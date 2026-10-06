use crate::ad::AD;
use crate::custom_op::CustomOp;
use crate::node::{NodeId, Variable};
use crate::scalar::Scalar;
use crate::tape::{OpRecord, Tape};
use smallvec::{smallvec, SmallVec};
use std::collections::HashMap;
use std::rc::Rc;

/// AD 上下文：管理 tape、伴随数组与叶子梯度。
///
/// 两种使用方式（设计文档 §2.4）：
/// 1. **显式路径**：持有 `&mut Context` 调用方法（checkpoint 集成、测试推荐）；
/// 2. **线程局部路径**：[`Context::enter`] 装载后用运算符重载 /
///    [`ContextGuard`](crate::ContextGuard) / [`with_context`]。
///
/// 梯度语义契约（设计文档 §4.1.4）：
/// - `backward` / `backward_from` / `backward_seeds`：叶子梯度**累加**到已有值
/// - [`Context::zero_grads`]：只清叶子梯度，不动 tape
/// - [`Context::clear_tape`]：释放 tape 与中间伴随，保留叶子注册与梯度；
///   之前 tape 产生的中间 `AD` 值随之失效（不得跨 clear 复用）
pub struct Context<S: Scalar> {
    pub(crate) tape: Tape<S>,
    /// 节点 ID 分配计数（单调递增；clear_tape 后重置到叶子水位之上）
    next_node: usize,
    /// 中间节点伴随值（backward 时按节点数分配）
    adjoints: Vec<S>,
    /// 叶子节点（创建顺序）
    leaves: Vec<NodeId>,
    /// 节点 ID -> 叶子下标
    leaf_index: HashMap<NodeId, usize>,
    /// 叶子梯度（跨 backward 累加，直到 zero_grads）
    leaf_gradients: Vec<S>,
    has_backwarded: bool,
    no_grad_depth: usize,
    /// 反向时检测首个非有限伴随（默认关）
    detect_anomaly: bool,
    anomaly_reported: bool,
    /// detect_anomaly 开启时记录的节点前向值（异常定位用，设计文档 §4.2.5）
    forward_values: Vec<S>,
    /// 自定义算子注册表（§12.3 第 42 条）：tape 记录存 `op_id` 而非 `Rc`，
    /// 同一算子的重复调用零引用计数开销；生命周期由 Context 持有。
    op_registry: Vec<Rc<dyn CustomOp<S>>>,
    /// `Rc::as_ptr` → 注册表槽位（去重：同一算子多次调用共享同一 id）
    op_ids: HashMap<usize, u32>,
}

impl<S: Scalar> Default for Context<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> Context<S> {
    pub fn new() -> Self {
        Context {
            tape: Tape::new(),
            next_node: 0,
            adjoints: Vec::new(),
            leaves: Vec::new(),
            leaf_index: HashMap::new(),
            leaf_gradients: Vec::new(),
            has_backwarded: false,
            no_grad_depth: 0,
            detect_anomaly: false,
            anomaly_reported: false,
            forward_values: Vec::new(),
            op_registry: Vec::new(),
            op_ids: HashMap::new(),
        }
    }

    /// 创建 context 并安装为线程局部当前实例，返回 guard（drop 时弹出）。
    pub fn enter(self) -> crate::guard::ContextGuard<S> {
        crate::guard::install(self)
    }

    // ---- 叶子与梯度 ----

    /// 分配一个新的可微变量（叶子）。
    pub fn var(&mut self, value: S) -> (AD<S>, Variable) {
        let id = self.alloc_node(value);
        let idx = self.leaves.len();
        self.leaves.push(id);
        self.leaf_index.insert(id, idx);
        self.leaf_gradients.push(S::zero());
        (AD::tracked(value, id), Variable::new(id))
    }

    /// 读取叶子梯度。`None` 表示尚未 backward 或该句柄不是本 context 的叶子。
    pub fn grad(&self, var: Variable) -> Option<S> {
        if !self.has_backwarded {
            return None;
        }
        self.leaf_index
            .get(&var.node)
            .map(|&i| self.leaf_gradients[i])
    }

    /// 按被追踪标量读取叶子梯度（`AD` 为本 context 创建的叶子时有效）。
    pub fn grad_of(&self, ad: AD<S>) -> Option<S> {
        if !self.has_backwarded {
            return None;
        }
        ad.node
            .and_then(|n| self.leaf_index.get(&n))
            .map(|&i| self.leaf_gradients[i])
    }

    /// 读取任意节点（含中间节点）最近一次 backward 的伴随值。
    /// 仅在下一次 backward / clear_tape 之前有效。
    pub fn adjoint(&self, ad: AD<S>) -> Option<S> {
        let node = ad.node?;
        if self.adjoints.len() == self.next_node {
            Some(self.adjoints[node.index()])
        } else {
            None
        }
    }

    /// 仅清零叶子梯度（tape 不动）。
    pub fn zero_grads(&mut self) {
        for g in &mut self.leaf_gradients {
            *g = S::zero();
        }
    }

    /// 按全局 L2 范数裁剪给定叶子集合的梯度（原位）：g ← g·min(1, max_norm/‖g‖)。
    /// 返回裁剪前的范数。范数非有限时不做任何缩放（交给异常检测定位）。
    pub fn clip_grad_norm(&mut self, vars: &[Variable], max_norm: S) -> S {
        let mut sq = S::zero();
        for &v in vars {
            if let Some(g) = self.grad(v) {
                sq = sq + g * g;
            }
        }
        let norm = sq.sqrt();
        if norm.is_finite() && norm > max_norm {
            let scale = max_norm / norm;
            for &v in vars {
                if let Some(&i) = self.leaf_index.get(&v.node) {
                    self.leaf_gradients[i] = self.leaf_gradients[i] * scale;
                }
            }
        }
        norm
    }

    /// 逐元素裁剪给定叶子集合的梯度到 [-value, value]。
    pub fn clip_grad_value(&mut self, vars: &[Variable], value: S) {
        for &v in vars {
            if let Some(&i) = self.leaf_index.get(&v.node) {
                let g = self.leaf_gradients[i];
                self.leaf_gradients[i] = if g > value {
                    value
                } else if g < -value {
                    -value
                } else {
                    g
                };
            }
        }
    }

    /// 清空 tape 与中间伴随（保留叶子注册和梯度），供下一轮前向复用。
    /// 此前 tape 产生的中间 `AD` 值失效。
    pub fn clear_tape(&mut self) {
        self.tape.clear();
        self.adjoints.clear();
        self.forward_values.clear();
        self.anomaly_reported = false;
        // 记录已全部丢弃 → op_id 引用清零，注册表可安全重置（否则
        // "每循环 Rc::new + call_custom" 的模式会累积注册项——容量保留，
        // 复用循环零重分配）
        self.op_registry.clear();
        self.op_ids.clear();
        self.next_node = self.leaves.iter().map(|n| n.index() + 1).max().unwrap_or(0);
    }

    /// 当前 tape 记录数（监控 tape 增速的探针，设计文档 §5.5）。
    pub fn tape_len(&self) -> usize {
        self.tape.len()
    }

    // ---- 反向传播 ----

    /// 反向传播：seed = 1 作用于 loss 节点。
    pub fn backward(&mut self, loss: AD<S>) {
        self.backward_from(loss, S::one());
    }

    /// 带种子的反向传播（VJP）：计算 vᵀ·(∂y/∂x)。
    pub fn backward_from(&mut self, y: AD<S>, seed: S) {
        self.backward_seeds(&[(y, seed)]);
    }

    /// 多种子反向传播：各 seed 的伴随先累加再统一反向。
    /// seeds 可指向任意被追踪节点（叶子或中间节点）。
    pub fn backward_seeds(&mut self, seeds: &[(AD<S>, S)]) {
        assert!(!seeds.is_empty(), "backward requires at least one seed");
        self.adjoints.clear();
        self.adjoints.resize(self.next_node, S::zero());
        self.anomaly_reported = false;

        let Context {
            tape,
            adjoints,
            leaves,
            leaf_gradients,
            detect_anomaly,
            anomaly_reported,
            forward_values,
            next_node,
            op_registry,
            ..
        } = self;

        for (y, s) in seeds {
            let node = y.node.unwrap_or_else(|| {
                panic!("backward: cannot differentiate a constant (no graph node)")
            });
            adjoints[node.index()] = adjoints[node.index()] + *s;
            if *detect_anomaly && !adjoints[node.index()].is_finite() && !*anomaly_reported {
                *anomaly_reported = true;
                panic!(
                    "gradient anomaly (seed): non-finite adjoint seeded at {:?}",
                    node
                );
            }
        }

        for (idx, rec) in tape.records().iter().enumerate().rev() {
            match rec {
                OpRecord::Native {
                    op,
                    output,
                    inputs,
                    jacobians,
                } => {
                    let g = adjoints[output.index()];
                    // 死分支剪枝（NaN != 0，不会误剪）
                    if g == S::zero() {
                        continue;
                    }
                    for (&i, j) in inputs.iter().zip(jacobians.iter()) {
                        let t = adjoints[i.index()] + g * *j;
                        if *detect_anomaly && !t.is_finite() && !*anomaly_reported {
                            *anomaly_reported = true;
                            panic!(
                                "gradient anomaly: non-finite adjoint at record #{} (op '{}'): \
                                 output {:?} adjoint {:?}, jacobian {:?}, input {:?} value {}",
                                idx,
                                op,
                                output,
                                g,
                                j,
                                i,
                                value_str(forward_values, *next_node, i)
                            );
                        }
                        adjoints[i.index()] = t;
                    }
                }
                OpRecord::Custom {
                    name,
                    op_id,
                    inputs,
                    output_base,
                    residual,
                } => {
                    // 算子经注册表查找（记录只存 id，§12.3 第 42 条）
                    let op = &*op_registry[*op_id as usize];
                    // 输出节点连续（NodeId 单调分配）：输出伴随是 adjoints 的
                    // 连续切片，直接以切片形式传入 VJP（无收集拷贝）
                    let n_out = op.num_outputs();
                    let base = output_base.index();
                    let gins = {
                        let gout = &adjoints[base..base + n_out];
                        if gout.iter().all(|&g| g == S::zero()) {
                            continue;
                        }
                        op.backward(&residual[..], gout)
                    };
                    debug_assert_eq!(
                        gins.len(),
                        op.num_inputs(),
                        "CustomOp '{}' returned wrong number of input gradients",
                        name
                    );
                    // gins 按**原始槽位**索引（常量输入占槽位但不占节点）——
                    // tracked 子集必须按 slot 取，不能与 gins 按位置 zip：
                    // 常量在尾部时 zip 恰好对齐（潜伏 bug，matvec 部分追踪
                    // 形态首踩：M 常量在前、v 在后，zip 错把 λM 路由给 λv）
                    for &(slot, inode) in inputs.iter() {
                        let g = gins[slot as usize];
                        let t = adjoints[inode.index()] + g;
                        if *detect_anomaly && !t.is_finite() && !*anomaly_reported {
                            *anomaly_reported = true;
                            panic!(
                                "gradient anomaly: non-finite adjoint at record #{} (custom op '{}', \
                                 input slot {} {:?}): output_base {:?}, contribution {:?}",
                                idx, name, slot, inode, output_base, g
                            );
                        }
                        adjoints[inode.index()] = t;
                    }
                }
            }
        }

        for (li, &node) in leaves.iter().enumerate() {
            leaf_gradients[li] = leaf_gradients[li] + adjoints[node.index()];
        }
        self.has_backwarded = true;
    }

    // ---- 前向记录 ----

    /// 记录一个一元算子（`ad-ops` 的实现基础）。
    pub fn unary(
        &mut self,
        a: AD<S>,
        op: &'static str,
        forward: impl Fn(S) -> S,
        jac: impl Fn(S) -> S,
    ) -> AD<S> {
        let value = forward(a.value);
        let Some(an) = a.node else {
            return AD::constant(value);
        };
        if self.no_grad_depth > 0 {
            return AD::constant(value);
        }
        let j = jac(a.value);
        let out = self.alloc_node(value);
        self.tape.push(OpRecord::Native {
            op,
            output: out,
            inputs: smallvec![an],
            jacobians: smallvec![j],
        });
        AD::tracked(value, out)
    }

    /// 记录一个二元算子（`ad-ops` 的实现基础）。
    /// 常量输入被折叠出记录（设计文档 §4.2.2）。
    pub fn binary(
        &mut self,
        a: AD<S>,
        b: AD<S>,
        op: &'static str,
        forward: impl Fn(S, S) -> S,
        jac_a: impl Fn(S, S) -> S,
        jac_b: impl Fn(S, S) -> S,
    ) -> AD<S> {
        let value = forward(a.value, b.value);
        if self.no_grad_depth > 0 {
            return AD::constant(value);
        }
        let j_a = jac_a(a.value, b.value);
        let j_b = jac_b(a.value, b.value);
        let mut inputs = SmallVec::<[NodeId; 4]>::new();
        let mut jacobians = SmallVec::<[S; 4]>::new();
        if let Some(n) = a.node {
            inputs.push(n);
            jacobians.push(j_a);
        }
        if let Some(n) = b.node {
            inputs.push(n);
            jacobians.push(j_b);
        }
        if inputs.is_empty() {
            return AD::constant(value);
        }
        let out = self.alloc_node(value);
        self.tape.push(OpRecord::Native {
            op,
            output: out,
            inputs,
            jacobians,
        });
        AD::tracked(value, out)
    }

    /// 记录一个 n 元算子（如 lerp）。
    pub fn nary(
        &mut self,
        inputs: &[AD<S>],
        op: &'static str,
        forward: impl Fn(&[S]) -> S,
        jacobians: impl Fn(&[S]) -> SmallVec<[S; 3]>,
    ) -> AD<S> {
        let vals: SmallVec<[S; 3]> = inputs.iter().map(|x| x.value).collect();
        let value = forward(&vals);
        if self.no_grad_depth > 0 {
            return AD::constant(value);
        }
        let jacs = jacobians(&vals);
        let mut tracked = SmallVec::<[NodeId; 4]>::new();
        let mut jacobians_kept = SmallVec::<[S; 4]>::new();
        for (x, j) in inputs.iter().zip(jacs.iter()) {
            if let Some(n) = x.node {
                tracked.push(n);
                jacobians_kept.push(*j);
            }
        }
        if tracked.is_empty() {
            return AD::constant(value);
        }
        let out = self.alloc_node(value);
        self.tape.push(OpRecord::Native {
            op,
            output: out,
            inputs: tracked,
            jacobians: jacobians_kept,
        });
        AD::tracked(value, out)
    }

    // ---- 基础算术（运算符重载的显式路径） ----

    pub fn add(&mut self, a: AD<S>, b: AD<S>) -> AD<S> {
        self.binary(a, b, "add", |a, b| a + b, |_, _| S::one(), |_, _| S::one())
    }

    pub fn sub(&mut self, a: AD<S>, b: AD<S>) -> AD<S> {
        self.binary(a, b, "sub", |a, b| a - b, |_, _| S::one(), |_, _| -S::one())
    }

    pub fn mul(&mut self, a: AD<S>, b: AD<S>) -> AD<S> {
        self.binary(a, b, "mul", |a, b| a * b, |_, b| b, |a, _| a)
    }

    pub fn div(&mut self, a: AD<S>, b: AD<S>) -> AD<S> {
        self.binary(
            a,
            b,
            "div",
            |a, b| a / b,
            |_, b| S::one() / b,
            |a, b| -(a / (b * b)),
        )
    }

    pub fn neg(&mut self, a: AD<S>) -> AD<S> {
        self.unary(a, "neg", |a| -a, |_| -S::one())
    }

    // ---- 自定义算子 ----

    /// 调用自定义算子：登记多输出节点，前向残差存入 tape（设计文档 §4.3.1）。
    pub fn call_custom<O: CustomOp<S> + 'static>(
        &mut self,
        op: O,
        inputs: &[AD<S>],
    ) -> SmallVec<[AD<S>; 4]> {
        let name = std::any::type_name::<O>();
        self.call_custom_dyn(Rc::new(op), name, inputs)
    }

    /// [`Context::call_custom`] 的 dyn 版本：算子已装箱（重放场景复用同一 Rc）。
    pub fn call_custom_dyn(
        &mut self,
        op: Rc<dyn CustomOp<S>>,
        name: &'static str,
        inputs: &[AD<S>],
    ) -> SmallVec<[AD<S>; 4]> {
        debug_assert_eq!(
            inputs.len(),
            op.num_inputs(),
            "input count mismatch for {}",
            name
        );
        let vals: SmallVec<[S; 8]> = inputs.iter().map(|x| x.value).collect();
        let (outs, residual) = op.forward(&vals);
        debug_assert_eq!(
            outs.len(),
            op.num_outputs(),
            "output count mismatch for {}",
            name
        );
        if self.no_grad_depth > 0 || inputs.iter().all(|x| x.node.is_none()) {
            return outs.iter().map(|&v| AD::constant(v)).collect();
        }
        // 算子注册：同一 Rc（指针相等）复用同一 id——重复调用零引用计数
        let op_id = if let Some(&id) = self.op_ids.get(&(Rc::as_ptr(&op) as *const u8 as usize)) {
            id
        } else {
            let id = self.op_registry.len() as u32;
            self.op_registry.push(Rc::clone(&op));
            self.op_ids.insert(Rc::as_ptr(&op) as *const u8 as usize, id);
            id
        };
        // 输出节点连续分配：记录 base，返回值直接用本地计数重建 id
        if outs.is_empty() {
            return SmallVec::new();
        }
        let output_base = self.alloc_node(outs[0]);
        for &v in &outs[1..] {
            self.alloc_node(v);
        }
        let tracked: SmallVec<[(u32, NodeId); 8]> = inputs
            .iter()
            .enumerate()
            .filter_map(|(slot, x)| x.node.map(|n| (slot as u32, n)))
            .collect();
        self.tape.push(OpRecord::Custom {
            name,
            op_id,
            inputs: tracked,
            output_base,
            residual,
        });
        let base = output_base.index();
        outs.iter()
            .enumerate()
            .map(|(k, &v)| AD::tracked(v, NodeId::new(base + k)))
            .collect()
    }

    // ---- 调试与作用域 ----

    /// 开启/关闭非有限梯度异常检测（设计文档 §4.2.5）。
    pub fn set_detect_anomaly(&mut self, on: bool) {
        self.detect_anomaly = on;
        if !on {
            self.forward_values.clear();
        }
    }

    /// 在区域内抑制 tape 记录（返回值退化为常量）。线程局部路径请用自由函数
    /// [`no_grad`](crate::no_grad)。
    pub fn no_grad<R>(&mut self, f: impl FnOnce(&mut Context<S>) -> R) -> R {
        self.no_grad_depth += 1;
        let r = f(self);
        self.no_grad_depth -= 1;
        r
    }

    pub(crate) fn begin_no_grad(&mut self) {
        self.no_grad_depth += 1;
    }

    pub(crate) fn end_no_grad(&mut self) {
        self.no_grad_depth = self.no_grad_depth.saturating_sub(1);
    }

    fn alloc_node(&mut self, value: S) -> NodeId {
        let id = NodeId::new(self.next_node);
        self.next_node += 1;
        if self.detect_anomaly {
            self.forward_values.push(value);
        }
        id
    }
}

fn value_str<S: Scalar>(forward_values: &[S], next_node: usize, node: NodeId) -> String {
    if forward_values.len() == next_node {
        if let Some(v) = forward_values.get(node.index()) {
            return format!("{:?}", v);
        }
    }
    "n/a (enable detect_anomaly before the forward pass to capture values)".to_string()
}
