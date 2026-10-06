# Rust 自研自动微分（AD）库 — 项目需求与设计文档

**版本**：v0.3（补充版）

**定位**：面向物理仿真场景的、纯 Rust 实现的反向模式自动微分库

**状态**：需求确定，API 设计已细化，待进入实现阶段

---

## 1. 项目背景与目标

### 1.1 背景

可微分物理仿真引擎的核心依赖是一个能够在物理计算（动力学算法、接触求解、约束处理）中正确、高效计算梯度的 AD 库。Rust 生态中已有 `tenflowers-autograd`（支持 tape-based eager 模式和自定义梯度）、`gad`（通用反向模式 AD，支持用户自定义算子）、`scirs2-autograd`（PyTorch 风格惰性求值，支持 checkpoint 和梯度验证）等早期项目，但它们**并非针对物理仿真场景设计**：物理仿真中的长时间轨迹、非光滑接触、递归算法（如 ABA）对 AD 库提出了不同于深度学习训练的需求。

本项目的目标是独立于物理引擎，先构建一个**正确性可验证、梯度质量可度量**的 AD 库，作为后续可微分物理仿真引擎的"可微分计算底座"。

### 1.2 核心目标

| 目标 | 说明 |
|------|------|
| 反向模式 AD | 支持基于 tape 的反向传播，计算标量 loss 对输入的梯度 |
| 物理算子可接入 | 提供接口让用户注册自定义算子及其局部 Jacobian / VJP |
| 梯度验证 | 内置与中心有限差分、双数（dual number）oracle 的对比验证工具 |
| 长轨迹内存可控 | 支持 checkpoint 机制（均匀 / 二项式 / 在线调度），避免 tape 无限增长 |
| 隐式求解器可微 | 以隐函数定理（IFT）伴随模式支持"穿过"迭代求解器的梯度 |
| 纯 Rust，无外部 C++ 依赖 | 不依赖 PyTorch、LibTorch 或 JAX |

### 1.3 明确不做（第一版）

- 前向模式 AD（JVP）——**对外不提供**；内部保留一份双数实现仅作测试 oracle 与 IFT 残差 Jacobian 构造（见 §4.5.4），不构成公开 API 承诺
- 高阶导数（Hessian、HVP）
- GPU / CUDA 后端
- 自动并行化（单条 rollout 内不做并行；批量 rollout 的并行由用户用 rayon 自行组织，见 §4.1.6）
- 完整的张量运算库（只做标量 + 最小向量/矩阵支持，见 §4.3.4）
- 与 Python 的绑定
- tape 的序列化 / 跨线程合并

---

## 2. 核心概念与设计决策

### 2.1 Tape-based 反向模式 AD

第一版采用 **Wengert List（tape-based）** 实现方式。每次前向操作将算子及其输入输出记录到线性 tape 中，反向传播时逆序遍历 tape，按链式法则累积梯度。`tenflowers-autograd` 的 GradientTape 和 `scirs2-autograd` 的 tape-based 梯度累积都采用了这一模式。

**线性 tape 的关键不变量**：tape 上的记录顺序即计算图的拓扑序——每条记录的输入节点要么是叶子，要么由更早的记录产生。因此**逆序遍历 tape 就是拓扑逆序**：处理某条记录时，它的所有"消费者"（下游记录）都已被处理，此时读到的该节点伴随值就是终值。无需显式的拓扑排序，也无需邻接表。这个不变量同时是 checkpoint 分段反向（§4.4.4）正确性的基石。

### 2.2 标量级 AD + 物理算子

AD 库的核心抽象是 **AD 标量类型** `AD<S>`，其中 `S` 是底层数值类型（第一版为 `f64`）。所有物理计算（力矩、加速度、接触力）都在 `AD<f64>` 上执行，自动记录计算图。

**关键设计**：物理仿真的核心算法（ABA、RNEA）是**递归的**，如果让 AD 自动穿透整个递归，计算图会爆炸。因此 AD 库必须提供 **自定义算子注册接口**，允许物理引擎手动定义局部 Jacobian，AD 库只负责链式组装。这与 JAX 的 `custom_vjp` 机制思路一致——用户定义 `f_fwd`（前向计算 + 保存残差）和 `f_bwd`（反向梯度计算），JAX 负责调用和链式组装。

### 2.3 与 PyTorch Autograd 的关系

设计上参考 PyTorch 的 autograd 机制，但有一个关键区别：PyTorch 的 `gradcheck` 使用中心差分验证梯度，而物理仿真的梯度验证需要**长轨迹下的梯度条件数检查**。研究表明，基于 tape 的 AD 框架（如 Newton）会继承 O(T) 的内存缩放问题，而全解析伴随方法可以实现 O(1) 的每步反向内存。本项目的 checkpoint 机制正是为了解决这一矛盾。

### 2.4 Tape 的所有权与 API 形态（Rust 特有的关键决策）

这是原文档缺失、但**必须最先定下来**的决策：运算符重载 `a * b` 在执行时如何拿到 tape？Rust 没有隐式的全局可变状态，四种可行方案的取舍如下：

| 方案 | 人体工学 | 安全性 | 性能 | 代表 |
|------|---------|--------|------|------|
| A. `Rc<RefCell<Tape>>` 内嵌在每个值里 | 好（可直接重载） | 差（运行期借用冲突可能 panic；`AD` 不能是 `Copy`） | 差（每次操作引用计数 + RefCell 检查） | 各类微型/micrograd 风格实现 |
| B. **线程局部 Context**（推荐） | 好（可重载，`AD` 保持 `Copy`） | 中（隐藏状态，需作用域纪律） | 好（TLS 一次访问 + 直接 push） | 本项目 |
| C. 显式 `ctx` 传参（`ctx.mul(a, b)`，无重载） | 差（样板代码多） | 最好（一切显式） | 最好 | `gad` 风格 |
| D. 生命周期 branding（`Variable<'ctx>` + generative 技术保证值不逃逸出作用域） | 中（生命周期签名传染） | 编译期最强 | 好 | grad_oxide（Enzyme 生态）风格 |

**决策：B 为对外形态，C 为内部实现路径**——即内部所有入带逻辑都是显式 `ctx` 的自由函数（如 §4.2.2 的 `register_binary_op`），运算符重载只是从线程局部槽取出 ctx 后对它的一层薄封装：

```rust
impl<S: Scalar> Mul for AD<S> {
    type Output = AD<S>;
    fn mul(self, rhs: AD<S>) -> AD<S> {
        with_context(|ctx| register_binary_op(
            ctx, self, rhs,
            |a, b| a * b,      // forward
            |_a, b| b,        // ∂(a·b)/∂a = b
            |a, _b| a,        // ∂(a·b)/∂b = a
        ))
    }
}
```

配套约定：

1. **安装与作用域**：`Context::enter()` 将 ctx 压入线程局部栈并返回 guard，drop 时弹出。支持嵌套（如在外层 ctx 中临时进入 `no_grad` 区域），杜绝"忘了设/设错"的全局状态泄漏。
2. **`Send` / `Sync`**：`AD<S>` 里的 `NodeId` 只对创建它的线程的 tape 有意义，因此 `AD<S>` 与 `Variable` 携带 `PhantomData<*mut ()>` 使其 `!Send + !Sync`，把跨线程误用变成编译错误（可用 cfg 开关放宽，见 §4.1.6）。
3. **方案 D 列为演进项**：若 B 的隐式性在实践中造成维护负担（如测试难以隔离），可迁移到 generative lifetime 方案，对外 API 形态基本不变。

---

## 3. 系统架构

### 3.1 分层结构

```
┌─────────────────────────────────────────────────────┐
│  verify/       梯度验证层：有限差分、双数 oracle、     │
│                梯度条件分析、属性测试                  │
├─────────────────────────────────────────────────────┤
│  checkpoint/   检查点层：调度算法（均匀/二项式/在线）、 │
│                分段重算、边界伴随传递                  │
├─────────────────────────────────────────────────────┤
│  tape/          Tape 层：算子记录、逆序遍历、伴随累积   │
├─────────────────────────────────────────────────────┤
│  op/            算子层：基础算子 + 自定义算子注册       │
├─────────────────────────────────────────────────────┤
│  scalar/       标量抽象层：AD<S> 类型、算术重载        │
│                （经线程局部 Context 入带）             │
├─────────────────────────────────────────────────────┤
│  core/         核心抽象：Variable、Context、Graph      │
└─────────────────────────────────────────────────────┘
```

### 3.2 Crate 划分

| Crate | 职责 | 依赖 |
|-------|------|------|
| `ad-core` | `AD<S>` 标量、`Variable`、`Context`、`Tape`、线程局部挂载、`no_grad`、异常检测 | `num-traits` |
| `ad-ops` | 基础算子（加、乘、幂、三角、ReLU 等）及局部 Jacobian、常量折叠、bulk 辅助算子（dot/axpy/norm） | `ad-core` |
| `ad-custom` | 自定义算子注册接口（`CustomOp`）、IFT 隐式求解模式、物理算子适配 | `ad-core` |
| `ad-checkpoint` | 检查点调度（均匀 / 二项式 / 在线）、分段反向、边界伴随传递 | `ad-core` |
| `ad-verify` | 梯度验证工具：有限差分、双数 oracle、梯度健康度、属性测试 | `ad-core`（dev: `proptest`） |
| `ad` | 用户接口 facade，`prelude`，整合所有模块 | 全部 |

---

## 4. 模块详细设计

### 4.1 `ad-core`：核心抽象

#### 4.1.1 AD 标量类型

```rust
/// AD 标量：承载数值 + 计算图节点标识
#[derive(Clone, Copy)]
pub struct AD<S: Scalar> {
    /// 前向数值
    pub value: S,
    /// 计算图节点 ID（None 表示该标量为常量，不入带）
    pub node: Option<NodeId>,
    /// 防止跨线程误用（NodeId 只对当前线程的 tape 有意义），零开销
    _local: PhantomData<*mut ()>,
}

impl<S: Scalar> AD<S> {
    /// 创建常量（不可微、不参与入带）
    pub fn constant(value: S) -> Self;
}
```

> API 清理（相对 v0.2）：删除了 `AD::variable(value, ctx)`，叶子变量统一由 `Context::var` 创建（避免两套入口语义漂移）。

**设计决策**：`node: Option<NodeId>` 而非 `NodeId`，使得常量在计算图中不占节点，反向传播时自动跳过。这可以减少 tape 大小。`tenflowers-autograd` 的 Tracked Tensors 机制也采用了类似的"仅追踪参与计算的张量"的思路。

**`Scalar` trait 定义**（第一版基于 `num-traits` 的 `Float`，轻量依赖）：

```rust
pub trait Scalar: Copy + Debug + PartialOrd + num_traits::Float {}
impl Scalar for f64 {}
impl Scalar for f32 {}   // 支持，但梯度验证容差需单独标定（见 §5.1）
```

#### 4.1.2 Variable 与 Context

```rust
/// 叶子变量句柄：轻量级引用，不持有数据
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Variable {
    pub node: NodeId,
}

/// AD 上下文：管理 Tape、伴随数组与叶子梯度
pub struct Context<S: Scalar> {
    tape: Tape<S>,
    /// 中间节点伴随值（backward 时按节点数分配）
    adjoints: Vec<S>,
    /// 叶子节点的梯度累积（跨多次 backward 持续累加，直到 zero_grads）
    leaf_gradients: Vec<S>,
    /// 节点到叶子变量的映射
    leaf_map: Vec<Variable>,
    /// 反向时检测首个非有限伴随（默认关，见 §4.2.5）
    detect_anomaly: bool,
}

impl<S: Scalar> Context<S> {
    /// 创建 context 并安装为线程局部当前实例，返回 guard
    pub fn enter(self) -> ContextGuard<S>;

    /// 分配一个新的可微变量（叶子）
    pub fn var(&mut self, value: S) -> (AD<S>, Variable);

    /// 反向传播：seed = 1 作用于 loss 节点（等价于 backward_from(loss, 1)）
    pub fn backward(&mut self, loss: AD<S>);

    /// 带种子的反向传播（VJP）：计算 vᵀ·(∂y/∂x)，用于多目标/加权梯度
    pub fn backward_from(&mut self, y: AD<S>, seed: S);

    /// 读取叶子梯度。None = 尚未 backward 或该句柄不是叶子
    pub fn grad(&self, var: Variable) -> Option<S>;

    /// 仅清零叶子梯度（下一次 backward 前调用；tape 不动）
    pub fn zero_grads(&mut self);

    /// 清空 tape 与中间伴随（保留叶子注册和梯度），供下一轮前向复用
    pub fn clear_tape(&mut self);

    /// 当前 tape 记录数（监控 tape 增速、发现"忘用 bulk 算子"的探针，见 §8）
    pub fn tape_len(&self) -> usize;
}
```

> API 清理（相对 v0.2）：原 `clear()` 更名为 `clear_tape()` 并把"清梯度"拆分为独立的 `zero_grads()`；`grad` 返回 `Option<S>`；新增 `backward_from`（VJP 种子）。

#### 4.1.3 Tape 结构（修正：支持多输出自定义算子）

v0.2 的 `OpRecord` 只有单 `output`、只存标量局部 Jacobian——**与 §4.3 的 `CustomOp` 多输出、需要保存残差的设计直接矛盾**（残差无处可存，多输出节点无法登记）。修正为枚举：

```rust
/// 算子记录（SSA 风格：每条记录产出新节点，不做原地更新）
enum OpRecord<S: Scalar> {
    /// 基础算子：局部 Jacobian 已在前向时算好
    Native {
        output: NodeId,
        inputs: SmallVec<[NodeId; 2]>,
        /// ∂output/∂input_i，与 inputs 一一对应
        jacobians: SmallVec<[S; 2]>,
    },
    /// 自定义算子：多输入多输出，反向调用 VJP
    Custom {
        inputs: SmallVec<[NodeId; 4]>,
        outputs: SmallVec<[NodeId; 4]>,
        op: Rc<dyn CustomOp<S>>,
        /// 前向保存的残差数据（f_fwd 风格），backward 时原样传回
        residual: SmallVec<[S; 8]>,
    },
}

/// Wengert List：线性 tape，天然按拓扑序排列
pub struct Tape<S: Scalar> {
    records: Vec<OpRecord<S>>,
}
```

> API 清理（相对 v0.2）：删除了冗余的 `op_order: Vec<usize>`——记录本来就是追加式线性序，逆序遍历即拓扑逆序，该字段无事可做。

**设计决策**：`Native` 的局部 Jacobian 在**前向传播时**计算并存储。这避免反向传播时重新计算中间量，但会增加 tape 内存。对于物理算子（如 ABA 的局部 Jacobian），前向计算时这些量本身就可获得，存储它们边际成本低。这与 JAX 的 `f_fwd` 返回残差数据的机制一致——残差在前向时保存，反向时使用。`Custom` 记录的 `residual` 同理，由 `CustomOp::forward` 返回并原地存入 tape（超出 SmallVec 内联容量时自动落到堆上，无需第二个 trait）。

#### 4.1.4 反向传播实现（伴随数组与梯度语义）

反向遍历使用按节点索引的伴随数组 `adjoints: Vec<S>`，算法如下：

```rust
fn backward(&mut self, loss: AD<S>, seed: S) {
    let n = self.tape.num_nodes();
    self.adjoints.clear();
    self.adjoints.resize(n, S::zero());
    self.adjoints[loss.node.unwrap()] = seed;   // 无节点 = 常量 loss，梯度全零

    for rec in self.tape.records.iter().rev() {
        match rec {
            OpRecord::Native { output, inputs, jacobians } => {
                let g = self.adjoints[*output];
                if g == S::zero() { continue; }          // 死分支剪枝（NaN != 0，不会误剪）
                for (i, j) in inputs.iter().zip(jacobians) {
                    self.adjoints[*i] += g * *j;
                }
            }
            OpRecord::Custom { inputs, outputs, op, residual } => {
                let gout: SmallVec<_> = outputs.iter().map(|o| self.adjoints[*o]).collect();
                let gins = op.backward(residual, &gout);
                for (i, g) in inputs.iter().zip(gins) {
                    self.adjoints[*i] += g;
                }
            }
        }
    }
    // 遍历结束后，叶子梯度 = adjoints[leaf.node]，累加进 leaf_gradients
}
```

**梯度语义契约**（显式文档化，避免实现与用户预期错位）：

| 调用 | 语义 |
|------|------|
| `backward(loss)` | seed = 1；叶子梯度**累加**到已有值（PyTorch 语义，支持梯度累积式优化） |
| `backward_from(y, v)` | 同上但 seed = v（VJP） |
| `zero_grads()` | 只清叶子梯度，不动 tape |
| `clear_tape()` | 释放 tape 与中间伴随，保留叶子注册；配合 `Vec::clear` 复用容量，长循环中 RSS 稳定（见 §5.5） |
| 常量输入 | 无节点，反向自动跳过 |
| 原地操作 | **不支持**。SSA 语义：`x = x + a` 生成新节点。物理引擎侧的状态更新应建模为"新状态 = 自定义算子(旧状态, …)" |

#### 4.1.5 `no_grad` 与 `detach`

```rust
// 区域内前向计算不记录 tape，返回的 AD 是常量
ctx.no_grad(|ctx| {
    let y = expensive_forward(a, b);   // 不增长 tape
    y
});

// 值拷贝并切断梯度（常量化）
let c = x.detach();
```

物理仿真中的实际用途：

1. **线搜索 / rollout 重评估**：优化循环中按同一前向公式重算 loss（不需要梯度）时避免 tape 增长；
2. **冻结参数**：把不参与优化的量声明为常量或 detach；
3. **阻断病态梯度路径**：例如阻断穿过接触求解器 warm-start 初值的梯度（初值只影响求解效率、不改变解，对它求导没有意义且引入噪声）。

#### 4.1.6 并发模型

- **单条 rollout = 单线程单 Context**。tape 的拓扑序不变量要求入带串行，第一版不做 rollout 内并行。
- `Context` 为 `!Sync`；`AD<S>` / `Variable` 为 `!Send + !Sync`（§2.4），跨线程误用编译期报错。
- **批量场景**（并行采样 N 条轨迹求梯度，MPC / RL 常见）：推荐 rayon，每个任务 `Context::new()` 各自独立，梯度在任务结束时聚合。库提供文档示例；不做隐式全局调度。

---

### 4.2 `ad-ops`：基础算子

#### 4.2.1 算子清单

| 算子 | 前向 | 局部 Jacobian | 备注 |
|------|------|--------------|------|
| Add | `a + b` | ∂/∂a = 1, ∂/∂b = 1 | |
| Sub | `a - b` | ∂/∂a = 1, ∂/∂b = -1 | |
| Mul | `a * b` | ∂/∂a = b, ∂/∂b = a | |
| Div | `a / b` | ∂/∂a = 1/b, ∂/∂b = -a/b² | b = 0 见 §4.2.4 |
| Powf | `a.powf(b)` | ∂/∂a = b·a^(b-1), ∂/∂b = a^b·ln(a) | 负底非整数幂见 §4.2.4 |
| Exp | `a.exp()` | ∂/∂a = exp(a) | |
| Ln | `a.ln()` | ∂/∂a = 1/a | x ≤ 0 见 §4.2.4 |
| Sin | `a.sin()` | ∂/∂a = cos(a) | |
| Cos | `a.cos()` | ∂/∂a = -sin(a) | |
| Tanh | `a.tanh()` | ∂/∂a = 1 - tanh²(a) | 物理中常用（有界） |
| Asin | `a.asin()` | ∂/∂a = 1/√(1-a²) | \|a\| < 1 |
| Acos | `a.acos()` | ∂/∂a = -1/√(1-a²) | \|a\| < 1 |
| Atan2 | `atan2(y, x)` | ∂/∂y = x/r², ∂/∂x = -y/r², r = √(x²+y²) | **关节角包裹必备**；(0,0) 见 §4.2.4 |
| Sqrt | `a.sqrt()` | ∂/∂a = 1/(2√a) | a = 0 时反向 inf |
| Recip | `1/a` | ∂/∂a = -1/a² | 比 Div(1, a) 语义清晰 |
| Abs | `a.abs()` | ∂/∂a = sign(a)（a≠0） | 非光滑，见 §4.2.3 |
| Min / Max | `min(a,b)` / `max(a,b)` | 1 加在 argmin / argmax 一侧（并列取第一个参数，与主流框架一致） | 非光滑 |
| Clamp | `clamp(a, lo, hi)` | 越界侧梯度为 0 | 非光滑；接触速度限幅常用 |
| ReLU | `a.relu()` | ∂/∂a = 1 if a>0 else 0 | 非光滑 |
| Sigmoid | `1/(1+exp(-a))` | ∂/∂a = σ(a)(1-σ(a)) | |
| Lerp | `a + t(b-a)` | ∂/∂a = 1-t, ∂/∂b = t, ∂/∂t = b-a | 插值/软化混合 |

向量级最小集（作为 bulk 自定义算子提供，见 §4.3.4）：`dot`、`axpy`、`norm2`。

#### 4.2.2 算子实现模式

```rust
/// 注册一个二元算子（内部实现路径：显式 ctx）
fn register_binary_op<S: Scalar>(
    ctx: &mut Context<S>,
    a: AD<S>,
    b: AD<S>,
    forward: impl Fn(S, S) -> S,
    jac_a: impl Fn(S, S) -> S,
    jac_b: impl Fn(S, S) -> S,
) -> AD<S> {
    let value = forward(a.value, b.value);

    // 常量折叠：所有输入都是常量 → 结果也是常量，不入带
    let (Some(a_node), Some(b_node)) = (a.node, b.node) else {
        return AD::constant(value);
    };

    let out_node = ctx.tape.push(OpRecord::Native {
        output: ctx.next_node_id(),
        inputs: smallvec![a_node, b_node],
        jacobians: smallvec![jac_a(a.value, b.value), jac_b(a.value, b.value)],
    });

    AD { value, node: Some(out_node), ..Default::default() }
}
```

运算符重载（`impl Mul/Add/Sub/Div/Neg for AD<S>` 及其引用组合形式）只是"线程局部取 ctx → 调用上述函数"的薄封装（§2.4）。常量折叠保证纯常量子表达式完全不消耗 tape——物理公式里大量系数运算是常量间的，这一条能把 tape 缩小可观的比例。

#### 4.2.3 非光滑点的处理

ReLU 在 0 点的导数定义为 0（与 PyTorch/TensorFlow 一致）。这保证整个程序是 **PAP（分段解析）函数**，AD 的正确性在"几乎处处"意义上成立。

**物理仿真的特殊挑战**：接触/分离、静摩擦/动摩擦的切换点会产生**梯度不连续**，这会阻碍基于梯度的优化，尤其是当解轨迹穿越分支边界时。AD 库需要提供**梯度裁剪接口**，允许物理引擎在反向传播前对梯度做平滑或裁剪。`tenflowers-autograd` 的 `CustomOp` 支持 `ClipGradient` 这样的自定义反向逻辑。

库层面提供两个叶子梯度工具（实现在 `ad-core`，供优化器使用）：

```rust
impl<S: Scalar> Context<S> {
    /// 按全局范数裁剪：g ← g · min(1, max_norm / ‖g‖)
    pub fn clip_grad_norm(&mut self, vars: &[Variable], max_norm: S) -> S;
    /// 逐元素裁剪：g ← clamp(g, -v, v)
    pub fn clip_grad_value(&mut self, vars: &[Variable], v: S);
}
```

#### 4.2.4 数值边界与定义域策略

原则：**不 panic、不静默饱和**。前向遵守 IEEE 754 语义（产生 NaN/Inf 并传播），由 §4.2.5 的异常检测在反向时定位首个出问题的算子；值域约束由调用方负责，库只负责文档化 + debug 断言（`debug_assert!` 可关）。

| 情形 | 前向值 | 反向值 | 策略 |
|------|--------|--------|------|
| `ln(x)`，x ≤ 0 | NaN | — | 传播，异常检测可定位 |
| `powf` 负底 + 非整数指数 | NaN | — | 同上 |
| `sqrt(0)` | 0 | **+inf** | 文档化；物理上常见（接触法向速度为 0），建议上游加 ε 或用软化模型 |
| `a / 0`（a≠0） | ±inf | 0（∂/∂b 路径） | 传播 |
| `abs(0)`、`relu(0)`、`clamp` 边界 | 0 | 0 | PAP 约定，见 §4.2.3 |
| `atan2(0, 0)` | 0（同 Rust std） | 无定义（r=0） | 文档化；调用方保证不同时为 0 |

#### 4.2.5 非有限梯度的异常检测

高刚度接触、除零、sqrt(0) 等会让梯度出现 NaN/Inf。裸的 NaN 梯度只告诉你"坏了"，不告诉你**哪里**坏的。类比 PyTorch 的 `torch.autograd.set_detect_anomaly`：

```rust
ctx.set_detect_anomaly(true);
```

开启后，反向遍历时**首个**产生非有限伴随值的记录会带上下文失败：算子种类（Native 微码 / CustomOp 的 Rust 类型名）、节点 ID、参与输入的前向值摘要。仅一次比较的额外开销，默认关闭。物理调试（"梯度从哪一步开始爆炸"）时是第一工具，配合 §4.5.3 的逐步范数追踪使用。

---

### 4.3 `ad-custom`：自定义算子与物理算子适配

#### 4.3.1 自定义算子 trait

参考 JAX 的 `custom_vjp` 设计：

```rust
/// 自定义算子：物理引擎实现此 trait 来接入 AD
pub trait CustomOp<S: Scalar> {
    /// 输入数量
    fn num_inputs(&self) -> usize;

    /// 输出数量
    fn num_outputs(&self) -> usize;

    /// 前向计算，返回 (输出值, 残差数据)
    /// 残差 = backward 需要但无法从输入输出恢复的中间量（ABA 的 U、D、Ia、pa 等）
    fn forward(&self, inputs: &[S]) -> (SmallVec<[S; 4]>, SmallVec<[S; 8]>);

    /// 反向计算（VJP）：给定残差和各输出的伴随值，返回各输入的梯度贡献
    fn backward(
        &self,
        residual: &[S],
        grad_output: &[S],
    ) -> SmallVec<[S; 4]>;
}

/// 调用入口：登记 Custom 记录，登记多输出节点
pub fn call_custom<S: Scalar>(ctx: &mut Context<S>, op: &dyn CustomOp<S>, inputs: &[AD<S>]) -> SmallVec<[AD<S>; 4]>;
```

**设计决策**：采用 JAX `custom_vjp` 风格的 `forward + backward` 接口，而非手动提供局部 Jacobian 矩阵。原因：

1. **更灵活**：`backward` 可以直接返回 VJP（vector-Jacobian product），不需要显式构造完整 Jacobian
2. **物理算子友好**：ABA 的反向传播天然是递归的，用 `backward` 形式更自然
3. **与主流框架一致**：PyTorch、JAX 的自定义算子都采用这种模式

注意：`forward` 的输入是**已剥离节点信息的纯数值切片**（`&[S]`）。自定义算子内部（如 ABA 的递归）用普通 `f64` 计算，**不会再入带**——这正是"计算图不爆炸"的机制本身；需要内部梯度的场景嵌套一层 `call_custom` 或用 IFT 模式（§4.3.3）。

**⚠️ backward 契约（实现期发现，§12.3 第 13 条）**：tape 只看见算子级的输入→输出边，
**算子内部的数据流（包括输出之间的耦合边）对 tape 不可见，backward 必须完整覆盖**。
典型陷阱：`x' = x + dt·v'` 且 `v'` 是同一算子的另一个输出——此时作用于 `a(x)` 路径的
伴随是 `λv' + dt·λx'`（而非裸的 `λv'`）。此类错误在单步上很小、随轨迹长度**复合放大**，
primitive 级单步逐坐标 FD 测试可可靠隔离（`scenarios::chain_one_step_per_coordinate_fd`）。

#### 4.3.2 物理算子示例：ABA 的一步

```rust
/// ABA 中单个 body 的更新算子
pub struct AbaBodyStep;

impl<S: Scalar> CustomOp<S> for AbaBodyStep {
    fn num_inputs(&self) -> usize { 6 } // I_A, p_A, S, v, c, X_T
    fn num_outputs(&self) -> usize { 4 } // I_a, p_a, u, D

    fn forward(&self, inputs: &[S]) -> (SmallVec<[S; 4]>, SmallVec<[S; 8]>) {
        // 标准 ABA 单步计算（纯 f64，不经过 AD）
        // 同时保存反向所需的中间量（U, D, I_a, p_a 等）作为残差
        aba_body_step_forward(inputs)
    }

    fn backward(&self, residual: &[S], grad_output: &[S]) -> SmallVec<[S; 4]> {
        // 手动推导的 ABA 单步反向
        aba_body_step_backward(residual, grad_output)
    }
}
```

**为什么这样做**：ABA 的递归反向传播如果交给 AD 自动处理，需要存储每一步的所有中间量，内存开销随 body 数量线性增长。手动定义 `backward` 后，AD 库只需在每一步之间做链式组装，内存开销可控。这类似于 DiffTaichi 的 source-code-transformation AD 和 Newton 的 tape-based AD 的权衡——前者通过源码变换避免 tape 开销，后者接受 O(T) 内存换取灵活性。

**定量验证这个决策**（估算，Native 记录约 48–64 B/条，见 §5.4）：

| 接入方式 | 每步 tape 增量 | 1000 步轨迹 tape 总量 |
|----------|---------------|---------------------|
| ABA 封装为 CustomOp（含残差） | ~200 条记录 ≈ 13 KB | **~13 MB（可行）** |
| AD 穿透 ABA 递归（逐步展开） | ~5×10³ 条记录 ≈ 320 KB | **~320 MB（不可行）** |

结论：**自定义算子封装不是优化项，而是长轨迹下的必要项**；checkpoint（§4.4）解决 10⁴ 步以上的场景。

#### 4.3.3 隐式求解器的可微：IFT / 伴随模式（可微物理的核心 trick）

**问题**：隐式积分器、接触求解器（LCP/MLCP）、约束投影的本质都是"迭代求解 `r(x; θ) = 0`"，其中 θ 是可微输入（参数/上一时刻状态），x 是解。两种朴素做法都不可行：

1. **展开迭代入带**：tape 随迭代次数爆炸，且梯度带"截断偏差"（迭代没收敛到底时梯度不正确）；
2. **不对求解器求导**：梯度直接错误。

**方案**：把整个 `solve(θ) → x*` 封装为**一个**自定义算子，用隐函数定理（IFT）推导其 backward——这正是 Neural ODE 伴随方法（Chen et al., 2018）在求解器上的对应物：

对 `r(x*(θ), θ) = 0` 两边求导得 `∂r/∂x · ẋ + ∂r/∂θ = 0`，即 `ẋ = -(∂r/∂x)⁻¹ · ∂r/∂θ`。给定 loss 对解的伴随 `ḡ = ∂L/∂x*`，反向只需：

1. **解一个线性伴随系统**：`(∂r/∂x)ᵀ λ = ḡ`（一次矩阵转置求解，不迭代、不入带）；
2. `θ̄ = -(∂r/∂θ)ᵀ λ`。

```rust
/// 隐式求解算子：solve(θ) 满足 r(x; θ) = 0
pub struct ImplicitSolve<R> {
    residual: R,          // 残差定义 r(x; θ)
    solver_cfg: SolverCfg,// 迭代求解器配置（Newton / GS 等，迭代次数固定，见 §4.4.5）
}

impl<S: Scalar> CustomOp<S> for ImplicitSolve<R> {
    // forward(inputs = θ):
    //   1. 用迭代求解器求 x*（普通 f64，不入带）
    //   2. 残差 = (θ, x*, 组装 ∂r/∂x 所需的信息)
    // backward:
    //   1. 解 (∂r/∂x(x*,θ))ᵀ λ = ḡ     —— 一次线性求解
    //   2. 返回 grad_θ = -(∂r/∂θ(x*,θ))ᵀ λ
}
```

**收益**：内存与迭代次数无关（O(1)）；梯度不受迭代截断偏差影响；反向成本 = 一次线性求解 + 两次 Jacobian 乘法。

**∂r/∂x、∂r/∂θ 从哪来**（三选一，库都支持）：

1. 物理引擎手写（最快，物理引擎通常已有解析 Jacobian）；
2. 用**本库**对残差函数 r 做反向、逐列构造（开发期验证用，昂贵）；
3. 用内部双数模块（§4.5.4）做逐列前向 JVP（比逐列反向便宜，量级 n 次前向）。

**风险与边界**：

- `∂r/∂x` 病态（高刚度接触）时伴随解不可靠——提供条件数探针接入 `ad-verify`（§4.5.3）；
- 多解分支：迭代求解器从不同初值可能收敛到不同解，反向推导假设解光滑依赖于 θ。快照必须保存求解器初值/warm-start（§4.4.5），否则 checkpoint 重算可能落到另一分支，梯度不一致。

**定位**：IFT 模式与 ABA 封装并列，是本库面向物理的两大一等公民模式。里程碑 2 交付一个完整示例（隐式欧拉弹簧-阻尼系统）。

#### 4.3.4 向量/矩阵的最小支持：逐元素 vs 整体（bulk）算子

第一版承诺"标量 + 最小向量/矩阵"，需要明确**接入方式的选择规则**，否则用户逐元素写线性代数会让 tape 爆炸：

| 操作 | 复杂度 | 接入方式 | 理由 |
|------|--------|---------|------|
| 逐元素 add / mul / sin | O(n) | 标量 AD（`Vec<AD<S>>` 逐元素） | n 条 Native 记录，可接受 |
| dot(a, b) | O(n) | **bulk CustomOp**（forward 保存 a、b；backward：ḁ += ḡ·b） | 1 条记录 vs n 条 |
| 矩阵-向量 M(q)·v | O(n²) | **bulk CustomOp**（物理引擎给 VJP） | 100×100 逐元素 ≈ 10⁴ 条记录 vs 1 条 |
| Cholesky / 线性求解 | O(n³) | **bulk CustomOp 或 IFT**（§4.3.3） | 同上，且差分病态 |
| 范数 ‖x‖₂ | O(n) | bulk（ḁ = ḡ·x/‖x‖；x=0 未定义，文档化） | 接触法向、约束违反量常用 |

**数据布局决策**：第一版提供 `Vec<AD<S>>`（array-of-structs）的逐元素便利接口——简单、与标量 AD 完全一致；**不引入 SoA 张量类型**（`values: Vec<S>` + 单节点 ID），因为 O(n²)+ 的操作已被 bulk 原则挡在带外，逐元素路径的 SIMD 收益不值得第一版付出类型系统复杂度。`ad-ops` 直接内置 dot / axpy / norm2 三个 bulk 算子作为示范。

---

### 4.4 `ad-checkpoint`：检查点机制

#### 4.4.1 问题定义

反向模式 AD 的 tape 长度与前向传播的操作数成正比。对于 10⁴+ 步的物理仿真，即使核心动力学已封装为自定义算子（§4.3.2），tape 仍会极大（13 KB/步 × 10⁴ ≈ 130 MB，尚可；含接触求解则数倍）。**检查点（checkpointing）**的核心思想是：不存储所有中间结果，而是定期存储状态快照，反向传播时从最近快照重算前向。

`scirs2-autograd` 的 checkpointing 可以减少 50-80% 的内存。PyTorch 的 `torch.utils.checkpoint` 也采用类似的 recompute 机制，其非 reentrant 实现利用 autograd engine 的 pack/unpack hooks 来避免手动存储前向输入。

#### 4.4.2 调度算法

v0.2 只列了"每 K 步 / sqrt(n)"两种策略名。这里把算法学补全（实现状态见 §12.3 第 17 条）：

| 策略 | 快照/状态内存 | 峰值段 tape | 重算开销 | 适用 |
|------|---------|---------|------|------|
| **Uniform（interval K）** | n/K 个快照 | ≈ n/K | 每步恰好一次（1× 前向） | 简单可控、默认推荐 |
| **Nested（budget m）**：二分嵌套反转（Revolve 思想的 tape 变体） | live 状态 m+1 个 | ≈ n/2^m | ≈ (m+1)/2 × 前向（递归 T(n,m) = ⌈n/2⌉ + T(⌊n/2⌋,m-1) + T(⌈n/2⌉,m-1)） | 状态大而单步 tape 小（快照比 tape 贵）；内存-重算旋钮 |
| **Online**（Stumm–Walther 思路） | 固定预算 m 个快照（保留最近 m 个） | ≈ n/m | 1× 前向 | 流式 rollout：MPC 滚动优化、轨迹长度未知 |

经典 Griewank–Walther **二项式调度**（Algorithm 799）最小化的是**纯伴随反转**（无
tape，每步局部 Jacobian 即用即弃）下的总前向次数，其 B(k, m-1) ≥ n-k 的窗口选择
不适用于 tape 架构——tape 变体改为平衡二分（见 §12.3 第 17 条的推导与实测）。

#### 4.4.3 接口设计（修正版）

v0.2 的 `restore() -> Option<(&Snapshot, &mut Context)>` 签名不能工作（返回自己不持有的 `&mut Context`），且 `Box<dyn Any>` 状态无法表达"如何重算"。修正为**由物理引擎实现可重算状态机**、管理器持有并调度：

```rust
/// 由物理引擎实现：完整、确定性的可重算仿真状态机
pub trait Recomputable {
    /// 完整动态状态：q, qd、接触状态、求解器 warm-start、RNG 状态等
    /// （漏掉任何一项，重算就会发散，见 §4.4.5）
    type State: Clone;

    fn save_state(&self) -> Self::State;
    fn load_state(&mut self, state: &Self::State);

    /// 从当前（已 load）状态确定性前进 1 步。
    /// 重放模式下：内部用 ctx.var 重建状态变量后 call_custom，使该步重新入带
    fn step(&mut self, ctx: &mut Context<f64>);

    /// 本步的状态向量如何以 AD 变量形式暴露（边界对齐用，见 §4.4.4）
    fn state_vars(&mut self, ctx: &mut Context<f64>) -> Vec<AD<f64>>;
}

pub enum CheckpointStrategy {
    /// 每 K 步存一次
    Uniform { interval: usize },
    /// 二项式调度（Revolve）；snapshot_budget 取 √n 即平方根策略
    Binomial { snapshot_budget: usize },
    /// 在线调度：固定内存预算，适合流式/未知长度 rollout
    Online { snapshot_budget: usize },
    /// 用户自定义调度表（step 索引列表）
    Custom(Box<dyn Fn(usize) -> Vec<usize>>),
}

pub struct CheckpointManager<R: Recomputable> { /* strategy, snapshots, 边界伴随表 */ }

impl<R: Recomputable> CheckpointManager<R> {
    pub fn new(strategy: CheckpointStrategy, sim: &mut R) -> Self;

    /// 前向传播循环中每步调用：按策略决定是否保存
    /// 快照内容 = (step, tape_len, sim.save_state())，不存 tape 本身
    pub fn maybe_checkpoint(&mut self, ctx: &Context<f64>, sim: &R, step: usize);

    /// 分段反向：从最后一个快照出发，逐段 重算 → 局部反向 → 传递边界伴随（§4.4.4），
    /// 结束后叶子梯度与全 tape 反向一致（§4.5.6）
    pub fn backward(&mut self, ctx: &mut Context<f64>, sim: &mut R, loss: &dyn Fn(&mut Context<f64>) -> AD<f64>);

    /// 反向完成后清理快照
    pub fn clear(&mut self);
}
```

**关键约束**：快照必须包含**完整的、决定性的物理状态**（q、qd、接触状态、warm-start、RNG 种子），否则重算出错。这也是放弃 `Box<dyn Any>`、改用关联类型 `State` 的原因——类型系统强迫实现者把"状态是什么"写成一个具体结构体，RNG/接触这类"隐藏状态"更容易被想起来。`scirs2-autograd` 的 checkpoint 支持 adaptive 策略（基于张量大小阈值）和 checkpoint groups（多输出操作），这些设计值得借鉴。

#### 4.4.4 分段反向与边界伴随传递（最容易写错的部分）

分段反向的流程：

1. **最后一段**：restore 最后一个快照，重算到轨迹末端（该段重新入带），构建 loss，对该段 tape 反向；
2. **边界伴随**：反向到段起点后，该段输入状态变量（= 上一段输出状态）的伴随值收集起来；
3. **前一段**：restore 前一个快照，重算该段（重新入带），将步骤 2 收集的伴随值作为**种子**注入该段起点状态变量（`backward_from` 语义），对该段反向；
4. 重复直到第一段，叶子梯度累加完成。

两个易错点，必须由管理器（而非用户）统一处理：

- **对齐按位置而非 NodeId**：重算生成的 tape 节点 ID 与原前向不同。边界伴随必须按"（step，状态分量下标）"对齐，通过 `Recomputable::state_vars` 声明的固定顺序状态向量完成映射；
- **叶子参数不需快照**：θ（质量、控制序列等叶子）全程不变，只有动态状态随步演进。快照只管动态状态，叶子梯度在各段中自然经由重算路径累加。

#### 4.4.5 确定性重算约束（正确性前提）

分段反向假设"从快照 S 重算 k 步得到的 tape 与原前向对应段**逐位一致**"。这把确定性要求加在物理引擎身上，必须文档化为接入规范：

1. **求解器迭代次数固定**——不允许容差早退（收敛判据依赖的状态必须全部入快照），或改用 IFT 模式（§4.3.3）把迭代整体封在算子内；
2. **RNG 状态入快照**——重算前恢复，保证扰动/采样序列一致；
3. **无并行归约顺序抖动**——段内若有并行计算，归约顺序必须固定；
4. **浮点环境一致**——同一二进制、同一编译参数下重算才保证 bit-exact（跨平台/跨 fast-math 设置不承诺）；
5. **接触求解器初值（warm-start）入快照**——否则重算可能落到另一解分支（§4.3.3）。

---

### 4.5 `ad-verify`：梯度验证

#### 4.5.1 有限差分验证

```rust
/// 梯度验证器
pub struct GradientChecker {
    /// 差分步长
    pub eps: f64,
    /// 相对误差容差
    pub rel_tolerance: f64,
    /// 绝对误差容差
    pub abs_tolerance: f64,
}

/// 验证结果
pub struct CheckResult {
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub passed: bool,
    pub details: Vec<CheckDetail>,
}

impl GradientChecker {
    /// 中心差分验证标量函数
    /// f: R^n -> R, 输入 x: &[f64]
    pub fn check_scalar<F>(
        &self,
        f: F,
        x: &[f64],
        analytical_grad: &[f64],
    ) -> CheckResult
    where F: Fn(&[f64]) -> f64;

    /// 随机方向验证（适合高维输入：沿随机方向做一维差分，与方向导数对比）
    pub fn check_random_direction<F>(
        &self,
        f: F,
        x: &[f64],
        analytical_grad: &[f64],
        n_directions: usize,
    ) -> CheckResult
    where F: Fn(&[f64]) -> f64;

    /// Taylor 余项测试（实现期采纳的科学计算社区标准验收法，dolfin-adjoint /
    /// Firedrake / pyadjoint 实践）：
    /// ratio(h) = |J(x+hδ)−J(x)−h·⟨∇J,δ⟩| / |J(x+hδ)−J(x)| = O(h)，
    /// 估计收敛阶并报告。同时约束前向值与梯度自洽，无绝对容差魔法数；
    /// 错误梯度（系数/方向错）与非光滑点都表现为 order ≈ 0。
    pub fn taylor_test<F>(
        &self, f: F, x: &[f64], grad: &[f64], direction: Option<&[f64]>,
    ) -> TaylorReport
    where F: Fn(&[f64]) -> f64;
}
```

#### 4.5.2 非光滑点的验证策略

在非光滑点（如 ReLU 的 0 点、接触切换点）附近，中心差分可能返回 0 或错误值。`∇Fuzz` 项目提出了一种采样策略：在点 x 附近采样 N 个随机邻居（`x + uniform(-δ, +δ)`，δ 默认 10⁻⁴），检查邻居的梯度和输出是否与点 x 一致。如果不一致，则判定该点非可微，过滤掉这个假阳性。

AD 库的验证工具应支持这种**可微性检查**，区分"梯度算错了"和"该点本身不可微"。

#### 4.5.3 梯度条件分析（物理仿真特有）

有限差分只验证"梯度是否算对"，不验证"梯度是否有用"。物理仿真的长轨迹中，梯度可能算对但条件数极差。研究表明（Howell et al., ICML 2022），可微仿真器的梯度即使数值正确，也可能与有限差分/随机扰动的经验梯度方向差异巨大，导致基于梯度的优化在接触丰富的场景失效——这正是需要健康度度量而不仅是正确性度量的原因。

```rust
/// 梯度健康度分析
pub struct GradientHealth {
    /// 梯度 L2 范数
    pub norm: f64,
    /// 梯度方向与有限差分方向的余弦相似度
    pub cosine_similarity: f64,
    /// 梯度中零元素占比
    pub zero_fraction: f64,
    /// 梯度中非有限值占比
    pub nonfinite_fraction: f64,
}

impl GradientChecker {
    /// 分析梯度健康度（不要求逐元素对齐）
    pub fn analyze_health(&self, grad: &[f64]) -> GradientHealth;

    /// 检查梯度是否随轨迹长度发散/消失
    pub fn check_trajectory_stability(
        &self,
        grads_per_step: &[Vec<f64>],
    ) -> TrajectoryStability;
}
```

配套：反向时按步记录叶子梯度的"逐步范数轨迹"（checkpoint 分段反向天然提供分段边界），`check_trajectory_stability` 据此判断 vanishing / exploding / 振荡三种模式。

#### 4.5.4 双数 oracle 与复步微分（新增的验证武器）

有限差分有步长两难（大步长截断误差、小步长舍入误差），且在非光滑点附近不可靠。两个更高精度的交叉验证手段：

**1. 双数（dual number）前向 oracle**——内部实现（feature `test-oracle`），不对外承诺前向模式 API：

```rust
/// 仅测试/内部使用：f(x) 求值同时得到 f'(x)·方向
#[derive(Clone, Copy)]
pub struct Dual { pub v: f64, pub d: f64 }

// 与 AD 相同的算子集在 Dual 上实现（前向模式极简、可信度高）：
// (f + g)' = f' + g'，(f·g)' = f'g + fg'，...
```

用途：反向模式对任意计算图的梯度 vs 双数 JVP，逐输入比对，容差可达 1e-12（无减法消除误差）。前向模式实现简单到"很难写错"，是反向模式的理想 oracle。它同时服务于 IFT 模式的 ∂r/∂x 逐列构造（§4.3.3）——一鱼两吃。

**2. 复步微分（complex-step，Martins et al., 2003）**：

`f'(x) ≈ Im(f(x + i·h)) / h`，h 可取到 1e-20 量级，**机器精度**且无步长两难、无减法消除。限制：仅适用于解析（实解析/全纯）函数——abs、relu、min/max 等非解析函数不适用（复数分支切割）。用于 Primitive 级（§5.1 第一层）验证每个基础算子的局部 Jacobian，比中心差分灵敏一个数量级以上。

#### 4.5.5 随机表达式属性测试

固定用例覆盖有限，用属性测试生成随机计算图做交叉验证：

```rust
proptest! {
    #[test]
    fn reverse_matches_dual_oracle(expr in random_expr_dag(depth 1..6, width 1..4, ops any)) {
        // 同一随机表达式分别以 AD（反向）与 Dual（前向）在随机点上求值，
        // 逐输入比较 ∂f/∂xᵢ，容差 1e-10；同时在光滑子集上与复步微分三向比对
    }
}
```

`random_expr_dag` 生成的表达式覆盖：嵌套深度、多分支扇出（同一子表达式被多次引用——验证梯度**累加**语义）、常量混合、算子全表。这是抓"累加遗漏""常量入带错误""雅可比排错位"这类结构性 bug 的最有效手段。

#### 4.5.6 检查点一致性验证

checkpoint 分段反向的正确性单独验证：

1. **一致性**：同一轨迹，分段反向 vs 全 tape 反向，叶子梯度应**逐位一致**（bit-exact，前提是满足 §4.4.5 确定性规范）；不满足确定性规范的引擎兜底容差 1e-5；
2. **策略无关性**：随机扰动快照间隔（Uniform K=7/13/29、Binomial、Online 混用），结果都应一致——验证边界伴随传递与调度解耦；
3. **段边界特化用例**：loss 恰好引用段边界状态变量、段内扇出到后续多段引用等边界情况。

---

### 4.6 端到端使用示例（目标 API 形态，示意）

```rust
use ad::prelude::*;   // AD, Context, CheckpointManager, CheckpointStrategy, call_custom

const T: usize = 10_000;

fn main() {
    let mut ctx = Context::new();          // 安装为线程局部（§2.4）

    // 1) 可微输入（叶子）：摆长、初始角、控制序列
    let (l, vl) = ctx.var(1.0);
    let (th0, vth0) = ctx.var(0.5);
    let taus: Vec<_> = (0..T).map(|_| ctx.var(0.01)).collect();

    // 2) rollout：物理引擎实现 Recomputable，每步整体封装为自定义算子
    let mut sim = Pendulum::new(th0);
    let mut ckpt = CheckpointManager::new(CheckpointStrategy::Binomial { snapshot_budget: 100 }, &mut sim);

    for t in 0..T {
        sim.apply_control(taus[t].value);
        sim.step(&mut ctx);                // 内部 call_custom(PendulumStep, ...)
        ckpt.maybe_checkpoint(&ctx, &sim, t);
    }

    // 3) loss：末态角度与目标的平方差
    let loss = {
        let theta = sim.theta_ad();
        (theta - AD::constant(std::f64::consts::FRAC_PI_2)).powi(2)
    };

    // 4) 分段反向（含快照恢复、逐段重算、边界伴随传递）
    let grads = ckpt.backward(&mut ctx, &mut sim, &|ctx| rebuild_loss(ctx, &mut sim));

    println!("dL/dθ0 = {:?}", grads.get(vth0));
    println!("dL/dl  = {:?}", grads.get(vl));

    // 5) 健康度分析（可选）
    let health = GradientChecker::default().analyze_health(&grads.as_slice(vth0, vl, &taus));
}
```

（示意代码：具体签名以实现为准，但展示的**生命周期与职责边界**——ctx 装载、叶子声明、checkpoint 前向钩子、分段反向入口、梯度读取——是设计承诺。）

---

## 5. 正确性保证策略

### 5.1 三层验证体系

| 层级 | 验证对象 | 方法 | 容差 |
|------|---------|------|------|
| **Primitive** | 每个基础算子的局部 Jacobian | 复步微分 + 中心差分，单次调用 | 1e-12（复步）/ 1e-6（差分） |
| **Cross-check** | 任意随机计算图（含扇出、常量混合） | 双数 oracle vs 反向（属性测试） | 1e-10 |
| **Composition** | 短计算图（2-10 个算子） | 中心差分，随机输入 | 1e-5 |
| **Physics Scenario** | 物理仿真的完整梯度 | 有限差分 + 梯度健康度 | 1e-4（相对） |

f32 用户：所有容差放宽至 ~1e-3 相对量级，文档单独给出标定值。

**f32 容差标定**（§12.3 第 30 条落地，`ad-ops/tests/f32.rs`）：

| 验证对象 | oracle | 可达精度 | 标定容差（相对） |
|---------|--------|---------|----------------|
| 算子表（f32 AD vs f64 AD 同点对拍） | f64 路径（已被双数 oracle 验证） | ~1e-7 | **1e-4** |
| bulk 算子（dot / norm2，f32 vs f64） | 同上 | ~1e-7 | 1e-4 |
| TLS 运算符重载 + 自由落体解析解 | 闭式解 | ~1e-6 | 1e-4 |

误差来源：f32 前向中间量的舍入（eps ≈ 1.19e-7）扰动 Jacobian 求值点，
故容差取 eps 的 ~10³ 倍而非机器精度；长轨迹场景下舍入随步数累积，
物理场景建议再放宽至 1e-3 并配合梯度健康度检查（f32 的梯度噪声本身
会抬高健康度基线，应重新标定 vanish/explode 阈值）。

### 5.2 非光滑点的处理策略

| 策略 | 适用场景 | 实现方式 |
|------|---------|---------|
| ReLU 式定义 | 通用激活函数 | 0 点导数定义为 0 |
| 平滑近似 | 接触/摩擦 | 用 smooth ReLU / soft clamp 替代（`ad-ops` 提供） |
| 梯度裁剪 | 反向传播 | `clip_grad_norm` / `clip_grad_value`（§4.2.3）或 `CustomOp::backward` 内裁剪 |
| 采样过滤 | 验证阶段 | 参考 ∇Fuzz 的邻居采样策略（§4.5.2） |

### 5.3 物理场景验证清单

| 场景 | 输入维度 | 验证重点 |
|------|---------|---------|
| 自由落体 | 1（重力） | 梯度与解析解一致 |
| 弹跳 | 2（刚度 k, 阻尼 d） | 接触梯度的有限差分误差 |
| 隐式欧拉弹簧-阻尼 | 3（k, d, Δt） | IFT 梯度 vs 展开迭代 AD 一致（§4.3.3） |
| 铰链摆动 10 步 | 10（控制序列） | 短轨迹梯度精度 |
| 铰链摆动 100 步 | 50 | 梯度健康度（范数是否有界） |
| 多体链 100 步 + 接触 | 20 | 混合场景梯度稳定性 |
| 多体链 10⁴ 步 + checkpoint | 20 | 分段反向 bit-exact 一致、重算开销（§4.5.6） |

### 5.4 性能验证与预算（新增）

**内存预算**（Native 记录按 48–64 B 估算：枚举标记 + 内联 SmallVec）：

| 场景 | tape 增量 | 结论 |
|------|----------|------|
| 1000 步，动力学封装 CustomOp | ~13 MB | 可行，无需 checkpoint |
| 1000 步，AD 穿透递归（错误用法） | ~320 MB | 不可行，验证 §4.3.4 bulk 原则的必要性 |
| 10⁴ 步 + Binomial(√n) checkpoint | 段内 ~13 MB + √n 个状态快照 | 目标形态 |

**性能目标表**（criterion 基准，CI 中跟踪回归）：

| 指标 | 目标 | 测法 |
|------|------|------|
| 标量前向开销 vs 纯 f64 | ≤ 3× | 随机表达式微基准 |
| 反向 / 前向耗时比 | ≤ 1.5× | 同上 |
| Native 记录内存 | ≤ 64 B/条 | 全局分配器统计 |
| CustomOp 主路径（整步动力学）梯度 vs 手写梯度 | ≤ 2× | 摆/小车摆手写解析梯度基线 |
| √n checkpoint 重算总开销 | ≤ 2.5× 单次前向 | 10⁴ 步演示场景 |

### 5.5 长稳与回归测试（新增）

- **tape 复用零泄漏**：`build → backward → clear_tape` 循环 10⁶ 次，RSS 波动 < 5%；
- **tape 增速探针**：`ctx.tape_len()` 暴露给物理引擎做步级断言（"每步增量超预期 = 忘了用 bulk/CustomOp"）；
- **异常检测自检**：构造已知 NaN 源（sqrt(0) 反向、1/0），断言检测器定位到正确记录；
- **no_grad / detach 语义**：区域内外 tape 长度、梯度路径断言；
- **fuzz**：随机操作序列（含 clear/zero_grads/嵌套 no_grad 交错）不 panic、不泄漏。

---

## 6. 里程碑计划

### 里程碑 1：AD 核心骨架（3-4 周）

**交付物**：

- `ad-core` + `ad-ops`
- `AD<S>` 标量、`Context`（线程局部挂载 + `enter` guard）、`Tape`（修正后的 `OpRecord` 枚举）
- 基础算子全表（§4.2.1）+ 常量折叠 + 运算符重载
- `backward` / `backward_from` / `zero_grads` / `clear_tape`、`no_grad` / `detach`
- 异常检测（`set_detect_anomaly`）
- 双数 oracle（`test-oracle` feature）+ proptest 随机表达式属性测试
- criterion 标量基线

**验收标准**：

- 所有基础算子的局部 Jacobian 与复步微分误差 < 1e-12
- 随机表达式属性测试 ≥ 10⁴ 组通过（反向 vs 双数 oracle，1e-10）
- 简单计算图（如 `y = sin(a * b + c)`）的梯度正确
- build→clear_tape 循环 10⁶ 次 RSS 稳定

### 里程碑 2：自定义算子 + 物理适配（3-4 周）

**交付物**：

- `ad-custom` crate
- `CustomOp` trait 及 `call_custom` 接口（多输出 + 残差存储）
- 向量 bulk 算子（dot / axpy / norm2）
- IFT 隐式求解模式 + 隐式欧拉弹簧-阻尼完整示例（§4.3.3）
- 物理算子的参考实现（如 SE(3) 变换、软化接触力）

**验收标准**：

- 用自定义算子实现一个简单的物理计算（如单摆的力矩计算），梯度与有限差分一致
- IFT 梯度与展开迭代的 AD 梯度在光滑、良态情形一致（< 1e-8）
- 多输出算子 + 扇出的属性测试通过

### 里程碑 3：检查点机制（2-3 周）

**交付物**：

- `ad-checkpoint` crate
- `Recomputable` trait、`CheckpointManager`、快照机制
- 三种调度：Uniform / Binomial（Revolve）/ Online
- 边界伴随传递 + 确定性重算接入规范文档（§4.4.4 / §4.4.5）
- 与轨迹循环的集成示例

**验收标准**：

- 100 步轨迹的分段反向与全 tape 反向**逐位一致**（确定性重算下），兜底 1e-5
- 10⁴ 步演示：内存随轨迹长度亚线性增长；√n 调度重算开销 ≤ 2.5× 单次前向
- 快照间隔随机扰动不影响结果（策略无关性，§4.5.6）

### 里程碑 4：梯度健康度分析（2-3 周）

**交付物**：

- `ad-verify` crate
- `GradientChecker`、`GradientHealth`、`TrajectoryStability`、∇Fuzz 式可微性检查
- 逐步范数轨迹记录（配合 checkpoint 分段边界）

**验收标准**：

- 能正确识别梯度消失/爆炸场景
- 有限差分对比工具可用于任意 `Fn(&[f64]) -> f64`

### 里程碑 5：性能固化与压测（1-2 周，新增）

**交付物**：

- §5.4 性能目标表的完整基准报告（criterion + 火焰图）
- 手写解析梯度基线对比（摆 / 小车摆）
- 长稳测试（10⁶ 循环、10⁴ 步轨迹）接入 CI

**验收标准**：

- §5.4 表中全部指标达标；未达标项给出分析结论与后续计划

---

## 7. 技术选型

| 模块 | 推荐 crate | 理由 |
|------|-----------|------|
| 小向量 | `smallvec` | 算子输入通常 ≤ 4，避免堆分配 |
| 数值 trait | `num-traits` | `Scalar` 基础（`Float` 等），极轻量、事实标准 |
| 测试 | `approx` | 浮点比较 |
| 属性测试 | `proptest` | 随机表达式交叉验证（§4.5.5） |
| 随机数 | `rand` | 验证采样、属性测试 |
| 基准 | `criterion` | 性能基准测试 |
| 并行（可选） | `rayon` | 批量 rollout 梯度（仅文档示例，不强制依赖） |
| 序列化 | 无（第一版） | 不需要 |

**不引入**：`nalgebra`、`burn`、`candle`。AD 库应该尽可能底层和轻量，物理引擎自己决定线性代数库。

---

## 8. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| Tape 内存爆炸 | 长轨迹无法反向传播 | checkpoint 机制是必选项，不是可选项；bulk 原则（§4.3.4）+ `tape_len()` 探针及早发现逐元素误用 |
| 物理算子局部 Jacobian 推导错误 | 梯度错误但不报错 | Primitive 级测试覆盖每个自定义算子 + 双数 oracle 属性测试 |
| 非光滑点导致梯度 NaN | 优化不收敛 | ReLU 式定义 + 梯度裁剪接口 + PAP 保证 + 异常检测定位（§4.2.5） |
| 段重算不确定 → checkpoint 梯度不一致 | 分段反向结果错误 | 确定性重算规范（§4.4.5：固定迭代上限、RNG/ warm-start 入快照、固定归约顺序）+ bit-exact 一致性测试 |
| 边界伴随对齐错误 | 分段反向梯度静默错误 | 对齐按（step，状态分量下标）而非 NodeId（§4.4.4）；策略无关性测试（§4.5.6） |
| 线程局部隐式状态误用（跨线程/嵌套泄漏） | 难查的运行时错误 | `AD`/`Variable` `!Send` 标记 + `enter` guard 作用域栈 + fuzz 交错测试（§5.5） |
| IFT 的 ∂r/∂x 病态（高刚度接触） | 伴随解不可靠 | 条件数探针接入 `ad-verify`；文档给出正则/直接法建议（§4.3.3） |
| f32 用户容差误用 | 验证假阴性/假阳性 | 文档分级容差标定（§5.1） |
| 与现有 AD 库重复造轮子 | 工作量浪费 | 本项目目标是"物理仿真专用"，现有库缺乏自定义算子接口和 checkpoint 的物理场景适配 |

---

## 9. 与同类项目的差异化

| 项目 | 语言 | 反向模式 | 自定义算子 | Checkpoint | 物理场景验证 |
|------|------|---------|-----------|------------|-------------|
| PyTorch Autograd | C++/Python | ✅ | ✅ | ✅（`torch.utils.checkpoint`） | ❌（通用） |
| JAX | C++/Python | ✅ | ✅（`custom_vjp`） | ✅（`jax.checkpoint`） | ❌（通用） |
| DiffTaichi | Python DSL | ✅（源码变换） | ✅ | ✅ | ✅（可微物理） |
| Warp (NVIDIA) | Python/CUDA | ✅（`warp.ad`） | ✅ | ✅ | ✅（可微物理） |
| MJX (MuJoCo) | Python/JAX | ✅ | ✅（JAX） | ✅（JAX） | ✅（可微物理） |
| Newton | C++/Python | ✅（tape） | ✅ | 视引擎配置 | ✅（可微物理） |
| tenflowers-autograd | Rust | ✅ | ✅（`CustomOp`） | ❌（未提及） | ❌ |
| gad | Rust | ✅ | ✅（`*Algebra` trait） | ❌ | ❌ |
| scirs2-autograd | Rust | ✅ | ✅（`custom_op`） | ✅（减少 50-80% 内存） | ❌ |
| rustograd | Rust | ✅（tape） | 有限 | ❌ | ❌ |
| num-dual / autodiff (iterate-ch) | Rust | ❌（前向/高阶） | — | — | ❌ |
| Enzyme（Rust 集成推进中） | Rust/LLVM | ✅（源码变换） | ✅ | ❌ | ❌ |
| **本项目** | **Rust** | **✅** | **✅（物理算子专用 + IFT 模式）** | **✅（均匀/Revolve/在线 + 边界伴随）** | **✅（梯度健康度 + bit-exact 一致性）** |

（表中第三方项目的能力描述基于其各自文档，随版本演进可能变化。）

**差异化的核心**：不是"又一个 Rust AD 库"，而是**第一个面向物理仿真场景的纯 Rust、独立、可嵌入的 AD 库**。GPU 系可微物理框架（DiffTaichi、Warp、MJX）证明了这个场景的价值，但它们不是 Rust 库、不可作为底座嵌入 Rust 物理引擎；Rust 现有 AD 库（tenflowers-autograd、gad、scirs2-autograd）都没有针对物理仿真的长轨迹梯度稳定性、非光滑接触和隐式求解器（IFT）做专门设计。本项目正确性保证的独特之处在于包含梯度条件分析与 checkpoint 分段反向的 bit-exact 一致性验证，而不仅仅是有限差分对比。

---

## 10. 参考文献

1. Baydin, Pearlmutter, Radul, Siskind. *Automatic Differentiation in Machine Learning: a Survey*. JMLR, 2018.（前向/反向模式、tape/源码变换谱系）
2. Griewank & Walther. *Evaluating Derivatives: Theory and Practice of Computational Differentiation*. 2nd ed., SIAM, 2008.（checkpointing 的理论基础）
3. Griewank & Walther. *Algorithm 799: revolve*. ACM TOMS, 2000.（二项式/Revolve 调度）
4. Stumm & Walther. 最优在线 checkpointing 算法. SIAM J. Sci. Comput.（流式/未知长度的内存受限调度）
5. Chen, Rubanova, Bettencourt, Duvenaud. *Neural Ordinary Differential Equations*. NeurIPS, 2018.（伴随方法，IFT 模式的同源思想）
6. Hu et al. *DiffTaichi: Differentiable Programming for Physical Simulation*. ICLR, 2020.（可微物理、源码变换 AD、checkpointing 实践）
7. Howell et al. *Do Differentiable Simulators Give Better Policy Gradients?* ICML, 2022.（可微仿真梯度质量/健康度问题）
8. Martins, Sturdza, Alonso. *The Complex-Step Derivative Approximation*. ACM TOMS, 2003.（复步微分验证）
9. ∇Fuzz：面向自动微分库的模糊测试（ICSE 2024 方向工作；非光滑点邻居采样过滤假阳性）
10. PyTorch 文档：`torch.autograd.set_detect_anomaly`、`torch.utils.checkpoint`；JAX 文档：`custom_vjp`、`jax.checkpoint`；NVIDIA Warp 文档：`warp.ad`。

---

## 11. 修订记录

**v0.3（本次）**——在 v0.2 基础上补充与修正：

1. **格式修复**：清理导出产生的 markdown 转义（`\#`、`\*\*`）、HTML 实体与冗余空行。
2. **§2.4（新增）**：Tape 所有权与 API 形态——Rust AD 必须最先回答的决策（Rc / 线程局部 / 显式传参 / 生命周期 branding 对比，定为线程局部 + 显式双路径），含 `Send/Sync` 约定。
3. **§4.1.3（修正）**：`OpRecord` 改为枚举——v0.2 单输出、无处存残差的结构与多输出 `CustomOp` 直接矛盾；删除冗余 `op_order`。
4. **§4.1.4/4.1.5（新增）**：反向传播算法与伴随数组、梯度语义契约（累加/清零/VJP 种子）、`no_grad` / `detach`。
5. **§4.1.6（新增）**：并发模型（单线程单 Context、批量 rayon 约定）。
6. **§4.2（扩充）**：算子表补 tanh/asin/acos/atan2/min/max/clamp/lerp/recip；新增常量折叠、数值边界策略（§4.2.4）、非有限梯度异常检测（§4.2.5）、叶子梯度裁剪工具。
7. **§4.3.3（新增）**：IFT / 伴随模式——隐式积分器与接触求解器的可微，可微物理的核心 trick。
8. **§4.3.4（新增）**：向量/矩阵接入规则——逐元素 vs bulk 算子的选择表与数据布局决策。
9. **§4.4（重写）**：调度算法补全（Uniform / Revolve / Online 对比）；接口修正（v0.2 `restore` 签名不可实现，改为 `Recomputable` 状态机 + 关联类型状态，替代 `Box<dyn Any>`）；新增分段反向的边界伴随传递（§4.4.4）与确定性重算约束（§4.4.5）。
10. **§4.5.4–4.5.6（新增）**：双数 oracle、复步微分、随机表达式属性测试、checkpoint 一致性（bit-exact / 策略无关）验证。
11. **§4.6（新增）**：端到端使用示例。
12. **§5.4/5.5（新增）**：定量内存预算、性能目标表、长稳与回归测试。
13. **§6**：里程碑按上述能力更新（M1 补 oracle/属性测试/异常检测，M2 补 bulk/IFT，M3 补三种调度与 bit-exact 验收，新增 M5 性能固化）。
14. **§7/§8/§9**：技术选型补 `num-traits`/`proptest`/`rand`；风险表补 5 项；对比表补 rustograd、num-dual、Enzyme、DiffTaichi、Warp、MJX、Newton，并把差异化表述修正为"纯 Rust、独立、可嵌入"。
15. **§10（新增）**：参考文献。

**v0.2**——API 设计细化（`AD<S>`、`Context`、`Tape`、`CustomOp`、`CheckpointManager`、`GradientChecker` 雏形，三层验证体系，里程碑 1-4）。

---

## 12. 实现状态与设计偏差（v0.1 实现后回写）

M1–M4 核心能力已实现并通过测试（约 60 个测试，`cargo test --workspace` 全绿）。
以下是实现过程中对本文档的**修正性发现**，已按实现为准回写：

### 12.1 架构层面的修正

1. **§4.4.2 checkpoint 调度：Revolve/二项式暂缓，Uniform 即本架构的内存最优。**
   实现采用"分段 tape 反向"架构：段 = 相邻快照之间；每段重算（重新入带）后立即整段反向，
   段 tape 用毕即弃。该架构下**每步恰好重算一次（总重算 = 1× 前向）**，快照位置只影响
   峰值段 tape 长度——均匀分布即最优（峰值段 ≈ n/m）。经典 Revolve 优化的是嵌套重算下的
   总前向次数，属于不同架构，列为后续工作。已实现：Uniform / Online（保留最近 m 个快照的
   流式调度）/ 自定义闭包。实测 10⁴ 步单摆：前向 1.2 ms + 分段反向 2.2 ms（含重算）。
2. **§4.5.6 checkpoint 一致性：全轨迹 bit-exact 不成立，修正为双重判据。**
   分段反向对叶子梯度的**求和顺序**与全 tape 反向不同（逐段累加 vs 全序遍历），
   多贡献路径的梯度存在 ULP 级差异。修正验收为：(a) 同策略重跑 **bit-exact**（确定性重算下成立）；
   (b) 分段 vs 全 tape 在 1e-10 相对容差内一致。测试覆盖 4 种调度策略 + 重跑一致性。
3. **`CustomOp` trait 落在 `ad-core` 而非 `ad-custom`**：tape 记录需要存储
   `Rc<dyn CustomOp>`，trait 属于 tape 数据模型；`ad-custom` 提供 IFT 模式与线性求解工具。

### 12.2 API 层面的修正与澄清

4. **线程局部路径与显式路径不可混用**：`with_context` 持有 `RefCell` 借用期间再入
   （闭包内用运算符重载）会 panic，并给出明确错误信息（重入防护是特性而非缺陷——
   它阻止了隐式状态错配）。文档 §2.4 方案 B/C 的"双路径"以此为准：表达式级别二选一。
5. **`bind_state` / `state()` 取代 v0.2 的 `state_vars`**：分段反向需要"段输出状态"的
   AD 句柄（作为边界种子）与"段输入状态"的叶子句柄（收集边界伴随），`Recomputable`
   为此提供两个视图，顺序固定即位置对齐锚点（§4.4.4）。
6. **`backward_seeds(&[(AD, S)])` 是原语**：`backward` / `backward_from` 是其特例；
   多种子（loss + 边界伴随同时注入）是分段反向的运行时必需能力。
7. **`AD::node` 收紧为 `pub(crate)`**（提供只读 `node()` 访问器），防止用户手工构造
   悬挂节点；`Variable` 不可在 crate 外构造（含 `PhantomData` 私有字段），跨线程误用
   由 `!Send + !Sync` 编译期拦截。
8. **clamp 边界约定**：与 PyTorch 一致——区间 [lo, hi] 含边界处梯度为 1，界外为 0
   （§4.2.1 表中"越界侧梯度 0"指严格界外）。
9. **`grad_of(AD)`**：按被追踪标量读叶子梯度，checkpoint 边界收集按位置对齐时使用，
   避免向调用方同时暴露两套句柄。

### 12.3 依赖与实现细节

10. **IFT 的残差系统要求方阵**（nr = nx），Newton + 单值分解未做；`∂r/∂x` 经内部双数
    逐列构造（`test-oracle` feature 复用于生产路径，验证了 §4.5.4 "一鱼两吃"的判断）。
    Newton 迭代次数固定（默认 8），满足 §4.4.5 确定性要求；从 x=0 起步要求 J(0) 非奇异，
    使用方需保证（或自行实现 `CustomOp` 接入带 warm-start 的求解器）。
11. **内部 RNG**：`ad-verify` 用确定性 xorshift64*（可复现采样），未引入 `rand` 依赖；
    实际依赖为 `num-traits`、`smallvec`、（dev）`proptest`、`num-complex`。
12. **长循环性能注意**：优化循环必须在每轮调用 `clear_tape()`，否则 tape 无限增长、
    每次 backward 全量遍历退化为 O(n²)（基准示例中实测 10 μs/iter → 119 ns/iter）。
    `tape_len()` 探针（§5.5）即为尽早暴露此类误用而设。
13. **CustomOp backward 的内部边陷阱（§4.3.1 已补契约）**：tape 只看见算子级输入→输出边，
    backward 必须覆盖算子内部全部数据流——包括**输出之间的耦合边**（如 `x' = x + dt·v'`
    中 v' 是本算子的另一个输出：作用于 a(x) 路径的伴随是 `λv' + dt·λx'`）。该错误单步
    极小、随轨迹复合放大，曾被弹跳球场景测试（400 步）与多体链单步逐坐标 FD 测试共同
    抓出；PendulumStep 未踩中是因为其输出只依赖**输入**（θ' = θ + dt·ω）。接入规范要求：
    新算子先过"单步逐坐标 FD"再上长轨迹。
14. **梯度裁剪已实现**（§4.2.3）：`Context::clip_grad_norm`（返回裁剪前范数；非有限范数
    不缩放，交给异常检测）/ `clip_grad_value`。
15. **§5.3 物理场景清单补全**：自由落体（解析解 1e-12）、弹跳球（smooth-relu 接触力 +
    手工 backward vs 前向 FD）、12 体半隐式弹簧链（26 维梯度，随机方向 FD + 健康度）
    已入库（`crates/ad/tests/scenarios.rs`）；单步逐坐标 FD 作为 CustomOp 接入的标准
    隔离器一并提供。
16. **Taylor 余项测试已采纳**（§4.5.1，AD 领域工业验收方法调研后的结论）：
    `ad-verify::GradientChecker::taylor_test` 实现 `ratio(h) = O(h)` 收敛阶判据
    （dolfin-adjoint / Firedrake / pyadjoint 实践），已接入三个物理场景测试。
    参照的工业测试基础设施谱系：跨工具基准 GradBench（MIT ADBench 继任者，无物理
    场景，列为潜在锚点）、PyTorch OpInfo 表驱动回归、Julia ChainRulesTestUtils
    `test_rule`、CUTEst/COPS 端到端问题库——本项目以"双数/复步 oracle + 属性测试 +
    Taylor 余项 + 场景清单"组合覆盖了同等的正确性方法学。
17. **嵌套反转（`Nested { budget }`）已实现**（§4.4.2）：二分嵌套是 Revolve 思想的
    tape 变体——峰值段 tape ≈ n/2^m、live 状态 m+1、重算 ≈ (m+1)/2×（budget=2 时
    2×、=6 时 3.5×，与递归 T(n,m) 的理论值逐步精确吻合）。经典二项式调度的
    B(k, m-1) ≥ n-k 窗口条件针对纯伴随反转（无 tape），不适用于 tape 架构，故取
    平衡二分。实现中踩掉两个隐蔽 bug，均属同一类**契约违反**：`Recomputable`
    的"load_state 后必须 bind_state"在嵌套路径被跳过——top 入口与 Phase C 的
    `load_state` 只刷新标量镜像，后续 no_grad 步进消费的是**过期的 AD 状态**
    （前向 pass 或上一窗口的遗留），梯度静默错误且随 budget 加深。教训已写入
    `ad-checkpoint` crate 文档与 `Recomputable` 契约注释。
18. **`ad-physics` 空间代数算子库已实现**（§4.3.2"物理算子参考实现"的落地）：
    6 个 CustomOp——空间叉积 `crm`、Plücker 运动/力变换、空间惯性作用量 `I·v`、
    空间惯性坐标系变换（6×6 拼装 `X*·I_B·X*ᵀ` + 提取）、SO(3) 指数映射（Rodrigues
    + 小角度分支），全部手写 VJP。验证：单步逐坐标 FD 隔离器逐分量对拍 +
    **动能坐标系不变性**物理先验测试（T 在 motion/force 变换下不变——该测试
    抓住了 force 变换转移项加错分量、惯性 B 块漏乘 m 两处 forward 语义错误）。
    实现期教训（均已写入算子文档）：
    (a) **vee 的符号约定**：`vee([a]×) = [−m₅, m₂, −m₁]`，两个负号漏掉会精确
    破坏 λ 的反对称分量（对称 λ 全过、反对称 λ 全错的诊断特征）；
    (b) **CustomOp 的 backward 必须覆盖算子内部全部数据流**（§4.3.1 契约再次
    应验）：半隐式欧拉的 `x' = x + dt·v'` 内部边，以及惯性变换中 pack 只读
    对称上三角——对称矩阵约定下梯度路由只能走上三角，逐坐标 FD 会强制暴露；
    (c) **直接透传输出（out[6] = m）的梯度不得进入中间量的 λY 路由**：
    RotateInertia 曾把 lm 放入 λY[5][5]，经共轭段产生虚假的 ∂Y55/∂E 依赖
    （Y55 = m 与 E 无关），恰好只污染 E 的第 2 行。
19. **`ad-physics` 接触模型套件已实现**（§5.2"平滑近似"的算子化）：
    `ContactNormalOp`（Hunt–Crossley 型法向力：softplus_ε 光滑单边 + `pen^p·(k−d·ḡap)`
    阻尼，接近时耗散）与 `RegularizedFrictionOp`（正则化库仑摩擦 `−μ·f_n·v_t/√(|v_t|²+ε²)`，
    严格满足摩擦锥）。验证：FD 隔离器逐分量对拍（含 ε 梯度——softplus 尾部对 ε
    的依赖不可省）、非粘着/接近耗散/摩擦锥/高速库仑极限物理先验、弹跳球刚度扫描
    （k 跨 3 个数量级：AD vs FD 一致、Taylor 余项通过、梯度范数随刚度增长但有限——
    设计文档 §4.2.4"高刚度梯度病态"风险的定量基线）。
    实现期教训：**FD oracle 的 loss 必须与 AD loss 逐项一致**——单输出算子时交叉项
    `0.15·o₀²` 曾被 `len ≥ 2` 守卫跳过，产生全坐标一致 1.32× 的系统性偏差
    （所有坐标同乘一个因子的失败模式 = 共同上游 λ 出错）。

20. **端到端轨迹优化基准已实现**（`ad-optim` crate，路线图方向 4）：
    零依赖 Armijo 回溯梯度下降（含坏梯度检测——线搜索连续失败 > 8 次即终止，
    是 AD 坏梯度的特征信号）。两个收敛基准：
    (a) **受控摆杆**（150 维控制序列、150 步纯表达式 rollout、clamp 限幅）：
    55 迭代收敛，θ_T = 1.187（目标 1.2），损失降 55×——正则项使损失下限非零，
    断言按物理结果（末态角误差 < 0.15）而非损失倍数；
    (b) **接触弹跳球**（ContactNormalOp，200 步、优化 v0 与阻尼 d）：
    21 迭代精确命中目标（z_T = 0.12）——穿透接触的梯度在优化中完全可用。
    两组基准均含起点处 FD 抽查与 Taylor 余项检验。证实了项目核心命题：
    **库产出的梯度不仅算得对，而且能驱动轨迹优化收敛**。iLQR（需二阶信息
    或 Gauss-Newton 近似）列为后续工作。
    实现注记：显式 ctx 代码中混用 TLS 算子（`ad::clamp`）会 panic（§2.4
    双路径约束的运行时表现）——基准改用 `clamp_with`。
21. **iLQR 求解器已实现**（`ad-optim::ilqr`，上面"后续工作"落地）：
    Tassa 正则化 iLQR——动力学 Jacobian 由 `Context::backward_seeds` 逐输出
    单位种子取出（每步 nx 次微型反向），代价限定标准二次型（Hessian 解析，
    无需二阶 AD），Q_uu 正则化自适应 + 前向 α 回溯按预期下降验收。
    双积分器（线性-二次）收敛到正则跟踪问题的离散最优；**受控摆杆同一问题
    iLQR 6 迭代 vs GD 55 迭代**（θ_T = 1.1985 vs 1.187），Jacobian 与 GD 的
    VJP 逐列对拍一致。"二阶信息"的实现路径即 §4.5.4 双数 oracle 的对偶用法
    ——前向逐列 Jacobian 或反向 VJP 皆可，无需 Hessian。
22. **推送基础设施**：HTTPS + gh 凭据为可用路径（SSH 需把 `id_ed25519.pub`
    注册到账号或解锁带密码的 `id_rsa`）；OAuth token 推送 `.github/workflows/`
    下文件需 `workflow` scope——CI workflow 以 `.github/ci.yml.pending` 形式
    暂存，scope 授权后移回。
23. **方向 3 落地——双关节摆（Acrobot 构型）混沌 iLQR 基准**：2 连杆点质量摆
    （相对关节角），Lagrangian 推导 M/c/g + 半隐式欧拉，**AD 直通实现**
    （~20 条记录/步，tape 自动 VJP）。验证：能量守恒先验（被动摆 2000 步
    能量振荡有界，漂移 0.38%——动力学方程正确的最严格检验）+ 单步 AD 梯度
    vs FD 逐分量对拍。iLQR 甩摆基准实证了 **Howell et al. 2022 的核心发现**：
    混沌系统 + 弱初始激励下，iLQR 7 迭代收敛到向目标部分移动的局部解
    （loss −43%，θ1_T 从 0.25 向 0.5 方向区域移动但不精确到达）——
    可微仿真的梯度数值正确 ≠ 优化全局收敛，梯度健康度体系（§4.5.3）因此必要。
24. **铰链链手写 VJP 的工程结论**（重要的反面教训）：2 连杆手写 VJP 在
    FD 隔离器 + 能量守恒双重检验下连续暴露 6+ 处错误（vee 符号、Coriolis
    推导把 ∂T/∂θ2 项误入 θ1 方程、重力二角项漏项、透传输出 λ 路由、
    对称打包梯度系数），修正成本远超预期。**工程结论**：此复杂度级别
    （三角耦合 + 多处 θ2 依赖）的动力学算子应优先 AD 直通（本实现），
    手写 CustomOp 保留给 O(n²)+ 的 bulk 操作（§4.3.4 原则的印证）。
    手写版可由直通版 + 机械推导管线后续补齐。
25. **3D 陀螺力学落地**（`ad-physics::gyro::GyroscopicStep`，方向 3 的 3D 深化）：
    自由刚体 Euler 顶方程 `ω̇ = −I⁻¹(ω×Iω)`（对角惯量）的手写 VJP 算子。
    **实现期最大的教训**：forward 加速度符号反了（缺 Euler 方程的负号），
    而动能守恒测试**未能发现**——时间反演对称的动力学，反向积分 KE 同样守恒；
    **角动量守恒测试抓住了它**（L_world = R·I·ω 漂移 13%，且不随 dt 缩小 →
    结构性公式错误的诊断特征）。守恒先验要**多样化**：每个不变量检验动力学
    的不同侧面。VJP 调试同样依赖数值仲裁：对角惯量的 ∂a/∂ω 循环闭式
    （如 ω3(I1−I3)/I2）曾被误写为 I3ω3/I1 等三种变体——数值 Jacobian
    逐项对比一次定位。测试套件：FD 隔离器、KE 守恒（1e-11）、角动量守恒
    （RK2 中点耦合积分，2 s 漂移 < 5e-3）、网球拍定理（中间轴失稳翻转，
    max|ω1| 从 1e-3 增长到 >0.5 而 KE 守恒）。
26. **火焰图性能工程完成**（M5 收尾；Windows 无 perf/dtrace，samply 需
    管理员权限 → 方法论：**计数分配器画像 + cdb poor-man's profiler**
    采样栈聚合，负载驱动 `examples/profile.rs`（按阶段打印 分配次数/字节/
    墙钟，`--loop` 模式供采样）。证据链：分配画像（checkpoint 每跑
    7160 次分配）与 CPU 叶子帧（~35% 在堆分配器、~25% 在 call_custom
    路径 + SmallVec 溢出/析构）互相印证。热点与修复：
    (a) SmallVec 内联容量 4 被 5+ 输入算子击穿（单摆步 5 输入）——
    `CustomOp` forward/backward 与 tape Custom 记录的 inputs/outputs
    内联容量升至 8（So3Exp 9 输出仍溢出，可接受）；
    (b) `call_custom` 返回堆 `Vec<AD>`（含 no_grad 前向）→ 改
    `SmallVec<[AD;4]>`；
    (c) `call_custom` 每次 `Rc::new(op)` → 高频场景缓存 `Rc<dyn CustomOp>`
    走 `call_custom_dyn`（PendulumSim 示范）。
    效果（criterion，1000 步单摆）：分段反向 288→141 µs（−51%）、
    全 tape 177→133 µs（−25%）；分配次数 checkpoint 7160→128（−98%）、
    全 tape 4016→16（−99.6%）；复用路径稳态本就零分配。教训：
    **复用 + clear_tape 的稳态无分配**意味着吞吐瓶颈全部在前向记录/
    自定义算子调用路径的意外堆分配——分配画像的"次数"比墙钟更早暴露。
27. **手写 CustomOp 版双关节摆补全**（`ad-physics::chain::DoublePendulumStep`，
    第 24 条工程结论的修正性补充——手写版从"不可行"变为"按策略可行"）：
    与 AD 直通版（第 23 条）数值等价的单算子形态，~1 条记录/步（直通版
    ~40 条）。**VJP 推导策略改进是关键**：不再一揽子展开三角耦合（第 24 条
    的失败模式），而是按中间量分解、与 tape 链式路径同构——
    `a = M⁻¹r` 段用 `s = M⁻¹λa`（M 对称）、`λr = s`、`λM_ij = −s_i·a_j`
    （δM 对称 ⇒ m12 梯度取全和，§4.6），`r = τ−c−g` 段逐变量局部偏导，
    半隐式欧拉内部边（§4.3.1）单独一层。FD 隔离器首跑即抓出 2 处错误
    （第 24 条 6+ 处 → 2 处）：`s12 = sin(θ1+θ2)` 的 **θ1 依赖漏项**
    （重力双角项进两个方程——第 24 条"重力二角漏项"家族重现），以及
    λdt 需用 **λω'(总计)**（θ' = θ + dt·ω' 使 ω' 是 θ' 的上游，内部边
    的 dt 直接边贡献含 λθ'·dt 分量）。验证四层：FD 隔离器（4 状态 ×
    7 坐标）、能量守恒（2000 步漂移 0.38%）、单步梯度与 Jacobian 对拍
    AD 直通（1e-9）、**iLQR 甩摆基准逐位一致**（loss 2.7839 → 1.573964，
    7 迭代收敛，θ1_T = 0.2246——与直通版完全相同）。工程结论更新：
    三角耦合动力学算子手写 VJP **可行但必须**（i）按中间量分解推导，
    （ii）FD 隔离器先行——隔离器一次运行的修正成本远低于长轨迹调试。
28. **工具链与场景补全 + 一个潜伏 bug**（§4.3.3/§4.3.4/§5.3/§5.4 的收尾轮）：
    (a) **条件数探针兑现**（§4.3.3 风险条目引用的 `ad-verify` 能力此前并不
    存在）：`condition_number_inf`（κ∞ = ‖A‖∞·‖A⁻¹‖∞，Gauss–Jordan 求逆，
    奇异返回 inf），`ad-custom` 文档接线 + Hilbert 病态用法示范测试；
    (b) **`matvec_with`**（§4.3.4 bulk 表 M·v 行补全，rows 条输出 = 1 条记录）
    ——其"部分追踪"测试（M 常量在前、v 在后）**首踩一个潜伏 bug**：Custom
    记录反向此前用 zip 按位置配对 tracked 子集与 gins，当常量输入占**非尾部**
    槽位时梯度错路由（既有算子的常量恰在尾部故从未暴露）；修复为按 slot
    索引取 gins。教训：**"恰好对齐"不是正确性证据**——新算子的非对称追踪
    形态（部分输入常量）应成为 FD 隔离器的标准用例之一；
    (c) **Custom 记录瘦身**：输出节点由 NodeId 单调分配保证连续，记录只存
    `output_base`（336→272B），反向直接传 `adjoints` 的连续切片（无收集
    拷贝）。全 tape 单摆 133→73 µs、分段 141→106 µs（较优化前基线累计
    −59% / −63%）；CustomOp 单发调用 1.0→0.59 µs；
    (d) **iLQR 工程**：rollout 与前向线搜索的 Context 跨步复用 + 缓冲区就地
    重填（消除每步 `Context::new` 分配热点）；Q_uu_reg 每步 LU 分解一次，
    k 与 K 的 nx 列右端共享（nx+1 次回代替代 nx+1 次 O(n³) 重分解），数值
    与旧路径逐位一致（双摆 iLQR loss 2.7839→1.573964 不变）；
    (e) **§5.3 末行场景补全**：10 体 20 维链 × 10⁴ 步、Uniform{interval:100}
    checkpoint——分段 vs 全 tape 1e-10 一致、同策略重跑 bit-exact、
    重算恰好 1×（steps_executed = 2T）、快照数 = T/100；
    (f) **§5.4"CustomOp 主路径 vs 手写梯度 ≤ 2×"验收：实测 ≈ 2.4–3.0×，
    未达标（记录偏差分析）**。新 bench `chain_step/*` 以手写伴随递推
    （同一 VJP 闭式、无 tape/无节点分配）为基线：1k 步稳态 130.6 vs
    43.7 µs。偏差构成：逐记录固定开销（tape push ~270B、伴随数组往返、
    叶子累加）约 90 ns/步，而 1 条记录/步的动力学算子 VJP 仅 ~44 ns——
    **"最小算子"恰是该指标的最坏情形**（框架开销占比最大化）；
    §4.3.4 的 O(n²)+ bulk 算子 VJP 随 n² 增长，比值趋近 1×（设计意图的
    主战场）。后续计划：OpRecord arena 分配器、Custom 变体存储再评估。
29. **IFT 泛化**（§4.3.3 第 10 条"方阵限制/Newton+SVD 未做"的收尾）：
    `Residual` 增加默认 `nr()`（方阵默认不变，向后兼容）——
    (a) **超定系统**（nr > nx）：要求 J_x 满列秩且相容（解流形存在），
    forward 走 Gauss–Newton 正规方程 `(JᵀJ + μI)Δx = -Jᵀr`（Tikhonov 阻尼
    μ 默认 0），伴随 `λ = J(JᵀJ)⁻¹ḡ` 为最小范数伴随（方阵时与 `J⁻ᵀ` 数学
    等价，方阵路径保持原 `solve_linear_transposed` 数值不变）；**欠定**
    （nr < nx）构造时 panic（解不唯一，IFT 不适用——显式拒绝优于静默
    最小范数解）；
    (b) **warm-start 作为显式输入槽位**（`with_warm_start`，inputs =
    [θ..., x0...]）：残差不依赖 x0（收敛解与初值无关）→ x0 槽位梯度恰为
    0，但迭代路径依赖 x0——这正是 §4.4.5"快照必须保存求解器初值"的
    tape 原生形态（checkpoint 重算天然保存并重放）。实现期发现：立方
    残差 x³ = θ 的冷启动 x0 = 0 恰是 Newton 奇异点（J = 3x² = 0），
    迭代卡死——初值属于收敛性行为的一部分，不是可有可无的优化；
    (c) `jacobian_x` 公开，配合条件数探针（第 28a 条）做病态诊断；
    (d) 测试：冗余约束对 + 极坐标超定（3 约束 2 未知量，c/s 叶子经 tape
    超越算子产生——展示 **IFT 算子与 tape 表达式可组合**，φ 的梯度经
    c/s 链式回传）FD 对拍、warm-start 跨初值一致性 + bit-exact 重算、
    欠定拒绝（should_panic）。
30. **iLQR box 约束与 MPC 滚动形态**（§12.3 第 21 条 iLQR 的能力补全）：
    (a) **clamped iLQR**：`IlqrCfg` 增 `u_min/u_max`，前向 pass 钳制
    u_new，backward pass 不感知边界——主动约束期 ΔV 预估偏乐观，由 α
    回溯的验收环节兜底（完整的 box-DDP 需逐步 QP，列为后续工作）；
    自标定测试：先解无约束问题取峰值控制定界（60% 峰值保证约束主动），
    验证界满足 + 有界解仍改进 + 有界损失 ≥ 无约束损失（可行性排序）；
    (b) **MPC 滚动时域**：每步解 horizon 上的 iLQR、只施加首控制、真
    动力学前进一步（`rollout` 单步）——§4.4.2 Online 行"MPC 滚动优化"
    承诺场景的 iLQR 侧对应物。双积分器测试：30 真实步 × horizon 15，
    |u| ≤ 0.5 下终点误差 < 0.05、全程界内。
31. **f32 路径覆盖与容差标定**（§8 风险表"f32 容差误用"的落地）：
    算子表对 `S: Scalar` 全泛型，f32 验证策略 = **同一泛型函数在 f64
    （已被双数 oracle 验证）与 f32 下同点对拍**（f32→f64 无损升采样保
    证同一数学点）。标定：单算子 1e-4 相对（f32 前向舍入 eps≈1.19e-7
    扰动 Jacobian 求值点，容差取 eps 的 ~10³ 倍；表见 §5.1）。覆盖：
    12 个一元 + 4 个二元算子、bulk（dot/norm2）、TLS 运算符重载路径、
    自由落体解析解 f32 形态。`CustomOp` 目前 f64 特化（trait 即 `f64`
    实例），f32 物理算子列为后续工作。
32. **rayon 批量示例**（§4.1.6/§7"文档示例、不强制依赖"的兑现）：
    `examples/batch_rollout.rs`，feature `rayon`（`required-features`
    门控，库本体零新增依赖）。演示并发模型纪律：闭包只捕获普通数值、
    每任务线程内自建 Context + 仿真（`AD`/`Context` 均 `!Send`，编译期
    阻止跨线程共享）、只把梯度数组送回主线程聚合。576 任务 × 1000 步
    checkpoint 分段反向 7.4 ms（12.8 µs/条）。
33. **`CustomOp` 契约验证器成为公开 API**（`ad_verify::op_check::
    validate_custom_op`，§4.3.1 测试模式的 API 化）——物理引擎作者手写
    VJP 后一行调用：前向确定性（逐位）+ VJP gins 长度契约（原为
    `debug_assert`，release 下消失）+ **四种追踪形态**逐坐标 FD 对拍
    （常量在头部/尾部/隔位——第 28b 条路由 bug 的形态由此固化为标准
    检查项）。库内 7 个算子（空间代数 6 + 双摆步 + 接触 2）全数通过；
    自验证测试刻意复刻三类破坏（zip 错路由、gins 短一截、forward
    不确定），全部被抓。**validator 自己也立竿见影**：第 36 条 fuzz
    算子的手写 VJP（2a²b 求导漏乘系数 2）首跑即被其抓住。
34. **接触求解器的 IFT 完整形态**（§4.3.3 动机难题的闭环；active-set +
    IFT 模式）：互补条件的光滑性由活动集承载——活动集稳定时对当前
    活动约束系统用 `ImplicitSolve` 求解并穿透求导（梯度域内精确），
    warm-start 承载上一时刻的活动集/冲量（IFT 梯度域内有效、checkpoint
    安全）。四个验证（oracle = 对求解器本身 FD——实证"IFT 伴随无迭代
    截断偏差"）：1-DOF 非线性接触（Hertz 型 kδ+cδ³）活动分支梯度、
    2×2 法向冲量、**3×2 冗余超定**（静不定接触，第三约束 = 前两个的
    凸组合，走正规方程 + 最小范数伴随，解与方阵一致）、warm-start λ
    跨时间步。实现期自踩第 19 条教训的现行重演：测试辅助 loss 漏加
    o0 二次项 → AD 与 FD oracle 不逐项一致 → 全坐标偏差——**教训的
    修法（两路共用同一 loss 函数）比教训本身更重要**，已固化为
    `ad_verify::op_check::mixed_output_loss`。
35. **健康度接入优化器输出**（§4.5.3 承诺"梯度健康度"与优化器的闭环）：
    iLQR 报告新增 `quu_cond_max`（全程 κ∞(Q_uu_reg) 最大值——控制通道
    二阶信息病态的直接读数）、`mu_final`（正则化被推高 = 实际下降达
    不到预期）、`line_search_rejections`（混沌接触特征）；GD 报告新增
    `grad_health`（范数/零占比/非有限占比画像）。验证：冗余双控制
    双积分器 R 从 1.0 压到 1e-4，κ(Q_uu)_max 显著增长（nu=1 时 Q_uu 是
    标量、条件数恒 1——健康度指标需要足够的问题维度才可观测）；坏
    梯度（方向翻转）被 Armijo 拒收信号捕捉。
36. **CustomOp 随机图 fuzz**（基础算子 proptest 的组装层补全）：
    3 个手写 VJP 算子（2×2 / 3×2 含零 Jacobian 槽位 / 6→1 bulk 形态）
    掺入 24 步随机 DAG，512 例反向 vs 双数 oracle。关键设计：
    (a) 输入槽位 25% 概率被常量替换（任意槽位——部分追踪形态）；
    (b) **先规划后双路执行**（tape 与 oracle 见到同一张图）；
    (c) 值模拟 + 2 的幂缩放步把 |v| 压回界内——无界图使梯度在 ~1e15
    量级的项之间灾难性对消，1% 级失真纯属测试负载病态而非库错误
    （pow2 缩放 FP 精确，两路逐位一致，不引入失真）。
37. **规模实证 + 用户侧教程**（"可嵌入底座"主张的数据支撑与采用通道）：
    (a) `examples/scale_stress.rs`——两个动力学家族的定量曲线：
    **稀疏链**（O(n)/步，真实 `CheckpointManager` 分段反向）4000 维 ×
    10⁴ 步：前向 0.31 s + 反向 0.93 s、峰值内存 37.9 MiB、分配 51 万次
    （相对 200 维仅 +46%——每步固定成本主导）；**稠密全耦合**（O(n²)/步
    全 tape）400 维 × 10³ 步 18 ms / 15.4 MiB。内存随 T 亚线性（快照数
    恒定 × 状态尺寸）即 §4.4 的承诺形态。实现期验证器再次立功：稠密
    手写 VJP 连抓 2 处（对角元漏 k/n、gx 漏 dt 因子）——"手写 VJP 必须先
    过验证器"从建议升级为流程；
    (b) `docs/guide.md`——用户侧集成教程：两条路径纪律 → CustomOp 三条
    契约 → 一行验证 → Recomputable/checkpoint → 验证+健康度 ritual →
    优化器与 rep 健康度字段解读 → 性能纪律 → **常见错误速查表**（全部
    实际踩过的坑，8 条）。
38. **f32 物理算子泛型化**（第 31 条"CustomOp 为 f64 特化"限制的解除，
    展示算子泛型路径）：`DoublePendulumStep` 与 `GyroscopicStep` 改为
    `impl<S: Scalar> CustomOp<S>`——结构常量字段保持 f64、按数值类型
    就地 cast（舍入发生在求值点），数学字面量用 `S::one()` 组合，
    超越函数经 `num_traits::Float`。**验证策略的分工**：泛型代码路径由
    f64 oracle（FD 隔离器 + 守恒先验 + 对拍直通）覆盖——同一份代码，
    f64 验证即代码验证；f32 专属风险只剩"求值点舍入"，由 f32/f64
    同点对拍（f32 精确表示的点，1e-4 相对）单独检查。f32 能量守恒
    实测漂移 3.84e-3，与 f64 的 0.38% 几乎一致——长轨迹漂移由半隐式
    欧拉的能量振荡主导，f32 舍入在此步长/步数下未显现（结论限定于
    该工况，高刚度/长轨迹仍需重标定）。剩余：接触/空间代数算子的
    泛型化（同模式机械推广）与 `op_check` 验证器的 f32 版本。
39. **box-DDP（control-limited backward pass）**（第 30a 条 clamped 启发式
    的正规化，Tassa et al. 2014）：`solve_kk_boxed`——无界时 LU 精确路径
    （数值与历史逐位一致）；有界时每步 box-QP 用**投影坐标下降**解
    （增量维护 `Q_u_f = Q_u + Q_uu·k`，沿维无约束极小
    `k_i −= Q_u_f,i/Q_uu,ii` 后投影到 `[lo−u, hi−u]`；收敛后自由行
    `K = −Q_uu_ff⁻¹Q_ux_f`、钳制行零，μ 正则保留在自由子块）。前向
    pass 的钳制保留（ΔV 预估仍可能乐观，α 回溯兜底）。性质验证
    （双积分器，凸 QP 直觉可仲裁）：松界逐位复现无界最优（差 1e-16）；
    损失随界宽单调不降（100%/60%/30% 界 → 0.131/0.149/0.336）；
    60% 界下 3/20 步精确饱和到界。**实现期教训**：首版标量更新误写为
    绝对赋值（`k = −Q_u_f/Q_uu` 而非增量步），单维即振荡 0→−k→0——
    **性质测试（松界等价）一次定位**，佐证"算法正确性用性质测试守护、
    不靠单点数值"的原则。

### 12.4 里程碑完成情况

| 里程碑 | 状态 | 验收证据 |
|--------|------|---------|
| M1 核心骨架 | ✅ | 算子表 vs 双数/复步（1e-10 / 1e-9）；proptest 512 组随机 DAG；异常检测定位 `sqrt`；TLS 重入/类型不匹配/无 context panic 测试 |
| M2 自定义算子 + 物理适配 | ✅ | 多输出扇出累加、多种子 VJP；PendulumStep 手工 backward vs AD 展开（1e-10）；IFT vs 闭式解 AD（1e-10）+ 非线性 Newton 收敛 |
| M3 检查点 | ✅ | 分段 vs 全 tape 4 种调度一致（1e-10）；重跑 bit-exact；快照预算受控（Uniform=28、Online=5）；初始状态伴随可读 |
| M4 梯度健康度 | ✅ | 有限差分抓错（10% 偏差）；随机方向（n=50, 8 方向）；轨迹 vanishing/exploding/oscillating/nonfinite 判定；∇Fuzz 式 kink 检测 |
| M5 性能固化 | ✅ | criterion 基准体系（`cargo bench -p ad`）：标量表达式 fresh ~0.24 µs / 复用+clear_tape ~0.14 µs（稳态零分配）；1000 步单摆分段反向 ~0.11 ms（≈1.4× 全 tape，满足 §5.4 ≤2.5×）。全局分配器计数的无泄漏长稳测试（复用 Context 50k 轮 / 每轮丢弃 Context 20k 轮，存活字节回到基线）。CI 工作流（fmt/clippy/test/wasm/演示构建/bench 冒烟）已入库。火焰图完成（§12.3 第 26 条）+ Custom 记录瘦身（第 28c 条）。§5.4 目标表逐项验收：唯一未达标项"CustomOp vs 手写梯度 ≤2×"实测 ≈2.4–3.0×，偏差分析见第 28f 条 |

### 12.5 wasm 与 web 演示（v0.1 附加交付）

整个库栈（ad-core → ad）无需修改即通过 `wasm32-unknown-unknown` 编译（线程局部挂载、
`Rc<dyn CustomOp>`、`PhantomData<*mut>` 的 !Send 标记在单线程 wasm 下均成立）。
`crates/ad-demo`（Yew 0.21 + trunk）提供交互式演示：标量求导、1000 步单摆 checkpoint
分段反向（SVG 轨迹）、IFT 隐式求解三个面板，均带**实时有限差分验证徽章**，计算在
浏览器主线程同步完成（合计 < 5 ms）。`cd crates/ad-demo && trunk serve` 启动。
浏览器实测：渲染、滑块交互→wasm 重算→实时更新、三面板 FD 徽章全绿。

**文档状态**：v0.4.0——M1–M5 全部完成；功能层（第 29–32 条）+ 体系闭环层
（第 33–36 条）+ 规模实证与用户教程（第 37 条，`docs/guide.md`）+
f32 物理算子泛型化（第 38 条）+ box-DDP control-limited backward pass（第 39 条）；48 套件全绿
