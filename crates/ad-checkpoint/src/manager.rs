//! 可重算仿真状态机接口（设计文档 §4.4.3）。
//!
//! 由物理引擎实现。关联类型 `State` 强迫实现者把"完整状态"写成一个具体
//! 结构体——RNG、接触状态、warm-start 这类隐藏状态更容易被想起来
//! （替代 v0.2 设计中无法表达重算语义的 `Box<dyn Any>`）。

use ad_core::{Context, AD};

/// 完整、确定性的可重算仿真状态机。
pub trait Recomputable {
    /// 完整动态状态（q、qd、接触状态、warm-start、RNG 状态等）。
    /// 漏掉任何一项，重算就会发散（设计文档 §4.4.5）。
    type State: Clone;

    /// 捕获当前完整状态（前向值，与 AD 追踪无关）。
    fn save_state(&self) -> Self::State;

    /// 恢复状态。恢复后必须调用 [`Recomputable::bind_state`] 重建 AD 视图。
    fn load_state(&mut self, state: &Self::State);

    /// 把当前状态的各分量创建为新的叶子变量并采纳为当前 AD 状态，
    /// 返回按固定顺序排列的状态变量（分段边界的对齐锚点，
    /// 设计文档 §4.4.4：按位置而非 NodeId 对齐）。
    fn bind_state(&mut self, ctx: &mut Context<f64>) -> Vec<AD<f64>>;

    /// 当前 AD 状态（按 bind_state 的固定顺序）。
    /// 段结束时这些是"段输出状态"，其伴随作为下一段的边界种子。
    fn state(&self) -> &[AD<f64>];

    /// 从当前状态确定性前进 1 步（重新入带）。
    /// 可微参数（质量、控制等叶子）应在本 trait 之外注册到 ctx 并由
    /// 实现者持有其 `AD` 句柄。
    fn step(&mut self, ctx: &mut Context<f64>);
}

/// 检查点管理器：调度快照 + 分段反向（含边界伴随传递）。
///
/// 前向循环每步调用 [`CheckpointManager::forward_step`]（内部以 no_grad
/// 前进并按策略存快照）；随后调用 [`CheckpointManager::backward`] 完成
/// 分段反向，叶子梯度与全 tape 反向一致。
pub struct CheckpointManager<R: Recomputable> {
    strategy: crate::strategy::CheckpointStrategy,
    /// (step, 状态)，状态为"执行第 step 步之前"；按 step 升序
    snapshots: Vec<(usize, R::State)>,
    /// 初始状态（step 0 之前），不占快照预算
    initial: R::State,
    /// 已前进的步数（= 下一个未执行步号）
    steps_seen: usize,
}

impl<R: Recomputable> CheckpointManager<R> {
    pub fn new(strategy: crate::strategy::CheckpointStrategy, sim: &R) -> Self {
        CheckpointManager {
            strategy,
            snapshots: Vec::new(),
            initial: sim.save_state(),
            steps_seen: 0,
        }
    }

    /// 前向推进 1 步：以 no_grad 前进（tape 不增长），随后按策略存快照。
    /// `t` 必须为当前步号（从 0 起连续递增）。
    pub fn forward_step(&mut self, ctx: &mut Context<f64>, sim: &mut R, t: usize) {
        debug_assert_eq!(
            t, self.steps_seen,
            "forward_step must be called for every step in order"
        );
        ctx.no_grad(|ctx| sim.step(ctx));
        self.steps_seen = t + 1;
        if self
            .strategy
            .should_snapshot::<R>(t + 1, &mut self.snapshots)
        {
            self.snapshots.push((t + 1, sim.save_state()));
        }
    }

    /// 当前快照数（内存监控用）。
    pub fn num_snapshots(&self) -> usize {
        self.snapshots.len()
    }

    fn state_at(&self, step: usize) -> &R::State {
        if step == 0 {
            return &self.initial;
        }
        self.snapshots
            .iter()
            .find(|(s, _)| *s == step)
            .map(|(_, st)| st)
            .unwrap_or_else(|| panic!("no snapshot at step {}", step))
    }

    /// 分段反向（设计文档 §4.4.4）。
    ///
    /// `loss` 在最后一段重算结束后被调用，应从 `sim.state()` 构建标量损失
    /// 并返回；`ctx.zero_grads()` 会在开始时自动执行（fresh 语义）。
    ///
    /// 返回初始状态（step 0 输入）的伴随 ∂L/∂x₀（如需对初值求导）。
    pub fn backward(
        &mut self,
        ctx: &mut Context<f64>,
        sim: &mut R,
        loss: &dyn Fn(&mut Context<f64>, &R) -> AD<f64>,
    ) -> Vec<f64> {
        assert!(self.steps_seen > 0, "no forward steps recorded");
        ctx.zero_grads();

        // 段边界：初始 0 + 快照步号；末段终点 = steps_seen
        let mut bounds: Vec<usize> = Vec::with_capacity(self.snapshots.len() + 2);
        bounds.push(0);
        for (s, _) in &self.snapshots {
            if *s < self.steps_seen && bounds.last().copied() != Some(*s) {
                bounds.push(*s);
            }
        }

        let mut boundary: Option<Vec<f64>> = None; // 段输出状态的伴随（来自后一段的输入收集）
        let mut first_segment_input_adj = Vec::new();

        // 逆序处理段：最后一段先反（带 loss 种子），逐段向前传递边界伴随
        for i in (0..bounds.len()).rev() {
            let start = bounds[i];
            let end = bounds.get(i + 1).copied().unwrap_or(self.steps_seen);
            if start > end {
                continue;
            }

            let st = self.state_at(start);
            sim.load_state(st);
            let input_vars = sim.bind_state(ctx);

            for _ in start..end {
                sim.step(ctx);
            }

            let mut seeds: Vec<(AD<f64>, f64)> = Vec::new();
            if let Some(adj) = &boundary {
                for (v, &a) in sim.state().iter().zip(adj.iter()) {
                    seeds.push((*v, a));
                }
            }
            if end == self.steps_seen {
                let l = loss(ctx, sim);
                seeds.push((l, 1.0));
            }
            ctx.backward_seeds(&seeds);

            let adj: Vec<f64> = input_vars
                .iter()
                .map(|&a| ctx.grad_of(a).unwrap_or(0.0))
                .collect();
            if start == 0 {
                first_segment_input_adj = adj;
            } else {
                boundary = Some(adj);
            }
            ctx.clear_tape();
        }

        first_segment_input_adj
    }
}
