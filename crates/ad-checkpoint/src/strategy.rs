//! 快照调度策略。
//!
//! 快照语义：`(step, 状态)`，状态为"执行第 step 步**之前**"的完整物理状态；
//! step 0 的初始状态由管理器在构造时保存，不占用预算。

use crate::manager::Recomputable;

#[derive(Clone)]
pub enum CheckpointStrategy {
    /// 每 `interval` 步存一次（步号 % interval == 0）。
    /// 分段反向架构下的内存最优位置（峰值段长 ≈ n/m）。
    Uniform { interval: usize },
    /// 固定快照预算 m：保留最近 m 个快照的在线调度，
    /// 适用于轨迹长度未知的流式 rollout（MPC 滚动优化）。
    Online { budget: usize },
    /// 用户自定义调度：`f(step) -> 是否在 step 前存快照`
    Custom(std::sync::Arc<dyn Fn(usize) -> bool + Send + Sync>),
}

impl std::fmt::Debug for CheckpointStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointStrategy::Uniform { interval } => write!(f, "Uniform({interval})"),
            CheckpointStrategy::Online { budget } => write!(f, "Online({budget})"),
            CheckpointStrategy::Custom(_) => write!(f, "Custom"),
        }
    }
}

impl CheckpointStrategy {
    pub(crate) fn should_snapshot<R: Recomputable>(
        &self,
        step: usize,
        snapshots: &mut Vec<(usize, R::State)>,
    ) -> bool {
        match self {
            CheckpointStrategy::Uniform { interval } => {
                *interval > 0 && step > 0 && step.is_multiple_of(*interval)
            }
            CheckpointStrategy::Online { budget } => {
                if *budget == 0 {
                    return false;
                }
                if snapshots.len() >= *budget {
                    // 保留最近的 m 个：淘汰最旧
                    snapshots.remove(0);
                }
                true
            }
            CheckpointStrategy::Custom(f) => step > 0 && f(step),
        }
    }
}
