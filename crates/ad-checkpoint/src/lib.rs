//! `ad-checkpoint`：长轨迹的检查点机制（设计文档 §4.4）。
//!
//! 架构（分段反向）：
//! 1. 前向 rollout 在 `no_grad` 下运行（tape 不增长），每步后按策略存快照
//!    （完整确定性物理状态）；
//! 2. 反向时从最后一个快照起逐段：恢复状态 → 重新入带跑该段 → 以边界伴随
//!    + loss 为种子做该段反向 → 收集段起点伴随作为上一段的种子 → 清段 tape。
//!
//! 内存 = m 个快照 + 最长段的 tape；重算 = 每步恰好一次（1× 前向）。
//! 快照位置在此架构下只影响峰值段 tape 长度，均匀分布即最优；
//! 经典 Revolve（嵌套重算、最小化总前向次数）列为后续工作（设计文档 §4.4.2）。
//!
//! 正确性前提：`Recomputable::step` 必须**确定性**——求解器迭代次数固定、
//! RNG 状态入快照、无归约顺序抖动、warm-start 入快照（设计文档 §4.4.5）。

pub mod manager;
pub mod sim;
pub mod strategy;

pub use manager::{CheckpointManager, Recomputable};
pub use sim::PendulumSim;
pub use strategy::CheckpointStrategy;
