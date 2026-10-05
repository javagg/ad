//! `ad-checkpoint`：长轨迹的检查点机制（设计文档 §4.4）。
//!
//! 两种反向架构：
//!
//! **平面分段**（Uniform / Online / Custom）：前向 no_grad rollout 并按策略存快照；
//! 反向时从最后一个快照起逐段：恢复状态 → 重新入带跑该段 → 以边界伴随 + loss
//! 为种子做该段反向 → 收集段起点伴随作为上一段的种子 → 清段 tape。
//! 内存 = m 个快照 + 最长段的 tape（≈ n/m）；重算 = 每步恰好一次（1× 前向）。
//!
//! **二分嵌套**（`Nested { budget }`）：前向不存快照；反向递归二分——前半段
//! no_grad 前进，先反尾段再恢复本窗起点状态反头段。峰值段 tape ≈ n/2^budget，
//! live 状态数 = budget+1，重算 ≈ (budget+1)/2 × 前向。适用于状态大而单步
//! tape 小的物理场景（快照比 tape 贵时的内存-重算旋钮）。
//!
//! 正确性前提：`Recomputable::step` 必须**确定性**——求解器迭代次数固定、
//! RNG 状态入快照、无归约顺序抖动、warm-start 入快照（设计文档 §4.4.5）；
//! 且 **load_state 之后必须 bind_state** 再 step（AD 视图刷新，嵌套路径的
//! 两个隐蔽 bug 均源于此，见设计文档 §12.3 第 17 条）。

pub mod manager;
pub mod sim;
pub mod strategy;

pub use manager::{CheckpointManager, Recomputable};
pub use sim::PendulumSim;
pub use strategy::CheckpointStrategy;
