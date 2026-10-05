//! `ad`：用户接口 facade，整合全部 `ad-*` 模块（设计文档 §3.2）。
//!
//! ```no_run
//! use ad::prelude::*;
//!
//! // 线程局部路径：运算符重载 + guard 便捷方法
//! let guard = Context::<f64>::new().enter();
//! let (x, vx) = guard.var(2.0);
//! let y = ad::sin(x * x) + x * 3.0;
//! guard.backward(y);
//! assert_eq!(guard.grad(vx), Some((2.0 * 2.0f64).cos() * 2.0 * 2.0 + 3.0));
//! ```

pub use ad_checkpoint::{CheckpointManager, CheckpointStrategy, PendulumSim, Recomputable};
pub use ad_custom::{linear_solve, ImplicitSolve, ImplicitSolveCfg, Residual};
pub use ad_ops::*;
pub use ad_verify::{
    GradientChecker, GradientHealth, NonSmoothnessReport, StabilityVerdict, TrajectoryStability,
};

pub use ad_core::{
    no_grad, with_context, Context, ContextGuard, CustomOp, NodeId, Scalar, Variable, AD,
};

pub mod prelude {
    pub use ad_checkpoint::{CheckpointManager, CheckpointStrategy, PendulumSim, Recomputable};
    pub use ad_core::{
        no_grad, with_context, Context, ContextGuard, CustomOp, NodeId, Scalar, Variable, AD,
    };
    pub use ad_custom::{ImplicitSolve, ImplicitSolveCfg, Residual};
    pub use ad_ops::*;
    pub use ad_verify::GradientChecker;
}
