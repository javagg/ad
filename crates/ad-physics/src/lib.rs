//! `ad-physics`：空间代数（Featherstone 风格）参考 CustomOp 库。
//!
//! 每个算子手写 VJP，并由测试中的"单步逐坐标 FD 隔离器"逐分量验证；
//! 跨算子约定一致性由**动能坐标系不变性**物理先验测试守护
//! （同一系统在两个坐标系中动能相等，且 ∂T/∂(E, r) = 0）。
//!
//! 约定见 [`spatial`] 模块文档：运动向量 `[ω, v]`、力向量 `[n, f]`、
//! 惯性三元组 `([Ī], m, c)`、旋转行主序 3×3。

pub mod chain;
pub mod contact;
pub mod gyro;
pub mod ops;
pub mod spatial;

pub use chain::DoublePendulumStep;
pub use contact::{ContactNormalOp, RegularizedFrictionOp};
pub use gyro::GyroscopicStep;
pub use ops::{
    InertiaApply, PluckerForce, PluckerMotion, RotateInertia, So3Exp, SpatialCrossMotion,
};
