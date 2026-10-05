//! `ad-custom`：自定义算子适配与 IFT（隐函数定理）隐式求解模式（设计文档 §4.3.3）。
//!
//! `CustomOp` trait 本体在 `ad-core`（tape 数据模型的一部分）；
//! 本 crate 提供：
//! - [`linear_solve`]：小型稠密线性求解（Gaussian 消元，IFT 反向用）；
//! - [`ift`]：把 `solve(θ) → x*`（满足 r(x\*; θ) = 0）整体封装为一个自定义算子，
//!   反向只解一次线性伴随系统，内存与迭代次数无关、无截断偏差。

pub mod ift;
pub mod linear_solve;

pub use ift::{ImplicitSolve, ImplicitSolveCfg, Residual};
