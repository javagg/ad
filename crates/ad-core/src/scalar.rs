use std::fmt::Debug;

/// 底层数值类型抽象。第一版支持 `f64` / `f32`。
///
/// `f32` 可用，但梯度验证容差需单独标定（设计文档 §5.1）。
pub trait Scalar: Copy + Debug + PartialOrd + num_traits::Float + 'static {}

impl Scalar for f64 {}
impl Scalar for f32 {}
