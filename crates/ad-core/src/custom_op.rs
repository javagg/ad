use crate::scalar::Scalar;
use smallvec::SmallVec;

/// 自定义算子：物理引擎实现此 trait 来接入 AD（`custom_vjp` 风格，设计文档 §4.3）。
///
/// - [`CustomOp::forward`]：前向计算，返回 (输出值, 残差)。残差是 backward 需要
///   但无法从输入输出恢复的中间量（如 ABA 的 U、D、Ia、pa）。
/// - [`CustomOp::backward`]：给定残差和各输出的伴随值，返回**每个输入槽位**的梯度
///   贡献（长度必须等于 `num_inputs`；常量输入的槽位梯度会被丢弃）。
///
/// forward/backward 只见纯数值切片，内部计算**不会再入带**——这正是
/// "计算图不爆炸"的机制本身。
pub trait CustomOp<S: Scalar> {
    /// 输入数量
    fn num_inputs(&self) -> usize;

    /// 输出数量
    fn num_outputs(&self) -> usize;

    /// 前向计算，返回 (输出值, 残差数据)
    fn forward(&self, inputs: &[S]) -> (SmallVec<[S; 4]>, SmallVec<[S; 8]>);

    /// 反向计算（VJP）：给定残差和各输出的伴随值，返回各输入槽位的梯度
    fn backward(&self, residual: &[S], grad_output: &[S]) -> SmallVec<[S; 4]>;

    /// 异常诊断用算子名
    fn name(&self) -> &'static str {
        "custom_op"
    }
}
