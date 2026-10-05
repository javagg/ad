//! `ad-core`: 核心抽象层。
//!
//! 包含 tape-based 反向模式 AD 的全部核心机制：
//! - [`AD`]：承载数值 + 计算图节点的可微标量
//! - [`Context`]：tape、伴随数组、叶子梯度管理
//! - [`CustomOp`]：自定义算子（`custom_vjp` 风格，见设计文档 §4.3）
//! - 线程局部 context 挂载（[`Context::enter`] / [`with_context`]，见设计文档 §2.4）
//!
//! ```text
//! 计算图约定（SSA）：
//!   tape 上的记录按拓扑序排列，逆序遍历即拓扑逆序；
//!   每条记录的输出节点由本记录独占产生。
//! ```

pub mod custom_op;
#[cfg(feature = "test-oracle")]
pub mod dual;

mod ad;
mod context;
mod guard;
mod node;
mod scalar;
mod tape;

pub use ad::AD;
pub use context::Context;
pub use custom_op::CustomOp;
pub use guard::{no_grad, with_context, ContextGuard};
pub use node::{NodeId, Variable};
pub use scalar::Scalar;

/// 预lude：常用类型与线程局部操作的便捷导入。
pub mod prelude {
    pub use crate::custom_op::CustomOp;
    pub use crate::{no_grad, with_context, Context, ContextGuard, NodeId, Scalar, Variable, AD};
}
