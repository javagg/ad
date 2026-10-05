use crate::guard::with_context;
use crate::node::NodeId;
use crate::scalar::Scalar;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// AD 标量：承载数值 + 计算图节点标识。
///
/// - [`AD::constant`]：常量，不入带、不可微
/// - 叶子变量由 [`Context::var`](crate::Context::var) 创建
///
/// 算术运算符（`+` `-` `*` `/`）通过**线程局部 context** 入带（设计文档 §2.4），
/// 因此要求当前线程已 `Context::enter`；未进入时 panic。
/// 显式传 ctx 的路径见各 `Context` 方法与 `ad-ops` 的 `*_with` 函数。
#[derive(Clone, Copy, Debug)]
pub struct AD<S: Scalar> {
    /// 前向数值
    pub value: S,
    pub(crate) node: Option<NodeId>,
    /// `NodeId` 只对创建它的线程的 tape 有意义；零开销阻止跨线程误用
    _not_send: PhantomData<*mut ()>,
}

impl<S: Scalar> AD<S> {
    /// 创建常量（不可微、不参与入带）。
    pub fn constant(value: S) -> Self {
        AD {
            value,
            node: None,
            _not_send: PhantomData,
        }
    }

    pub(crate) fn tracked(value: S, node: NodeId) -> Self {
        AD {
            value,
            node: Some(node),
            _not_send: PhantomData,
        }
    }

    /// 计算图节点；`None` 表示常量。
    pub fn node(&self) -> Option<NodeId> {
        self.node
    }

    /// 是否被追踪（非常量）。
    pub fn is_tracked(&self) -> bool {
        self.node.is_some()
    }

    /// 值拷贝并切断梯度（常量化）。之后参与的计算不会再对其求导。
    pub fn detach(&self) -> Self {
        AD {
            value: self.value,
            node: None,
            _not_send: PhantomData,
        }
    }
}

impl<S: Scalar> From<S> for AD<S> {
    fn from(value: S) -> Self {
        AD::constant(value)
    }
}

// ---- 运算符重载：线程局部路径 ----

impl<S: Scalar> Add for AD<S> {
    type Output = AD<S>;
    fn add(self, rhs: AD<S>) -> AD<S> {
        with_context(|ctx| ctx.add(self, rhs))
    }
}

impl<S: Scalar> Sub for AD<S> {
    type Output = AD<S>;
    fn sub(self, rhs: AD<S>) -> AD<S> {
        with_context(|ctx| ctx.sub(self, rhs))
    }
}

impl<S: Scalar> Mul for AD<S> {
    type Output = AD<S>;
    fn mul(self, rhs: AD<S>) -> AD<S> {
        with_context(|ctx| ctx.mul(self, rhs))
    }
}

impl<S: Scalar> Div for AD<S> {
    type Output = AD<S>;
    fn div(self, rhs: AD<S>) -> AD<S> {
        with_context(|ctx| ctx.div(self, rhs))
    }
}

impl<S: Scalar> Neg for AD<S> {
    type Output = AD<S>;
    fn neg(self) -> AD<S> {
        with_context(|ctx| ctx.neg(self))
    }
}

impl<S: Scalar> Add<S> for AD<S> {
    type Output = AD<S>;
    fn add(self, rhs: S) -> AD<S> {
        with_context(|ctx| ctx.add(self, AD::constant(rhs)))
    }
}

impl<S: Scalar> Sub<S> for AD<S> {
    type Output = AD<S>;
    fn sub(self, rhs: S) -> AD<S> {
        with_context(|ctx| ctx.sub(self, AD::constant(rhs)))
    }
}

impl<S: Scalar> Mul<S> for AD<S> {
    type Output = AD<S>;
    fn mul(self, rhs: S) -> AD<S> {
        with_context(|ctx| ctx.mul(self, AD::constant(rhs)))
    }
}

impl<S: Scalar> Div<S> for AD<S> {
    type Output = AD<S>;
    fn div(self, rhs: S) -> AD<S> {
        with_context(|ctx| ctx.div(self, AD::constant(rhs)))
    }
}

// 标量在左侧：orphan rule 不允许 `impl Mul<AD<S>> for S`，按具体类型展开。
macro_rules! impl_scalar_lhs {
    ($t:ty) => {
        impl Add<AD<$t>> for $t {
            type Output = AD<$t>;
            fn add(self, rhs: AD<$t>) -> AD<$t> {
                with_context(|ctx| ctx.add(AD::constant(self), rhs))
            }
        }
        impl Sub<AD<$t>> for $t {
            type Output = AD<$t>;
            fn sub(self, rhs: AD<$t>) -> AD<$t> {
                with_context(|ctx| ctx.sub(AD::constant(self), rhs))
            }
        }
        impl Mul<AD<$t>> for $t {
            type Output = AD<$t>;
            fn mul(self, rhs: AD<$t>) -> AD<$t> {
                with_context(|ctx| ctx.mul(AD::constant(self), rhs))
            }
        }
        impl Div<AD<$t>> for $t {
            type Output = AD<$t>;
            fn div(self, rhs: AD<$t>) -> AD<$t> {
                with_context(|ctx| ctx.div(AD::constant(self), rhs))
            }
        }
    };
}

impl_scalar_lhs!(f64);
impl_scalar_lhs!(f32);
