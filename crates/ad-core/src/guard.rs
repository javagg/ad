use crate::ad::AD;
use crate::context::Context;
use crate::custom_op::CustomOp;
use crate::node::Variable;
use crate::scalar::Scalar;
use std::any::{type_name, Any};
use std::cell::RefCell;
use std::marker::PhantomData;

thread_local! {
    static STACK: RefCell<Vec<Box<dyn Any>>> = const { RefCell::new(Vec::new()) };
}

/// 在当前线程的活动 [`Context`] 上执行 `f`。
///
/// 由运算符重载与 [`ContextGuard`] 的便捷方法使用；不支持重入
/// （在 `with_context` 闭包内再用运算符重载会 panic——闭包内请用显式的
/// `&mut Context` 方法）。
pub fn with_context<S: Scalar, R>(f: impl FnOnce(&mut Context<S>) -> R) -> R {
    STACK.with(|stack| {
        let mut stack = stack.try_borrow_mut().unwrap_or_else(|_| {
            panic!(
                "reentrant access to the thread-local AD context: overloaded operators or \
                 ContextGuard methods cannot be used inside another with_context closure; \
                 use the explicit &mut Context methods instead"
            )
        });
        let ctx = stack.last_mut().expect(
            "no active AD context on this thread: create one with Context::new().enter(), \
             or pass &mut Context explicitly",
        );
        let ctx = ctx.downcast_mut::<Context<S>>().unwrap_or_else(|| {
            panic!(
                "the active AD context holds a different scalar type (expected Context<{}>)",
                type_name::<S>()
            )
        });
        f(ctx)
    })
}

/// `ContextGuard` 安装入口（见 [`Context::enter`]）。
pub(crate) fn install<S: Scalar>(ctx: Context<S>) -> ContextGuard<S> {
    STACK.with(|stack| stack.borrow_mut().push(Box::new(ctx)));
    ContextGuard {
        _not_send: PhantomData,
    }
}

/// 线程局部 context 的作用域 guard：drop 时弹出（支持嵌套）。
///
/// 便捷方法（`var` / `backward` / `grad` / ...）内部走 [`with_context`]；
/// 复杂逻辑可用 [`ContextGuard::with`] 拿到 `&mut Context`——闭包内不要再
/// 使用运算符重载（会重入 panic），用显式方法。
pub struct ContextGuard<S: Scalar> {
    _not_send: PhantomData<*mut S>,
}

impl<S: Scalar> ContextGuard<S> {
    pub fn with<R>(&self, f: impl FnOnce(&mut Context<S>) -> R) -> R {
        with_context(f)
    }

    /// 弹出并取回 context。弹出的是**栈顶** context（嵌套场景请按相反顺序退出）。
    pub fn exit(self) -> Context<S> {
        let ctx = STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            let top = stack
                .pop()
                .expect("ContextGuard::exit: context stack is empty");
            *top.downcast::<Context<S>>()
                .expect("ContextGuard::exit: scalar type mismatch")
        });
        std::mem::forget(self);
        ctx
    }

    pub fn var(&self, value: S) -> (AD<S>, Variable) {
        with_context(|c: &mut Context<S>| c.var(value))
    }

    pub fn grad(&self, var: Variable) -> Option<S> {
        with_context(|c: &mut Context<S>| c.grad(var))
    }

    pub fn backward(&self, loss: AD<S>) {
        with_context(|c: &mut Context<S>| c.backward(loss))
    }

    pub fn backward_from(&self, y: AD<S>, seed: S) {
        with_context(|c: &mut Context<S>| c.backward_from(y, seed))
    }

    pub fn zero_grads(&self) {
        with_context(|c: &mut Context<S>| c.zero_grads())
    }

    pub fn clear_tape(&self) {
        with_context(|c: &mut Context<S>| c.clear_tape())
    }

    pub fn tape_len(&self) -> usize {
        with_context(|c: &mut Context<S>| c.tape_len())
    }

    pub fn set_detect_anomaly(&self, on: bool) {
        with_context(|c: &mut Context<S>| c.set_detect_anomaly(on))
    }

    pub fn call_custom<O: CustomOp<S> + 'static>(&self, op: O, inputs: &[AD<S>]) -> Vec<AD<S>> {
        with_context(|c: &mut Context<S>| c.call_custom(op, inputs))
    }
}

impl<S: Scalar> Drop for ContextGuard<S> {
    fn drop(&mut self) {
        STACK.with(|stack| {
            stack.borrow_mut().pop();
        });
    }
}

/// 线程局部路径的 no_grad 区域：区域内前向计算不入带（设计文档 §4.1.5）。
pub fn no_grad<S: Scalar, R>(f: impl FnOnce() -> R) -> R {
    with_context::<S, _>(|ctx| ctx.begin_no_grad());
    let _guard = NoGradGuard::<S>(PhantomData);
    f()
}

struct NoGradGuard<S: Scalar>(PhantomData<*mut S>);

impl<S: Scalar> Drop for NoGradGuard<S> {
    fn drop(&mut self) {
        STACK.with(|stack| {
            if let Ok(mut stack) = stack.try_borrow_mut() {
                if let Some(top) = stack.last_mut() {
                    if let Some(ctx) = top.downcast_mut::<Context<S>>() {
                        ctx.end_no_grad();
                    }
                }
            }
        });
    }
}
