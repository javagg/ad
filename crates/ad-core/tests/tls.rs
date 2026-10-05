//! 线程局部 context 路径（设计文档 §2.4）：运算符重载、嵌套、重入防护、类型不匹配。

use ad_core::{no_grad, Context, AD};
use ad_ops::{sin, sqrt};

#[test]
fn operator_overloads_via_tls() {
    let guard = Context::<f64>::new().enter();
    let (a, va) = guard.var(1.0);
    let (b, vb) = guard.var(2.0);
    let (c, vc) = guard.var(3.0);

    let y = sin(a * b + c);
    guard.backward(y);

    let expect = 5.0f64.cos();
    assert!((guard.grad(va).unwrap() - expect * 2.0).abs() < 1e-12);
    assert!((guard.grad(vb).unwrap() - expect).abs() < 1e-12);
    assert!((guard.grad(vc).unwrap() - expect).abs() < 1e-12);
}

#[test]
fn scalar_mixed_arithmetic() {
    let guard = Context::<f64>::new().enter();
    let (x, vx) = guard.var(3.0);
    let y = 2.0 * x + x * 0.5 - 1.0 + x / 6.0; // = 6 + 1.5 - 1 + 0.5 = 7
    guard.backward(y);
    assert!((y.value - 7.0).abs() < 1e-12);
    assert!((guard.grad(vx).unwrap() - (2.0 + 0.5 + 1.0 / 6.0)).abs() < 1e-12);
}

#[test]
fn no_grad_tls_region() {
    let guard = Context::<f64>::new().enter();
    let (x, _vx) = guard.var(2.0);
    let len_before = guard.tape_len();
    let y = no_grad::<f64, _>(|| {
        let z = x * x;
        assert!(!z.is_tracked());
        z
    });
    assert!(!y.is_tracked());
    assert_eq!(guard.tape_len(), len_before);
}

#[test]
fn nested_contexts_isolate() {
    let guard1 = Context::<f64>::new().enter();
    let (a, va) = guard1.var(1.0);

    // 嵌套第二个 context：运算符落到栈顶
    {
        let guard2 = Context::<f32>::new().enter();
        let (b, vb) = guard2.var(2.0f32);
        let y2 = b * b;
        guard2.backward(y2);
        assert_eq!(guard2.grad(vb), Some(4.0f32));
    } // guard2 drop → 弹出栈顶

    // 外层 context 不受影响
    let y1 = a * 5.0;
    guard1.backward(y1);
    assert_eq!(guard1.grad(va), Some(5.0));
}

#[test]
fn guard_exit_returns_context() {
    let guard = Context::<f64>::new().enter();
    let (x, vx) = guard.var(2.0);
    let mut ctx = guard.exit();
    let y = ctx.mul(x, x);
    ctx.backward(y);
    assert_eq!(ctx.grad(vx), Some(4.0));
}

#[test]
#[should_panic(expected = "reentrant")]
fn reentrant_with_context_panics() {
    let guard = Context::<f64>::new().enter();
    let (a, _va) = guard.var(1.0);
    guard.with(|_ctx| {
        let _ = a * a; // 内部再进 with_context → 重入 panic
    });
}

#[test]
#[should_panic(expected = "no active AD context")]
fn operators_without_context_panic() {
    let a = AD::<f64>::constant(1.0);
    let _ = a * a;
}

#[test]
#[should_panic(expected = "different scalar type")]
fn scalar_type_mismatch_panics() {
    let _guard = Context::<f64>::new().enter();
    let a = AD::<f32>::constant(1.0);
    let _ = a * a; // with_context::<f32> 遇到 Context<f64> → panic
}

#[test]
fn sqrt_zero_tls_semantics_consistent() {
    let guard = Context::<f64>::new().enter();
    let (x, vx) = guard.var(0.0);
    let y = sqrt(x);
    guard.backward(y);
    assert_eq!(guard.grad(vx), Some(f64::INFINITY));
}
