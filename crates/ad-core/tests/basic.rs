//! M1 验收：核心梯度语义、常量折叠、no_grad/detach、clear_tape 复用、异常检测。

use ad_core::{Context, AD};
use ad_ops::{sin_with, sqrt_with};

#[test]
fn sin_of_mul_add_composition() {
    // y = sin(a*b + c)，a=1, b=2, c=3
    let mut ctx = Context::<f64>::new();
    let (a, va) = ctx.var(1.0);
    let (b, vb) = ctx.var(2.0);
    let (c, vc) = ctx.var(3.0);
    let ab = ctx.mul(a, b);
    let s = ctx.add(ab, c);
    let y = sin_with(&mut ctx, s);
    ctx.backward(y);

    let expect = 5.0f64.cos();
    assert!((ctx.grad(va).unwrap() - expect * 2.0).abs() < 1e-12);
    assert!((ctx.grad(vb).unwrap() - expect * 1.0).abs() < 1e-12);
    assert!((ctx.grad(vc).unwrap() - expect).abs() < 1e-12);
    // loss 节点伴随 = seed
    assert_eq!(ctx.adjoint(y), Some(1.0));
}

#[test]
fn constant_folding_keeps_tape_small() {
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(2.0);

    // 纯常量子表达式不入带
    let k = ctx.mul(AD::constant(2.0), AD::constant(3.0));
    assert!(ctx.tape_len() == 0);
    assert!(!k.is_tracked());

    // 常量输入被折叠出记录：x * 6.0 只产生 1 条记录
    let y = ctx.mul(x, k);
    assert_eq!(ctx.tape_len(), 1);
    ctx.backward(y);
    assert_eq!(ctx.grad(vx), Some(6.0));
}

#[test]
fn gradient_accumulation_and_zero_grads() {
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(3.0);
    let y1 = ctx.mul(x, x); // d/dx = 2x = 6
    ctx.backward(y1);
    let y2 = ctx.mul(x, x);
    ctx.backward(y2);
    assert_eq!(ctx.grad(vx), Some(12.0)); // 累加语义
    ctx.zero_grads();
    assert_eq!(ctx.grad(vx), Some(0.0));
    let y3 = ctx.mul(x, x);
    ctx.backward(y3);
    assert_eq!(ctx.grad(vx), Some(6.0));
}

#[test]
fn backward_from_with_custom_seed() {
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(2.0);
    let y = ctx.mul(x, x);
    ctx.backward_from(y, 2.0);
    assert_eq!(ctx.grad(vx), Some(8.0)); // 2 * 2x
}

#[test]
fn unconnected_leaf_grads_zero() {
    let mut ctx = Context::<f64>::new();
    let (_x, vx) = ctx.var(1.0);
    let (y, vy) = ctx.var(5.0);
    let z = ctx.mul(y, y);
    ctx.backward(z);
    assert_eq!(ctx.grad(vx), Some(0.0));
    assert_eq!(ctx.grad(vy), Some(10.0));
}

#[test]
fn backward_on_constant_panics() {
    let mut ctx = Context::<f64>::new();
    let c = AD::constant(1.0);
    let y = ctx.mul(c, c);
    // 常量运算不入带，结果也是常量
    assert!(!y.is_tracked());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ctx.backward(y);
    }));
    assert!(result.is_err());
}

#[test]
fn no_grad_suppresses_recording() {
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(2.0);
    assert_eq!(ctx.tape_len(), 0);
    let y = ctx.no_grad(|ctx| ctx.mul(x, x));
    assert_eq!(ctx.tape_len(), 0);
    assert!(!y.is_tracked());

    // no_grad 之外的表达式梯度只沿未阻断路径流动
    let z = ctx.mul(y, x);
    ctx.backward(z);
    assert_eq!(ctx.grad(vx), Some(4.0)); // z = (x*x 常量化) * x → d/dx = x = 2? 见下
    let _ = vx;
}

#[test]
fn detach_cuts_gradient_path() {
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(3.0);
    let d = x.detach();
    assert!(!d.is_tracked());
    let y = ctx.mul(d, x); // y = 3 * x
    ctx.backward(y);
    assert_eq!(ctx.grad(vx), Some(3.0));
}

#[test]
fn clear_tape_reuse_keeps_leaf_grads() {
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(2.0);
    for i in 0..1000 {
        let y = ctx.mul(x, x);
        ctx.backward(y);
        assert_eq!(ctx.grad(vx), Some(4.0), "iteration {}", i);
        ctx.clear_tape();
        ctx.zero_grads();
        assert_eq!(ctx.tape_len(), 0);
        // 中间 AD 已失效，但叶子句柄依旧有效
    }
    let y = ctx.mul(x, x);
    ctx.backward(y);
    assert_eq!(ctx.grad(vx), Some(4.0));
}

#[test]
fn anomaly_detection_locates_offending_op() {
    let mut ctx = Context::<f64>::new();
    ctx.set_detect_anomaly(true);
    let (x, _vx) = ctx.var(0.0);
    let y = sqrt_with(&mut ctx, x); // d/dx sqrt(0) = +inf
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ctx.backward(y);
    }));
    let msg = result
        .err()
        .and_then(|e| e.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    assert!(msg.contains("gradient anomaly"), "message: {}", msg);
    assert!(msg.contains("sqrt"), "message: {}", msg);
}

#[test]
fn without_anomaly_detection_grads_flow_nonfinite() {
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(0.0);
    let y = sqrt_with(&mut ctx, x);
    ctx.backward(y);
    assert_eq!(ctx.grad(vx), Some(f64::INFINITY));
}

#[test]
fn multiple_seeds_accumulate() {
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(2.0);
    let (y, vy) = ctx.var(3.0);
    let u = ctx.mul(x, y);
    let v = ctx.mul(x, x);
    // seed u with 1 and v with 10 simultaneously
    ctx.backward_seeds(&[(u, 1.0), (v, 10.0)]);
    assert_eq!(ctx.grad(vx), Some(3.0 + 40.0));
    assert_eq!(ctx.grad(vy), Some(2.0));
}
