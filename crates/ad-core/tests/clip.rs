//! 梯度裁剪工具（设计文档 §4.2.3）：clip_grad_norm / clip_grad_value。

use ad_core::{Context, AD};

/// loss = 2xy（x=3, y=4 → grads (8, 6)，范数 10）
fn build_2xy(ctx: &mut Context<f64>) -> (AD<f64>, ad_core::Variable, ad_core::Variable) {
    let (x, vx) = ctx.var(3.0);
    let (y, vy) = ctx.var(4.0);
    let xy1 = ctx.mul(x, y);
    let xy2 = ctx.mul(y, x);
    let loss = ctx.add(xy1, xy2);
    (loss, vx, vy)
}

#[test]
fn clip_grad_norm_scales_to_max_preserving_direction() {
    let mut ctx = Context::<f64>::new();
    let (loss, vx, vy) = build_2xy(&mut ctx);
    ctx.backward(loss);

    let norm = ctx.clip_grad_norm(&[vx, vy], 1.0);
    assert_eq!(norm, 10.0);
    let gx = ctx.grad(vx).unwrap();
    let gy = ctx.grad(vy).unwrap();
    assert!((gx - 0.8).abs() < 1e-12);
    assert!((gy - 0.6).abs() < 1e-12);
    // 方向保持、范数等于 max
    let n2 = (gx * gx + gy * gy).sqrt();
    assert!((n2 - 1.0).abs() < 1e-12);
}

#[test]
fn clip_grad_norm_noop_below_max() {
    let mut ctx = Context::<f64>::new();
    let (loss, vx, vy) = build_2xy(&mut ctx);
    ctx.backward(loss);
    let norm = ctx.clip_grad_norm(&[vx, vy], 100.0);
    assert_eq!(norm, 10.0);
    assert_eq!(ctx.grad(vx), Some(8.0));
    assert_eq!(ctx.grad(vy), Some(6.0));
}

#[test]
fn clip_grad_value_clamps_elements() {
    let mut ctx = Context::<f64>::new();
    let (loss, vx, vy) = build_2xy(&mut ctx);
    ctx.backward(loss);
    ctx.clip_grad_value(&[vx, vy], 7.0);
    assert_eq!(ctx.grad(vx), Some(7.0));
    assert_eq!(ctx.grad(vy), Some(6.0));
}

#[test]
fn clip_grad_norm_ignores_nonfinite() {
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(0.0);
    let y = ctx.div(AD::constant(1.0), x); // 1/0 → inf 前向
    ctx.backward(y); // grad wrt x: -1/x² = -inf
    let norm = ctx.clip_grad_norm(&[vx], 1.0);
    assert!(norm.is_infinite());
    // 非有限范数 → 不缩放（梯度保持 inf，由异常检测/上游处理）
    assert!(ctx.grad(vx).unwrap().is_infinite());
}
