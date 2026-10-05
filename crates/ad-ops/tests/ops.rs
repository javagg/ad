//! M1 验收：基础算子局部 Jacobian（设计文档 §5.1 Primitive 层）。
//!
//! 双重 oracle：双数前向（全域）+ 复步微分（解析子集，机器精度）。

use ad_core::dual::Dual;
use ad_core::{Context, AD};
use ad_ops::*;
use num_complex::Complex;

fn check_unary(
    name: &str,
    ad_fn: fn(&mut Context<f64>, AD<f64>) -> AD<f64>,
    dual_fn: fn(Dual) -> Dual,
    points: &[f64],
) {
    for &x0 in points {
        let mut ctx = Context::<f64>::new();
        let (x, vx) = ctx.var(x0);
        let y = ad_fn(&mut ctx, x);
        if !y.value.is_finite() {
            continue; // 定义域边界，边界语义单独测
        }
        ctx.backward(y);
        let g = ctx.grad(vx).unwrap();
        let d = dual_fn(Dual::seed(x0));
        assert!(
            (g - d.du).abs() <= 1e-10 * (1.0 + g.abs()),
            "{} at x={}: ad {} vs dual {}",
            name,
            x0,
            g,
            d.du
        );
    }
}

fn check_binary(
    name: &str,
    ad_fn: fn(&mut Context<f64>, AD<f64>, AD<f64>) -> AD<f64>,
    dual_fn: fn(Dual, Dual) -> Dual,
    points: &[(f64, f64)],
) {
    for &(a, b) in points {
        let mut ctx = Context::<f64>::new();
        let (x, vx) = ctx.var(a);
        let (y, vy) = ctx.var(b);
        let z = ad_fn(&mut ctx, x, y);
        if !z.value.is_finite() {
            continue;
        }
        ctx.backward(z);
        let gx = ctx.grad(vx).unwrap();
        let gy = ctx.grad(vy).unwrap();
        let dx = dual_fn(Dual::seed(a), Dual::constant(b));
        let dy = dual_fn(Dual::constant(a), Dual::seed(b));
        assert!(
            (gx - dx.du).abs() <= 1e-10 * (1.0 + gx.abs()),
            "{} d/da at ({},{}): ad {} vs dual {}",
            name,
            a,
            b,
            gx,
            dx.du
        );
        assert!(
            (gy - dy.du).abs() <= 1e-10 * (1.0 + gy.abs()),
            "{} d/db at ({},{}): ad {} vs dual {}",
            name,
            a,
            b,
            gy,
            dy.du
        );
    }
}

#[test]
fn unary_table_matches_dual_oracle() {
    check_unary("exp", exp_with, |d| d.exp(), &[-2.0, 0.0, 1.5]);
    check_unary("ln", ln_with, |d| d.ln(), &[0.1, 0.5, 3.0, 100.0]);
    check_unary("sqrt", sqrt_with, |d| d.sqrt(), &[0.25, 2.0, 1e4]);
    check_unary("sin", sin_with, |d| d.sin(), &[-3.1, 0.0, 1.0, 6.3]);
    check_unary("cos", cos_with, |d| d.cos(), &[-3.1, 0.0, 1.0, 6.3]);
    check_unary("tanh", tanh_with, |d| d.tanh(), &[-2.0, 0.0, 2.0]);
    check_unary("asin", asin_with, |d| d.asin(), &[-0.9, 0.0, 0.9]);
    check_unary("acos", acos_with, |d| d.acos(), &[-0.9, 0.0, 0.9]);
    check_unary("recip", recip_with, |d| d.recip(), &[-4.0, 0.5, 10.0]);
    check_unary("abs", abs_with, |d| d.abs(), &[-3.0, 2.0]);
    check_unary(
        "relu",
        relu_with,
        |d| {
            if d.re > 0.0 {
                d
            } else {
                Dual::constant(0.0)
            }
        },
        &[-1.0, 0.5],
    );
    check_unary("sigmoid", sigmoid_with, |d| d.sigmoid(), &[-5.0, 0.0, 5.0]);
    check_unary(
        "powi",
        |c, x| powi_with(c, x, 3),
        |d| d.powi(3),
        &[-2.0, 0.0, 1.5],
    );
    check_unary("powi0", |c, x| powi_with(c, x, 0), |d| d.powi(0), &[1.5]);
    check_unary(
        "clamp",
        |c, x| clamp_with(c, x, -0.5, 0.5),
        |d| {
            if d.re < -0.5 || d.re > 0.5 {
                Dual::constant(d.re.clamp(-0.5, 0.5))
            } else {
                d
            }
        },
        &[-1.0, 0.0, 1.0],
    );
}

#[test]
fn binary_table_matches_dual_oracle() {
    check_binary(
        "add",
        |c, a, b| c.add(a, b),
        |a, b| a + b,
        &[(1.0, 2.0), (-1.0, 0.5)],
    );
    check_binary("sub", |c, a, b| c.sub(a, b), |a, b| a - b, &[(1.0, 2.0)]);
    check_binary("mul", |c, a, b| c.mul(a, b), |a, b| a * b, &[(1.5, -2.0)]);
    check_binary("div", |c, a, b| c.div(a, b), |a, b| a / b, &[(3.0, 4.0)]);
    check_binary(
        "powf",
        powf_with,
        |a, b| a.powf(b),
        &[(2.0, 0.5), (0.5, 3.0)],
    );
    check_binary(
        "atan2",
        atan2_with,
        |a, b| a.atan2(b),
        &[(1.0, 2.0), (-1.0, -2.0)],
    );
    check_binary(
        "min",
        min_with,
        |a, b| {
            if a.re <= b.re {
                a
            } else {
                b
            }
        },
        &[(1.0, 2.0), (3.0, 2.0)],
    );
    check_binary(
        "max",
        max_with,
        |a, b| {
            if a.re >= b.re {
                a
            } else {
                b
            }
        },
        &[(1.0, 2.0), (3.0, 2.0)],
    );
}

#[test]
fn lerp_three_way() {
    // lerp(a, b, t) = a + t(b - a)，三个输入都可微
    let mut ctx = Context::<f64>::new();
    let (a, va) = ctx.var(1.0);
    let (b, vb) = ctx.var(5.0);
    let (t, vt) = ctx.var(0.25);
    let y = lerp_with(&mut ctx, a, b, t);
    assert!((y.value - 2.0).abs() < 1e-15);
    ctx.backward(y);
    assert_eq!(ctx.grad(va), Some(0.75));
    assert_eq!(ctx.grad(vb), Some(0.25));
    assert_eq!(ctx.grad(vt), Some(4.0));
}

#[test]
fn lerp_matches_dual() {
    // 三个输入分别带种子做前向对拍
    let check = |seed: usize| -> f64 {
        let mut seeded = [0.0f64; 3];
        seeded[seed] = 1.0;
        let a = Dual::new(1.0, seeded[0]);
        let b = Dual::new(5.0, seeded[1]);
        let t = Dual::new(0.25, seeded[2]);
        (a + t * (b - a)).du
    };

    let mut ctx = Context::<f64>::new();
    let (a, va) = ctx.var(1.0);
    let (b, vb) = ctx.var(5.0);
    let (t, vt) = ctx.var(0.25);
    let y = lerp_with(&mut ctx, a, b, t);
    ctx.backward(y);
    assert!((ctx.grad(va).unwrap() - check(0)).abs() < 1e-15); // 1 - t = 0.75
    assert!((ctx.grad(vb).unwrap() - check(1)).abs() < 1e-15); // t = 0.25
    assert!((ctx.grad(vt).unwrap() - check(2)).abs() < 1e-15); // b - a = 4
}

#[test]
fn non_smooth_zero_point_conventions() {
    // PAP 约定：|0|、relu(0)、clamp 边界 → 导数 0（设计文档 §4.2.3）
    for (name, f) in [
        ("abs", abs_with as fn(&mut Context<f64>, AD<f64>) -> AD<f64>),
        ("relu", |c: &mut Context<f64>, x| relu_with(c, x)),
    ] {
        let mut ctx = Context::<f64>::new();
        let (x, vx) = ctx.var(0.0);
        let y = f(&mut ctx, x);
        ctx.backward(y);
        assert_eq!(ctx.grad(vx), Some(0.0), "{} at 0", name);
    }
    // clamp 边界约定与 PyTorch 一致：区间 [lo, hi] 内（含边界）梯度 1，界外 0
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(-0.5);
    let y = clamp_with(&mut ctx, x, -0.5, 0.5); // 恰在下边界 → 梯度 1
    ctx.backward(y);
    assert_eq!(ctx.grad(vx), Some(1.0));
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(-0.7);
    let y = clamp_with(&mut ctx, x, -0.5, 0.5); // 界外 → 梯度 0
    ctx.backward(y);
    assert_eq!(ctx.grad(vx), Some(0.0));

    // sqrt(0) 反向 +inf（文档化，不静默饱和）
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(0.0);
    let y = sqrt_with(&mut ctx, x);
    ctx.backward(y);
    assert_eq!(ctx.grad(vx), Some(f64::INFINITY));
}

#[test]
fn min_max_tie_goes_to_first_argument() {
    let mut ctx = Context::<f64>::new();
    let (a, va) = ctx.var(1.0);
    let (b, vb) = ctx.var(1.0);
    let y = max_with(&mut ctx, a, b);
    ctx.backward(y);
    assert_eq!(ctx.grad(va), Some(1.0));
    assert_eq!(ctx.grad(vb), Some(0.0));
}

// ---- 复步微分交叉验证（设计文档 §4.5.4） ----
// f'(x) ≈ Im(f(x + i·h)) / h，h = 1e-20，机器精度、无步长两难。
// 仅适用于解析函数（abs/relu/min/max 等非解析函数不适用）。

const H: f64 = 1e-20;

fn complex_step_check(
    name: &str,
    ad_fn: fn(&mut Context<f64>, AD<f64>) -> AD<f64>,
    c_fn: fn(Complex<f64>) -> Complex<f64>,
    x0: f64,
) {
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(x0);
    let y = ad_fn(&mut ctx, x);
    ctx.backward(y);
    let g = ctx.grad(vx).unwrap();

    let z = c_fn(Complex::new(x0, H));
    let d = z.im / H;
    assert!(
        (g - d).abs() <= 1e-9 * (1.0 + d.abs()),
        "{} at x={}: ad {} vs complex-step {}",
        name,
        x0,
        g,
        d
    );
}

#[test]
fn complex_step_cross_validation() {
    complex_step_check("exp", exp_with, |z| z.exp(), 1.3);
    complex_step_check("ln", ln_with, |z| z.ln(), 2.5);
    complex_step_check("sqrt", sqrt_with, |z| z.sqrt(), 3.7);
    complex_step_check("sin", sin_with, |z| z.sin(), 0.9);
    complex_step_check("cos", cos_with, |z| z.cos(), 0.9);
    complex_step_check("tanh", tanh_with, |z| z.tanh(), 0.6);
    complex_step_check(
        "sigmoid",
        sigmoid_with,
        |z| Complex::new(1.0, 0.0) / (Complex::new(1.0, 0.0) + (-z).exp()),
        0.4,
    );
    complex_step_check("recip", recip_with, |z| Complex::new(1.0, 0.0) / z, 1.7);
}

#[test]
fn complex_step_powf() {
    // x^p 对 x 的导数（p 为实常数）：x^p = exp(p·ln x)
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(2.0);
    let y = powf_with(&mut ctx, x, AD::constant(3.5));
    ctx.backward(y);
    let g = ctx.grad(vx).unwrap();

    let p = 3.5;
    let z = (Complex::new(2.0, H)).powc(Complex::new(p, 0.0));
    assert!((g - z.im / H).abs() <= 1e-9 * (1.0 + (g).abs()));
}
