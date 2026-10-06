//! f32 路径覆盖与容差标定（设计文档 §5.1 / §12.3 第 31 条）。
//!
//! 方法：整个算子表对 `S: Scalar` 泛型——同一泛型函数在 f64（已被双数
//! oracle 验证至 1e-10）与 f32 下各跑一遍，逐算子对比。f32 的误差来源：
//! 前向中间量的舍入（~1.2e-7 相对）扰动 Jacobian 求值点，标定容差
//! **1e-4 相对**（远松于 f64 的 1e-10，远紧于"能跑就行"）。
//! 另含 TLS 运算符重载路径与 bulk 算子的 f32 抽查。

use ad_core::{Context, AD};
use ad_ops::*;

type UnaryFn<S> = fn(&mut Context<S>, AD<S>) -> AD<S>;
type BinaryFn<S> = fn(&mut Context<S>, AD<S>, AD<S>) -> AD<S>;

fn unary_grad_f32(op: UnaryFn<f32>, x0: f32) -> f32 {
    let mut ctx = Context::<f32>::new();
    let (x, vx) = ctx.var(x0);
    let y = op(&mut ctx, x);
    assert!(y.value.is_finite(), "non-finite forward at {x0}");
    ctx.backward(y);
    ctx.grad(vx).unwrap()
}

fn unary_grad_f64(op: UnaryFn<f64>, x0: f64) -> f64 {
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(x0);
    let y = op(&mut ctx, x);
    ctx.backward(y);
    ctx.grad(vx).unwrap()
}

fn binary_grads_f32(op: BinaryFn<f32>, a: f32, b: f32) -> (f32, f32) {
    let mut ctx = Context::<f32>::new();
    let (x, vx) = ctx.var(a);
    let (y, vy) = ctx.var(b);
    let z = op(&mut ctx, x, y);
    assert!(z.value.is_finite(), "non-finite forward at ({a},{b})");
    ctx.backward(z);
    (ctx.grad(vx).unwrap(), ctx.grad(vy).unwrap())
}

fn binary_grads_f64(op: BinaryFn<f64>, a: f64, b: f64) -> (f64, f64) {
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(a);
    let (y, vy) = ctx.var(b);
    let z = op(&mut ctx, x, y);
    ctx.backward(z);
    (ctx.grad(vx).unwrap(), ctx.grad(vy).unwrap())
}

const TOL: f64 = 1e-4;

fn assert_close(name: &str, at: &str, g32: f64, g64: f64) {
    assert!(
        (g32 - g64).abs() <= TOL * (1.0 + g64.abs()),
        "{name} {at}: f32 {g32:.7} vs f64 参考 {g64:.7}"
    );
}

#[test]
fn f32_unary_table_matches_f64_reference() {
    let cases: &[(&str, UnaryFn<f32>, UnaryFn<f64>, &[f32])] = &[
        ("exp", exp_with, exp_with, &[0.5, 1.0, -1.5]),
        ("ln", ln_with, ln_with, &[0.5, 2.0, 100.0]),
        ("sqrt", sqrt_with, sqrt_with, &[0.25, 2.0, 9.0]),
        ("sin", sin_with, sin_with, &[0.5, 1.0, -2.0]),
        ("cos", cos_with, cos_with, &[0.5, 1.0, -2.0]),
        ("tanh", tanh_with, tanh_with, &[0.5, 2.0, -1.0]),
        ("asin", asin_with, asin_with, &[0.3, 0.8]),
        ("acos", acos_with, acos_with, &[0.3, 0.8]),
        ("recip", recip_with, recip_with, &[0.5, 4.0, -2.0]),
        ("abs", abs_with, abs_with, &[1.5, -2.5]),
        ("relu", relu_with, relu_with, &[1.5, -0.5]),
        ("sigmoid", sigmoid_with, sigmoid_with, &[0.0, 1.0, -3.0]),
    ];
    for &(name, op32, op64, pts) in cases {
        for &p in pts {
            let g32 = unary_grad_f32(op32, p) as f64;
            let g64 = unary_grad_f64(op64, p as f64);
            assert_close(name, &format!("x={p}"), g32, g64);
        }
    }
}

#[test]
fn f32_binary_table_matches_f64_reference() {
    let cases: &[(&str, BinaryFn<f32>, BinaryFn<f64>, &[(f32, f32)])] = &[
        ("powf", powf_with, powf_with, &[(2.0, 3.0), (4.0, 0.5)]),
        ("atan2", atan2_with, atan2_with, &[(1.0, 2.0), (-0.5, 1.5)]),
        ("min", min_with, min_with, &[(1.0, 2.0), (3.0, -1.0)]),
        ("max", max_with, max_with, &[(1.0, 2.0), (3.0, -1.0)]),
    ];
    for &(name, op32, op64, pts) in cases {
        for &(a, b) in pts {
            let (ga32, gb32) = binary_grads_f32(op32, a, b);
            let (ga64, gb64) = binary_grads_f64(op64, a as f64, b as f64);
            let at = format!("({a},{b})");
            assert_close(name, &at, ga32 as f64, ga64);
            assert_close(name, &at, gb32 as f64, gb64);
        }
    }
}

#[test]
fn f32_bulk_ops_match_f64_reference() {
    // dot / norm2 / matvec（4 维，值全部 f32 精确表示）
    let a: Vec<f32> = [0.5, -1.25, 2.0, 0.125].to_vec();
    let b: Vec<f32> = [1.5, 0.25, -0.75, 4.0].to_vec();

    let grad_dot = |av: &[f32]| -> Vec<f32> {
        let mut ctx = Context::<f32>::new();
        let (ad_a, va): (Vec<AD<f32>>, Vec<ad_core::Variable>) =
            av.iter().map(|&v| ctx.var(v)).unzip();
        let (ad_b, _): (Vec<AD<f32>>, Vec<ad_core::Variable>) =
            b.iter().map(|&v| ctx.var(v)).unzip();
        let y = dot_with(&mut ctx, &ad_a, &ad_b);
        ctx.backward(y);
        va.iter().map(|&v| ctx.grad(v).unwrap()).collect()
    };
    let g32 = grad_dot(&a);
    let g64: Vec<f64> = {
        let mut ctx = Context::<f64>::new();
        let (ad_a, va): (Vec<AD<f64>>, Vec<ad_core::Variable>) =
            a.iter().map(|&v| ctx.var(v as f64)).unzip();
        let (ad_b, _): (Vec<AD<f64>>, Vec<ad_core::Variable>) =
            b.iter().map(|&v| ctx.var(v as f64)).unzip();
        let y = dot_with(&mut ctx, &ad_a, &ad_b);
        ctx.backward(y);
        va.iter().map(|&v| ctx.grad(v).unwrap()).collect()
    };
    for (i, (&g32, &g64)) in g32.iter().zip(g64.iter()).enumerate() {
        assert_close("dot", &format!("a[{i}]"), g32 as f64, g64);
    }

    // norm2
    let mut ctx = Context::<f32>::new();
    let (ad_x, vx): (Vec<AD<f32>>, Vec<ad_core::Variable>) =
        a.iter().map(|&v| ctx.var(v)).unzip();
    let n = norm2_with(&mut ctx, &ad_x);
    ctx.backward(n);
    let norm64 = a.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
    for (i, &v) in a.iter().enumerate() {
        let g = ctx.grad(vx[i]).unwrap() as f64;
        let want = v as f64 / norm64;
        assert_close("norm2", &format!("x[{i}]"), g, want);
    }
}

#[test]
fn f32_tls_overload_path_and_free_fall_analytic() {
    // TLS 运算符重载（f32）+ 自由落体解析梯度（§5.3 场景 1 的 f32 形态）
    let guard = Context::<f32>::new().enter();
    let (g, vg) = guard.var(9.81f32);
    let (t, vt) = guard.var(0.5f32);
    // h = g·t²/2；∂h/∂g = t²/2，∂h/∂t = g·t
    let h = g * t * t * 0.5f32;
    guard.backward(h);
    let gg = guard.grad(vg).unwrap() as f64;
    let gt = guard.grad(vt).unwrap() as f64;
    assert!((gg - 0.125).abs() < 1e-6, "∂h/∂g = {gg}");
    assert!((gt - 9.81 * 0.5).abs() < 1e-4, "∂h/∂t = {gt}");
}
