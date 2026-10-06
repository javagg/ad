//! M2 验收：IFT 隐式求解（设计文档 §4.3.3）——隐式欧拉弹簧-阻尼系统。
//!
//! 隐式欧拉（m = 1）：q₁ = q₀ + h·v₁；v₁ = v₀ - h(k·q₁ + c·v₁)
//! 残差：r = [q₁ - q₀ - h·v₁, v₁ - v₀ + h·k·q₁ + h·c·v₁]
//! 闭式解：v₁ = (v₀ - h·k·q₀) / (1 + h·c + h²·k)；q₁ = q₀ + h·v₁
//! 对照组：闭式解的 AD 表达式（同参数叶子）→ 梯度应一致。

use ad_core::{Context, AD};
use ad_custom::{ImplicitSolve, Residual};
use num_traits::Num;

#[derive(Clone, Copy)]
struct SpringDamper;

impl Residual for SpringDamper {
    fn nx(&self) -> usize {
        2
    }
    fn ntheta(&self) -> usize {
        5
    }
    fn residual<N: Num + Copy>(&self, x: &[N], theta: &[N], r: &mut [N]) {
        // theta = [q0, v0, h, k, c]，x = [q1, v1]
        let (q0, v0, h, k, c) = (theta[0], theta[1], theta[2], theta[3], theta[4]);
        let (q1, v1) = (x[0], x[1]);
        r[0] = q1 - q0 - h * v1;
        r[1] = v1 - v0 + h * k * q1 + h * c * v1;
    }
}

fn build_params(ctx: &mut Context<f64>, q0: f64, v0: f64, h: f64, k: f64, c: f64) -> Vec<AD<f64>> {
    let mut ads = Vec::new();
    for v in [q0, v0, h, k, c] {
        let (ad, _var) = ctx.var(v);
        ads.push(ad);
    }
    ads
}

#[test]
fn ift_matches_closed_form_ad() {
    let (q0, v0, h, k, c) = (1.0, 0.5, 0.1, 3.0, 0.4);

    // --- IFT 路径 ---
    let mut ctx = Context::<f64>::new();
    let theta = build_params(&mut ctx, q0, v0, h, k, c);
    let out = ctx.call_custom(ImplicitSolve::new(SpringDamper), &theta);
    assert_eq!(out.len(), 2);
    // 前向解正确性
    let v1 = (v0 - h * k * q0) / (1.0 + h * c + h * h * k);
    let q1 = q0 + h * v1;
    assert!(
        (out[0].value - q1).abs() < 1e-12,
        "q1 {} vs {}",
        out[0].value,
        q1
    );
    assert!(
        (out[1].value - v1).abs() < 1e-12,
        "v1 {} vs {}",
        out[1].value,
        v1
    );
}

#[test]
fn ift_gradients_match_closed_form_ad_param_wise() {
    let (q0, v0, h, k, c) = (1.0, 0.5, 0.1, 3.0, 0.4);

    let mut ctx = Context::<f64>::new();
    let mut vars = Vec::new();
    let mut theta = Vec::new();
    for v in [q0, v0, h, k, c] {
        let (ad, var) = ctx.var(v);
        theta.push(ad);
        vars.push(var);
    }
    let out = ctx.call_custom(ImplicitSolve::new(SpringDamper), &theta);
    let q1s = ctx.mul(out[0], out[0]);
    let two_v1 = ctx.mul(AD::constant(2.0), out[1]);
    let loss = ctx.add(q1s, two_v1);
    ctx.backward(loss);

    let mut ctx2 = Context::<f64>::new();
    let mut vars2 = Vec::new();
    let mut theta2 = Vec::new();
    for v in [q0, v0, h, k, c] {
        let (ad, var) = ctx2.var(v);
        theta2.push(ad);
        vars2.push(var);
    }
    let [q0a, v0a, ha, ka, ca] = [theta2[0], theta2[1], theta2[2], theta2[3], theta2[4]];
    let hk = ctx2.mul(ha, ka);
    let hkq0 = ctx2.mul(hk, q0a);
    let num = ctx2.sub(v0a, hkq0);
    let hc = ctx2.mul(ha, ca);
    let h2 = ctx2.mul(ha, ha);
    let h2k = ctx2.mul(h2, ka);
    let one_plus_hc = ctx2.add(AD::constant(1.0), hc);
    let den = ctx2.add(one_plus_hc, h2k);
    let v1a = ctx2.div(num, den);
    let hv1 = ctx2.mul(ha, v1a);
    let q1a = ctx2.add(q0a, hv1);
    let q1s2 = ctx2.mul(q1a, q1a);
    let two_v1b = ctx2.mul(AD::constant(2.0), v1a);
    let loss2 = ctx2.add(q1s2, two_v1b);
    ctx2.backward(loss2);

    for (i, name) in ["q0", "v0", "h", "k", "c"].iter().enumerate() {
        let g1 = ctx.grad(vars[i]).unwrap();
        let g2 = ctx2.grad(vars2[i]).unwrap();
        assert!(
            (g1 - g2).abs() <= 1e-10 * (1.0 + g1.abs()),
            "dL/d{} mismatch: ift {} vs closed-form {}",
            name,
            g1,
            g2
        );
    }
}

#[test]
fn ift_solver_value_matches_direct_computation() {
    // 非线性残差的 Newton 收敛：r(x) = x³ + x - θ（单调，J = 3x²+1 处处非奇异，
    // 从 x=0 起步收敛；x³-θ 在 0 处奇异会卡死）
    struct Cubic;
    impl Residual for Cubic {
        fn nx(&self) -> usize {
            1
        }
        fn ntheta(&self) -> usize {
            1
        }
        fn residual<N: Num + Copy>(&self, x: &[N], theta: &[N], r: &mut [N]) {
            r[0] = x[0] * x[0] * x[0] + x[0] - theta[0];
        }
    }
    let theta_val = 7.5f64;
    let mut ctx = Context::<f64>::new();
    let (theta, vt) = ctx.var(theta_val);
    let out = ctx.call_custom(
        ImplicitSolve::with_cfg(Cubic, ad_custom::ImplicitSolveCfg { max_iters: 32, ..Default::default() }),
        &[theta],
    );
    // 解 x*：x³ + x = θ
    let expect = {
        // 简单二分求参考解
        let (mut lo, mut hi) = (0.0f64, theta_val);
        for _ in 0..200 {
            let m = 0.5 * (lo + hi);
            if m * m * m + m < theta_val {
                lo = m;
            } else {
                hi = m;
            }
        }
        0.5 * (lo + hi)
    };
    assert!(
        (out[0].value - expect).abs() < 1e-10,
        "{} vs {}",
        out[0].value,
        expect
    );

    // d(x*)/dθ = 1 / (3x² + 1)；loss = x²
    let loss = ctx.mul(out[0], out[0]);
    ctx.backward(loss);
    let expect_grad = 2.0 * expect / (3.0 * expect * expect + 1.0);
    assert!((ctx.grad(vt).unwrap() - expect_grad).abs() < 1e-10);
}
