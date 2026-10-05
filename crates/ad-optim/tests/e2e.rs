//! 端到端轨迹优化基准（路线图方向 4）：检验 AD 梯度是否**可用于优化**。
//!
//! 基准 1：受控单摆（150 步 rollout，150 维控制序列）——控制序列把摆从
//! θ₀ = 0.5 推到目标角 1.2，带控制量正则。检验长 rollout + 表达式组合
//! 梯度（非 CustomOp）的优化可用性。
//!
//! 基准 2：接触弹跳球（ContactNormalOp，200 步）——优化初速度与阻尼参数，
//! 使末段高度贴近目标。检验穿透接触梯度（高刚度、近非光滑）的优化可用性。
//!
//! 两组基准同时对照：优化轨迹 vs 有限差分梯度在同一点的偏差（应通过
//! GradientChecker），并断言损失下降倍数与末态精度。

use ad::prelude::*;
use ad::{Context, Variable};

// ============================================================ 基准 1：受控单摆

const T: usize = 150;
const DT: f64 = 0.05;
const THETA0: f64 = 0.5;
const OMEGA0: f64 = 0.0;
const G: f64 = 9.81;
const LEN: f64 = 4.0; // g/L = 2.45，控制幅值 1.5 可推动
const TARGET: f64 = 1.2;
const U_MAX: f64 = 1.5;
const EFFORT: f64 = 1e-3;

/// 受控单摆 rollout + 损失/梯度（控制序列为自由参数）。
fn pendulum_loss_and_grad(u: &[f64], g: &mut [f64]) -> f64 {
    let mut ctx = Context::<f64>::new();
    let mut u_vars: Vec<Variable> = Vec::with_capacity(u.len());
    let mut u_ad: Vec<AD<f64>> = Vec::with_capacity(u.len());
    for &v in u {
        let (ad, var) = ctx.var(v);
        u_vars.push(var);
        u_ad.push(ad);
    }
    // 初始状态叶子
    let (mut th, _vth) = ctx.var(THETA0);
    let (mut om, _vom) = ctx.var(OMEGA0);
    let g_ad = AD::constant(G);
    let len_ad = AD::constant(LEN);
    let dt = AD::constant(DT);

    for t in 0..T {
        // 控制限幅（clamp 算子，非光滑边界按 PyTorch 约定）
        let torque = clamp_with(&mut ctx, u_ad[t], -U_MAX, U_MAX);
        // ω' = ω + dt·(u − (g/L)·sin θ)
        let sn = sin_with(&mut ctx, th);
        let grav = ctx.mul(g_ad, sn);
        let grav_over_l = ctx.div(grav, len_ad);
        let net = ctx.sub(torque, grav_over_l);
        let dw = ctx.mul(dt, net);
        om = ctx.add(om, dw);
        // θ' = θ + dt·ω'
        let dth = ctx.mul(dt, om);
        th = ctx.add(th, dth);
    }
    // loss = (θ_T − target)² + 0.1·ω_T² + EFFORT·Σu²
    let dth_t = ctx.sub(th, AD::constant(TARGET));
    let term1 = ctx.mul(dth_t, dth_t);
    let om_sq = ctx.mul(om, om);
    let term2 = ctx.mul(AD::constant(0.1), om_sq);
    let mut loss = ctx.add(term1, term2);
    for &uv in &u_ad {
        let sq = ctx.mul(uv, uv);
        let effort = ctx.mul(AD::constant(EFFORT), sq);
        loss = ctx.add(loss, effort);
    }
    ctx.backward(loss);
    for (i, &v) in u_vars.iter().enumerate() {
        g[i] = ctx.grad(v).unwrap();
    }
    loss.value
}

#[test]
fn benchmark_pendulum_trajectory_optimization() {
    let u0 = vec![0.0; T];
    let (loss0, g0) = {
        let mut g = vec![0.0; T];
        (pendulum_loss_and_grad(&u0, &mut g), g)
    };

    // 梯度 vs 有限差分抽查（起点处）
    let mut grad_fd = vec![0.0; T];
    let h = 1e-6;
    for j in [0usize, T / 2, T - 1] {
        let mut up = u0.clone();
        up[j] += h;
        let mut um = u0.clone();
        um[j] -= h;
        grad_fd[j] = (pendulum_loss_and_grad(&up, &mut vec![0.0; T])
            - pendulum_loss_and_grad(&um, &mut vec![0.0; T]))
            / (2.0 * h);
        assert!(
            (g0[j] - grad_fd[j]).abs() <= 1e-5 * (1.0 + g0[j].abs()),
            "grad[{j}]: ad {} vs fd {}",
            g0[j],
            grad_fd[j]
        );
    }

    let cfg = ad_optim::OptimizerCfg {
        max_iters: 400,
        lr0: 0.3,
        tol_grad: 1e-4,
        ..Default::default()
    };
    let (u_opt, rep) = ad_optim::minimize_gradient_descent(&u0, &cfg, pendulum_loss_and_grad);

    let loss_f = rep.loss;
    let iters = rep.iters;
    let gn = rep.grad_norm;
    let conv = rep.converged;
    eprintln!(
        "摆杆优化：loss {loss0:.4} → {loss_f:.6}（{iters} 迭代，‖g‖∞ = {gn:.2e}，收敛 = {conv}）"
    );
    assert!(
        rep.loss < loss0 * 0.05,
        "loss should drop >20x: {loss0} -> {}（effort 正则使损失下限非零）",
        rep.loss
    );

    // 末态验证：用优化后的控制重放，θ_T 贴近目标
    let mut ctx = Context::<f64>::new();
    let u_ad: Vec<AD<f64>> = u_opt.iter().map(|&v| AD::constant(v)).collect();
    let (mut th, _) = ctx.var(THETA0);
    let (mut om, _) = ctx.var(OMEGA0);
    let g_ad = AD::constant(G);
    let len_ad = AD::constant(LEN);
    let dt = AD::constant(DT);
    for t in 0..T {
        let torque = clamp_with(&mut ctx, u_ad[t], -U_MAX, U_MAX);
        let sn = sin_with(&mut ctx, th);
        let grav = ctx.mul(g_ad, sn);
        let gol = ctx.div(grav, len_ad);
        let net = ctx.sub(torque, gol);
        let dw = ctx.mul(dt, net);
        om = ctx.add(om, dw);
        let dth = ctx.mul(dt, om);
        th = ctx.add(th, dth);
    }
    eprintln!("θ_T = {:.4}（目标 {TARGET}）", th.value);
    assert!(
        (th.value - TARGET).abs() < 0.15,
        "θ_T {} too far from {TARGET}",
        th.value
    );
}

// ============================================================ 基准 2：接触弹跳球

const T_B: usize = 200;
const DT_B: f64 = 0.002;
const KK: f64 = 200.0;
const PP: f64 = 1.5;

const EPS_C: f64 = 1e-4;

/// 弹跳球：优化 (v0, d) 使 z_T 贴近目标高度 h*。
fn ball_loss_and_grad(p: &[f64], g: &mut [f64]) -> f64 {
    let (v0, d) = (p[0], p[1]);
    let mut ctx = Context::<f64>::new();
    let (z_ad, _vz) = ctx.var(0.4); // z0 固定
    let (v_ad, vv) = ctx.var(v0);
    let (d_ad, vd) = ctx.var(d);
    let k_ad = AD::constant(KK);
    let dt = AD::constant(DT_B);
    let (mut z, mut v) = (z_ad, v_ad);
    for _ in 0..T_B {
        let f = ctx.call_custom(
            ad::ContactNormalOp,
            &[z, v, k_ad, AD::constant(PP), d_ad, AD::constant(EPS_C)],
        );
        let dv = ctx.mul(dt, f[0]); // m = 1，力沿 +z
        v = ctx.add(v, dv);
        let dz = ctx.mul(dt, v);
        z = ctx.add(z, dz);
    }
    // loss = (z_T − h*)²
    let d = ctx.sub(z, AD::constant(0.12));
    let loss = ctx.mul(d, d);
    ctx.backward(loss);
    g[0] = ctx.grad(vv).unwrap();
    g[1] = ctx.grad(vd).unwrap();
    loss.value
}

/// 纯数值 rollout（FD/Taylor oracle）
fn ball_loss(p: &[f64]) -> f64 {
    let (v0, d) = (p[0], p[1]);
    let mut z = 0.4f64;
    let mut v = v0;
    for _ in 0..T_B {
        let (o, _) = ad::ContactNormalOp.forward(&[z, v, KK, PP, d, EPS_C]);
        v += DT_B * o[0];
        z += DT_B * v;
    }
    let dd = z - 0.12;
    dd * dd
}

#[test]
fn benchmark_contact_ball_optimization() {
    let p0 = [1.5f64, 0.02];
    let mut g0 = [0.0; 2];
    let loss0 = ball_loss_and_grad(&p0, &mut g0);

    // 梯度 vs FD
    let check = GradientChecker::default().check_scalar(ball_loss, &p0, &g0);
    assert!(check.passed, "FD check failed: {:?}", check.max_rel_error);

    // Taylor 余项（前向值与梯度自洽——接触近非光滑的关键检验）
    let taylor = GradientChecker::default().taylor_test(ball_loss, &p0, &g0, None);
    assert!(taylor.passed, "taylor order {}", taylor.estimated_order);

    let cfg = ad_optim::OptimizerCfg {
        max_iters: 200,
        lr0: 0.5,
        tol_grad: 1e-5,
        ..Default::default()
    };
    let (p_opt, rep) = ad_optim::minimize_gradient_descent(&p0, &cfg, ball_loss_and_grad);
    let loss_f = rep.loss;
    let iters = rep.iters;
    let v_star = p_opt[0];
    let d_star = p_opt[1];
    eprintln!(
        "接触弹跳球优化：loss {loss0:.6} → {loss_f:.8}（{iters} 迭代，v* = {v_star:+.4}，d* = {d_star:+.4}）"
    );
    assert!(rep.loss < loss0 * 1e-2, "loss {loss0} -> {}", rep.loss);
    let z_t = ball_loss(&p_opt).sqrt() + 0.12;
    eprintln!("z_T = {z_t:.4}（目标 0.12）");
    assert!((z_t - 0.12).abs() < 0.02, "z_T {z_t} too far from 0.12");
}
