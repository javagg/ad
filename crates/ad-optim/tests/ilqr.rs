//! iLQR 验收：
//! 1. 双积分器（线性-二次）——少量迭代精确收敛到目标；
//! 2. 受控单摆（与 GD 基准同一问题）——iLQR 迭代次数应远少于 GD 的 55 次；
//! 3. 动力学 Jacobian 逐列与中心差分对拍。

use ad::prelude::*;
use ad::{clamp_with, Context};
use ad_optim::{rollout, solve_ilqr, Dynamics, IlqrCfg, QuadraticCost};

// ============================================================ 1. 双积分器

/// x' = [x₁ + dt·x₂; x₂ + dt·u]，nx = 2，nu = 1
struct DoubleIntegrator {
    dt: f64,
}

impl Dynamics for DoubleIntegrator {
    fn nx(&self) -> usize {
        2
    }
    fn nu(&self) -> usize {
        1
    }
    fn step(&self, ctx: &mut Context<f64>, x: &[AD<f64>], u: &[AD<f64>]) -> Vec<AD<f64>> {
        let dt = AD::constant(self.dt);
        let wx = ctx.mul(dt, x[1]);
        let x1 = ctx.add(x[0], wx);
        let wu = ctx.mul(dt, u[0]);
        let x2 = ctx.add(x[1], wu);
        vec![x1, x2]
    }
}

#[test]
fn ilqr_double_integrator_exact() {
    let d = DoubleIntegrator { dt: 0.1 };
    let t_total = 40;
    let cost = QuadraticCost {
        q: vec![1.0, 1.0],
        r: vec![0.1],
        qf: vec![10.0, 10.0],
        goal: vec![1.0, 0.0],
    };
    let u0: Vec<Vec<f64>> = vec![vec![0.0]; t_total];
    let cfg = IlqrCfg {
        max_iters: 20,
        ..Default::default()
    };
    let (u, rep) = solve_ilqr(&d, &[0.0, 0.0], &u0, &cost, &cfg);

    // R=0.1 的控制代价使最优解有有限残差：收敛判据 = expected 改进归零
    assert!(rep.converged, "report: {rep:?}");
    assert!(
        rep.loss < rep.loss0 * 0.3,
        "loss {} → {}",
        rep.loss0,
        rep.loss
    );
    let (x, _) = rollout(&d, &[0.0, 0.0], &u, &cost);
    assert!(
        (x[t_total][0] - 1.0).abs() < 5e-2,
        "x_T[0] = {}",
        x[t_total][0]
    );
}

// ============================================================ 2. 受控单摆（对照 GD）

const T: usize = 150;
const DT: f64 = 0.05;
const G: f64 = 9.81;
const LEN: f64 = 4.0;
const TARGET: f64 = 1.2;
const U_MAX: f64 = 1.5;
const THETA0: f64 = 0.5;

/// 与 GD 基准同一动力学（θ, ω, u）
struct PendulumDyn;

impl Dynamics for PendulumDyn {
    fn nx(&self) -> usize {
        2
    }
    fn nu(&self) -> usize {
        1
    }
    fn step(&self, ctx: &mut Context<f64>, x: &[AD<f64>], u: &[AD<f64>]) -> Vec<AD<f64>> {
        let (th, om) = (x[0], x[1]);
        let torque = clamp_with(ctx, u[0], -U_MAX, U_MAX);
        let g_ad = AD::constant(G);
        let len_ad = AD::constant(LEN);
        let dt = AD::constant(DT);
        let sn = ad::sin_with(ctx, th);
        let grav = ctx.mul(g_ad, sn);
        let gol = ctx.div(grav, len_ad);
        let net = ctx.sub(torque, gol);
        let dw = ctx.mul(dt, net);
        let om1 = ctx.add(om, dw);
        let dth = ctx.mul(dt, om1);
        let th1 = ctx.add(th, dth);
        vec![th1, om1]
    }
}

#[test]
fn ilqr_pendulum_beats_gd_iterations() {
    let d = PendulumDyn;
    let t_total = T;
    let cost = QuadraticCost {
        q: vec![0.0, 0.0], // 运行状态不罚（中途摆动自由）
        r: vec![1e-3],
        qf: vec![10.0, 1.0],
        goal: vec![TARGET, 0.0],
    };
    let u0: Vec<Vec<f64>> = vec![vec![0.0]; t_total];
    let cfg = IlqrCfg {
        max_iters: 80,
        ..Default::default()
    };
    let (u, rep) = solve_ilqr(&d, &[THETA0, 0.0], &u0, &cost, &cfg);

    eprintln!(
        "iLQR 摆杆：loss {:.4} → {:.6}（{} 迭代，converged = {}）",
        rep.loss0, rep.loss, rep.iters, rep.converged
    );
    assert!(rep.loss < rep.loss0 * 0.05, "report: {rep:?}");

    // 终态：重放验证 θ_T
    let mut ctx = Context::<f64>::new();
    let mut th = AD::constant(THETA0);
    let mut om = AD::constant(0.0);
    for t in 0..T {
        let out = d.step(&mut ctx, &[th, om], &[AD::constant(u[t][0])]);
        th = out[0];
        om = out[1];
    }
    eprintln!("θ_T = {:.4}（目标 {TARGET}）", th.value);
    assert!((th.value - TARGET).abs() < 0.1, "θ_T = {}", th.value);

    // 对照：GD 用了 55 次迭代才把 loss 降 55×；iLQR 应显著更少
    assert!(
        rep.iters < 55,
        "iLQR should beat GD's 55 iterations, took {}",
        rep.iters
    );
}

// ============================================================ 3. Jacobian 数值对拍

#[test]
fn ilqr_jacobian_matches_finite_difference() {
    let d = PendulumDyn;
    let x = [0.4f64, -0.3];
    let u = [0.8f64];

    let (a, b) = jacobians_pub(&d, &x, &u);

    let h = 1e-7;
    for j in 0..2 {
        let mut xp = x;
        xp[j] += h;
        let mut xm = x;
        xm[j] -= h;
        let fp = d.step(
            &mut Context::new(),
            &[AD::constant(xp[0]), AD::constant(xp[1])],
            &[AD::constant(u[0])],
        );
        let fm = d.step(
            &mut Context::new(),
            &[AD::constant(xm[0]), AD::constant(xm[1])],
            &[AD::constant(u[0])],
        );
        for i in 0..2 {
            let fd = (fp[i].value - fm[i].value) / (2.0 * h);
            assert!(
                (a[i][j] - fd).abs() < 1e-5,
                "A[{i}][{j}] an {} vs fd {fd}",
                a[i][j]
            );
        }
    }
    // u 列
    let mut up = u;
    up[0] += h;
    let mut um = u;
    um[0] -= h;
    let fp = d.step(
        &mut Context::new(),
        &[AD::constant(x[0]), AD::constant(x[1])],
        &[AD::constant(up[0])],
    );
    let fm = d.step(
        &mut Context::new(),
        &[AD::constant(x[0]), AD::constant(x[1])],
        &[AD::constant(um[0])],
    );
    for i in 0..2 {
        let fd = (fp[i].value - fm[i].value) / (2.0 * h);
        assert!(
            (b[i][0] - fd).abs() < 1e-5,
            "B[{i}][0] an {} vs fd {fd}",
            b[i][0]
        );
    }
}

fn jacobians_pub(d: &dyn Dynamics, x: &[f64], u: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut ctx = Context::<f64>::new();
    let mut xv = Vec::new();
    let mut uv = Vec::new();
    let mut x_vars = Vec::new();
    let mut u_vars = Vec::new();
    for &v in x {
        let (ad, var) = ctx.var(v);
        xv.push(ad);
        x_vars.push(var);
    }
    for &v in u {
        let (ad, var) = ctx.var(v);
        uv.push(ad);
        u_vars.push(var);
    }
    let out = d.step(&mut ctx, &xv, &uv);
    let (nx, nu) = (d.nx(), d.nu());
    let mut a = vec![vec![0.0; nx]; nx];
    let mut b = vec![vec![0.0; nu]; nx];
    for i in 0..nx {
        ctx.backward_seeds(&[(out[i], 1.0)]);
        for j in 0..nx {
            a[i][j] = ctx.grad(x_vars[j]).unwrap_or(0.0);
        }
        for j in 0..nu {
            b[i][j] = ctx.grad(u_vars[j]).unwrap_or(0.0);
        }
        ctx.zero_grads();
    }
    (a, b)
}
