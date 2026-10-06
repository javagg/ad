//! 优化器健康度诊断闭环（设计文档 §4.5.3 的承诺落地；§12.3 第 35 条）。
//!
//! 项目核心命题：梯度数值正确 ≠ 优化可用。本文件把健康度信号接进两个
//! 优化器的**输出**，使"为什么收敛不好"可以被机器读到而不是人工注释：
//! 1. iLQR：`quu_cond_max`（控制通道二阶信息的条件数，随 R/刚度病态化）、
//!    `mu_final`（正则化被推高 = 实际下降达不到预期）、
//!    `line_search_rejections`（混沌接触问题的特征）；
//! 2. GD：`grad_health`（范数/零占比/非有限占比）+ 既有
//!    `line_search_failures`（梯度与损失不一致的坏梯度信号）。


use ad::{Context, AD};
use ad_optim::{minimize_gradient_descent, solve_ilqr, Dynamics, IlqrCfg, OptimizerCfg, QuadraticCost};

/// 双积分器 + 冗余双控制：p' = p + dt·v；v' = v + dt·(u1 + u2)。
/// nu=2 但两控制作用共线 → R 压低时 Q_uu 趋向秩亏（nu=1 时 Q_uu 是标量，
/// 条件数恒 1，无法演示病态化）。
struct DoubleIntegrator {
    dt: f64,
}

impl Dynamics for DoubleIntegrator {
    fn nx(&self) -> usize {
        2
    }
    fn nu(&self) -> usize {
        2
    }
    fn step(&self, ctx: &mut Context<f64>, x: &[AD<f64>], u: &[AD<f64>]) -> Vec<AD<f64>> {
        let dt = AD::constant(self.dt);
        let dv = ctx.mul(dt, x[1]);
        let p1 = ctx.add(x[0], dv);
        let usum = ctx.add(u[0], u[1]);
        let du = ctx.mul(dt, usum);
        let v1 = ctx.add(x[1], du);
        vec![p1, v1]
    }
}

#[test]
fn ilqr_quu_condition_grows_as_control_cheapens() {
    // R（控制权重）从 1.0 压到 1e-4：Q_uu = R + BᵀV_xx B 中 R 项缩水，
    // 而共线控制使 BᵀV_xxB 秩亏 → 条件数病态化——条件数探针应当读出来
    let d = DoubleIntegrator { dt: 0.1 };
    let t_total = 20;
    let x0 = [0.0, 0.6];
    let u0: Vec<Vec<f64>> = (0..t_total).map(|_| vec![0.05, 0.05]).collect();

    let mut conds = Vec::new();
    for &r in &[1.0f64, 1e-2, 1e-4] {
        let cost = QuadraticCost {
            q: vec![1.0, 0.1],
            r: vec![r, r],
            qf: vec![10.0, 1.0],
            goal: vec![0.0, 0.0],
        };
        let cfg = IlqrCfg {
            max_iters: 40,
            ..Default::default()
        };
        let (_, rep) = solve_ilqr(&d, &x0, &u0, &cost, &cfg);
        eprintln!("R = {r:.0e}: κ(Q_uu)_max = {:.3e}, mu_final = {:.3e}, rejections = {}",
            rep.quu_cond_max, rep.mu_final, rep.line_search_rejections);
        assert!(rep.quu_cond_max.is_finite() && rep.quu_cond_max > 0.0);
        conds.push(rep.quu_cond_max);
    }
    // 控制越便宜（R 越小），Q_uu 越接近奇异 → 条件数单调显著增长
    assert!(
        conds[2] > conds[0] * 10.0,
        "R 跨 4 个量级而条件数未增长: {conds:?}"
    );
}

#[test]
fn gd_reports_gradient_health_and_catches_inconsistent_grad() {
    // Rosenbrock（AD 表达式）：良态梯度 → 收敛 + 健康画像干净
    fn rosenbrock_ad(x: &[f64], g: &mut [f64]) -> f64 {
        let mut ctx = Context::<f64>::new();
        let (x0, vx0) = ctx.var(x[0]);
        let (x1, vx1) = ctx.var(x[1]);
        let xx = ctx.mul(x0, x0);
        let d = ctx.sub(x1, xx);
        let sq = ctx.mul(d, d);
        let t1 = ctx.mul(AD::constant(100.0), sq);
        let e = ctx.sub(AD::constant(1.0), x0);
        let t2 = ctx.mul(e, e);
        let loss = ctx.add(t1, t2);
        ctx.backward(loss);
        g[0] = ctx.grad(vx0).unwrap();
        g[1] = ctx.grad(vx1).unwrap();
        loss.value
    }

    let cfg = OptimizerCfg {
        max_iters: 2000,
        ..Default::default()
    };
    let (_x, rep) = minimize_gradient_descent(&[-1.2, 1.0], &cfg, rosenbrock_ad);
    assert!(rep.grad_health.nonfinite_fraction == 0.0, "健康梯度不应有非有限分量");
    assert!(rep.grad_health.norm.is_finite() && rep.grad_health.norm >= 0.0);
    eprintln!(
        "Rosenbrock GD：iters = {}，‖∇‖ = {:.3e}，zero_frac = {:.2}",
        rep.iters, rep.grad_health.norm, rep.grad_health.zero_fraction
    );

    // 坏梯度（与损失不一致：方向翻转 + 缩放）→ Armijo 连续拒收，
    // 既有信号 line_search_failures 把它读出来
    fn broken_grad(x: &[f64], g: &mut [f64]) -> f64 {
        rosenbrock_ad(x, g);
        g[0] *= -0.9;
        g[1] *= -0.9;
        // f(x) 原样返回
        let (x0, x1) = (x[0], x[1]);
        100.0 * (x1 - x0 * x0) * (x1 - x0 * x0) + (1.0 - x0) * (1.0 - x0)
    }
    let (_xb, rep_bad) = minimize_gradient_descent(&[-1.2, 1.0], &cfg, broken_grad);
    eprintln!(
        "坏梯度 GD：line_search_failures = {}，converged = {}，‖∇‖ = {:.3e}",
        rep_bad.line_search_failures, rep_bad.converged, rep_bad.grad_health.norm
    );
    assert!(
        rep_bad.line_search_failures > 0,
        "不一致梯度必须被 Armijo 拒收信号捕捉"
    );
}
