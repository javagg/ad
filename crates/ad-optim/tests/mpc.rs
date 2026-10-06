//! iLQR 控制约束与 MPC 滚动形态（设计文档 §4.3.3 后续工作、§12.3 第 30 条）：
//! 1. box 约束（clamped iLQR）：前向 pass 钳制 u，backward 不感知边界——
//!    自标定测试：先解无约束问题取峰值控制定界，再验证约束满足 + 仍有改进；
//! 2. MPC 滚动时域：每步解 horizon 上的 iLQR、只施加首控制、前进一步——
//!    Online checkpoint 的承诺场景（MPC 滚动优化）在 iLQR 侧的对应物。

use ad::prelude::*;
use ad_optim::{solve_ilqr, rollout, Dynamics, IlqrCfg, QuadraticCost};

/// 双积分器：p' = p + dt·v；v' = v + dt·u（nx=2, nu=1）。
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
        let dv = ctx.mul(dt, x[1]);
        let p1 = ctx.add(x[0], dv);
        let du = ctx.mul(dt, u[0]);
        let v1 = ctx.add(x[1], du);
        vec![p1, v1]
    }
}

fn cost(goal: [f64; 2], qf_scale: f64) -> QuadraticCost {
    QuadraticCost {
        q: vec![1.0, 0.1],
        r: vec![0.01],
        qf: vec![10.0 * qf_scale, 1.0 * qf_scale],
        goal: vec![goal[0], goal[1]],
    }
}

/// 无约束解的峰值 |u|（自标定：把界取在其 60% 处保证约束主动）
fn peak_control(u: &[Vec<f64>]) -> f64 {
    u.iter().map(|ui| ui[0].abs()).fold(0.0f64, f64::max)
}

#[test]
fn ilqr_box_constraints_saturate_and_improve() {
    let d = DoubleIntegrator { dt: 0.1 };
    let t_total = 20;
    let x0 = [0.0, 0.6]; // 带初速度，需减速
    let c = cost([0.0, 0.0], 1.0);

    // 无约束解：u0 给一个非平凡波形
    let u0: Vec<Vec<f64>> = (0..t_total).map(|t| vec![0.1 * (t as f64 * 0.3).sin()]).collect();
    let cfg = IlqrCfg {
        max_iters: 80,
        ..Default::default()
    };
    let (u_free, rep_free) = solve_ilqr(&d, &x0, &u0, &c, &cfg);
    assert!(rep_free.loss < rep_free.loss0);
    let u_peak = peak_control(&u_free);
    eprintln!("无约束峰值 |u| = {u_peak:.4}");

    // 界 = 60% 峰值 → 约束必然主动
    let bound = 0.6 * u_peak;
    let cfg_box = IlqrCfg {
        max_iters: 80,
        u_min: Some(vec![-bound]),
        u_max: Some(vec![bound]),
        ..Default::default()
    };
    let (u_box, rep_box) = solve_ilqr(&d, &x0, &u0, &c, &cfg_box);
    assert!(
        u_box.iter().all(|ui| ui[0] >= -bound - 1e-12 && ui[0] <= bound + 1e-12),
        "控制越界"
    );
    assert!(
        rep_box.loss < rep_box.loss0,
        "box 约束下损失未下降：{} → {}",
        rep_box.loss0,
        rep_box.loss
    );
    // 约束只会更差（无约束解是可行域下界）
    assert!(
        rep_box.loss >= rep_free.loss - 1e-9,
        "box 解 ({}) 好于无约束解 ({})？",
        rep_box.loss,
        rep_free.loss
    );
    eprintln!(
        "box iLQR：bound = {bound:.4}，loss {} → {}（无约束 {}）",
        rep_box.loss0, rep_box.loss, rep_free.loss
    );
}

#[test]
fn mpc_receding_horizon_reaches_goal_under_bounds() {
    let d = DoubleIntegrator { dt: 0.1 };
    let (horizon, t_real) = (15usize, 30usize);
    let bound = 0.5f64;
    let goal = [0.3, 0.0];

    let mut x = vec![0.0, 0.0];
    let mut applied: Vec<f64> = Vec::new();
    for _ in 0..t_real {
        let c = cost(goal, 0.1); // 滚动时域：终端权重适度
        let u0: Vec<Vec<f64>> = (0..horizon).map(|_| vec![0.0]).collect();
        let cfg = IlqrCfg {
            max_iters: 30,
            u_min: Some(vec![-bound]),
            u_max: Some(vec![bound]),
            ..Default::default()
        };
        let (u_opt, _) = solve_ilqr(&d, &x, &u0, &c, &cfg);
        let u1 = u_opt[0][0];
        applied.push(u1);
        // 真实动力学前进一步（与求解器同一 Dynamics——确定性重算）
        let (xs, _) = rollout(&d, &x, &[vec![u1]], &cost(goal, 0.1));
        x = xs[1].clone();
    }

    assert!(
        applied.iter().all(|&u| u >= -bound - 1e-12 && u <= bound + 1e-12),
        "MPC 施加的控制越界"
    );
    let (err_p, err_v) = ((x[0] - goal[0]).abs(), x[1].abs());
    eprintln!("MPC：p_T = {:.4}（目标 {}），v_T = {:.4}", x[0], goal[0], x[1]);
    assert!(err_p < 0.05, "位置误差 {err_p}");
    assert!(err_v < 0.05, "速度误差 {err_v}");
}
