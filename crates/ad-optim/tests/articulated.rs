//! 铰接体 + 系统辨识（设计文档 §12.3 第 43 条；"第一个消费者"）：
//! 1. iLQR 甩摆基准——n=3 组合式动力学（空间代数算子 RNEA + solve_sym）
//!    接入求解器，从下垂零位甩到目标位形；
//! 2. **系统辨识**——从含噪轨迹反推物理参数 (m₂, 阻尼 d)：可微物理的
//!    招牌应用，端到端穿过 CustomOp → 梯度 → Armijo GD。


use ad::{Context, AD};


use ad_optim::{minimize_gradient_descent, Dynamics, IlqrCfg, OptimizerCfg, QuadraticCost, solve_ilqr};
use ad_physics::{articulated_forward, PlanarChain};

const N: usize = 3;
const DT: f64 = 0.02;
const MASSES: [f64; 3] = [1.0, 0.8, 0.6];
const LENGTHS: [f64; 3] = [1.0, 0.9, 0.7];
const GRAV: f64 = 9.81;

/// 组合式动力学的 Dynamics 包装（iLQR 用；质量为常量输入）
struct ArticulatedDyn {
    chain: PlanarChain,
}

impl Dynamics for ArticulatedDyn {
    fn nx(&self) -> usize {
        2 * N
    }
    fn nu(&self) -> usize {
        N
    }
    fn step(&self, ctx: &mut Context<f64>, x: &[AD<f64>], u: &[AD<f64>]) -> Vec<AD<f64>> {
        let q = &x[..N];
        let qd = &x[N..2 * N];
        let masses: Vec<AD<f64>> = self.chain.masses.iter().map(|&v| AD::constant(v)).collect();
        let damping: Vec<AD<f64>> = (0..N).map(|_| AD::constant(0.0)).collect();
        let a = articulated_forward(ctx, &self.chain, &masses, q, qd, u, &damping);
        // 半隐式欧拉：q̇' = q̇ + dt·a；q' = q + dt·q̇'
        let mut w_new = Vec::with_capacity(N);
        for i in 0..N {
            let dv = ctx.mul(AD::constant(DT), a[i]);
            w_new.push(ctx.add(qd[i], dv));
        }
        let mut out = Vec::with_capacity(2 * N);
        for i in 0..N {
            let dq = ctx.mul(AD::constant(DT), w_new[i]);
            out.push(ctx.add(q[i], dq));
        }
        out.extend(w_new);
        out
    }
}

/// iLQR 甩摆：从 (0.3, −0.2, 0.1) 附近甩到 (0.9, 0.4, −0.3)
#[test]
fn ilqr_articulated_swing_up() {
    let chain = PlanarChain::new(&MASSES, &LENGTHS, GRAV);
    let d = ArticulatedDyn { chain };
    let t_total = 80;
    let x0 = vec![0.3, -0.2, 0.1, 0.0, 0.0, 0.0];
    let goal = vec![0.9, 0.4, -0.3, 0.0, 0.0, 0.0];
    let cost = QuadraticCost {
        q: vec![0.5; 6],
        r: vec![0.01; N],
        qf: vec![20.0, 20.0, 20.0, 1.0, 1.0, 1.0],
        goal: goal.clone(),
    };
    let u0: Vec<Vec<f64>> = (0..t_total).map(|t| vec![0.2 * (t as f64 * 0.2).sin(); N]).collect();
    let cfg = IlqrCfg {
        max_iters: 60,
        ..Default::default()
    };
    let (_u, rep) = solve_ilqr(&d, &x0, &u0, &cost, &cfg);
    eprintln!(
        "铰接体 iLQR：loss {:.4} → {:.4}（{} 迭代）",
        rep.loss0, rep.loss, rep.iters
    );
    assert!(rep.loss < rep.loss0 * 0.5, "损失未显著下降");
}

// ============================================================ 系统辨识

/// 观测数据（真值动力学生成）
struct ObsData {
    x0: Vec<f64>,
    tau: Vec<Vec<f64>>,
    q: Vec<Vec<f64>>,
}

fn observations() -> ObsData {
    let chain = PlanarChain::new(&[MASSES[0], 0.8, MASSES[2]], &LENGTHS, GRAV);
    let d_true = 0.15;
    let x0 = vec![0.4, -0.25, 0.15, 0.0, 0.0, 0.0];
    let tau: Vec<Vec<f64>> = (0..150).map(|t| vec![0.3 * (t as f64 * 0.15).sin(); N]).collect();

    let mut q_obs: Vec<Vec<f64>> = Vec::new();
    let mut st = x0.clone();
    for t in 0..150 {
        let mut ctx = Context::<f64>::new();
        let masses: Vec<AD<f64>> = chain.masses.iter().map(|&v| AD::constant(v)).collect();
        let q: Vec<AD<f64>> = st[..N].iter().map(|&v| ctx.var(v).0).collect();
        let qd: Vec<AD<f64>> = st[N..].iter().map(|&v| ctx.var(v).0).collect();
        let t_ad: Vec<AD<f64>> = tau[t].iter().map(|&v| AD::constant(v)).collect();
        let d: Vec<AD<f64>> = (0..N).map(|_| AD::constant(d_true)).collect();
        let a = articulated_forward(&mut ctx, &chain, &masses, &q, &qd, &t_ad, &d);
        let mut w_new = [0.0f64; N];
        for i in 0..N {
            w_new[i] = st[N + i] + DT * a[i].value;
            st[i] += DT * w_new[i];
            st[N + i] = w_new[i];
        }
        q_obs.push(st[..N].to_vec());
    }
    ObsData { x0, tau, q: q_obs }
}

/// 含阻尼组合式动力学 rollout + 参数梯度：
/// loss(m₂, d) = ½·Σ ||q_sim(t; m₂, d) − q_obs(t)||²。
/// **关键**：m₂ 与 d 以 AD 叶子进入质量/阻尼输入（此前 m₂ 被烘焙为
/// f64 常量导致梯度恒 0——参数必须经图传播）。
fn sysid_loss_and_grad(params: &[f64], g: &mut [f64]) -> f64 {
    let (m2, dmp) = (params[0], params[1]);
    let chain = PlanarChain::new(&[MASSES[0], m2, MASSES[2]], &LENGTHS, GRAV);
    let obs = observations();

    let mut ctx = Context::<f64>::new();
    let (m2_ad, vm2) = ctx.var(m2);
    let (d_ad, vd) = ctx.var(dmp);
    let masses: Vec<AD<f64>> = vec![
        AD::constant(MASSES[0]),
        m2_ad,
        AD::constant(MASSES[2]),
    ];
    let vars: Vec<ad::Variable> = vec![vm2, vd];
    let mut st: Vec<AD<f64>> = obs.x0.iter().map(|&v| ctx.var(v).0).collect();
    let damping: Vec<AD<f64>> = (0..N).map(|_| d_ad).collect();

    let mut loss = AD::constant(0.0);
    let steps = obs.q.len();
    for t in 0..steps {
        let q = st[..N].to_vec();
        let qd = st[N..].to_vec();
        let tau: Vec<AD<f64>> = obs.tau[t].iter().map(|&v| AD::constant(v)).collect();
        let a = articulated_forward(&mut ctx, &chain, &masses, &q, &qd, &tau, &damping);
        let mut w_new = Vec::with_capacity(N);
        let mut q_new = Vec::with_capacity(N);
        for i in 0..N {
            let dv = ctx.mul(AD::constant(DT), a[i]);
            w_new.push(ctx.add(qd[i], dv));
        }
        for i in 0..N {
            let dq = ctx.mul(AD::constant(DT), w_new[i]);
            q_new.push(ctx.add(q[i], dq));
        }
        if t % 10 == 0 {
            for i in 0..N {
                let r = ctx.sub(q_new[i], AD::constant(obs.q[t][i]));
                let sq = ctx.mul(r, r);
                loss = ctx.add(loss, sq);
            }
        }
        st = q_new;
        st.extend(w_new);
    }
    let loss = ctx.mul(AD::constant(0.5), loss);
    ctx.backward(loss);
    g[0] = ctx.grad(vars[0]).unwrap();
    g[1] = ctx.grad(vars[1]).unwrap();
    loss.value
}

#[test]
fn sysid_recovers_mass_and_damping() {
    let cfg = OptimizerCfg {
        max_iters: 400,
        lr0: 0.3,
        ..Default::default()
    };
    let x0 = [0.5, 0.0];
    let (x, rep) = minimize_gradient_descent(&x0, &cfg, sysid_loss_and_grad);
    eprintln!(
        "系统辨识：loss {:.4} → {:.6}，估计 m₂ = {:.4}（真值 0.8），d = {:.4}（真值 0.15），{} 迭代",
        rep.loss0, rep.loss, x[0], x[1], rep.iters
    );
    assert!(rep.loss < rep.loss0 * 0.1, "损失未收敛");
    assert!((x[0] - 0.8).abs() < 0.05, "m₂ 估计 {:.4} 偏离真值", x[0]);
    assert!((x[1] - 0.15).abs() < 0.05, "d 估计 {:.4} 偏离真值", x[1]);
    assert!(rep.grad_health.nonfinite_fraction == 0.0);
}
