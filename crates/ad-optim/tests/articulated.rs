//! 铰接体 + 系统辨识（设计文档 §12.3 第 43 条；"第一个消费者"）：
//! 1. iLQR 甩摆基准——n=3 组合式动力学（空间代数算子 RNEA + solve_sym）
//!    接入求解器，从下垂零位甩到目标位形；
//! 2. **系统辨识**——从含噪轨迹反推物理参数 (m₂, 阻尼 d)：可微物理的
//!    招牌应用，端到端穿过 CustomOp → 梯度 → Armijo GD；
//! 3. **积分器对照**（§9.2 / 第 46 条）：同一组合动力学，RK4 vs 半隐式
//!    欧拉——rollout 精度（对细步长参考）与 iLQR 收敛的量化对比。

use ad::{Context, AD};

use ad_optim::{
    minimize_gradient_descent, solve_ilqr, Dynamics, IlqrCfg, OptimizerCfg, QuadraticCost,
};
use ad_physics::{articulated_forward, Integrator, PlanarChain, Rk4, SemiImplicitEuler};

const N: usize = 3;
const DT: f64 = 0.02;
const MASSES: [f64; 3] = [1.0, 0.8, 0.6];
const LENGTHS: [f64; 3] = [1.0, 0.9, 0.7];
const GRAV: f64 = 9.81;

/// 组合式动力学的 Dynamics 包装（iLQR 用；质量为常量输入；
/// 积分器为可插拔策略——第 46 条）
struct ArticulatedDyn {
    chain: PlanarChain,
    integ: &'static dyn Integrator<f64>,
}

impl ArticulatedDyn {
    fn new(chain: PlanarChain, integ: &'static dyn Integrator<f64>) -> Self {
        ArticulatedDyn { chain, integ }
    }
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
        let chain = self.chain.clone();
        let u_owned: Vec<AD<f64>> = u.to_vec();
        let accel = move |ctx: &mut Context<f64>, q: &[AD<f64>], qd: &[AD<f64>], u: &[AD<f64>]| {
            articulated_forward(ctx, &chain, &masses, q, qd, u, &damping)
        };
        let (q_new, w_new) = self.integ.step(ctx, &accel, q, qd, &u_owned, DT);
        let mut out = q_new;
        out.extend(w_new);
        out
    }
}

/// iLQR 甩摆：从 (0.3, −0.2, 0.1) 附近甩到 (0.9, 0.4, −0.3)
#[test]
fn ilqr_articulated_swing_up() {
    let chain = PlanarChain::new(&MASSES, &LENGTHS, GRAV);
    let d = ArticulatedDyn::new(chain, &SemiImplicitEuler);
    let t_total = 80;
    let x0 = vec![0.3, -0.2, 0.1, 0.0, 0.0, 0.0];
    let goal = vec![0.9, 0.4, -0.3, 0.0, 0.0, 0.0];
    let cost = QuadraticCost {
        q: vec![0.5; 6],
        r: vec![0.01; N],
        qf: vec![20.0, 20.0, 20.0, 1.0, 1.0, 1.0],
        goal: goal.clone(),
    };
    let u0: Vec<Vec<f64>> = (0..t_total)
        .map(|t| vec![0.2 * (t as f64 * 0.2).sin(); N])
        .collect();
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

// ============================================================ 积分器对照（第 46 条）

/// 被动 rollout（无控制——零阶保持的输入失配会主导轨迹差，淹没积分器阶数）
fn rollout_states(integ: &'static dyn Integrator<f64>, dt: f64, steps: usize) -> Vec<[f64; N]> {
    let chain = PlanarChain::new(&MASSES, &LENGTHS, GRAV);
    let masses: Vec<AD<f64>> = chain.masses.iter().map(|&v| AD::constant(v)).collect();
    let damping: Vec<AD<f64>> = (0..N).map(|_| AD::constant(0.0)).collect();
    let zeros: Vec<AD<f64>> = (0..N).map(|_| AD::constant(0.0)).collect();
    let mut q: Vec<f64> = vec![0.3, -0.2, 0.1];
    let mut qd: Vec<f64> = vec![0.5, -0.3, 0.2];
    let mut traj = Vec::with_capacity(steps);
    for _ in 0..steps {
        let mut ctx = Context::<f64>::new();
        let q_ad: Vec<AD<f64>> = q.iter().map(|&v| ctx.var(v).0).collect();
        let qd_ad: Vec<AD<f64>> = qd.iter().map(|&v| ctx.var(v).0).collect();
        let u_empty: Vec<AD<f64>> = Vec::new();
        let accel = |ctx: &mut Context<f64>, q: &[AD<f64>], qd: &[AD<f64>], u: &[AD<f64>]| {
            let _ = u;
            articulated_forward(ctx, &chain, &masses, q, qd, &zeros, &damping)
        };
        let (qn, wn) = integ.step(&mut ctx, &accel, &q_ad, &qd_ad, &u_empty, dt);
        for i in 0..N {
            q[i] = qn[i].value;
            qd[i] = wn[i].value;
        }
        traj.push([q[0], q[1], q[2]]);
    }
    traj
}

/// 组合式动力学的 iLQR 端到端损失（给定积分器）
fn ilqr_final_loss(integ: &'static dyn Integrator<f64>) -> (f64, f64, usize) {
    let chain = PlanarChain::new(&MASSES, &LENGTHS, GRAV);
    let d = ArticulatedDyn::new(chain, integ);
    let t_total = 80;
    let x0 = vec![0.3, -0.2, 0.1, 0.0, 0.0, 0.0];
    let goal = vec![0.9, 0.4, -0.3, 0.0, 0.0, 0.0];
    let cost = QuadraticCost {
        q: vec![0.5; 6],
        r: vec![0.01; N],
        qf: vec![20.0, 20.0, 20.0, 1.0, 1.0, 1.0],
        goal,
    };
    let u0: Vec<Vec<f64>> = (0..t_total)
        .map(|t| vec![0.2 * (t as f64 * 0.2).sin(); N])
        .collect();
    let cfg = IlqrCfg {
        max_iters: 60,
        ..Default::default()
    };
    let (_u, rep) = solve_ilqr(&d, &x0, &u0, &cost, &cfg);
    (rep.loss0, rep.loss, rep.iters)
}

#[test]
fn integrator_comparison_rk4_vs_euler() {
    // 1) 单步局部截断误差（0.02 s 同一物理区间；多步长程会被混沌放大
    //    打到误差本底、淹没阶数差——首跑 20/80 步实测比值仅 1.5–2×）：
    //    Euler(dt) 与 RK4(dt) 各推 1 步，对照 RK4(dt/10) 推 10 步的参考
    let ref_traj = rollout_states(&Rk4, DT / 10.0, 10);
    let q_ref = ref_traj[9];
    let one_step = |integ: &'static dyn Integrator<f64>| {
        let traj = rollout_states(integ, DT, 1);
        (0..N)
            .map(|i| (traj[0][i] - q_ref[i]).abs())
            .fold(0.0, f64::max)
    };
    let (err_e, err_r) = (one_step(&SemiImplicitEuler), one_step(&Rk4));
    eprintln!(
        "单步局部截断（dt={DT}）：Euler {err_e:.3e}，RK4 {err_r:.3e}（比值 {:.0}×）",
        err_e / err_r
    );
    assert!(
        err_r * 100.0 < err_e,
        "RK4 单步误差 {err_r:.2e} 应比 Euler {err_e:.2e} 低两个数量级以上"
    );

    // 2) 能量漂移（无驱动、无阻尼，4 s @ dt=0.002——dt=0.01 对此链的
    //    高频模式太大，两种积分器同样漂）：半隐式欧拉辛——有界振荡 O(dt)；
    //    RK4 漂移 O(dt⁴)——数量级差
    let chain = PlanarChain::new(&MASSES, &LENGTHS, GRAV);
    let masses: Vec<AD<f64>> = chain.masses.iter().map(|&v| AD::constant(v)).collect();
    let zeros: Vec<AD<f64>> = (0..N).map(|_| AD::constant(0.0)).collect();
    // 独立能量公式：势能 = −Σ mᵢ·g·深度（基座 x 向下）；动能用闭式速度链。
    // **θ̇_j 必须是关节速率的累积和**（勘误见 ad-physics 同名测试/第 46 条）
    let energy = |q: &[f64; N], w: &[f64; N]| {
        let mut kin = 0.0;
        let mut pot = 0.0;
        let mut th_acc = 0.0;
        let mut x_acc = 0.0;
        for i in 0..N {
            th_acc += q[i];
            x_acc += LENGTHS[i] * th_acc.cos();
            let (mut vx, mut vz) = (0.0, 0.0);
            let mut thj = 0.0;
            let mut wdj = 0.0;
            for j in 0..=i {
                thj += q[j];
                wdj += w[j];
                vx += -thj.sin() * LENGTHS[j] * wdj;
                vz += -thj.cos() * LENGTHS[j] * wdj;
            }
            kin += 0.5 * MASSES[i] * (vx * vx + vz * vz);
            pot -= MASSES[i] * GRAV * x_acc;
        }
        kin + pot
    };
    let drift = |integ: &'static dyn Integrator<f64>| {
        let (mut q, mut w) = ([0.4f64, -0.3, 0.2], [0.0f64; N]);
        let e0 = energy(&q, &w);
        let mut worst = 0.0f64;
        for t in 0..2000 {
            let mut ctx = Context::<f64>::new();
            let q_ad: Vec<AD<f64>> = q.iter().map(|&v| ctx.var(v).0).collect();
            let w_ad: Vec<AD<f64>> = w.iter().map(|&v| ctx.var(v).0).collect();
            let u_empty: Vec<AD<f64>> = Vec::new();
            let accel = |ctx: &mut Context<f64>, q: &[AD<f64>], qd: &[AD<f64>], u: &[AD<f64>]| {
                let _ = u;
                articulated_forward(ctx, &chain, &masses, q, qd, &zeros, &zeros)
            };
            let (qn, wn) = integ.step(&mut ctx, &accel, &q_ad, &w_ad, &u_empty, 0.002);
            for i in 0..N {
                q[i] = qn[i].value;
                w[i] = wn[i].value;
            }
            if t % 100 == 0 {
                worst = worst.max((energy(&q, &w) - e0).abs());
            }
        }
        worst
    };
    let (d_e, d_r) = (drift(&SemiImplicitEuler), drift(&Rk4));
    eprintln!(
        "无驱动能量漂移（4 s, dt=0.002）：Euler {d_e:.3e}，RK4 {d_r:.3e}（比值 {:.0}×）",
        d_e / d_r
    );
    assert!(
        d_r * 10.0 < d_e,
        "RK4 能量漂移 {d_r:.2e} 应远小于 Euler {d_e:.2e}"
    );

    // 3) iLQR 收敛：两种积分器都显著收敛；RK4 的离散问题更接近真实
    //    动力学（对照数字入 design.md 第 46 条，不做强序断言）
    let (l0_e, l_e, it_e) = ilqr_final_loss(&SemiImplicitEuler);
    let (l0_r, l_r, it_r) = ilqr_final_loss(&Rk4);
    eprintln!(
        "iLQR 甩摆：Euler loss {l0_e:.4} → {l_e:.4}（{it_e} 迭代）；RK4 loss {l0_r:.4} → {l_r:.4}（{it_r} 迭代）"
    );
    assert!(l_e < l0_e * 0.5, "Euler iLQR 未收敛：{l_e} vs {l0_e}");
    assert!(l_r < l0_r * 0.5, "RK4 iLQR 未收敛：{l_r} vs {l0_r}");
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
    let tau: Vec<Vec<f64>> = (0..150)
        .map(|t| vec![0.3 * (t as f64 * 0.15).sin(); N])
        .collect();

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
    let masses: Vec<AD<f64>> = vec![AD::constant(MASSES[0]), m2_ad, AD::constant(MASSES[2])];
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

// ============================================================ 数据驱动链描述（第 47 条）

/// 描述构建（URDF-lite）驱动的 iLQR 重跑：与硬编码构建的优化轨迹
/// **逐位一致**（同一构建产物 → 同一 tape → 同一求解轨迹）。
#[test]
fn ilqr_from_chain_desc_bit_identical() {
    let desc = ad_physics::ChainDesc::new(GRAV)
        .joint(1.0, 1.0, 0.0)
        .joint(0.8, 0.9, 0.0)
        .joint(0.6, 0.7, 0.0);
    let (chain_desc, _damping) = desc.to_chain();
    let chain_hard = PlanarChain::new(&MASSES, &LENGTHS, GRAV);

    let run = |chain: PlanarChain| -> (f64, f64, usize) {
        let d = ArticulatedDyn::new(chain, &SemiImplicitEuler);
        let t_total = 80;
        let x0 = vec![0.3, -0.2, 0.1, 0.0, 0.0, 0.0];
        let goal = vec![0.9, 0.4, -0.3, 0.0, 0.0, 0.0];
        let cost = QuadraticCost {
            q: vec![0.5; 6],
            r: vec![0.01; N],
            qf: vec![20.0, 20.0, 20.0, 1.0, 1.0, 1.0],
            goal,
        };
        let u0: Vec<Vec<f64>> = (0..t_total)
            .map(|t| vec![0.2 * (t as f64 * 0.2).sin(); N])
            .collect();
        let cfg = IlqrCfg {
            max_iters: 60,
            ..Default::default()
        };
        let (_u, rep) = solve_ilqr(&d, &x0, &u0, &cost, &cfg);
        (rep.loss0, rep.loss, rep.iters)
    };
    let (l0_d, l_d, it_d) = run(chain_desc);
    let (l0_h, l_h, it_h) = run(chain_hard);
    eprintln!(
        "描述驱动 iLQR：{l0_d:.6} → {l_d:.6}（{it_d} 迭代）；硬编码：{l0_h:.6} → {l_h:.6}（{it_h} 迭代）"
    );
    assert_eq!(l0_d.to_bits(), l0_h.to_bits(), "初始损失非逐位一致");
    assert_eq!(l_d.to_bits(), l_h.to_bits(), "最终损失非逐位一致");
    assert_eq!(it_d, it_h);
}
