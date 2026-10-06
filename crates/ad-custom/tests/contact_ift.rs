//! 接触求解器的 IFT 完整形态（设计文档 §4.3.3 的动机难题；§12.3 第 34 条）。
//!
//! 模式：**active-set + IFT**——互补条件 (λ ≥ 0, φ ≥ 0, λφ = 0) 的光滑性
//! 由活动集承载：活动集稳定时，对"当前活动集上的等式系统"用
//! [`ImplicitSolve`] 求解并穿透求导（梯度在该域内精确）；活动集切换是
//! 半光滑边界（warm-start 承载上一时刻的活动集，IFT 梯度在集内有效）。
//!
//! 四个验证（oracle = 对求解器本身做 FD——IFT 伴随 vs 差分穿透）：
//! 1. 1-DOF 非线性接触（隐式欧拉 + Hertz 型 `kδ + cδ³`）的活动分支；
//! 2. 2×2 法向冲量求解（双质点静置）；
//! 3. 3×2 冗余超定（第三约束 = 前两个的凸组合——静不定接触）走正规方程；
//! 4. warm-start λ：多时间步间传递上一时刻的冲量（checkpoint 安全形态），
//!    x0 槽位梯度恒 0。

use ad_core::{Context, CustomOp, AD};
use ad_custom::{ImplicitSolve, ImplicitSolveCfg, Residual};
use ad_verify::op_check::mixed_output_loss;
use num_traits::Num;
use std::rc::Rc;

/// 通过求解器本身的中心差分 oracle：`loss(op.forward(θ±h))`。
/// IFT 伴随绕过迭代路径（O(1)、无截断偏差），FD 穿透含 Newton 截断——
/// 两者在收敛解处一致（这正是"IFT 梯度无截断偏差"的实证方式）。
fn fd_through_solver(op: &dyn CustomOp<f64>, theta: &[f64], coord: usize, h: f64) -> f64 {
    let mut tp = theta.to_vec();
    tp[coord] += h;
    let mut tm = theta.to_vec();
    tm[coord] -= h;
    let lp = mixed_output_loss(&op.forward(&tp).0);
    let lm = mixed_output_loss(&op.forward(&tm).0);
    (lp - lm) / (2.0 * h)
}

fn ad_grads(op: Rc<dyn CustomOp<f64>>, theta: &[f64]) -> Vec<f64> {
    let mut ctx = Context::<f64>::new();
    let mut ads = Vec::new();
    let mut vars = Vec::new();
    for &v in theta {
        let (ad, var) = ctx.var(v);
        ads.push(ad);
        vars.push(var);
    }
    let outs = ctx.call_custom_dyn(Rc::clone(&op), "contact_ift_op", &ads);
    let lin0 = ctx.mul(AD::constant(0.3), outs[0]);
    let sq0 = ctx.mul(outs[0], outs[0]);
    let quad0 = ctx.mul(AD::constant(0.2), sq0);
    let mut loss = ctx.add(lin0, quad0);
    for (i, o) in outs.iter().enumerate().skip(1) {
        let lin = ctx.mul(AD::constant(0.3 + 0.11 * i as f64), *o);
        let sq = ctx.mul(*o, *o);
        let quad = ctx.mul(AD::constant(0.2), sq);
        let lq = ctx.add(lin, quad);
        loss = ctx.add(loss, lq);
    }
    if outs.len() >= 2 {
        let xy = ctx.mul(outs[0], outs[outs.len() - 1]);
        let cross = ctx.mul(AD::constant(0.15), xy);
        loss = ctx.add(loss, cross);
    }
    ctx.backward(loss);
    vars.iter().map(|&v| ctx.grad(v).unwrap()).collect()
}

fn crosscheck(op: Rc<dyn CustomOp<f64>>, theta: &[f64], tol: f64, label: &str) {
    let gs = ad_grads(Rc::clone(&op), theta);
    let mut bad = 0;
    for (j, &g) in gs.iter().enumerate() {
        let fd = fd_through_solver(op.as_ref(), theta, j, 1e-6);
        if (g - fd).abs() > tol * (1.0 + g.abs() + fd.abs()) {
            eprintln!("{label}: θ[{j}] ift {g:.10} vs fd {fd:.10}");
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{label}: {bad} 个坐标不一致");
}

// ============================================================ 1. 1-DOF 非线性接触

/// 隐式欧拉 + Hertz 型接触弹簧的活动分支。未知 x = q⁺（高度），
/// θ = [q0, v0, h, k, c, m, g, dt]；穿透 δ = h − q⁺ > 0 时
/// `r = m(q⁺ − q0 − dt·v0) + dt²(kδ + cδ³ + m·g) = 0`。
struct ContactResidual;

impl Residual for ContactResidual {
    fn nx(&self) -> usize {
        1
    }
    fn ntheta(&self) -> usize {
        8
    }
    fn residual<N: Num + Copy>(&self, x: &[N], theta: &[N], r: &mut [N]) {
        let (q0, v0, h, k, c, m, g, dt) = (
            theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7],
        );
        let d = h - x[0]; // δ⁺
        let d3 = d * d * d;
        r[0] = m * (x[0] - q0 - dt * v0) + dt * dt * (k * d + c * d3 + m * g);
    }
}

#[test]
fn contact_1dof_active_branch_gradients() {
    let theta = [0.05, -0.5, 0.1, 50.0, 10.0, 1.0, 9.81, 0.01];
    let op: Rc<dyn CustomOp<f64>> = Rc::new(ImplicitSolve::new(ContactResidual));

    // 前向：解存在且活动集稳定（穿透 δ⁺ > 0，接触力 > 0）
    let (outs, _) = op.forward(&theta);
    let qp = outs[0];
    let delta = theta[2] - qp;
    assert!(delta > 1e-4, "活动分支前提不成立：δ⁺ = {delta}");
    let force = theta[3] * delta + theta[4] * delta * delta * delta;
    assert!(force > 0.0, "接触力必须为正（压缩）: {force}");

    // IFT 伴随 vs 穿透求解器的 FD（8 个参数坐标全查）
    crosscheck(Rc::clone(&op), &theta, 1e-6, "contact_1dof");
}

// ============================================================ 2. 2×2 法向冲量

/// 双质点静置：r = [v1 + λ1/m1, v2 + λ2/m2]（解后法向速度归零）。
/// θ = [v1, v2, m1, m2]，未知 λ = [λ1, λ2]。
struct ImpulseResidual;

impl Residual for ImpulseResidual {
    fn nx(&self) -> usize {
        2
    }
    fn ntheta(&self) -> usize {
        4
    }
    fn residual<N: Num + Copy>(&self, x: &[N], theta: &[N], r: &mut [N]) {
        let (v1, v2, m1, m2) = (theta[0], theta[1], theta[2], theta[3]);
        r[0] = v1 + x[0] / m1;
        r[1] = v2 + x[1] / m2;
    }
}

#[test]
fn impulse_2x2_gradients() {
    let theta = [-1.2, 0.4, 2.0, 0.8];
    let op: Rc<dyn CustomOp<f64>> = Rc::new(ImplicitSolve::new(ImpulseResidual));

    let (outs, _) = op.forward(&theta);
    // 闭式：λ_i = −m_i·v_i
    assert!((outs[0] - (-theta[2] * theta[0])).abs() < 1e-12);
    assert!((outs[1] - (-theta[3] * theta[1])).abs() < 1e-12);

    crosscheck(Rc::clone(&op), &theta, 1e-6, "impulse_2x2");
}

// ============================================================ 3. 3×2 冗余超定（静不定接触）

/// 第三个约束 = 前两个的凸组合（r3 = ½(r1 + r2)）——J 满列秩、系统相容，
/// 走 Gauss–Newton 正规方程路径。解与 2×2 相同（行空间不变）。
struct RedundantImpulseResidual;

impl Residual for RedundantImpulseResidual {
    fn nx(&self) -> usize {
        2
    }
    fn ntheta(&self) -> usize {
        4
    }
    fn nr(&self) -> usize {
        3
    }
    fn residual<N: Num + Copy>(&self, x: &[N], theta: &[N], r: &mut [N]) {
        let (v1, v2, m1, m2) = (theta[0], theta[1], theta[2], theta[3]);
        r[0] = v1 + x[0] / m1;
        r[1] = v2 + x[1] / m2;
        // r3 = ½(r1 + r2)：行空间不变 → 相容冗余（解与 2×2 相同）
        let half = N::one() / (N::one() + N::one());
        r[2] = (r[0] + r[1]) * half;
    }
}

#[test]
fn redundant_impulse_3x2_matches_square_solution() {
    let theta = [-1.2, 0.4, 2.0, 0.8];
    let sq = Rc::new(ImplicitSolve::new(ImpulseResidual));
    let rect: Rc<dyn CustomOp<f64>> = Rc::new(ImplicitSolve::new(RedundantImpulseResidual));
    let rect_concrete = ImplicitSolve::new(RedundantImpulseResidual);

    // 行空间不变 → 解相同
    let (o_sq, _) = sq.forward(&theta);
    let (o_rect, _) = rect.forward(&theta);
    for (a, b) in o_sq.iter().zip(o_rect.iter()) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    // 超定伴随（正规方程 + 最小范数 λ）vs 穿透 FD
    crosscheck(Rc::clone(&rect), &theta, 1e-6, "impulse_3x2");

    // 与条件数探针的衔接：正规矩阵 JᵀJ 的 κ 可查（κ(JᵀJ) = κ(J)²）
    let cols = rect_concrete.jacobian_x(&o_rect, &theta); // cols[j][p] = ∂r_p/∂x_j
    let jtj = vec![
        vec![
            cols[0].iter().map(|v| v * v).sum::<f64>(),
            cols[0].iter().zip(cols[1].iter()).map(|(a, b)| a * b).sum(),
        ],
        vec![
            cols[0].iter().zip(cols[1].iter()).map(|(a, b)| a * b).sum(),
            cols[1].iter().map(|v| v * v).sum::<f64>(),
        ],
    ];
    let kappa = ad_verify::condition_number_inf(&jtj);
    assert!(kappa.is_finite() && kappa > 0.0, "κ(JᵀJ) = {kappa}");
    eprintln!("3×2 冲量系统：κ(JᵀJ) = {kappa:.3}");
}

// ============================================================ 4. warm-start λ（多时间步）

#[test]
fn warm_started_impulse_across_steps() {
    let cfg = ImplicitSolveCfg {
        max_iters: 8,
        ..Default::default()
    };
    let op: Rc<dyn CustomOp<f64>> = Rc::new(ImplicitSolve::with_warm_start(ImpulseResidual, cfg));
    let theta1 = [-1.2, 0.4, 2.0, 0.8];
    let theta2 = [-0.9, 0.3, 2.0, 0.8]; // 下一时间步（速度衰减）

    // 步 1：冷启动求解 λ₁
    let (lam1, _) = op.forward(&[theta1.as_slice(), &[0.0, 0.0]].concat());
    // 步 2：以 λ₁ 为初值（上一时刻活动集的延续）
    let input2 = [theta2.as_slice(), lam1.as_slice()].concat();
    let (lam2, _) = op.forward(&input2);
    // 步 2 的解与冷启动一致（线性系统 + 收敛解与初值无关）
    let (lam2_cold, _) = op.forward(&[theta2.as_slice(), &[0.0, 0.0]].concat());
    for (a, b) in lam2.iter().zip(lam2_cold.iter()) {
        assert!((a - b).abs() < 1e-12);
    }

    // 步 2 梯度（warm-start 形态）：θ 坐标 vs FD；x0 槽位梯度恒 0
    crosscheck(Rc::clone(&op), &input2, 1e-6, "impulse_warm");
    let gs = ad_grads(Rc::clone(&op), &input2);
    for (k, &g) in gs.iter().enumerate().skip(4) {
        assert_eq!(g, 0.0, "x0 槽位 {k} 梯度应为 0，得到 {g}");
    }
}
