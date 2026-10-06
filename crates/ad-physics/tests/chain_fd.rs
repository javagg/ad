//! 双关节摆手写 CustomOp（`DoublePendulumStep`）的验证体系：
//! 1. 单步逐坐标 FD 隔离器（§4.3.1 契约标准流程，覆盖全部 7 输入 × 4 输出）；
//! 2. 能量守恒物理先验（被动摆半隐式欧拉，最严格的动力学方程正确性检验）。

use ad_core::{Context, CustomOp, AD};
use ad_physics::DoublePendulumStep;

/// 平滑混合全部输出的标量损失（与 ops_fd.rs 同约定，固定权重确定性）。
/// AD 路径中的 loss 表达式与它逐项一致。
fn loss_of_out(out: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, &o) in out.iter().enumerate() {
        s += (0.3 + 0.11 * i as f64) * o + 0.2 * o * o;
    }
    if out.len() >= 2 {
        s += 0.15 * out[0] * out[out.len() - 1];
    }
    s
}

/// 单步逐坐标 FD：AD（call_custom + 反向）vs `loss(op.forward(x±h))` 中心差分
fn check_op(name: &str, op: DoublePendulumStep, inputs: &[f64], tol: f64) {
    let mut ctx = Context::<f64>::new();
    let mut vars = Vec::new();
    let mut ad_in = Vec::new();
    for &v in inputs {
        let (ad, var) = ctx.var(v);
        ad_in.push(ad);
        vars.push(var);
    }
    let out = ctx.call_custom(op, &ad_in);
    let n = out.len();

    // loss = Σ w_i·o_i + 0.2Σo_i² + 0.15·o_0·o_last（与 loss_of_out 一致）
    let mut l = {
        let lin = ctx.mul(AD::constant(0.3), out[0]);
        let quad0 = ctx.mul(out[0], out[0]);
        let quad = ctx.mul(AD::constant(0.2), quad0);
        ctx.add(lin, quad)
    };
    for (i, o) in out.iter().enumerate().skip(1) {
        let lin = ctx.mul(AD::constant(0.3 + 0.11 * i as f64), *o);
        let sq = ctx.mul(*o, *o);
        let quad = ctx.mul(AD::constant(0.2), sq);
        let sum = ctx.add(lin, quad);
        l = ctx.add(l, sum);
    }
    let cross_in = ctx.mul(out[0], out[n - 1]);
    let cross = ctx.mul(AD::constant(0.15), cross_in);
    l = ctx.add(l, cross);
    ctx.backward(l);
    let ad_grads: Vec<f64> = vars.iter().map(|&v| ctx.grad(v).unwrap()).collect();

    let h = 1e-6;
    let mut bad = 0;
    for j in 0..inputs.len() {
        let mut tp = inputs.to_vec();
        tp[j] += h;
        let mut tm = inputs.to_vec();
        tm[j] -= h;
        let fp = loss_of_out(&op.forward(&tp).0);
        let fm = loss_of_out(&op.forward(&tm).0);
        let fd = (fp - fm) / (2.0 * h);
        let g = ad_grads[j];
        if (g - fd).abs() > tol * (1.0 + g.abs() + fd.abs()) {
            eprintln!("{name}: input[{j}] ad {g:.10} vs fd {fd:.10}");
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{name}: {bad} mismatched coordinates");
}

#[test]
fn fd_double_pendulum_step_rest() {
    // 弱激励状态（接近悬挂平衡）
    let op = DoublePendulumStep::default();
    check_op(
        "dchain/rest",
        op,
        &[0.1, -0.05, 0.02, -0.01, 0.0, 0.0, 0.002],
        1e-5,
    );
}

#[test]
fn fd_double_pendulum_step_swing() {
    // 一般甩摆状态 + 非零力矩
    let op = DoublePendulumStep::default();
    check_op(
        "dchain/swing",
        op,
        &[0.3, -0.2, 0.1, 0.4, 0.2, -0.1, 0.002],
        1e-5,
    );
}

#[test]
fn fd_double_pendulum_step_chaotic() {
    // 大角度混沌状态（能量守恒基准的初始条件附近）
    let op = DoublePendulumStep::default();
    check_op(
        "dchain/chaotic",
        op,
        &[0.8, 0.6, -1.5, 0.7, 0.1, -0.05, 0.002],
        1e-5,
    );
}

#[test]
fn fd_double_pendulum_step_large_dt() {
    // 大 dt（检查 dt 槽位的梯度路径与高阶误差）
    let op = DoublePendulumStep::default();
    check_op(
        "dchain/large_dt",
        op,
        &[0.5, 0.3, 0.2, -0.3, 0.0, 0.0, 0.05],
        1e-5,
    );
}

// ============================================================ 能量守恒先验

/// 独立能量公式（与 ad-optim/tests/chain.rs 的 energy 一致）
fn energy(op: &DoublePendulumStep, th1: f64, th2: f64, w1: f64, w2: f64) -> f64 {
    let (m1, m2, l1, l2, g) = (op.m1, op.m2, op.l1, op.l2, op.g);
    let c2 = th2.cos();
    let d11 = (m1 + m2) * l1 * l1 + m2 * l2 * l2 + 2.0 * m2 * l1 * l2 * c2;
    let d12 = m2 * l2 * l2 + m2 * l1 * l2 * c2;
    let d22 = m2 * l2 * l2;
    let kin = 0.5 * (d11 * w1 * w1 + 2.0 * d12 * w1 * w2 + d22 * w2 * w2);
    let pot = -(m1 + m2) * g * l1 * th1.cos() - m2 * g * l2 * (th1 + th2).cos();
    kin + pot
}

#[test]
fn energy_conservation_passive_chain_op() {
    let op = DoublePendulumStep::default();
    let dt = 0.002;
    let mut st = [0.8f64, 0.6, 0.0, 0.0]; // [θ1, θ2, ω1, ω2]
    let e0 = energy(&op, st[0], st[1], st[2], st[3]);

    for step in 0..2000 {
        let inputs = [st[0], st[1], st[2], st[3], 0.0, 0.0, dt];
        let outs = op.forward(&inputs).0;
        st = [outs[0], outs[1], outs[2], outs[3]];
        if step % 500 == 0 {
            let e = energy(&op, st[0], st[1], st[2], st[3]);
            eprintln!("step {step}: E = {e:.6}");
        }
    }
    let e1 = energy(&op, st[0], st[1], st[2], st[3]);
    let drift = (e1 - e0).abs() / e0.abs().max(1.0);
    eprintln!("E0 = {e0:.4}, E1 = {e1:.4}, drift = {drift:.2e}");
    assert!(
        drift < 0.02,
        "energy drift {drift:.2e} exceeds 2% — forward dynamics equations wrong"
    );
}

#[test]
fn forward_matches_ad_passthrough_reference() {
    // forward 与直通版参考数值一致性：对照解析单步（独立实现，防两处同错）
    // 参考实现直接用文献公式重写一遍（不与本算子共享任何代码路径）
    let (m1, m2, l1, l2, g, dt) = (1.0f64, 0.8, 1.0, 0.9, 9.81, 0.002);
    let op = DoublePendulumStep::default();
    let (th1, th2, w1, w2, t1, t2): (f64, f64, f64, f64, f64, f64) =
        (0.3, -0.2, 0.1, 0.4, 0.2, -0.1);
    let c2 = th2.cos();
    let m11 = (m1 + m2) * l1 * l1 + m2 * l2 * l2 + 2.0 * m2 * l1 * l2 * c2;
    let m12 = m2 * l2 * l2 + m2 * l1 * l2 * c2;
    let m22 = m2 * l2 * l2;
    let h = m2 * l1 * l2 * th2.sin();
    let r1 = t1 + h * (2.0 * w1 * w2 + w2 * w2)
        - (m1 + m2) * g * l1 * th1.sin()
        - m2 * g * l2 * (th1 + th2).sin();
    let r2 = t2 - h * w1 * w1 - m2 * g * l2 * (th1 + th2).sin();
    let det = m11 * m22 - m12 * m12;
    let a1 = (m22 * r1 - m12 * r2) / det;
    let a2 = (m11 * r2 - m12 * r1) / det;

    let want = [
        th1 + dt * (w1 + dt * a1),
        th2 + dt * (w2 + dt * a2),
        w1 + dt * a1,
        w2 + dt * a2,
    ];
    let got = op
        .forward(&[th1, th2, w1, w2, t1, t2, dt])
        .0
        .to_vec();
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-14, "forward {g} vs reference {w}");
    }
}
