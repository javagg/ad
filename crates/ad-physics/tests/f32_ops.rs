//! f32 物理算子验证（§12.3 第 38 条）：泛型 CustomOp 的 f32 形态。
//!
//! 验证策略与 `ad-ops/tests/f32.rs` 一致——**同一份泛型代码**在 f64
//! （已被 FD 隔离器 + 守恒先验验证至 1e-10）与 f32 下同点对拍：
//! 泛型代码路径由 f64 oracle 覆盖，f32 专属风险只剩"求值点舍入"，
//! 由对拍（1e-4 相对，f32 eps ≈ 1.19e-7 的 ~10³ 倍）单独检查。
//! 另含 f32 能量守恒先验（长时间尺度上 f32 舍入的实测漂移）与
//! f32 AD 全栈（Context + call_custom + backward）通路。

use ad_core::{Context, CustomOp, AD};
use ad_physics::{DoublePendulumStep, GyroscopicStep};

const TOL: f64 = 1e-4;

fn chain_grads_f32(op: &DoublePendulumStep, x: &[f32]) -> Vec<f32> {
    let mut ctx = Context::<f32>::new();
    let mut ads = Vec::new();
    let mut vars = Vec::new();
    for &v in x {
        let (ad, var) = ctx.var(v);
        ads.push(ad);
        vars.push(var);
    }
    let outs = ctx.call_custom(*op, &ads);
    // loss = Σ w·o + 0.2·o²（线性 + 二次，覆盖全部 4 个输出通道）
    let mut loss = {
        let lin = ctx.mul(AD::constant(0.3), outs[0]);
        let sq = ctx.mul(outs[0], outs[0]);
        let lq = ctx.mul(AD::constant(0.2), sq);
        ctx.add(lin, lq)
    };
    for (i, o) in outs.iter().enumerate().skip(1) {
        let lin = ctx.mul(AD::constant(0.3 + 0.11 * i as f32), *o);
        let sq = ctx.mul(*o, *o);
        let quad = ctx.mul(AD::constant(0.2), sq);
        let term = ctx.add(lin, quad);
        loss = ctx.add(loss, term);
    }
    ctx.backward(loss);
    vars.iter().map(|&v| ctx.grad(v).unwrap()).collect()
}

fn chain_grads_f64(op: &DoublePendulumStep, x: &[f64]) -> Vec<f64> {
    let mut ctx = Context::<f64>::new();
    let mut ads = Vec::new();
    let mut vars = Vec::new();
    for &v in x {
        let (ad, var) = ctx.var(v);
        ads.push(ad);
        vars.push(var);
    }
    let outs = ctx.call_custom(*op, &ads);
    let mut loss = {
        let lin = ctx.mul(AD::constant(0.3), outs[0]);
        let sq = ctx.mul(outs[0], outs[0]);
        let lq = ctx.mul(AD::constant(0.2), sq);
        ctx.add(lin, lq)
    };
    for (i, o) in outs.iter().enumerate().skip(1) {
        let lin = ctx.mul(AD::constant(0.3 + 0.11 * i as f64), *o);
        let sq = ctx.mul(*o, *o);
        let quad = ctx.mul(AD::constant(0.2), sq);
        let term = ctx.add(lin, quad);
        loss = ctx.add(loss, term);
    }
    ctx.backward(loss);
    vars.iter().map(|&v| ctx.grad(v).unwrap()).collect()
}

#[test]
fn f32_chain_matches_f64_reference() {
    let op = DoublePendulumStep::default();
    // 测试点全部为 f32 精确值（升采样无损，保证两路同一数学点）
    let points: [[f32; 7]; 3] = [
        [0.3125, -0.25, 0.125, 0.5, 0.25, -0.125, 0.001953125],
        [-0.5, 0.75, -1.0, 0.25, 0.125, 0.5, 0.001953125],
        [1.0, -1.5, 0.5, -0.75, -0.25, 0.375, 0.03125],
    ];
    for x in points {
        let g32 = chain_grads_f32(&op, &x);
        let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let g64 = chain_grads_f64(&op, &x64);
        for (k, (&a, &b)) in g32.iter().zip(g64.iter()).enumerate() {
            let rel = (a as f64 - b).abs() / (1.0 + b.abs());
            assert!(rel < TOL, "coord {k}: f32 {a:.7} vs f64 {b:.7} (rel {rel:.2e})");
        }
    }
}

#[test]
fn f32_gyro_matches_f64_reference() {
    let grad_f32 = |x: [f32; 7]| -> Vec<f32> {
        let mut ctx = Context::<f32>::new();
        let mut ads = Vec::new();
        let mut vars = Vec::new();
        for &v in &x {
            let (ad, var) = ctx.var(v);
            ads.push(ad);
            vars.push(var);
        }
        let outs = ctx.call_custom(GyroscopicStep, &ads);
        let mut loss = ctx.mul(outs[0], outs[0]);
        for o in &outs[1..] {
            let sq = ctx.mul(*o, *o);
            loss = ctx.add(loss, sq);
        }
        ctx.backward(loss);
        vars.iter().map(|&v| ctx.grad(v).unwrap()).collect()
    };
    let grad_f64 = |x: [f64; 7]| -> Vec<f64> {
        let mut ctx = Context::<f64>::new();
        let mut ads = Vec::new();
        let mut vars = Vec::new();
        for &v in &x {
            let (ad, var) = ctx.var(v);
            ads.push(ad);
            vars.push(var);
        }
        let outs = ctx.call_custom(GyroscopicStep, &ads);
        let mut loss = ctx.mul(outs[0], outs[0]);
        for o in &outs[1..] {
            let sq = ctx.mul(*o, *o);
            loss = ctx.add(loss, sq);
        }
        ctx.backward(loss);
        vars.iter().map(|&v| ctx.grad(v).unwrap()).collect()
    };
    let points: [[f32; 7]; 2] = [
        [0.5, -1.0, 0.25, 2.0, 3.0, 1.5, 0.01],
        [-0.75, 0.5, 1.0, 1.0, 2.0, 4.0, 0.005],
    ];
    for x in points {
        let g32 = grad_f32(x);
        let x64: [f64; 7] = x.map(|v| v as f64);
        let g64 = grad_f64(x64);
        for (k, (&a, &b)) in g32.iter().zip(g64.iter()).enumerate() {
            let rel = (a as f64 - b).abs() / (1.0 + b.abs());
            assert!(rel < TOL, "coord {k}: f32 {a:.7} vs f64 {b:.7} (rel {rel:.2e})");
        }
    }
}

// ============================================================ f32 能量守恒先验

fn energy_f32(op: &DoublePendulumStep, th1: f64, th2: f64, w1: f64, w2: f64) -> f64 {
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
fn f32_energy_conservation_passive_chain() {
    // 被动摆 2000 步（f32 前向累积）——f64 同条件漂移 0.38%；
    // f32 舍入随步数累积，容差放宽至 2%（实测值见断言前的输出）
    let op = DoublePendulumStep::default();
    let dt = 0.002f32;
    let mut st = [0.8f32, 0.6, 0.0, 0.0];
    let e0 = energy_f32(&op, st[0] as f64, st[1] as f64, st[2] as f64, st[3] as f64);

    for _ in 0..2000 {
        let outs = op.forward(&[st[0], st[1], st[2], st[3], 0.0, 0.0, dt]).0;
        st = [outs[0], outs[1], outs[2], outs[3]];
    }
    let e1 = energy_f32(&op, st[0] as f64, st[1] as f64, st[2] as f64, st[3] as f64);
    let drift = (e1 - e0).abs() / e0.abs().max(1.0);
    eprintln!("f32 能量守恒：E0 = {e0:.4}, E1 = {e1:.4}, drift = {drift:.2e}");
    assert!(drift < 0.02, "f32 能量漂移 {drift:.2e} 超过 2%");
}

// ============================================================ f32 直接验证（§12.3 第 40 条）

/// 泛型验证器在 f32 下直接跑接触与链算子——fd_step 1e-3 / 容差 5e-3
/// （f32 中心差分的 roundoff/truncation 平衡点）。
#[test]
fn validator_directly_validates_f32_ops() {
    use std::rc::Rc;

    let mut rng = ad_verify::Rng::new(11);
    let mut pts = |n: usize| -> Vec<Vec<f32>> {
        (0..3)
            .map(|_| (0..n).map(|_| 0.5f32 + rng.next_f64() as f32).collect())
            .collect()
    };
    for (name, op, np) in [
        (
            "chain_f32",
            Rc::new(DoublePendulumStep::default()) as Rc<dyn CustomOp<f32>>,
            7,
        ),
        ("gyro_f32", Rc::new(GyroscopicStep) as Rc<dyn CustomOp<f32>>, 7),
        (
            "contact_f32",
            Rc::new(ad_physics::ContactNormalOp) as Rc<dyn CustomOp<f32>>,
            6,
        ),
        (
            "friction_f32",
            Rc::new(ad_physics::RegularizedFrictionOp) as Rc<dyn CustomOp<f32>>,
            5,
        ),
    ] {
        let report = ad_verify::op_check::validate_custom_op(op, &pts(np), 1e-3, 5e-3);
        assert!(report.passed, "{name}:\n{report}");
    }
}

/// f32 摩擦锥性质（f64 版在 contact_fd.rs——同一性质在 f32 严格成立）
#[test]
fn f32_friction_cone_property() {
    let op = ad_physics::RegularizedFrictionOp;
    let cases: [(f32, f32, f32); 4] = [
        (3.0, 0.4, -0.7),
        (0.0, 2.0, 1.0),   // 无法向力 → 无摩擦
        (5.0, 0.0, 0.0),   // 无滑移 → 无摩擦
        (1.0, 100.0, 0.0), // 高速 → |ft| → μ·fn
    ];
    for (fn_, vx, vy) in cases {
        let (o, _) = CustomOp::<f32>::forward(&op, &[fn_, vx, vy, 0.6, 0.01]);
        let ft_mag = (o[0] * o[0] + o[1] * o[1]).sqrt();
        assert!(
            ft_mag <= 0.6 * fn_ + 1e-6,
            "cone violated (f32): {ft_mag} > {}",
            0.6 * fn_
        );
    }
}
