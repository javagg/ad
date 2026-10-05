//! 陀螺力学验收（方向 3 深化：3D 自由刚体）：
//! 1. 单步逐坐标 FD 隔离器——手写 VJP 逐分量对拍；
//! 2. 动能守恒——Euler 陀螺方程不改变动能（先验）；
//! 3. 网球拍定理——绕中间主轴旋转的失稳翻转（物理标志性现象）；
//! 4. 角动量守恒——So3Exp 跟踪姿态，惯性系动量矩恒定（跨算子联用验证）。

use ad_core::{Context, CustomOp};
use ad_physics::spatial::so3_exp;
use ad_physics::GyroscopicStep;

fn op() -> GyroscopicStep {
    GyroscopicStep
}

#[test]
fn fd_gyroscopic_step() {
    let inputs = [0.4f64, 2.0, -0.7, 2.0, 1.0, 3.0, 0.01];
    let mut ctx = Context::<f64>::new();
    let mut vars = Vec::new();
    let mut ad_in = Vec::new();
    for &v in &inputs {
        let (ad, var) = ctx.var(v);
        ad_in.push(ad);
        vars.push(var);
    }
    let out = ctx.call_custom(op(), &ad_in);
    let mut l = ctx.mul(out[0], out[0]);
    for o in &out[1..] {
        let sq = ctx.mul(*o, *o);
        l = ctx.add(l, sq);
    }
    ctx.backward(l);

    let h = 1e-7;
    let loss_of = |o: &[f64]| -> f64 { o.iter().map(|v| v * v).sum() };
    let mut bad = 0;
    for j in 0..inputs.len() {
        let mut tp = inputs;
        tp[j] += h;
        let mut tm = inputs;
        tm[j] -= h;
        let fp = loss_of(&op().forward(&tp).0);
        let fm = loss_of(&op().forward(&tm).0);
        let fd = (fp - fm) / (2.0 * h);
        let g = ctx.grad(vars[j]).unwrap();
        if (g - fd).abs() > 1e-5 * (1.0 + g.abs() + fd.abs()) {
            eprintln!("input[{j}] ad {g:.10} vs fd {fd:.10}");
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad} mismatched coordinates");
}

/// 动能 = ½ωᵀIω（对角）
fn kinetic(w: [f64; 3], i: [f64; 3]) -> f64 {
    0.5 * (i[0] * w[0] * w[0] + i[1] * w[1] * w[1] + i[2] * w[2] * w[2])
}

#[test]
fn kinetic_energy_conserved() {
    // 无外力矩：Euler 方程保动能（动能 ≠ 功的来源，纯运动学耦合不产生/消耗）
    let op = op();
    let i_diag = [2.0f64, 1.0, 3.0];
    let mut w = [0.4f64, 3.0, 0.1]; // 靠近中间轴的高速旋转
    let dt = 2e-5;
    let ke0 = kinetic(w, i_diag);
    for _ in 0..100_000 {
        let (o, _) = op.forward(&[w[0], w[1], w[2], i_diag[0], i_diag[1], i_diag[2], dt]);
        w = [o[0], o[1], o[2]];
    }
    let ke1 = kinetic(w, i_diag);
    let drift = (ke1 - ke0).abs() / ke0;
    eprintln!(
        "KE {ke0:.6} -> {ke1:.6} (drift {drift:.2e}), |ω| = {:.3}",
        (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt()
    );
    assert!(drift < 1e-3, "kinetic energy drift {drift:.2e}");
}

/// 网球拍定理：绕中间主轴（I2）的旋转不稳定。初始 ω ≈ 3ω2 轴向 + 1e-3 扰动，
/// 不稳定性使 ω1 从 1e-3 指数增长到 O(1)（增长率 σ = ω2·√((I1−I2)(I2−I3)/(I1I3))），
/// 全程 KE 守恒。
#[test]
fn tennis_racket_intermediate_axis_flip() {
    let op = op();
    let i_diag = [3.0f64, 2.0, 1.0]; // I1 > I2 > I3（中间是 I2）
    let mut w = [0.0f64, 3.0, 1e-3];
    let dt = 1e-4;
    let ke0 = kinetic(w, i_diag);
    let mut max_w1 = 0.0f64;
    let mut w2_end = 0.0;

    for _ in 0..150_000 {
        let (o, _) = op.forward(&[w[0], w[1], w[2], i_diag[0], i_diag[1], i_diag[2], dt]);
        w = [o[0], o[1], o[2]];
        max_w1 = max_w1.max(w[0].abs());
        w2_end = w[1];
    }
    let ke1 = kinetic(w, i_diag);
    eprintln!(
        "tennis racket: max|ω1| = {max_w1:.4}（初始 1e-3），|ω2| 末值 = {w2_end:.3}，KE drift = {:.2e}",
        (ke1 - ke0).abs() / ke0
    );
    assert!(
        max_w1 > 0.5,
        "intermediate-axis instability expected, max|ω1| = {max_w1}"
    );
    assert!(
        (ke1 - ke0).abs() / ke0 < 1e-3,
        "KE must stay conserved through flips"
    );
}

/// 角动量守恒：L_world = R·(I·ω)，RK2 中点法耦合积分 (ω, R)（二阶），
/// 姿态指数映射用 [`so3_exp`]（So3Exp 算子的同源闭式）。
/// Euler 顶可积无混沌——漂移应为纯离散误差（小且有界）。
#[test]
fn angular_momentum_conserved_via_so3exp() {
    let op = op();
    let i_diag = [2.0f64, 1.0, 3.0];
    let i_mat = [
        [i_diag[0], 0.0, 0.0],
        [0.0, i_diag[1], 0.0],
        [0.0, 0.0, i_diag[2]],
    ];
    let mut w = [0.5f64, 2.0, -0.3];
    let mut r = so3_exp(&[0.3, -0.4, 0.2]);
    let dt = 2e-4;
    let momentum = |r: &[f64; 9], w: &[f64; 3]| -> [f64; 3] {
        let lb = [i_mat[0][0] * w[0], i_mat[1][1] * w[1], i_mat[2][2] * w[2]];
        [
            r[0] * lb[0] + r[1] * lb[1] + r[2] * lb[2],
            r[3] * lb[0] + r[4] * lb[1] + r[5] * lb[2],
            r[6] * lb[0] + r[7] * lb[1] + r[8] * lb[2],
        ]
    };
    let mul = |a: &[f64; 9], b: &[f64; 9]| -> [f64; 9] {
        let mut o = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    o[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
                }
            }
        }
        o
    };
    let a_of = |w: &[f64; 3]| -> [f64; 3] {
        let (o, _) = op.forward(&[w[0], w[1], w[2], i_diag[0], i_diag[1], i_diag[2], 1.0]);
        [(o[0] - w[0]), (o[1] - w[1]), (o[2] - w[2])]
    };
    let l0 = momentum(&r, &w);
    for _ in 0..10_000 {
        // RK2 中点：耦合 (ω, R)
        let a0 = a_of(&w);
        let w_mid = [
            w[0] + 0.5 * dt * a0[0],
            w[1] + 0.5 * dt * a0[1],
            w[2] + 0.5 * dt * a0[2],
        ];
        let r_mid = mul(
            &r,
            &so3_exp(&[0.5 * dt * w[0], 0.5 * dt * w[1], 0.5 * dt * w[2]]),
        );
        let a_mid = a_of(&w_mid);
        let w_n = [
            w[0] + dt * a_mid[0],
            w[1] + dt * a_mid[1],
            w[2] + dt * a_mid[2],
        ];
        let r_n = mul(
            &r_mid,
            &so3_exp(&[
                0.5 * dt * w_mid[0],
                0.5 * dt * w_mid[1],
                0.5 * dt * w_mid[2],
            ]),
        );
        w = w_n;
        r = r_n;
    }
    let l1 = momentum(&r, &w);
    let err = (0..3).map(|i| (l1[i] - l0[i]).abs()).fold(0.0f64, f64::max);
    eprintln!("L0 = {l0:?}, L1 = {l1:?}, max err = {err:.2e}");
    assert!(err < 5e-3, "angular momentum drift {err:.2e}");
}
