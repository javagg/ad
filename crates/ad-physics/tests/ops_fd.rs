//! 单步逐坐标 FD 隔离器（设计文档 §4.3.1 契约的标准验证流程）：
//! 每个空间代数算子的手写 VJP 逐分量对照
//! `loss(op.forward(x ± h))` 的中心差分。
//! loss 刻意混合全部输出分量（线性 + 二次 + 交叉项），
//! 确保 VJP 的每个输出通道都被覆盖。

use ad_core::{Context, CustomOp, AD};
use std::rc::Rc;

use ad_physics::spatial;
use ad_physics::{chain::DoublePendulumStep, ContactNormalOp, RegularizedFrictionOp, GyroscopicStep, SpatialForceCross};
use ad_physics::{
    InertiaApply, PluckerForce, PluckerMotion, RotateInertia, So3Exp, SpatialCrossMotion,
};

/// 平滑混合全部输出的标量损失（固定伪随机权重，确定性）。
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
fn check_op<O>(name: &str, op: O, inputs: &[f64], tol: f64)
where
    O: CustomOp<f64> + Copy + 'static,
{
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
fn fd_spatial_cross_motion() {
    let x = [
        0.4, -1.2, 0.7, 0.3, -0.5, 2.0, -0.6, 0.9, 0.1, 1.5, 0.2, -0.8,
    ];
    check_op("spatial_cross_motion", SpatialCrossMotion, &x, 1e-5);
}

#[test]
fn fd_plucker_motion() {
    let e = spatial::so3_exp(&[0.3, -0.8, 0.5]);
    let x = [
        e.as_slice(),
        &[0.4, -0.3, 0.9],
        &[0.7, 0.2, -1.1, 0.5, 0.3, -0.4],
    ]
    .concat();
    check_op("plucker_motion", PluckerMotion, &x, 1e-5);
}

#[test]
fn fd_plucker_force() {
    let e = spatial::so3_exp(&[-0.5, 0.4, 1.1]);
    let x = [
        e.as_slice(),
        &[0.2, 0.8, -0.4],
        &[1.0, -0.3, 0.6, 0.2, -0.9, 0.4],
    ]
    .concat();
    check_op("plucker_force", PluckerForce, &x, 1e-5);
}

#[test]
fn fd_inertia_apply() {
    let x = [
        2.0, 3.0, 1.5, 0.1, -0.2, 0.15, // Ī（6，正定）
        1.2,  // m
        0.3, -0.2, 0.5, // c
        0.8, -0.4, 1.1, 0.2, -0.6, -0.3, // v
    ];
    check_op("inertia_apply", InertiaApply, &x, 1e-5);
}

#[test]
fn fd_rotate_inertia() {
    let e = spatial::so3_exp(&[0.4, 0.6, -0.9]);
    let x = [
        2.0, 3.0, 1.5, 0.1, -0.2, 0.15, // Ī_B（6，正定）
        1.2,  // m
        0.3, -0.2, 0.5, // c
        e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], // E
        0.4, -0.3, 0.9, // r
    ];
    check_op("rotate_inertia", RotateInertia, &x, 1e-5);
}

#[test]
fn fd_so3_exp() {
    check_op("so3_exp", So3Exp, &[0.3, -0.8, 0.5], 1e-5);
    check_op("so3_exp", So3Exp, &[-1.0, 0.2, 0.7], 1e-5);
}

#[test]
fn so3_exp_forward_properties() {
    // RᵀR = I，det R = 1
    for w in [[0.3, -0.8, 0.5], [1.0, 1.0, 1.0], [-0.1, 0.0, 0.0]] {
        let w: [f64; 3] = w;
        let r = spatial::so3_exp(&w);
        let rt = spatial::mat3_t(&r);
        let rtr = spatial::mat3_mul(&rt, &r);
        for (i, j) in [(0usize, 0usize), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)] {
            let expect = if i == j { 1.0 } else { 0.0 };
            assert!(
                (rtr[i * 3 + j] - expect).abs() < 1e-12,
                "RᵀR[{i},{j}] = {}",
                rtr[i * 3 + j]
            );
        }
        let det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6])
            + r[2] * (r[3] * r[7] - r[4] * r[6]);
        assert!((det - 1.0).abs() < 1e-12);
    }
}

#[test]
fn so3_exp_small_angle_continuity() {
    // 主分支与小角度分支的梯度应连续（差 O(θ)）
    let grads = |w: [f64; 3]| -> [f64; 3] {
        let mut ctx = Context::<f64>::new();
        let (x, vx) = ctx.var(w[0]);
        let (y, vy) = ctx.var(w[1]);
        let (z, vz) = ctx.var(w[2]);
        let out = ctx.call_custom(So3Exp, &[x, y, z]);
        let mut l = ctx.mul(out[0], out[0]);
        for o in &out[1..] {
            let sq = ctx.mul(*o, *o);
            l = ctx.add(l, sq);
        }
        ctx.backward(l);
        [
            ctx.grad(vx).unwrap(),
            ctx.grad(vy).unwrap(),
            ctx.grad(vz).unwrap(),
        ]
    };
    let small = grads([1e-6, -2e-6, 5e-7]);
    let mid = grads([2e-5, -4e-5, 1e-5]);
    for i in 0..3 {
        assert!(
            (small[i] - mid[i]).abs() < 1e-3,
            "branch continuity: {small:?} vs {mid:?}"
        );
    }
}

// ============================================================ 公开验证器（§12.3 第 33 条）
//
// `ad_verify::op_check::validate_custom_op` 一行调用：前向确定性 + VJP 契约 +
// 四种追踪形态逐坐标 FD 对拍——库自己的空间代数算子必须全数通过。

use ad_verify::op_check::validate_custom_op;

fn expect_pass(name: &str, op: Rc<dyn CustomOp<f64>>, points: &[Vec<f64>]) {
    let report = validate_custom_op(op, points, 1e-6, 1e-5);
    assert!(report.passed, "{name}: {}", report);
}

#[test]
fn validator_passes_all_library_ops() {
    let e = spatial::so3_exp(&[0.3, -0.8, 0.5]);
    let pts = |n: usize| -> Vec<Vec<f64>> {
        let mut rng = ad_verify::Rng::new(42);
        (0..2)
            .map(|_| (0..n).map(|_| 0.4 + 0.5 * rng.next_f64() - 0.2).collect())
            .collect()
    };
    expect_pass("spatial_cross_motion", Rc::new(SpatialCrossMotion), &pts(12));
    expect_pass("inertia_apply", Rc::new(InertiaApply), &pts(16));
    expect_pass("so3_exp", Rc::new(So3Exp), &pts(3));
    expect_pass("gyroscopic_step", Rc::new(GyroscopicStep), &pts(7));
    expect_pass("spatial_force_cross", Rc::new(SpatialForceCross), &pts(12));
    expect_pass("contact_normal", Rc::new(ContactNormalOp), &pts(6));
    expect_pass("regularized_friction", Rc::new(RegularizedFrictionOp), &pts(5));
    expect_pass(
        "double_pendulum_step",
        Rc::new(DoublePendulumStep::default()),
        &pts(7),
    );
    // PluckerMotion / PluckerForce / RotateInertia 的输入含旋转矩阵（定义域受限），
    // 自动点会落在非正交矩阵上——传入合法点集
    expect_pass(
        "plucker_motion",
        Rc::new(PluckerMotion),
        &[[
            e.as_slice(),
            &[0.4, -0.3, 0.9],
            &[0.7, 0.2, -1.1, 0.5, 0.3, -0.4],
        ]
        .concat()],
    );
}
