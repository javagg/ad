//! 接触模型验收（设计文档 §5.2 / §4.3.1）：
//! 1. 单步逐坐标 FD 隔离器——两个接触算子的手写 VJP 逐分量对拍；
//! 2. 物理先验：非粘着（f ≥ 0）、接近耗散（f·gap_vel ≤ 0）、摩擦锥（|ft| ≤ μ·fn）；
//! 3. 刚度扫描：弹跳球 rollout 上 AD vs FD + Taylor 余项 + 梯度健康度。

use ad_core::{Context, CustomOp, AD};
use ad_physics::{ContactNormalOp, RegularizedFrictionOp};
use ad_verify::GradientChecker;

/// 平滑混合全部输出的标量损失（确定性），与 AD 路径的 loss 表达式逐项一致
fn loss_of_out(out: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, &o) in out.iter().enumerate() {
        s += (0.3 + 0.11 * i as f64) * o + 0.2 * o * o;
    }
    // 交叉项对 len==1 也成立（out[0]·out[0]）——单输出算子的 AD loss 含此项，
    // FD oracle 必须一致，否则 λf 差一个因子
    s += 0.15 * out[0] * out[out.len() - 1];
    s
}

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
    let mut l = {
        let lin = ctx.mul(AD::constant(0.3), out[0]);
        let sq = ctx.mul(out[0], out[0]);
        let quad = ctx.mul(AD::constant(0.2), sq);
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
        let g = ctx.grad(vars[j]).unwrap();
        if (g - fd).abs() > tol * (1.0 + g.abs() + fd.abs()) {
            eprintln!("{name}: input[{j}] ad {g:.10} vs fd {fd:.10}");
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{name}: {bad} mismatched coordinates");
}

#[test]
fn fd_contact_normal() {
    // gap < 0（穿透）、k、p ∈ [1,2]、d、ε
    check_op(
        "contact_normal",
        ContactNormalOp,
        &[-0.05, -0.8, 50.0, 1.5, 0.3, 0.01],
        1e-5,
    );
    check_op(
        "contact_normal",
        ContactNormalOp,
        &[-0.003, 0.5, 50.0, 1.0, 0.2, 0.01],
        1e-5,
    );
    // 分离态（gap > 0）：力 ≈ 0 但 softplus 尾部仍光滑可导
    check_op(
        "contact_normal",
        ContactNormalOp,
        &[0.02, -0.4, 50.0, 1.5, 0.3, 0.01],
        1e-5,
    );
}

#[test]
fn fd_regularized_friction() {
    check_op(
        "regularized_friction",
        RegularizedFrictionOp,
        &[3.0, 0.4, -0.7, 0.6, 0.01],
        1e-5,
    );
    // 滑移速度 ≪ ε（正则化线性区）
    check_op(
        "regularized_friction",
        RegularizedFrictionOp,
        &[3.0, 1e-4, -2e-4, 0.6, 0.01],
        1e-5,
    );
}

#[test]
fn contact_normal_physics_priors() {
    let op = ContactNormalOp;
    // 非粘着：k, d ≥ 0 且 |d·gap_vel| < k 时 f ≥ 0
    for &(gap, gv) in &[(-0.05f64, -0.8), (-0.001, -3.0), (0.02, 0.0), (0.0, -1.0)] {
        let (o, _) = op.forward(&[gap, gv, 50.0, 1.5, 0.3, 0.01]);
        assert!(
            o[0] >= 0.0,
            "adhesive force at gap={gap}, gv={gv}: {}",
            o[0]
        );
    }
    // 接近耗散：gap_vel < 0 → 接触功率 f·gap_vel ≤ 0
    for &(gap, gv) in &[(-0.05f64, -0.8), (-0.001, -3.0), (-0.01, -0.1)] {
        let (o, _) = op.forward(&[gap, gv, 50.0, 1.5, 0.3, 0.01]);
        assert!(o[0] * gv <= 0.0, "non-dissipative at gap={gap}, gv={gv}");
    }
    // 分离时力趋于 0（softplus 尾部 ~ k·(ε²/4·gap)^1.5 ≈ 1.8e-5 @ gap=0.5）
    let (o, _) = op.forward(&[0.5, 0.0, 50.0, 1.5, 0.0, 0.01]);
    assert!(o[0] < 1e-4, "separated force {}", o[0]);
}

#[test]
fn friction_cone_property() {
    let op = RegularizedFrictionOp;
    for &(fn_, vx, vy) in &[
        (3.0f64, 0.4, -0.7),
        (0.0, 2.0, 1.0),   // 无法向力 → 无摩擦
        (5.0, 0.0, 0.0),   // 无滑移 → 无摩擦
        (1.0, 100.0, 0.0), // 高速 → |ft| → μ·fn
    ] {
        let (o, _) = op.forward(&[fn_, vx, vy, 0.6f64, 0.01]);
        let ft_mag = (o[0] * o[0] + o[1] * o[1]).sqrt();
        assert!(
            ft_mag <= 0.6 * fn_ + 1e-12,
            "cone violated: {ft_mag} > {}",
            0.6 * fn_
        );
    }
    // 高速极限 ≈ μ·fn
    // 高速极限 ≈ μ·fn（vt 沿 x → ft_x 承担全部摩擦）
    let (o, _) = op.forward(&[1.0f64, 100.0, 0.0, 0.6, 0.01]);
    assert!(
        (o[0].abs() - 0.6).abs() < 1e-3,
        "coulomb limit {}",
        o[0].abs()
    );
}

// （ stiffness 扫描报告见测试输出：损失与梯度范数随刚度 decades 的变化）/
// ---- 刚度扫描：梯度健康度随刚度数量的行为（设计文档 §4.2.4/§5.4） ----

/// 弹跳球（半隐式欧拉）：ContactNormalOp 提供力 + 普通 ctx 组合积分。
/// FD oracle 用 ContactNormal.forward 的纯数值路径。
mod bounce {
    use super::*;
    use ad_physics::ContactNormalOp;

    pub const DT: f64 = 0.002;
    pub const P: f64 = 1.5;
    pub const D: f64 = 0.05;
    pub const EPS: f64 = 1e-4;

    /// AD rollout：z0, v0, k 为叶子；gap = z（地面在 0，向上为正）。
    /// 返回 (loss, [∂L/∂z0, ∂L/∂v0, ∂L/∂k], tape 记录数)。
    pub fn rollout_ad(z0: f64, v0: f64, k: f64, t_total: usize) -> (f64, [f64; 3], usize) {
        let mut ctx = Context::<f64>::new();
        let (mut z, vz) = ctx.var(z0);
        let (mut v, vv) = ctx.var(v0);
        let (k_ad, vk) = ctx.var(k);
        let dt = AD::constant(DT);
        let one = AD::constant(1.0);
        for _ in 0..t_total {
            // gap = z；gap_vel = v
            let f = ctx.call_custom(
                ContactNormalOp,
                &[
                    z,
                    v,
                    k_ad,
                    AD::constant(P),
                    AD::constant(D),
                    AD::constant(EPS),
                ],
            );
            // 加速度 = +f/m（力沿 gap 增大方向推）
            let acc = ctx.div(f[0], one);
            let dv = ctx.mul(dt, acc);
            v = ctx.add(v, dv);
            let dz = ctx.mul(dt, v);
            z = ctx.add(z, dz);
        }
        let loss = ctx.mul(z, z); // (z_T)²
        ctx.backward(loss);
        (
            loss.value,
            [
                ctx.grad(vz).unwrap(),
                ctx.grad(vv).unwrap(),
                ctx.grad(vk).unwrap(),
            ],
            ctx.tape_len(),
        )
    }

    /// 纯数值 rollout（FD oracle 用）
    pub fn rollout_loss(z0: f64, v0: f64, k: f64, t_total: usize) -> f64 {
        let mut z = z0;
        let mut v = v0;
        for _ in 0..t_total {
            let (o, _) = ContactNormalOp.forward(&[z, v, k, P, D, EPS]);
            v += DT * o[0];
            z += DT * v;
        }
        z * z
    }
}

#[test]
fn stiffness_sweep_gradient_health() {
    use bounce::*;
    let (z0, v0) = (0.05f64, -1.0);
    let t_total = 400;

    for k in [10.0, 100.0, 1000.0] {
        let loss_fd_fn = |p: &[f64]| rollout_loss(p[0], p[1], p[2], t_total);
        let (loss_ad, grads, tape) = rollout_ad(z0, v0, k, t_total);
        let _ = tape;
        let params = [z0, v0, k];

        // AD vs 中心差分（平滑接触 → 高精度成立）
        let check = GradientChecker::default().check_scalar(loss_fd_fn, &params, &grads);
        assert!(
            check.passed,
            "k={k}: FD check failed, max_rel {:e}",
            check.max_rel_error
        );

        // 健康度：有限值
        let health = GradientChecker::default().analyze_health(&grads);
        assert_eq!(health.nonfinite_fraction, 0.0, "k={k}: non-finite grads");

        // Taylor 余项：前向值与梯度自洽
        let report = GradientChecker::default().taylor_test(loss_fd_fn, &params, &grads, None);
        assert!(
            report.passed,
            "k={k}: taylor order {}",
            report.estimated_order
        );

        eprintln!(
            "k={k:>7}: loss={loss_ad:.3e}, ‖∂L/∂(z0,v0,k)‖={:.3e}, |∂L/∂k|={:.3e}",
            grads.iter().take(2).map(|g| g * g).sum::<f64>().sqrt(),
            grads[2].abs()
        );
    }
}
