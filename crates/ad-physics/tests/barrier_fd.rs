//! BarrierContactOp 验收（HANDOFF §9.1 / design.md 第 45 条）：
//! 1. 泛型验证器 `validate_custom_op` f64 + f32——手写 VJP 逐分量 FD 对拍
//!    （四种追踪形态 + 前向确定性 + gins 契约），点集覆盖穿透/零/带内/
//!    激活边界/分离五段（第 33 条公开验证器取代手写隔离器）；
//! 2. 物理先验：非粘着（f ≥ 0）、远处无感（gap ≥ d̂ 力为 0）、
//!    单调阻塞（gap 越小力越大）、静力平衡间隙 ~ m·g/f'（柔度可辨识）；
//! 3. **C² 光滑探针**——激活边界处二阶差分商连续性（clamp 式对照
//!    组必须被同一探针抓住）；
//! 4. 刚度 κ 扫描：弹跳球 rollout 上 AD vs FD + Taylor 余项 + 梯度健康度
//!    + 高 κ 最近逼近受抑（刚性换光滑的直接实证）。

use ad_core::{Context, CustomOp, AD};
use ad_physics::BarrierContactOp;
use ad_verify::GradientChecker;
use std::rc::Rc;

const OP: BarrierContactOp = BarrierContactOp;

// ============================================================ 泛型验证器（f64 + f32）

#[test]
fn validator_barrier_f64() {
    // 点集覆盖穿透/零/带内/激活边界/分离五段（κ=50、d̂=0.1、ε=1e-3）
    let pts: Vec<Vec<f64>> = vec![
        vec![-0.05, 50.0, 0.1, 1e-3],  // 深穿透：g̃ 的 softplus 尾部，非线性最强
        vec![0.0, 20.0, 0.05, 1e-3],   // gap=0：g̃ = ε/2，两段 softplus 同时活跃
        vec![0.03, 50.0, 0.1, 1e-3],   // 接近带内
        vec![0.0999, 50.0, 0.1, 1e-3], // 激活边界：max-clamp 式实现在此暴露非光滑
        vec![0.2, 50.0, 0.1, 1e-3],    // 分离态：力 ≈ 0 但尾部仍光滑可导
    ];
    let report = ad_verify::op_check::validate_custom_op(
        Rc::new(BarrierContactOp) as Rc<dyn CustomOp<f64>>,
        &pts,
        1e-6,
        1e-5,
    );
    assert!(report.passed, "barrier f64:\n{report}");
}

#[test]
fn validator_barrier_f32() {
    // f32 点集避开深穿透尾部（g̃ ~ ε²/(4|gap|) 区 FD 截断误差固有 >5%，
    // f64 已覆盖该区）：穿透点取 −0.001（g̃ ≈ ε/2 邻域，变化平缓），
    // ε 统一 0.02；点值为 f32 友好表示
    let pts: Vec<Vec<f32>> = vec![
        vec![-0.001, 50.0, 0.1, 0.02],
        vec![0.0, 20.0, 0.05, 0.02],
        vec![0.03125, 50.0, 0.1, 0.02],
        vec![0.25, 50.0, 0.1, 0.02],
    ];
    let report = ad_verify::op_check::validate_custom_op(
        Rc::new(BarrierContactOp) as Rc<dyn CustomOp<f32>>,
        &pts,
        1e-3,
        5e-3,
    );
    assert!(report.passed, "barrier f32:\n{report}");
}

// ============================================================ 物理先验

#[test]
fn barrier_physics_priors() {
    // 非粘着：f ≥ 0 对全 gap 域成立（穿透/零/带内/带外）
    for &(gap, k, dhat) in &[
        (-0.5f64, 50.0, 0.1),
        (-0.01, 50.0, 0.1),
        (0.0, 50.0, 0.1),
        (0.05, 50.0, 0.1),
        (0.5, 50.0, 0.1),
        (-0.02, 1e3, 0.05),
    ] {
        let (o, _) = OP.forward(&[gap, k, dhat, 1e-3]);
        assert!(
            o[0] >= 0.0 && o[0].is_finite(),
            "f({gap}, k={k}) = {}",
            o[0]
        );
    }
    // 远处无感：gap ≥ 2·d̂ 时力仅为 softplus 尾部水平（比带内低 5 个数量级）
    let (o_far, _) = OP.forward(&[0.2f64, 50.0, 0.1, 1e-3]);
    let (o_band, _) = OP.forward(&[0.03, 50.0, 0.1, 1e-3]);
    assert!(
        o_far[0] < 1e-4 * o_band[0],
        "far {} vs band {}",
        o_far[0],
        o_band[0]
    );
    // 单调阻塞：带内 gap 越小力越大
    let mut prev = 0.0f64;
    for gap in [0.09, 0.07, 0.05, 0.03, 0.01, -0.01] {
        let (o, _) = OP.forward(&[gap, 50.0, 0.1, 1e-3]);
        assert!(o[0] > prev, "非单调 @ gap={gap}: {prev} -> {}", o[0]);
        prev = o[0];
    }
    // log-barrier 极限：ε→小时带内力趋于 κ·d̂/gap − κ（1/gap 族）
    let (o_eps, _) = OP.forward(&[0.02f64, 50.0, 0.1, 1e-9]);
    let expect = 50.0f64 * 0.1 / 0.02 - 50.0;
    assert!(
        (o_eps[0] - expect).abs() < 1e-4 * expect,
        "log-barrier limit {} vs {expect}",
        o_eps[0]
    );
    // 静力平衡柔度：m·g 的负载下平衡间隙 δ 满足 f(δ) = m·g（f 在带内单调降，
    // 根唯一），且 δ 随 κ 增大向 d̂ 收缩（κ 可由静平衡位置辨识——第 45 条的
    // 系统辨识叙事；barrier 的"柔度"是平衡点落在激活带内，而非穿透）。
    // 用二分法（避免对 Newton 导数的手工微分再引入测试 bug）。
    let (m, g) = (1.0f64, 9.81);
    let mut eq_gaps = Vec::new();
    for k in [100.0f64, 1000.0] {
        let f_of = |gap: f64| OP.forward(&[gap, k, 0.1, 1e-6]).0[0];
        let (mut lo, mut hi) = (1e-9f64, 0.1); // f(lo)=+∞ > mg > 0 = f(hi)
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if f_of(mid) > m * g {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let gap_eq = 0.5 * (lo + hi);
        let err = (f_of(gap_eq) - m * g).abs();
        eprintln!("静平衡：κ={k:.0} → 平衡间隙 δ = {gap_eq:.6}（f 误差 {err:.1e}）");
        assert!(err < 1e-3, "k={k}: 平衡力误差 {err}");
        assert!(
            gap_eq > 0.0 && gap_eq < 0.1,
            "平衡点应落在激活带内：{gap_eq}"
        );
        eq_gaps.push(gap_eq);
    }
    assert!(
        eq_gaps[1] > eq_gaps[0],
        "κ 增大应使平衡间隙向 d̂ 收缩：{:?}",
        eq_gaps
    );
}

/// C² 光滑探针：激活边界 gap = d̂ 处二阶差分商 f''_fd 的连续性。
/// softplus 激活窗在此处过渡；max(0,·)-clamp 式 barrier 在此处 f'' 跳变
/// 100%（带内 = −f' > 0，带外 = 0）。h = 1e-3 ≪ ε = 0.01（过渡宽度），
/// 光滑函数的 f'' 在 2h 邻域内变化 ≪ clamp 跳变。附**对照组**：clamp
/// 式实现在同一探针下必须失败——证明探针有区分力（第 33 条"验证器
/// 必须能抓住破坏"的方法论）。
#[test]
fn barrier_c2_smoothness_probe() {
    let probe = |f: &dyn Fn(f64) -> f64| -> [f64; 3] {
        // f''_fd 于 d̂−h、d̂、d̂+h（h=1e-3，全部窗口落在同一过渡尺度内）
        let (x0, h) = (0.1f64, 1e-3);
        let d2 = |x: f64| (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
        [d2(x0 - h), d2(x0), d2(x0 + h)]
    };
    let smooth = |gap: f64| OP.forward(&[gap, 50.0, 0.1, 0.01]).0[0];
    // clamp 式对照：带内 κ(d̂−gap)/gap，带外 0（C¹ 但 f'' 在 d̂ 跳变）
    let clamped = |gap: f64| {
        if gap < 0.1 {
            50.0 * (0.1 - gap) / gap
        } else {
            0.0
        }
    };

    let s = probe(&smooth);
    let c = probe(&clamped);
    // 散度只比较边界两侧的内点估计（分母取二者尺度——跳变点自身的
    // 混叠值不进分母，否则 clamp 信号被自己压没）
    let spread = |v: &[f64; 3]| (v[0] - v[2]).abs() / (1e-9 + v[0].abs().max(v[2].abs()));
    let (s_spread, c_spread) = (spread(&s), spread(&c));
    eprintln!(
        "C² 探针 @ d̂：smooth f'' = [{:.3e}, {:.3e}, {:.3e}]（散度 {s_spread:.3e}）；\
         clamp f'' = [{:.3e}, {:.3e}, {:.3e}]（散度 {c_spread:.3e}）",
        s[0], s[1], s[2], c[0], c[1], c[2]
    );
    // 光滑实现：f'' 在边界两侧连续（相对散度小）
    assert!(s_spread < 0.35, "smooth f'' 散度 {s_spread} 过大——非 C²？");
    // 对照组必须被抓住（探针区分力的自证）：clamp 的内点估计一侧为 0
    assert!(
        c_spread > 0.5 && c_spread > 2.0 * s_spread,
        "clamp 对照组未被探针区分（散度 {c_spread}）——探针无区分力"
    );
}

// ============================================================ 刚度扫描（§4.2.4 模式）

/// 弹跳球（半隐式欧拉 + 重力）：BarrierContactOp 提供阻塞力 + 普通 ctx
/// 组合积分。**参数域耦合**（首跑实证）：softplus_ε 尾部力标尺 ≈ κε²/4z²，
/// ε=0.02、κ=500 时 z=2d̂ 处尾部力 ≈ 重力——球被"不可见的软肩"弹起、
/// 能量翻倍；ε 取 d̂/10 = 0.005 后尾部力 ≪ g，三个数量级的 κ 全部稳定
/// （ω·Δt = √(κ·d̂)/g̃·Δt ≤ 0.6）。刚性换光滑：κ 大 → 最近逼近间隙
/// g̃_min 越接近 d̂（越浅）。
mod bounce {
    use super::*;

    pub const DT: f64 = 0.002;
    pub const DHAT: f64 = 0.05;
    pub const EPS: f64 = 0.005;
    pub const G: f64 = 9.81;

    /// AD rollout：z0, v0, k 为叶子；gap = z（地面在 0，向上为正）。
    /// 返回 (loss, [∂L/∂z0, ∂L/∂v0, ∂L/∂k], 最近逼近间隙, 轨迹最高点)。
    pub fn rollout_ad(z0: f64, v0: f64, k: f64, t_total: usize) -> (f64, [f64; 3], f64, f64) {
        let mut ctx = Context::<f64>::new();
        let (mut z, vz) = ctx.var(z0);
        let (mut v, vv) = ctx.var(v0);
        let (k_ad, vk) = ctx.var(k);
        let dt = AD::constant(DT);
        let mut min_gap = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;
        for _ in 0..t_total {
            // gap = z；barrier 力沿 gap 增大方向推（+z）；重力 −g；m = 1
            let f = ctx.call_custom(
                BarrierContactOp,
                &[z, k_ad, AD::constant(DHAT), AD::constant(EPS)],
            );
            let acc = ctx.sub(f[0], AD::constant(G));
            let dv = ctx.mul(dt, acc);
            v = ctx.add(v, dv);
            let dz = ctx.mul(dt, v);
            z = ctx.add(z, dz);
            min_gap = min_gap.min(z.value);
            max_z = max_z.max(z.value);
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
            min_gap,
            max_z,
        )
    }

    /// 纯数值 rollout（FD oracle 用）——与 AD 路径逐项一致
    pub fn rollout_loss(z0: f64, v0: f64, k: f64, t_total: usize) -> f64 {
        let mut z = z0;
        let mut v = v0;
        for _ in 0..t_total {
            let (o, _) = BarrierContactOp.forward(&[z, k, DHAT, EPS]);
            v += DT * (o[0] - G);
            z += DT * v;
        }
        z * z
    }
}

#[test]
fn stiffness_sweep_barrier() {
    use bounce::*;
    let (z0, v0) = (0.06f64, -1.0);
    let t_total = 500;

    let mut min_gaps = Vec::new();
    for k in [5.0, 50.0, 500.0] {
        let loss_fd_fn = |p: &[f64]| rollout_loss(p[0], p[1], p[2], t_total);
        let params = [z0, v0, k];
        let (loss_ad, grads, min_gap, max_z) = rollout_ad(z0, v0, k, t_total);

        // AD vs 中心差分（C^∞ 力 → 高精度成立，这是与 LCP 路线对比的基线）
        let check = GradientChecker::default().check_scalar(loss_fd_fn, &params, &grads);
        assert!(
            check.passed,
            "k={k}: FD check failed, max_rel {:e}",
            check.max_rel_error
        );
        // eprintln 层面记录精度（供第 45 条引用）
        eprintln!("k={k:.0}: FD max_rel = {:e}", check.max_rel_error);

        // 健康度：有限值
        let health = GradientChecker::default().analyze_health(&grads);
        assert_eq!(health.nonfinite_fraction, 0.0, "k={k}: non-finite grads");

        // Taylor 余项：前向值与梯度自洽（C² 的直接数值证据）
        let report = GradientChecker::default().taylor_test(loss_fd_fn, &params, &grads, None);
        assert!(
            report.passed,
            "k={k}: taylor order {}",
            report.estimated_order
        );

        // 弹跳物理：重力 + 保守 barrier → 有界往复。理论上界
        // z0 + v0²/2g（总能量全转势能），辛欧拉多次弹跳能漂留 3e-2 余量；
        // 发散表现为 max_z 爆炸
        let z_energy = z0 + v0 * v0 / (2.0 * G);
        assert!(
            loss_ad.is_finite() && max_z < z_energy + 0.03,
            "k={k}: 轨迹发散 loss={loss_ad}, max_z={max_z} > {z_energy}"
        );
        // 刚性换光滑的直接实证：κ 越大最近逼近间隙越接近 d̂（越浅）
        eprintln!(
            "k={k:>6}: loss={loss_ad:.3e}, |∂L/∂k|={:.3e}, 最近逼近间隙={min_gap:.4} m",
            grads[2].abs()
        );
        min_gaps.push(min_gap);
    }
    assert!(
        min_gaps[2] > min_gaps[1] && min_gaps[1] > min_gaps[0],
        "κ 增大应使最近逼近变浅：{min_gaps:?}"
    );
}
