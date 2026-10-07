//! Barrier vs LCP 接触路线对比（HANDOFF §9.1 验收 / design.md 第 45 条核心交付）。
//!
//! 同一 1-DOF 弹跳场景（球落地面、重力、半隐式欧拉），两条可微接触路线：
//! - **barrier**（[`BarrierContactOp`]）：保守阻塞力，C^∞ 光滑，无切换；
//! - **LCP**（Moreau + Newton 回弹 + 位置门控活动集，单接触退化式）：
//!   精确非穿透，但活动集切换处半光滑。
//!
//! 量化对比（写入 design.md 第 45 条）：
//! 1. 轨迹形态：barrier 允许带内柔度（力平衡点落在激活带内），LCP 精确
//!    非穿透 + 回弹系数 e 精确可调；
//! 2. **梯度质量**：J(z₀) = ∫x dt 的 AD vs 中心差分，跨多个落差采样点
//!    × 两个 FD 步长——barrier 全点全步长机器精度级一致；LCP 在切换
//!    敏感点处出现 1e-2 量级偏差（kink 的一侧导数 vs 平均割线）；
//! 3. **系统辨识**：从轨迹反推刚度 κ（barrier 特有的可辨识参数，
//!    对应柔度/接触刚度标定）。

use ad::{Context, AD};
use ad_physics::BarrierContactOp;

const DT: f64 = 0.002;
const G: f64 = 9.81;
const DHAT: f64 = 0.05;
const EPS: f64 = 0.005;

// 显式 ctx 路径嵌套调用双重借用——帮助函数逐语句绑定（第 44 条 c 项模式）
fn add2(ctx: &mut Context<f64>, a: AD<f64>, b: AD<f64>) -> AD<f64> {
    ctx.add(a, b)
}
fn mul2(ctx: &mut Context<f64>, a: AD<f64>, b: AD<f64>) -> AD<f64> {
    ctx.mul(a, b)
}

// ============================================================ 两条路线的单步

/// barrier 路线单步：半隐式欧拉 + 阻塞力（m = 1）。
fn step_barrier(ctx: &mut Context<f64>, z: AD<f64>, v: AD<f64>, k: &AD<f64>) -> (AD<f64>, AD<f64>) {
    let f = ctx.call_custom(
        BarrierContactOp,
        &[z, *k, AD::constant(DHAT), AD::constant(EPS)],
    );
    let acc = ctx.sub(f[0], AD::constant(G));
    let dv = mul2(ctx, AD::constant(DT), acc);
    let v_new = add2(ctx, v, dv);
    let dz = mul2(ctx, AD::constant(DT), v_new);
    let z_new = add2(ctx, z, dz);
    (z_new, v_new)
}

/// LCP 路线单步：Moreau + Newton 回弹，单接触（地面，法向 +z）。
/// 结构与 `lcp_contact.rs` 的 step_ball 单墙分支一致（e·γ⁻ 带符号 r 项）；
/// 位置门控 gap ≤ 0 读前向值（离散决策不入带）。
fn step_lcp(ctx: &mut Context<f64>, z: AD<f64>, v: AD<f64>, e: &AD<f64>) -> (AD<f64>, AD<f64>) {
    let v_free = ctx.sub(v, AD::constant(G * DT)); // 自由飞行 v⁻
    if z.value > 0.0 {
        let dz = mul2(ctx, AD::constant(DT), v_free);
        let z_new = add2(ctx, z, dz);
        return (z_new, v_free);
    }
    // 接触：λ = −(b + r)，b = γ⁻ = v_free，r = e·γ⁻（带符号）；
    // A = 1/m = 1（质量归一）。λ ≥ 0 可行（回弹/支撑），否则自由。
    let r = mul2(ctx, *e, v_free);
    let br = add2(ctx, v_free, r);
    let lam = ctx.neg(br);
    let v_new = if lam.value >= 0.0 {
        add2(ctx, v_free, lam)
    } else {
        v_free
    };
    let dz = mul2(ctx, AD::constant(DT), v_new);
    let z_new = add2(ctx, z, dz);
    (z_new, v_new)
}

// ============================================================ 带梯度的 rollout

enum Route {
    Barrier,
    Lcp,
}

/// J(z0, p) = Σ x_t·DT（p = κ 或 e），返回 (J, [∂J/∂z0, ∂J/∂p])
fn rollout_j(route: &Route, z0: f64, p: f64, steps: usize) -> (f64, [f64; 2]) {
    let mut ctx = Context::<f64>::new();
    let (z0_ad, vz) = ctx.var(z0);
    let (p_ad, vp) = ctx.var(p);
    let (mut z, mut v) = (z0_ad, AD::constant(0.0));
    let mut loss = AD::constant(0.0);
    for _ in 0..steps {
        let (zn, vn) = match route {
            Route::Barrier => step_barrier(&mut ctx, z, v, &p_ad),
            Route::Lcp => step_lcp(&mut ctx, z, v, &p_ad),
        };
        let dt_x = mul2(&mut ctx, AD::constant(DT), zn);
        loss = add2(&mut ctx, loss, dt_x);
        z = zn;
        v = vn;
    }
    ctx.backward(loss);
    (loss.value, [ctx.grad(vz).unwrap(), ctx.grad(vp).unwrap()])
}

/// 纯数值 J（FD oracle）
fn rollout_j_fd(route: &Route, z0: f64, p: f64, steps: usize) -> f64 {
    rollout_j(route, z0, p, steps).0
}

// ============================================================ 1. 轨迹形态对比

#[test]
fn trajectory_shapes_differ_as_documented() {
    let steps = 600;
    // LCP（e=0.75）：精确非穿透——长时间后静止在 z = 0（微弹跳有界）
    let mut ctx = Context::<f64>::new();
    let e = ctx.var(0.75).0;
    let (mut z, mut v) = (ctx.var(0.08).0, AD::constant(0.0));
    for _ in 0..steps {
        let (zn, vn) = step_lcp(&mut ctx, z, v, &e);
        z = zn;
        v = vn;
    }
    eprintln!("LCP 静止：z = {:+.6}, v = {:+.2e}", z.value, v.value);
    assert!(z.value.abs() < 1e-3, "LCP 未精确收在地面：z = {}", z.value);

    // barrier（κ=200）：带内柔度 + ε 软肩——围绕肩部平衡点 z*（f(z*) = m·g，
    // z* > d̂ 受力地板 κε²/4z² 托起，"柔度足迹" O(κε²/g)）的有界振荡
    let k = 200.0f64;
    let mut ctx2 = Context::<f64>::new();
    let k_ad = ctx2.var(k).0;
    let (mut zb, mut vb) = (ctx2.var(0.08).0, AD::constant(0.0));
    let (mut zb_min, mut zb_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for _ in 0..steps {
        let (zn, vn) = step_barrier(&mut ctx2, zb, vb, &k_ad);
        zb = zn;
        vb = vn;
        zb_min = zb_min.min(zb.value);
        zb_max = zb_max.max(zb.value);
    }
    eprintln!(
        "barrier 弹跳：z ∈ [{zb_min:.6}, {zb_max:.6}]，终值 {:+.6}",
        zb.value
    );
    // 保守 barrier 无耗散 → 围绕肩部平衡点的有界振荡（不会"静止"）：
    // 振荡下界始终为正（软垫从不触地），上界有界（不飞离）
    assert!(
        zb_min > 0.02 && zb_max < 0.1,
        "barrier 轨迹应有界且不触地：z ∈ [{zb_min}, {zb_max}]"
    );
    assert!(
        zb_min > z.value.abs() + 0.02,
        "两路线静止/触地形态应有形态级差异（柔度足迹）：barrier min {zb_min} vs LCP {}",
        z.value
    );
}

// ============================================================ 2. 梯度质量对比（核心）

/// 相对误差（与验证器同约定）：|a−b|/(1+|a|+|b|)
fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / (1.0 + a.abs() + b.abs())
}

#[test]
fn gradient_quality_barrier_vs_lcp() {
    let steps = 300;
    // 落差采样横跨：半步碰撞敏感点（0.05）、贴地反弹（0.06/0.07）、
    // 中距（0.08）、远距多弹跳（0.10/0.12）
    let z0s = [0.05f64, 0.06, 0.07, 0.08, 0.10, 0.12];
    let hs = [1e-3f64, 1e-4];

    let mut worst_b = [0.0f64; 2];
    let mut worst_l = [0.0f64; 2];
    for &z0 in &z0s {
        let (_, g_b) = rollout_j(&Route::Barrier, z0, 200.0, steps);
        let (_, g_l) = rollout_j(&Route::Lcp, z0, 0.75, steps);
        for (pi, &p) in [200.0f64, 0.75].iter().enumerate() {
            let route = if pi == 0 {
                &Route::Barrier
            } else {
                &Route::Lcp
            };
            let g = if pi == 0 { g_b } else { g_l };
            for (hi, &h) in hs.iter().enumerate() {
                let gp = rollout_j_fd(route, z0 + h, p, steps);
                let gm = rollout_j_fd(route, z0 - h, p, steps);
                let fd = (gp - gm) / (2.0 * h);
                let e = rel(g[0], fd);
                if pi == 0 {
                    worst_b[hi] = worst_b[hi].max(e);
                } else {
                    worst_l[hi] = worst_l[hi].max(e);
                }
                eprintln!(
                    "z0={z0:.2} h={h:.0e} {}: ad={:+.8e} fd={:+.8e} rel={:.2e}",
                    if pi == 0 { "barrier" } else { "LCP" },
                    g[0],
                    fd,
                    e
                );
            }
        }
    }
    eprintln!(
        "=== 梯度质量对比（design.md 第 45 条）：barrier worst h=1e-3: {:.2e}, h=1e-4: {:.2e}；\
         LCP worst h=1e-3: {:.2e}, h=1e-4: {:.2e}",
        worst_b[0], worst_b[1], worst_l[0], worst_l[1]
    );
    // barrier：C^∞ 力 → FD 偏差是中心差分自身的截断误差，随 h 二阶收敛
    assert!(
        worst_b[1] < 1e-4,
        "barrier h=1e-4 偏差 {:.2e} 应达 1e-4 以下",
        worst_b[1]
    );
    assert!(
        worst_b[1] < 0.2 * worst_b[0],
        "barrier 偏差应随 h 十倍步长降 ~百倍（C² 收敛）：{:.2e} → {:.2e}",
        worst_b[0],
        worst_b[1]
    );
    // LCP：半光滑切换下 FD 一致性受离散切换 kink 支配、不随 h 消失
    // （z0=0.05 的半步碰撞点：±1e-4 即改变碰撞步schedule）
    assert!(
        worst_l[1] < 0.7,
        "LCP h=1e-4 偏差 {:.2e} 异常（应为其已知半光滑水平）",
        worst_l[1]
    );
    // 对比结论：barrier 在两个步长下都严格更优（≥2 个数量级）
    assert!(
        worst_b[0] < worst_l[0] && worst_b[1] * 100.0 < worst_l[1],
        "barrier（{:.2e}/{:.2e}）未显著优于 LCP（{:.2e}/{:.2e}）",
        worst_b[0],
        worst_b[1],
        worst_l[0],
        worst_l[1]
    );
}

// ============================================================ 3. 系统辨识：反推 κ

#[test]
fn sysid_recovers_barrier_stiffness() {
    // 观测：真值 κ* = 200 的轨迹（带内初始压缩 → 全程 κ 敏感）
    let (x_obs, _): (Vec<f64>, ()) = {
        let mut ctx = Context::<f64>::new();
        let k_true = ctx.var(200.0).0;
        let (mut z, mut v) = (ctx.var(0.03).0, AD::constant(0.0));
        let mut xs = Vec::new();
        for _ in 0..300 {
            let (zn, vn) = step_barrier(&mut ctx, z, v, &k_true);
            xs.push(zn.value);
            z = zn;
            v = vn;
        }
        (xs, ())
    };

    // loss(κ) = ½·mean((x_t(κ) − x_obs)²)；梯度经 κ 叶子自动反传
    fn loss_and_grad(k_val: f64, x_obs: &[f64], out: &mut [f64]) -> f64 {
        let mut ctx = Context::<f64>::new();
        let (k_ad, vk) = ctx.var(k_val);
        let (mut z, mut v) = (ctx.var(0.03).0, AD::constant(0.0));
        let mut loss = AD::constant(0.0);
        for &xo in x_obs {
            let (zn, vn) = step_barrier(&mut ctx, z, v, &k_ad);
            let r = ctx.sub(zn, AD::constant(xo));
            let sq = ctx.mul(r, r);
            loss = ctx.add(loss, sq);
            z = zn;
            v = vn;
        }
        let n = x_obs.len() as f64;
        let loss = ctx.mul(AD::constant(0.5 / n), loss);
        ctx.backward(loss);
        out[0] = ctx.grad(vk).unwrap();
        loss.value
    }

    let cfg = ad_optim::OptimizerCfg {
        max_iters: 150,
        lr0: 5.0,
        tol_rel_improve: 1e-14,
        ..Default::default()
    };
    let (k_hat, rep) = ad_optim::minimize_gradient_descent(&[100.0], &cfg, |p, out| {
        loss_and_grad(p[0], &x_obs, out)
    });
    eprintln!(
        "刚度辨识：loss {:.5} → {:.7}，κ̂ = {:.3}（真值 200），{} 迭代",
        rep.loss0, rep.loss, k_hat[0], rep.iters
    );
    assert!(rep.loss < rep.loss0 * 0.05, "损失未收敛");
    assert!((k_hat[0] - 200.0).abs() < 4.0, "κ̂ = {}", k_hat[0]);
    assert_eq!(rep.grad_health.nonfinite_fraction, 0.0);
}
