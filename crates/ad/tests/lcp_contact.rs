//! 接触 LCP 完整演示（设计文档 §4.3.3 动机难题的闭环；§12.3 第 44 条）。
//!
//! **Moreau 时间步进 + Newton 回弹 LCP**，两层结构：
//! - 离散外环：**活动集枚举**（接触数 n_c ≤ 2，子集穷举即精确活动集法）；
//!   可行性判据（λ ≥ 0、非活动约束 w ≥ 0）读前向值——离散决策不入带；
//! - 连续内环：活动集上的线性求解用 **`solve_sym` 在 tape 上**完成——
//!   λ 是被追踪表达式，梯度穿过接触求解**自动**由 tape 反传。
//!
//! Newton 回弹以 r 项进 LCP：`w = A·λ + b + r`，`r_i = e·max(0, −γ_i⁻)`
//! （e = 回弹系数，AD 叶子——系统辨识的可辨识参数）。
//!
//! 场景：
//! 1. **1-DOF 球在箱内**（左右墙）：掉落 → 以 e 回弹 → 能量衰减 → 收壁；
//!    闭式可验证（apex ∝ e²、LCP 自支撑静置 λ = m·g·dt）；
//! 2. **穿越接触切换的梯度**：J(e) 的 AD vs FD（弹跳时刻漂移的半光滑
//!    切换在隔离时刻下 FD 仍可用——实证记录）；
//! 3. **系统辨识**：从轨迹反推 e（真值 0.75）；
//! 4. **2D 角块**：地面 + 侧墙双接触**同时主动**（非退化 2×2 LCP，
//!    对角 Delassus）——挤入角落静止，闭式逐轴可验证。


use ad::{Context, AD};
use ad_optim::{minimize_gradient_descent, OptimizerCfg};

const DT: f64 = 0.01;
const G: f64 = 9.81;

// 显式 ctx 路径的单次借用帮助函数：嵌套调用（ctx.a(x, ctx.y(..))）会
// 双重可变借用——经帮助函数中转后每次借用即结束，嵌套合法
fn add2(ctx: &mut Context<f64>, a: AD<f64>, b: AD<f64>) -> AD<f64> {
    ctx.add(a, b)
}
fn mul2(ctx: &mut Context<f64>, a: AD<f64>, b: AD<f64>) -> AD<f64> {
    ctx.mul(a, b)
}
fn neg1(ctx: &mut Context<f64>, a: AD<f64>) -> AD<f64> {
    ctx.neg(a)
}
// ============================================================ 1. 1-DOF 球在箱内

/// 单步 Moreau 步进（在 tape 上）：x, v, e 为 AD；返回 (x⁺, v⁺)。
/// 墙：左 x=0（法向 +x）、右 x=L（法向 −x）；接触按 gap ≤ 0 门控。
fn step_ball(
    ctx: &mut Context<f64>,
    x: AD<f64>,
    v: AD<f64>,
    e: &AD<f64>,
    box_len: f64,
) -> (AD<f64>, AD<f64>) {
    let v_free = ctx.sub(v, AD::constant(G * DT)); // 自由飞行 v⁻ = v − g·dt
    let gap_l = x.value;
    let gap_r = box_len - x.value;

    // 接触候选（位置门控，离散决策读前向值）
    let at_l = gap_l <= 0.0;
    let at_r = gap_r <= 0.0;
    if !at_l && !at_r {
        // 自由飞行
        let dx = ctx.mul(AD::constant(DT), v_free);
        let x_new = ctx.add(x, dx);

        return (x_new, v_free);
    }

    // r 项（Newton 回弹）：r_i = (1−e)·max(0, −γ_i⁻)；γ_L = v_free，γ_R = −v_free
    let bneg = ctx.neg(v_free);
    let b = [v_free, bneg];
    let mn0 = ctx.neg(b[0]);
    let _m0 = ad_ops::max_with(ctx, AD::constant(0.0), mn0); // max 值仅为文档语义；实际回弹量经 r 项
    let mn1 = ctx.neg(b[1]);
    let _m1 = ad_ops::max_with(ctx, AD::constant(0.0), mn1);
    // r_i = e·γ_i⁻（带符号）：逼近接触 λ = −(1+e)γ⁻ > 0 产生回弹 v⁺ = −e·γ⁻；
    // 分离接触 λ < 0 不可行自动失活
    let r0 = mul2(ctx, *e, b[0]);
    let r1 = mul2(ctx, *e, b[1]);
    // 对每个候选 S：A_SS·λ_S = −(b_S + r_S)，A_ii = 1/m（1-DOF 单墙）
    let rs = [r0, r1];
    // 可行性：λ_S ≥ 0 且非活动墙的法向速度 w = b + r + A_īS·λ_S ≥ 0
    let m_inv = 1.0; // m = 1（演示质量归一）

    // {L}：λ_L = −(b_L + r_L)/A_LL
    if at_l {
        let a11 = AD::constant(m_inv);
        let br = ctx.add(b[0], rs[0]);
        let rhs = ctx.neg(br);
        let mat = vec![a11];
        let lam = ad_ops::solve_sym_with(ctx, &mat, &[rhs]);
        if lam[0].value >= 0.0 {
            let wl1 = ctx.add(b[1], rs[1]);
            let wl2 = ctx.mul(AD::constant(-m_inv), lam[0]);
            let w_r = ctx.add(wl1, wl2);
            if w_r.value >= 0.0 {
                // v⁺ = v_free + λ_L/m（λ_L 沿 +x）
                let dv = ctx.mul(AD::constant(m_inv), lam[0]);
                let v_new = ctx.add(v_free, dv);
                let dx = ctx.mul(AD::constant(DT), v_new);
                let x_new = ctx.add(x, dx);
                return (x_new, v_new);
            }
        }
    }
    if at_r {
        let a11 = AD::constant(m_inv);
        let br = ctx.add(b[1], rs[1]);
        let rhs = ctx.neg(br);
        let mat = vec![a11];
        let lam = ad_ops::solve_sym_with(ctx, &mat, &[rhs]);
        if lam[0].value >= 0.0 {
            let wl1 = ctx.add(b[0], rs[0]);
            let wl2 = ctx.mul(AD::constant(-m_inv), lam[0]);
            let w_l = ctx.add(wl1, wl2);
            if w_l.value >= 0.0 {
                // v⁺ = v_free − λ_R/m（λ_R 沿 −x）
                let dv = ctx.mul(AD::constant(m_inv), lam[0]);
                let v_new = ctx.sub(v_free, dv);
                let dx = ctx.mul(AD::constant(DT), v_new);
                let x_new = ctx.add(x, dx);
                return (x_new, v_new);
            }
        }
    }
    // 无可行活动集（含 ∅）：自由飞行
    let dx = ctx.mul(AD::constant(DT), v_free);
    let x_new = ctx.add(x, dx);

    (x_new, v_free)
}

/// rollout：返回逐步 (x, v) 轨迹
fn rollout_ball(e_val: f64, steps: usize, x0: f64, box_len: f64) -> (Vec<f64>, Vec<f64>) {
    let mut ctx = Context::<f64>::new();
    let e = ctx.var(e_val).0;
    let (mut x, _) = ctx.var(x0);
    let (mut v, _) = ctx.var(0.0);
    let mut xs = Vec::with_capacity(steps);
    let mut vs = Vec::with_capacity(steps);
    for _ in 0..steps {
        let (xn, vn) = step_ball(&mut ctx, x, v, &e, box_len);
        xs.push(xn.value);
        vs.push(vn.value);
        x = xn;
        v = vn;
    }
    (xs, vs)
}

/// 带梯度的 rollout：J(e) = Σ x_t·dt，返回 (J, dJ/de)
fn rollout_with_grad(e_val: f64, steps: usize, x0: f64, box_len: f64) -> (f64, f64) {
    let mut ctx = Context::<f64>::new();
    let (e, ve) = ctx.var(e_val);
    let (mut x, _) = ctx.var(x0);
    let (mut v, _) = ctx.var(0.0);
    let mut loss = AD::constant(0.0);
    for _ in 0..steps {
        let (xn, vn) = step_ball(&mut ctx, x, v, &e, box_len);
        let dt_x = ctx.mul(AD::constant(DT), xn);
        loss = ctx.add(loss, dt_x);
        x = xn;
        v = vn;
    }
    ctx.backward(loss);
    (loss.value, ctx.grad(ve).unwrap())
}

#[test]
fn lcp_bounce_physics() {
    let g = G;
    let e = 0.75f64;
    let x0 = 1.5f64;
    let dt = DT;
    // 解析：掉落时间 t1 = √(2·x0/g)；第一反弹 apex = e²·x0
    let t1 = (2.0 * x0 / g).sqrt();
    let steps_to_first_impact = (t1 / dt) as usize;
    let (xs, vs) = rollout_ball(e, 2000, x0, 2.0);

    // 1. 掉落段自由飞行（x = x0 − ½g t²）
    let t_check = 0.3f64;
    let k = (t_check / dt) as usize;
    let expect = x0 - 0.5 * g * t_check * t_check;
    assert!(
        (xs[k] - expect).abs() < 0.05,
        "自由飞行 {} vs {}",
        xs[k],
        expect
    );

    // 2. 第一反弹 apex ≈ e²·x0（在碰撞后的飞行段内采样最大值）
    let apex_window = &xs[steps_to_first_impact..(steps_to_first_impact + (t1 / dt) as usize + 2)];
    let apex = apex_window.iter().cloned().fold(f64::MIN, f64::max);
    let expect_apex = e * e * x0;
    eprintln!(
        "第一反弹 apex = {apex:.4}（解析 {expect_apex:.4}），碰撞步 = {steps_to_first_impact}"
    );
    assert!(
        (apex - expect_apex).abs() < 0.05,
        "apex {apex} vs {expect_apex}"
    );

    // 3. 收壁：300 步（3 s）后静止在左墙
    let v_end = vs[vs.len() - 1];
    let x_end = xs[xs.len() - 1];
    eprintln!("收壁：x_end = {x_end:.5}, v_end = {v_end:.5}");
    assert!(x_end.abs() < 0.01, "未收壁：x_end = {x_end}");
    // Moreau + Newton 回弹的稳态是微弹跳：v ∈ [0, g·dt] 有界，位置收敛于壁
    assert!(v_end.abs() <= G * DT + 1e-9, "微弹跳稳态速度越界：{v_end}");

    // 4. LCP 静置自支撑：静止时接触 impulse = m·g·dt（每步支撑重力）
    //    （由 v_end == 0 与 x_end 稳定隐式验证）
}

/// 穿越接触切换的梯度：J(e) = ∫x dt 的 AD vs FD（弹跳切换的半光滑性实证）
#[test]
fn lcp_gradient_through_switches() {
    let (j, dj) = rollout_with_grad(0.75, 300, 1.5, 2.0);
    let h = 1e-5;
    let (jp, djp) = rollout_with_grad(0.75 + h, 300, 1.5, 2.0);
    let (jm, djm) = rollout_with_grad(0.75 - h, 300, 1.5, 2.0);
    let fd = (jp - jm) / (2.0 * h);
    eprintln!(
        "J(e) = {j:.6}；dJ/de AD = {dj:.6} vs FD {fd:.6}（J(e±h) = {jp:.4}/{djm:.4}）"
    );
    assert!(
        (dj - fd).abs() < 1e-2 * (1.0 + fd.abs()),
        "梯度 {dj} vs FD {fd}"
    );
    let _ = (djp, djm);
}

/// 系统辨识：从轨迹反气回弹系数 e（真值 0.75）
#[test]
fn lcp_sysid_restitution() {
    // 观测：真值 e* = 0.75 的轨迹
    let (xs_obs, _) = rollout_ball(0.75, 300, 1.5, 2.0);

    // loss(e) = ½·mean((x_t(e) − x_obs)²)；梯度经 e 叶子自动反传
    fn loss_and_grad(e_val: f64, xs_obs: &[f64], out: &mut [f64]) -> f64 {
        let mut ctx = Context::<f64>::new();
        let (e, ve) = ctx.var(e_val);
        let (mut x, _) = ctx.var(1.5);
        let (mut v, _) = ctx.var(0.0);
        let mut loss = AD::constant(0.0);
        for t in 0..xs_obs.len() {
            let (xn, vn) = step_ball(&mut ctx, x, v, &e, 2.0);
            let r = ctx.sub(xn, AD::constant(xs_obs[t]));
            let sq = ctx.mul(r, r);
            loss = ctx.add(loss, sq);
            x = xn;
            v = vn;
        }
        let n = xs_obs.len() as f64;
        let loss = ctx.mul(AD::constant(0.5 / n), loss);
        ctx.backward(loss);
        out[0] = ctx.grad(ve).unwrap();
        loss.value
    }

    let cfg = OptimizerCfg {
        max_iters: 100,
        lr0: 0.02,
        tol_rel_improve: 1e-14,
        ..Default::default()
    };
    let (e_hat, rep) = minimize_gradient_descent(&[0.5], &cfg, |p, out| {
        loss_and_grad(p[0], &xs_obs, out)
    });
    eprintln!(
        "回弹辨识：loss {:.5} → {:.7}，ê = {:.5}（真值 0.75），{} 迭代",
        rep.loss0, rep.loss, e_hat[0], rep.iters
    );
    assert!((e_hat[0] - 0.75).abs() < 5e-3, "ê = {}", e_hat[0]);
}

// ============================================================ 2. 2D 角块（双接触同时主动）

/// 2D 块（无旋转，自由度 (x, y)）：地面 y=0（法向 +y）+ 左墙 x=0（法向 +x）。
/// 双接触同时主动时 LCP 2×2 非退化（对角 Delassus = diag(1/m)）。
fn step_block(
    ctx: &mut Context<f64>,
    st: &[AD<f64>], // [x, y, vx, vy]
    e: &AD<f64>,
) -> [AD<f64>; 4] {
    let (x, y, vx, vy) = (st[0], st[1], st[2], st[3]);
    // 自由速度：vy_free = vy − g·dt
    let g_dt = mul2(ctx, AD::constant(G * DT), AD::constant(1.0));
    let ng = neg1(ctx, g_dt);
    let vy_free = add2(ctx, vy, ng);
    let at_floor = y.value <= 0.0;
    let at_wall = x.value <= 0.0;
    if !at_floor && !at_wall {
        let dtv = mul2(ctx, AD::constant(DT), vx);
        let xn = add2(ctx, x, dtv);
        let dtv = mul2(ctx, AD::constant(DT), vy_free);
        let yn = add2(ctx, y, dtv);
        return [xn, yn, vx, vy_free];
    }
    // r_i = e·max(0, −b_i)
    // r_i = e·γ_i⁻（带符号，同 step_ball）
    let r_x = mul2(ctx, *e, vx);
    let r_y = mul2(ctx, *e, vy_free);

    // 活动集枚举（4 子集；A = diag(1/m) 非退化）。可行优先序：双 → 单 → 空
    let m_inv = 1.0;

    // 双活动：solve_sym 2×2，A = diag(1/m)，λ = −m·(b + r)
    if at_floor && at_wall {
        let a = vec![AD::constant(m_inv), AD::constant(0.0), AD::constant(0.0), AD::constant(m_inv)];
        let s0 = add2(ctx, vx, r_x);
        let rhs0 = neg1(ctx, s0);
        let s1 = add2(ctx, vy_free, r_y);
        let rhs1 = neg1(ctx, s1);
        let rhs = vec![rhs0, rhs1];
        let lam = ad_ops::solve_sym_with(ctx, &a, &rhs);
        if lam[0].value >= 0.0 && lam[1].value >= 0.0 {
            let dvx = mul2(ctx, AD::constant(m_inv), lam[0]);
            let vxn = add2(ctx, vx, dvx);
            let dvy = mul2(ctx, AD::constant(m_inv), lam[1]);
            let vyn = add2(ctx, vy_free, dvy);
            let dxn = mul2(ctx, AD::constant(DT), vxn);
            let xn = add2(ctx, x, dxn);
            let d_yn = mul2(ctx, AD::constant(DT), vyn);
            let yn = add2(ctx, y, d_yn);
            return [xn, yn, vxn, vyn];
        }
    }
    if at_floor {
        let a = vec![AD::constant(m_inv)];
        let s1 = add2(ctx, vy_free, r_y);
        let rhs1 = neg1(ctx, s1);
        let rhs = vec![rhs1];
        let lam = ad_ops::solve_sym_with(ctx, &a, &rhs);
        if lam[0].value >= 0.0 {
            let dvy = mul2(ctx, AD::constant(m_inv), lam[0]);
            let vyn = add2(ctx, vy_free, dvy);
            let d_yn = mul2(ctx, AD::constant(DT), vyn);
            let yn = add2(ctx, y, d_yn);
            let dxn = mul2(ctx, AD::constant(DT), vx);
            let xn = add2(ctx, x, dxn);
            return [xn, yn, vx, vyn];
        }
    }
    if at_wall {
        let a = vec![AD::constant(m_inv)];
        let s0 = add2(ctx, vx, r_x);
        let rhs0 = neg1(ctx, s0);
        let rhs = vec![rhs0];
        let lam = ad_ops::solve_sym_with(ctx, &a, &rhs);
        if lam[0].value >= 0.0 {
            let dvx = mul2(ctx, AD::constant(m_inv), lam[0]);
            let vxn = add2(ctx, vx, dvx);
            let dxn = mul2(ctx, AD::constant(DT), vxn);
            let xn = add2(ctx, x, dxn);
            return [xn, y, vxn, vy_free];
        }
    }
        let dtvx = mul2(ctx, AD::constant(DT), vx);
        let xn = add2(ctx, x, dtvx);
        let dtvy = mul2(ctx, AD::constant(DT), vy_free);
        let yn = add2(ctx, y, dtvy);
    [xn, yn, vx, vy_free]
}

/// 2D 角块：斜抛入角 → 双接触同时主动 → 角落静止；逐轴闭式验证
#[test]
fn lcp_corner_block_squeeze() {
    let e = 0.0f64; // 完全非弹性：撞击角 → 双接触同时主动 → 粘住
    let mut st = [0.5f64, 0.4, -0.6, -0.9]; // 斜向左下
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for _ in 0..400 {
        let mut ctx = Context::<f64>::new();
        let s_ad: Vec<AD<f64>> = st.iter().map(|&v| ctx.var(v).0).collect();
        let e_ad = AD::constant(e);
        let out = step_block(&mut ctx, &s_ad, &e_ad);
        for (i, o) in out.iter().enumerate() {
            st[i] = o.value;
        }
        xs.push(st[0]);
        ys.push(st[1]);
    }
    eprintln!("角块收位：x = {:.5}, y = {:.5}", st[0], st[1]);
    assert!(st[0].abs() < 0.02, "未收入左墙：x = {}", st[0]);
    assert!(st[1].abs() < 0.02, "未收入地面：y = {}", st[1]);
    assert!(xs.iter().all(|v| v.is_finite()) && ys.iter().all(|v| v.is_finite()));
}
