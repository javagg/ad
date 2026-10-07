//! 可插拔积分器族（HANDOFF §9.2 / design.md 第 46 条）。
//!
//! 二阶系统（q, q̇）的标准接口：加速度函数 `accel(ctx, q, q̇, u) -> q̈`
//! 全部在 tape 上求值（被追踪表达式），积分器只是表达式的组合方式——
//! 换积分器 = 换组合，梯度自动穿过。
//!
//! - [`SemiImplicitEuler`]：`q̇' = q̇ + dt·a`、`q' = q + dt·q̇'`（辛，
//!   LCP/Moreau 组合依赖其结构——接触冲量作用在速度上的离散分层）；
//! - [`Rk4`]：标准四级 Runge–Kutta（把 (q, q̇) 当一阶系统
//!   `ẋ = (q̇, a)`），局部误差 O(dt⁵)、全局 O(dt⁴)——iLQR 精度立涨，
//!   代价是每步 4 次加速度求值（tape ~4×）。
//!
//! 变分/辛高阶积分器：远期（§9.2）。

use ad_core::{Context, Scalar, AD};

/// 二阶系统加速度函数：`q̈ = f(q, q̇, u)`，全 AD 表达式。
pub type AccelFn<'a, S> = dyn Fn(&mut Context<S>, &[AD<S>], &[AD<S>], &[AD<S>]) -> Vec<AD<S>> + 'a;

/// 二阶系统积分器策略。
pub trait Integrator<S: Scalar> {
    /// 给定 (q, q̇, u) 推进一步，返回 (q', q̇')。
    fn step(
        &self,
        ctx: &mut Context<S>,
        accel: &AccelFn<'_, S>,
        q: &[AD<S>],
        qd: &[AD<S>],
        u: &[AD<S>],
        dt: S,
    ) -> (Vec<AD<S>>, Vec<AD<S>>);

    fn name(&self) -> &'static str;
}

/// 半隐式（辛）欧拉：先更新速度、再用**新**速度更新位置。
pub struct SemiImplicitEuler;

impl<S: Scalar> Integrator<S> for SemiImplicitEuler {
    fn step(
        &self,
        ctx: &mut Context<S>,
        accel: &AccelFn<'_, S>,
        q: &[AD<S>],
        qd: &[AD<S>],
        u: &[AD<S>],
        dt: S,
    ) -> (Vec<AD<S>>, Vec<AD<S>>) {
        let a = accel(ctx, q, qd, u);
        let mut qd_new = Vec::with_capacity(q.len());
        for (w, ai) in qd.iter().zip(&a) {
            let dw = ctx.mul(AD::constant(dt), *ai);
            qd_new.push(ctx.add(*w, dw));
        }
        let mut q_new = Vec::with_capacity(q.len());
        for (x, w) in q.iter().zip(&qd_new) {
            let dx = ctx.mul(AD::constant(dt), *w);
            q_new.push(ctx.add(*x, dx));
        }
        (q_new, qd_new)
    }

    fn name(&self) -> &'static str {
        "semi_implicit_euler"
    }
}

/// 经典四级 Runge–Kutta（一阶化 ẋ = (q̇, a(q, q̇, u))，控制零阶保持）。
pub struct Rk4;

impl<S: Scalar> Integrator<S> for Rk4 {
    fn step(
        &self,
        ctx: &mut Context<S>,
        accel: &AccelFn<'_, S>,
        q: &[AD<S>],
        qd: &[AD<S>],
        u: &[AD<S>],
        dt: S,
    ) -> (Vec<AD<S>>, Vec<AD<S>>) {
        let half = dt / (S::one() + S::one());
        let sixth = dt / (S::from(6.0).expect("6.0"));
        let n = q.len();

        // k1 = (q̇, a(q, q̇))
        let a1 = accel(ctx, q, qd, u);
        // k2 = (q̇ + dt/2·a1, a(q + dt/2·q̇, q̇ + dt/2·a1))
        let q_m2 = lin_comb(ctx, q, qd, half);
        let w_m2 = lin_comb(ctx, qd, &a1, half);
        let a2 = accel(ctx, &q_m2, &w_m2, u);
        // k3 = (q̇ + dt/2·a2, a(q + dt/2·w_m2, q̇ + dt/2·a2))
        let q_m3 = lin_comb(ctx, q, &w_m2, half);
        let w_m3 = lin_comb(ctx, qd, &a2, half);
        let a3 = accel(ctx, &q_m3, &w_m3, u);
        // k4 = (q̇ + dt·a3, a(q + dt·k3q, q̇ + dt·a3))；k3q = q̇ + dt/2·a2 = w_m3
        let q_m4 = lin_comb(ctx, q, &w_m3, dt);
        let w_m4 = lin_comb(ctx, qd, &a3, dt);
        let a4 = accel(ctx, &q_m4, &w_m4, u);

        // 位置分量的 k 项中 q̇ 系数和恰为 6 → 精确折叠：
        //   q' = q + dt·q̇ + dt²/6·(a1 + a2 + a3)
        //   q̇' = q̇ + dt/6·(a1 + 2a2 + 2a3 + a4)
        let mut q_new = Vec::with_capacity(n);
        let mut qd_new = Vec::with_capacity(n);
        let two = S::one() + S::one();
        let drift_c = dt * dt / (S::from(6.0).expect("6.0"));
        for i in 0..n {
            let dq = ctx.mul(AD::constant(dt), qd[i]);
            let q1 = ctx.add(q[i], dq);
            let d12 = ctx.add(a1[i], a2[i]);
            let drift = ctx.add(d12, a3[i]);
            let drift = ctx.mul(AD::constant(drift_c), drift);
            let qn = ctx.add(q1, drift);
            let s1 = ctx.mul(AD::constant(two), a2[i]);
            let s2 = ctx.mul(AD::constant(two), a3[i]);
            let sum = ctx.add(a1[i], s1);
            let sum = ctx.add(sum, s2);
            let sum = ctx.add(sum, a4[i]);
            let wsum = ctx.mul(AD::constant(sixth), sum);
            let wn = ctx.add(qd[i], wsum);
            q_new.push(qn);
            qd_new.push(wn);
        }
        (q_new, qd_new)
    }

    fn name(&self) -> &'static str {
        "rk4"
    }
}

/// `out = base + c·vec` 的逐分量 AD 组合（RK4 中间级的线性部分）
fn lin_comb<S: Scalar>(ctx: &mut Context<S>, base: &[AD<S>], vec: &[AD<S>], c: S) -> Vec<AD<S>> {
    let mut out = Vec::with_capacity(base.len());
    for (x, v) in base.iter().zip(vec) {
        let cv = ctx.mul(AD::constant(c), *v);
        out.push(ctx.add(*x, cv));
    }
    out
}
