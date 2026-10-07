//! 可插拔积分器验证（HANDOFF §9.2 / design.md 第 46 条）：
//! 1. 收敛阶：谐振子（闭式解）上 Euler 全局误差 O(dt)、RK4 O(dt⁴)——
//!    步长减半误差比 ≈ 2 vs ≈ 16；
//! 2. 能量行为：无阻尼非线性摆长程积分，RK4 能量漂移 ≪ 半隐式欧拉
//!    的有界振荡（同为守恒形态，精度差数量级）。

use ad_core::{Context, AD};
use ad_physics::{AccelFn, Integrator, Rk4, SemiImplicitEuler};

/// 谐振子 q̈ = −ω²q
fn harmonic_accel(
    omega2: f64,
) -> impl Fn(&mut Context<f64>, &[AD<f64>], &[AD<f64>], &[AD<f64>]) -> Vec<AD<f64>> {
    move |ctx, q, _qd, _u| {
        let mut a = Vec::with_capacity(q.len());
        for &x in q {
            let s = ctx.mul(AD::constant(-omega2), x);
            a.push(s);
        }
        a
    }
}

/// 非线性单摆 q̈ = −sin(q)
fn pendulum_accel() -> impl Fn(&mut Context<f64>, &[AD<f64>], &[AD<f64>], &[AD<f64>]) -> Vec<AD<f64>>
{
    |ctx, q, _qd, _u| {
        q.iter()
            .map(|&x| {
                let s = ad_ops::sin_with(ctx, x);
                ctx.neg(s)
            })
            .collect()
    }
}

/// 从 (q0, v0) 积分 steps 步，返回末态 (q, w)
fn integrate_to(
    integ: &dyn Integrator<f64>,
    accel: &AccelFn<'_, f64>,
    q0: f64,
    v0: f64,
    dt: f64,
    steps: usize,
) -> (f64, f64) {
    let mut ctx = Context::<f64>::new();
    let (mut q, _) = ctx.var(q0);
    let (mut w, _) = ctx.var(v0);
    let u_empty: Vec<AD<f64>> = Vec::new();
    for _ in 0..steps {
        let (qn, wn) = integ.step(&mut ctx, accel, &[q], &[w], &u_empty, dt);
        q = qn[0];
        w = wn[0];
    }
    (q.value, w.value)
}

#[test]
fn convergence_orders_harmonic_oscillator() {
    let accel = harmonic_accel(1.0);
    let (q0, v0, t_total) = (1.0f64, 0.0f64, 1.0f64);
    let q_exact = q0 * t_total.cos() + v0 * t_total.sin();

    let err = |integ: &dyn Integrator<f64>, dt: f64| {
        let accel_dyn: &AccelFn<'_, f64> = &accel;
        let (q, _) = integrate_to(integ, accel_dyn, q0, v0, dt, (t_total / dt) as usize);
        (q - q_exact).abs()
    };

    // Euler：一阶 → 减半误差比 ≈ 2
    let (e1, e2) = (
        err(&SemiImplicitEuler, 0.01),
        err(&SemiImplicitEuler, 0.005),
    );
    let ratio_e = e1 / e2;
    eprintln!("Euler: err(0.01)={e1:.3e}, err(0.005)={e2:.3e}, ratio={ratio_e:.2}");
    assert!(
        (1.8..2.4).contains(&ratio_e),
        "Euler 误差比 {ratio_e} 应 ≈ 2"
    );

    // RK4：四阶 → 减半误差比 ≈ 16
    let (r1, r2) = (err(&Rk4, 0.05), err(&Rk4, 0.025));
    let ratio_r = r1 / r2;
    eprintln!("RK4: err(0.05)={r1:.3e}, err(0.025)={r2:.3e}, ratio={ratio_r:.2}");
    assert!(
        (12.0..20.0).contains(&ratio_r),
        "RK4 误差比 {ratio_r} 应 ≈ 16"
    );

    // 同步长下 RK4 精度碾压：dt=0.05 的 RK4 比 dt=0.005 的 Euler 还准
    let r_coarse = err(&Rk4, 0.05);
    let e_fine = err(&SemiImplicitEuler, 0.005);
    assert!(
        r_coarse < e_fine / 10.0,
        "RK4(dt=0.05)={r_coarse:.2e} 应远优于 Euler(dt=0.005)={e_fine:.2e}"
    );
}

#[test]
fn energy_behavior_pendulum_long_run() {
    let accel = pendulum_accel();
    let (q0, w0) = (1.2f64, 0.0);
    let energy = |q: f64, w: f64| 0.5 * w * w + 1.0 - q.cos();
    let e0 = energy(q0, w0);

    let max_drift = |integ: &dyn Integrator<f64>, dt: f64, steps: usize| {
        let accel_dyn: &AccelFn<'_, f64> = &accel;
        let mut ctx = Context::<f64>::new();
        let (mut q, _) = ctx.var(q0);
        let (mut w, _) = ctx.var(w0);
        let u_empty: Vec<AD<f64>> = Vec::new();
        let mut worst = 0.0f64;
        for step in 0..steps {
            let (qn, wn) = integ.step(&mut ctx, accel_dyn, &[q], &[w], &u_empty, dt);
            q = qn[0];
            w = wn[0];
            if step % 100 == 0 {
                worst = worst.max((energy(q.value, w.value) - e0).abs());
            }
        }
        worst
    };

    // 同一物理时长（20 s）：Euler 有界振荡 ~O(dt)，RK4 漂移 ~O(dt⁴)
    let (steps, dt) = (2000, 0.01);
    let d_e = max_drift(&SemiImplicitEuler, dt, steps);
    let d_r = max_drift(&Rk4, dt, steps);
    eprintln!(
        "能量漂移（20 s, dt={dt}）：Euler {d_e:.3e} vs RK4 {d_r:.3e}（比值 {:.0}×）",
        d_e / d_r
    );
    assert!(
        d_r < d_e / 20.0,
        "RK4 能量漂移 {d_r:.2e} 应远小于 Euler {d_e:.2e}"
    );
}
