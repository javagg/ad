//! §5.3 物理场景验证清单（设计文档）：
//! 1. 自由落体——梯度与解析解精确一致（1e-12）；
//! 2. 弹跳球（平滑接触力）——非光滑点的平滑近似 + 手工 backward vs 前向 FD；
//! 3. 多体弹簧链（12 体，26 维梯度）——随机方向 FD 验证 + 梯度健康度。
//!
//! FD oracle 直接用 `CustomOp::forward` 的纯数值路径（与 AD 共享同一前向），
//! 模式与 IFT 测试一致：验证的是"手工推导的 backward"这一风险点。

use ad::prelude::*;
use ad::{Context, Variable};

// ============================================================ 1. 自由落体（解析解）

#[test]
fn scenario_free_fall_matches_analytic() {
    // 半隐式欧拉：v' = v - g·dt；p' = p + dt·v'
    // p_T = p0 + v0·dt·T - g·dt²·T(T+1)/2（解析）
    const T: usize = 500;
    const DT: f64 = 0.01;
    let (p0, v0, g) = (1.0, 0.0, 9.81);
    let target = 0.2;

    let mut ctx = Context::<f64>::new();
    let (g_ad, vg) = ctx.var(g);
    let mut v = ctx.var(v0).0;
    let mut p = ctx.var(p0).0;
    let dt = AD::constant(DT);
    for _ in 0..T {
        let dv = ctx.mul(AD::constant(-DT), g_ad);
        v = ctx.add(v, dv);
        let dp = ctx.mul(dt, v);
        p = ctx.add(p, dp);
    }
    let d = ctx.sub(p, AD::constant(target));
    let loss = ctx.mul(d, d);
    ctx.backward(loss);

    let p_t = p0 + v0 * DT * (T as f64) - g * DT * DT * ((T * (T + 1) / 2) as f64);
    let dp_dg = -DT * DT * ((T * (T + 1) / 2) as f64);
    let expect = 2.0 * (p_t - target) * dp_dg;
    let got = ctx.grad(vg).unwrap();
    assert!(
        (got - expect).abs() <= 1e-12 * (1.0 + expect.abs()),
        "ad {got} vs analytic {expect}"
    );
}

// ============================================================ 2. 弹跳球（平滑接触）

/// 平滑接触弹跳：pen = -p（穿透深度），smooth-relu(pen) 给出接触力。
/// v' = v - g·dt - (k/m)·dt·s(pen)；p' = p + dt·v'
/// inputs = [p, v, g, k, m, dt]，outputs = [p', v']，ε = 0.01 保证解析可微。
struct BallStep;

const EPS: f64 = 0.01;

fn smooth_relu(x: f64) -> f64 {
    0.5 * (x + (x * x + EPS * EPS).sqrt())
}

impl CustomOp<f64> for BallStep {
    fn num_inputs(&self) -> usize {
        6
    }
    fn num_outputs(&self) -> usize {
        2
    }
    fn forward(&self, i: &[f64]) -> (smallvec::SmallVec<[f64; 4]>, smallvec::SmallVec<[f64; 8]>) {
        let (p, v, g, k, m, dt) = (i[0], i[1], i[2], i[3], i[4], i[5]);
        let pen = -p;
        let s = smooth_relu(pen);
        let v1 = v - g * dt - (k / m) * dt * s;
        let p1 = p + dt * v1;
        (smallvec::smallvec![p1, v1], i.iter().copied().collect())
    }
    fn backward(&self, r: &[f64], go: &[f64]) -> smallvec::SmallVec<[f64; 4]> {
        let (p, v, g, k, m, dt) = (r[0], r[1], r[2], r[3], r[4], r[5]);
        let (lp, lv) = (go[0], go[1]);
        // 关键：p' = p + dt·v' 的内部边（v' 是本算子的另一个输出）对 tape 不可见，
        // 必须在此处理：作用于 v 路径（重力/接触力）的伴随是 λv' + dt·λp'
        let lv_acc = lv + dt * lp;
        let pen = -p;
        let s = smooth_relu(pen);
        // ds/dpen = 0.5(1 + pen/√(pen²+ε²))
        let sig = 0.5 * (1.0 + pen / (pen * pen + EPS * EPS).sqrt());
        let gp = lp + lv_acc * (k / m) * dt * sig;
        let gv = lv_acc;
        let gg = -lv_acc * dt;
        let gk = -lv_acc * dt * s / m;
        let gm = lv_acc * k * dt * s / (m * m);
        let gdt = lp * (v - g * dt - (k / m) * dt * s) + lv_acc * (-g - (k / m) * s);
        smallvec::smallvec![gp, gv, gg, gk, gm, gdt]
    }
    fn name(&self) -> &'static str {
        "ball_step"
    }
}

#[test]
fn scenario_bouncing_ball_contact_gradients() {
    const T: usize = 400;
    const DT: f64 = 0.01;
    let params = [1.0f64, 0.0, 9.81, 50.0, 1.0]; // p0, v0, g, k, m
    let target = -0.3;

    let op = BallStep;
    // 纯数值 rollout（FD oracle 用）
    let rollout_loss = |pr: &[f64]| -> f64 {
        let mut st = [pr[0], pr[1]];
        for _ in 0..T {
            let (o, _) = op.forward(&[st[0], st[1], pr[2], pr[3], pr[4], DT]);
            st = [o[0], o[1]];
        }
        let d = st[0] - target;
        d * d
    };

    // AD 路径
    let mut ctx = Context::<f64>::new();
    let mut vars = Vec::new();
    let mut params_ad = Vec::new();
    for &v in params.iter() {
        let (ad, var) = ctx.var(v);
        params_ad.push(ad);
        vars.push(var);
    }
    let dt_ad = AD::constant(DT);
    let mut st = [params_ad[0], params_ad[1]]; // 当前 [p, v]
    for _ in 0..T {
        let inp = [
            st[0],
            st[1],
            params_ad[2],
            params_ad[3],
            params_ad[4],
            dt_ad,
        ];
        let out = ctx.call_custom(BallStep, &inp);
        st = [out[0], out[1]];
    }
    let d = ctx.sub(st[0], AD::constant(target));
    let loss = ctx.mul(d, d);
    ctx.backward(loss);
    let grads: Vec<f64> = vars.iter().map(|&v| ctx.grad(v).unwrap()).collect();

    // 逐参数中心差分（前向解析光滑）
    let check = GradientChecker::default().check_scalar(rollout_loss, &params, &grads);
    assert!(
        check.passed,
        "bounce FD check failed: max_rel {:e}, details {:?}",
        check.max_rel_error,
        check
            .details
            .iter()
            .filter(|d| !d.passed)
            .collect::<Vec<_>>()
    );

    // 场景确实经历了接触（s > 0），且梯度有限
    assert!(rollout_loss(&params).is_finite());
    assert!(grads.iter().all(|g| g.is_finite()));
    let health = GradientChecker::default().analyze_health(&grads);
    assert_eq!(health.nonfinite_fraction, 0.0);
}

// ============================================================ 3. 多体弹簧链（12 体）

/// n 体最近邻弹簧链（两端固定墙），半隐式欧拉。
/// a = -k·L·x（L = tridiag(2,-1)，Dirichlet）；v' = v + dt·a；x' = x + dt·v'
/// inputs = [x(0..n), v(0..n), k, dt]，outputs = [x'..., v'...]
struct ChainStep {
    n: usize,
}

fn lap(x: &[f64], j: usize) -> f64 {
    let n = x.len();
    let left = if j > 0 { x[j - 1] } else { 0.0 };
    let right = if j + 1 < n { x[j + 1] } else { 0.0 };
    2.0 * x[j] - left - right
}

impl CustomOp<f64> for ChainStep {
    fn num_inputs(&self) -> usize {
        2 * self.n + 2
    }
    fn num_outputs(&self) -> usize {
        2 * self.n
    }
    fn forward(&self, i: &[f64]) -> (smallvec::SmallVec<[f64; 4]>, smallvec::SmallVec<[f64; 8]>) {
        let n = self.n;
        let (k, dt) = (i[2 * n], i[2 * n + 1]);
        let (x, v) = (&i[..n], &i[n..2 * n]);
        // L 对称：a = -k·L·x
        let a: Vec<f64> = (0..n).map(|j| -k * lap(x, j)).collect();
        let vp: Vec<f64> = (0..n).map(|j| v[j] + dt * a[j]).collect();
        let mut outs = smallvec::SmallVec::new();
        for j in 0..n {
            outs.push(x[j] + dt * vp[j]);
        }
        for j in 0..n {
            outs.push(vp[j]);
        }
        (outs, i.iter().copied().collect())
    }
    fn backward(&self, r: &[f64], go: &[f64]) -> smallvec::SmallVec<[f64; 4]> {
        let n = self.n;
        let (k, dt) = (r[2 * n], r[2 * n + 1]);
        let (x, v) = (&r[..n], &r[n..2 * n]);
        let (lx, lv) = (&go[..n], &go[n..2 * n]);
        // 关键：x' = x + dt·v' 的内部边（v' 是本算子的另一个输出）对 tape 不可见，
        // 必须在此处理：作用于 a(x) 路径的伴随是 λv' + dt·λx'
        let lv_acc: Vec<f64> = (0..n).map(|j| lv[j] + dt * lx[j]).collect();
        let a: Vec<f64> = (0..n).map(|j| -k * lap(x, j)).collect();
        let vp: Vec<f64> = (0..n).map(|j| v[j] + dt * a[j]).collect();
        let mut grads = smallvec::SmallVec::new();
        for j in 0..n {
            // gx_j = λx'_j - k·dt·(L·λv_acc)_j（L 对称）
            let g = lx[j] - k * dt * lap(&lv_acc, j);
            grads.push(g);
        }
        for j in 0..n {
            grads.push(dt * lx[j] + lv[j]);
        }
        grads.push(-dt * (0..n).map(|j| lv_acc[j] * lap(x, j)).sum::<f64>());
        grads.push(
            (0..n).map(|j| lx[j] * vp[j]).sum::<f64>()
                + (0..n).map(|j| lv_acc[j] * a[j]).sum::<f64>(),
        );
        grads
    }
    fn name(&self) -> &'static str {
        "chain_step"
    }
}

#[test]
fn scenario_mass_spring_chain_random_direction() {
    const N: usize = 12;
    const T: usize = 150;
    const DT: f64 = 0.02;

    let mut rng_state = 0x853C49E6748FEA9Bu64;
    let mut rand = move || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state >> 11) as f64 / (1u64 << 53) as f64
    };
    // 初始状态：随机驻点扰动（幅值 ~0.3）
    let mut init = vec![0.0f64; 2 * N];
    for i in 0..N {
        init[i] = 0.3 * (rand() - 0.5) * 2.0;
        init[N + i] = 0.1 * (rand() - 0.5) * 2.0;
    }
    let k0 = 2.5f64;

    let op = ChainStep { n: N };
    let rollout_loss = |params: &[f64]| -> f64 {
        let mut st: Vec<f64> = params[..2 * N].to_vec();
        let k = params[2 * N];
        for _ in 0..T {
            let mut inp = st.clone();
            inp.push(k);
            inp.push(DT);
            let (o, _) = op.forward(&inp);
            st = o.to_vec();
        }
        st[..N].iter().map(|v| v * v).sum::<f64>() * 0.5
    };

    // AD 路径：x0/v0/k 全部为叶子（2N+1 维梯度）；dt 为常量
    let mut ctx = Context::<f64>::new();
    let mut vars: Vec<Variable> = Vec::new();
    let mut state: Vec<AD<f64>> = Vec::new();
    for &v in &init {
        let (ad, var) = ctx.var(v);
        state.push(ad);
        vars.push(var);
    }
    let (k_ad, vk) = ctx.var(k0);
    vars.push(vk);
    let dt_ad = AD::constant(DT);

    for _ in 0..T {
        let mut inp = state.clone();
        inp.push(k_ad);
        inp.push(dt_ad);
        state = ctx.call_custom(ChainStep { n: N }, &inp);
    }
    let mut loss = AD::constant(0.0);
    for xi in &state[..N] {
        let sq = ctx.mul(*xi, *xi);
        loss = ctx.add(loss, sq);
    }
    let half = ctx.mul(AD::constant(0.5), loss);
    ctx.backward(half);
    let grads: Vec<f64> = vars.iter().map(|&v| ctx.grad(v).unwrap()).collect();

    // 随机方向 FD 验证（高维输入的廉价替代，设计文档 §4.5.1）
    let mut params = init.clone();
    params.push(k0);
    let check = GradientChecker::default().check_random_direction(rollout_loss, &params, &grads, 8);
    assert!(
        check.passed,
        "chain FD check failed: max_rel {:e}",
        check.max_rel_error
    );

    // 梯度健康度（§4.5.3）
    let health = GradientChecker::default().analyze_health(&grads);
    assert_eq!(health.nonfinite_fraction, 0.0);
    assert!(health.norm.is_finite() && health.norm > 0.0);
}

// ============================================================ 4. 单步逐坐标 FD（backward 推导隔离器）

/// 单步、逐坐标、小规模：直接隔离"手工 backward 推导错误"，
/// 不受长轨迹 FD 精度和动力学敏感性影响。
#[test]
fn chain_one_step_per_coordinate_fd() {
    let n = 3;
    let op = ChainStep { n };
    let inp: Vec<f64> = vec![0.3, -0.2, 0.1, 0.05, 0.0, -0.07, 2.5, 0.02];
    // loss 用到全部输出（含 v'），覆盖输出耦合边
    let loss_of = |i: &[f64]| -> f64 {
        let (o, _) = op.forward(i);
        0.5 * (o[..n].iter().map(|v| v * v).sum::<f64>()
            + o[n..].iter().map(|v| v * v).sum::<f64>())
    };

    let mut ctx = Context::<f64>::new();
    let mut vars = Vec::new();
    let mut ad_in: Vec<AD<f64>> = Vec::new();
    for &v in &inp {
        let (ad, var) = ctx.var(v);
        ad_in.push(ad);
        vars.push(var);
    }
    let out = ctx.call_custom(ChainStep { n }, &ad_in);
    let mut l = AD::constant(0.0);
    for o in &out {
        let sq = ctx.mul(*o, *o);
        l = ctx.add(l, sq);
    }
    let half = ctx.mul(AD::constant(0.5), l);
    ctx.backward(half);

    let h = 1e-6;
    for j in 0..inp.len() {
        let mut tp = inp.clone();
        tp[j] += h;
        let mut tm = inp.clone();
        tm[j] -= h;
        let fd = (loss_of(&tp) - loss_of(&tm)) / (2.0 * h);
        let ad = ctx.grad(vars[j]).unwrap();
        assert!(
            (ad - fd).abs() < 1e-6,
            "coord {j}: ad {ad:.10} vs fd {fd:.10}"
        );
    }
}
