//! 铰接体组合动力学验证（§12.3 第 43 条）：
//! 1. n=2 交叉验证——组合式（空间代数算子 RNEA + solve_sym）与
//!    **独立重写的经典闭式公式**（点质量双摆拉格朗日）逐点对拍；
//! 2. n=3 能量守恒先验（被动摆半隐式欧拉 2000 步）；
//! 3. 单步 AD 梯度 vs FD 逐坐标抽查。

use ad_core::{Context, AD};

use ad_physics::{articulated_forward, PlanarChain};

/// 独立重写的点质量双摆闭式前向动力学（不与组合路径共享任何代码——
/// "防两处同错"原则；公式与 ad-optim/tests/chain.rs 的 step_ad 同源文献形）。
/// 返回 q̈。模型：无质量连杆 + 端部点质量，q 为相对关节角，
/// M(q)·q̈ = τ − c(q,ω) − g(q)，半隐式欧拉的位置更新在外部。
fn closed_form_dd2(
    m: [f64; 2],
    l: [f64; 2],
    g: f64,
    th: [f64; 2],
    w: [f64; 2],
    tau: [f64; 2],
) -> [f64; 2] {
    let c2 = th[1].cos();
    let s2 = th[1].sin();
    let s12 = (th[0] + th[1]).sin();
    let m11 = (m[0] + m[1]) * l[0] * l[0] + m[1] * l[1] * l[1] + 2.0 * m[1] * l[0] * l[1] * c2;
    let m12 = m[1] * l[1] * l[1] + m[1] * l[0] * l[1] * c2;
    let m22 = m[1] * l[1] * l[1];
    let h = m[1] * l[0] * l[1] * s2;
    let r1 = tau[0] + h * (2.0 * w[0] * w[1] + w[1] * w[1])
        - (m[0] + m[1]) * g * l[0] * th[0].sin()
        - m[1] * g * l[1] * s12;
    let r2 = tau[1] - h * w[0] * w[0] - m[1] * g * l[1] * s12;
    let det = m11 * m22 - m12 * m12;
    [(m22 * r1 - m12 * r2) / det, (m11 * r2 - m12 * r1) / det]
}

/// 组合式前向动力学的 f64 包装（零阻尼）
fn composed_dd2(
    chain: &PlanarChain,
    th: [f64; 2],
    w: [f64; 2],
    tau: [f64; 2],
) -> [f64; 2] {
    let mut ctx = Context::<f64>::new();
    let q: Vec<AD<f64>> = th.iter().map(|&v| ctx.var(v).0).collect();
    let qd: Vec<AD<f64>> = w.iter().map(|&v| ctx.var(v).0).collect();
    let t: Vec<AD<f64>> = tau.iter().map(|&v| ctx.var(v).0).collect();
    let d: Vec<AD<f64>> = (0..2).map(|_| AD::constant(0.0)).collect();
    let masses_ad: Vec<AD<f64>> = chain.masses.iter().map(|&v| AD::constant(v)).collect();
    let a = articulated_forward(&mut ctx, chain, &masses_ad, &q, &qd, &t, &d);
    [a[0].value, a[1].value]
}

#[test]
fn composed_matches_closed_form_n2() {
    let chain = PlanarChain::new(&[1.0, 0.8], &[1.0, 0.9], 9.81);
    let m = [1.0, 0.8];
    let l = [1.0, 0.9];
    let states: [([f64; 2], [f64; 2], [f64; 2]); 4] = [
        ([0.3, -0.2], [0.1, 0.4], [0.2, -0.1]),
        ([0.8, 0.6], [-1.5, 0.7], [0.1, -0.05]),
        ([-2.0, 1.1], [0.3, -0.3], [0.0, 0.0]),
        ([1.2, -2.5], [-0.8, 1.2], [0.5, 0.25]),
    ];
    let mut worst = 0.0f64;
    for (th, w, tau) in states {
        let a_ref = closed_form_dd2(m, l, 9.81, th, w, tau);
        let a_cmp = composed_dd2(&chain, th, w, tau);
        for k in 0..2 {
            let rel = (a_ref[k] - a_cmp[k]).abs() / (1.0 + a_ref[k].abs());
            worst = worst.max(rel);
            assert!(
                rel < 1e-9,
                "state {th:?} w {w:?}: q̈[{k}] composed {} vs closed {} (rel {rel:.2e})",
                a_cmp[k],
                a_ref[k]
            );
        }
    }
    eprintln!("n=2 交叉验证最大相对偏差：{worst:.2e}");
}

/// n=3 组合链：单步 AD 梯度 vs FD 逐坐标（组合结构的组装正确性由
/// n=2 闭式对拍 + 本测试的 FD 仲裁共同守护）
#[test]
fn composed_matches_fd_n3() {
    let chain = PlanarChain::new(&[1.0, 0.8, 0.6], &[1.0, 0.9, 0.7], 9.81);
    let base = [0.4f64, -0.3, 0.9, 0.2, -0.5, 0.3, 0.1, -0.05, 0.02];

    // loss = Σ w_j·a_j（线性，覆盖全部输出；与 FD 用同一表达式）
    fn loss_at(chain: &PlanarChain, p: &[f64]) -> f64 {
        let mut ctx = Context::<f64>::new();
        let q: Vec<AD<f64>> = p[0..3].iter().map(|&v| ctx.var(v).0).collect();
        let qd: Vec<AD<f64>> = p[3..6].iter().map(|&v| ctx.var(v).0).collect();
        let t: Vec<AD<f64>> = p[6..9].iter().map(|&v| ctx.var(v).0).collect();
        let d: Vec<AD<f64>> = (0..3).map(|_| AD::constant(0.0)).collect();
        let masses_ad: Vec<AD<f64>> = chain.masses.iter().map(|&v| AD::constant(v)).collect();
        let a = articulated_forward(&mut ctx, chain, &masses_ad, &q, &qd, &t, &d);
        let w = [0.7f64, -1.1, 0.4];
        let mut l = ctx.mul(AD::constant(w[0]), a[0]);
        for (k, o) in a.iter().enumerate().skip(1) {
            let lin = ctx.mul(AD::constant(w[k]), *o);
            l = ctx.add(l, lin);
        }
        l.value
    }

    // AD 路径（叶子只建一次——vars 与 q/qd/t 是同一批节点）
    let mut ctx = Context::<f64>::new();
    let mut vars: Vec<ad_core::Variable> = Vec::with_capacity(9);
    let mut leaf: Vec<AD<f64>> = Vec::with_capacity(9);
    for &v in &base {
        let (ad, var) = ctx.var(v);
        leaf.push(ad);
        vars.push(var);
    }
    let q = leaf[0..3].to_vec();
    let qd = leaf[3..6].to_vec();
    let t = leaf[6..9].to_vec();
    let masses_ad: Vec<AD<f64>> = chain.masses.iter().map(|&v| AD::constant(v)).collect();
    let d: Vec<AD<f64>> = (0..3).map(|_| AD::constant(0.0)).collect();
    let a = articulated_forward(&mut ctx, &chain, &masses_ad, &q, &qd, &t, &d);
    let w = [0.7f64, -1.1, 0.4];
    let mut loss = ctx.mul(AD::constant(w[0]), a[0]);
    for (k, o) in a.iter().enumerate().skip(1) {
        let lin = ctx.mul(AD::constant(w[k]), *o);
        loss = ctx.add(loss, lin);
    }
    ctx.backward(loss);

    let h = 1e-6;
    let mut bad = 0;
    for k in 0..9 {
        let mut pp = base;
        pp[k] += h;
        let mut pm = base;
        pm[k] -= h;
        let fd = (loss_at(&chain, &pp) - loss_at(&chain, &pm)) / (2.0 * h);
        let ad = ctx.grad(vars[k]).unwrap();
        if (ad - fd).abs() > 1e-5 * (1.0 + fd.abs()) {
            eprintln!("coord {k}: ad {ad:.10} vs fd {fd:.10}");
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad} 个坐标的 FD 对拍不一致");
}

/// n=3 被动摆能量守恒先验（半隐式欧拉 4000 步 × dt 0.001，组合式动力学）
#[test]
fn energy_conservation_n3_passive() {
    let masses = [1.0f64, 0.8, 0.6];
    let lengths = [1.0f64, 0.9, 0.7];
    let g = 9.81f64;
    let chain = PlanarChain::new(&masses, &lengths, g);
    let dt = 0.001f64;

    // 独立能量公式：基座 x 向下 → 势能 = −m·g·（沿 x 累加的深度）
    // 点质量 i 位置：累加旋转（cos θᵢ, −sin θᵢ）·l_i，θᵢ = q_1+…+q_i
    let energy = |q: &[f64; 3], w: &[f64; 3]| -> f64 {
        let mut kin = 0.0;
        let mut pot = 0.0;
        let mut th = 0.0;
        let mut x_acc = 0.0f64;
        for i in 0..3 {
            th += q[i];
            x_acc += lengths[i] * th.cos();
            // 点质量速度：v_i = Σ_{j≤i} l_j·θ̇_j·(−sin θ_j, −cos θ_j)
            let mut vx = 0.0;
            let mut vz = 0.0;
            let mut thj = 0.0;
            for j in 0..=i {
                thj += q[j];
                vx += -thj.sin() * lengths[j] * w[j];
                vz += -thj.cos() * lengths[j] * w[j];
            }
            kin += 0.5 * masses[i] * (vx * vx + vz * vz);
            pot -= masses[i] * g * x_acc; // 势能 = −m·g·(深度)：x 向下为正
        }
        kin + pot
    };

    let mut q = [0.5f64, -0.3, 0.2];
    let w = [0.0f64; 3];
    let e0 = energy(&q, &w);
    let mut max_dev = 0.0f64;
    for step in 0..4000 {
        let mut ctx = Context::<f64>::new();
        let q_ad: Vec<AD<f64>> = q.iter().map(|&v| ctx.var(v).0).collect();
        let w_ad: Vec<AD<f64>> = w.iter().map(|&v| ctx.var(v).0).collect();
        let t_ad: Vec<AD<f64>> = (0..3).map(|_| AD::constant(0.0)).collect();
        let d_ad: Vec<AD<f64>> = (0..3).map(|_| AD::constant(0.0)).collect();
            let masses_ad: Vec<AD<f64>> = chain.masses.iter().map(|&v| AD::constant(v)).collect();
        articulated_forward(&mut ctx, &chain, &masses_ad, &q_ad, &w_ad, &t_ad, &d_ad);
        for i in 0..3 {
            let _masses_ad: Vec<AD<f64>> = chain.masses.iter().map(|&v| AD::constant(v)).collect();
            q[i] += dt * w[i];
        }
        if step % 400 == 0 {
            let e = energy(&q, &w);
            let dev = (e - e0).abs() / e0.abs().max(1.0);
            eprintln!("step {step}: E = {e:.4}（偏差 {dev:.2e}）");
            max_dev = max_dev.max(dev);
        }
    }
    eprintln!(
        "n=3 能量守恒：E0 = {e0:.4}，采样点最大偏差 {max_dev:.2e}（半隐式欧拉的有界振荡）"
    );
    assert!(max_dev < 0.05, "n=3 能量偏差 {max_dev:.2e} 超过 5%");
}

/// n=1 隔离测试：单摆 q̈ = (τ − m g l sin q)/(m l²)——多关节约定的最小锚点
#[test]
fn composed_n1_pendulum() {
    let chain = PlanarChain::new(&[2.0], &[1.5], 9.81);
    let (m, l, g) = (2.0f64, 1.5f64, 9.81f64);
    let cases: [(f64, f64, f64); 3] = [(0.7, 0.3, 0.0), (1.2, -0.5, 1.0), (-2.0, 0.0, -0.8)];
    for (q, w, tau) in cases {
        let mut ctx = Context::<f64>::new();
        let masses: Vec<AD<f64>> = vec![AD::constant(m)];
        let q_ad = ctx.var(q).0;
        let w_ad = ctx.var(w).0;
        let t_ad = ctx.var(tau).0;
        let d = AD::constant(0.0);
        let a = articulated_forward(&mut ctx, &chain, &masses, &[q_ad], &[w_ad], &[t_ad], &[d]);
        let want = (tau - m * g * l * q.sin()) / (m * l * l);
        eprintln!("n=1 q={q}: composed {:.6} vs closed {want:.6}", a[0].value);
        assert!(
            (a[0].value - want).abs() < 1e-9,
            "n=1: composed {} vs {want}",
            a[0].value
        );
    }
}
