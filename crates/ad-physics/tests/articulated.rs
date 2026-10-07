//! 铰接体组合动力学验证（§12.3 第 43 条）：
//! 1. n=2 交叉验证——组合式（空间代数算子 RNEA + solve_sym）与
//!    **独立重写的经典闭式公式**（点质量双摆拉格朗日）逐点对拍；
//! 2. n=3 能量守恒先验（被动摆半隐式欧拉 2000 步）；
//! 3. 单步 AD 梯度 vs FD 逐坐标抽查。

use ad_core::{Context, AD};

use ad_physics::{articulated_forward, rnea_torques, ChainDesc, PlanarChain};

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
fn composed_dd2(chain: &PlanarChain, th: [f64; 2], w: [f64; 2], tau: [f64; 2]) -> [f64; 2] {
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

// ============================================================ 独立能量/拉格朗日公式（能量守恒 + n=3 oracle 两测试共用；
// 与组合式路径零共享代码——"防两处同错"原则）

/// 动能：点质量 i 位置 pᵢ = Σ_{j≤i} l_j·(cos θ_j, −sin θ_j)，θ_j 绝对角，
/// vᵢ = Σ_{j≤i} l_j·θ̇_j·(−sin θ_j, −cos θ_j)——**θ̇_j 是关节速率的累积和**
/// （w₁+…+w_j），非 w[j] 本身（第 46 条勘误）。
fn chain_ke(q: &[f64; 3], w: &[f64; 3], lengths: &[f64; 3], masses: &[f64; 3]) -> f64 {
    let mut kin = 0.0;
    for i in 0..3 {
        let (mut vx, mut vz) = (0.0, 0.0);
        let mut thj = 0.0;
        let mut wdj = 0.0;
        for j in 0..=i {
            thj += q[j];
            wdj += w[j];
            vx += -thj.sin() * lengths[j] * wdj;
            vz += -thj.cos() * lengths[j] * wdj;
        }
        kin += 0.5 * masses[i] * (vx * vx + vz * vz);
    }
    kin
}

/// 势能：基座 x 向下 → −Σ mᵢ·g·（沿 x 累加的深度）
fn chain_pot(q: &[f64; 3], lengths: &[f64; 3], masses: &[f64; 3], g: f64) -> f64 {
    let mut pot = 0.0;
    let mut th = 0.0;
    let mut x_acc = 0.0;
    for i in 0..3 {
        th += q[i];
        x_acc += lengths[i] * th.cos();
        pot -= masses[i] * g * x_acc;
    }
    pot
}

fn chain_energy(q: &[f64; 3], w: &[f64; 3], lengths: &[f64; 3], masses: &[f64; 3], g: f64) -> f64 {
    chain_ke(q, w, lengths, masses) + chain_pot(q, lengths, masses, g)
}

/// M_ij = ∂²KE/∂w_i∂w_j（4 点中心二阶差分，步长 h）
fn mass_ij(
    q: &[f64; 3],
    w: &[f64; 3],
    lengths: &[f64; 3],
    masses: &[f64; 3],
    i: usize,
    j: usize,
    h: f64,
) -> f64 {
    let mut wpp = *w;
    wpp[i] += h;
    wpp[j] += h;
    let mut wpm = *w;
    wpm[i] += h;
    wpm[j] -= h;
    let mut wmp = *w;
    wmp[i] -= h;
    wmp[j] += h;
    let mut wmm = *w;
    wmm[i] -= h;
    wmm[j] -= h;
    (chain_ke(q, &wpp, lengths, masses)
        - chain_ke(q, &wpm, lengths, masses)
        - chain_ke(q, &wmp, lengths, masses)
        + chain_ke(q, &wmm, lengths, masses))
        / (4.0 * h * h)
}

/// 高斯消元 3×3（部分主元）
fn solve3(m: &[[f64; 3]; 3], b: &[f64; 3]) -> [f64; 3] {
    let mut a = *m;
    let mut x = *b;
    for col in 0..3 {
        let mut piv = col;
        for r in col + 1..3 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        x.swap(col, piv);
        for r in col + 1..3 {
            let f = a[r][col] / a[col][col];
            for c in col..3 {
                a[r][c] -= f * a[col][c];
            }
            x[r] -= f * x[col];
        }
    }
    for r in (0..3).rev() {
        for c in r + 1..3 {
            x[r] -= a[r][c] * x[c];
        }
        x[r] /= a[r][r];
    }
    x
}

/// n=3 被动摆能量守恒先验（半隐式欧拉 4000 步 × dt 0.001，组合式动力学）。
///
/// **实现期勘误（第 46 条）**：本测试首版有两处相互抵消的 bug——
/// (a) 速度从不更新（w 恒 0，"守恒"的是冻结系统）；
/// (b) 动能公式把关节速率 w[j] 直接当连杆绝对角速率（缺累积和），
///     静止时两者同为 0 → 检不出。
/// 修正后能量才真正守恒（有界振荡），并首次以运动态验证 n=3 组合动力学。
#[test]
fn energy_conservation_n3_passive() {
    let masses = [1.0f64, 0.8, 0.6];
    let lengths = [1.0f64, 0.9, 0.7];
    let g = 9.81f64;
    let chain = PlanarChain::new(&masses, &lengths, g);
    let dt = 0.001f64;

    let mut q = [0.5f64, -0.3, 0.2];
    let mut w = [0.3f64, -0.2, 0.1]; // 非零初速：动能通道必须真的参与
    let e0 = chain_energy(&q, &w, &lengths, &masses, g);
    let mut max_dev = 0.0f64;
    for step in 0..4000 {
        let mut ctx = Context::<f64>::new();
        let q_ad: Vec<AD<f64>> = q.iter().map(|&v| ctx.var(v).0).collect();
        let w_ad: Vec<AD<f64>> = w.iter().map(|&v| ctx.var(v).0).collect();
        let t_ad: Vec<AD<f64>> = (0..3).map(|_| AD::constant(0.0)).collect();
        let d_ad: Vec<AD<f64>> = (0..3).map(|_| AD::constant(0.0)).collect();
        let masses_ad: Vec<AD<f64>> = chain.masses.iter().map(|&v| AD::constant(v)).collect();
        let a = articulated_forward(&mut ctx, &chain, &masses_ad, &q_ad, &w_ad, &t_ad, &d_ad);
        // 半隐式欧拉（先速度后位置）
        let mut w_new = [0.0f64; 3];
        for i in 0..3 {
            w_new[i] = w[i] + dt * a[i].value;
        }
        for i in 0..3 {
            q[i] += dt * w_new[i];
        }
        w = w_new;
        if step % 400 == 0 {
            let e = chain_energy(&q, &w, &lengths, &masses, g);
            let dev = (e - e0).abs() / e0.abs().max(1.0);
            max_dev = max_dev.max(dev);
        }
    }
    eprintln!("n=3 能量守恒：E0 = {e0:.4}，采样点最大偏差 {max_dev:.2e}（半隐式欧拉的有界振荡）");
    assert!(max_dev < 0.05, "n=3 能量偏差 {max_dev:.2e} 超过 5%");
}

/// n=3 运动态 vs 独立拉格朗日 oracle（第 46 条补上的第一个 n=3 物理
/// oracle）：M 与 V 从独立 KE/势能公式数值导出（M_ij = ∂²KE/∂w_i∂w_j、
/// g_i = ∂V/∂q_i、Coriolis 用 Christoffel 符号），q̈ = M⁻¹(τ − Cw − g)
/// 高斯消元求解，与组合式在多个运动态对拍。此前的 oracle 链
/// （n=2 闭式 + AD-vs-FD）到 n=3 断裂：FD 对拍是"同一函数的自检"，
/// 能量测试的速度冻结使它只覆盖了重力通道。
#[test]
fn composed_matches_lagrangian_oracle_n3() {
    let masses = [1.0f64, 0.8, 0.6];
    let lengths = [1.0f64, 0.9, 0.7];
    let g = 9.81f64;
    let chain = PlanarChain::new(&masses, &lengths, g);
    let h = 1e-4f64;
    let states: [([f64; 3], [f64; 3]); 4] = [
        ([0.4, -0.3, 0.2], [-0.138, 0.288, -0.224]),
        ([0.4, -0.3, 0.2], [0.5, -0.3, 0.2]),
        ([1.0, 0.5, -0.8], [-0.6, 0.9, 0.4]),
        ([-0.7, 1.2, 0.3], [1.0, -0.5, 0.4]),
    ];
    for (q, w) in states {
        // 数值 M、g（h=1e-4 平衡截断与噪声）
        let mut m = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = mass_ij(&q, &w, &lengths, &masses, i, j, h);
            }
        }
        let mut gvec = [0.0f64; 3];
        for i in 0..3 {
            let mut qp = q;
            qp[i] += h;
            let mut qm = q;
            qm[i] -= h;
            gvec[i] = (chain_pot(&qp, &lengths, &masses, g) - chain_pot(&qm, &lengths, &masses, g))
                / (2.0 * h);
        }
        // Coriolis：(Cw)_i = ½ Σ_jk (∂M_ij/∂q_k + ∂M_ik/∂q_j − ∂M_jk/∂q_i) w_j w_k
        let dmq = |i: usize, j: usize, k: usize| -> f64 {
            let mut qp = q;
            qp[k] += h;
            let mut qm = q;
            qm[k] -= h;
            (mass_ij(&qp, &w, &lengths, &masses, i, j, h)
                - mass_ij(&qm, &w, &lengths, &masses, i, j, h))
                / (2.0 * h)
        };
        let mut cw = [0.0f64; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    cw[i] += 0.5 * (dmq(i, j, k) + dmq(i, k, j) - dmq(j, k, i)) * w[j] * w[k];
                }
            }
        }
        // q̈ = M⁻¹(τ − Cw − g)，τ = 0
        let rhs = [-cw[0] - gvec[0], -cw[1] - gvec[1], -cw[2] - gvec[2]];
        let a_ref = solve3(&m, &rhs);

        // 组合式
        let mut ctx = Context::<f64>::new();
        let q_ad: Vec<AD<f64>> = q.iter().map(|&v| ctx.var(v).0).collect();
        let w_ad: Vec<AD<f64>> = w.iter().map(|&v| ctx.var(v).0).collect();
        let zeros: Vec<AD<f64>> = (0..3).map(|_| AD::constant(0.0)).collect();
        let masses_ad: Vec<AD<f64>> = chain.masses.iter().map(|&v| AD::constant(v)).collect();
        let a_cmp = articulated_forward(&mut ctx, &chain, &masses_ad, &q_ad, &w_ad, &zeros, &zeros);

        let scale = (0..3)
            .map(|i| a_ref[i].abs())
            .fold(0.0f64, f64::max)
            .max(1.0);
        for i in 0..3 {
            let err = (a_cmp[i].value - a_ref[i]).abs();
            assert!(
                err < 2e-2 * scale,
                "state {q:?} w {w:?}: q̈[{i}] composed {:.6} vs lagrangian {:.6}",
                a_cmp[i].value,
                a_ref[i]
            );
        }
    }
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

// ============================================================ 数据驱动链描述（第 47 条）

/// URDF-lite 描述构建 vs 手写硬编码：f64 槽位、RNEA 力矩、150 步 rollout
/// 全部**逐位一致**（构建器只做原值拷贝、无算术重排——"数据驱动换掉
/// 硬编码而不改变任何数值行为"的验收）。
#[test]
fn chain_desc_bit_identical_to_hardcoded() {
    let desc = ChainDesc::new(9.81)
        .joint(1.0, 1.0, 0.0)
        .joint(0.8, 0.9, 0.15)
        .joint(0.6, 0.7, 0.0);
    let (chain, damping) = desc.to_chain();
    let hardcoded = PlanarChain::new(&[1.0, 0.8, 0.6], &[1.0, 0.9, 0.7], 9.81);
    let damping_hard = [0.0f64, 0.15, 0.0];

    // 槽位逐位
    for i in 0..3 {
        assert_eq!(chain.masses[i].to_bits(), hardcoded.masses[i].to_bits());
        assert_eq!(chain.lengths[i].to_bits(), hardcoded.lengths[i].to_bits());
        assert_eq!(damping[i].to_bits(), damping_hard[i].to_bits());
    }
    assert_eq!(chain.g.to_bits(), hardcoded.g.to_bits());

    // RNEA 力矩逐位（运动态，速度非零）
    let sample = (
        [0.4f64, -0.3, 0.9],
        [0.2f64, -0.5, 0.3],
        [0.1f64, -0.05, 0.02],
    );
    let torques_bits = |chain: &PlanarChain| -> Vec<u64> {
        let mut ctx = Context::<f64>::new();
        let q: Vec<AD<f64>> = sample.0.iter().map(|&v| ctx.var(v).0).collect();
        let w: Vec<AD<f64>> = sample.1.iter().map(|&v| ctx.var(v).0).collect();
        let m: Vec<AD<f64>> = chain.masses.iter().map(|&v| AD::constant(v)).collect();
        rnea_torques(&mut ctx, chain, &m, &q, &w, &sample.2, true)
            .iter()
            .map(|x| x.value.to_bits())
            .collect()
    };
    assert_eq!(torques_bits(&chain), torques_bits(&hardcoded));

    // 150 步带阻尼 rollout 逐位
    let rollout_bits = |chain: &PlanarChain, dmp: &[f64; 3]| -> Vec<u64> {
        let mut ctx = Context::<f64>::new();
        let mut q: Vec<AD<f64>> = sample.0.iter().map(|&v| ctx.var(v).0).collect();
        let mut w: Vec<AD<f64>> = sample.1.iter().map(|&v| ctx.var(v).0).collect();
        let mut bits = Vec::new();
        for _ in 0..150 {
            let t: Vec<AD<f64>> = sample.2.iter().map(|&v| AD::constant(v)).collect();
            let m: Vec<AD<f64>> = chain.masses.iter().map(|&v| AD::constant(v)).collect();
            let d: Vec<AD<f64>> = dmp.iter().map(|&v| AD::constant(v)).collect();
            let a = articulated_forward(&mut ctx, chain, &m, &q, &w, &t, &d);
            let mut w_new = Vec::with_capacity(3);
            for i in 0..3 {
                let dv = ctx.mul(AD::constant(0.02), a[i]);
                w_new.push(ctx.add(w[i], dv));
            }
            let mut q_new = Vec::with_capacity(3);
            for i in 0..3 {
                let dq = ctx.mul(AD::constant(0.02), w_new[i]);
                q_new.push(ctx.add(q[i], dq));
            }
            q = q_new;
            w = w_new;
            bits.push(q[0].value.to_bits());
        }
        bits
    };
    assert_eq!(
        rollout_bits(&chain, &damping[..].try_into().unwrap()),
        rollout_bits(&hardcoded, &damping_hard)
    );
}
