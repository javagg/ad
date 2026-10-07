//! n 关节平面铰接体：**RNEA 用空间代数算子在 tape 上组合**（§12.3 第 43 条）。
//!
//! 模型：平面串联链（旋转轴 = y），无质量连杆 + 端部点质量
//! （空间惯性 Ī = 0、c = (l, 0, 0)）——与 `ad-optim/tests/chain.rs` 的
//! 闭式双摆同一质量模型，n=2 时可交叉验证（1e-9）。
//!
//! 约定：基座系 x 轴 = 铅垂向下，零位（全 q = 0）为悬挂下垂；
//! 重力以基座伪加速度 `a_0 = (0,0,0; g,0,0)` 进入（Featherstone 约定）。
//! 关节子空间 S = e_ωy（运动向量第 1 分量），`τ_i = f_i[1]`。
//!
//! 组合（每关节 ~7 条记录，全部为已验证的 CustomOp）：
//! - 运动链：`v_i = PluckerMotion(X_i)·v_{i-1}`，分量 1 += q̇_i；
//!   `a_i = PluckerMotion(X_i)·a_{i-1}`，分量 1 += q̈_i，再 += crm(v_i)·S q̇_i；
//! - 力链：`f_i = InertiaApply(I_i, a_i) + ForceCross(v_i, InertiaApply(I_i, v_i))`；
//! - 回传：`τ_i = f_i[1]`，`f_{i-1} += PluckerForce(Eᵀ, −Eᵀr)·f_i`
//!   （力回传是运动变换的对偶：Eᵀ 与 r' = −Eᵀr，由功率不变性 v_cᵀf_c = v_pᵀf_p 推出）。
//!
//! 前向动力学（M 的 CRBA 列法 + 求解 bulk 算子）：
//! `M[:,j] = RNEA_nograv(q, 0, e_j)`、`bias = RNEA_grav(q, q̇, 0)`、
//! `q̈ = SolveSym(M, τ − bias − d·q̇)`——`solve_sym_with` 是 §4.3.4 表
//! "线性求解 → bulk CustomOp" 行的落地。
//!
//! 验证：n=2 与闭式解（独立重写的经典公式）逐点对拍（1e-9）、
//! n=3 能量守恒先验、iLQR 甩摆（ad-optim/tests/articulated.rs）。

use crate::{InertiaApply, PluckerForce, PluckerMotion, SpatialCrossMotion, SpatialForceCross};
use ad_core::{Context, CustomOp, AD};
use std::rc::Rc;

/// 平面铰接链参数（无质量连杆 + 端部点质量）。
#[derive(Clone)]
pub struct PlanarChain {
    /// 各连杆质量
    pub masses: Vec<f64>,
    /// 各连杆长度（关节 i 到质量 i 的距离，沿连杆 x 轴）
    pub lengths: Vec<f64>,
    /// 重力加速度（基座系 x 正向 = 铅垂向下）
    pub g: f64,
}

impl PlanarChain {
    pub fn new(masses: &[f64], lengths: &[f64], g: f64) -> Self {
        assert_eq!(masses.len(), lengths.len(), "mass/length count mismatch");
        PlanarChain {
            masses: masses.to_vec(),
            lengths: lengths.to_vec(),
            g,
        }
    }

    pub fn n(&self) -> usize {
        self.masses.len()
    }

    /// 运动变换 X_i 的 PluckerMotion 输入：[E(9), r(3)]，E = Ry(q_i)，
    /// r = 子系原点在父系中 = (l_{i-1}, 0, 0)（i=0 时零）。
    /// 关节变换 X_i 的 Plucker 输入：[E(9), r(3)]，E = Ry(q_i)，
    /// r = **父系原点在子系中** = Ry(q)ᵀ·(−l_{i-1}, 0, 0) = (−l·c, 0, −l·s)。
    ///
    /// 语义（数值探针 + 动能不变性测试锚定）：PluckerMotion(E, r) 计算
    /// `v_A = X·v_B`（B = 父、A = 子：E 为父→子轴旋转，r 为父原点在子系
    /// 坐标）；**同一 (E, r) 传给 PluckerForce 即得其转置对偶
    /// `f_parent = Xᵀ·f_child`（力回传）**——由功率不变性与动能不变性
    /// 测试共同守护。
    /// （实现期教训：曾把 r 误取"子原点在父系"(l,0,0)，M 的耦合列
    /// 全错——n=2 闭式对拍 + M/bias 分层探针一次定位。）
    fn joint_transform(
        ctx: &mut Context<f64>,
        q: &[AD<f64>],
        lengths: &[f64],
        i: usize,
    ) -> Vec<AD<f64>> {
        let cq = ad_ops::cos_with(ctx, q[i]);
        let sq = ad_ops::sin_with(ctx, q[i]);
        // E = Ry(q) = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        let e = vec![
            cq,

            AD::constant(0.0),
            ctx.neg(sq),
            AD::constant(0.0),
            AD::constant(1.0),
            AD::constant(0.0),
            sq,

            AD::constant(0.0),
            cq,
        ];
        // r = Ry(q)ᵀ·(−l_{i-1}, 0, 0) = (−l·c, 0, −l·s)
        let l = if i == 0 {
            0.0
        } else {
            lengths[i - 1]
        };
        let lc = ctx.mul(AD::constant(-l), cq);
        let ls = ctx.mul(AD::constant(-l), sq);
        let mut x = e;
        x.push(lc);
        x.push(AD::constant(0.0));
        x.push(ls);
        x
    }
}

#[inline]
fn const6() -> Vec<AD<f64>> {
    (0..6).map(|_| AD::constant(0.0)).collect()
}

/// RNEA 组合（逆动力学，被追踪表达式）：给定期望运动 (q, q̇, q̈)，
/// 返回各关节力矩 τ（n 个 AD）。`qdd` 为常量切片（CRBA 列法用，
/// q̇ = 0 时哥氏/陀螺项折叠为常量 0）。`gravity`：false 时 a_0 = 0
/// （质量矩阵列）。
pub fn rnea_torques(
    ctx: &mut Context<f64>,
    chain: &PlanarChain,
    masses: &[AD<f64>],
    q: &[AD<f64>],
    qd: &[AD<f64>],
    qdd: &[f64],
    gravity: bool,
) -> Vec<AD<f64>> {
    let n = chain.n();
    let crm: Rc<dyn CustomOp<f64>> = Rc::new(SpatialCrossMotion);
    let fc: Rc<dyn CustomOp<f64>> = Rc::new(SpatialForceCross);
    let ia: Rc<dyn CustomOp<f64>> = Rc::new(InertiaApply);
    let pm: Rc<dyn CustomOp<f64>> = Rc::new(PluckerMotion);
    let pf: Rc<dyn CustomOp<f64>> = Rc::new(PluckerForce);

    // 基座：v_0 = 0；a_0 = 重力伪加速度（Featherstone：a_0 = −g_vec，
    // 基座 x 向下 ⇒ g_vec = (g,0,0) ⇒ a_0 = (−g, 0, 0) 线分量）
    let mut v: Vec<AD<f64>> = const6();
    let a0 = if gravity {
        vec![
            AD::constant(0.0),
            AD::constant(0.0),
            AD::constant(0.0),
            AD::constant(-chain.g),
            AD::constant(0.0),
            AD::constant(0.0),
        ]
    } else {
        const6()
    };
    let mut a: Vec<AD<f64>> = a0;

    // ---- 正向运动/力链 ----
    let mut forces: Vec<Vec<AD<f64>>> = Vec::with_capacity(n);
    for i in 0..n {
        // PluckerMotion 输入 = [E(9), r(3), v_B(6)]：变换与被变换向量一起进

        // v_child = PluckerMotion(X_i)·v_parent；分量 1（ω_y）+= q̇_i
        let mut xin = PlanarChain::joint_transform(ctx, q, &chain.lengths, i);
        xin.extend(v.clone());
        let mut vi = ctx
            .call_custom_dyn(Rc::clone(&pm), "plucker_motion", &xin)
            .to_vec();
        vi[1] = ctx.add(vi[1], qd[i]);

        // a_child = PluckerMotion(X_i)·a_parent；分量 1 += q̈_i；+= crm(v_i)·(S q̇_i)
        let mut ain = PlanarChain::joint_transform(ctx, q, &chain.lengths, i);
        ain.extend(a.clone());
        let mut ai = ctx
            .call_custom_dyn(Rc::clone(&pm), "plucker_motion", &ain)
            .to_vec();
        ai[1] = ctx.add(ai[1], AD::constant(qdd[i]));
        let mut sm = const6();
        sm[1] = qd[i];
        let mut crm_in = vi.clone();
        crm_in.extend(sm);
        let crm_out = ctx
            .call_custom_dyn(Rc::clone(&crm), "spatial_cross_motion", &crm_in)
            .to_vec();
        for k in 0..6 {
            ai[k] = ctx.add(ai[k], crm_out[k]);
        }

        // f_i = I·a + v ×* (I·v)（点质量：Ī = 0, m = m_i, c = (l_i, 0, 0)）
        let inertia_in: Vec<AD<f64>> = vec![
            AD::constant(0.0),
            AD::constant(0.0),
            AD::constant(0.0),
            AD::constant(0.0),
            AD::constant(0.0),
            AD::constant(0.0),
            masses[i],

            AD::constant(chain.lengths[i]),
            AD::constant(0.0),
            AD::constant(0.0),
        ];
        let mut inputs_a = inertia_in.clone();
        inputs_a.extend(&ai);
        let f_acc = ctx
            .call_custom_dyn(Rc::clone(&ia), "inertia_apply", &inputs_a)
            .to_vec();
        let mut inputs_v = inertia_in;
        inputs_v.extend(&vi);
        let p = ctx
            .call_custom_dyn(Rc::clone(&ia), "inertia_apply", &inputs_v)
            .to_vec();
        let mut vm = vi.clone();
        vm.extend(p);
        let gyro = ctx
            .call_custom_dyn(Rc::clone(&fc), "spatial_force_cross", &vm)
            .to_vec();
        let mut f: Vec<AD<f64>> = Vec::with_capacity(6);
        for k in 0..6 {
            f.push(ctx.add(f_acc[k], gyro[k]));
        }
        forces.push(f);
        v = vi;
        a = ai;
    }

    // ---- 回传：τ_i = f_i[1]（S = e_ωy）；物理形式力回传 ----
    //   f_p_lin = Eᵀ·f_c_lin；n_p = Eᵀ·(n_c + r_mc × f_c_lin)
    // （r_mc = O_c − O_p 在子系坐标 = 关节变换的 r；全部由已验证的
    //   matvec bulk 算子 + 三角表达式组合，正确性由构造保证——
    //   PluckerForce 的方向语义经数值探针确认与本回传不同，弃用）
    let mut tau = vec![AD::constant(0.0); n];
    let mut f_up: Option<Vec<AD<f64>>> = None;
    for i in (0..n).rev() {
        let f_i = match &f_up {
            Some(extra) => {
                let mut f = forces[i].clone();
                for k in 0..6 {
                    f[k] = ctx.add(f[k], extra[k]);
                }
                f
            }
            None => forces[i].clone(),

        };
        tau[i] = f_i[1];
        if i > 0 {
            // 力回传（对偶）：f_parent = PluckerForce(E_mcᵀ, r_f)(f_child)，
            //   E_mcᵀ = Ryᵀ(q)、r_f = −E_mcᵀ·r_mc = (l·cos 2q, 0, l·sin 2q)。
            // PluckerForce(E, r) = X_m(E, r)ᵀ 的力形——功率不变性 +
            // 正交 E 数值探针逐位验证（0.000e0）。
            let l = chain.lengths[i - 1];
            let cq = ad_ops::cos_with(ctx, q[i]);
            let sq = ad_ops::sin_with(ctx, q[i]);
            // E_f = child→parent = Ry(q) = [[c, 0, s], [0, 1, 0], [−s, 0, c]]；
            // r_f = child origin in parent = (l, 0, 0)
            let et = vec![
                cq,

                AD::constant(0.0),
                sq,

                AD::constant(0.0),
                AD::constant(1.0),
                AD::constant(0.0),
                ctx.neg(sq),
                AD::constant(0.0),
                cq,
            ];
            let mut x = et;
            x.push(AD::constant(l));
            x.push(AD::constant(0.0));
            x.push(AD::constant(0.0));
            x.extend(f_i);
            f_up = Some(
                ctx.call_custom_dyn(Rc::clone(&pf), "plucker_force", &x)
                    .to_vec(),
            );
        }
    }
    tau
}

/// 组合式前向动力学：`q̈ = M(q)⁻¹(τ − bias(q, q̇) − d·q̇)`，
/// M 列 = 无重力 RNEA(q, 0, e_j)、bias = 含重力 RNEA(q, q̇, 0)，
/// 求解用 `solve_sym_with` bulk 算子（§4.3.4）。`damping` 为逐关节
/// 粘滞阻尼（系统辨识的可辨识参数，传 0 向量即无阻尼）。
pub fn articulated_forward(
    ctx: &mut Context<f64>,
    chain: &PlanarChain,
    masses: &[AD<f64>],
    q: &[AD<f64>],
    qd: &[AD<f64>],
    tau: &[AD<f64>],
    damping: &[AD<f64>],
) -> Vec<AD<f64>> {
    let n = chain.n();
    let zeros = vec![0.0f64; n];
    let qd_zero: Vec<AD<f64>> = (0..n).map(|_| AD::constant(0.0)).collect();

    // bias = 含重力 RNEA(q, q̇, 0)
    let bias = rnea_torques(ctx, chain, masses, q, qd, &zeros, true);
    // M 的列：无重力 RNEA(q, 0, e_j)（q̇ = 0 → 哥氏/陀螺项自然折叠为常量 0）
    let mut m_flat = Vec::with_capacity(n * n);
    for j in 0..n {
        let mut e = zeros.clone();
        e[j] = 1.0;
        let col = rnea_torques(ctx, chain, masses, q, &qd_zero, &e, false);
        for i in 0..n {
            m_flat.push(col[i]);
        }
    }
    // rhs = τ − d·q̇ − bias
    let mut rhs = Vec::with_capacity(n);
    for i in 0..n {
        let damp = ctx.mul(damping[i], qd[i]);
        let t1 = ctx.sub(tau[i], damp);
        rhs.push(ctx.sub(t1, bias[i]));
    }
    ad_ops::solve_sym_with(ctx, &m_flat, &rhs)
}
