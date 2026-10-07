//! 平滑接触模型 CustomOp 集（设计文档 §5.2"平滑近似"路线的算子化）。
//!
//! 约定：`gap < 0` 表示穿透（标准运动学间隙），`gap_vel = d(gap)/dt`，
//! 力沿 gap 增大方向推。所有平滑由 ε 参数控制，非光滑点被 C¹∞-化的
//! softplus/softsign 替代——梯度处处存在，优化友好。
//!
//! - [`ContactNormalOp`]：Hunt–Crossley 型法向力，`f = pen^p·(k − d·gap_vel)`
//!   （接近时 `gap_vel < 0` 力增大 → 耗散）；
//! - [`RegularizedFrictionOp`]：正则化库仑摩擦，`f_t = −μ·f_n·v_t/√(|v_t|²+ε²)`，
//!   严格满足 |f_t| ≤ μ·f_n（摩擦锥内）；
//! - [`BarrierContactOp`]：IPC 思路的 log-barrier 阻塞力（§9.1 路线）——
//!   纯位置依赖（保守）、C^∞ 光滑、无活动集切换，对梯度质量敏感的
//!   优化（iLQR/系统辨识）最稳的一条接触路线。
//!
//! 泛型实现（§12.3 第 38 条模式）：同一份 forward/手写 VJP 服务 f64 与 f32
//! （`S: Scalar`）；f64 路径由 FD 隔离器 + 接触锥先验验证，f32 由同点对拍
//! 单独检查（`tests/f32_ops.rs`）。

use ad_core::{CustomOp, Scalar};
use smallvec::{smallvec, SmallVec};

/// 光滑非负部分：`softplus_ε(x) = ½(x + √(x² + ε²)) ≈ max(0, x)`，C^∞。
pub fn softplus_eps<S: Scalar>(x: S, eps: S) -> S {
    let half = S::one() / (S::one() + S::one());
    half * (x + (x * x + eps * eps).sqrt())
}

/// Hunt–Crossley 型平滑法向接触力。
///
/// inputs = [gap, gap_vel, k, p, d, ε]（6），output = [f]（1）：
/// - `pen = softplus_ε(−gap)`（光滑穿透深度，恒 > 0）；
/// - `f = pen^p·(k − d·gap_vel)`，接近（gap_vel < 0）时阻尼增大受力 → 耗散。
///
/// 定义域：`pen > 0` 恒成立（softplus 下界 ε/2），`p ≥ 1`；要求
/// `k − d·gap_vel ≥ 0`（合理阻尼比），否则力可能变负（粘着）——文档化而非静默截断。
#[derive(Clone, Copy)]
pub struct ContactNormalOp;

impl<S: Scalar> CustomOp<S> for ContactNormalOp {
    fn num_inputs(&self) -> usize {
        6
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn forward(&self, i: &[S]) -> (SmallVec<[S; 8]>, SmallVec<[S; 8]>) {
        let (gap, gv, k, p, d, eps) = (i[0], i[1], i[2], i[3], i[4], i[5]);
        let half = S::one() / (S::one() + S::one());
        let rho = (gap * gap + eps * eps).sqrt();
        let pen = half * (-gap + rho);
        let f = pen.powf(p) * (k - d * gv);
        (smallvec![f], i.iter().copied().collect())
    }
    fn backward(&self, r: &[S], go: &[S]) -> SmallVec<[S; 8]> {
        let (gap, gv, k, p, d, eps) = (r[0], r[1], r[2], r[3], r[4], r[5]);
        let lf = go[0];
        let half = S::one() / (S::one() + S::one());
        let two = S::one() + S::one();
        let rho = (gap * gap + eps * eps).sqrt();
        let pen = half * (-gap + rho);
        // ∂pen/∂gap = ½(−1 + gap/ρ)
        let dpen = half * (-S::one() + gap / rho);
        let base = k - d * gv; // k − d·gap_vel
        let pen_p = pen.powf(p);
        let pen_pm1 = pen.powf(p - S::one());
        smallvec![
            lf * p * pen_pm1 * base * dpen,              // ∂/∂gap
            lf * (-d * pen_p),                           // ∂/∂gap_vel
            lf * pen_p,                                  // ∂/∂k
            lf * pen_p * pen.ln() * base,                // ∂/∂p
            lf * (-pen_p * gv),                          // ∂/∂d
            lf * p * pen_pm1 * base * eps / (two * rho), // ∂pen/∂ε = ε/(2ρ)
        ]
    }
    fn name(&self) -> &'static str {
        "contact_normal"
    }
}

/// 正则化库仑摩擦（切向平面内）。
///
/// inputs = [fn, vt_x, vt_y, μ, ε]（5），outputs = [ft_x, ft_y]（2）：
/// `ft = −μ·fn·v_t/√(|v_t|² + ε²)`。
///
/// 性质（由测试守护）：|ft| ≤ μ·fn 严格成立（摩擦锥内）；滑移速度大时退化为
/// 库仑摩擦，滑移速度 ≪ ε 时线性化（正则化）。
#[derive(Clone, Copy)]
pub struct RegularizedFrictionOp;

impl<S: Scalar> CustomOp<S> for RegularizedFrictionOp {
    fn num_inputs(&self) -> usize {
        5
    }
    fn num_outputs(&self) -> usize {
        2
    }
    fn forward(&self, i: &[S]) -> (SmallVec<[S; 8]>, SmallVec<[S; 8]>) {
        let (fn_, vx, vy, mu, eps) = (i[0], i[1], i[2], i[3], i[4]);
        let r = (vx * vx + vy * vy + eps * eps).sqrt();
        (
            smallvec![-mu * fn_ * vx / r, -mu * fn_ * vy / r],
            i.iter().copied().collect(),
        )
    }
    fn backward(&self, r: &[S], go: &[S]) -> SmallVec<[S; 8]> {
        let (fn_, vx, vy, mu, eps) = (r[0], r[1], r[2], r[3], r[4]);
        let (lx, ly) = (go[0], go[1]);
        let s2 = vx * vx + vy * vy;
        let r3 = (s2 + eps * eps).sqrt().powi(3);
        let vxvy = vx * vy;
        // ∂ft_x/∂vx = −μ·fn·(vy²+ε²)/r³；∂ft_x/∂vy = μ·fn·vx·vy/r³；y 对称
        let g_vx = -mu * fn_ * ((vy * vy + eps * eps) * lx - vxvy * ly) / r3;
        let g_vy = -mu * fn_ * (-vxvy * lx + (vx * vx + eps * eps) * ly) / r3;
        let g_fn = -mu * (vx * lx + vy * ly) / (s2 + eps * eps).sqrt();
        let g_mu = -fn_ * (vx * lx + vy * ly) / (s2 + eps * eps).sqrt();
        let g_eps = mu * fn_ * eps * (vx * lx + vy * ly) / r3;
        smallvec![g_fn, g_vx, g_vy, g_mu, g_eps]
    }
    fn name(&self) -> &'static str {
        "regularized_friction"
    }
}

/// IPC 思路的 log-barrier 阻塞力（HANDOFF §9.1 / design.md 第 45 条）。
///
/// 纯 log-barrier 力 `f = κ/gap`（能量 `−κ·ln(gap)` 的梯度）在 gap ≤ 0
/// 无定义且 gap → 0⁺ 时发散到无穷；IPC 用"激活距离 d̂ + 多项式窗"把
/// 阻塞限制在接近带内。本算子取其光滑化形态（全部由 C^∞ 的
/// softplus_ε 复合而成，处处二阶可导）：
///
/// - `g̃ = softplus_ε(gap)`——正则化间隙（≈ max(0, gap)，恒 > 0）；
/// - `â = softplus_ε(d̂ − g̃)`——光滑激活窗（gap ≥ d̂ 后指数衰减到 0）；
/// - `f = κ·â/g̃`——阻塞力 ≥ 0，恒沿 gap 增大方向推。
///
/// 极限行为（ε → 0⁺、gap ∈ (0, d̂)）：`f ≈ κ·(d̂ − gap)/gap`——
/// `κ·d̂/gap − κ` 正是 log-barrier 力族；gap → 0⁺ 时如 1/gap 发散
/// （穿透被指数级排斥），gap ≥ d̂ 时力恒为 0（远处无感知）。
/// ε > 0 时全实轴有定义：深度穿透下 `g̃ ≈ ε²/(4|gap|)`，
/// f 随穿透深度线性增长（等效刚度 4κ·d̂/ε²）。
///
/// **与另两条接触路线的分工**（§12.3 第 45 条有量化对比）：
/// Hunt–Crossley（连续、需调刚度）/ LCP 活动集（精确、半光滑切换）/
/// barrier（保守阻塞、C² 光滑、梯度场无 kink——iLQR/系统辨识首选）。
/// 代价是**刚性换光滑**：高 κ 需要更小步长或更大 ε（显式积分稳定域
/// ω·Δt < 2，ω ≈ √(κ·d̂)/g̃），刚度扫描测试实证。
///
/// inputs = [gap, κ, d̂, ε]（4），output = [f]（1）。
/// 定义域：κ > 0、d̂ > 0、ε > 0（ε 是正则化半径，非形状参数）。
#[derive(Clone, Copy)]
pub struct BarrierContactOp;

impl<S: Scalar> CustomOp<S> for BarrierContactOp {
    fn num_inputs(&self) -> usize {
        4
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn forward(&self, i: &[S]) -> (SmallVec<[S; 8]>, SmallVec<[S; 8]>) {
        let (gap, k, dhat, eps) = (i[0], i[1], i[2], i[3]);
        let g = softplus_eps(gap, eps);
        let a = softplus_eps(dhat - g, eps);
        (smallvec![k * a / g], i.iter().copied().collect())
    }
    fn backward(&self, r: &[S], go: &[S]) -> SmallVec<[S; 8]> {
        let (gap, k, dhat, eps) = (r[0], r[1], r[2], r[3]);
        let lf = go[0];
        let two = S::one() + S::one();
        // g̃ = softplus_ε(gap)：∂g̃/∂gap = ½(1 + gap/ρ)，∂g̃/∂ε = ε/(2ρ)
        let rho = (gap * gap + eps * eps).sqrt();
        let g = (gap + rho) / two;
        let gp = (S::one() + gap / rho) / two;
        let ge = eps / (two * rho);
        // â = softplus_ε(d̂ − g̃)：u = d̂ − g̃，∂â/∂u = ½(1 + u/σ)
        let u = dhat - g;
        let sig = (u * u + eps * eps).sqrt();
        let a = (u + sig) / two;
        let ah = (S::one() + u / sig) / two;
        // ∂â/∂ε = ε/(2σ) − ah·ge（u 随 ε 经 g̃ 间接依赖）
        let ae = eps / (two * sig) - ah * ge;
        let g2 = g * g;
        smallvec![
            // f = κ·â/g̃；∂â/∂gap = −ah·gp
            lf * k * (-ah * gp / g - a * gp / g2), // ∂/∂gap
            lf * a / g,                            // ∂/∂κ
            lf * k * ah / g,                       // ∂/∂d̂
            lf * k * (ae / g - a * ge / g2),       // ∂/∂ε
        ]
    }
    fn name(&self) -> &'static str {
        "barrier_contact"
    }
}
