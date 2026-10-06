//! 真实铰链多体链：双关节摆（Acrobot 构型）的手写 CustomOp（方向 3 补全）。
//!
//! 2 连杆点质量摆的拉格朗日动力学 `M(q)·q̈ + c(q,ω) + g(q) = τ`
//! （M/c/g 闭式见 [`DoublePendulumStep::forward`]），半隐式欧拉积分。
//! 与 AD 直通版（`ad-optim/tests/chain.rs` 的 `step_ad`，逐表达式入带）
//! 数值等价——本模块是其"单算子 + 手写 VJP"形态：
//! ~1 条 tape 记录/步（直通版 ~40 条），动力学 Jacobian 由 VJP 闭式给出。
//!
//! VJP 推导策略（对比 §12.3 第 24 条"直连展开 6+ 处错误"的教训）：
//! **按中间量分解**，与 AD tape 的链式路径同构，避免一揽子展开三角耦合——
//! 1. `a = M⁻¹(τ − c − g)` 的 VJP：`s = M⁻¹·λa`（M 对称 ⇒ M⁻ᵀ = M⁻¹），
//!    `λr = s`，`λM_ij = −s_i·a_j`（δM 对称 ⇒ m12 的梯度取**全和**
//!    `−(s₀a₂+s₁a₁)`，§4.6 约定）；λM 经 ∂m/∂θ2 = (−2h, −h) 传给 θ2。
//! 2. `r = τ − c − g` 的偏导是逐变量的局部项（重力二角、Coriolis 双因子）。
//! 3. 半隐式欧拉的**内部边**（§4.3.1 契约）：`θ' = θ + dt·ω'` 中 ω' 是本
//!    算子的另一个输出 ⇒ `λω'(总计) = λω' + dt·λθ'`、`λa = dt·λω'(总计)`。
//!
//! 验证体系：单步逐坐标 FD 隔离器（`tests/chain_fd.rs`）+ 能量守恒先验 +
//! 对拍 AD 直通梯度与 iLQR 收敛（`ad-optim/tests/chain.rs`）。

use ad_core::{CustomOp, Scalar};
use smallvec::{smallvec, SmallVec};

/// 双关节摆单步算子：inputs = [θ1, θ2, ω1, ω2, τ1, τ2, dt]（7），
/// outputs = [θ1', θ2', ω1', ω2']（4）。
///
/// 质量参数（m1, m2, l1, l2, g）是算子实例的常量字段——它们不进 VJP
/// （对参数的梯度走直通版或另行展开）；dt 保持为输入槽位（同
/// `PendulumStep` 约定，可对其求导）。
#[derive(Clone, Copy)]
pub struct DoublePendulumStep {
    pub m1: f64,
    pub m2: f64,
    pub l1: f64,
    pub l2: f64,
    pub g: f64,
}

impl Default for DoublePendulumStep {
    fn default() -> Self {
        DoublePendulumStep {
            m1: 1.0,
            m2: 0.8,
            l1: 1.0,
            l2: 0.9,
            g: 9.81,
        }
    }
}

impl DoublePendulumStep {
    /// 常量组合（forward/backward 共用）：M 的常数块与重力幅度。
    #[inline]
    fn consts(&self) -> (f64, f64, f64, f64, f64) {
        let DoublePendulumStep {
            m1,
            m2,
            l1,
            l2,
            g,
        } = *self;
        let a11 = (m1 + m2) * l1 * l1 + m2 * l2 * l2; // m11 的 θ2 无关块
        let b = m2 * l1 * l2; // 耦合幅度（m11 系数 2、m12 系数 1、h）
        let c22 = m2 * l2 * l2; // m22（= m12 的 θ2 无关块）
        let p = (m1 + m2) * g * l1; // 重力项 1 幅度（乘 sin θ1）
        let q = m2 * g * l2; // 重力项 2 幅度（乘 sin(θ1+θ2)，进两个方程）
        (a11, b, c22, p, q)
    }
}

impl CustomOp<f64> for DoublePendulumStep {
    fn num_inputs(&self) -> usize {
        7
    }

    fn num_outputs(&self) -> usize {
        4
    }

    /// 前向：M/c/g 闭式 + 2×2 解析逆 + 半隐式欧拉。
    /// 残差 = [θ1, θ2, ω1, ω2, dt, a1, a2]（7，SmallVec 内联上限内；
    /// ω1'/ω2' 由 ω + dt·a 重算，避免残差超 8 溢出堆）。
    fn forward(&self, i: &[f64]) -> (SmallVec<[f64; 8]>, SmallVec<[f64; 8]>) {
        let (th1, th2, w1, w2, t1, t2, dt) = (i[0], i[1], i[2], i[3], i[4], i[5], i[6]);
        let (a11, b, c22, p, q) = self.consts();

        let c2 = th2.cos();
        let s2 = th2.sin();
        let s12 = (th1 + th2).sin();

        let m11 = a11 + 2.0 * b * c2;
        let m12 = c22 + b * c2;
        let h = b * s2;
        let kappa = 2.0 * w1 * w2 + w2 * w2;

        // r = τ − c − g：c1 = −h·κ，c2f = h·ω1²，g1 = P·sinθ1 + Q·s12，g2 = Q·s12
        let r1 = t1 + h * kappa - p * th1.sin() - q * s12;
        let r2 = t2 - h * w1 * w1 - q * s12;

        let det = m11 * c22 - m12 * m12;
        let a1 = (c22 * r1 - m12 * r2) / det;
        let a2 = (m11 * r2 - m12 * r1) / det;

        let w1n = w1 + dt * a1;
        let w2n = w2 + dt * a2;
        let th1n = th1 + dt * w1n;
        let th2n = th2 + dt * w2n;

        (
            smallvec![th1n, th2n, w1n, w2n],
            smallvec![th1, th2, w1, w2, dt, a1, a2],
        )
    }

    // VJP。记 λθ1', λθ2', λω1', λω2' 为 go[0..4]，a = M⁻¹r（与 forward 同式）。
    //
    // 内部边（§4.3.1）：θ' = θ + dt·ω'（ω' 是本算子输出）⇒
    //   λω'(总计) = λω' + dt·λθ'，  λa = dt·λω'(总计)
    //
    // M⁻¹ 的 VJP（M 对称）：s = M⁻¹·λa；λr = s；
    //   λM_ij = −s_i·a_j，δM 对称 ⇒ λm12 = −(s₀a₂ + s₁a₁)（全和，§4.6）
    // λm 经 ∂m11/∂θ2 = −2h、∂m12/∂θ2 = −h 进入 θ2（m22、λm22 为常量路径，无效）。
    //
    // r 的偏导（局部项）：
    //   ∂r1/∂θ1 = −P·cosθ1 − Q·cos(θ1+θ2)；∂r2/∂θ1 = −Q·cos(θ1+θ2)
    //     （s12 = sin(θ1+θ2) 同样依赖 θ1——"重力双角"项进两个方程，§12.3 第 24 条的漏项家族）
    //   ∂r1/∂θ2 = B·cosθ2·κ − Q·cos(θ1+θ2)；∂r2/∂θ2 = −B·cosθ2·ω1² − Q·cos(θ1+θ2)
    //   ∂r1/∂ω1 = 2h·ω2；∂r2/∂ω1 = −2h·ω1；∂r1/∂ω2 = h·(2ω1+2ω2)；∂r2/∂ω2 = 0
    //   ∂r1/∂τ1 = 1；∂r2/∂τ2 = 1
    //
    // λdt：∂ω'/∂dt = a、∂θ'/∂dt = ω'（ω' = ω + dt·a 由残差重算）。
    //   注意 dt 的 ω' 路径要用 λω'(总计)——θ' = θ + dt·ω' 使 ω' 同时是
    //   θ' 的上游（内部边），其对 dt 的直接边贡献含 λθ'·dt 分量。
    fn backward(&self, r: &[f64], go: &[f64]) -> SmallVec<[f64; 8]> {
        let (th1, th2, w1, w2, dt, a1, a2) = (r[0], r[1], r[2], r[3], r[4], r[5], r[6]);
        let (lth1, lth2, lw1, lw2) = (go[0], go[1], go[2], go[3]);
        let (a11, b, c22, p, q) = self.consts();

        let c2 = th2.cos();
        let s2 = th2.sin();
        let c12 = (th1 + th2).cos();
        let c1t = th1.cos();

        let m11 = a11 + 2.0 * b * c2;
        let m12 = c22 + b * c2;
        let h = b * s2;
        let kappa = 2.0 * w1 * w2 + w2 * w2;
        let det = m11 * c22 - m12 * m12;

        // 内部边：λa、λω'(总计)
        let lw1t = lw1 + dt * lth1;
        let lw2t = lw2 + dt * lth2;
        let la1 = dt * lw1t;
        let la2 = dt * lw2t;

        // M⁻¹ VJP：s = M⁻¹·λa（与 forward 的 a = M⁻¹r 同一个 2×2 解析逆）
        let s0 = (c22 * la1 - m12 * la2) / det;
        let s1 = (m11 * la2 - m12 * la1) / det;

        // λM（全和约定）与 θ2 的 M 路径
        let lm11 = -s0 * a1;
        let lm12 = -(s0 * a2 + s1 * a1);

        // 汇总各输入槽位
        let g_th1 = lth1 + s0 * (-p * c1t - q * c12) + s1 * (-q * c12);
        let g_th2 = lth2
            + lm11 * (-2.0 * h)
            + lm12 * (-h)
            + s0 * (b * c2 * kappa - q * c12)
            + s1 * (-b * c2 * w1 * w1 - q * c12);
        let g_w1 = lw1t + s0 * (2.0 * h * w2) + s1 * (-2.0 * h * w1);
        let g_w2 = lw2t + s0 * (h * (2.0 * w1 + 2.0 * w2));
        let g_t1 = s0;
        let g_t2 = s1;
        let g_dt = lw1t * a1 + lw2t * a2 + lth1 * (w1 + dt * a1) + lth2 * (w2 + dt * a2);

        smallvec![g_th1, g_th2, g_w1, g_w2, g_t1, g_t2, g_dt]
    }

    fn name(&self) -> &'static str {
        "double_pendulum_step"
    }
}

// 保持 Scalar 在约束说明中被引用
#[allow(unused)]
fn _s<S: Scalar>() {}
