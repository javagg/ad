//! 3×3 / 6×6 空间代数的泛型辅助函数（不算子，供 CustomOp 的
//! forward / backward 复用；`S: Scalar` 同时服务 f64 与 f32——§12.3 第 41 条）。
//!
//! 约定（本 crate 全局一致，由动能不变性测试守护跨算子一致性）：
//! - 空间运动向量 `v = [ω(3), v(3)]`（角速度在前），力向量 `f = [n(3), f(3)]`（力矩在前）
//! - 3×3 矩阵按行主序展平为 9 个分量
//! - 空间惯性 `I = [[Ī, m[c]×], [−m[c]×, m·1]]`（关于坐标系原点），
//!   以 `(Ī 的 6 个对称分量, m, c)` 三元组参数化；动能
//!   `T = ½ωᵀĪ_Bω + ½m|v_O + ω×c|²`

use ad_core::Scalar;
use num_traits::NumCast;

#[inline]
pub(crate) fn cast<S: Scalar>(v: f64) -> S {
    NumCast::from(v).expect("spatial constant cast f64→S")
}

/// 3×3 反对称矩阵 `[a]×`（行主序 9 分量）
pub fn skew<S: Scalar>(a: &[S; 3]) -> [S; 9] {
    let z = S::zero();
    [
        z, -a[2], a[1], //
        a[2], z, -a[0], //
        -a[1], a[0], z,
    ]
}

/// `vee([a]×) = a`：从反对称阵提取向量。
/// 约定 [a]× 行主序 = [[0,−a2,a1],[a2,0,−a0],[−a1,a0,0]] → m1=−a2, m2=a1, m5=−a0
pub fn vee<S: Scalar>(m: [S; 9]) -> [S; 3] {
    [-m[5], m[2], -m[1]]
}

/// 3×3 矩阵乘法（行主序）
pub fn mat3_mul<S: Scalar>(a: &[S; 9], b: &[S; 9]) -> [S; 9] {
    let mut o = [S::zero(); 9];
    for i in 0..3 {
        for j in 0..3 {
            o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        }
    }
    o
}

pub fn mat3_t<S: Scalar>(a: &[S; 9]) -> [S; 9] {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}

pub fn mat3_vec<S: Scalar>(a: &[S; 9], v: &[S; 3]) -> [S; 3] {
    [
        a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
        a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
        a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
    ]
}

pub fn cross<S: Scalar>(a: &[S; 3], b: &[S; 3]) -> [S; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// 对称 3×3 的 6 分量打包顺序：[xx, yy, zz, xy, xz, yz]
pub mod sym3 {
    use super::Scalar;

    /// 打包：取上三角
    pub fn pack<S: Scalar>(m: &[S; 9]) -> [S; 6] {
        [m[0], m[4], m[8], m[1], m[2], m[5]]
    }

    /// 解包（打包形式本身已对称）
    pub fn unpack<S: Scalar>(s: &[S; 6]) -> [S; 9] {
        [
            s[0], s[3], s[4], //
            s[3], s[1], s[5], //
            s[4], s[5], s[2],
        ]
    }
}

/// SO(3) 指数映射：Rodrigues 公式。`w = θ·u`；
/// `R = cosθ·I + (1−cosθ)·uuᵀ + sinθ·[u]×`。
/// θ → 0 时按级数退化（确定性分支；阈值 1e-8 对 f32 亦安全——u = w/θ
/// 在该量级仍有 ~1e-7 相对精度，sin_cos 由库函数保证）。
pub fn so3_exp<S: Scalar>(w: &[S; 3]) -> [S; 9] {
    let theta = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    let small: S = cast(1e-8);
    if theta < small {
        // R ≈ I + [w]× + ½[w]×²
        let wx = skew(w);
        let wx2 = mat3_mul(&wx, &wx);
        let half: S = cast(0.5);
        let mut r = [S::zero(); 9];
        for i in 0..9 {
            let diag = if i == 0 || i == 4 || i == 8 { S::one() } else { S::zero() };
            r[i] = diag + wx[i] + half * wx2[i];
        }
        return r;
    }
    let u = [w[0] / theta, w[1] / theta, w[2] / theta];
    let (s, c) = theta.sin_cos();
    let ux = skew(&u);
    let one_m_c = S::one() - c;
    let mut r = [S::zero(); 9];
    for i in 0..9 {
        let diag = if i == 0 || i == 4 || i == 8 { S::one() } else { S::zero() };
        r[i] = c * diag + one_m_c * u[i / 3] * u[i % 3] + s * ux[i];
    }
    r
}
/// Rodrigues 的 VJP 组件：∂R/∂θ = [u]×R（列作用）。
pub(crate) fn so3_grad_theta<S: Scalar>(u: &[S; 3], r: &[S; 9], lam: &[S; 9]) -> S {
    // ⟨λ, [u]×R⟩ = uᵀ·Σ_j (R(:,j) × λ(:,j))
    let mut acc = [S::zero(); 3];
    for j in 0..3 {
        let col_r = [r[j], r[3 + j], r[6 + j]];
        let col_l = [lam[j], lam[3 + j], lam[6 + j]];
        let cr = cross(&col_r, &col_l);
        acc[0] = acc[0] + cr[0];
        acc[1] = acc[1] + cr[1];
        acc[2] = acc[2] + cr[2];
    }
    u[0] * acc[0] + u[1] * acc[1] + u[2] * acc[2]
}
