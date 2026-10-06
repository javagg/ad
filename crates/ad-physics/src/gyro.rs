//! 陀螺力学：自由刚体的 Euler 顶方程（方向 3 的 3D 深化）。
//!
//! 对角惯量张量 I = diag(I1, I2, I3) 下，本体角速度满足 Euler 方程：
//! `ω̇ = I⁻¹(ω × Iω)`
//! （无外力矩时刚体动量矩守恒，ω 的演化由陀螺耦合驱动——网球拍定理的来源）。
//!
//! VJP 闭式（对角惯量，推导见 `backward` 注释）：
//! `∂a1/∂ω2 = I3ω3/I1`，`∂a1/∂ω3 = −I2ω2/I1`，`∂a1/∂ω1 = 0`，其余循环。
//!
//! 泛型实现（§12.3 第 38 条）：同一份 forward/手写 VJP 服务 f64 与 f32
//! （`S: Scalar`，超越函数与常量经 `num_traits::Float`）。f64 路径的
//! FD 隔离器 + 角动量守恒验证覆盖全部代码路径；f32 专属的求值点舍入
//! 由 f32/f64 同点对拍测试单独验证。

use ad_core::{CustomOp, Scalar};
use smallvec::{smallvec, SmallVec};

/// 陀螺步：inputs = [ω1, ω2, ω3, I1, I2, I3, dt]（7），
/// outputs = [ω1', ω2', ω3']（3），ω' = ω + dt·I⁻¹(ω × Iω)。
#[derive(Clone, Copy)]
pub struct GyroscopicStep;

#[inline]
fn accel<S: Scalar>(w: [S; 3], i: [S; 3]) -> [S; 3] {
    let l = [i[0] * w[0], i[1] * w[1], i[2] * w[2]];
    let x = [
        w[1] * l[2] - w[2] * l[1],
        w[2] * l[0] - w[0] * l[2],
        w[0] * l[1] - w[1] * l[0],
    ];
    [x[0] / i[0], x[1] / i[1], x[2] / i[2]]
}

impl<S: Scalar> CustomOp<S> for GyroscopicStep {
    fn num_inputs(&self) -> usize {
        7
    }

    fn num_outputs(&self) -> usize {
        3
    }

    fn forward(&self, i: &[S]) -> (SmallVec<[S; 8]>, SmallVec<[S; 8]>) {
        let w = [i[0], i[1], i[2]];
        let ia = [i[3], i[4], i[5]];
        let dt = i[6];
        let a = accel(w, ia);
        // Euler 方程：ω̇ = −I⁻¹(ω×Iω)（注意负号——(ω×L) 形式天然多一个负号）
        (
            smallvec![w[0] - dt * a[0], w[1] - dt * a[1], w[2] - dt * a[2]],
            i.iter().copied().collect(),
        )
    }

    // VJP。记 `a = +I⁻¹(ω×Iω)`，物理 ω̇ = −a（Euler 方程的符号），步长取
    // `ω' = ω − dt·a` → `λa = −dt·λω'`（Jacobian 项保持 a 的导数形式）：
    // - `a1 = ω2ω3(I3−I2)/I1`，`a2 = ω1ω3(I1−I3)/I2`，`a3 = ω1ω2(I2−I1)/I3`
    // - `∂a1/∂ω2 = I3ω3/I1`，`∂a1/∂ω3 = −I2ω2/I1`，`∂a1/∂ω1 = 0`（循环）
    // - `λω = λω' − dt·(∂a/∂ω)ᵀλω'`；`λdt = −Σλω'·a`；`λI = −dt·Σλω'·∂a/∂I`
    //   （I 的导数含 −a/I 自身项，双重负号相消，系数保持不变）
    fn backward(&self, r: &[S], go: &[S]) -> SmallVec<[S; 8]> {
        let (w1, w2, w3) = (r[0], r[1], r[2]);
        let (i1, i2, i3) = (r[3], r[4], r[5]);
        let dt = r[6];
        let (l1, l2, l3) = (go[0], go[1], go[2]);
        let la1 = -dt * l1;
        let la2 = -dt * l2;
        let la3 = -dt * l3;

        // a 的值（λdt 与 λI 需要）
        let a1 = w2 * w3 * (i3 - i2) / i1;
        let a2 = w1 * w3 * (i1 - i3) / i2;
        let a3 = w1 * w2 * (i2 - i1) / i3;

        // ∂a/∂ωᵀ · λa（对角惯量闭式）
        let g_w1 = l1 + (w3 * (i1 - i3) / i2 * la2 + w2 * (i2 - i1) / i3 * la3);
        let g_w2 = l2 + (w3 * (i3 - i2) / i1 * la1 + w1 * (i2 - i1) / i3 * la3);
        let g_w3 = l3 + (w2 * (i3 - i2) / i1 * la1 + w1 * (i1 - i3) / i2 * la2);

        // λI = la·Σ ∂a_i/∂I
        let g_i1 = la1 * (-a1 / i1) + la2 * (w1 * w3 / i2) + la3 * (-w1 * w2 / i3);
        let g_i2 = la1 * (-w2 * w3 / i1) + la2 * (-a2 / i2) + la3 * (w1 * w2 / i3);
        let g_i3 = la1 * (w2 * w3 / i1) + la2 * (-w1 * w3 / i2) + la3 * (-a3 / i3);

        // λdt：∂ω'/∂dt = −a（la 已含 −dt 因子，此处需除回）
        let g_dt = (la1 * a1 + la2 * a2 + la3 * a3) / dt;

        smallvec![g_w1, g_w2, g_w3, g_i1, g_i2, g_i3, g_dt]
    }

    fn name(&self) -> &'static str {
        "gyroscopic_step"
    }
}
