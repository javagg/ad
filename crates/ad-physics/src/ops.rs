//! 空间代数参考 CustomOp 集（手写 VJP，全部由"单步逐坐标 FD"测试守护）。
//!
//! 输入/输出均为展平的 `S` 切片（`S: Scalar`——同一份 forward/手写 VJP
//! 服务 f64 与 f32，§12.3 第 41 条），布局见各算子文档。约定见 [`crate::spatial`]。
//! 泛型代码路径由 f64 的 FD 隔离器 + 动能不变性测试覆盖；f32 由泛型
//! 验证器直接验证（`tests/f32_ops.rs`）。

use crate::spatial::{self, sym3};
use ad_core::{CustomOp, Scalar};
use num_traits::NumCast;
use smallvec::{smallvec, SmallVec};

#[inline]
fn c<S: Scalar>(v: f64) -> S {
    NumCast::from(v).expect("spatial constant cast f64→S")
}

/// `out = crm(v1)·v2`（空间运动叉积）：
/// `out = (ω1×ω2 ; v1×ω2 + ω1×v2)`。
/// inputs = [v1(6), v2(6)]，outputs = [out(6)]。
#[derive(Clone, Copy)]
pub struct SpatialCrossMotion;

impl<S: Scalar> CustomOp<S> for SpatialCrossMotion {
    fn num_inputs(&self) -> usize {
        12
    }
    fn num_outputs(&self) -> usize {
        6
    }
    fn forward(&self, i: &[S]) -> (SmallVec<[S; 8]>, SmallVec<[S; 8]>) {
        let (w1, v1) = (&i[0..3], &i[3..6]);
        let (w2, v2) = (&i[6..9], &i[9..12]);
        let a = |x: &[S], y: &[S]| spatial::cross(&[x[0], x[1], x[2]], &[y[0], y[1], y[2]]);
        let w = a(w1, w2);
        let t1 = a(v1, w2);
        let t2 = a(w1, v2);
        (
            smallvec![
                w[0],
                w[1],
                w[2],
                t1[0] + t2[0],
                t1[1] + t2[1],
                t1[2] + t2[2]
            ],
            i.iter().copied().collect(),
        )
    }
    fn backward(&self, r: &[S], go: &[S]) -> SmallVec<[S; 8]> {
        let (w1, v1) = (&r[0..3], &r[3..6]);
        let (lw, lv) = (&go[0..3], &go[3..6]);
        let cr = |x: &[S], y: &[S]| spatial::cross(&[x[0], x[1], x[2]], &[y[0], y[1], y[2]]);
        // g_v1 = (ω2×λω + v2×λv ; ω2×λv)
        let w2: &[S; 3] = (&r[6..9]).try_into().unwrap();
        let v2: &[S; 3] = (&r[9..12]).try_into().unwrap();
        let a = cr(w2, lw);
        let b = cr(v2, lv);
        let gv1w = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
        let gv1v = cr(w2, lv);
        // g_v2 = crm(v1)ᵀ·λ = (λω×ω1 + λv×v1 ; λv×ω1)
        let c1 = cr(lw, w1);
        let c2 = cr(lv, v1);
        let gv2w = [c1[0] + c2[0], c1[1] + c2[1], c1[2] + c2[2]];
        let gv2v = cr(lv, w1);
        smallvec![
            gv1w[0], gv1w[1], gv1w[2], gv1v[0], gv1v[1], gv1v[2], gv2w[0], gv2w[1], gv2w[2],
            gv2v[0], gv2v[1], gv2v[2],
        ]
    }
    fn name(&self) -> &'static str {
        "spatial_cross_motion"
    }
}

/// Plücker 运动向量变换（B 系 → A 系）：
/// `v_A = (E·ω_B ; E·v_B + r×(E·ω_B))`，
/// 其中 E 为 B→A 旋转（行主序 3×3），r 为 B 原点在 A 系中的位置。
/// inputs = [E(9), r(3), v_B(6)]，outputs = [v_A(6)]。
#[derive(Clone, Copy)]
pub struct PluckerMotion;

impl<S: Scalar> CustomOp<S> for PluckerMotion {
    fn num_inputs(&self) -> usize {
        18
    }
    fn num_outputs(&self) -> usize {
        6
    }
    fn forward(&self, i: &[S]) -> (SmallVec<[S; 8]>, SmallVec<[S; 8]>) {
        let e: &[S; 9] = (&i[0..9]).try_into().unwrap();
        let r: &[S; 3] = (&i[9..12]).try_into().unwrap();
        let (w, v) = (&i[12..15], &i[15..18]);
        let w_a = spatial::mat3_vec(e, &[w[0], w[1], w[2]]);
        let v_a0 = spatial::mat3_vec(e, &[v[0], v[1], v[2]]);
        let t = spatial::cross(r, &w_a);
        (
            smallvec![
                w_a[0],
                w_a[1],
                w_a[2],
                v_a0[0] + t[0],
                v_a0[1] + t[1],
                v_a0[2] + t[2]
            ],
            i.iter().copied().collect(),
        )
    }
    fn backward(&self, r_in: &[S], go: &[S]) -> SmallVec<[S; 8]> {
        let e: &[S; 9] = (&r_in[0..9]).try_into().unwrap();
        let r: &[S; 3] = (&r_in[9..12]).try_into().unwrap();
        let (w, v) = (&r_in[12..15], &r_in[15..18]);
        let (lw, lv) = (&go[0..3], &go[3..6]);
        let et = spatial::mat3_t(e);
        // g_v = Eᵀ·λv（out_v = E·v 的贡献）
        let gv = spatial::mat3_vec(&et, &[lv[0], lv[1], lv[2]]);
        // g_ω = Eᵀ·λω + Eᵀ·[r]×ᵀ·λv = Eᵀ·(λω − r×λv)（out_ω = E·ω 与 out_v 中
        // r×(E·ω) 两条路径；[r]×ᵀ = −[r]×）
        let rxlv = spatial::cross(r, &[lv[0], lv[1], lv[2]]);
        let arg = [lw[0] - rxlv[0], lw[1] - rxlv[1], lw[2] - rxlv[2]];
        let gw = spatial::mat3_vec(&et, &arg);
        // g_r = (E·ω)×λv
        let wa = spatial::mat3_vec(e, &[w[0], w[1], w[2]]);
        let gr = spatial::cross(&wa, &[lv[0], lv[1], lv[2]]);
        // g_E = λω·ωᵀ + λv·vᵀ − (r×λv)·ωᵀ
        let mut ge = [S::zero(); 9];
        for i in 0..3 {
            for j in 0..3 {
                ge[i * 3 + j] = lw[i] * w[j] + lv[i] * v[j] - rxlv[i] * w[j];
            }
        }
        smallvec![
            ge[0], ge[1], ge[2], ge[3], ge[4], ge[5], ge[6], ge[7], ge[8], gr[0], gr[1], gr[2],
            gw[0], gw[1], gw[2], gv[0], gv[1], gv[2],
        ]
    }
    fn name(&self) -> &'static str {
        "plucker_motion"
    }
}

/// Plücker 力向量变换（B 系 → A 系）：
/// `f_A = (E·n_B + r×(E·f_B) ; E·f_B)`。
/// inputs = [E(9), r(3), f_B(6)]，outputs = [f_A(6)]。
#[derive(Clone, Copy)]
pub struct PluckerForce;

impl<S: Scalar> CustomOp<S> for PluckerForce {
    fn num_inputs(&self) -> usize {
        18
    }
    fn num_outputs(&self) -> usize {
        6
    }
    fn forward(&self, i: &[S]) -> (SmallVec<[S; 8]>, SmallVec<[S; 8]>) {
        let e: &[S; 9] = (&i[0..9]).try_into().unwrap();
        let r: &[S; 3] = (&i[9..12]).try_into().unwrap();
        let (n, f) = (&i[12..15], &i[15..18]);
        let fa = spatial::mat3_vec(e, &[f[0], f[1], f[2]]);
        let na = spatial::mat3_vec(e, &[n[0], n[1], n[2]]);
        // 力向量的转移项在力矩分量（与运动向量对偶：线速度分量拿 r×(E·ω)）
        let t = spatial::cross(r, &fa);
        (
            smallvec![
                na[0] + t[0],
                na[1] + t[1],
                na[2] + t[2],
                fa[0],
                fa[1],
                fa[2]
            ],
            i.iter().copied().collect(),
        )
    }
    fn backward(&self, r_in: &[S], go: &[S]) -> SmallVec<[S; 8]> {
        let e: &[S; 9] = (&r_in[0..9]).try_into().unwrap();
        let r: &[S; 3] = (&r_in[9..12]).try_into().unwrap();
        let (n, f) = (&r_in[12..15], &r_in[15..18]);
        let (ln, lf) = (&go[0..3], &go[3..6]);
        let et = spatial::mat3_t(e);
        let gn = spatial::mat3_vec(&et, &[ln[0], ln[1], ln[2]]);
        let rxln = spatial::cross(r, &[ln[0], ln[1], ln[2]]);
        let arg = [lf[0] - rxln[0], lf[1] - rxln[1], lf[2] - rxln[2]];
        let gf = spatial::mat3_vec(&et, &arg);
        let fa = spatial::mat3_vec(e, &[f[0], f[1], f[2]]);
        let gr = spatial::cross(&fa, &[ln[0], ln[1], ln[2]]);
        let mut ge = [S::zero(); 9];
        for i in 0..3 {
            for j in 0..3 {
                ge[i * 3 + j] = ln[i] * n[j] + lf[i] * f[j] - rxln[i] * f[j];
            }
        }
        smallvec![
            ge[0], ge[1], ge[2], ge[3], ge[4], ge[5], ge[6], ge[7], ge[8], gr[0], gr[1], gr[2],
            gn[0], gn[1], gn[2], gf[0], gf[1], gf[2],
        ]
    }
    fn name(&self) -> &'static str {
        "plucker_force"
    }
}

/// 空间惯性作用量：`f = I_O·v`（v = [ω; u]，u 为原点线速度）。
/// 动量共轭约定（与动能 `T = ½ωᵀĪω + ½m|u + ω×c|²` 一致，由不变性测试守护）：
/// `n = (Ī − m[c]×[c]×)ω + m(c×u)`；`f = m(u + ω×c)`。
/// inputs = [Ī(6 对称), m, c(3), v(6)]（16），outputs = [n(3), f(3)]（6）。
#[derive(Clone, Copy)]
pub struct InertiaApply;

impl<S: Scalar> CustomOp<S> for InertiaApply {
    fn num_inputs(&self) -> usize {
        16
    }
    fn num_outputs(&self) -> usize {
        6
    }
    fn forward(&self, i: &[S]) -> (SmallVec<[S; 8]>, SmallVec<[S; 8]>) {
        let ib = sym3::unpack(&[i[0], i[1], i[2], i[3], i[4], i[5]]);
        let m = i[6];
        let c: &[S; 3] = (&i[7..10]).try_into().unwrap();
        let (w, u) = (&i[10..13], &i[13..16]);
        let iw = spatial::mat3_vec(&ib, &[w[0], w[1], w[2]]);
        let ccw = spatial::cross(c, &spatial::cross(c, &[w[0], w[1], w[2]]));
        let cu = spatial::cross(c, &[u[0], u[1], u[2]]);
        let wxc = spatial::cross(&[w[0], w[1], w[2]], c);
        (
            smallvec![
                iw[0] - m * ccw[0] + m * cu[0],
                iw[1] - m * ccw[1] + m * cu[1],
                iw[2] - m * ccw[2] + m * cu[2],
                m * (u[0] + wxc[0]),
                m * (u[1] + wxc[1]),
                m * (u[2] + wxc[2]),
            ],
            i.iter().copied().collect(),
        )
    }
    fn backward(&self, r: &[S], go: &[S]) -> SmallVec<[S; 8]> {
        let ib = sym3::unpack(&[r[0], r[1], r[2], r[3], r[4], r[5]]);
        let m = r[6];
        let c: &[S; 3] = (&r[7..10]).try_into().unwrap();
        let (w, u) = (&r[10..13], &r[13..16]);
        let (ln, lf) = (&go[0..3], &go[3..6]);
        let cr = |x: &[S; 3], y: &[S; 3]| spatial::cross(x, y);
        let dot = |x: &[S], y: &[S]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];

        // g_ω = Īλn − m c×(c×λn) + m c×λf
        let iln = spatial::mat3_vec(&ib, &[ln[0], ln[1], ln[2]]);
        let ccln = cr(c, &cr(c, &[ln[0], ln[1], ln[2]]));
        let clf = cr(c, &[lf[0], lf[1], lf[2]]);
        let gw = [
            iln[0] - m * ccln[0] + m * clf[0],
            iln[1] - m * ccln[1] + m * clf[1],
            iln[2] - m * ccln[2] + m * clf[2],
        ];
        // g_u = m(λf − c×λn)
        let cln = cr(c, &[ln[0], ln[1], ln[2]]);
        let gu = [
            m * (lf[0] - cln[0]),
            m * (lf[1] - cln[1]),
            m * (lf[2] - cln[2]),
        ];
        // g_Ī（对称 6-打包）：每个打包分量控制矩阵两个对称位置 → 梯度 = 完整和
        let g_ixx = ln[0] * w[0];
        let g_iyy = ln[1] * w[1];
        let g_izz = ln[2] * w[2];
        let g_ixy = ln[0] * w[1] + ln[1] * w[0];
        let g_ixz = ln[0] * w[2] + ln[2] * w[0];
        let g_iyz = ln[1] * w[2] + ln[2] * w[1];
        // g_m = λn·(c×u − c×(c×ω)) + λf·(u + ω×c)
        let cu = cr(c, &[u[0], u[1], u[2]]);
        let ccw = cr(c, &cr(c, &[w[0], w[1], w[2]]));
        let wxc = cr(&[w[0], w[1], w[2]], c);
        let gm = dot(ln, &[cu[0] - ccw[0], cu[1] - ccw[1], cu[2] - ccw[2]])
            + dot(lf, &[u[0] + wxc[0], u[1] + wxc[1], u[2] + wxc[2]]);
        // g_c = −m[(c·ω)λn + (λn·c)ω − 2(λn·ω)c] + m(u×λn) + m(λf×ω)
        let cdw = dot(c, w);
        let lnc = dot(ln, c);
        let lnw = dot(ln, w);
        let uxln = cr(&[u[0], u[1], u[2]], &[ln[0], ln[1], ln[2]]);
        let lfxw = cr(&[lf[0], lf[1], lf[2]], &[w[0], w[1], w[2]]);
        let two = S::one() + S::one();
        let gc = [
            -m * (cdw * ln[0] + lnc * w[0] - two * lnw * c[0]) + m * (uxln[0] + lfxw[0]),
            -m * (cdw * ln[1] + lnc * w[1] - two * lnw * c[1]) + m * (uxln[1] + lfxw[1]),
            -m * (cdw * ln[2] + lnc * w[2] - two * lnw * c[2]) + m * (uxln[2] + lfxw[2]),
        ];
        smallvec![
            g_ixx, g_iyy, g_izz, g_ixy, g_ixz, g_iyz, gm, gc[0], gc[1], gc[2], gw[0], gw[1], gw[2],
            gu[0], gu[1], gu[2],
        ]
    }
    fn name(&self) -> &'static str {
        "inertia_apply"
    }
}

/// 空间惯性坐标系变换（B 系 → A 系），按 6×6 拼装实现：
/// `I_B = Lin(Ī_B, m, c) = [[Ī_B − m[c]×[c]×, m[c]×],[−m[c]×, m·1]]`；
/// `Y = X*·I_B·X*ᵀ`（X* = 力变换 `[[E, r×E],[0, E]]`）；
/// 提取：`m = Y₅₅`，`c_A = vee(Y_tr/m)`，`Ī_A = Y_A + m[c_A]×[c_A]×`。
/// inputs = [Ī_B(6), m, c(3), E(9), r(3)]（22），outputs = [Ī_A(6), m, c_A(3)]（10）。
#[derive(Clone, Copy)]
pub struct RotateInertia;

#[inline]
fn m6<S: Scalar>(a: &[S], i: usize, j: usize) -> S {
    a[i * 6 + j]
}

#[inline]
fn m6_set<S: Scalar>(a: &mut [S], i: usize, j: usize, v: S) {
    a[i * 6 + j] = v;
}

fn put3<S: Scalar>(a: &mut [S], r0: usize, c0: usize, b: &[S; 9]) {
    for i in 0..3 {
        for j in 0..3 {
            a[(r0 + i) * 6 + c0 + j] = b[i * 3 + j];
        }
    }
}

fn get3<S: Scalar>(a: &[S], r0: usize, c0: usize) -> [S; 9] {
    let mut b = [S::zero(); 9];
    for i in 0..3 {
        for j in 0..3 {
            b[i * 3 + j] = a[(r0 + i) * 6 + c0 + j];
        }
    }
    b
}

/// Y = X·M·Xᵀ（6×6）
fn conj6<S: Scalar>(x: &[S], m: &[S]) -> [S; 36] {
    let mut tmp = [S::zero(); 36];
    for i in 0..6 {
        for j in 0..6 {
            let mut acc = S::zero();
            for k in 0..6 {
                acc = acc + m6(x, i, k) * m6(m, k, j);
            }
            m6_set(&mut tmp, i, j, acc);
        }
    }
    let mut y = [S::zero(); 36];
    for i in 0..6 {
        for j in 0..6 {
            let mut acc = S::zero();
            for k in 0..6 {
                acc = acc + m6(&tmp, i, k) * m6(x, j, k);
            }
            m6_set(&mut y, i, j, acc);
        }
    }
    y
}

fn diag3<S: Scalar>(m: S) -> [S; 9] {
    let z = S::zero();
    [m, z, z, z, m, z, z, z, m]
}

impl<S: Scalar> CustomOp<S> for RotateInertia {
    fn num_inputs(&self) -> usize {
        22
    }
    fn num_outputs(&self) -> usize {
        10
    }
    fn forward(&self, i: &[S]) -> (SmallVec<[S; 8]>, SmallVec<[S; 8]>) {
        let ib = sym3::unpack(&[i[0], i[1], i[2], i[3], i[4], i[5]]);
        let m = i[6];
        let c: &[S; 3] = (&i[7..10]).try_into().unwrap();
        let e: &[S; 9] = (&i[10..19]).try_into().unwrap();
        let rv: &[S; 3] = (&i[19..22]).try_into().unwrap();

        let cc = spatial::mat3_mul(&spatial::skew(c), &spatial::skew(c));
        let mut a_blk = [S::zero(); 9];
        for k in 0..9 {
            a_blk[k] = ib[k] - m * cc[k];
        }
        let mut ib6 = [S::zero(); 36];
        let mb = spatial::skew(c);
        let mut b_blk = [S::zero(); 9];
        let mut bl_blk = [S::zero(); 9];
        for k in 0..9 {
            b_blk[k] = m * mb[k];
            bl_blk[k] = -(m * mb[k]);
        }
        put3(&mut ib6, 0, 0, &a_blk);
        put3(&mut ib6, 0, 3, &b_blk);
        put3(&mut ib6, 3, 0, &bl_blk);
        put3(&mut ib6, 3, 3, &diag3(m));
        let re = spatial::mat3_mul(&spatial::skew(rv), e);
        let mut xs = [S::zero(); 36];
        put3(&mut xs, 0, 0, e);
        put3(&mut xs, 0, 3, &re);
        put3(&mut xs, 3, 3, e);

        let y = conj6(&xs, &ib6);

        let y_tr = get3(&y, 0, 3);
        let ca = spatial::vee([
            y_tr[0] / m,
            y_tr[1] / m,
            y_tr[2] / m,
            y_tr[3] / m,
            y_tr[4] / m,
            y_tr[5] / m,
            y_tr[6] / m,
            y_tr[7] / m,
            y_tr[8] / m,
        ]);
        let y_a = get3(&y, 0, 0);
        let caca = spatial::mat3_mul(&spatial::skew(&ca), &spatial::skew(&ca));
        let mut ia = [S::zero(); 9];
        for k in 0..9 {
            ia[k] = y_a[k] + m * caca[k];
        }
        let mut out = SmallVec::new();
        out.extend(sym3::pack(&ia));
        out.push(m);
        out.extend(ca);
        (out, i.iter().copied().collect())
    }
    fn backward(&self, r: &[S], go: &[S]) -> SmallVec<[S; 8]> {
        let ib = sym3::unpack(&[r[0], r[1], r[2], r[3], r[4], r[5]]);
        let m = r[6];
        let c: &[S; 3] = (&r[7..10]).try_into().unwrap();
        let e: &[S; 9] = (&r[10..19]).try_into().unwrap();
        let rv: &[S; 3] = (&r[19..22]).try_into().unwrap();
        let lm = go[6];
        let lca: &[S; 3] = (&go[7..10]).try_into().unwrap();
        let two = S::one() + S::one();

        // ---- 前向量重建 ----
        let cc = spatial::mat3_mul(&spatial::skew(c), &spatial::skew(c));
        let mut a_blk = [S::zero(); 9];
        for k in 0..9 {
            a_blk[k] = ib[k] - m * cc[k];
        }
        let mut ib6 = [S::zero(); 36];
        let mb = spatial::skew(c);
        let mut b_blk = [S::zero(); 9];
        let mut bl_blk = [S::zero(); 9];
        for k in 0..9 {
            b_blk[k] = m * mb[k];
            bl_blk[k] = -(m * mb[k]);
        }
        put3(&mut ib6, 0, 0, &a_blk);
        put3(&mut ib6, 0, 3, &b_blk);
        put3(&mut ib6, 3, 0, &bl_blk);
        put3(&mut ib6, 3, 3, &diag3(m));
        let re = spatial::mat3_mul(&spatial::skew(rv), e);
        let mut xs = [S::zero(); 36];
        put3(&mut xs, 0, 0, e);
        put3(&mut xs, 0, 3, &re);
        put3(&mut xs, 3, 3, e);
        let y = conj6(&xs, &ib6);
        let y_tr = get3(&y, 0, 3);
        let ca = spatial::vee([
            y_tr[0] / m,
            y_tr[1] / m,
            y_tr[2] / m,
            y_tr[3] / m,
            y_tr[4] / m,
            y_tr[5] / m,
            y_tr[6] / m,
            y_tr[7] / m,
            y_tr[8] / m,
        ]);

        // ---- 提取段 VJP（严格镜像 forward 的实际读取） ----
        // forward 读取：pack(Ī_A) 只读 Y_A 上三角 [00,11,22,01,02,12]；
        // c_A = vee(Y_tr/m) 读 Y[1][5]、Y[0][5]、Y[0][4]；m 读 Y[5][5]。
        // caca 上三角：[c0²−|c|², c1²−|c|², c2²−|c|², c0c1, c0c2, c1c2]
        let mut ly = [S::zero(); 36]; // λY
        let cabsq = ca[0] * ca[0] + ca[1] * ca[1] + ca[2] * ca[2];
        let caca_up = [
            ca[0] * ca[0] - cabsq,
            ca[1] * ca[1] - cabsq,
            ca[2] * ca[2] - cabsq,
            ca[0] * ca[1],
            ca[0] * ca[2],
            ca[1] * ca[2],
        ];
        // ∂(m·caca_up)/∂c_A 与 ∂/∂m（caca 项）。
        // gm 累加提取段的 m 路径：caca 项 + c_A 缩放项（−g_ca·c_A/m）。
        // 直接输出 out[6] = 输入 m 的 lm 也在种子中计入。
        // 注意提取段不读 Y[5][5]，λY[5][5] 保持 0——若把 lm 放入 λY[5][5]，
        // 共轭段会产生虚假的 ∂Y55/∂E 依赖（Y55 = m 与 E 无关）。
        let mut g_ca = [S::zero(); 3];
        for i in 0..3 {
            g_ca[i] = g_ca[i] + lca[i];
        }
        let mut gm = lm;
        for k in 0..6 {
            // ∂pack_k(ia)/∂c_A · go[k]
            let d0 = match k {
                0 => S::zero(),
                1 => -(two * ca[0]),
                2 => -(two * ca[0]),
                3 => ca[1],
                4 => ca[2],
                _ => S::zero(),
            };
            let d1 = match k {
                0 => -(two * ca[1]),
                1 => S::zero(),
                2 => -(two * ca[1]),
                3 => ca[0],
                4 => S::zero(),
                _ => ca[2],
            };
            let d2 = match k {
                0 => -(two * ca[2]),
                1 => -(two * ca[2]),
                2 => S::zero(),
                3 => S::zero(),
                4 => ca[0],
                _ => ca[1],
            };
            g_ca[0] = g_ca[0] + m * go[k] * d0;
            g_ca[1] = g_ca[1] + m * go[k] * d1;
            g_ca[2] = g_ca[2] + m * go[k] * d2;
            // ∂pack_k(ia)/∂m = caca_up[k]
            gm = gm + go[k] * caca_up[k];
        }
        // c_A = vee(Y_tr/m)：λY 位置 [1][5]、[0][5]、[0][4]，系数用总 g_ca
        m6_set(&mut ly, 1, 5, -(g_ca[0] / m));
        m6_set(&mut ly, 0, 5, g_ca[1] / m);
        m6_set(&mut ly, 0, 4, -(g_ca[2] / m));
        gm = gm - (g_ca[0] * ca[0] + g_ca[1] * ca[1] + g_ca[2] * ca[2]) / m;
        // Y_A 上三角（pack 读取位置）
        m6_set(&mut ly, 0, 0, go[0]);
        m6_set(&mut ly, 1, 1, go[1]);
        m6_set(&mut ly, 2, 2, go[2]);
        m6_set(&mut ly, 0, 1, go[3]);
        m6_set(&mut ly, 0, 2, go[4]);
        m6_set(&mut ly, 1, 2, go[5]);
        // 注意：提取段不读 Y[5][5]（forward 的 out[6] = 输入 m，非 Y55），
        // λY[5][5] 必须为 0——若把 lm 放入 λY[5][5]，共轭段会产生虚假的
        // ∂Y55/∂E 依赖（Y55 = m 与 E 无关）。直接路径已在 gm 种子中计入。

        // ---- 共轭段：g_X = (λY + λYᵀ)·X*·I_B；g_I_B = X*ᵀ·λY·X* ----
        let mut lys = ly;
        for i in 0..6 {
            for j in 0..6 {
                let v = m6(&lys, i, j) + m6(&lys, j, i);
                m6_set(&mut lys, i, j, v);
            }
        }
        // xib = X*·I_B
        let mut xib = [S::zero(); 36];
        for i in 0..6 {
            for j in 0..6 {
                let mut acc = S::zero();
                for k in 0..6 {
                    acc = acc + m6(&xs, i, k) * m6(&ib6, k, j);
                }
                m6_set(&mut xib, i, j, acc);
            }
        }
        let mut gx = [S::zero(); 36];
        for i in 0..6 {
            for j in 0..6 {
                let mut acc = S::zero();
                for k in 0..6 {
                    acc = acc + m6(&lys, i, k) * m6(&xib, k, j);
                }
                m6_set(&mut gx, i, j, acc);
            }
        }
        // g_I_B = X*ᵀ·λY·X*
        let mut t1 = [S::zero(); 36];
        for i in 0..6 {
            for j in 0..6 {
                let mut acc = S::zero();
                for k in 0..6 {
                    acc = acc + m6(&xs, k, i) * m6(&ly, k, j);
                }
                m6_set(&mut t1, i, j, acc);
            }
        }
        let mut g6 = [S::zero(); 36];
        for i in 0..6 {
            for j in 0..6 {
                let mut acc = S::zero();
                for k in 0..6 {
                    acc = acc + m6(&t1, i, k) * m6(&xs, k, j);
                }
                m6_set(&mut g6, i, j, acc);
            }
        }
        let g_a = get3(&g6, 0, 0);
        let g_b = get3(&g6, 0, 3);
        let g_bl = get3(&g6, 3, 0);

        // ---- X* 结构段：E 在 TL/BR，TR = [r]×·E ----
        let gx_tl = get3(&gx, 0, 0);
        let gx_br = get3(&gx, 3, 3);
        let gx_tr = get3(&gx, 0, 3);
        // ⟨G, [r]×E⟩ 对 r：= ⟨G·Eᵀ, [r]×⟩ = r·vee_c(GEᵀ − (GEᵀ)ᵀ)
        let g = spatial::mat3_mul(&gx_tr, &spatial::mat3_t(e));
        let gmt = spatial::mat3_t(&g);
        let gr = spatial::vee([
            g[0] - gmt[0],
            g[1] - gmt[1],
            g[2] - gmt[2],
            g[3] - gmt[3],
            g[4] - gmt[4],
            g[5] - gmt[5],
            g[6] - gmt[6],
            g[7] - gmt[7],
            g[8] - gmt[8],
        ]);
        // 对 E：∂⟨G, [r]×E⟩/∂E_kj = Σ_i G_ij [r]×_ik = ([r]×ᵀ·G)_kj
        let rx = spatial::skew(rv);
        let rxt_g = spatial::mat3_mul(&spatial::mat3_t(&rx), &gx_tr);
        let mut ge = [S::zero(); 9];
        for k in 0..9 {
            ge[k] = gx_tl[k] + gx_br[k] + rxt_g[k];
        }

        // ---- Lin 段：Ī_B、m、c ----
        let gib = [
            g_a[0],
            g_a[4],
            g_a[8],
            g_a[1] + g_a[3],
            g_a[2] + g_a[6],
            g_a[5] + g_a[7],
        ];
        let cgac: S = (0..3)
            .map(|i| (0..3).map(|j| c[i] * g_a[i * 3 + j] * c[j]).fold(S::zero(), |a, v| a + v))
            .fold(S::zero(), |a, v| a + v);
        let tr_ga = g_a[0] + g_a[4] + g_a[8];
        let c2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
        gm = gm + tr_ga * c2 - cgac;
        let skew_coef = |mm: &[S; 9]| -> [S; 3] {
            let mt = spatial::mat3_t(mm);
            spatial::vee([
                mm[0] - mt[0],
                mm[1] - mt[1],
                mm[2] - mt[2],
                mm[3] - mt[3],
                mm[4] - mt[4],
                mm[5] - mt[5],
                mm[6] - mt[6],
                mm[7] - mt[7],
                mm[8] - mt[8],
            ])
        };
        let wb = skew_coef(&g_b);
        let wbl = skew_coef(&g_bl);
        gm = gm + (c[0] * wb[0] + c[1] * wb[1] + c[2] * wb[2]);
        gm = gm - (c[0] * wbl[0] + c[1] * wbl[1] + c[2] * wbl[2]);
        // D = m·I：∂⟨g_D, m·I⟩/∂m = tr(g_D)
        gm = gm + (g6[3 * 6 + 3] + g6[4 * 6 + 4] + g6[5 * 6 + 5]);
        let mut gc = [S::zero(); 3];
        for i in 0..3 {
            // ∂(cᵀg_Ac)/∂c = (g_A + g_Aᵀ)c —— g_A 不对称（λY 非对称），不能省转置
            let gac = (g_a[i * 3] + g_a[i]) * c[0]
                + (g_a[i * 3 + 1] + g_a[3 + i]) * c[1]
                + (g_a[i * 3 + 2] + g_a[6 + i]) * c[2];
            gc[i] = -(m * (gac - two * tr_ga * c[i])) + m * (wb[i] - wbl[i]);
        }

        smallvec![
            gib[0], gib[1], gib[2], gib[3], gib[4], gib[5], gm, gc[0], gc[1], gc[2], ge[0], ge[1],
            ge[2], ge[3], ge[4], ge[5], ge[6], ge[7], ge[8], gr[0], gr[1], gr[2],
        ]
    }
    fn name(&self) -> &'static str {
        "rotate_inertia"
    }
}

/// SO(3) 指数映射：`w`（旋量向量，3）→ `E`（3×3 行主序，9）。
/// Rodrigues：`R = cosθ·I + (1−cosθ)·uuᵀ + sinθ·[u]×`，`u = w/θ`。
#[derive(Clone, Copy)]
pub struct So3Exp;

impl<S: Scalar> CustomOp<S> for So3Exp {
    fn num_inputs(&self) -> usize {
        3
    }
    fn num_outputs(&self) -> usize {
        9
    }
    fn forward(&self, i: &[S]) -> (SmallVec<[S; 8]>, SmallVec<[S; 8]>) {
        let w: &[S; 3] = (&i[0..3]).try_into().unwrap();
        let r = spatial::so3_exp(w);
        (
            smallvec![r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]],
            smallvec![w[0], w[1], w[2]],
        )
    }
    fn backward(&self, r_in: &[S], go: &[S]) -> SmallVec<[S; 8]> {
        let w: &[S; 3] = (&r_in[0..3]).try_into().unwrap();
        let lam: &[S; 9] = (&go[0..9]).try_into().unwrap();
        let theta = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        if theta < c(1e-5) {
            // R ≈ I + [w]×：g_w = vee(λ − λᵀ)
            let mut lmlt = [S::zero(); 9];
            let lt = spatial::mat3_t(lam);
            for k in 0..9 {
                lmlt[k] = lam[k] - lt[k];
            }
            let v = spatial::vee(lmlt);
            return smallvec![v[0], v[1], v[2]];
        }
        let u = [w[0] / theta, w[1] / theta, w[2] / theta];
        let (s, cth) = theta.sin_cos();
        let r_mat = spatial::so3_exp(w);
        // g_θ = ⟨λ, [u]×R⟩
        let g_theta = spatial::so3_grad_theta(&u, &r_mat, lam);
        // g_u = (1−c)(λu + λᵀu) + s·vee(λ − λᵀ)
        let lu = [
            lam[0] * u[0] + lam[1] * u[1] + lam[2] * u[2],
            lam[3] * u[0] + lam[4] * u[1] + lam[5] * u[2],
            lam[6] * u[0] + lam[7] * u[1] + lam[8] * u[2],
        ];
        let ltu = [
            lam[0] * u[0] + lam[3] * u[1] + lam[6] * u[2],
            lam[1] * u[0] + lam[4] * u[1] + lam[7] * u[2],
            lam[2] * u[0] + lam[5] * u[1] + lam[8] * u[2],
        ];
        let mut lmlt = [S::zero(); 9];
        let lt = spatial::mat3_t(lam);
        for k in 0..9 {
            lmlt[k] = lam[k] - lt[k];
        }
        let vs = spatial::vee(lmlt);
        let one = S::one();
        let one_m_c = one - cth;
        let gu = [
            one_m_c * (lu[0] + ltu[0]) + s * vs[0],
            one_m_c * (lu[1] + ltu[1]) + s * vs[1],
            one_m_c * (lu[2] + ltu[2]) + s * vs[2],
        ];
        // g_w = g_θ·u + (I − uuᵀ)·g_u/θ
        let ugu = u[0] * gu[0] + u[1] * gu[1] + u[2] * gu[2];
        let mut gw = [S::zero(); 3];
        for i in 0..3 {
            gw[i] = g_theta * u[i] + (gu[i] - u[i] * ugu) / theta;
        }
        smallvec![gw[0], gw[1], gw[2]]
    }
    fn name(&self) -> &'static str {
        "so3_exp"
    }
}

/// 空间力叉积（对偶于 [`SpatialCrossMotion`]）：`out = v ×* m`。
/// v = [ω; v]、m = [n; f] → `out = (ω×n + v×f ; ω×f)`。
/// RNEA 的陀螺项 `v ×* (I·v)` 由 [`InertiaApply`]（动量）与本算子组合。
/// inputs = [v(6), m(6)]，outputs = [out(6)]。
#[derive(Clone, Copy)]
pub struct SpatialForceCross;

impl<S: Scalar> CustomOp<S> for SpatialForceCross {
    fn num_inputs(&self) -> usize {
        12
    }
    fn num_outputs(&self) -> usize {
        6
    }
    fn forward(&self, i: &[S]) -> (SmallVec<[S; 8]>, SmallVec<[S; 8]>) {
        let (w, v) = (&i[0..3], &i[3..6]);
        let (n, f) = (&i[6..9], &i[9..12]);
        let wn = spatial::cross(&[w[0], w[1], w[2]], &[n[0], n[1], n[2]]);
        let vf = spatial::cross(&[v[0], v[1], v[2]], &[f[0], f[1], f[2]]);
        let wf = spatial::cross(&[w[0], w[1], w[2]], &[f[0], f[1], f[2]]);
        (
            smallvec![
                wn[0] + vf[0],
                wn[1] + vf[1],
                wn[2] + vf[2],
                wf[0],
                wf[1],
                wf[2]
            ],
            i.iter().copied().collect(),
        )
    }
    // VJP（λ = [λn; λf]）：
    //   d(out_n) = dω×n + ω×dn + dv×f + v×df；d(out_f) = dω×f + ω×df
    //   → g_ω = n×λn + f×λf；g_v = f×λn；g_n = λn×ω；g_f = λn×v + λf×ω
    fn backward(&self, r: &[S], go: &[S]) -> SmallVec<[S; 8]> {
        let (w, v) = (&r[0..3], &r[3..6]);
        let (n, f) = (&r[6..9], &r[9..12]);
        let (ln, lf) = (&go[0..3], &go[3..6]);
        let cr = |x: &[S], y: &[S]| spatial::cross(&[x[0], x[1], x[2]], &[y[0], y[1], y[2]]);
        let nxln = cr(n, ln);
        let fxlf = cr(f, lf);
        let g_w = [nxln[0] + fxlf[0], nxln[1] + fxlf[1], nxln[2] + fxlf[2]];
        let g_v = cr(f, ln);
        let g_n = cr(ln, w);
        let lnxv = cr(ln, v);
        let lfxw = cr(lf, w);
        let g_f = [lnxv[0] + lfxw[0], lnxv[1] + lfxw[1], lnxv[2] + lfxw[2]];
        smallvec![
            g_w[0], g_w[1], g_w[2], g_v[0], g_v[1], g_v[2], g_n[0], g_n[1], g_n[2], g_f[0],
            g_f[1], g_f[2],
        ]
    }
    fn name(&self) -> &'static str {
        "spatial_force_cross"
    }
}
