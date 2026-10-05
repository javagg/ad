//! RotateInertia forward 的物理先验验证：动能坐标系不变性。
//! T = ½ωᵀĪω + ½m|u + ω×c|² 在运动向量变换 v_A = X_motion·v_B 下不变，
//! 且 (Ī_A, c_A) 恰为使该不变性成立的坐标变换结果。

use ad_core::CustomOp;
use ad_physics::{spatial, RotateInertia};

fn kinetic_energy(i6: &[f64; 6], m: f64, c: &[f64; 3], v: &[f64; 6]) -> f64 {
    let w: [f64; 3] = v[0..3].try_into().unwrap();
    let u: [f64; 3] = v[3..6].try_into().unwrap();
    let iw = spatial::mat3_vec(&spatial::sym3::unpack(i6), &w);
    let wxc = spatial::cross(&w, c);
    let uu = [u[0] + wxc[0], u[1] + wxc[1], u[2] + wxc[2]];
    0.5 * (w[0] * iw[0] + w[1] * iw[1] + w[2] * iw[2])
        + 0.5 * m * (uu[0] * uu[0] + uu[1] * uu[1] + uu[2] * uu[2])
}

#[test]
fn check_forward_semantics() {
    let e = spatial::so3_exp(&[0.4, 0.6, -0.9]);
    let m = 1.2f64;
    let c: [f64; 3] = [0.3, -0.2, 0.5];
    let r: [f64; 3] = [0.4, -0.3, 0.9];
    let ib: [f64; 6] = [2.0, 3.0, 1.5, 0.1, -0.2, 0.15];
    let x = [ib.as_slice(), &[m], &c, e.as_slice(), &r].concat();
    let (out, _) = RotateInertia.forward(&x);

    // c_A = E·c + r
    let ec = spatial::mat3_vec(&e, &c);
    let ca: [f64; 3] = [out[7], out[8], out[9]];
    assert!((ca[0] - (ec[0] + r[0])).abs() < 1e-12);
    assert!((ca[1] - (ec[1] + r[1])).abs() < 1e-12);
    assert!((ca[2] - (ec[2] + r[2])).abs() < 1e-12);
    assert_eq!(out[6], m);

    // 动能不变性：½v_Aᵀ I_A v_A = ½v_Bᵀ I_B v_B，v_A = X_motion v_B
    let vb: [f64; 6] = [0.7, -0.4, 0.2, -0.3, 0.5, 1.0];
    let t_b = kinetic_energy(&ib, m, &c, &vb);

    let wa = spatial::mat3_vec(&e, &[vb[0], vb[1], vb[2]]);
    let ua = spatial::mat3_vec(&e, &[vb[3], vb[4], vb[5]]);
    let rxe = spatial::mat3_mul(&spatial::skew(&r), &e);
    let rxwa = spatial::mat3_vec(&rxe, &[vb[0], vb[1], vb[2]]);
    let va: [f64; 6] = [
        wa[0],
        wa[1],
        wa[2],
        ua[0] + rxwa[0],
        ua[1] + rxwa[1],
        ua[2] + rxwa[2],
    ];
    let ia6: [f64; 6] = out[0..6].try_into().unwrap();
    let t_a = kinetic_energy(&ia6, out[6], &ca, &va);
    assert!(
        (t_b - t_a).abs() < 1e-12,
        "kinetic energy not invariant: T_B {t_b} vs T_A {t_a}"
    );
}

/// 动能坐标系不变性（跨算子约定一致性守护，设计文档 §4.3.1）：
/// I_A = X*·I_B·X*ᵀ 且 v_A = X·v_B ⟹ ½v_AᵀI_Av_A = ½v_BᵀI_Bv_B。
/// 该测试能抓住 motion/force 变换、惯性块布局、[c]× 符号等跨算子不一致。
#[test]
fn kinetic_energy_invariance_under_transform() {
    let e = spatial::so3_exp(&[0.4, 0.6, -0.9]);
    let m = 1.2f64;
    let c: [f64; 3] = [0.3, -0.2, 0.5];
    let r: [f64; 3] = [0.4, -0.3, 0.9];
    let ib: [f64; 6] = [2.0, 3.0, 1.5, 0.1, -0.2, 0.15];
    let x = [ib.as_slice(), &[m], &c, e.as_slice(), &r].concat();

    // RotInertia 前向：c_A = E·c + r，Ī_A 由 6×6 共轭给出
    let (out, _) = RotateInertia.forward(&x);
    let ca: [f64; 3] = [out[7], out[8], out[9]];
    let ec = spatial::mat3_vec(&e, &c);
    for i in 0..3 {
        assert!((ca[i] - (ec[i] + r[i])).abs() < 1e-12, "c_A mismatch");
    }

    let vb: [f64; 6] = [0.7, -0.4, 0.2, -0.3, 0.5, 1.0];
    let t = |i6: &[f64; 6], mm: f64, cc: &[f64; 3], v: &[f64; 6]| -> f64 {
        let w: [f64; 3] = v[0..3].try_into().unwrap();
        let u: [f64; 3] = v[3..6].try_into().unwrap();
        let iw = spatial::mat3_vec(&spatial::sym3::unpack(i6), &w);
        let wxc = spatial::cross(&w, cc);
        let uu = [u[0] + wxc[0], u[1] + wxc[1], u[2] + wxc[2]];
        0.5 * (w[0] * iw[0] + w[1] * iw[1] + w[2] * iw[2])
            + 0.5 * mm * (uu[0] * uu[0] + uu[1] * uu[1] + uu[2] * uu[2])
    };
    let t_b = t(&ib, m, &c, &vb);

    // v_A = X_motion·v_B（ω_A = Eω_B；u_A = E u_B + r×(E ω_B)）
    let wa = spatial::mat3_vec(&e, &[vb[0], vb[1], vb[2]]);
    let ua = spatial::mat3_vec(&e, &[vb[3], vb[4], vb[5]]);
    let rxe = spatial::mat3_mul(&spatial::skew(&r), &e);
    let rxwa = spatial::mat3_vec(&rxe, &[vb[0], vb[1], vb[2]]);
    let va: [f64; 6] = [
        wa[0],
        wa[1],
        wa[2],
        ua[0] + rxwa[0],
        ua[1] + rxwa[1],
        ua[2] + rxwa[2],
    ];
    let ia6: [f64; 6] = out[0..6].try_into().unwrap();
    let t_a = t(&ia6, out[6], &ca, &va);
    assert!(
        (t_b - t_a).abs() < 1e-12,
        "kinetic energy not invariant: T_B {t_b} vs T_A {t_a}"
    );
}
