//! bulk 向量算子（设计文档 §4.3.4）：1 条 tape 记录 + 梯度与逐元素展开一致。
//! n = 7 特意超过 SmallVec 内联容量，覆盖堆溢出路径。

use ad_core::{Context, AD};
use ad_ops::{axpy_with, dot_with, norm2_with};

const N: usize = 7;

fn vec_vars(ctx: &mut Context<f64>, vals: &[f64]) -> (Vec<AD<f64>>, Vec<ad_core::Variable>) {
    let mut ads = Vec::with_capacity(vals.len());
    let mut vars = Vec::with_capacity(vals.len());
    for &v in vals {
        let (ad, var) = ctx.var(v);
        ads.push(ad);
        vars.push(var);
    }
    (ads, vars)
}

fn read_all(ctx: &Context<f64>, vars: &[ad_core::Variable]) -> Vec<f64> {
    vars.iter().map(|&v| ctx.grad(v).unwrap()).collect()
}

#[test]
fn dot_matches_elementwise_expansion() {
    let a: Vec<f64> = (0..N).map(|i| 0.5 + i as f64 * 0.3).collect();
    let b: Vec<f64> = (0..N).map(|i| 1.0 - i as f64 * 0.11).collect();

    let mut ctx = Context::<f64>::new();
    let (ad_a, va) = vec_vars(&mut ctx, &a);
    let (ad_b, vb) = vec_vars(&mut ctx, &b);
    let y = dot_with(&mut ctx, &ad_a, &ad_b);
    assert_eq!(ctx.tape_len(), 1, "bulk 算子应只占 1 条记录");
    ctx.backward(y);
    let ga = read_all(&ctx, &va);
    let gb = read_all(&ctx, &vb);

    // 参考实现：逐元素展开
    let mut ctx2 = Context::<f64>::new();
    let (ad_a2, va2) = vec_vars(&mut ctx2, &a);
    let (ad_b2, vb2) = vec_vars(&mut ctx2, &b);
    let mut acc = ctx2.mul(ad_a2[0], ad_b2[0]);
    for i in 1..N {
        let prod = ctx2.mul(ad_a2[i], ad_b2[i]);
        acc = ctx2.add(acc, prod);
    }
    ctx2.backward(acc);
    for i in 0..N {
        assert!(
            (ga[i] - ctx2.grad(va2[i]).unwrap()).abs() < 1e-12,
            "da[{}]",
            i
        );
        assert!(
            (gb[i] - ctx2.grad(vb2[i]).unwrap()).abs() < 1e-12,
            "db[{}]",
            i
        );
    }
}

#[test]
fn axpy_matches_elementwise_expansion() {
    let alpha = 2.5;
    let x: Vec<f64> = (0..N).map(|i| i as f64 + 1.0).collect();
    let y0: Vec<f64> = (0..N).map(|i| -(i as f64) * 0.7).collect();

    let mut ctx = Context::<f64>::new();
    let (al, val) = ctx.var(alpha);
    let (ad_x, vx) = vec_vars(&mut ctx, &x);
    let (ad_y, vy) = vec_vars(&mut ctx, &y0);
    let out = axpy_with(&mut ctx, al, &ad_x, &ad_y);
    assert_eq!(ctx.tape_len(), 1);
    // loss = Σ outᵢ²
    let mut loss = ctx.mul(out[0], out[0]);
    for o in &out[1..] {
        let sq = ctx.mul(*o, *o);
        loss = ctx.add(loss, sq);
    }
    ctx.backward(loss);

    let mut ctx2 = Context::<f64>::new();
    let (al2, val2) = ctx2.var(alpha);
    let (ad_x2, vx2) = vec_vars(&mut ctx2, &x);
    let (ad_y2, vy2) = vec_vars(&mut ctx2, &y0);
    let mut loss2 = AD::constant(0.0);
    for i in 0..N {
        let prod = ctx2.mul(al2, ad_x2[i]);
        let o = ctx2.add(prod, ad_y2[i]);
        let sq = ctx2.mul(o, o);
        loss2 = ctx2.add(loss2, sq);
    }
    ctx2.backward(loss2);

    assert!((ctx.grad(val).unwrap() - ctx2.grad(val2).unwrap()).abs() < 1e-12);
    for i in 0..N {
        assert!(
            (read_all(&ctx, &vx)[i] - ctx2.grad(vx2[i]).unwrap()).abs() < 1e-12,
            "dx[{}]",
            i
        );
        assert!(
            (read_all(&ctx, &vy)[i] - ctx2.grad(vy2[i]).unwrap()).abs() < 1e-12,
            "dy[{}]",
            i
        );
    }
}

#[test]
fn norm2_matches_reference() {
    let x: Vec<f64> = (0..N).map(|i| (i as f64 - 3.0) * 0.9).collect();

    let mut ctx = Context::<f64>::new();
    let (ad_x, vx) = vec_vars(&mut ctx, &x);
    let n = norm2_with(&mut ctx, &ad_x);
    assert_eq!(ctx.tape_len(), 1);
    ctx.backward(n);
    let g = read_all(&ctx, &vx);

    // 解析：d‖x‖/dxᵢ = xᵢ/‖x‖
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    for i in 0..N {
        assert!((g[i] - x[i] / norm).abs() < 1e-12, "i={}", i);
    }
}

#[test]
fn bulk_ops_with_constant_inputs_degrade_gracefully() {
    // 全常量输入 → 不入带，返回常量
    let mut ctx = Context::<f64>::new();
    let ca: Vec<AD<f64>> = (0..4).map(|k: i32| AD::constant(k as f64)).collect();
    let cb: Vec<AD<f64>> = (0..4).map(|k: i32| AD::constant(k as f64)).collect();
    let d = dot_with(&mut ctx, &ca, &cb);
    assert_eq!(ctx.tape_len(), 0);
    assert!(!d.is_tracked());
}

// ============================================================ matvec（§4.3.4 O(n²) bulk）

use ad_ops::matvec_with;

#[test]
fn matvec_matches_elementwise_expansion() {
    // 3×4 行主序矩阵 × 4 维向量（12 个 M 元素 > SmallVec 内联，覆盖溢出路径）
    const R: usize = 3;
    const C: usize = 4;
    let m: Vec<f64> = (0..R * C).map(|k| (k as f64 * 0.37 - 1.2).sin()).collect();
    let v: Vec<f64> = (0..C).map(|j| 0.8 - j as f64 * 0.25).collect();

    let mut ctx = Context::<f64>::new();
    let (ad_m, vm) = vec_vars(&mut ctx, &m);
    let (ad_v, vv) = vec_vars(&mut ctx, &v);
    let y = matvec_with(&mut ctx, &ad_m, &ad_v);
    assert_eq!(ctx.tape_len(), 1, "matvec 应只占 1 条记录");
    // loss = Σ yᵢ²（让全部 rows 的伴随非零）
    let mut loss = ctx.mul(y[0], y[0]);
    for o in &y[1..] {
        let sq = ctx.mul(*o, *o);
        loss = ctx.add(loss, sq);
    }
    ctx.backward(loss);
    let gm = read_all(&ctx, &vm);
    let gv = read_all(&ctx, &vv);

    // 参考实现：逐元素展开 yᵢ = Σⱼ Mᵢⱼ·vⱼ
    let mut ctx2 = Context::<f64>::new();
    let (ad_m2, vm2) = vec_vars(&mut ctx2, &m);
    let (ad_v2, vv2) = vec_vars(&mut ctx2, &v);
    let mut loss2 = AD::constant(0.0);
    for i in 0..R {
        let mut acc = ctx2.mul(ad_m2[i * C], ad_v2[0]);
        for j in 1..C {
            let term = ctx2.mul(ad_m2[i * C + j], ad_v2[j]);
            acc = ctx2.add(acc, term);
        }
        let sq = ctx2.mul(acc, acc);
        loss2 = ctx2.add(loss2, sq);
    }
    ctx2.backward(loss2);

    for k in 0..R * C {
        assert!(
            (gm[k] - ctx2.grad(vm2[k]).unwrap()).abs() < 1e-12,
            "dM[{k}]"
        );
    }
    for j in 0..C {
        assert!(
            (gv[j] - ctx2.grad(vv2[j]).unwrap()).abs() < 1e-12,
            "dv[{j}]"
        );
    }
}

#[test]
fn matvec_forward_values_match_reference() {
    // 前向数值与朴素乘法一致（防 forward 索引错位——FD 隔离器之外的直接断言）
    const R: usize = 4;
    const C: usize = 3;
    let m: Vec<f64> = (0..R * C).map(|k| (k % 5) as f64 - 2.0).collect();
    let v: Vec<f64> = (0..C).map(|j| (j as f64 + 1.0) * 0.5).collect();

    let mut ctx = Context::<f64>::new();
    let (ad_m, _) = vec_vars(&mut ctx, &m);
    let (ad_v, _) = vec_vars(&mut ctx, &v);
    let y = matvec_with(&mut ctx, &ad_m, &ad_v);
    for i in 0..R {
        let want: f64 = (0..C).map(|j| m[i * C + j] * v[j]).sum();
        assert!((y[i].value - want).abs() < 1e-14, "y[{i}]");
    }
}

#[test]
fn matvec_constant_matrix_still_tracks_vector() {
    // M 为常量、v 被追踪：只走 v 的梯度路径（部分追踪的混合形态）
    let m = [1.0, 2.0, 3.0, 4.0]; // 2×2 行主序
    let v = vec![0.5, -1.5];

    let mut ctx = Context::<f64>::new();
    let ad_m: Vec<AD<f64>> = m.iter().map(|&x| AD::constant(x)).collect();
    let (ad_v, vv) = vec_vars(&mut ctx, &v);
    let y = matvec_with(&mut ctx, &ad_m, &ad_v);
    let loss = ctx.add(y[0], y[1]);
    ctx.backward(loss);
    // d(y0+y1)/dv = [1+3, 2+4]
    assert!((ctx.grad(vv[0]).unwrap() - 4.0).abs() < 1e-12);
    assert!((ctx.grad(vv[1]).unwrap() - 6.0).abs() < 1e-12);
}


// ============================================================ solve_sym（§4.3.4 线性求解行）

use ad_core::CustomOp;
use ad_ops::solve_sym_with;

/// 已知系统 + 解析梯度：M=[[4,1],[1,3]]，b=[9,10]
#[test]
fn solve_sym_forward_and_analytic_grads() {
    let m = vec![4.0, 1.0, 1.0, 3.0];
    let b = vec![9.0, 10.0];
    let mut ctx = Context::<f64>::new();
    let (ad_m, vm) = vec_vars(&mut ctx, &m);
    let (ad_b, vb) = vec_vars(&mut ctx, &b);
    let x = solve_sym_with(&mut ctx, &ad_m, &ad_b);
    for i in 0..2 {
        let r: f64 = (0..2).map(|j| m[i * 2 + j] * x[j].value).sum();
        assert!((r - b[i]).abs() < 1e-12, "residual row {i}");
    }
    assert_eq!(ctx.tape_len(), 1, "solve_sym 应只占 1 条记录");

    // loss = x[0]+x[1] → λb = M⁻¹λ、λM_ij = −z_i·x_j（z = M⁻¹λ）
    let loss = ctx.add(x[0], x[1]);
    ctx.backward(loss);
    let det = 4.0 * 3.0 - 1.0;
    let minv = [[3.0 / det, -1.0 / det], [-1.0 / det, 4.0 / det]];
    for j in 0..2 {
        let want: f64 = minv[0][j] + minv[1][j];
        assert!((ctx.grad(vb[j]).unwrap() - want).abs() < 1e-12, "db[{j}]");
    }
    let z = [minv[0][0] + minv[1][0], minv[0][1] + minv[1][1]];
    for (idx, (i, j)) in [(0usize, (0usize, 0usize)), (1, (0, 1)), (2, (1, 0)), (3, (1, 1))] {
        let want = -(z[i] * x[j].value);
        assert!((ctx.grad(vm[idx]).unwrap() - want).abs() < 1e-12, "dM[{i}][{j}]");
    }
}

/// 泛型验证器直接对拍 solve_sym 的 VJP（f64 与 f32 双标量）
#[test]
fn solve_sym_passes_validator_both_scalars() {
    use std::rc::Rc;

    let pts64 = vec![vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0, 1.0, 2.0, 3.0]];
    let op64: Rc<dyn CustomOp<f64>> = Rc::new(ad_ops::SolveSymOp { n: 3 });
    let report = ad_verify::op_check::validate_custom_op(op64, &pts64, 1e-6, 1e-5);
    assert!(report.passed, "solve_sym f64:\n{report}");

    let pts32: Vec<Vec<f32>> = pts64
        .clone()
        .into_iter()
        .map(|p| p.into_iter().map(|v| v as f32).collect())
        .collect();
    let op32: Rc<dyn CustomOp<f32>> = Rc::new(ad_ops::SolveSymOp { n: 3 });
    let report = ad_verify::op_check::validate_custom_op(op32, &pts32, 1e-3, 5e-3);
    assert!(report.passed, "solve_sym f32:\n{report}");
}
