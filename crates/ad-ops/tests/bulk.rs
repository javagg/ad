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
