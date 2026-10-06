//! M5 基准体系（设计文档 §5.4）。
//! 运行：cargo bench -p ad
//!
//! 指标对照 §5.4 目标表：
//! - scalar_expr：前向 + 反向微基准（相对纯 f64 的开销可由对比推导）；
//! - pendulum：整步动力学 CustomOp 路径（含 checkpoint 分段反向）。

use ad::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

/// y = sin(a·b + c)：每轮独立 Context（含分配、记录、反向、销毁）
fn bench_scalar_expr_fresh(c: &mut Criterion) {
    c.bench_function("scalar_expr/fresh_context", |b| {
        b.iter(|| {
            let mut ctx = Context::<f64>::new();
            let (a, _) = ctx.var(black_box(1.1));
            let (bb, _) = ctx.var(black_box(2.2));
            let (cc, _) = ctx.var(black_box(3.3));
            let ab = ctx.mul(a, bb);
            let s = ctx.add(ab, cc);
            let y = ad::sin_with(&mut ctx, s);
            ctx.backward(y);
            y.value
        })
    });
}

/// 同一表达式，复用 Context + clear_tape（优化循环的目标形态）
fn bench_scalar_expr_reuse(c: &mut Criterion) {
    let guard = Context::<f64>::new().enter();
    let (a, va) = guard.var(1.1);
    let (bb, vb) = guard.var(2.2);
    let (cc, vc) = guard.var(3.3);
    c.bench_function("scalar_expr/reuse_clear_tape", |b| {
        b.iter(|| {
            let y = ad::sin(a * bb + cc);
            guard.backward(y);
            let g = guard.grad(va).unwrap() + guard.grad(vb).unwrap() + guard.grad(vc).unwrap();
            guard.zero_grads();
            guard.clear_tape();
            g
        })
    });
    let _ = (a, va, vb, vc);
}

/// 1000 步单摆：no_grad 前向 + 快照 + 分段反向（含逐段重算）
fn bench_pendulum_checkpoint(c: &mut Criterion) {
    let mut group = c.benchmark_group("pendulum");
    group.throughput(criterion::Throughput::Elements(1000));

    group.bench_function("T=1000 fwd+segmented_bwd", |b| {
        b.iter(|| {
            let mut ctx = Context::<f64>::new();
            let (g_ad, vg) = ctx.var(9.81);
            let (l_ad, vl) = ctx.var(1.0);
            let mut sim = PendulumSim::new(&mut ctx, 0.5, 0.0, g_ad, l_ad, 0.01);
            let mut ckpt =
                CheckpointManager::new(CheckpointStrategy::Uniform { interval: 32 }, &sim);
            for t in 0..1000 {
                ckpt.forward_step(&mut ctx, &mut sim, t);
            }
            let loss = |ctx: &mut Context<f64>, sim: &PendulumSim| {
                let th = sim.state()[0];
                let d = ctx.sub(th, AD::constant(1.0));
                ctx.mul(d, d)
            };
            ckpt.backward(&mut ctx, &mut sim, &loss);
            ctx.grad(vg).unwrap() + ctx.grad(vl).unwrap()
        })
    });

    // 对照：同长度全 tape（不 checkpoint）
    group.bench_function("T=1000 fwd+full_tape_bwd", |b| {
        b.iter(|| {
            let mut ctx = Context::<f64>::new();
            let (g_ad, vg) = ctx.var(9.81);
            let (l_ad, vl) = ctx.var(1.0);
            let mut sim = PendulumSim::new(&mut ctx, 0.5, 0.0, g_ad, l_ad, 0.01);
            for _ in 0..1000 {
                sim.step(&mut ctx);
            }
            let th = sim.state()[0];
            let d = ctx.sub(th, AD::constant(1.0));
            let loss = ctx.mul(d, d);
            ctx.backward(loss);
            ctx.grad(vg).unwrap() + ctx.grad(vl).unwrap()
        })
    });

    group.finish();
}

/// §5.4 目标表"CustomOp 主路径梯度 vs 手写梯度 ≤ 2×"的验收基准：
/// 同一双关节摆步进（`DoublePendulumStep`）的两种求梯度方式——
/// (a) `op_path_1k`：1000 步 recorded rollout + tape 反向（库框架完整路径）；
/// (b) `handwritten_1k`：同一 VJP 闭式 + 手写伴随递推（λ 保持在局部变量，
///     无 tape / 无节点分配 / 无伴随数组）——"手写解析梯度"基线。
/// 另附单发形态（`op_path_once` vs `handwritten_once`）作参考：其比值被
/// Context/叶子创建的一次性开销主导，非本目标的稳态语义。
fn bench_chain_step(c: &mut Criterion) {
    use ad::{CustomOp, DoublePendulumStep};
    use std::rc::Rc;

    let op = DoublePendulumStep::default();
    let op_rc: Rc<dyn CustomOp<f64>> = Rc::new(op);
    let dt = 0.002f64;
    let st0 = [0.3f64, -0.2, 0.1, 0.4]; // [θ1,θ2,ω1,ω2]
    const T: usize = 1000;

    let mut group = c.benchmark_group("chain_step");

    // (a) 框架路径：1000 步全 tape + 一次反向（损失 ½Σx_T²）
    group.bench_function("op_path_1k", |b| {
        b.iter(|| {
            let mut ctx = Context::<f64>::new();
            let (t1, vt1) = ctx.var(0.2);
            let (t2, vt2) = ctx.var(-0.1);
            let mut st: smallvec::SmallVec<[AD<f64>; 4]> =
                st0.iter().map(|&v| ctx.var(v).0).collect();
            let mut inputs = Vec::with_capacity(7);
            for _ in 0..T {
                inputs.clear();
                inputs.extend_from_slice(&st);
                inputs.push(t1);
                inputs.push(t2);
                inputs.push(AD::constant(dt));
                st = ctx.call_custom(DoublePendulumStep::default(), &inputs);
            }
            let mut loss = ctx.mul(st[0], st[0]);
            for o in &st[1..] {
                let sq = ctx.mul(*o, *o);
                loss = ctx.add(loss, sq);
            }
            ctx.backward(loss);
            let g: f64 = ctx.grad(vt1).unwrap() + ctx.grad(vt2).unwrap();
            black_box(g)
        })
    });

    // (b) 手写基线：同一 forward/backward 闭式，伴随递推手写（无 tape）
    group.bench_function("handwritten_1k", |b| {
        b.iter(|| {
            let mut st = st0;
            let mut residuals: Vec<smallvec::SmallVec<[f64; 8]>> = Vec::with_capacity(T);
            for _ in 0..T {
                let (outs, residual) = op_rc.forward(&[
                    st[0], st[1], st[2], st[3], 0.2, -0.1, dt,
                ]);
                st = [outs[0], outs[1], outs[2], outs[3]];
                residuals.push(residual);
            }
            // λ 递推：λ_T = ∇½Σx² = x_T；λ_t = backward(residual_t, λ_{t+1}) 的状态分量
            let mut lam = [st[0], st[1], st[2], st[3]];
            let mut g_tau = 0.0f64;
            for r in residuals.iter().rev() {
                let gins = op_rc.backward(r, &lam);
                lam = [gins[0], gins[1], gins[2], gins[3]];
                g_tau += gins[4] + gins[5];
            }
            black_box(g_tau + lam.iter().sum::<f64>())
        })
    });

    // 参考：单发形态（比值含 Context/叶子创建的一次性开销）
    let x = [0.3f64, -0.2, 0.1, 0.4, 0.2, -0.1, dt];
    group.bench_function("op_path_once", |b| {
        b.iter(|| {
            let mut ctx = Context::<f64>::new();
            let mut vars = Vec::with_capacity(7);
            let mut inputs = Vec::with_capacity(7);
            for &v in x.iter() {
                let (ad, var) = ctx.var(v);
                inputs.push(ad);
                vars.push(var);
            }
            let outs = ctx.call_custom(DoublePendulumStep::default(), &inputs);
            let seeds: Vec<(AD<f64>, f64)> = outs.iter().map(|&o| (o, 1.0)).collect();
            ctx.backward_seeds(&seeds);
            let g: f64 = vars.iter().map(|&v| ctx.grad(v).unwrap()).sum();
            black_box(g)
        })
    });
    group.bench_function("handwritten_once", |b| {
        b.iter(|| {
            let (outs, residual) = op_rc.forward(&x);
            let go: smallvec::SmallVec<[f64; 8]> = outs.iter().map(|_| 1.0).collect();
            let gins = op_rc.backward(&residual, &go);
            black_box(gins.iter().sum::<f64>())
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_scalar_expr_fresh,
    bench_scalar_expr_reuse,
    bench_pendulum_checkpoint,
    bench_chain_step
);
criterion_main!(benches);
