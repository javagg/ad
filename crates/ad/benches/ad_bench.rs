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
    drop(a);
    let _ = (vb, vc);
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

criterion_group!(
    benches,
    bench_scalar_expr_fresh,
    bench_scalar_expr_reuse,
    bench_pendulum_checkpoint
);
criterion_main!(benches);
