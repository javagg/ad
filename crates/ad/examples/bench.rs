//! 简易基准（criterion 体系化为 M5 任务，此处提供快速量级参考）。
//! 运行：cargo run -p ad --example bench --release

use ad::prelude::*;
use std::time::Instant;

fn bench_scalar_expr() {
    // y = sin(a·b + c) 的前向 + 反向
    const N: usize = 100_000;
    let guard = Context::<f64>::new().enter();
    let (a, _va) = guard.var(1.1);
    let (b, _vb) = guard.var(2.2);
    let (c, _vc) = guard.var(3.3);

    let t0 = Instant::now();
    for _ in 0..N {
        let y = ad::sin(a * b + c);
        guard.backward(y);
        guard.zero_grads();
        guard.clear_tape();
    }
    let dt = t0.elapsed();
    println!(
        "scalar expr (sin(a*b+c)): {:?}/iter fwd+bwd ({} iters)",
        dt / N as u32,
        N
    );
}

fn bench_checkpoint_rollout() {
    // 10⁴ 步单摆：no_grad 前向 + 分段反向
    const T: usize = 10_000;
    let mut ctx = Context::<f64>::new();
    let (g_ad, vg) = ctx.var(9.81);
    let (l_ad, vl) = ctx.var(1.0);
    let mut sim = PendulumSim::new(&mut ctx, 0.5, 0.0, g_ad, l_ad, 0.01);
    let mut ckpt = CheckpointManager::new(CheckpointStrategy::Uniform { interval: 100 }, &sim);

    let t0 = Instant::now();
    for t in 0..T {
        ckpt.forward_step(&mut ctx, &mut sim, t);
    }
    let t_fwd = t0.elapsed();

    let loss = |ctx: &mut Context<f64>, sim: &PendulumSim| {
        let th = sim.state()[0];
        let d = ctx.sub(th, AD::constant(1.0));
        ctx.mul(d, d)
    };
    let t1 = Instant::now();
    let _ = ckpt.backward(&mut ctx, &mut sim, &loss);
    let t_bwd = t1.elapsed();

    println!(
        "pendulum T={}: fwd(no_grad+ckpt) {:?}，segmented bwd {:?}（含各段重算）",
        T, t_fwd, t_bwd
    );
    println!(
        "  梯度抽查：dL/dg = {:.3e}，dL/dL = {:.3e}，快照数 = {}",
        ctx.grad(vg).unwrap(),
        ctx.grad(vl).unwrap(),
        ckpt.num_snapshots()
    );
}

fn main() {
    bench_scalar_expr();
    bench_checkpoint_rollout();
}
