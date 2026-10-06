//! 批量 rollout 并行梯度采样（设计文档 §4.1.6 的参考实现）。
//!
//! 并发模型：**单条 rollout = 单线程单 Context**（tape 拓扑序不变量要求入带
//! 串行）；批量场景（MPC 多假设采样 / RL 策略梯度 / 参数扫描）用 rayon 把
//! 独立的 rollout 派发到线程池——每个任务在线程内自建 `Context` + 仿真，
//! **不跨线程共享任何 AD 类型**（`AD`/`Context` 均 `!Send`，由编译期强制），
//! 只把普通数值（梯度数组）送回主线程聚合。
//!
//! 运行：`cargo run --release -p ad --example batch_rollout --features rayon`

use ad::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

/// 单个 rollout 任务：给定 (g, L) 参数，返回 1000 步单摆末态损失的
/// 梯度 [∂L/∂g, ∂L/∂L]（checkpoint 分段反向）。
/// 闭包只捕获普通数值——`Context` 在线程内自建（§4.1.6 的关键纪律）。
fn rollout_grad(g: f64, len: f64, steps: usize) -> [f64; 2] {
    let mut ctx = Context::<f64>::new();
    let (g_ad, vg) = ctx.var(g);
    let (l_ad, vl) = ctx.var(len);
    let mut sim = PendulumSim::new(&mut ctx, 0.5, 0.0, g_ad, l_ad, 0.01);
    let mut ckpt = CheckpointManager::new(CheckpointStrategy::Uniform { interval: 32 }, &sim);
    for t in 0..steps {
        ckpt.forward_step(&mut ctx, &mut sim, t);
    }
    let loss = |ctx: &mut Context<f64>, sim: &PendulumSim| {
        let th = sim.state()[0];
        let d = ctx.sub(th, AD::constant(1.0));
        ctx.mul(d, d)
    };
    ckpt.backward(&mut ctx, &mut sim, &loss);
    [ctx.grad(vg).unwrap(), ctx.grad(vl).unwrap()]
}

fn main() {
    let (n_g, n_l, steps) = (24usize, 24usize, 1000);
    let gs: Vec<f64> = (0..n_g).map(|i| 6.0 + 8.0 * i as f64 / (n_g - 1) as f64).collect();
    let ls: Vec<f64> = (0..n_l).map(|i| 0.5 + 1.2 * i as f64 / (n_l - 1) as f64).collect();

    let n = gs.len() * ls.len();
    println!("批量 rollout：{n} 个任务 × {steps} 步（rayon 线程池）");

    let t0 = Instant::now();
    let grads: Vec<(usize, usize, [f64; 2])> = gs
        .par_iter()
        .enumerate()
        .flat_map(|(i, &g)| {
            ls.par_iter()
                .enumerate()
                .map(move |(j, &l)| (i, j, rollout_grad(g, l, steps)))
        })
        .collect();
    let dt = t0.elapsed();

    // 聚合：梯度范数的均值/最大值（MPC 多假设筛选、参数敏感度扫描的典型输出）
    let mut sum_sq = 0.0;
    let mut max_norm = 0.0f64;
    for (_, _, g) in &grads {
        let norm = (g[0] * g[0] + g[1] * g[1]).sqrt();
        sum_sq += norm * norm;
        max_norm = max_norm.max(norm);
    }
    let rms = (sum_sq / n as f64).sqrt();
    println!(
        "完成 {n} 条：耗时 {dt:.1?}（{:.1} µs/条），‖∇L‖ RMS = {rms:.4}，max = {max_norm:.4}",
        dt.as_micros() as f64 / n as f64
    );
    // 抽样打印一个角落与中心
    let pick = |(i, j): (usize, usize)| {
        let (_, _, g) = grads[i * ls.len() + j];
        println!("  (g={:.2}, L={:.2}): ∂L/∂g = {:.5}, ∂L/∂L = {:.5}", gs[i], ls[j], g[0], g[1]);
    };
    pick((0, 0));
    pick((n_g / 2, n_l / 2));
}
