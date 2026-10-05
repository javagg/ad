//! 端到端示例（设计文档 §4.6 的可运行版本）：
//! 10⁴ 步单摆 rollout + checkpoint 分段反向 + 梯度健康度分析。
//!
//! 运行：cargo run -p ad --example pendulum_checkpoint --release

use ad::prelude::*;

const T: usize = 10_000;
const DT: f64 = 0.01;

fn main() {
    let mut ctx = Context::<f64>::new();

    // 1) 可微输入（叶子）：重力、摆长
    let (g_ad, vg) = ctx.var(9.81);
    let (l_ad, vl) = ctx.var(1.0);

    // 2) rollout：每步整体封装为自定义算子；checkpoint 每 100 步存一次快照
    let mut sim = PendulumSim::new(&mut ctx, 0.5, 0.0, g_ad, l_ad, DT);
    let mut ckpt = CheckpointManager::new(CheckpointStrategy::Uniform { interval: 100 }, &sim);
    for t in 0..T {
        ckpt.forward_step(&mut ctx, &mut sim, t);
    }
    println!("步数 = {T}，快照数 = {}", ckpt.num_snapshots());

    // 3) 分段反向：loss = (θ_T − π/2)²，返回初始状态伴随 ∂L/∂x₀
    let loss = |ctx: &mut Context<f64>, sim: &PendulumSim| {
        let th = sim.state()[0];
        let d = ctx.sub(th, AD::constant(std::f64::consts::FRAC_PI_2));
        ctx.mul(d, d)
    };
    let init_adj = ckpt.backward(&mut ctx, &mut sim, &loss);

    // 4) 梯度读取 + 健康度分析
    let grads = [ctx.grad(vg).unwrap(), ctx.grad(vl).unwrap()];
    println!("dL/dg = {:.6}", grads[0]);
    println!("dL/dL = {:.6}", grads[1]);
    println!("dL/dθ0 = {:.6}  dL/dω0 = {:.6}", init_adj[0], init_adj[1]);

    let health = GradientChecker::default().analyze_health(&grads);
    println!(
        "梯度健康度：‖g‖ = {:.3e}，零占比 {:.2}，非有限占比 {:.2}",
        health.norm, health.zero_fraction, health.nonfinite_fraction
    );
}
