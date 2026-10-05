//! 物理算子验收（设计文档 M2 验收标准）：
//! PendulumStep（手工 backward 的 CustomOp）vs 同一递推的 AD 逐元素展开。
//! 两者梯度必须一致——这是"物理算子局部 Jacobian 推导错误"风险的直接防线。

use ad_checkpoint::{PendulumSim, Recomputable};
use ad_core::{Context, AD};
use ad_ops::sin_with;

const STEPS: usize = 50;
const DT: f64 = 0.03;
const P0: [f64; 4] = [0.4, -0.2, 9.81, 1.1]; // θ₀, ω₀, g, L

fn loss(ctx: &mut Context<f64>, th: AD<f64>, om: AD<f64>) -> AD<f64> {
    let s = ctx.add(th, om);
    ctx.mul(s, s) // (θ + ω)²，同时使用两个输出（扇出）
}

#[test]
fn pendulum_step_matches_ad_passthrough() {
    // --- CustomOp 路径（手工 backward） ---
    let mut ctx = Context::<f64>::new();
    let (g_ad, vg) = ctx.var(P0[2]);
    let (l_ad, vl) = ctx.var(P0[3]);
    let mut sim = PendulumSim::new(&mut ctx, P0[0], P0[1], g_ad, l_ad, DT);
    let init = sim.bind_state(&mut ctx);
    for _ in 0..STEPS {
        sim.step(&mut ctx);
    }
    let (th, om) = (sim.state()[0], sim.state()[1]);
    let lc = loss(&mut ctx, th, om);
    ctx.backward(lc);
    let g_custom = [
        ctx.grad_of(init[0]).unwrap(),
        ctx.grad_of(init[1]).unwrap(),
        ctx.grad(vg).unwrap(),
        ctx.grad(vl).unwrap(),
    ];

    // --- AD 逐元素展开路径（自动微分参考） ---
    let mut ctx2 = Context::<f64>::new();
    let (th0, vth0) = ctx2.var(P0[0]);
    let (om0, vom0) = ctx2.var(P0[1]);
    let (g2, vg2) = ctx2.var(P0[2]);
    let (l2, vl2) = ctx2.var(P0[3]);
    let dt = AD::constant(DT);
    let mut th = th0;
    let mut om = om0;
    for _ in 0..STEPS {
        // θ' = θ + dt·ω
        let dth = ctx2.mul(dt, om);
        let th1 = ctx2.add(th, dth);
        // ω' = ω - dt·(g/L)·sin(θ)
        let ratio = ctx2.div(g2, l2);
        let s = sin_with(&mut ctx2, th);
        let ks = ctx2.mul(ratio, s);
        let dks = ctx2.mul(dt, ks);
        let om1 = ctx2.sub(om, dks);
        th = th1;
        om = om1;
    }
    let lr = loss(&mut ctx2, th, om);
    ctx2.backward(lr);
    let g_ref = [
        ctx2.grad(vth0).unwrap(),
        ctx2.grad(vom0).unwrap(),
        ctx2.grad(vg2).unwrap(),
        ctx2.grad(vl2).unwrap(),
    ];

    for (i, (c, r)) in g_custom.iter().zip(g_ref.iter()).enumerate() {
        assert!(
            (c - r).abs() <= 1e-10 * (1.0 + c.abs() + r.abs()),
            "grad[{}] custom {} vs passthrough {}",
            i,
            c,
            r
        );
    }
}
