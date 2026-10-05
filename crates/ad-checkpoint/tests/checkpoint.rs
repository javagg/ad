//! M3 验收（设计文档 §4.4 / §5.3）：
//! 1. 分段反向（checkpoint）与全 tape 反向的梯度一致；
//! 2. 快照间隔扰动不影响结果（策略无关性）；
//! 3. 快照预算受控（内存亚线性）；
//! 4. 确定性重算：同策略重跑结果逐位一致；
//! 5. 初始状态伴随 ∂L/∂x₀ 可获取。

use ad_checkpoint::{CheckpointManager, CheckpointStrategy, PendulumSim, Recomputable};
use ad_core::{Context, AD};

const T: usize = 200;
const DT: f64 = 0.05;
const THETA0: f64 = 0.5;
const OMEGA0: f64 = 0.0;
const G: f64 = 9.81;
const LEN: f64 = 1.3;

fn loss_of(ctx: &mut Context<f64>, sim: &PendulumSim) -> AD<f64> {
    let th = sim.state()[0];
    let d = ctx.sub(th, AD::constant(1.0));
    ctx.mul(d, d)
}

/// 全 tape 参考：正向全程入带 + 一次反向。
/// 返回 (∂L/∂g, ∂L/∂L, ∂L/∂θ₀, ∂L/∂ω₀)。
fn reference_grads() -> [f64; 4] {
    let mut ctx = Context::<f64>::new();
    let (g_ad, g_var) = ctx.var(G);
    let (l_ad, l_var) = ctx.var(LEN);
    let mut sim = PendulumSim::new(&mut ctx, THETA0, OMEGA0, g_ad, l_ad, DT);
    // 重新绑定以捕获初始状态叶子（构造时的绑定叶子成为孤立节点，无害）
    let init_state = sim.bind_state(&mut ctx);
    for _ in 0..T {
        sim.step(&mut ctx);
    }
    let loss = loss_of(&mut ctx, &sim);
    ctx.backward(loss);
    [
        ctx.grad(g_var).unwrap(),
        ctx.grad(l_var).unwrap(),
        ctx.grad_of(init_state[0]).unwrap(),
        ctx.grad_of(init_state[1]).unwrap(),
    ]
}

fn compare(name: &str, strategy: CheckpointStrategy) {
    let reference = reference_grads();
    let (grads, _snapshots) = run_checkpoint(strategy, name);
    for (i, (r, c)) in reference.iter().zip(grads.iter()).enumerate() {
        let scale = 1.0 + r.abs() + c.abs();
        assert!(
            (r - c).abs() <= 1e-10 * scale,
            "{}: grad[{}] mismatch: reference {} vs checkpoint {}",
            name,
            i,
            r,
            c
        );
    }
}

/// 实际执行的 checkpoint 流程（供 compare 使用）。
fn run_checkpoint(strategy: CheckpointStrategy, name: &str) -> ([f64; 4], usize) {
    let mut ctx = Context::<f64>::new();
    let (g_ad, g_var) = ctx.var(G);
    let (l_ad, l_var) = ctx.var(LEN);
    let mut sim = PendulumSim::new(&mut ctx, THETA0, OMEGA0, g_ad, l_ad, DT);
    let mut ckpt = CheckpointManager::new(strategy, &sim);

    for t in 0..T {
        ckpt.forward_step(&mut ctx, &mut sim, t);
    }
    let snapshots = ckpt.num_snapshots();
    let init_adj = ckpt.backward(&mut ctx, &mut sim, &loss_of);

    let grads = [
        ctx.grad(g_var).unwrap(),
        ctx.grad(l_var).unwrap(),
        init_adj[0],
        init_adj[1],
    ];
    let _ = name;
    (grads, snapshots)
}

#[test]
fn checkpoint_matches_full_tape_uniform() {
    compare("uniform-7", CheckpointStrategy::Uniform { interval: 7 });
}

#[test]
fn checkpoint_matches_full_tape_uniform_stepwise() {
    compare("uniform-1", CheckpointStrategy::Uniform { interval: 1 });
}

#[test]
fn checkpoint_matches_full_tape_online() {
    compare("online-5", CheckpointStrategy::Online { budget: 5 });
}

#[test]
fn checkpoint_matches_full_tape_custom() {
    // 每 13 步存一次（与 uniform 的 7 不同 → 策略无关性）
    compare(
        "custom-13",
        CheckpointStrategy::Custom(std::sync::Arc::new(|s| s % 13 == 0)),
    );
}

#[test]
fn snapshot_budget_respected() {
    // Uniform{7}：在 7, 14, ..., 196 处存 → 28 个
    let (_, n) = run_checkpoint(CheckpointStrategy::Uniform { interval: 7 }, "u7");
    assert_eq!(n, T / 7);
    // Online{5}：固定预算 5
    let (_, n) = run_checkpoint(CheckpointStrategy::Online { budget: 5 }, "o5");
    assert_eq!(n, 5);
}

#[test]
fn deterministic_rerun_bit_exact() {
    let a = run_checkpoint(CheckpointStrategy::Uniform { interval: 11 }, "d1").0;
    let b = run_checkpoint(CheckpointStrategy::Uniform { interval: 11 }, "d2").0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "grad[{}] not bit-exact across reruns",
            i
        );
    }
}

#[test]
fn no_checkpoint_falls_back_to_full_tape() {
    // 预算 0 / interval 0 → 单段全量重算（内存换正确性的退化路径）
    compare("none", CheckpointStrategy::Online { budget: 0 });
}

// ---- 嵌套反转（Nested 策略，设计文档 §4.4.2）----

/// 嵌套反转运行。返回 (∂L/∂g, ∂L/∂L, ∂L/∂θ₀, ∂L/∂ω₀, 反向阶段重算步数)。
fn run_nested(budget: usize) -> ([f64; 4], usize) {
    let mut ctx = Context::<f64>::new();
    let (g_ad, g_var) = ctx.var(G);
    let (l_ad, l_var) = ctx.var(LEN);
    let mut sim = PendulumSim::new(&mut ctx, THETA0, OMEGA0, g_ad, l_ad, DT);
    let mut ckpt = CheckpointManager::new(CheckpointStrategy::Nested { budget }, &sim);

    for t in 0..T {
        ckpt.forward_step(&mut ctx, &mut sim, t);
    }
    assert_eq!(ckpt.num_snapshots(), 0, "Nested 前向不存快照");
    let before = sim.steps_executed;
    let init_adj = ckpt.backward(&mut ctx, &mut sim, &loss_of);
    let recompute = sim.steps_executed - before;

    let grads = [
        ctx.grad(g_var).unwrap(),
        ctx.grad(l_var).unwrap(),
        init_adj[0],
        init_adj[1],
    ];
    (grads, recompute)
}

#[test]
fn nested_matches_full_tape() {
    let reference = reference_grads();
    for budget in [1usize, 2, 3, 6] {
        let (grads, _) = run_nested(budget);
        for (i, (r, c)) in reference.iter().zip(grads.iter()).enumerate() {
            let scale = 1.0 + r.abs() + c.abs();
            assert!(
                (r - c).abs() <= 1e-10 * scale,
                "nested budget={budget}: grad[{i}] reference {r} vs got {c}"
            );
        }
    }
}

#[test]
fn nested_recompute_within_bound() {
    // 重算上界：T(n, m) = ⌈n/2⌉ + T(⌊n/2⌋, m-1) + T(⌈n/2⌉, m-1)，T(n, 0) = n
    // → T(200, 2) = 400，T(200, 3) = 500，T(200, 6) ≈ 800
    for (budget, bound) in [(2usize, 400), (3, 500), (6, 812)] {
        let (_, recompute) = run_nested(budget);
        assert!(
            recompute <= bound,
            "budget={budget}: recompute {recompute} > bound {bound}"
        );
        assert!(recompute >= T, "budget={budget}: recompute {recompute} < T");
    }
}

#[test]
fn nested_budget_zero_degenerates_to_full_tape() {
    let reference = reference_grads();
    let (grads, recompute) = run_nested(0);
    for (i, (r, c)) in reference.iter().zip(grads.iter()).enumerate() {
        let scale = 1.0 + r.abs() + c.abs();
        assert!(
            (r - c).abs() <= 1e-10 * scale,
            "grad[{i}] reference {r} vs got {c}"
        );
    }
    // budget=0 → 单窗口全量：重算恰好 = T
    assert_eq!(recompute, T);
}
