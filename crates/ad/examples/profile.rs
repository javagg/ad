//! 性能剖析用例（M5 火焰图工程的负载驱动）。
//!
//! 两个用途：
//! 1. **CPU 采样**：`samply record target/release/examples/profile.exe`（或 perf）——
//!    各阶段串行执行、迭代数悬殊，采样火焰图按符号归因；
//! 2. **分配画像**：内置计数分配器按阶段打印 分配次数/字节/峰值（不依赖 dhat，
//!    与 ad-core/tests/leak_*.rs 的全局分配器同一机制）。
//!
//! 运行：`cargo run --release -p ad --example profile`

use ad::prelude::*;
use std::hint::black_box;
use std::time::Instant;

// ---- 计数分配器（分配画像） ----

#[derive(Default)]
struct AllocStats {
    live_bytes: usize,
    peak_bytes: usize,
    allocs: usize,
    bytes_total: usize,
}

static STATS: std::sync::atomic::AtomicPtr<AllocStats> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

struct CountingAlloc;

unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let ptr = std::alloc::System.alloc(layout);
        if !ptr.is_null() {
            let stats = STATS.load(std::sync::atomic::Ordering::Relaxed);
            if !stats.is_null() {
                unsafe {
                    (*stats).live_bytes += layout.size();
                    (*stats).peak_bytes = (*stats).peak_bytes.max((*stats).live_bytes);
                    (*stats).allocs += 1;
                    (*stats).bytes_total += layout.size();
                }
            }
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        let stats = STATS.load(std::sync::atomic::Ordering::Relaxed);
        if !stats.is_null() {
            unsafe {
                (*stats).live_bytes -= layout.size();
            }
        }
        std::alloc::System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        let stats = STATS.load(std::sync::atomic::Ordering::Relaxed);
        let new_ptr = std::alloc::System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() && !stats.is_null() {
            unsafe {
                (*stats).live_bytes = (*stats).live_bytes + new_size - layout.size();
                (*stats).peak_bytes = (*stats).peak_bytes.max((*stats).live_bytes);
                (*stats).allocs += 1;
                (*stats).bytes_total += new_size;
            }
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// 单摆损失（与 bench 一致）
fn pendulum_loss(ctx: &mut Context<f64>, sim: &PendulumSim) -> AD<f64> {
    let th = sim.state()[0];
    let d = ctx.sub(th, AD::constant(1.0));
    ctx.mul(d, d)
}

fn phase_scalar_fresh(n: usize) {
    let mut stats = AllocStats::default();
    STATS.store(&mut stats, std::sync::atomic::Ordering::Relaxed);
    let t0 = Instant::now();
    for _ in 0..n {
        let mut ctx = Context::<f64>::new();
        let (a, _) = ctx.var(black_box(1.1));
        let (b, _) = ctx.var(black_box(2.2));
        let (c, _) = ctx.var(black_box(3.3));
        let ab = ctx.mul(a, b);
        let s = ctx.add(ab, c);
        let y = ad::sin_with(&mut ctx, s);
        ctx.backward(y);
        black_box(y.value);
    }
    let dt = t0.elapsed();
    STATS.store(std::ptr::null_mut(), std::sync::atomic::Ordering::Relaxed);
    println!(
        "scalar_fresh        {:>7} iters  {:>10.1?}  {:>7.2} ns/iter  allocs/iter {:>6.1}  bytes/iter {:>8.1}  peak {:>8}",
        n,
        dt,
        dt.as_nanos() as f64 / n as f64,
        stats.allocs as f64 / n as f64,
        stats.bytes_total as f64 / n as f64,
        stats.peak_bytes
    );
}

fn phase_scalar_reuse(n: usize) {
    let guard = Context::<f64>::new().enter();
    let (a, va) = guard.var(1.1);
    let (b, vb) = guard.var(2.2);
    let (c, vc) = guard.var(3.3);
    // 预热一次，建立 Vec 容量（复用形态的稳态）
    {
        let y = ad::sin(a * b + c);
        guard.backward(y);
        guard.zero_grads();
        guard.clear_tape();
    }
    let mut stats = AllocStats::default();
    STATS.store(&mut stats, std::sync::atomic::Ordering::Relaxed);
    let t0 = Instant::now();
    for _ in 0..n {
        let y = ad::sin(a * b + c);
        guard.backward(y);
        black_box(
            guard.grad(va).unwrap() + guard.grad(vb).unwrap() + guard.grad(vc).unwrap(),
        );
        guard.zero_grads();
        guard.clear_tape();
    }
    let dt = t0.elapsed();
    STATS.store(std::ptr::null_mut(), std::sync::atomic::Ordering::Relaxed);
    println!(
        "scalar_reuse        {:>7} iters  {:>10.1?}  {:>7.2} ns/iter  allocs/iter {:>6.1}  bytes/iter {:>8.1}",
        n,
        dt,
        dt.as_nanos() as f64 / n as f64,
        stats.allocs as f64 / n as f64,
        stats.bytes_total as f64 / n as f64
    );
}

fn run_checkpoint(ctx: &mut Context<f64>, dt_alloc: bool) {
    let (g_ad, vg) = ctx.var(9.81);
    let (l_ad, vl) = ctx.var(1.0);
    let mut sim = PendulumSim::new(ctx, 0.5, 0.0, g_ad, l_ad, 0.01);
    let mut ckpt = CheckpointManager::new(CheckpointStrategy::Uniform { interval: 32 }, &sim);
    for t in 0..1000 {
        ckpt.forward_step(ctx, &mut sim, t);
    }
    ckpt.backward(ctx, &mut sim, &pendulum_loss);
    black_box(ctx.grad(vg).unwrap() + ctx.grad(vl).unwrap());
    if dt_alloc {
        ctx.clear_tape();
    }
}

fn phase_pendulum_checkpoint(n: usize) {
    let mut stats = AllocStats::default();
    STATS.store(&mut stats, std::sync::atomic::Ordering::Relaxed);
    let t0 = Instant::now();
    for _ in 0..n {
        let mut ctx = Context::<f64>::new();
        run_checkpoint(&mut ctx, false);
    }
    let dt = t0.elapsed();
    STATS.store(std::ptr::null_mut(), std::sync::atomic::Ordering::Relaxed);
    println!(
        "pendulum_checkpoint {:>7} runs   {:>10.1?}  {:>9.1} us/run   allocs/run  {:>6.1}  bytes/run  {:>8.1}",
        n,
        dt,
        dt.as_secs_f64() * 1e6 / n as f64,
        stats.allocs as f64 / n as f64,
        stats.bytes_total as f64 / n as f64
    );
}

fn phase_pendulum_fulltape(n: usize) {
    let mut stats = AllocStats::default();
    STATS.store(&mut stats, std::sync::atomic::Ordering::Relaxed);
    let t0 = Instant::now();
    for _ in 0..n {
        let mut ctx = Context::<f64>::new();
        let (g_ad, vg) = ctx.var(9.81);
        let (l_ad, vl) = ctx.var(1.0);
        let mut sim = PendulumSim::new(&mut ctx, 0.5, 0.0, g_ad, l_ad, 0.01);
        for _ in 0..1000 {
            sim.step(&mut ctx);
        }
        let loss = pendulum_loss(&mut ctx, &sim);
        ctx.backward(loss);
        black_box(ctx.grad(vg).unwrap() + ctx.grad(vl).unwrap());
    }
    let dt = t0.elapsed();
    STATS.store(std::ptr::null_mut(), std::sync::atomic::Ordering::Relaxed);
    println!(
        "pendulum_fulltape   {:>7} runs   {:>10.1?}  {:>9.1} us/run   allocs/run  {:>6.1}  bytes/run  {:>8.1}",
        n,
        dt,
        dt.as_secs_f64() * 1e6 / n as f64,
        stats.allocs as f64 / n as f64,
        stats.bytes_total as f64 / n as f64
    );
}

fn main() {
    // --loop <秒>：长时间循环所有阶段（供 cdb 采样 poor-man's profiler 使用）
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--loop") {
        let secs: u64 = args.get(pos + 1).and_then(|s| s.parse().ok()).unwrap_or(30);
        println!("looping for {secs}s (pid sample target)");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
        while std::time::Instant::now() < deadline {
            phase_scalar_fresh(50_000);
            phase_scalar_reuse(50_000);
            phase_pendulum_checkpoint(100);
            phase_pendulum_fulltape(100);
        }
        return;
    }

    let n_scalar = 200_000usize;
    let n_pendulum = 300usize;
    println!("=== 分配画像（计数分配器）+ 墙钟 ===");
    println!("--- 先各跑一遍把缓存/分配器预热排除 ---");
    phase_scalar_reuse(10_000);
    phase_pendulum_checkpoint(5);
    println!("--- 正式测量 ---");
    phase_scalar_fresh(n_scalar);
    phase_scalar_reuse(n_scalar);
    phase_pendulum_checkpoint(n_pendulum);
    phase_pendulum_fulltape(n_pendulum);
}
