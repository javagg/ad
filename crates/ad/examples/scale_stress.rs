//! 规模实证（设计文档 §12.3 第 37 条）：10²–10³ 自由度、单步 1 个 CustomOp、
//! 10⁴ 步 + Uniform checkpoint 的定量压力曲线——"粗粒度 CustomOp + checkpoint
//! = 引擎可扩展"主张的数据支撑。
//!
//! 两个动力学家族：
//! - **稀疏链**（O(n)/步）：N 体弹簧链，N ∈ {100, 500, 2000}（状态 2N 维），
//!   T = 10⁴ 步，走**真实 CheckpointManager 分段反向**（引擎接入形态）；
//! - **稠密全耦合**（O(n²)/步）：N 体全对耦合，N ∈ {50, 100, 200}，T = 10³ 步
//!   全 tape——§4.3.4 "O(n²) 必须 bulk" 的定量印证。
//!
//! 指标：前向/反向墙钟、分配次数与峰值内存（计数分配器）、梯度有限性。
//! 运行：`cargo run --release -p ad --example scale_stress`

use ad::prelude::*;
use ad::{Context, Recomputable, AD};
use smallvec::SmallVec;
use std::hint::black_box;
use std::rc::Rc;
use std::time::Instant;

// ---- 计数分配器（与 profile.rs 同机制，Box 固定统计块） ----

#[derive(Default, Clone)]
struct AllocStats {
    live_bytes: usize,
    peak_bytes: usize,
    allocs: usize,
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
            }
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn stats_begin() -> *mut AllocStats {
    let s = Box::into_raw(Box::new(AllocStats::default()));
    STATS.store(s, std::sync::atomic::Ordering::Relaxed);
    s
}

fn stats_end(ptr: *mut AllocStats) -> AllocStats {
    STATS.store(std::ptr::null_mut(), std::sync::atomic::Ordering::Relaxed);
    unsafe { *Box::from_raw(ptr) }
}

// ============================================================ 两个动力学算子

fn lap(x: &[f64], j: usize) -> f64 {
    let left = if j > 0 { x[j - 1] } else { 0.0 };
    let right = if j + 1 < x.len() { x[j + 1] } else { 0.0 };
    2.0 * x[j] - left - right
}

/// 稀疏链单步（O(n)/步，1 条记录）：inputs = [q..., v..., k, dt]。
/// 手写 VJP 已由 scenarios.rs 的同构算子（ChainStep）验证——此处仅放大 n。
struct SparseChainStep {
    n: usize,
}

impl CustomOp<f64> for SparseChainStep {
    fn num_inputs(&self) -> usize {
        2 * self.n + 2
    }
    fn num_outputs(&self) -> usize {
        2 * self.n
    }
    fn forward(&self, i: &[f64]) -> (SmallVec<[f64; 8]>, SmallVec<[f64; 8]>) {
        let n = self.n;
        let (k, dt) = (i[2 * n], i[2 * n + 1]);
        let (x, v) = (&i[..n], &i[n..2 * n]);
        let mut outs = SmallVec::new();
        let mut vp = vec![0.0f64; n];
        for j in 0..n {
            let a = -k * lap(x, j);
            vp[j] = v[j] + dt * a;
            outs.push(x[j] + dt * vp[j]);
        }
        outs.extend_from_slice(&vp);
        (outs, i.iter().copied().collect())
    }
    fn backward(&self, r: &[f64], go: &[f64]) -> SmallVec<[f64; 8]> {
        let n = self.n;
        let (k, dt) = (r[2 * n], r[2 * n + 1]);
        let (x, v) = (&r[..n], &r[n..2 * n]);
        let (lx, lv) = (&go[..n], &go[n..2 * n]);
        let lv_acc: Vec<f64> = (0..n).map(|j| lv[j] + dt * lx[j]).collect();
        let mut grads = SmallVec::new();
        for j in 0..n {
            grads.push(lx[j] - k * dt * lap(&lv_acc, j));
        }
        for j in 0..n {
            grads.push(dt * lx[j] + lv[j]);
        }
        grads.push(-dt * (0..n).map(|j| lv_acc[j] * lap(x, j)).sum::<f64>());
        let mut acc_dt = 0.0;
        for j in 0..n {
            let a = -k * lap(x, j);
            let vp = v[j] + dt * a;
            acc_dt += lx[j] * vp + lv_acc[j] * a;
        }
        grads.push(acc_dt);
        grads
    }
    fn name(&self) -> &'static str {
        "sparse_chain_step"
    }
}

/// 稠密全耦合单步（O(n²)/步，1 条记录）：a_i = −(k/n)·Σ_{j≠i} (x_i − x_j)。
struct DenseCoupleStep {
    n: usize,
}

impl CustomOp<f64> for DenseCoupleStep {
    fn num_inputs(&self) -> usize {
        2 * self.n + 2
    }
    fn num_outputs(&self) -> usize {
        2 * self.n
    }
    fn forward(&self, i: &[f64]) -> (SmallVec<[f64; 8]>, SmallVec<[f64; 8]>) {
        let n = self.n;
        let (k, dt) = (i[2 * n], i[2 * n + 1]);
        let (x, v) = (&i[..n], &i[n..2 * n]);
        let total: f64 = x.iter().sum();
        let mut outs = SmallVec::new();
        let mut vp = vec![0.0f64; n];
        for j in 0..n {
            // Σ_{j'≠j} (x_j − x_{j'}) = n·x_j − total
            let a = -(k / n as f64) * (n as f64 * x[j] - total);
            vp[j] = v[j] + dt * a;
            outs.push(x[j] + dt * vp[j]);
        }
        outs.extend_from_slice(&vp);
        (outs, i.iter().copied().collect())
    }
    fn backward(&self, r: &[f64], go: &[f64]) -> SmallVec<[f64; 8]> {
        let n = self.n;
        let (k, dt) = (r[2 * n], r[2 * n + 1]);
        let (x, v) = (&r[..n], &r[n..2 * n]);
        let (lx, lv) = (&go[..n], &go[n..2 * n]);
        let lv_acc: Vec<f64> = (0..n).map(|j| lv[j] + dt * lx[j]).collect();
        // a_p = −(k/n)·(n·x_p − total) → ∂a_p/∂x_q = −k·δ_pq + k/n；
        // ∂v'/∂x = dt·∂a/∂x（gx 必须带 dt 因子）
        let mut gx = vec![0.0f64; n];
        for q in 0..n {
            let mut acc = 0.0;
            for p in 0..n {
                let dap_dq = if p == q { -k + k / n as f64 } else { k / n as f64 };
                acc += dap_dq * lv_acc[p];
            }
            gx[q] = lx[q] + dt * acc;
        }
        let mut grads = SmallVec::new();
        grads.extend_from_slice(&gx);
        for j in 0..n {
            grads.push(dt * lx[j] + lv[j]);
        }
        // λk、λdt：∂a_j/∂k = −(n·x_j − total)/n；λdt 经 vp 与 a 两条路
        let total: f64 = x.iter().sum();
        let mut gk = 0.0;
        let mut gdt = 0.0;
        for j in 0..n {
            let a = -(k / n as f64) * (n as f64 * x[j] - total);
            let vp = v[j] + dt * a;
            gk += lv_acc[j] * (-(n as f64 * x[j] - total) / n as f64);
            gdt += lx[j] * vp + lv_acc[j] * a;
        }
        grads.push(dt * gk);
        grads.push(gdt);
        grads
    }
    fn name(&self) -> &'static str {
        "dense_couple_step"
    }
}

// ============================================================ Recomputable 封装（稀疏链走真实 checkpoint）

struct ChainSim {
    op: Rc<dyn CustomOp<f64>>,
    #[allow(dead_code)]
    n: usize,
    dt: f64,
    k: AD<f64>,
    x: Vec<f64>,
    state_ad: Vec<AD<f64>>,
    op_name: &'static str,
}

impl ChainSim {
    fn new(op: Rc<dyn CustomOp<f64>>, op_name: &'static str, n: usize, dt: f64, k: AD<f64>, x: Vec<f64>, ctx: &mut Context<f64>) -> Self {
        let mut sim = ChainSim {
            op,
            n,
            dt,
            k,
            x,
            state_ad: Vec::new(),
            op_name,
        };
        sim.bind_state(ctx);
        sim
    }
}

impl Recomputable for ChainSim {
    type State = Vec<f64>;
    fn save_state(&self) -> Vec<f64> {
        self.x.clone()
    }
    fn load_state(&mut self, s: &Vec<f64>) {
        self.x = s.clone();
    }
    fn bind_state(&mut self, ctx: &mut Context<f64>) -> Vec<AD<f64>> {
        self.state_ad = self.x.iter().map(|&v| ctx.var(v).0).collect();
        self.state_ad.clone()
    }
    fn state(&self) -> &[AD<f64>] {
        &self.state_ad
    }
    fn step(&mut self, ctx: &mut Context<f64>) {
        let mut inp: Vec<AD<f64>> = self.state_ad.clone();
        inp.push(self.k);
        inp.push(AD::constant(self.dt));
        let outs = ctx.call_custom_dyn(Rc::clone(&self.op), self.op_name, &inp);
        for (i, o) in outs.iter().enumerate() {
            self.x[i] = o.value;
        }
        self.state_ad = outs.to_vec();
    }
}

// ============================================================ 压力运行

struct StressResult {
    label: String,
    state_dim: usize,
    steps: usize,
    fwd_ms: f64,
    bwd_ms: f64,
    allocs: usize,
    peak_mib: f64,
    grad_ok: bool,
}

fn bench_sparse(n: usize, t: usize) -> StressResult {
    let dt = 0.02f64;
    let init: Vec<f64> = (0..2 * n)
        .map(|i| 0.3 * ((i * 7 + 3) % 13) as f64 / 13.0 - 0.15)
        .collect();

    let stat = stats_begin();
    let t0 = Instant::now();
    let mut ctx = Context::<f64>::new();
    let (k_ad, vk) = ctx.var(2.5);
    let mut sim = ChainSim::new(
        Rc::new(SparseChainStep { n }),
        "sparse_chain_step",
        n,
        dt,
        k_ad,
        init.clone(),
        &mut ctx,
    );
    let mut ckpt = CheckpointManager::new(CheckpointStrategy::Uniform { interval: 100 }, &sim);
    for s in 0..t {
        ckpt.forward_step(&mut ctx, &mut sim, s);
    }
    let fwd = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let init_adj = ckpt.backward(&mut ctx, &mut sim, &|ctx: &mut Context<f64>, sim: &ChainSim| {
        let mut l = AD::constant(0.0);
        for xi in &sim.state()[..n] {
            let sq = ctx.mul(*xi, *xi);
            l = ctx.add(l, sq);
        }
        ctx.mul(AD::constant(0.5), l)
    });
    let bwd = t1.elapsed().as_secs_f64() * 1e3;
    let s = stats_end(stat);
    let gk = ctx.grad(vk).unwrap();

    StressResult {
        label: format!("sparse N={n}"),
        state_dim: 2 * n,
        steps: t,
        fwd_ms: fwd,
        bwd_ms: bwd,
        allocs: s.allocs,
        peak_mib: s.peak_bytes as f64 / (1024.0 * 1024.0),
        grad_ok: init_adj.iter().all(|g| g.is_finite()) && gk.is_finite(),
    }
}

fn bench_dense(n: usize, t: usize) -> StressResult {
    let dt = 0.02f64;
    let init: Vec<f64> = (0..2 * n)
        .map(|i| 0.3 * ((i * 5 + 1) % 11) as f64 / 11.0 - 0.15)
        .collect();

    let stat = stats_begin();
    let t0 = Instant::now();
    let mut ctx = Context::<f64>::new();
    let (k_ad, _) = ctx.var(0.5);
    let op: Rc<dyn CustomOp<f64>> = Rc::new(DenseCoupleStep { n });
    let leaves: Vec<AD<f64>> = init.iter().map(|&v| ctx.var(v).0).collect();
    // 全 tape recorded rollout——第一步消费叶子（梯度经重算路径回传）
    let mut inp: Vec<AD<f64>> = Vec::new();
    inp.extend(leaves.iter().copied());
    inp.push(k_ad);
    inp.push(AD::constant(dt));
    let mut outs = ctx.call_custom_dyn(Rc::clone(&op), "dense", &inp);
    for _ in 1..t {
        inp.clear();
        inp.extend(outs.iter().copied());
        inp.push(k_ad);
        inp.push(AD::constant(dt));
        outs = ctx.call_custom_dyn(Rc::clone(&op), "dense", &inp);
    }
    let fwd = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let mut loss = ctx.mul(outs[0], outs[0]);
    for o in &outs[1..n] {
        let sq = ctx.mul(*o, *o);
        loss = ctx.add(loss, sq);
    }
    ctx.backward(loss);
    let grads: Vec<f64> = leaves.iter().map(|&a| ctx.grad_of(a).unwrap_or(0.0)).collect();
    let bwd = t1.elapsed().as_secs_f64() * 1e3;
    let s = stats_end(stat);
    StressResult {
        label: format!("dense  N={n}"),
        state_dim: 2 * n,
        steps: t,
        fwd_ms: fwd,
        bwd_ms: bwd,
        allocs: s.allocs,
        peak_mib: s.peak_bytes as f64 / (1024.0 * 1024.0),
        grad_ok: grads.iter().all(|g| g.is_finite()) && grads.iter().any(|&g| g != 0.0),
    }
}

fn main() {
    // 自检：两个手写 VJP 先过 FD 对拍（公开验证器模式，n=4 逐坐标）——
    // "grad ok" 的可信度由验证器保证，而非仅有限性
    for (name, _n, op) in [
        ("sparse", 4usize, Rc::new(SparseChainStep { n: 4 }) as Rc<dyn CustomOp<f64>>),
        ("dense", 4, Rc::new(DenseCoupleStep { n: 4 }) as Rc<dyn CustomOp<f64>>),
    ] {
        let report = ad_verify::op_check::validate_custom_op(op, &[], 1e-5);
        assert!(report.passed, "{name} VJP 失败:\n{report}");
    }

    black_box(bench_sparse(10, 100)); // 预热

    println!(
        "{:<15} {:>7} {:>7} {:>10} {:>10} {:>10} {:>11} {:>6}",
        "family", "state", "steps", "fwd(ms)", "bwd(ms)", "allocs", "peak(MiB)", "grad"
    );
    for &(n, t) in &[(100usize, 10_000usize), (500, 10_000), (2000, 10_000)] {
        let r = bench_sparse(n, t);
        println!(
            "{:<15} {:>7} {:>7} {:>10.1} {:>10.1} {:>10} {:>11.1} {:>6}",
            r.label,
            r.state_dim,
            r.steps,
            r.fwd_ms,
            r.bwd_ms,
            r.allocs,
            r.peak_mib,
            if r.grad_ok { "ok" } else { "BAD" }
        );
    }
    for &(n, t) in &[(50usize, 1_000usize), (100, 1_000), (200, 1_000)] {
        let r = bench_dense(n, t);
        println!(
            "{:<15} {:>7} {:>7} {:>10.1} {:>10.1} {:>10} {:>11.1} {:>6}",
            r.label,
            r.state_dim,
            r.steps,
            r.fwd_ms,
            r.bwd_ms,
            r.allocs,
            r.peak_mib,
            if r.grad_ok { "ok" } else { "BAD" }
        );
    }
}
