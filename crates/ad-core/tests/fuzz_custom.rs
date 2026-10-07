//! CustomOp 随机图 fuzz（§12.3 第 36 条）：在随机 DAG 中掺入手写 VJP 的
//! CustomOp，反向 vs 双数 oracle 逐例对拍。
//!
//! 覆盖重点（基础算子 fuzz 的盲区）：
//! - **CustomOp 的链式组装**：算子输出扇出、多算子级联、与基础算子混合；
//! - **部分追踪形态**：每个输入槽位以 25% 概率被常量替换——常量可以落在
//!   **任意槽位（含非尾部）**，这正是暴露过梯度路由潜伏 bug 的形态
//!   （§12.3 第 28b 条 / HANDOFF §4.11）；
//! - **部分输出依赖**：TriOp 的 out1 不依赖 a——零 Jacobian 槽位的路由。
//!
//! 三个算子的手写 VJP 先经 `ad_verify::op_check::validate_custom_op`
//! （四种追踪形态 FD 对拍）独立验证，再进入 fuzz——验证器抓"算子错了"，
//! fuzz 抓"组装错了"。
//!
//! oracle 方法：双数单通道，对 4 个叶子各重放一遍图（第 k 遍第 k 叶种子，
//! 其余常量），loss 的 du 即 ∂L/∂(leaf k)。两路执行**同一份预规划的图**
//! （随机决策先规划后执行，保证 tape 与 oracle 结构逐位同构）。

use ad_core::dual::Dual;
use ad_core::{Context, CustomOp, AD};
use num_traits::{Num, Zero};
use smallvec::{smallvec, SmallVec};
use std::rc::Rc;

// ============================================================ 三个 fuzz 算子

/// 2 入 2 出：out = [2a²·b, a + 3b³]
#[derive(Clone, Copy)]
struct SqMixOp;

impl SqMixOp {
    fn fwd<N: Num + Copy>(&self, i: &[N]) -> SmallVec<[N; 8]> {
        let (a, b) = (i[0], i[1]);
        let two = N::one() + N::one();
        let three = two + N::one();
        smallvec![two * a * a * b, a + three * b * b * b]
    }
}

impl CustomOp<f64> for SqMixOp {
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        2
    }
    fn forward(&self, i: &[f64]) -> (SmallVec<[f64; 8]>, SmallVec<[f64; 8]>) {
        (self.fwd(i), i.iter().copied().collect())
    }
    fn backward(&self, r: &[f64], go: &[f64]) -> SmallVec<[f64; 8]> {
        let (a, b) = (r[0], r[1]);
        // out0 = 2a²b → ∂/∂a = 4ab, ∂/∂b = 2a²；out1 = a + 3b³ → ∂/∂b = 9b²
        smallvec![go[0] * 4.0 * a * b + go[1], go[0] * 2.0 * a * a + go[1] * 9.0 * b * b]
    }
    fn name(&self) -> &'static str {
        "sq_mix"
    }
}

/// 3 入 2 出：out = [a·b + c², (b−c)·(b+c)]——out1 **不依赖 a**（零 Jacobian 槽位）。
#[derive(Clone, Copy)]
struct TriOp;

impl TriOp {
    fn fwd<N: Num + Copy>(&self, i: &[N]) -> SmallVec<[N; 8]> {
        let (a, b, c) = (i[0], i[1], i[2]);
        smallvec![a * b + c * c, (b - c) * (b + c)]
    }
}

impl CustomOp<f64> for TriOp {
    fn num_inputs(&self) -> usize {
        3
    }
    fn num_outputs(&self) -> usize {
        2
    }
    fn forward(&self, i: &[f64]) -> (SmallVec<[f64; 8]>, SmallVec<[f64; 8]>) {
        (self.fwd(i), i.iter().copied().collect())
    }
    fn backward(&self, r: &[f64], go: &[f64]) -> SmallVec<[f64; 8]> {
        let (a, b, c) = (r[0], r[1], r[2]);
        let two_c = 2.0 * c;
        smallvec![go[0] * b, go[0] * a + go[1] * (b + b), go[0] * two_c - go[1] * two_c]
    }
    fn name(&self) -> &'static str {
        "tri"
    }
}

/// 6 入 1 出：out = Σ aᵢ·bᵢ（inputs = [a0,a1,a2,b0,b1,b2]，bulk dot 形态）。
#[derive(Clone, Copy)]
struct Dot3Op;

impl Dot3Op {
    fn fwd<N: Num + Copy>(&self, i: &[N]) -> SmallVec<[N; 8]> {
        let mut s = N::zero();
        for k in 0..3 {
            s = s + i[k] * i[3 + k];
        }
        smallvec![s]
    }
}

impl CustomOp<f64> for Dot3Op {
    fn num_inputs(&self) -> usize {
        6
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn forward(&self, i: &[f64]) -> (SmallVec<[f64; 8]>, SmallVec<[f64; 8]>) {
        (self.fwd(i), i.iter().copied().collect())
    }
    fn backward(&self, r: &[f64], go: &[f64]) -> SmallVec<[f64; 8]> {
        smallvec![
            go[0] * r[3],
            go[0] * r[4],
            go[0] * r[5],
            go[0] * r[0],
            go[0] * r[1],
            go[0] * r[2],
        ]
    }
    fn name(&self) -> &'static str {
        "dot3"
    }
}

// ============================================================ 验证器先行

#[test]
fn fuzz_ops_pass_the_validator() {
    let mut rng = Rng::new(7);
    let mut pts = |n: usize| -> Vec<Vec<f64>> {
        (0..3)
            .map(|_| (0..n).map(|_| 0.5 + 1.0 * rng.next_f64()).collect())
            .collect()
    };
    for (name, n, op) in [
        ("sq_mix", 2usize, Rc::new(SqMixOp) as Rc<dyn CustomOp<f64>>),
        ("tri", 3, Rc::new(TriOp) as Rc<dyn CustomOp<f64>>),
        ("dot3", 6, Rc::new(Dot3Op) as Rc<dyn CustomOp<f64>>),
    ] {
        let report = ad_verify::op_check::validate_custom_op(op, &pts(n), 1e-6, 1e-5);
        assert!(report.passed, "{name}: {}", report);
    }
}

// ============================================================ 随机图规划与双路执行

/// 确定性 xorshift（与 scenarios.rs 同款，可复现）
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// [0.5, 1.5] 或 [-1.5, -0.5]
    fn value(&mut self) -> f64 {
        let v = 0.5 + self.next_f64();
        if self.next_u64() & 1 == 0 {
            v
        } else {
            -v
        }
    }
    fn index(&mut self, len: usize) -> usize {
        (self.next_u64() % len as u64) as usize
    }
    /// 25% 概率替换为常量（值独立抽取）
    fn maybe_const(&mut self) -> (bool, f64) {
        (self.next_u64().is_multiple_of(4), self.value())
    }
}

/// 一次随机决策（先规划、后双路执行）
enum Step {
    /// 0=add 1=sub 2=mul
    Prim { op: u8, i: usize, j: usize },
    /// 乘以 2 的幂常量（FP 精确缩放，把 |v| 压回界内防梯度对消；
    /// 两路执行逐位一致）
    Scale { i: usize, c: f64 },
    SqMix {
        i: usize,
        j: usize,
        ca: bool,
        cb: bool,
        va: f64,
        vb: f64,
    },
    Tri {
        i: usize,
        j: usize,
        k: usize,
        ca: bool,
        cb: bool,
        cc: bool,
        va: f64,
        vb: f64,
        vc: f64,
    },
    Dot3 {
        idx: [usize; 6],
        cs: [bool; 6],
        vs: [f64; 6],
    },
}

/// 界：|v| ≤ 64 时无需缩放；缩放把 |v| 压到 ≤ 16。
const BOUND: f64 = 64.0;

fn plan_case(seed: u64) -> (Vec<f64>, Vec<Step>, [f64; 3]) {
    let mut rng = Rng::new(seed);
    let leaves: Vec<f64> = (0..4).map(|_| rng.value()).collect();
    let mut steps = Vec::new();
    // 值模拟（与执行路径同一算术）：用于决定是否插入 Scale——
    // 无界的图会让梯度在 ~1e15 量级的项之间灾难性对消，fuzz 失真
    let mut sim: Vec<f64> = leaves.clone();
    let mut n_vals = 4usize;
    for _ in 0..24 {
        let roll = rng.next_u64() % 8;
        if roll < 4 || n_vals < 6 {
            let op = (rng.next_u64() % 3) as u8;
            let (i, j) = (rng.index(n_vals), rng.index(n_vals));
            let v = match op {
                0 => sim[i] + sim[j],
                1 => sim[i] - sim[j],
                _ => sim[i] * sim[j],
            };
            sim.push(v);
            steps.push(Step::Prim { op, i, j });
            n_vals += 1;
        } else if roll < 6 {
            let (ca, va) = rng.maybe_const();
            let (cb, vb) = rng.maybe_const();
            let (i, j) = (rng.index(n_vals), rng.index(n_vals));
            let out0 = 2.0 * sim[i] * sim[i] * sim[j];
            let out1 = sim[i] + 3.0 * sim[j] * sim[j] * sim[j];
            sim.push(out0);
            sim.push(out1);
            steps.push(Step::SqMix {
                i,
                j,
                ca,
                cb,
                va,
                vb,
            });
            n_vals += 2;
        } else if roll == 6 {
            let (ca, va) = rng.maybe_const();
            let (cb, vb) = rng.maybe_const();
            let (cc, vc) = rng.maybe_const();
            let (i, j, k) = (rng.index(n_vals), rng.index(n_vals), rng.index(n_vals));
            let out0 = sim[i] * sim[j] + sim[k] * sim[k];
            let out1 = (sim[j] - sim[k]) * (sim[j] + sim[k]);
            sim.push(out0);
            sim.push(out1);
            steps.push(Step::Tri {
                i,
                j,
                k,
                ca,
                cb,
                cc,
                va,
                vb,
                vc,
            });
            n_vals += 2;
        } else {
            let mut idx = [0usize; 6];
            let mut cs = [false; 6];
            let mut vs = [0.0f64; 6];
            for s in 0..6 {
                idx[s] = rng.index(n_vals);
                let (c, v) = rng.maybe_const();
                cs[s] = c;
                vs[s] = v;
            }
            let mut acc = 0.0;
            for s in 0..3 {
                let a = if cs[s] { vs[s] } else { sim[idx[s]] };
                let b = if cs[s + 3] { vs[s + 3] } else { sim[idx[s + 3]] };
                acc += a * b;
            }
            sim.push(acc);
            steps.push(Step::Dot3 { idx, cs, vs });
            n_vals += 1;
        }
        // 值越界 → 插入 2 的幂缩放步（FP 精确）。Scale 的输入 = 刚产出的
        // 那个值（下标 n_vals−1）；缩放副本追加为其后一个下标。
        let mut target = n_vals - 1;
        while sim[target].abs() > BOUND {
            let mut c = 1.0f64;
            while sim[target].abs() * c > 16.0 {
                c *= 0.5;
            }
            sim.push(sim[target] * c);
            steps.push(Step::Scale { i: target, c });
            n_vals += 1;
            target = n_vals - 1;
        }
    }
    let weights = [0.5 + rng.next_f64(), 0.5 + rng.next_f64(), 0.5 + rng.next_f64()];
    (leaves, steps, weights)
}

fn run_tape(leaves: &[f64], steps: &[Step], weights: &[f64; 3]) -> Vec<f64> {
    let mut ctx = Context::<f64>::new();
    let mut vals: Vec<AD<f64>> = Vec::new();
    let mut vars = Vec::new();
    for &v in leaves {
        let (ad, var) = ctx.var(v);
        vals.push(ad);
        vars.push(var);
    }
    for step in steps {
        match *step {
            Step::Prim { op, i, j } => {
                let (a, b) = (vals[i], vals[j]);
                let v = match op {
                    0 => ctx.add(a, b),
                    1 => ctx.sub(a, b),
                    _ => ctx.mul(a, b),
                };
                vals.push(v);
            }
            Step::Scale { i, c } => {
                let v = ctx.mul(AD::constant(c), vals[i]);
                vals.push(v);
            }
            Step::SqMix { i, j, ca, cb, va, vb } => {
                let ia = if ca { AD::constant(va) } else { vals[i] };
                let ib = if cb { AD::constant(vb) } else { vals[j] };
                vals.extend(ctx.call_custom(SqMixOp, &[ia, ib]));
            }
            Step::Tri { i, j, k, ca, cb, cc, va, vb, vc } => {
                let ia = if ca { AD::constant(va) } else { vals[i] };
                let ib = if cb { AD::constant(vb) } else { vals[j] };
                let ic = if cc { AD::constant(vc) } else { vals[k] };
                vals.extend(ctx.call_custom(TriOp, &[ia, ib, ic]));
            }
            Step::Dot3 { idx, cs, vs } => {
                let inputs: Vec<AD<f64>> = (0..6)
                    .map(|s| {
                        if cs[s] {
                            AD::constant(vs[s])
                        } else {
                            vals[idx[s]]
                        }
                    })
                    .collect();
                vals.extend(ctx.call_custom(Dot3Op, &inputs));
            }
        }
    }
    // loss = Σ_{末 3 个值} (w·v + 0.07·v²)。**只对被追踪的项入带**——
    // 末 3 个值可能全是常量输出（其输入全为常量时算子退化为常量），
    // 对常量 backward 会 panic；常量项对 du 贡献为 0，跳过两路语义一致。
    let n = vals.len();
    let mut loss: Option<AD<f64>> = None;
    for (kk, w) in weights.iter().enumerate() {
        let v = vals[n - 3 + kk];
        if !v.is_tracked() {
            continue;
        }
        let lin = ctx.mul(AD::constant(*w), v);
        let sq = ctx.mul(v, v);
        let quad = ctx.mul(AD::constant(0.07), sq);
        let term = ctx.add(lin, quad);
        loss = Some(match loss {
            Some(l) => ctx.add(l, term),
            None => term,
        });
    }
    if let Some(l) = loss {
        ctx.backward(l);
    }
    vars.iter().map(|&v| ctx.grad(v).unwrap_or(0.0)).collect()
}

fn run_dual_leaf(leaves: &[f64], steps: &[Step], weights: &[f64; 3], k: usize) -> f64 {
    let mut duals: Vec<Dual> = leaves
        .iter()
        .enumerate()
        .map(|(i, &v)| if i == k {
            Dual::new(v, 1.0)
        } else {
            Dual::constant(v)
        })
        .collect();
    for step in steps {
        match *step {
            Step::Prim { op, i, j } => {
                let v = match op {
                    0 => duals[i] + duals[j],
                    1 => duals[i] - duals[j],
                    _ => duals[i] * duals[j],
                };
                duals.push(v);
            }
            Step::Scale { i, c } => {
                duals.push(Dual::constant(c) * duals[i]);
            }
            Step::SqMix { i, j, ca, cb, va, vb } => {
                let ia = if ca { Dual::constant(va) } else { duals[i] };
                let ib = if cb { Dual::constant(vb) } else { duals[j] };
                duals.extend(SqMixOp.fwd(&[ia, ib]));
            }
            Step::Tri { i, j, k, ca, cb, cc, va, vb, vc } => {
                let ia = if ca { Dual::constant(va) } else { duals[i] };
                let ib = if cb { Dual::constant(vb) } else { duals[j] };
                let ic = if cc { Dual::constant(vc) } else { duals[k] };
                duals.extend(TriOp.fwd(&[ia, ib, ic]));
            }
            Step::Dot3 { idx, cs, vs } => {
                let inputs: Vec<Dual> = (0..6)
                    .map(|s| {
                        if cs[s] {
                            Dual::constant(vs[s])
                        } else {
                            duals[idx[s]]
                        }
                    })
                    .collect();
                duals.extend(Dot3Op.fwd(&inputs));
            }
        }
    }
    let n = duals.len();
    let mut loss = Dual::zero();
    for (kk, w) in weights.iter().enumerate() {
        let v = duals[n - 3 + kk];
        loss = loss + Dual::constant(*w) * v + Dual::constant(0.07) * v * v;
    }
    loss.du
}


// ============================================================ 随机图 fuzz 主测试

#[test]
fn fuzz_custom_random_graphs_match_dual_oracle() {
    let mut total_mismatch = 0usize;
    for seed in 1..=512u64 {
        let (leaves, steps, weights) = plan_case(seed);
        let tape_grads = run_tape(&leaves, &steps, &weights);
        for k in 0..4 {
            let du = run_dual_leaf(&leaves, &steps, &weights, k);
            let g = tape_grads[k];
            if (g - du).abs() > 1e-10 * (1.0 + du.abs()) {
                eprintln!("seed {seed} leaf {k}: ad {g:.12} vs dual {du:.12}");
                total_mismatch += 1;
            }
        }
    }
    assert_eq!(total_mismatch, 0, "{total_mismatch} 个 (seed, leaf) 不一致");
}

