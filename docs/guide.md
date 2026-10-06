# `ad` 集成指南 — 从自定义算子到轨迹优化

面向物理引擎作者的使用教程。设计原理与正确性方法学见
[design.md](design.md)；本文只讲"怎么接"。

完整的可运行示例：`crates/ad/examples/`（`batch_rollout`、`scale_stress`）、
各 crate 的 `tests/`（本文所有代码片段均摘自这些文件）。

---

## 0. 两条路径，一个纪律

库有两种入带方式（设计文档 §2.4），**表达式级别二选一，不可混用**：

```rust
use ad::prelude::*;

// 线程局部路径：运算符重载，适合顶层表达式
let guard = Context::<f64>::new().enter();
let (x, vx) = guard.var(2.0);
let y = ad::sin(x * x) + x * 3.0;
guard.backward(y);

// 显式路径：&mut Context 传参，适合引擎集成代码（Dynamics::step 等）
fn torque(ctx: &mut Context<f64>, th: AD<f64>) -> AD<f64> {
    let s = ad::sin_with(ctx, th);      // 显式版本算子：*_with
    ctx.mul(s, AD::constant(3.0))
}
```

混用会 panic（重入防护是特性）。**规则：仿真步进函数里只用显式路径。**

---

## 1. 写一个 CustomOp（物理算子的标准接入方式）

物理动力学封装为**一个** CustomOp = 长轨迹下 tape 每步只增长 1 条记录
（逐元素展开是 ~40 条，ABA 递归展开是 ~200 条——§4.3.4）。

```rust
use ad_core::{CustomOp, AD};
use smallvec::{smallvec, SmallVec};

/// 单摆单步：inputs = [θ, ω, g, L, dt]，outputs = [θ', ω']。
/// 半隐式欧拉；残差保存 backward 需要的中间量（f_fwd 风格）。
struct PendulumStep;

impl CustomOp<f64> for PendulumStep {
    fn num_inputs(&self) -> usize { 5 }
    fn num_outputs(&self) -> usize { 2 }

    fn forward(&self, i: &[f64]) -> (SmallVec<[f64; 8]>, SmallVec<[f64; 8]>) {
        let (th, om, g, l, dt) = (i[0], i[1], i[2], i[3], i[4]);
        let th1 = th + dt * om;
        let om1 = om - dt * (g / l) * th.sin();
        (smallvec![th1, om1], smallvec![th.sin(), th.cos(), om, dt, g, l])
    }

    fn backward(&self, r: &[f64], go: &[f64]) -> SmallVec<[f64; 8]> {
        // go = [λθ', λω']；残差 r = [sinθ, cosθ, ω, dt, g, L]
        let (lam1, lam2) = (go[0], go[1]);
        smallvec![
            lam1 + lam2 * (-r[3] * (r[4] / r[5]) * r[1]), // ∂/∂θ
            lam1 * r[3] + lam2,                            // ∂/∂ω
            lam2 * (-r[3] * r[0] / r[5]),                  // ∂/∂g
            lam2 * (r[3] * r[4] * r[0] / (r[5] * r[5])),   // ∂/∂L
            lam1 * r[2] + lam2 * (-(r[4] / r[5]) * r[0]),  // ∂/∂dt
        ]
    }

    fn name(&self) -> &'static str { "pendulum_step" }
}
```

### 接入前的三条契约（违反 = 静默错误梯度）

1. **内部边**（§4.3.1）：`θ' = θ + dt·ω'` 里 ω' 是本算子的另一个输出，
   tape 看不见这条边——backward 必须自己处理（`λω'总计 = λω' + dt·λθ'`）。
2. **gins 按原始槽位对齐**：返回长度恒等于 `num_inputs`；常量输入占槽位
   但不占节点（其梯度被框架丢弃）。**不要假设常量都在尾部**——
   库自己曾在此翻车（§12.3 第 28b 条）。
3. **forward 必须确定**（逐位）：checkpoint 重算依赖它（§4.4.5）。
   迭代求解器要固定迭代次数、RNG/warm-start 入残差或输入。

## 2. 一行验证（不要跳过）

手写 VJP 的错误**不报错**。库提供公开验证器（§12.3 第 33 条）：

```rust
use ad_verify::op_check::validate_custom_op;
use std::rc::Rc;
use ad_core::CustomOp;

let report = validate_custom_op(
    Rc::new(PendulumStep),
    &[vec![0.5, 0.0, 9.81, 1.0, 0.01]], // 定义域内的测试点
    1e-5,                                 // 相对容差
);
assert!(report.passed, "{}", report);
```

检查内容：前向确定性（逐位）+ gins 长度契约 + **四种追踪形态**（全追踪 /
常量在头部 / 在尾部 / 隔位）下的逐坐标 FD 对拍。历史战绩：库内 7 个算子
全过；fuzz 算子与压力测试算子的手写 VJP 各被抓出 1–2 处（§12.3 第 33/37 条）。

**tol 选择**：f64 用 1e-5；接触/高刚度算子放大到 1e-4（FD 步长 1e-6 下的
截断误差随二阶导增长）。f32 自定义算子目前不适用（CustomOp 为 f64 特化）。

## 3. 接入 checkpoint（长轨迹的内存形态）

实现 `Recomputable` 状态机，交给 `CheckpointManager`（设计文档 §4.4）：

```rust
use ad::{CheckpointManager, CheckpointStrategy, Recomputable, Context, AD};

struct PendulumSim {
    theta: f64, omega: f64,
    state_ad: Vec<AD<f64>>,
    g: AD<f64>, len: AD<f64>, dt: f64,
}

impl Recomputable for PendulumSim {
    type State = (f64, f64);                     // 完整动态状态
    fn save_state(&self) -> (f64, f64) { (self.theta, self.omega) }
    fn load_state(&mut self, s: &(f64, f64)) { self.theta = s.0; self.omega = s.1; }
    fn bind_state(&mut self, ctx: &mut Context<f64>) -> Vec<AD<f64>> {
        let (th, _) = ctx.var(self.theta);       // load_state 后必须 bind_state！
        let (om, _) = ctx.var(self.omega);       // （§4.4.5：否则消费过期 AD 视图）
        self.state_ad = vec![th, om];
        self.state_ad.clone()
    }
    fn state(&self) -> &[AD<f64>] { &self.state_ad }
    fn step(&mut self, ctx: &mut Context<f64>) {
        let inputs = [self.state_ad[0], self.state_ad[1], self.g, self.len,
                      AD::constant(self.dt)];
        let outs = ctx.call_custom(PendulumStep, &inputs);
        self.theta = outs[0].value; self.omega = outs[1].value;
        self.state_ad = outs.to_vec();
    }
}
```

使用：

```rust
let mut sim = PendulumSim::new(&mut ctx, 0.5, 0.0, g_ad, l_ad, 0.01);
let mut ckpt = CheckpointManager::new(CheckpointStrategy::Uniform { interval: 32 }, &sim);
for t in 0..1000 {
    ckpt.forward_step(&mut ctx, &mut sim, t);   // no_grad 前进 + 按策略快照
}
let init_adj = ckpt.backward(&mut ctx, &mut sim, &loss_fn); // 分段反向（含重算）
// init_adj = ∂L/∂x₀（初始状态伴随）；参数梯度经 ctx.grad(var) 读取
```

策略选择：`Uniform{interval}` 默认（重算恰 1× 前向）；`Nested{budget}`
状态大而单步 tape 贵时用（内存-重算旋钮）；`Online` MPC 流式 rollout。
**同策略重跑 bit-exact** 是库的验收判据（确定性重算下成立）。

规模参考（`examples/scale_stress.rs`，release）：4000 维状态 × 10⁴ 步
checkpoint 分段反向 ~0.93 s、峰值内存 37.9 MiB；200 维 × 10³ 步全 tape
~5 ms。

## 4. 梯度验证 + 健康度（上线前 ritual）

```rust
use ad_verify::GradientChecker;

let checker = GradientChecker::default();
let r = checker.check_scalar(&|x| loss(x), &x, &grads);   // 逐坐标 FD
assert!(r.passed);
let health = checker.analyze_health(&grads);              // 范数/零占比/非有限
let traj  = checker.check_trajectory_stability(&grads_per_step); // vanishing/exploding
```

Taylor 余项检验（比单点 FD 更强的判据）：`checker.taylor_test(...)`。
**健康度 ≠ 正确性**：数值正确的梯度在混沌/接触问题上仍可能不可用
（Howell et al. 2022）——优化器的健康度输出（下一节）就是为此。

## 5. 轨迹优化（GD 与 iLQR）

```rust
use ad_optim::{Dynamics, IlqrCfg, QuadraticCost, solve_ilqr};

struct MyDyn { /* 参数 */ }
impl Dynamics for MyDyn {
    fn nx(&self) -> usize { 4 }
    fn nu(&self) -> usize { 2 }
    fn step(&self, ctx: &mut Context<f64>, x: &[AD<f64>], u: &[AD<f64>]) -> Vec<AD<f64>> {
        // 显式路径组装单步动力学（可复用你的 CustomOp：ctx.call_custom）
        // 禁止运算符重载（TLS 重入 panic）
    }
}

let cost = QuadraticCost { q, r, qf, goal };
let cfg = IlqrCfg { max_iters: 60, ..Default::default() };
let (u, rep) = solve_ilqr(&dyn_impl, &x0, &u0, &cost, &cfg);
```

读 `rep` 的健康度字段判断"为什么没收敛"（§12.3 第 35 条）：
- `quu_cond_max` ≫ 1e8：控制通道二阶信息病态 → 加大 R 或收紧 box；
- `mu_final` 被推高 / `line_search_rejections` 多：实际下降达不到 iLQR 的
  二阶预期——混沌接触问题的特征（局部解，非梯度错误）；
- GD 的 `grad_health.nonfinite_fraction > 0` 或 `line_search_failures > 8`：
  梯度与损失不一致，先用 §4 的验证器定位算子。

box 约束：`IlqrCfg { u_min: Some(...), u_max: Some(...) }`（clamped 前向）。
MPC 滚动形态见 `tests/mpc.rs`。

## 6. 性能纪律（压测数据支撑，§12.3 第 26/28 条）

1. 优化循环每轮 `clear_tape()`——否则 tape 无限增长退化为 O(n²)；
2. 高频 CustomOp 缓存 `Rc<dyn CustomOp>` 走 `call_custom_dyn`
   （`PendulumSim` 是示范：每步 0 次堆分配）；
3. O(n²)+ 的操作（matvec、全对力）**必须**整体封装为 CustomOp；
4. 剖面工具：`cargo run --release -p ad --example profile`（分配画像 +
   墙钟）；`--loop N` 模式供 cdb 采样。

## 7. 常见错误速查（全部踩过的坑）

| 症状 | 根因 | 出处 |
|------|------|------|
| 所有坐标同乘一个因子 | FD oracle 与 AD loss 不逐项一致 / 共同上游 λ 错 | §12.3 第 19 条 |
| 长轨迹梯度错、单步对 | 内部边（输出间耦合）未处理 | 第 13 条 |
| 常量在非尾部槽位时梯度错 | backward 按 zip 位置而非 slot 索引 | 第 28b 条 |
| 对称 λ 全对、反对称全错 | vee 符号约定（两个负号） | 第 18a 条 |
| 反向积分 KE 守恒但动力学错 | 只测能量守恒——加角动量/动量守恒 | 第 25 条 |
| checkpoint 重算梯度错 | load_state 后未 bind_state | 第 17 条 |
| 优化越跑越慢 | 忘了 clear_tape（tape_len 探针可查） | 第 12 条 |
