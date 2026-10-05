# `ad` — 面向物理仿真的 Rust 反向模式自动微分库

纯 Rust、无 C++ 依赖的 tape-based 反向模式 AD，为可微分物理仿真引擎设计：
长轨迹内存可控（checkpoint）、物理算子可整体封装（`CustomOp`）、隐式求解器可微（IFT）、
梯度质量可度量（健康度分析 + 多重 oracle 验证）。

设计文档：[docs/design.md](docs/design.md)。

## Web 演示（Yew + trunk）

交互式 demo 页面（三个面板：标量求导 / 1000 步单摆 checkpoint 分段反向 + SVG 轨迹 / IFT 隐式求解，
全部带实时有限差分验证徽章），整个 AD 库栈编译到 `wasm32-unknown-unknown` 在浏览器中运行：

```bash
rustup target add wasm32-unknown-unknown
cargo install trunk
cd crates/ad-demo
trunk serve        # → http://localhost:8080
```

## 状态

v0.1：M1–M5 里程碑的核心能力已实现并有测试覆盖（详见设计文档"实现状态"一节）。

| 能力 | 状态 |
|------|------|
| 反向模式 AD（标量 `AD<f64>` / `AD<f32>`，Wengert tape） | ✅ |
| 线程局部 context + 运算符重载；显式 `&mut Context` 路径 | ✅ |
| 常量折叠、`no_grad` / `detach`、VJP 种子、梯度累加语义 | ✅ |
| 非有限梯度异常检测（定位首个出错算子） | ✅ |
| 基础算子全表 + bulk 向量算子（dot / axpy / norm2） | ✅ |
| `CustomOp`（custom_vjp 风格，多输出 + 残差） | ✅ |
| IFT 隐式求解模式（Newton + 双数 Jacobian + 伴随线性系统） | ✅ |
| checkpoint 分段反向（Uniform / Online / 自定义调度，边界伴随传递） | ✅ |
| 二分嵌套反转（`Nested { budget }`：峰值段 tape ≈ n/2^m，重算 ≈ (m+1)/2×，内存-重算旋钮） | ✅ |
| 验证：双数 oracle、复步微分、中心差分、随机方向、proptest、∇Fuzz 式可微性检查 | ✅ |
| Taylor 余项测试（科学计算社区标准验收法，dolfin-adjoint 实践） | ✅ |
| 梯度裁剪（`clip_grad_norm` / `clip_grad_value`） | ✅ |
| §5.3 物理场景清单：自由落体（解析解）、弹跳接触、12 体弹簧链（26 维） | ✅ |
| 空间代数算子库（`ad-physics`：6 个 CustomOp + 动能不变性先验测试） | ✅ |
| 平滑接触模型（Hunt–Crossley 法向 + 正则化库仑摩擦 + 刚度扫描健康度） | ✅ |
| 端到端轨迹优化基准（`ad-optim`：摆杆控制序列 150 维 + 接触弹跳球目标优化，梯度下降 + Armijo 收敛） | ✅ |
| iLQR 求解器（Tassa 正则化，AD 逐列 Jacobian；摆杆 6 迭代 vs GD 55 迭代） | ✅ |
| 双关节摆（Acrobot 构型）：能量守恒先验 + 混沌 iLQR 局部性实证 | ✅ |
| 3D 陀螺力学（`GyroscopicStep`：Euler 顶方程手写 VJP + 动能/角动量守恒 + 网球拍定理） | ✅ |
| criterion 基准 + 全局分配器无泄漏长稳测试 | ✅ |
| wasm32 编译 + Yew 交互式 web demo | ✅ |
| CI（fmt / clippy / test / wasm / 演示构建 / bench 冒烟） | ✅ |
| 经典 Revolve 二项式调度（纯伴随反转架构，无 tape） | ⏳ 见设计文档 §4.4.2/§12.3 |

## Workspace 结构

| Crate | 职责 |
|-------|------|
| [`ad-core`](crates/ad-core) | `AD<S>`、`Context`、`Tape`、`CustomOp` trait、线程局部挂载、双数 oracle（`test-oracle` feature） |
| [`ad-ops`](crates/ad-ops) | 基础算子（局部 Jacobian）+ bulk 向量算子 |
| [`ad-custom`](crates/ad-custom) | IFT 隐式求解模式、小型稠密线性求解 |
| [`ad-checkpoint`](crates/ad-checkpoint) | `Recomputable` 状态机、快照调度、分段反向 + 边界伴随；含 `PendulumSim` 参考实现 |
| [`ad-verify`](crates/ad-verify) | 有限差分 / 随机方向验证、梯度健康度、轨迹稳定性、可微性检查 |
| [`ad-physics`](crates/ad-physics) | 空间代数（Featherstone 风格）参考 CustomOp：Plücker 运动/力变换、空间惯性作用量与坐标系变换、SO(3) 指数映射、空间叉积，全部手写 VJP + 逐坐标 FD + 动能不变性验证；含平滑接触模型（Hunt–Crossley 法向 + 正则化库仑摩擦） |
| [`ad-optim`](crates/ad-optim) | 优化器：Armijo 回溯梯度下降 + Tassa 正则化 iLQR（AD 逐列 Jacobian）；含端到端收敛基准（受控摆杆 / 接触弹跳球 / 双关节摆混沌甩摆） |
| [`ad`](crates/ad) | facade：`use ad::prelude::*`；含 criterion 基准（`cargo bench -p ad`） |
| [`ad-demo`](crates/ad-demo) | Yew + trunk web 演示（wasm32），见上方"Web 演示" |

## 快速开始

线程局部路径（运算符重载）：

```rust
use ad::prelude::*;

let guard = Context::<f64>::new().enter();
let (x, vx) = guard.var(2.0);
let y = ad::sin(x * x) + x * 3.0;
guard.backward(y);
assert_eq!(guard.grad(vx), Some((2.0 * 2.0f64).cos() * 2.0 * 2.0 + 3.0));
```

长轨迹 + checkpoint（10⁴ 步单摆，可运行示例）：

```text
cargo run -p ad --example pendulum_checkpoint --release
```

物理引擎集成三步（详见设计文档 §4.3 / §4.4）：

1. 把每步动力学封装为 `CustomOp`（forward 返回 (输出, 残差)，backward 手工 VJP）；
2. 实现 `Recomputable`（完整确定性状态 + `bind_state` / `step`）；
3. 前向循环调 `ckpt.forward_step(...)`，结束时 `ckpt.backward(ctx, &sim, &loss)`。

## 测试与验证

```text
cargo test --workspace
```

验证体系（设计文档 §5）：

- **Primitive**：每个算子 vs 双数前向 oracle + 复步微分（机器精度）
- **Cross-check**：proptest 随机表达式 DAG（反向 vs 双数，512 组/次）
- **Physics**：PendulumStep 手工 backward vs AD 逐元素展开；IFT vs 闭式解 AD
- **Checkpoint**：分段反向 vs 全 tape 反向（多策略一致）；同策略重跑 bit-exact

## 关键语义约定

- 叶子梯度跨 `backward` **累加**；`zero_grads()` 只清梯度；`clear_tape()` 释放
  tape 并使中间 `AD` 失效（叶子句柄保持有效），长循环复用容量
- 运算符重载走线程局部 context（需 `Context::enter`），集成代码走 `*_with(ctx, ...)`
  显式路径；两者不可混用在同一个表达式里（重入会 panic 并给出明确信息）
- 数值边界遵守 IEEE 754（不 panic、不静默饱和）；`sqrt(0)` 反向 +inf 已文档化；
  开启 `set_detect_anomaly(true)` 可定位首个产生非有限梯度的记录
