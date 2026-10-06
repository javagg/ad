# HANDOFF — 项目状态与新 Session 接入指南

> 写于 2026-10-06，同日两次更新。最后一次全量验证：**40 套件全绿**、clippy 零警告、
> wasm32 编译通过。本文档目标：新 session 零上下文即可继续推进，不丢任何关键决策/陷阱/路径。

## 1. 项目概要

**`ad`**：面向物理仿真的纯 Rust 反向模式自动微分库。设计文档 `docs/design.md`（v0.3，含完整 API 设计 + 实现回写 §12）。

**核心理念**：梯度不仅要算得对（多重 oracle 验证），还要能用（优化收敛基准）且健康（健康度分析）。

**仓库**：`https://github.com/javagg/ad.git`（公开）。注意：CI workflow 文件因 OAuth token 缺 `workflow` scope 暂存在 `.github/ci.yml.pending`，待授权后移回 `.github/workflows/ci.yml`。

## 2. Crate 架构（9 个，依赖方向自上而下）

```
ad-core    AD<S>标量、Context、Tape、CustomOp trait、线程局部挂载、双数oracle、no_grad/detach/异常检测
ad-ops     基础算子全表（含 sin/cos/tanh/atan2/clamp/lerp 等）+ bulk 向量（dot/axpy/norm2）
ad-custom  IFT 隐式求解模式（ImplicitSolve）、稠密线性求解
ad-checkpoint  Recomputable 状态机、快照调度（Uniform/Nested/Online/Custom）、分段反向+边界伴随、PendulumSim
ad-physics 空间代数算子（spatial.rs 辅助 + ops.rs 6个CustomOp + contact.rs 接触力）
ad-verify  FD/随机方向/Taylor余项/健康度/轨迹稳定性/可微性检查
ad-optim   Armijo GD + Tassa正则化iLQR（Dynamics trait + AD逐列Jacobian）
ad         facade：re-exports 全部 + prelude
ad-demo    Yew + trunk wasm32 web demo（三面板：标量/单摆checkpoint/IFT）
```

## 3. 已完成里程碑（按提交时间序）

| 提交 | 内容 |
|------|------|
| `a81b459` | M1-M5 核心 + Yew web demo |
| `a26f42f` | 梯度裁剪 + 物理场景测试 + CI + CustomOp 内部边 bug 修复 |
| `aae43e6` | Taylor 余项测试（ad-verify） |
| `4c6f8d6` | Nested{budget} 嵌套反转检查点 |
| `2fe2a08` | 3D 陀螺力学 GyroscopicStep |
| `a532b6c` | 空间代数 CustomOp 库（ad-physics） |
| `ab11864` | 平滑接触模型套件 |
| `f8430f8` | iLQR 求解器 |
| `c0c0a2f` | 端到端轨迹优化基准（GD + 接触球） |
| `abdef4e` | 接触 iLQR 基准 |
| `972641a` | 火焰图性能工程（M5 收尾）：分配热点消除，分段反向 −51%、分配 −98%；profile 剖析用例 |
| （本次） | 手写 CustomOp 双关节摆补全（`ad-physics::chain::DoublePendulumStep`）+ iLQR 逐位对拍；文档 v0.3.4 |

## 4. 关键设计决策与陷阱（新 session 必读）

### 4.1 CustomOp backward 的内部边陷阱（§4.3.1 契约）
tape 只看见算子级输入→输出边。backward 必须覆盖算子内部全部数据流，包括**输出之间的耦合边**。
例：`x' = x + dt·v'`（v' 是同一算子的另一个输出）→ 作用于 a(x) 路径的伴随 = `λv' + dt·λx'`。
**接入规范：新 CustomOp 先过单步逐坐标 FD 隔离器（参照 `ad-physics/tests/ops_fd.rs` 的 `check_op`），再上长轨迹。**

### 4.2 Recomputable 的 "load_state 后必须 bind_state"
`load_state` 只刷新标量镜像；后续 `step` 消费 AD 状态——若不 `bind_state` 重建，会用**过期的 AD 视图**（嵌套反转路径两个 bug 的根因）。

### 4.3 FD oracle 与 AD loss 必须逐项一致
FD 隔离器的 loss（如 `loss_of_out`）与 AD 路径的 loss 表达式必须**逐项相同**——曾有 `len >= 2` 守卫跳过单输出交叉项，产生全坐标一致 1.32× 偏差。失败模式：**所有坐标同乘一个因子 = 共同上游 λ 出错**。

### 4.4 守恒先验必须多样化
动能守恒（时间反演对称）不能发现陀螺 forward 符号反转——角动量守恒才能。每个不变量检验动力学不同侧面。已入库：KE、角动量、能量（双关节摆）、Taylor 余项、FD 逐坐标。

### 4.5 TLS 与显式路径不可混用
运算符重载（`+`/`*`/`sin()`）走线程局部 context（需 `Context::enter`）；集成代码走 `*_with(ctx, ...)` 显式路径。同一表达式混用会 panic（重入防护）。**Dynamics::step 实现中必须全部用显式方法。**

### 4.6 对称打包（sym3）梯度是完整和（非 ½）
每个打包分量控制矩阵两个对称位置 → 梯度 = `g_ij + g_ji`（非 `½(g_ij+g_ji)`）。

### 4.7 vee 符号约定
`vee([a]×) = [−m₅, m₂, −m₁]`（两个负号）——漏掉时对称 λ 全过、反对称 λ 全错（特征性诊断模式）。

### 4.8 手写 VJP 的工程结论（§12.3 第 24 条）
三角耦合 + 多处参数依赖的动力学算子（双关节摆）手写 VJP 修正成本远超预期（6+ 处错误）。
**工程结论**：此复杂度级别优先 AD 直通；手写 CustomOp 保留给 O(n²)+ bulk 操作。
（2026-10-06 更新：第 27 条修正——手写版**可行**，前提是按 M⁻¹ 中间量分解推导
VJP 并先过 FD 隔离器；见 `ad-physics::chain::DoublePendulumStep`。）
已被此管线抓出的 bug 类型：vee 符号、Coriolis 项放错方程、重力二角漏项、内部边耦合、λ 路由、透传输出 λ。

### 4.9 iLQR 的接触局部性
穿透接触 + 混沌动力学下，iLQR 收敛到局部解（数值正确的梯度 ≠ 全局优化收敛）。
基准断言按实际可达水平定标，并记录为 Howell et al. 2022 的实证。

### 4.10 性能注意
优化循环必须每轮 `clear_tape()`，否则 tape 无限增长退化为 O(n²)。`tape_len()` 探针用于检测。

## 5. 测试体系总览（40 套件）

| 层级 | 位置 | 方法 |
|------|------|------|
| Primitive | ad-ops/tests/ops.rs | 逐算子 vs 双数前向 + 复步微分（1e-12） |
| Cross-check | ad-core/tests/fuzz.rs | proptest 随机 DAG（反向 vs 双数，512 组） |
| 场景 | ad/tests/scenarios.rs | 自由落体(解析解)/弹跳接触/12体链(26维) |
| 物理 | ad-physics/tests/ | FD隔离器/动能守恒/角动量/网球拍/接触锥 |
| 检查点 | ad-checkpoint/tests/ | 分段vs全tape(4策略)/嵌套反转/重跑bit-exact |
| 优化 | ad-optim/tests/e2e.rs | GD/iLQR 摆杆+接触球收敛基准 |
| 长稳 | ad-core/tests/leak_*.rs | 全局分配器计数，50k 轮无泄漏 |
| 基准 | ad/benches/ad_bench.rs | criterion（标量/单摆/全tape对照） |
| web demo | crates/ad-demo | Yew + trunk serve（三面板 + FD 徽章） |

## 6. 下一步（按优先级）

### 6.1 ~~火焰图性能工程（M5 收尾）~~ ✅ 已完成（2026-10-06，§12.3 第 26 条）
方法论：Windows 无 perf/dtrace、samply 需管理员 → **计数分配器画像 + cdb
poor-man's profiler**（负载驱动 `cargo run --release -p ad --example profile`，
`--loop <秒>` 模式供 cdb 采样）。成果：criterion 分段反向 288→141 µs（−51%）、
全 tape 177→133 µs（−25%）；分配次数 checkpoint 7160→128、全 tape 4016→16。
修复：SmallVec 内联 4→8（CustomOp forward/backward、tape Custom 记录）、
`call_custom` 返回 SmallVec、PendulumSim 缓存 Rc 走 call_custom_dyn。

### 6.2 ~~真实铰链多体链（方向 3 手写 CustomOp 补全）~~ ✅ 已完成（2026-10-06，§12.3 第 27 条）
`ad-physics::chain::DoublePendulumStep`：inputs = [θ1,θ2,ω1,ω2,τ1,τ2,dt]，
outputs = [θ1',θ2',ω1',ω2']，VJP 按 **M⁻¹ 分解**推导（`s = M⁻¹λa`、`λr = s`、
`λM_ij = −s_i·a_j` 全和约定）。四层验证全过：FD 隔离器（4 状态）、能量守恒
（漂移 0.38%）、单步梯度/Jacobian 对拍 AD 直通（1e-9）、iLQR 甩摆基准
**逐位一致**（loss 2.7839→1.573964，7 迭代）。隔离器首跑抓出 2 处错误
（重力双角 θ1 依赖漏项 + 内部边 λdt 需用 λω'(总计)）——教训已入算子文档。
iLQR 对拍测试在 `ad-optim/tests/chain.rs`（ad-optim dev-dep ad-physics）。

### 6.3 CI workflow 恢复（需要用户操作）
`gh auth refresh -h github.com -s workflow` → 浏览器授权 →
`git mv .github/ci.yml.pending .github/workflows/ci.yml && git commit && git push`

### 6.4 推送待办
最新提交（手写 CustomOp 双关节摆 + 文档）与此前 `abdef4e`、`972641a` 均在本地
master，网络恢复后 `git push`。

## 7. 常用命令

```bash
cargo test --workspace          # 40 套件全绿
cargo bench -p ad               # criterion 基准
cargo clippy --workspace --all-targets  # 零警告
cargo check --workspace --target wasm32-unknown-unknown
cargo run --release -p ad --example profile  # 分配画像 + 墙钟（--loop N 供 cdb 采样）
cd crates/ad-demo && trunk serve  # web demo → localhost:8080
git push                        # 推送（本地领先远端 3+ 提交）
```

## 8. 文件路径速查

| 文件 | 内容 |
|------|------|
| `docs/design.md` | 设计文档 v0.3 + §12 实现回写（12.3 有 27 条教训） |
| `docs/design.md` §4.3.1 | CustomOp backward 契约（内部边 + 单步 FD 规范） |
| `docs/design.md` §12.3 | 实现期偏差与教训（27 条，含全部 bug 复盘） |
| `crates/ad-core/src/context.rs` | Context 核心（backward_seeds/Jacobian/clip_grad） |
| `crates/ad-physics/src/ops.rs` | 6 个空间代数 CustomOp |
| `crates/ad-physics/src/contact.rs` | 接触力 CustomOp |
| `crates/ad-physics/src/chain.rs` | 手写 CustomOp 双关节摆（M⁻¹ 分解 VJP） |
| `crates/ad/examples/profile.rs` | 性能剖析用例（分配画像 + `--loop` 采样模式） |
| `crates/ad-physics/tests/ops_fd.rs` | FD 隔离器参考实现（新算子照此写） |
| `crates/ad-optim/src/ilqr.rs` | iLQR 求解器 |
| `crates/ad-optim/tests/e2e.rs` | 端到端收敛基准（GD + 接触 iLQR） |
| `crates/ad-optim/tests/chain.rs` | 双关节摆直通动力学 + 能量守恒 + iLQR + 手写算子对拍 |
| `crates/ad-checkpoint/src/manager.rs` | 嵌套反转实现（reverse_window 递归） |
