# HANDOFF — 项目状态与新 Session 接入指南

> 写于 2026-10-06。**项目收尾版 v0.5.0**：最后全量验证 **48 套件全绿**、
> clippy 零警告、wasm32 编译通过。42 条实现教训全部回写 design.md。
> 本文目标：新 session 零上下文即可继续，不丢任何关键决策/陷阱/路径。

## 1. 项目概要

**`ad`**：面向物理仿真的纯 Rust 反向模式自动微分库。设计文档 `docs/design.md`（v0.3，含完整 API 设计 + 实现回写 §12）。

**核心理念**：梯度不仅要算得对（多重 oracle 验证），还要能用（优化收敛基准）且健康（健康度分析）。

**仓库**：`https://github.com/javagg/ad.git`（公开）。注意：CI workflow 文件因 OAuth token 缺 `workflow` scope 暂存在 `.github/ci.yml.pending`，待授权后移回 `.github/workflows/ci.yml`。

## 2. Crate 架构（9 个，依赖方向自上而下）

```
ad-core    AD<S>标量（f64/f32）、Context、Tape（u32 节点 + 算子注册表）、CustomOp trait、线程局部挂载、双数oracle、no_grad/detach/异常检测
ad-ops     基础算子全表（含 sin/cos/tanh/atan2/clamp/lerp 等）+ bulk 向量（dot/axpy/norm2）
ad-custom  IFT 隐式求解模式（ImplicitSolve）、稠密线性求解
ad-checkpoint  Recomputable 状态机、快照调度（Uniform/Nested/Online/Custom）、分段反向+边界伴随、PendulumSim
ad-physics 空间代数算子（泛型 f64/f32：spatial.rs 辅助 + ops.rs 6 CustomOp + contact.rs 接触力 + chain.rs 双摆 + gyro.rs 陀螺）
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
| `9d1e682` | 手写 CustomOp 双关节摆补全（`ad-physics::chain::DoublePendulumStep`）+ iLQR 逐位对拍；文档 v0.3.4 |
| （本次 2） | IFT 泛化（矩形残差/warm-start/欠定拒绝，第 29 条）、iLQR box 约束 + MPC 滚动（第 30 条）、f32 标定（第 31 条）、rayon 批量示例（第 32 条）；文档 v0.3.6 |
| （本次 3） | 体系闭环（§12.3 第 33–36 条）：CustomOp 公开验证器、接触 IFT active-set 模式、健康度接入优化器（κ/μ/拒收计数 + GD 健康度画像）、CustomOp 随机图 fuzz；文档 v0.3.7 |
| （本次 4） | 规模实证（scale_stress，4000 维 × 10⁴ 步 0.93 s / 37.9 MiB）+ 用户教程 docs/guide.md；文档 v0.3.8 |
| （本次 5） | f32 物理算子泛型化（DoublePendulumStep + GyroscopicStep，f32/f64 对拍 + f32 能量守恒漂移 3.8e-3）；文档 v0.3.9 |
| （本次 6） | box-DDP control-limited backward pass（solve_kk_boxed，投影坐标下降；松界逐位等价 / 界宽单调 / 饱和断言）；文档 v0.4.0 |
| （本次 7） | f32 体系收尾（第 40 条）：接触算子泛型化 + 验证器泛型化（validate_custom_op<S>，f32 直接 FD 验证）；文档 v0.4.1 |
| （本次 8） | 项目收尾（第 41–42 条）：空间代数 f32 全量泛型化 + tape 瘦身/算子注册表（字节 −30~44%）；v0.5.0 |
| （本次） | 工具链与场景补全（§12.3 第 28 条）：条件数探针、matvec、**Custom 记录槽位路由潜伏 bug 修复**、记录瘦身（单摆 −59%/−63% 累计）、iLQR LU 多右端、10⁴ 步链场景、§5.4 验收 bench；文档 v0.3.5 |

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
iLQR 的 rollout/前向线搜索已改为 Context 跨步复用 + 缓冲区重填；高频 CustomOp 调用
应缓存 `Rc<dyn CustomOp>` 走 `call_custom_dyn`（`PendulumSim` 是示范）。

### 4.11 Custom 记录的梯度按原始槽位索引（§12.3 第 28b 条，潜伏 bug 教训）
`CustomOp::backward` 返回的 gins 长度 = `num_inputs`，与**原始槽位**一一对应；
context 路由必须按 `tracked` 的 slot 取 `gins[slot]`。早期实现用 zip 按位置配对，
常量输入恰好都在尾部时"碰巧对齐"——`matvec`（M 常量在前、v 在后）首踩：
λM 被错路由给 λv。**FD 隔离器应包含部分追踪形态用例**（部分输入常量、
且常量不在尾部）。

## 5. 测试体系总览（43 套件）

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

### 6.1 ~~火焰图性能工程（M5 收尾）~~ ✅ 已完成（2026-10-06，§12.3 第 26/28c 条）
方法论：Windows 无 perf/dtrace、samply 需管理员 → **计数分配器画像 + cdb
poor-man's profiler**（负载驱动 `cargo run --release -p ad --example profile`，
`--loop <秒>` 模式供 cdb 采样）。成果：criterion 分段反向 288→**106 µs**（−63%）、
全 tape 177→**73 µs**（−59%）；分配次数 checkpoint 7160→128、全 tape 4016→16。
修复：SmallVec 内联 4→8（CustomOp forward/backward、tape Custom 记录）、
`call_custom` 返回 SmallVec、PendulumSim 缓存 Rc 走 call_custom_dyn、
Custom 记录存连续输出 base（336→272B，反向直传 adjoints 切片）。

### 6.2 ~~真实铰链多体链（方向 3 手写 CustomOp 补全）~~ ✅ 已完成（2026-10-06，§12.3 第 27 条）
`ad-physics::chain::DoublePendulumStep`：inputs = [θ1,θ2,ω1,ω2,τ1,τ2,dt]，
outputs = [θ1',θ2',ω1',ω2']，VJP 按 **M⁻¹ 分解**推导（`s = M⁻¹λa`、`λr = s`、
`λM_ij = −s_i·a_j` 全和约定）。四层验证全过：FD 隔离器（4 状态）、能量守恒
（漂移 0.38%）、单步梯度/Jacobian 对拍 AD 直通（1e-9）、iLQR 甩摆基准
**逐位一致**（loss 2.7839→1.573964，7 迭代）。隔离器首跑抓出 2 处错误
（重力双角 θ1 依赖漏项 + 内部边 λdt 需用 λω'(总计)）——教训已入算子文档。
iLQR 对拍测试在 `ad-optim/tests/chain.rs`（ad-optim dev-dep ad-physics）。

### 6.3 ~~§5.3/§5.4 收尾~~ ✅ 已完成（2026-10-06，§12.3 第 28 条）
- 条件数探针 `ad_verify::condition_number_inf`（§4.3.3 文档引用兑现）+ ad-custom 接线；
- `matvec_with` bulk 算子（§4.3.4 M·v 行）——**首踩 Custom 记录槽位路由潜伏 bug**（§4.11）；
- 10⁴ 步 20 维链 Uniform checkpoint 场景（bit-exact 重跑 + 1e-10 全 tape 对拍 + 重算 1×）；
- §5.4 验收 bench `chain_step/*`：CustomOp vs 手写伴随递推 ≈ **2.4–3.0×，未达标**，
  偏差分析入 §12.3 第 28f 条（"最小算子"是该指标最坏情形；bulk 算子比值趋近 1×）。

### 6.4 ~~剩余方向盘点~~ ✅ 全部落地（2026-10-06 两轮，§12.3 第 29–36 条）

**第一轮（第 29–32 条）**：IFT 泛化（超定正规方程 + warm-start 显式槽位 + 欠定
拒绝，`ad-custom/tests/ift_rect.rs`）、iLQR box 约束 + MPC 滚动（`mpc.rs`）、
f32 标定（`f32.rs` + §5.1 表）、rayon 批量示例（`batch_rollout.rs`）。

**第二轮（第 33–36 条，体系闭环）**：
1. ~~CustomOp 公开验证器~~ ✅ `ad_verify::op_check::validate_custom_op`——一行
   调用：前向确定性 + gins 契约 + 四种追踪形态 FD 对拍；库内 7 算子全过，
   三类人为破坏全被抓（`ad-verify/tests/op_check.rs`）；
2. ~~接触 IFT~~ ✅ active-set 模式（1-DOF Hertz 接触 / 2×2 冲量 / 3×2 冗余超定 /
   warm-start λ 跨步），oracle = 对求解器本身 FD（`contact_ift.rs`）；
3. ~~健康度接入优化器~~ ✅ iLQR `quu_cond_max`/`mu_final`/`line_search_rejections`
   + GD `grad_health`；冗余双控制 R 扫描 κ 增长验证（`health.rs`）；
4. ~~CustomOp 随机图 fuzz~~ ✅ 3 算子 × 24 步随机 DAG × 512 例 vs 双数 oracle，
   部分追踪形态 + 先规划后执行 + 值模拟缩放防对消（`fuzz_custom.rs`）。

**仍开放的方向**（收尾后仅剩可选项）：crates.io 发布（用户指示暂缓）；
经典 Revolve / GradBench 跨工具锚点（研究性）。

**第四轮（第 41–42 条，项目收尾）**：
- ~~空间代数 f32 化~~ ✅ spatial 辅助模块 + 全部 6 算子泛型化；f64 回归全过 +
  f32 泛型验证器直接验证——**f32 覆盖至此为全量**（基础算子/bulk/全部物理算子）；
- ~~OpRecord arena~~ ✅ 落地为记录瘦身 + 算子注册表：NodeId usize→u32（AD
  24→16B）、Custom 记录去 Rc（注册表按指针去重，clear_tape 生命周期重置——
  漏掉时被 50k 轮零泄漏测试当场抓出）；确定性收益：full-tape 字节/run
  −44%、checkpoint −30%、scalar fresh −21%。墙钟 A/B 因本机噪声 2.5×
  不可信，如实记录（字节为确定性指标）。
**第三轮（第 37 条，规模实证与采用通道）**：
- ~~规模实证~~ ✅ `examples/scale_stress.rs`：稀疏链 4000 维 × 10⁴ 步
  checkpoint 分段反向 0.93 s / 峰值 37.9 MiB（内存随 T 亚线性 ✓）；
  稠密全耦合 400 维 × 10³ 步全 tape 18 ms。验证器在 dense 手写 VJP
  连抓 2 处（对角元 + dt 因子）；
- ~~用户侧集成教程~~ ✅ `docs/guide.md`（两条路径纪律 → CustomOp 契约 →
  验证 → checkpoint → 优化器健康度解读 → 常见错误速查表）；
  README 已链接。

### 6.5 CI workflow 恢复（需要用户操作）
`gh auth refresh -h github.com -s workflow` → 浏览器授权 →
`git mv .github/ci.yml.pending .github/workflows/ci.yml && git commit && git push`
### 6.6 推送状态
✅ 已全部推送（2026-10-06 晚网络恢复）：远端 master = `eb3c634`（box-DDP），
含体系闭环（33–36）、规模实证 + 教程（37）、f32 物理算子（38）。工作区干净。

## 7. 常用命令
`git push`（本地领先远端 3 提交）。此前提交（至 `8e12fce`）均已推送。工作区干净。

## 7. 常用命令

```bash
cargo test --workspace          # 43 套件全绿
cargo bench -p ad               # criterion 基准
cargo clippy --workspace --all-targets  # 零警告
cargo check --workspace --target wasm32-unknown-unknown
cargo run --release -p ad --example profile  # 分配画像 + 墙钟（--loop N 供 cdb 采样）
cd crates/ad-demo && trunk serve  # web demo → localhost:8080
git push                        # 推送
```

## 8. 文件路径速查

| 文件 | 内容 |
|------|------|
| `docs/design.md` | 设计文档 v0.3.8 + §12 实现回写（12.3 有 37 条教训） |
| `docs/design.md` §4.3.1 | CustomOp backward 契约（内部边 + 单步 FD 规范） |
| `docs/design.md` §12.3 | 实现期偏差与教训（37 条，含全部 bug 复盘） |
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
