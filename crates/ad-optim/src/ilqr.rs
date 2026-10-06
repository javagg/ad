//! iLQR（迭代线性二次调节器）求解器——基于 `ad-core` 的动力学 Jacobian。
//!
//! 设计文档 §12.3 第 20 条标注的"后续工作"落地：iLQR 需要
//! ∂f/∂x、∂f/∂u（动力学 Jacobian），由 [`Dynamics`] 实现方在
//! `Context` 上记录前向，反向传播用单位种子逐列取出（`backward_seeds`）。
//!
//! 代价限定为标准二次型（Hessian 解析，无需二阶 AD）：
//! `½Σ(xₜ−x*)ᵀQ(xₜ−x*) + ½uₜᵀRuₜ + ½(x_T−x*)ᵀQ_f(x_T−x*)`。
//!
//! 算法：Tassa 正则化 iLQR（backward pass 用 μ-正则化的 Q_uu 求 k/K，
//! 前向 pass 对 α 做回溯线搜索，按预期下降 ΔV 验收；Q_uu 不可逆时增大 μ）。

use ad_core::{Context, AD};

/// 离散动力学：`x' = f(x, u)`。实现方在 `ctx` 上用 AD 表达式记录前向。
pub trait Dynamics {
    fn nx(&self) -> usize;
    fn nu(&self) -> usize;
    fn step(&self, ctx: &mut Context<f64>, x: &[AD<f64>], u: &[AD<f64>]) -> Vec<AD<f64>>;
}

/// iLQR 配置。
#[derive(Clone, Debug)]
pub struct IlqrCfg {
    pub max_iters: usize,
    /// Q_uu 正则化初始值
    pub mu0: f64,
    pub mu_max: f64,
    /// 正则化增/减因子
    pub mu_mult: f64,
    /// 前向线搜索 α 序列起点（1, α, α², …）
    pub alpha_shrink: f64,
    /// 期望下降的最低验收比例
    pub accept_ratio: f64,
    pub tol: f64,
    /// 控制下界（box 约束，None = 无界）。backward pass 为 control-limited
    /// 形式（Tassa et al. 2014：投影坐标下降解每步 box-QP，钳制维耦合进入
    /// 自由维标量解）；前向 pass 钳制 u_new，主动约束期 ΔV 预估仍可能偏乐观，
    /// 由 α 回溯的验收环节兜底（设计文档 §12.3 第 30/39 条）。
    pub u_min: Option<Vec<f64>>,
    /// 控制上界（与 u_min 等长）
    pub u_max: Option<Vec<f64>>,
}

impl Default for IlqrCfg {
    fn default() -> Self {
        IlqrCfg {
            max_iters: 60,
            mu0: 1.0,
            mu_max: 1e8,
            mu_mult: 10.0,
            alpha_shrink: 0.5,
            accept_ratio: 0.0,
            tol: 1e-6,
            u_min: None,
            u_max: None,
        }
    }
}

/// 二次代价权重（均为对角阵）。
pub struct QuadraticCost {
    /// 运行状态权重 Q（nx）
    pub q: Vec<f64>,
    /// 控制权重 R（nu）
    pub r: Vec<f64>,
    /// 终端状态权重 Q_f（nx）
    pub qf: Vec<f64>,
    /// 目标状态 x*（nx）
    pub goal: Vec<f64>,
}

/// iLQR 求解结果。
#[derive(Clone, Debug)]
pub struct IlqrReport {
    pub loss: f64,
    pub loss0: f64,
    pub iters: usize,
    pub converged: bool,
    /// 末次前向线搜索的 α
    pub last_alpha: f64,
    /// 退出时的 Q_uu 正则化强度 μ——被推到 `mu_max` 附近而收敛失败时，
    /// 通常是病态/混沌问题（梯度健康但问题难）的信号（§12.3 第 35 条）
    pub mu_final: f64,
    /// 前向线搜索的 α 拒收总次数——与 μ 升级互相印证的"实际下降 ≠ 预期下降"
    /// 信号（混沌接触问题的特征）
    pub line_search_rejections: usize,
    /// 全部时间步上 `κ∞(Q_uu_reg)` 的最大值（条件数探针，§12.3 第 28a 条）。
    /// Q_uu 病态 = 控制通道的有效二阶信息病态；inf 表示某步奇异（触发 μ 升级）。
    pub quu_cond_max: f64,
}

/// 在 (x, u) 处计算动力学 Jacobian：A[i][j] = ∂out_i/∂x_j，B[i][j] = ∂out_i/∂u_j。
/// 逐输出单位种子反向传播，`zero_grads` 隔离各列。
fn jacobians<D: Dynamics>(d: &D, x: &[f64], u: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let (nx, nu) = (d.nx(), d.nu());
    let mut ctx = Context::<f64>::new();
    let mut xv = Vec::with_capacity(nx);
    let mut uv = Vec::with_capacity(nu);
    let mut x_vars = Vec::with_capacity(nx);
    let mut u_vars = Vec::with_capacity(nu);
    for &v in x {
        let (ad, var) = ctx.var(v);
        xv.push(ad);
        x_vars.push(var);
    }
    for &v in u {
        let (ad, var) = ctx.var(v);
        uv.push(ad);
        u_vars.push(var);
    }
    let out = d.step(&mut ctx, &xv, &uv);

    let mut a = vec![vec![0.0; nx]; nx];
    let mut b = vec![vec![0.0; nu]; nx];
    for i in 0..nx {
        ctx.backward_seeds(&[(out[i], 1.0)]);
        for j in 0..nx {
            a[i][j] = ctx.grad(x_vars[j]).unwrap_or(0.0);
        }
        for j in 0..nu {
            b[i][j] = ctx.grad(u_vars[j]).unwrap_or(0.0);
        }
        ctx.zero_grads();
    }
    (a, b)
}

/// 沿 (U) 从 x0 rollout，返回逐时刻状态（含 x0，共 T+1）与损失。
///
/// 前向为纯数值路径（全部输入常量，tape 不增长）：Context 跨步复用、
/// 状态/控制缓冲区就地重填——rollout 在求解器内每迭代被调用 2 次，
/// 每步 Context::new 的分配热点（性能画像 §12.3 第 26 条）在此放大。
pub fn rollout<D: Dynamics>(
    d: &D,
    x0: &[f64],
    u: &[Vec<f64>],
    cost: &QuadraticCost,
) -> (Vec<Vec<f64>>, f64) {
    let (nx, nu) = (d.nx(), d.nu());
    let t_total = u.len();
    let mut x = vec![vec![0.0; nx]; t_total + 1];
    x[0].copy_from_slice(x0);
    let mut loss = 0.0;
    let mut ctx = Context::<f64>::new();
    let mut xv: Vec<AD<f64>> = Vec::with_capacity(nx);
    let mut uv: Vec<AD<f64>> = Vec::with_capacity(nu);
    for t in 0..t_total {
        // 运行代价（x_t, u_t）
        for j in 0..nx {
            let dx = x[t][j] - cost.goal[j];
            loss += 0.5 * cost.q[j] * dx * dx;
        }
        for j in 0..nu {
            loss += 0.5 * cost.r[j] * u[t][j] * u[t][j];
        }
        // 前向（纯数值：以常量 AD 复用 Dynamics::step 表达式）
        ctx.clear_tape();
        xv.clear();
        xv.extend(x[t].iter().map(|&v| AD::constant(v)));
        uv.clear();
        uv.extend(u[t].iter().map(|&v| AD::constant(v)));
        let out = d.step(&mut ctx, &xv, &uv);
        for j in 0..nx {
            x[t + 1][j] = out[j].value;
        }
    }
    for j in 0..nx {
        let dx = x[t_total][j] - cost.goal[j];
        loss += 0.5 * cost.qf[j] * dx * dx;
    }
    (x, loss)
}

/// iLQR 主循环。
pub fn solve_ilqr<D: Dynamics>(
    d: &D,
    x0: &[f64],
    u0: &[Vec<f64>],
    cost: &QuadraticCost,
    cfg: &IlqrCfg,
) -> (Vec<Vec<f64>>, IlqrReport) {
    let (nx, nu) = (d.nx(), d.nu());
    let t_total = u0.len();
    assert_eq!(x0.len(), nx, "x0 length must match dynamics nx");
    for u in u0 {
        assert_eq!(u.len(), nu, "control length must match dynamics nu");
    }

    let mut mu = cfg.mu0;
    let mut u = u0.to_vec();
    let (mut x, mut loss) = rollout(d, x0, &u, cost);
    let loss0 = loss;
    let mut iters = 0usize;
    let mut converged = false;
    let mut last_alpha = 0.0;
    let mut line_search_rejections = 0usize;
    let mut quu_cond_max = 0.0f64;

    for _ in 0..cfg.max_iters {
        if converged {
            break;
        }
        // ---- backward pass ----
        // 终端 V
        let mut v_x = vec![0.0; nx];
        let mut v_xx = vec![vec![0.0; nx]; nx];
        for j in 0..nx {
            v_x[j] = cost.qf[j] * (x[t_total][j] - cost.goal[j]);
            v_xx[j][j] = cost.qf[j];
        }
        let mut ks = vec![vec![0.0; nu]; t_total];
        let mut kmat = vec![vec![vec![0.0; nx]; nu]; t_total];
        let mut expected = 0.0;
        let mut backward_ok = true;

        for t in (0..t_total).rev() {
            let (a, b) = jacobians(d, &x[t], &u[t]);
            // Q 项
            let mut q_x = vec![0.0; nx];
            let mut q_u = vec![0.0; nu];
            for i in 0..nx {
                q_x[i] = cost.q[i] * (x[t][i] - cost.goal[i]);
                for j in 0..nx {
                    q_x[i] += a[j][i] * v_x[j]; // Aᵀ V_x：A[j][i] = ∂out_j/∂x_i
                }
            }
            for i in 0..nu {
                for j in 0..nx {
                    q_u[i] += b[j][i] * v_x[j];
                }
                q_u[i] += cost.r[i] * u[t][i];
            }
            let mut q_xx = vec![vec![0.0; nx]; nx];
            for i in 0..nx {
                q_xx[i][i] = cost.q[i];
            }
            // Aᵀ V_xx A：先算 V_xx A（nx×nx），再 Aᵀ 乘
            let mut vxx_a = vec![vec![0.0; nx]; nx];
            for i in 0..nx {
                for j in 0..nx {
                    let mut acc = 0.0;
                    for k in 0..nx {
                        acc += v_xx[i][k] * a[k][j];
                    }
                    vxx_a[i][j] = acc;
                }
            }
            for i in 0..nx {
                for j in 0..nx {
                    let mut acc = 0.0;
                    for k in 0..nx {
                        acc += a[k][i] * vxx_a[k][j];
                    }
                    q_xx[i][j] += acc;
                }
            }
            // Bᵀ V_xx A（nu×nx）
            let mut q_ux = vec![vec![0.0; nx]; nu];
            for i in 0..nu {
                for j in 0..nx {
                    let mut acc = 0.0;
                    for k in 0..nx {
                        acc += b[k][i] * vxx_a[k][j];
                    }
                    q_ux[i][j] = acc;
                }
            }
            // Q_uu = R + Bᵀ V_xx B
            let mut vxx_b = vec![vec![0.0; nu]; nx];
            for i in 0..nx {
                for j in 0..nu {
                    let mut acc = 0.0;
                    for k in 0..nx {
                        acc += v_xx[i][k] * b[k][j];
                    }
                    vxx_b[i][j] = acc;
                }
            }
            let mut q_uu = vec![vec![0.0; nu]; nu];
            for i in 0..nu {
                for j in 0..nu {
                    q_uu[i][j] = if i == j { cost.r[i] } else { 0.0 };
                    for k in 0..nx {
                        q_uu[i][j] += b[k][i] * vxx_b[k][j];
                    }
                }
            }

            // 正则化求 k, K：Q_uu_reg 只分解一次，k 与 K 的 nx 列右端
            // 共享同一 LU（每步 nx+1 次回代，替代此前 nx+1 次 O(n³) 重分解）
            let mut q_uu_reg = q_uu.clone();
            for i in 0..nu {
                q_uu_reg[i][i] += mu;
            }
            // 健康度探针：控制通道二阶信息的条件数（报告取全程最大值）
            let cond = ad_verify::condition_number_inf(&q_uu_reg);
            if cond > quu_cond_max {
                quu_cond_max = cond;
            }
            // k/K 求解：无界走 LU 精确路径（数值与历史逐位一致）；有界走
            // control-limited 迭代分解（Tassa et al. 2014，第 30a 条启发式的
            // 正规化——backward 感知边界，钳制维的耦合进入自由维的标量解）
            if !solve_kk_boxed(
                &q_uu,
                &q_u,
                &q_ux,
                &u[t],
                cfg.u_min.as_deref(),
                cfg.u_max.as_deref(),
                mu,
                &mut ks[t],
                &mut kmat[t],
            ) {
                backward_ok = false;
                break;
            }
            if !backward_ok {
                break;
            }

            // 预期下降与 V 更新（Tassa 完整形式）
            // QuuK = Q_uu_reg·K（nu×nx）、Quu_k = Q_uu_reg·k（nu）
            let mut quu_k = vec![vec![0.0; nx]; nu];
            let mut quu_k1 = vec![0.0; nu];
            for p in 0..nu {
                quu_k1[p] = (0..nu).map(|q| q_uu_reg[p][q] * ks[t][q]).sum::<f64>();
                for j in 0..nx {
                    quu_k[p][j] = (0..nu).map(|q| q_uu_reg[p][q] * kmat[t][q][j]).sum::<f64>();
                }
            }
            for i in 0..nu {
                expected += ks[t][i] * q_u[i];
                expected += 0.5 * ks[t][i] * quu_k1[i];
            }
            // V_x = Q_x + KᵀQ_uu k + KᵀQ_u + Q_uxᵀ k
            // V_xx = Q_xx + KᵀQ_uu K + KᵀQ_ux + Q_uxᵀ K
            for i in 0..nx {
                let mut vx = q_x[i];
                for p in 0..nu {
                    vx +=
                        kmat[t][p][i] * quu_k1[p] + kmat[t][p][i] * q_u[p] + q_ux[p][i] * ks[t][p];
                }
                v_x[i] = vx;
            }
            for i in 0..nx {
                for j in 0..nx {
                    let mut vxx = q_xx[i][j];
                    for p in 0..nu {
                        vxx += kmat[t][p][i] * quu_k[p][j]
                            + kmat[t][p][i] * q_ux[p][j]
                            + q_ux[p][i] * kmat[t][p][j];
                    }
                    v_xx[i][j] = vxx;
                }
            }
        }

        if !backward_ok {
            mu *= cfg.mu_mult;
            if mu > cfg.mu_max {
                break;
            }
            continue;
        }

        // ---- forward pass（α 回溯） ----
        let mut alpha = 1.0f64;
        let mut accepted = false;
        let mut ctx_fwd = Context::<f64>::new();
        let mut xv: Vec<AD<f64>> = Vec::with_capacity(nx);
        let mut uv: Vec<AD<f64>> = Vec::with_capacity(nu);
        while alpha >= 1e-3 {
            let mut u_new = u.clone();
            let mut x_new = vec![vec![0.0; nx]; t_total + 1];
            x_new[0].copy_from_slice(x0);
            for t in 0..t_total {
                for i in 0..nu {
                    let dx: Vec<f64> = (0..nx).map(|j| x_new[t][j] - x[t][j]).collect();
                    let k_term: f64 = (0..nx).map(|j| kmat[t][i][j] * dx[j]).sum::<f64>();
                    u_new[t][i] += alpha * ks[t][i] + k_term;
                    // box 约束：前向钳制 + control-limited backward（§12.3 第 39 条）
                    if let Some(lo) = &cfg.u_min {
                        if u_new[t][i] < lo[i] {
                            u_new[t][i] = lo[i];
                        }
                    }
                    if let Some(hi) = &cfg.u_max {
                        if u_new[t][i] > hi[i] {
                            u_new[t][i] = hi[i];
                        }
                    }
                }
                ctx_fwd.clear_tape();
                xv.clear();
                xv.extend(x_new[t].iter().map(|&v| AD::constant(v)));
                uv.clear();
                uv.extend(u_new[t].iter().map(|&v| AD::constant(v)));
                let out = d.step(&mut ctx_fwd, &xv, &uv);
                for j in 0..nx {
                    x_new[t + 1][j] = out[j].value;
                }
            }
            let (_, loss_new) = rollout(d, x0, &u_new, cost);
            if loss_new.is_finite() && loss_new <= loss + cfg.accept_ratio * alpha * expected {
                u = u_new;
                x = x_new;
                loss = loss_new;
                last_alpha = alpha;
                accepted = true;
                break;
            }
            line_search_rejections += 1;
            alpha *= cfg.alpha_shrink;
        }

        iters += 1;
        if !accepted {
            mu *= cfg.mu_mult;
            if mu > cfg.mu_max {
                break;
            }
            continue;
        }
        mu /= cfg.mu_mult;
        mu = mu.max(1e-9);
        // 收敛判据：预期下降归零 **且** 正则已基本关闭——μ 大时 k 被压缩，
        // expected 小并不代表到达最优
        if (-expected).abs() < cfg.tol && mu <= 1e-6 {
            converged = true;
        }
    }

    (
        u,
        IlqrReport {
            loss,
            loss0,
            iters,
            converged,
            last_alpha,
            mu_final: mu,
            line_search_rejections,
            quu_cond_max,
        },
    )
}

/// 单步 k/K 求解（box-DDP，Tassa et al. 2014）。
///
/// - **无界**（u_min/u_max 均为 None）：LU 精确路径，数值与历史实现逐位一致；
/// - **有界**：自由/钳制维迭代分解——k 从 0 起步，逐控制维检查
///   `u_i + k_i` 是否越界：界内做标量除法解（`k_i = −Q_u_f/i / Q_uu_ii`）
///   并以 `Δk_i` 更新 `Q_u_f` 的耦合；越界则钳制 `k_i` 到界并标记维。
///   维集合稳定且 k 收敛后，K 的钳制行置零、自由行由自由子块的 LU 给出
///   （`K_free = −Q_uu_ff⁻¹ Q_ux_f`）。正则化 μ 已含在传入的 q_uu 中？
///   否——本函数内部加 μ（与调用方的条件数探针共享同一 reg 值）。
///
/// 返回 false = 数值失败（主元奇异），调用方升级 μ 重试。
#[allow(clippy::too_many_arguments)]
fn solve_kk_boxed(
    q_uu: &[Vec<f64>],
    q_u: &[f64],
    q_ux: &[Vec<f64>],
    u: &[f64],
    u_min: Option<&[f64]>,
    u_max: Option<&[f64]>,
    mu: f64,
    ks_out: &mut [f64],
    kmat_out: &mut [Vec<f64>],
) -> bool {
    let nu = q_u.len();
    let nx = if kmat_out.is_empty() { 0 } else { kmat_out[0].len() };
    let mut q_uu_reg = q_uu.to_vec();
    for i in 0..nu {
        q_uu_reg[i][i] += mu;
    }

    let (Some(lo), Some(hi)) = (u_min, u_max) else {
        let Some(lu) = lu_factor(&q_uu_reg) else {
            return false;
        };
        let kk = lu_solve(&lu, q_u);
        for i in 0..nu {
            ks_out[i] = -kk[i];
        }
        for j in 0..nx {
            let rhs: Vec<f64> = (0..nu).map(|i| q_ux[i][j]).collect();
            let kcol = lu_solve(&lu, &rhs);
            for i in 0..nu {
                kmat_out[i][j] = -kcol[i];
            }
        }
        return true;
    };

    // ---- control-limited：投影坐标下降（box-QP，Quu_reg ≽ 0 保证收敛）----
    // 维护增量形式 qu_f = Q_u + Q_uu_reg·k；沿第 i 维的无约束极小
    // k_i^unc = k_i − qu_f_i/Q_uu_ii，越界则投影到界。收敛后以"处于界上"
    // 的维划分自由/钳制集。
    let mut k = vec![0.0f64; nu];
    let mut qu_f: Vec<f64> = q_u.to_vec();
    for _ in 0..(6 * nu + 6) {
        let mut max_dk = 0.0f64;
        for i in 0..nu {
            let k_unc = k[i] - qu_f[i] / q_uu_reg[i][i];
            let u_cand = u[i] + k_unc;
            let k_new = if u_cand < lo[i] {
                lo[i] - u[i]
            } else if u_cand > hi[i] {
                hi[i] - u[i]
            } else {
                k_unc
            };
            let delta = k_new - k[i];
            if delta != 0.0 {
                for r in 0..nu {
                    qu_f[r] += q_uu_reg[r][i] * delta;
                }
                k[i] = k_new;
                let ad = delta.abs();
                if ad > max_dk {
                    max_dk = ad;
                }
            }
        }
        if max_dk < 1e-10 {
            break;
        }
    }

    for i in 0..nu {
        ks_out[i] = k[i];
        for j in 0..nx {
            kmat_out[i][j] = 0.0;
        }
    }

    // K：自由行（未处于界上的维）= −Q_uu_ff⁻¹ Q_ux_f（保留 μ 正则的自由子块）；
    // 钳制行零
    let free: Vec<usize> = (0..nu)
        .filter(|&i| ((u[i] + k[i]) - lo[i]).abs() > 1e-9 && ((u[i] + k[i]) - hi[i]).abs() > 1e-9)
        .collect();
    if free.is_empty() {
        return true;
    }
    let nf = free.len();
    let a: Vec<Vec<f64>> = (0..nf)
        .map(|p| (0..nf).map(|q| q_uu_reg[free[p]][free[q]]).collect())
        .collect();
    let Some(lu) = lu_factor(&a) else {
        return false;
    };
    for j in 0..nx {
        let rhs: Vec<f64> = free.iter().map(|&i| q_ux[i][j]).collect();
        let kcol = lu_solve(&lu, &rhs);
        for (p, &i) in free.iter().enumerate() {
            kmat_out[i][j] = -kcol[p];
        }
    }
    true
}

/// LU 分解（部分主元，就地存 L 下三角于消元块）：奇异主元（< 1e-12）返回 None。
/// 与旧 solve_spd 的消元次序一致（同一主元选择 + 同序回代），数值结果逐位等价。
fn lu_factor(a: &[Vec<f64>]) -> Option<(Vec<Vec<f64>>, Vec<usize>)> {    let n = a.len();
    let mut m = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| m[i][col].abs().partial_cmp(&m[j][col].abs()).unwrap())
            .unwrap_or(col);
        m.swap(col, piv);
        perm.swap(col, piv);
        let d = m[col][col];
        if d.abs() < 1e-12 {
            return None;
        }
        for r in col + 1..n {
            let f = m[r][col] / d;
            m[r][col] = f; // L 因子
            for c in col + 1..n {
                m[r][c] -= f * m[col][c];
            }
        }
    }
    Some((m, perm))
}

/// 解 `LU x = P b`（前代 + 回代）。
fn lu_solve(lu: &(Vec<Vec<f64>>, Vec<usize>), b: &[f64]) -> Vec<f64> {
    let (m, perm) = lu;
    let n = m.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[perm[i]];
        for j in 0..i {
            s -= m[i][j] * y[j];
        }
        y[i] = s;
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in i + 1..n {
            s -= m[i][j] * x[j];
        }
        x[i] = s / m[i][i];
    }
    x
}
