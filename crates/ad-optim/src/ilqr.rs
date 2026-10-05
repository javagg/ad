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
pub fn rollout<D: Dynamics>(
    d: &D,
    x0: &[f64],
    u: &[Vec<f64>],
    cost: &QuadraticCost,
) -> (Vec<Vec<f64>>, f64) {
    let t_total = u.len();
    let mut x = vec![vec![0.0; d.nx()]; t_total + 1];
    x[0].copy_from_slice(x0);
    let mut loss = 0.0;
    for t in 0..t_total {
        // 运行代价（x_t, u_t）
        for j in 0..d.nx() {
            let dx = x[t][j] - cost.goal[j];
            loss += 0.5 * cost.q[j] * dx * dx;
        }
        for j in 0..d.nu() {
            loss += 0.5 * cost.r[j] * u[t][j] * u[t][j];
        }
        // 前向（纯数值：以常量 AD 复用 Dynamics::step 表达式）
        let mut ctx = Context::<f64>::new();
        let xv: Vec<AD<f64>> = x[t].iter().map(|&v| AD::constant(v)).collect();
        let uv: Vec<AD<f64>> = u[t].iter().map(|&v| AD::constant(v)).collect();
        let out = d.step(&mut ctx, &xv, &uv);
        for j in 0..d.nx() {
            x[t + 1][j] = out[j].value;
        }
    }
    for j in 0..d.nx() {
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

            // 正则化求 k, K
            let mut q_uu_reg = q_uu.clone();
            for i in 0..nu {
                q_uu_reg[i][i] += mu;
            }
            let Some(kk) = solve_spd(&q_uu_reg, &q_u) else {
                backward_ok = false;
                break;
            };
            for i in 0..nu {
                ks[t][i] = -kk[i];
            }
            // K = −Q_uu_reg⁻¹ Q_ux：每列右端
            for j in 0..nx {
                let rhs: Vec<f64> = (0..nu).map(|i| q_ux[i][j]).collect();
                let Some(kcol) = solve_spd(&q_uu_reg, &rhs) else {
                    backward_ok = false;
                    break;
                };
                for i in 0..nu {
                    kmat[t][i][j] = -kcol[i];
                }
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
        while alpha >= 1e-3 {
            let mut u_new = u.clone();
            let mut x_new = vec![vec![0.0; nx]; t_total + 1];
            x_new[0].copy_from_slice(x0);
            for t in 0..t_total {
                for i in 0..nu {
                    let dx: Vec<f64> = (0..nx).map(|j| x_new[t][j] - x[t][j]).collect();
                    let k_term: f64 = (0..nx).map(|j| kmat[t][i][j] * dx[j]).sum::<f64>();
                    u_new[t][i] += alpha * ks[t][i] + k_term;
                }
                let mut ctx = Context::<f64>::new();
                let xv: Vec<AD<f64>> = x_new[t].iter().map(|&v| AD::constant(v)).collect();
                let uv: Vec<AD<f64>> = u_new[t].iter().map(|&v| AD::constant(v)).collect();
                let out = d.step(&mut ctx, &xv, &uv);
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
        if (-expected).abs() < cfg.tol {
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
        },
    )
}

/// 对称正定（含正则项）线性求解：部分主元 Gaussian 消元；奇异返回 None。
fn solve_spd(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m = a.to_vec();
    let mut y = b.to_vec();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| m[i][col].abs().partial_cmp(&m[j][col].abs()).unwrap())
            .unwrap_or(col);
        m.swap(col, piv);
        y.swap(col, piv);
        let d = m[col][col];
        if d.abs() < 1e-12 {
            return None;
        }
        for r in col + 1..n {
            let f = m[r][col] / d;
            for c in col..n {
                m[r][c] -= f * m[col][c];
            }
            y[r] -= f * y[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for c in i + 1..n {
            s -= m[i][c] * x[c];
        }
        x[i] = s / m[i][i];
    }
    Some(x)
}
