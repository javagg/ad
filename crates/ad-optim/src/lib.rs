//! `ad-optim`：最小梯度优化器——端到端轨迹优化基准的工具端。
//!
//! 刻意保持零依赖、零 AD 耦合：`minimize_gradient_descent` 只见
//! `FnMut(&[f64], &mut [f64]) -> f64`（输入 → (损失, 梯度)）。AD 由使用方
//! 接线（见 `ad-optim` 的基准测试）。
//!
//! 优化器：梯度下降 + **Armijo 回溯线搜索** + 自适应步长伸缩。
//! 基准目的是检验"AD 梯度是否可用于优化"，而非追求 SOTA——一阶方法
//! 对梯度误差最敏感，正是合适的探针。

/// 优化器配置。
#[derive(Clone, Debug)]
pub struct OptimizerCfg {
    /// 最大迭代数
    pub max_iters: usize,
    /// 初始步长（每轮按 Armijo 接受性自适应伸缩）
    pub lr0: f64,
    /// Armijo 充分下降系数
    pub armijo_c: f64,
    /// 失败时步长收缩因子
    pub shrink: f64,
    /// 成功时步长增长因子（与 shrink 构成有界伸缩循环）
    pub grow: f64,
    /// 梯度无穷范数收敛阈值
    pub tol_grad: f64,
    /// 相对损失改进收敛阈值
    pub tol_rel_improve: f64,
}

impl Default for OptimizerCfg {
    fn default() -> Self {
        OptimizerCfg {
            max_iters: 300,
            lr0: 1.0,
            armijo_c: 1e-4,
            shrink: 0.5,
            grow: 1.5,
            tol_grad: 1e-6,
            tol_rel_improve: 1e-10,
        }
    }
}

/// 优化结果报告。
#[derive(Clone, Debug)]
pub struct OptReport {
    pub loss0: f64,
    pub loss: f64,
    pub iters: usize,
    /// ‖∇f‖∞（终止时）
    pub grad_norm: f64,
    /// 达到 tol_grad 或 tol_rel_improve
    pub converged: bool,
    /// 线搜索彻底失败（Armijo 拒绝一切步长）的迭代次数。
    /// 持续 > 8 次即停止——通常是梯度与损失不一致（AD 坏梯度的特征信号）。
    pub line_search_failures: usize,
}

/// 梯度下降 + Armijo 回溯线搜索。
///
/// `loss_and_grad(x, g)` 写入 `g` = ∇f(x) 并返回 f(x)。`g` 与 `x` 不得别名。
pub fn minimize_gradient_descent<F>(
    x0: &[f64],
    cfg: &OptimizerCfg,
    mut loss_and_grad: F,
) -> (Vec<f64>, OptReport)
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    assert!(!x0.is_empty(), "x0 must be non-empty");
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut x_trial = x0.to_vec();
    let mut g = vec![0.0; n];

    let loss0 = loss_and_grad(&x, &mut g);
    let mut loss = loss0;
    let mut norm = g.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let mut lr = cfg.lr0;
    let mut iters = 0usize;
    let mut consecutive_failures = 0usize;
    let mut failures = 0usize;
    let mut converged = norm < cfg.tol_grad;

    while iters < cfg.max_iters && !converged {
        // Armijo 回溯：d = −g，充分下降条件 f(x − t·g) ≤ f(x) − c·t·‖g‖²
        let gd = norm * norm;
        let x_base = x.clone();
        let mut t = lr;
        let mut accepted = false;
        let mut loss_new = loss;
        for _ in 0..60 {
            for i in 0..n {
                x_trial[i] = x_base[i] - t * g[i];
            }
            loss_new = loss_and_grad(&x_trial, &mut g);
            if loss_new.is_finite() && loss_new <= loss - cfg.armijo_c * t * gd {
                accepted = true;
                break;
            }
            t *= cfg.shrink;
        }
        if !accepted {
            failures += 1;
            consecutive_failures += 1;
            lr *= cfg.shrink;
            // 恢复 x 与梯度到基点（线搜索试探污染了 g）
            x.copy_from_slice(&x_base);
            loss = loss_and_grad(&x, &mut g);
            if consecutive_failures > 8 {
                break; // 梯度与损失不一致（AD 坏梯度的特征信号）
            }
            continue;
        }
        consecutive_failures = 0;
        x.copy_from_slice(&x_trial);
        let old_loss = loss;
        loss = loss_new;
        norm = g.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        lr = (lr * cfg.grow).min(1e12);
        iters += 1;
        converged = norm < cfg.tol_grad
            || (old_loss - loss).abs() <= cfg.tol_rel_improve * (1.0 + old_loss.abs());
    }

    (
        x,
        OptReport {
            loss0,
            loss,
            iters,
            grad_norm: norm,
            converged,
            line_search_failures: failures,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_converges() {
        // f = ½‖x − c‖²（10 维），最优 x = c
        let c: Vec<f64> = (0..10).map(|i| 1.0 + i as f64 * 0.3).collect();
        let f = |x: &[f64], g: &mut [f64]| -> f64 {
            let mut s = 0.0;
            for i in 0..x.len() {
                let d = x[i] - c[i];
                s += 0.5 * d * d;
                g[i] = d;
            }
            s
        };
        let x0 = vec![0.0; 10];
        let (x, rep) = minimize_gradient_descent(&x0, &OptimizerCfg::default(), f);
        assert!(rep.converged, "report: {rep:?}");
        for i in 0..10 {
            assert!((x[i] - c[i]).abs() < 1e-4, "x[{i}] = {}", x[i]);
        }
    }

    #[test]
    fn rosenbrock_converges_to_minimum() {
        // Rosenbrock（经典病态香蕉谷），最优 (1, 1)
        let f = |x: &[f64], g: &mut [f64]| -> f64 {
            let (a, b) = (1.0, 100.0);
            g[0] = -2.0 * (a - x[0]) - 4.0 * b * x[0] * (x[1] - x[0] * x[0]);
            g[1] = 2.0 * b * (x[1] - x[0] * x[0]);
            (a - x[0]).powi(2) + b * (x[1] - x[0] * x[0]).powi(2)
        };
        let cfg = OptimizerCfg {
            max_iters: 20000,
            lr0: 0.1,
            ..Default::default()
        };
        let (x, rep) = minimize_gradient_descent(&[-1.2, 1.0], &cfg, f);
        // 一阶 GD 在 Rosenbrock 弯曲谷地上收敛慢是已知特性，2e-2 为正常精度
        assert!((x[0] - 1.0).abs() < 2e-2, "x = {x:?}, report: {rep:?}");
        assert!((x[1] - 1.0).abs() < 2e-2, "x = {x:?}, report: {rep:?}");
        assert!(rep.loss < 1e-4, "loss = {}", rep.loss);
    }

    #[test]
    fn detects_inconsistent_gradient() {
        // 梯度与损失不一致（坏 AD 的模拟）：线搜索应连续失败并终止
        let f = |x: &[f64], g: &mut [f64]| -> f64 {
            g[0] = 1.0; // 恒定"梯度"，与损失无关
            0.5 * x[0] * x[0]
        };
        let cfg = OptimizerCfg {
            max_iters: 100,
            ..Default::default()
        };
        let (_x, rep) = minimize_gradient_descent(&[1.0], &cfg, f);
        assert!(rep.line_search_failures > 0, "report: {rep:?}");
        assert!(!rep.converged);
    }
}
