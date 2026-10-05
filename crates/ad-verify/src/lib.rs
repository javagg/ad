//! `ad-verify`：梯度验证工具（设计文档 §4.5）。
//!
//! - [`GradientChecker::check_scalar`]：逐坐标中心差分；
//! - [`GradientChecker::check_random_direction`]：随机方向导数对比（高维友好）；
//! - [`GradientChecker::analyze_health`]：梯度健康度（范数/零占比/非有限占比）；
//! - [`GradientChecker::check_trajectory_stability`]：轨迹梯度发散/消失/振荡判定；
//! - [`GradientChecker::check_differentiability`]：∇Fuzz 式邻居采样，区分
//!   "梯度算错"与"该点本身不可微"（设计文档 §4.5.2）。
//!
//! 内部随机数使用确定性 xorshift（可复现，无外部依赖）。

/// 中心差分梯度验证器。
#[derive(Clone, Debug)]
pub struct GradientChecker {
    /// 差分步长（f64 推荐 ~1e-6）
    pub eps: f64,
    /// 相对误差容差
    pub rel_tolerance: f64,
    /// 绝对误差容差
    pub abs_tolerance: f64,
}

impl Default for GradientChecker {
    fn default() -> Self {
        GradientChecker {
            eps: 1e-6,
            rel_tolerance: 1e-5,
            abs_tolerance: 1e-8,
        }
    }
}

/// 单坐标验证明细。
#[derive(Clone, Copy, Debug)]
pub struct CheckDetail {
    pub index: usize,
    pub analytical: f64,
    pub numerical: f64,
    pub abs_error: f64,
    pub rel_error: f64,
    pub passed: bool,
}

/// 验证结果。
#[derive(Clone, Debug)]
pub struct CheckResult {
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub passed: bool,
    pub details: Vec<CheckDetail>,
}

/// 梯度健康度（设计文档 §4.5.3：验证"梯度是否有用"，而非仅"算对"）。
#[derive(Clone, Copy, Debug)]
pub struct GradientHealth {
    /// 梯度 L2 范数
    pub norm: f64,
    /// 与参考梯度的余弦相似度（无参考时为 `None`）
    pub cosine_similarity: Option<f64>,
    /// 零元素占比
    pub zero_fraction: f64,
    /// 非有限值占比
    pub nonfinite_fraction: f64,
}

/// 轨迹梯度稳定性判定。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StabilityVerdict {
    Stable,
    Vanishing,
    Exploding,
    Oscillating,
    NonFinite,
}

#[derive(Clone, Debug)]
pub struct TrajectoryStability {
    pub norms: Vec<f64>,
    pub verdict: StabilityVerdict,
    /// 末段/首段范数比（对数尺度增长因子）
    pub growth_ratio: f64,
}

/// 可微性（非光滑点）检查报告。
#[derive(Clone, Copy, Debug)]
pub struct NonSmoothnessReport {
    /// 邻居梯度与中心点梯度的最大相对偏差
    pub max_relative_deviation: f64,
    /// true = 邻域内梯度一致（未发现非光滑迹象）
    pub consistent: bool,
}

/// 确定性 xorshift64* 随机数（可复现采样）。
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// [0, 1) 均匀
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// [-1, 1] 均匀
    pub fn next_signed(&mut self) -> f64 {
        2.0 * self.next_f64() - 1.0
    }
}

impl GradientChecker {
    pub fn new(eps: f64, rel_tolerance: f64, abs_tolerance: f64) -> Self {
        GradientChecker {
            eps,
            rel_tolerance,
            abs_tolerance,
        }
    }

    fn central_diff(&self, f: &dyn Fn(&[f64]) -> f64, x: &mut [f64], i: usize) -> f64 {
        let h = self.eps;
        let orig = x[i];
        x[i] = orig + h;
        let fp = f(x);
        x[i] = orig - h;
        let fm = f(x);
        x[i] = orig;
        (fp - fm) / (2.0 * h)
    }

    /// 逐坐标中心差分验证 `f: R^n -> R` 的解析梯度。
    pub fn check_scalar<F>(&self, f: F, x: &[f64], analytical_grad: &[f64]) -> CheckResult
    where
        F: Fn(&[f64]) -> f64,
    {
        assert_eq!(x.len(), analytical_grad.len());
        let mut fx = x.to_vec();
        let mut details = Vec::with_capacity(x.len());
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        let mut all_passed = true;

        for i in 0..x.len() {
            let num = self.central_diff(&f, &mut fx, i);
            let ana = analytical_grad[i];
            let abs_err = (ana - num).abs();
            let rel_err = abs_err / ana.abs().max(num.abs()).max(1.0);
            let passed = abs_err <= self.abs_tolerance || rel_err <= self.rel_tolerance;
            all_passed &= passed;
            max_abs = max_abs.max(abs_err);
            max_rel = max_rel.max(rel_err);
            details.push(CheckDetail {
                index: i,
                analytical: ana,
                numerical: num,
                abs_error: abs_err,
                rel_error: rel_err,
                passed,
            });
        }

        CheckResult {
            max_abs_error: max_abs,
            max_rel_error: max_rel,
            passed: all_passed,
            details,
        }
    }

    /// 随机方向验证：n_directions 个随机单位方向上，方向导数 vs 中心差分。
    /// 高维输入的廉价替代（设计文档 §4.5.1）。
    pub fn check_random_direction<F>(
        &self,
        f: F,
        x: &[f64],
        analytical_grad: &[f64],
        n_directions: usize,
    ) -> CheckResult
    where
        F: Fn(&[f64]) -> f64,
    {
        assert_eq!(x.len(), analytical_grad.len());
        let mut rng = Rng::new(0x9E3779B97F4A7C15);
        let mut fx = x.to_vec();
        let mut details = Vec::with_capacity(n_directions);
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        let mut all_passed = true;

        for d in 0..n_directions {
            // 随机单位向量
            let mut v: Vec<f64> = (0..x.len()).map(|_| rng.next_signed()).collect();
            let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
            if norm == 0.0 {
                continue;
            }
            for a in &mut v {
                *a /= norm;
            }
            let h = self.eps;
            for (i, &a) in v.iter().enumerate() {
                fx[i] = x[i] + h * a;
            }
            let fp = f(&fx);
            for (i, &a) in v.iter().enumerate() {
                fx[i] = x[i] - h * a;
            }
            let fm = f(&fx);
            let num = (fp - fm) / (2.0 * h);
            let ana: f64 = v.iter().zip(analytical_grad).map(|(a, g)| a * g).sum();

            let abs_err = (ana - num).abs();
            let rel_err = abs_err / ana.abs().max(num.abs()).max(1.0);
            let passed = abs_err <= self.abs_tolerance || rel_err <= self.rel_tolerance;
            all_passed &= passed;
            max_abs = max_abs.max(abs_err);
            max_rel = max_rel.max(rel_err);
            details.push(CheckDetail {
                index: d,
                analytical: ana,
                numerical: num,
                abs_error: abs_err,
                rel_error: rel_err,
                passed,
            });
        }

        CheckResult {
            max_abs_error: max_abs,
            max_rel_error: max_rel,
            passed: all_passed,
            details,
        }
    }

    /// 梯度健康度（无参考方向）。
    pub fn analyze_health(&self, grad: &[f64]) -> GradientHealth {
        self.analyze_health_vs(grad, None)
    }

    /// 梯度健康度（带参考梯度，计算余弦相似度）。
    pub fn analyze_health_vs(&self, grad: &[f64], reference: Option<&[f64]>) -> GradientHealth {
        let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        let zero_fraction = if grad.is_empty() {
            0.0
        } else {
            grad.iter().filter(|g| **g == 0.0).count() as f64 / grad.len() as f64
        };
        let nonfinite_fraction = if grad.is_empty() {
            0.0
        } else {
            grad.iter().filter(|g| !g.is_finite()).count() as f64 / grad.len() as f64
        };
        let cosine = reference.map(|r| {
            let rn = r.iter().map(|g| g * g).sum::<f64>().sqrt();
            if norm == 0.0 || rn == 0.0 {
                return 0.0;
            }
            grad.iter().zip(r).map(|(a, b)| a * b).sum::<f64>() / (norm * rn)
        });
        GradientHealth {
            norm,
            cosine_similarity: cosine,
            zero_fraction,
            nonfinite_fraction,
        }
    }

    /// 轨迹梯度稳定性（逐步叶子梯度范数序列）。
    /// 判定为启发式，阈值基于对数尺度（设计文档 §4.5.3）。
    pub fn check_trajectory_stability(&self, grads_per_step: &[Vec<f64>]) -> TrajectoryStability {
        let norms: Vec<f64> = grads_per_step
            .iter()
            .map(|g| g.iter().map(|v| v * v).sum::<f64>().sqrt())
            .collect();
        let growth = match (norms.first(), norms.last()) {
            (Some(&f), Some(&l)) if f > 0.0 => l / f,
            _ => f64::NAN,
        };
        let verdict = if norms.is_empty() || norms.iter().any(|n| !n.is_finite()) {
            StabilityVerdict::NonFinite
        } else {
            let ln: Vec<f64> = norms.iter().map(|n| n.max(1e-300).ln()).collect();
            let total = ln.last().unwrap() - ln[0];
            if total > 9.2 {
                // > e^9.2 ≈ 1e4 倍
                StabilityVerdict::Exploding
            } else if total < -9.2 {
                StabilityVerdict::Vanishing
            } else {
                // 相邻步范数比的最大绝对对数跳变
                let max_jump = ln
                    .windows(2)
                    .map(|w| (w[1] - w[0]).abs())
                    .fold(0.0f64, f64::max);
                if max_jump > 4.6 {
                    // 单步跳变 > 100 倍
                    StabilityVerdict::Oscillating
                } else {
                    StabilityVerdict::Stable
                }
            }
        };
        TrajectoryStability {
            norms,
            verdict,
            growth_ratio: growth,
        }
    }

    /// ∇Fuzz 式可微性检查（设计文档 §4.5.2）：在 x 附近采样邻居
    /// `x + U(-δ, +δ)`，比较邻居处与中心点的差分梯度。梯度跳变（非光滑）
    /// 会表现为显著的相对偏差。
    pub fn check_differentiability<F>(
        &self,
        f: F,
        x: &[f64],
        n_samples: usize,
        delta: f64,
        tolerance: f64,
    ) -> NonSmoothnessReport
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = Rng::new(0xDEADBEEFCAFEBABE);
        let mut fx = x.to_vec();
        let base: Vec<f64> = (0..x.len())
            .map(|i| self.central_diff(&f, &mut fx, i))
            .collect();

        let mut max_dev = 0.0f64;
        for _ in 0..n_samples {
            let mut nb = x.to_vec();
            for v in &mut nb {
                *v += delta * rng.next_signed();
            }
            for i in 0..nb.len() {
                let num = self.central_diff(&f, &mut nb, i);
                let dev = (num - base[i]).abs() / (base[i].abs() + 1.0);
                max_dev = max_dev.max(dev);
            }
        }
        NonSmoothnessReport {
            max_relative_deviation: max_dev,
            consistent: max_dev <= tolerance,
        }
    }
}
