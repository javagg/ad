//! M4 验收（设计文档 §4.5）：有限差分、随机方向、健康度、轨迹稳定性、可微性检查。

use ad_verify::{GradientChecker, StabilityVerdict};

fn rosenbrock(x: &[f64]) -> f64 {
    let (a, b) = (1.0, 100.0);
    let (x, y) = (x[0], x[1]);
    (a - x) * (a - x) + b * (y - x * x) * (y - x * x)
}

fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
    let b = 100.0;
    let (x, y) = (x[0], x[1]);
    vec![
        -2.0 * (1.0 - x) - 4.0 * b * x * (y - x * x),
        2.0 * b * (y - x * x),
    ]
}

#[test]
fn check_scalar_passes_correct_gradient() {
    let checker = GradientChecker::default();
    let x = [0.3, -0.7];
    let r = checker.check_scalar(rosenbrock, &x, &rosenbrock_grad(&x));
    assert!(r.passed, "details: {:?}", r.details);
    assert!(r.max_rel_error < 1e-8);
}

#[test]
fn check_scalar_catches_wrong_gradient() {
    let checker = GradientChecker::default();
    let x = [0.3, -0.7];
    let mut wrong = rosenbrock_grad(&x);
    wrong[1] *= 1.1; // 10% 偏差
    let r = checker.check_scalar(rosenbrock, &x, &wrong);
    assert!(!r.passed);
}

#[test]
fn check_random_direction_high_dimensional() {
    // n = 50：逐坐标差分 100 次前向 vs 随机方向 8 次前向
    let checker = GradientChecker::default();
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 - n as f64 / 2.0) * 0.01).collect();
    // f = Σ xᵢ²·(i+1)
    let f = |x: &[f64]| {
        x.iter()
            .enumerate()
            .map(|(i, v)| v * v * (i + 1) as f64)
            .sum()
    };
    let grad: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, v)| 2.0 * v * (i + 1) as f64)
        .collect();

    let r = checker.check_random_direction(f, &x, &grad, 8);
    assert!(r.passed, "details: {:?}", r.details);
}

#[test]
fn analyze_health_reports_fractions() {
    let checker = GradientChecker::default();
    let h = checker.analyze_health(&[3.0, 0.0, -4.0, 0.0]);
    assert!((h.norm - 5.0).abs() < 1e-12);
    assert!((h.zero_fraction - 0.5).abs() < 1e-12);
    assert_eq!(h.nonfinite_fraction, 0.0);
    assert!(h.cosine_similarity.is_none());

    let h2 = checker.analyze_health_vs(&[1.0, 0.0], Some(&[2.0, 0.0]));
    assert!((h2.cosine_similarity.unwrap() - 1.0).abs() < 1e-12);

    let h3 = checker.analyze_health(&[1.0, f64::NAN]);
    assert!((h3.nonfinite_fraction - 0.5).abs() < 1e-12);
}

#[test]
fn trajectory_stability_verdicts() {
    let checker = GradientChecker::default();
    let mk = |norm: f64| vec![norm / 2.0f64.sqrt(), norm / 2.0f64.sqrt()];

    // 稳定：范数恒定
    let stable: Vec<Vec<f64>> = (0..20).map(|_| mk(1.0)).collect();
    assert_eq!(
        checker.check_trajectory_stability(&stable).verdict,
        StabilityVerdict::Stable
    );

    // 爆炸：每步 ×10
    let exploding: Vec<Vec<f64>> = (0..12).map(|i| mk(10f64.powi(i))).collect();
    assert_eq!(
        checker.check_trajectory_stability(&exploding).verdict,
        StabilityVerdict::Exploding
    );

    // 消失：每步 ×0.1
    let vanishing: Vec<Vec<f64>> = (0..12).map(|i| mk(10f64.powi(-i))).collect();
    assert_eq!(
        checker.check_trajectory_stability(&vanishing).verdict,
        StabilityVerdict::Vanishing
    );

    // 非有限
    let nonfinite = vec![mk(1.0), vec![f64::NAN, 0.0]];
    assert_eq!(
        checker.check_trajectory_stability(&nonfinite).verdict,
        StabilityVerdict::NonFinite
    );
}

#[test]
fn differentiability_check_distinguishes_kink_from_smooth() {
    let checker = GradientChecker::default();

    // 光滑点：|x|³ 在 x=1 处 C²
    let f_smooth = |x: &[f64]| x[0].abs().powi(3);
    let r = checker.check_differentiability(f_smooth, &[1.0], 20, 1e-4, 1e-3);
    assert!(r.consistent, "deviation {}", r.max_relative_deviation);

    // 非光滑点：|x| 在 x=0 处梯度跳变
    let f_kink = |x: &[f64]| x[0].abs();
    let r = checker.check_differentiability(f_kink, &[0.0], 20, 1e-4, 1e-3);
    assert!(!r.consistent, "deviation {}", r.max_relative_deviation);
}

#[test]
fn deterministic_rng_reproduces() {
    let checker = GradientChecker::default();
    let x = [0.3, -0.7];
    let g = rosenbrock_grad(&x);
    let a = checker.check_random_direction(rosenbrock, &x, &g, 4);
    let b = checker.check_random_direction(rosenbrock, &x, &g, 4);
    for (da, db) in a.details.iter().zip(b.details.iter()) {
        assert_eq!(da.numerical.to_bits(), db.numerical.to_bits());
    }
}

// ---- Taylor 余项测试（设计文档 §4.5.1，科学计算社区标准验收方法） ----

use ad_verify::TaylorReport;

#[test]
fn taylor_test_passes_for_correct_gradient() {
    let checker = GradientChecker::default();
    let x = [0.3, -0.7];
    let g = rosenbrock_grad(&x);
    let report = checker.taylor_test(rosenbrock, &x, &g, None);
    assert!(report.passed, "ratios: {:?}", report.ratios);
    // 正确一阶梯度：ratio(h) = O(h) → 收敛阶 ≈ 1
    assert!(
        (report.estimated_order - 1.0).abs() < 0.25,
        "estimated order {}",
        report.estimated_order
    );
}

#[test]
fn taylor_test_fails_for_scaled_gradient() {
    let checker = GradientChecker::default();
    let x = [0.3, -0.7];
    let mut wrong = rosenbrock_grad(&x);
    for w in &mut wrong {
        *w *= 1.1; // 系数错 10%：ratio → 非零常数，order ≈ 0
    }
    let report = checker.taylor_test(rosenbrock, &x, &wrong, None);
    assert!(!report.passed);
    assert!(
        report.estimated_order < 0.5,
        "order {}",
        report.estimated_order
    );
}

#[test]
fn taylor_test_fails_for_rotated_gradient() {
    let checker = GradientChecker::default();
    let x = [0.3, -0.7];
    let g = rosenbrock_grad(&x);
    // 方向错（分量互换）：与真梯度成大角度 → ratio 不衰减
    let wrong = [g[1], g[0]];
    let report = checker.taylor_test(rosenbrock, &x, &wrong, None);
    assert!(!report.passed);
}

#[test]
fn taylor_test_fails_at_kink() {
    let checker = GradientChecker::default();
    // |x| 在 0 处（PAP 约定梯度为 0）：ratio ≡ 1，order ≈ 0
    let report = checker.taylor_test(|x: &[f64]| x[0].abs(), &[0.0], &[0.0], Some(&[1.0]));
    assert!(!report.passed, "ratios: {:?}", report.ratios);
}

#[test]
fn taylor_test_explicit_direction() {
    let checker = GradientChecker::default();
    let f = |x: &[f64]| (x[0] * x[1] + x[2]).sin();
    let x = [1.0, 2.0, 3.0];
    let g = [2.0 * 5.0f64.cos(), 1.0 * 5.0f64.cos(), 5.0f64.cos()];
    let report: TaylorReport = checker.taylor_test(f, &x, &g, Some(&[0.6, 0.8, 0.0]));
    assert!(report.passed);
}

// ============================================================ 条件数探针

use ad_verify::condition_number_inf;

#[test]
fn condition_number_diagonal_exact() {
    // diag(1, 2, 4)：‖A‖∞ = 4，‖A⁻¹‖∞ = 1 → κ∞ = 4
    let a = vec![vec![1.0, 0.0, 0.0], vec![0.0, 2.0, 0.0], vec![0.0, 0.0, 4.0]];
    let k = condition_number_inf(&a);
    assert!((k - 4.0).abs() < 1e-12, "κ∞ = {k}");
}

#[test]
fn condition_number_scaled_identity_is_one() {
    // 缩放不变性：κ∞(αI) = 1（良态），误差按 α² 放大也不改变条件数
    let a = vec![vec![1e6, 0.0], vec![0.0, 1e-6]];
    // 注意：κ∞(diag(1e6, 1e-6)) = 1e6 · 1e6 = 1e12（各向异性缩放是病态）
    let k = condition_number_inf(&a);
    assert!((k - 1e12).abs() < 1e6, "κ∞ = {k}");
}

#[test]
fn condition_number_hilbert_ill_conditioned() {
    // 4 阶 Hilbert 矩阵的经典病态（2-范数条件数 ≈ 1.55e4，∞-范数同量级）
    let a: Vec<Vec<f64>> = (0..4)
        .map(|i| (0..4).map(|j| 1.0 / ((i + j + 1) as f64)).collect())
        .collect();
    let k = condition_number_inf(&a);
    assert!(k > 1e4, "κ∞ = {k} should be large");
}

#[test]
fn condition_number_singular_is_infinite() {
    let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
    assert!(condition_number_inf(&a).is_infinite());
}

#[test]
fn condition_number_identity_is_one() {
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    assert!((condition_number_inf(&a) - 1.0).abs() < 1e-12);
}

#[test]
fn condition_number_matches_direct_inverse() {
    // 与直接求逆交叉验证：2×2 解析逆
    let a = vec![vec![3.0, 1.0], vec![1.0, 2.0]];
    let det = 3.0 * 2.0 - 1.0;
    let inv = vec![vec![2.0 / det, -1.0 / det], vec![-1.0 / det, 3.0 / det]];
    let norm = |m: &[Vec<f64>]| {
        m.iter()
            .map(|r| r.iter().fold(0.0, |s, &v| s + v.abs()))
            .fold(0.0, f64::max)
    };
    let want = norm(&a) * norm(&inv);
    let got = condition_number_inf(&a);
    assert!((got - want).abs() < 1e-9 * want, "{got} vs {want}");
}
