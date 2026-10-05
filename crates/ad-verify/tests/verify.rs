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
