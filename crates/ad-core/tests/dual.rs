//! 双数 oracle 自检：前向模式的正确性独立于反向实现（设计文档 §4.5.4）。

use ad_core::dual::Dual;

#[test]
fn dual_composite_matches_central_difference() {
    // f(x) = sin(x²) + exp(x)·ln(x+1)
    let f = |x: Dual| -> Dual { (x * x).sin() + x.exp() * (x + Dual::constant(1.0)).ln() };
    let x0 = 0.7f64;
    let d = f(Dual::seed(x0)).du;

    let h = 1e-6;
    let plain = |t: f64| -> f64 { (t * t).sin() + t.exp() * (t + 1.0).ln() };
    let fd = (plain(x0 + h) - plain(x0 - h)) / (2.0 * h);
    assert!((d - fd).abs() < 1e-8, "dual {} vs fd {}", d, fd);
}

#[test]
fn dual_arithmetic_rules() {
    let x = Dual::seed(2.0);
    let y = Dual::seed(3.0);
    let y3 = Dual::constant(3.0); // 不带种子
                                  // (xy)' = x'y + xy' = 3 + 2 = 5
    assert_eq!((x * y).du, 5.0);
    // (x/y)' 沿 x = 1/y = 1/3
    assert!(((x / y3).du - 1.0 / 3.0).abs() < 1e-15);
    // (-x)' = -1
    assert_eq!((-x).du, -1.0);
    // powf: x^3 at 2 沿 x = 3·2² = 12
    assert!((x.powf(y3)).du - 12.0 < 1e-12);
    // 双种子方向：d(x^y) = y·x^(y-1)·dx + x^y·ln(x)·dy = 12 + 8·ln2
    assert!(((x.powf(y)).du - (12.0 + 8.0 * 2.0f64.ln())).abs() < 1e-12);
    // 链式：(x + y)² 沿方向 (dx,dy)=(1,1)：2(x+y)·(1+1) = 20
    assert_eq!(((x + y) * (x + y)).du, 20.0);
}

#[test]
fn dual_transcendentals() {
    let x = Dual::seed(0.8);
    // tanh' = 1 - tanh²
    let t = x.tanh();
    assert!((t.du - (1.0 - t.re * t.re)).abs() < 1e-15);
    // atan2(y, x) at (1, 2) 沿 y：x/r² = 2/5
    let y = Dual::seed(1.0);
    let xc = Dual::constant(2.0);
    assert!((y.atan2(xc).du - 2.0 / 5.0).abs() < 1e-15);
    // asin at 0.5：1/sqrt(1-x²)
    let a = Dual::seed(0.5);
    assert!((a.asin().du - 1.0 / (1.0 - 0.25f64).sqrt()).abs() < 1e-12);
}
