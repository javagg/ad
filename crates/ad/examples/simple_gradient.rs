//! 最小示例：线程局部路径（运算符重载）+ 梯度验证。

use ad::prelude::*;

fn main() {
    // ---- 反向模式：y = sin(a·b + c) ----
    let guard = Context::<f64>::new().enter();
    let (a, va) = guard.var(1.0);
    let (b, vb) = guard.var(2.0);
    let (c, vc) = guard.var(3.0);

    let y = ad::sin(a * b + c);
    guard.backward(y);

    println!("y            = {}", y.value);
    println!(
        "dy/da        = {} (期望 {})",
        guard.grad(va).unwrap(),
        5.0f64.cos() * 2.0
    );
    println!(
        "dy/db        = {} (期望 {})",
        guard.grad(vb).unwrap(),
        5.0f64.cos()
    );
    println!(
        "dy/dc        = {} (期望 {})",
        guard.grad(vc).unwrap(),
        5.0f64.cos()
    );

    // ---- 有限差分交叉验证 ----
    let f = |x: &[f64]| (x[0] * x[1] + x[2]).sin();
    let x = [1.0, 2.0, 3.0];
    let g = [
        guard.grad(va).unwrap(),
        guard.grad(vb).unwrap(),
        guard.grad(vc).unwrap(),
    ];
    let check = GradientChecker::default().check_scalar(f, &x, &g);
    println!(
        "有限差分验证  = {} (max rel err {:.2e})",
        check.passed, check.max_rel_error
    );
}
