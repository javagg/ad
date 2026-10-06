//! IFT 泛化验证（设计文档 §4.3.3 / §12.3 第 29 条）：
//! 超定相容残差系统（nr > nx，Gauss–Newton 正规方程 + 最小范数伴随）、
//! warm-start 形态（x0 显式槽位，checkpoint 安全）、欠定系统拒绝。

use ad_core::{Context, AD};
use ad_custom::{ImplicitSolve, ImplicitSolveCfg, Residual};
use num_traits::Num;

// ============================================================ 超定：冗余约束对

/// r1 = x − θ，r2 = (x − θ)·θ：零点集相同（相容冗余），θ ≠ 0 时
/// J = [1, θ]ᵀ 满列秩。解 x* = θ，∂x*/∂θ = 1。
struct Redundant;

impl Residual for Redundant {
    fn nx(&self) -> usize {
        1
    }
    fn ntheta(&self) -> usize {
        1
    }
    fn nr(&self) -> usize {
        2
    }
    fn residual<N: Num + Copy>(&self, x: &[N], theta: &[N], r: &mut [N]) {
        let d = x[0] - theta[0];
        r[0] = d;
        r[1] = d * theta[0];
    }
}

#[test]
fn overdetermined_redundant_constraint_fd_crosscheck() {
    let th0 = 0.8f64;
    let mut ctx = Context::<f64>::new();
    let (th, vth) = ctx.var(th0);
    let outs = ctx.call_custom(ImplicitSolve::new(Redundant), &[th]);
    // x* = θ
    assert!((outs[0].value - th0).abs() < 1e-12, "x* = {}", outs[0].value);
    let l = ctx.mul(outs[0], outs[0]);
    ctx.backward(l);
    let g = ctx.grad(vth).unwrap();
    // d(x*²)/dθ = 2θ
    assert!((g - 2.0 * th0).abs() < 1e-10, "dL/dθ = {g}");
}

// ============================================================ 超定：2 未知量 + 3 约束

/// 极坐标到直角坐标（2 未知量 + 3 约束的相容超定系统）：
/// r1 = x − θ·c，r2 = y − θ·s，r3 = x·s − y·c（r1/r2 的线性组合 → 相容冗余），
/// 其中 c = cos φ、s = sin φ 由调用方作为 AD 叶子传入（c/s 进入 θ）——
/// 展示 IFT 算子与 tape 表达式的组合：φ 的梯度经 c/s 叶子链式回传。
/// 解 x* = θ c，y* = θ s。
struct PolarRedundant;

impl Residual for PolarRedundant {
    fn nx(&self) -> usize {
        2
    }
    fn ntheta(&self) -> usize {
        3
    }
    fn nr(&self) -> usize {
        3
    }
    fn residual<N: Num + Copy>(&self, x: &[N], theta: &[N], r: &mut [N]) {
        let (th, c, s) = (theta[0], theta[1], theta[2]);
        r[0] = x[0] - th * c;
        r[1] = x[1] - th * s;
        r[2] = x[0] * s - x[1] * c;
    }
}

#[test]
fn overdetermined_polar_system_fd_crosscheck() {
    // loss = x + 0.5·y² + 0.3·x·y（混合输出）
    let (th0, phi0) = (1.3f64, 0.7);
    let mut ctx = Context::<f64>::new();
    let (th, vth) = ctx.var(th0);
    let (phi, vphi) = ctx.var(phi0);
    // c/s 由 tape 的超越算子产生（IFT 输入可以是任意被追踪表达式）
    let c = ad_ops::cos_with(&mut ctx, phi);
    let s = ad_ops::sin_with(&mut ctx, phi);
    let outs = ctx.call_custom(ImplicitSolve::new(PolarRedundant), &[th, c, s]);
    let a = ctx.mul(AD::constant(1.0), outs[0]);
    let y2 = ctx.mul(outs[1], outs[1]);
    let b = ctx.mul(AD::constant(0.5), y2);
    let xy = ctx.mul(outs[0], outs[1]);
    let cross = ctx.mul(AD::constant(0.3), xy);
    let ab = ctx.add(a, b);
    let l = ctx.add(ab, cross);
    ctx.backward(l);
    let g_th = ctx.grad(vth).unwrap();
    let g_phi = ctx.grad(vphi).unwrap(); // 经 c/s 链式回传

    // FD oracle：显式解 x(θ,φ) = (θ cos φ, θ sin φ) 的 loss 梯度
    let loss = |th: f64, phi: f64| {
        let (x, y) = (th * phi.cos(), th * phi.sin());
        x + 0.5 * y * y + 0.3 * x * y
    };
    let h = 1e-7;
    let fd_th = (loss(th0 + h, phi0) - loss(th0 - h, phi0)) / (2.0 * h);
    let fd_phi = (loss(th0, phi0 + h) - loss(th0, phi0 - h)) / (2.0 * h);
    assert!((g_th - fd_th).abs() < 1e-8, "dL/dθ {g_th} vs {fd_th}");
    assert!((g_phi - fd_phi).abs() < 1e-8, "dL/dφ {g_phi} vs {fd_phi}");
}

// ============================================================ warm-start

/// 非线性方阵：r = x³ − θ（解 x* = θ^(1/3)，Newton 对初值敏感 → warm-start 有意义）
struct Cubic;

impl Residual for Cubic {
    fn nx(&self) -> usize {
        1
    }
    fn ntheta(&self) -> usize {
        1
    }
    fn residual<N: Num + Copy>(&self, x: &[N], theta: &[N], r: &mut [N]) {
        r[0] = x[0] * x[0] * x[0] - theta[0];
    }
}

#[test]
fn warm_start_matches_across_initial_guesses() {
    let cfg = ImplicitSolveCfg {
        max_iters: 16,
        ..Default::default()
    };
    let th0 = 3.5f64;

    // 两个不同初值 → 同一收敛解、同一梯度
    // （注：冷启动 x0 = 0 对立方系统恰好是 Newton 奇异点 J = 3x² = 0，
    //   迭代卡死——这正说明初值是收敛性行为的一部分，须纳入快照 §4.4.5）
    let run = |x_init: f64| -> (f64, f64, f64) {
        let mut ctx = Context::<f64>::new();
        let (th, vth) = ctx.var(th0);
        let (x0, vx0) = ctx.var(x_init);
        let outs = ctx.call_custom(ImplicitSolve::with_warm_start(Cubic, cfg), &[th, x0]);
        let xw = outs[0].value;
        let l = ctx.mul(outs[0], outs[0]);
        ctx.backward(l);
        (xw, ctx.grad(vth).unwrap(), ctx.grad(vx0).unwrap())
    };
    let (x1, g1, gx0_1) = run(1.0);
    let (x2, g2, gx0_2) = run(2.0);

    assert!((x1 - x2).abs() < 1e-12, "warm {x1} vs {x2}");
    assert!((g1 - g2).abs() < 1e-10, "grad {g1} vs {g2}");
    // ∂r/∂x0 = 0 → x0 槽位梯度恰为 0（收敛解与初值无关）
    assert_eq!(gx0_1, 0.0);
    assert_eq!(gx0_2, 0.0);
    // 解析：d(x*²)/dθ，x* = θ^(1/3)
    let want = 2.0 * th0.powf(1.0 / 3.0) * (1.0 / 3.0) * th0.powf(-2.0 / 3.0);
    assert!((g1 - want).abs() < 1e-9, "{g1} vs {want}");
}

#[test]
fn warm_start_recompute_is_bit_exact() {
    let cfg = ImplicitSolveCfg {
        max_iters: 16,
        ..Default::default()
    };
    let run = || {
        let mut ctx = Context::<f64>::new();
        let (th, _) = ctx.var(3.5);
        let (x0, _) = ctx.var(1.0);
        let outs = ctx.call_custom(ImplicitSolve::with_warm_start(Cubic, cfg), &[th, x0]);
        let l = ctx.mul(outs[0], outs[0]);
        ctx.backward(l);
        (outs[0].value, ctx.adjoint(outs[0]).unwrap())
    };
    let (v1, a1) = run();
    let (v2, a2) = run();
    assert_eq!(v1.to_bits(), v2.to_bits());
    assert_eq!(a1.to_bits(), a2.to_bits());
}

// ============================================================ 欠定拒绝

/// nr = 1 < nx = 2：解流形一维，IFT 不适用。
struct Underdetermined;

impl Residual for Underdetermined {
    fn nx(&self) -> usize {
        2
    }
    fn ntheta(&self) -> usize {
        1
    }
    fn nr(&self) -> usize {
        1
    }
    fn residual<N: Num + Copy>(&self, _x: &[N], _theta: &[N], _r: &mut [N]) {
        unreachable!()
    }
}

#[test]
#[should_panic(expected = "underdetermined")]
fn underdetermined_system_is_rejected() {
    let mut ctx = Context::<f64>::new();
    let (th, _) = ctx.var(1.0);
    let _ = ctx.call_custom(ImplicitSolve::new(Underdetermined), &[th]);
}
