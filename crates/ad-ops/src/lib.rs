//! `ad-ops`：基础算子（局部 Jacobian，设计文档 §4.2）+ bulk 向量算子（§4.3.4）。
//!
//! 每个算子提供两种形式：
//! - `fn(x)`：**线程局部路径**，要求当前线程已 `Context::enter`；
//! - `fn_with(ctx, x)`：**显式路径**，适合 checkpoint / 物理引擎集成代码。
//!
//! 数值边界策略（设计文档 §4.2.4）：不 panic、不静默饱和——前向遵守 IEEE 754
//! 语义，NaN/Inf 传播，由 `set_detect_anomaly` 在反向时定位首个出错算子。

use ad_core::{Context, CustomOp, Scalar, AD};
use smallvec::{smallvec, SmallVec};

// ---- 一元算子 ----

/// `exp(x)`
pub fn exp_with<S: Scalar>(ctx: &mut Context<S>, x: AD<S>) -> AD<S> {
    ctx.unary(x, "exp", |v| v.exp(), |v| v.exp())
}

/// `ln(x)`；x <= 0 时按 IEEE 传播 NaN
pub fn ln_with<S: Scalar>(ctx: &mut Context<S>, x: AD<S>) -> AD<S> {
    ctx.unary(x, "ln", |v| v.ln(), |v| S::one() / v)
}

/// `sqrt(x)`；x = 0 时反向 +inf（设计文档 §4.2.4）
pub fn sqrt_with<S: Scalar>(ctx: &mut Context<S>, x: AD<S>) -> AD<S> {
    ctx.unary(
        x,
        "sqrt",
        |v| v.sqrt(),
        |v| S::one() / (v.sqrt() + v.sqrt()),
    )
}

/// `sin(x)`
pub fn sin_with<S: Scalar>(ctx: &mut Context<S>, x: AD<S>) -> AD<S> {
    ctx.unary(x, "sin", |v| v.sin(), |v| v.cos())
}

/// `cos(x)`
pub fn cos_with<S: Scalar>(ctx: &mut Context<S>, x: AD<S>) -> AD<S> {
    ctx.unary(x, "cos", |v| v.cos(), |v| -v.sin())
}

/// `tanh(x)`
pub fn tanh_with<S: Scalar>(ctx: &mut Context<S>, x: AD<S>) -> AD<S> {
    ctx.unary(
        x,
        "tanh",
        |v| v.tanh(),
        |v| {
            let t = v.tanh();
            S::one() - t * t
        },
    )
}

/// `asin(x)`；|x| >= 1 时反向 inf/NaN
pub fn asin_with<S: Scalar>(ctx: &mut Context<S>, x: AD<S>) -> AD<S> {
    ctx.unary(
        x,
        "asin",
        |v| v.asin(),
        |v| S::one() / (S::one() - v * v).sqrt(),
    )
}

/// `acos(x)`；|x| >= 1 时反向 inf/NaN
pub fn acos_with<S: Scalar>(ctx: &mut Context<S>, x: AD<S>) -> AD<S> {
    ctx.unary(
        x,
        "acos",
        |v| v.acos(),
        |v| -S::one() / (S::one() - v * v).sqrt(),
    )
}

/// `1/x`
pub fn recip_with<S: Scalar>(ctx: &mut Context<S>, x: AD<S>) -> AD<S> {
    ctx.unary(x, "recip", |v| S::one() / v, |v| -S::one() / (v * v))
}

/// `|x|`；x = 0 处导数定义为 0（PAP 约定，设计文档 §4.2.3）
pub fn abs_with<S: Scalar>(ctx: &mut Context<S>, x: AD<S>) -> AD<S> {
    ctx.unary(
        x,
        "abs",
        |v| v.abs(),
        |v| {
            if v > S::zero() {
                S::one()
            } else if v < S::zero() {
                -S::one()
            } else {
                S::zero()
            }
        },
    )
}

/// `max(0, x)`；x = 0 处导数定义为 0
pub fn relu_with<S: Scalar>(ctx: &mut Context<S>, x: AD<S>) -> AD<S> {
    ctx.unary(
        x,
        "relu",
        |v| {
            if v > S::zero() {
                v
            } else {
                S::zero()
            }
        },
        |v| {
            if v > S::zero() {
                S::one()
            } else {
                S::zero()
            }
        },
    )
}

fn sigmoid_fwd<S: Scalar>(v: S) -> S {
    S::one() / (S::one() + (-v).exp())
}

/// `1 / (1 + exp(-x))`
pub fn sigmoid_with<S: Scalar>(ctx: &mut Context<S>, x: AD<S>) -> AD<S> {
    ctx.unary(x, "sigmoid", sigmoid_fwd, |v| {
        let s = sigmoid_fwd(v);
        s * (S::one() - s)
    })
}

// ---- 二元 / n 元算子 ----

/// `x^p`；负底非整数指数按 IEEE 传播 NaN
pub fn powf_with<S: Scalar>(ctx: &mut Context<S>, x: AD<S>, p: AD<S>) -> AD<S> {
    ctx.binary(
        x,
        p,
        "powf",
        |a, b| a.powf(b),
        |a, b| b * a.powf(b - S::one()),
        |a, b| a.powf(b) * a.ln(),
    )
}

/// `x^n`（整数幂）
pub fn powi_with<S: Scalar>(ctx: &mut Context<S>, x: AD<S>, n: i32) -> AD<S> {
    let nf = <S as num_traits::NumCast>::from(n).expect("powi: exponent out of range");
    ctx.unary(
        x,
        "powi",
        |v| v.powi(n),
        move |v| {
            if n == 0 {
                S::zero()
            } else {
                nf * v.powi(n - 1)
            }
        },
    )
}

/// `atan2(y, x)`；关节角包裹必备，(0,0) 处反向无定义（设计文档 §4.2.1）
pub fn atan2_with<S: Scalar>(ctx: &mut Context<S>, y: AD<S>, x: AD<S>) -> AD<S> {
    ctx.binary(
        y,
        x,
        "atan2",
        |a, b| a.atan2(b),
        |a, b| b / (a * a + b * b),
        |a, b| -a / (a * a + b * b),
    )
}

/// `min(a, b)`；并列时梯度加在第一个参数（与主流框架一致）
pub fn min_with<S: Scalar>(ctx: &mut Context<S>, a: AD<S>, b: AD<S>) -> AD<S> {
    ctx.binary(
        a,
        b,
        "min",
        |a, b| {
            if a <= b {
                a
            } else {
                b
            }
        },
        |a, b| {
            if a <= b {
                S::one()
            } else {
                S::zero()
            }
        },
        |a, b| {
            if a <= b {
                S::zero()
            } else {
                S::one()
            }
        },
    )
}

/// `max(a, b)`；并列时梯度加在第一个参数
pub fn max_with<S: Scalar>(ctx: &mut Context<S>, a: AD<S>, b: AD<S>) -> AD<S> {
    ctx.binary(
        a,
        b,
        "max",
        |a, b| {
            if a >= b {
                a
            } else {
                b
            }
        },
        |a, b| {
            if a >= b {
                S::one()
            } else {
                S::zero()
            }
        },
        |a, b| {
            if a >= b {
                S::zero()
            } else {
                S::one()
            }
        },
    )
}

/// `clamp(a, lo, hi)`（标量边界）；越界侧梯度为 0。接触速度限幅常用。
pub fn clamp_with<S: Scalar>(ctx: &mut Context<S>, a: AD<S>, lo: S, hi: S) -> AD<S> {
    ctx.unary(
        a,
        "clamp",
        move |v| {
            if v < lo {
                lo
            } else if v > hi {
                hi
            } else {
                v
            }
        },
        move |v| {
            if v < lo || v > hi {
                S::zero()
            } else {
                S::one()
            }
        },
    )
}

/// `a + t(b - a)`，t 可微（∂/∂t = b - a）
pub fn lerp_with<S: Scalar>(ctx: &mut Context<S>, a: AD<S>, b: AD<S>, t: AD<S>) -> AD<S> {
    ctx.nary(
        &[a, b, t],
        "lerp",
        |v| v[0] + v[2] * (v[1] - v[0]),
        |v| smallvec![S::one() - v[2], v[2], v[1] - v[0]],
    )
}

// ---- 线程局部版本（`Context::enter` 后使用） ----

macro_rules! tls_unary {
    ($name:ident, $with:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $name<S: Scalar>(x: AD<S>) -> AD<S> {
            ad_core::with_context(|c: &mut Context<S>| $with(c, x))
        }
    };
}

tls_unary!(exp, exp_with, "`exp(x)`");
tls_unary!(ln, ln_with, "`ln(x)`");
tls_unary!(sqrt, sqrt_with, "`sqrt(x)`");
tls_unary!(sin, sin_with, "`sin(x)`");
tls_unary!(cos, cos_with, "`cos(x)`");
tls_unary!(tanh, tanh_with, "`tanh(x)`");
tls_unary!(asin, asin_with, "`asin(x)`");
tls_unary!(acos, acos_with, "`acos(x)`");
tls_unary!(recip, recip_with, "`1/x`");
tls_unary!(abs, abs_with, "`|x|`");
tls_unary!(relu, relu_with, "`max(0, x)`");
tls_unary!(sigmoid, sigmoid_with, "`1 / (1 + exp(-x))`");

macro_rules! tls_binary {
    ($name:ident, $with:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $name<S: Scalar>(a: AD<S>, b: AD<S>) -> AD<S> {
            ad_core::with_context(|c: &mut Context<S>| $with(c, a, b))
        }
    };
}

tls_binary!(powf, powf_with, "`x^p`");
tls_binary!(atan2, atan2_with, "`atan2(y, x)`");
tls_binary!(min, min_with, "`min(a, b)`");
tls_binary!(max, max_with, "`max(a, b)`");

/// `x^n`：线程局部版本
pub fn powi<S: Scalar>(x: AD<S>, n: i32) -> AD<S> {
    ad_core::with_context(|c: &mut Context<S>| powi_with(c, x, n))
}

/// `clamp(a, lo, hi)`：线程局部版本
pub fn clamp<S: Scalar>(a: AD<S>, lo: S, hi: S) -> AD<S> {
    ad_core::with_context(|c: &mut Context<S>| clamp_with(c, a, lo, hi))
}

/// `a + t(b - a)`：线程局部版本
pub fn lerp<S: Scalar>(a: AD<S>, b: AD<S>, t: AD<S>) -> AD<S> {
    ad_core::with_context(|c: &mut Context<S>| lerp_with(c, a, b, t))
}

// ---- bulk 向量算子（设计文档 §4.3.4：O(n²)+ 操作必须整体入带） ----

/// `dot(a, b) = Σ aᵢbᵢ`：1 条 tape 记录，而非 n 条逐元素记录。
pub fn dot_with<S: Scalar>(ctx: &mut Context<S>, a: &[AD<S>], b: &[AD<S>]) -> AD<S> {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let mut inputs = Vec::with_capacity(2 * a.len());
    inputs.extend_from_slice(a);
    inputs.extend_from_slice(b);
    ctx.call_custom(DotOp { n: a.len() }, &inputs).remove(0)
}

/// `out = α·x + y`（BLAS axpy 语义）：1 条记录。
pub fn axpy_with<S: Scalar>(
    ctx: &mut Context<S>,
    alpha: AD<S>,
    x: &[AD<S>],
    y: &[AD<S>],
) -> SmallVec<[AD<S>; 4]> {
    assert_eq!(x.len(), y.len(), "axpy: length mismatch");
    let mut inputs = Vec::with_capacity(1 + 2 * x.len());
    inputs.push(alpha);
    inputs.extend_from_slice(x);
    inputs.extend_from_slice(y);
    ctx.call_custom(AxpyOp { n: x.len() }, &inputs)
}

/// `‖x‖₂`；x = 0 时反向 inf（文档化，同 sqrt(0)，设计文档 §4.2.4）
pub fn norm2_with<S: Scalar>(ctx: &mut Context<S>, x: &[AD<S>]) -> AD<S> {
    ctx.call_custom(Norm2Op { n: x.len() }, x).remove(0)
}

/// `dot(a, b)`：线程局部版本
pub fn dot<S: Scalar>(a: &[AD<S>], b: &[AD<S>]) -> AD<S> {
    ad_core::with_context(|c: &mut Context<S>| dot_with(c, a, b))
}

/// `out = α·x + y`：线程局部版本
pub fn axpy<S: Scalar>(alpha: AD<S>, x: &[AD<S>], y: &[AD<S>]) -> SmallVec<[AD<S>; 4]> {
    ad_core::with_context(|c: &mut Context<S>| axpy_with(c, alpha, x, y))
}

/// `‖x‖₂`：线程局部版本
pub fn norm2<S: Scalar>(x: &[AD<S>]) -> AD<S> {
    ad_core::with_context(|c: &mut Context<S>| norm2_with(c, x))
}

// ---- bulk CustomOp 实现 ----

/// dot：inputs = [a..., b...]，residual = [a..., b...]
struct DotOp {
    n: usize,
}

impl<S: Scalar> CustomOp<S> for DotOp {
    fn num_inputs(&self) -> usize {
        2 * self.n
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn forward(&self, inputs: &[S]) -> (SmallVec<[S; 8]>, SmallVec<[S; 8]>) {
        let n = self.n;
        let mut sum = S::zero();
        for i in 0..n {
            sum = sum + inputs[i] * inputs[n + i];
        }
        (smallvec![sum], inputs.iter().copied().collect())
    }
    fn backward(&self, residual: &[S], grad_output: &[S]) -> SmallVec<[S; 8]> {
        let n = self.n;
        let g = grad_output[0];
        let mut grads = SmallVec::new();
        for i in 0..n {
            grads.push(g * residual[n + i]); // ∂/∂aᵢ = bᵢ
        }
        for i in 0..n {
            grads.push(g * residual[i]); // ∂/∂bᵢ = aᵢ
        }
        grads
    }
    fn name(&self) -> &'static str {
        "dot"
    }
}

/// axpy：inputs = [α, x..., y...]，residual = [α, x...]
struct AxpyOp {
    n: usize,
}

impl<S: Scalar> CustomOp<S> for AxpyOp {
    fn num_inputs(&self) -> usize {
        1 + 2 * self.n
    }
    fn num_outputs(&self) -> usize {
        self.n
    }
    fn forward(&self, inputs: &[S]) -> (SmallVec<[S; 8]>, SmallVec<[S; 8]>) {
        let n = self.n;
        let alpha = inputs[0];
        let mut outs = SmallVec::new();
        for i in 0..n {
            outs.push(alpha * inputs[1 + i] + inputs[1 + n + i]);
        }
        let mut residual = SmallVec::new();
        residual.push(alpha);
        residual.extend_from_slice(&inputs[1..1 + n]); // x
        (outs, residual)
    }
    fn backward(&self, residual: &[S], grad_output: &[S]) -> SmallVec<[S; 8]> {
        let n = self.n;
        let alpha = residual[0];
        let mut grads = SmallVec::new();
        let mut g_alpha = S::zero();
        for i in 0..n {
            g_alpha = g_alpha + grad_output[i] * residual[1 + i];
        }
        grads.push(g_alpha);
        for i in 0..n {
            grads.push(grad_output[i] * alpha); // ∂/∂xᵢ
        }
        for i in 0..n {
            grads.push(grad_output[i]); // ∂/∂yᵢ
        }
        grads
    }
    fn name(&self) -> &'static str {
        "axpy"
    }
}

/// norm2：inputs = [x...]，residual = [x...]
struct Norm2Op {
    n: usize,
}

impl<S: Scalar> CustomOp<S> for Norm2Op {
    fn num_inputs(&self) -> usize {
        self.n
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn forward(&self, inputs: &[S]) -> (SmallVec<[S; 8]>, SmallVec<[S; 8]>) {
        let mut sq = S::zero();
        for &v in inputs {
            sq = sq + v * v;
        }
        (smallvec![sq.sqrt()], inputs.iter().copied().collect())
    }
    fn backward(&self, residual: &[S], grad_output: &[S]) -> SmallVec<[S; 8]> {
        let norm = (residual.iter().fold(S::zero(), |a, &v| a + v * v)).sqrt();
        let g = grad_output[0];
        residual.iter().map(|&v| g * v / norm).collect()
    }
    fn name(&self) -> &'static str {
        "norm2"
    }
}
