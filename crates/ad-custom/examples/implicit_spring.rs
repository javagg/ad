//! IFT 隐式求解示例（设计文档 §4.3.3）：隐式欧拉弹簧-阻尼系统。
//! 运行：cargo run -p ad-custom --example implicit_spring --release

use ad_core::{Context, AD};
use ad_custom::{ImplicitSolve, Residual};
use num_traits::Num;

/// 隐式欧拉弹簧-阻尼（m=1）残差：
/// theta = [q0, v0, h, k, c]，x = [q1, v1]
/// r = [q1 - q0 - h·v1, v1 - v0 + h·k·q1 + h·c·v1]
struct SpringDamper;

impl Residual for SpringDamper {
    fn nx(&self) -> usize {
        2
    }
    fn ntheta(&self) -> usize {
        5
    }
    fn residual<N: Num + Copy>(&self, x: &[N], theta: &[N], r: &mut [N]) {
        let (q0, v0, h, k, c) = (theta[0], theta[1], theta[2], theta[3], theta[4]);
        let (q1, v1) = (x[0], x[1]);
        r[0] = q1 - q0 - h * v1;
        r[1] = v1 - v0 + h * k * q1 + h * c * v1;
    }
}

fn main() {
    let (q0, v0, h, k, c) = (1.0, 0.5, 0.1, 3.0, 0.4);

    let mut ctx = Context::<f64>::new();
    let mut vars = Vec::new();
    let mut theta = Vec::new();
    for (name, v) in [("q0", q0), ("v0", v0), ("h", h), ("k", k), ("c", c)] {
        let (ad, var) = ctx.var(v);
        theta.push(ad);
        vars.push((name, var));
    }

    // 整个求解器 = 一个自定义算子：内存 O(1)，无迭代截断偏差
    let out = ctx.call_custom(ImplicitSolve::new(SpringDamper), &theta);

    let q1s = ctx.mul(out[0], out[0]);
    let two_v1 = ctx.mul(AD::constant(2.0), out[1]);
    let loss = ctx.add(q1s, two_v1);
    ctx.backward(loss);

    println!("q1 = {:.6}  v1 = {:.6}", out[0].value, out[1].value);
    println!("loss = {:.6}", loss.value);
    for (name, var) in &vars {
        println!("dL/d{} = {:.6}", name, ctx.grad(*var).unwrap());
    }
}
