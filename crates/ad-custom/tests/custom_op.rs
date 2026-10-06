//! M2 验收：CustomOp 机制——多输出、扇出累加、残差回传（设计文档 §4.3.1）。

use ad_core::{Context, CustomOp, AD};
use smallvec::{smallvec, SmallVec};

/// outputs = [x², x³]，residual = [x]。
/// backward: ∂/∂x = g₁·2x + g₂·3x²
struct PolyOp;

impl CustomOp<f64> for PolyOp {
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        2
    }
    fn forward(&self, inputs: &[f64]) -> (SmallVec<[f64; 8]>, SmallVec<[f64; 8]>) {
        let x = inputs[0];
        (smallvec![x * x, x * x * x], smallvec![x])
    }
    fn backward(&self, residual: &[f64], grad_output: &[f64]) -> SmallVec<[f64; 8]> {
        let x = residual[0];
        smallvec![grad_output[0] * 2.0 * x + grad_output[1] * 3.0 * x * x]
    }
    fn name(&self) -> &'static str {
        "poly"
    }
}

#[test]
fn multi_output_fanout_accumulates() {
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(2.0);
    let outs = ctx.call_custom(PolyOp, &[x]);
    assert_eq!(outs.len(), 2);
    assert_eq!(outs[0].value, 4.0);
    assert_eq!(outs[1].value, 8.0);
    assert_eq!(ctx.tape_len(), 1, "多输出只占 1 条记录");

    // loss 同时使用两个输出（扇出）→ backward 梯度应正确累加
    let loss = ctx.add(outs[0], outs[1]); // x² + x³
    ctx.backward(loss);
    // d/dx = 2x + 3x² = 4 + 12 = 16
    assert_eq!(ctx.grad(vx), Some(16.0));
}

#[test]
fn multi_output_multi_seed_vjp() {
    // 分别以 (1, 0) 和 (0, 1) 为 VJP 种子，验证 backward 收到的 grad_output 分量
    let mut ctx = Context::<f64>::new();
    let (x, vx) = ctx.var(2.0);
    let outs = ctx.call_custom(PolyOp, &[x]);
    ctx.backward_seeds(&[(outs[0], 1.0), (outs[1], 0.0)]);
    assert_eq!(ctx.grad(vx), Some(4.0)); // 2x
    ctx.zero_grads();
    ctx.backward_seeds(&[(outs[0], 0.0), (outs[1], 1.0)]);
    assert_eq!(ctx.grad(vx), Some(12.0)); // 3x²
}

#[test]
fn constant_inputs_skip_grad_slots() {
    // 常量输入占槽位但不占节点；backward 返回的对应槽位梯度被丢弃
    let mut ctx = Context::<f64>::new();
    let x = AD::constant(2.0);
    let outs = ctx.call_custom(PolyOp, &[x]);
    assert!(!outs[0].is_tracked());
    assert_eq!(ctx.tape_len(), 0);
}

#[test]
fn no_grad_degrades_to_constants() {
    let mut ctx = Context::<f64>::new();
    let (x, _vx) = ctx.var(2.0);
    let outs = ctx.no_grad(|ctx| ctx.call_custom(PolyOp, &[x]));
    assert!(!outs[0].is_tracked());
    assert_eq!(outs[0].value, 4.0);
    assert_eq!(ctx.tape_len(), 0);
}
