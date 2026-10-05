//! M1 验收：随机表达式 DAG 上反向模式 vs 双数 oracle（设计文档 §4.5.5）。
//!
//! 覆盖：嵌套深度、扇出（同一子表达式多次引用 → 验证梯度累加）、
//! 常量混合、算子全表。域保护（abs+0.1 作 ln/sqrt/div 分母）保证有限值。

use ad_core::dual::Dual;
use ad_core::{Context, AD};
use ad_ops::{abs_with, cos_with, sigmoid_with, sin_with, sqrt_with, tanh_with};
use proptest::prelude::*;

#[derive(Clone, Copy, Debug)]
struct RawInstr {
    op: u8, // 0..10
    a: u8,
    b: u8,
}

fn prog_strategy() -> impl Strategy<Value = Vec<RawInstr>> {
    prop::collection::vec(
        (0u8..10, any::<u8>(), any::<u8>()).prop_map(|(op, a, b)| RawInstr { op, a, b }),
        1..12,
    )
}

/// 在 AD（反向）上求值，返回 (终值, 各叶子梯度)
fn eval_ad(prog: &[RawInstr], leaves: &[f64]) -> (f64, Vec<f64>) {
    let mut ctx = Context::<f64>::new();
    let mut vars = Vec::with_capacity(leaves.len());
    let mut vals: Vec<AD<f64>> = Vec::with_capacity(leaves.len() + prog.len());
    for &v in leaves {
        let (ad, var) = ctx.var(v);
        vals.push(ad);
        vars.push(var);
    }
    for &ins in prog {
        let i = (ins.a as usize) % vals.len();
        let j = (ins.b as usize) % vals.len();
        let (a, b) = (vals[i], vals[j]);
        let v = match ins.op {
            0 => ctx.add(a, b),
            1 => ctx.sub(a, b),
            2 => ctx.mul(a, b),
            3 => {
                // 域保护：a / (|b| + 0.1)
                let d = abs_with(&mut ctx, b);
                let d = ctx.add(d, AD::constant(0.1));
                ctx.div(a, d)
            }
            4 => sin_with(&mut ctx, a),
            5 => cos_with(&mut ctx, a),
            6 => tanh_with(&mut ctx, a),
            7 => {
                let d = abs_with(&mut ctx, a);
                let d = ctx.add(d, AD::constant(0.1));
                sqrt_with(&mut ctx, d)
            }
            8 => {
                let d = abs_with(&mut ctx, a);
                let d = ctx.add(d, AD::constant(0.1));
                ad_ops::ln_with(&mut ctx, d)
            }
            _ => sigmoid_with(&mut ctx, a),
        };
        vals.push(v);
    }
    let out = *vals.last().unwrap();
    ctx.backward(out);
    let grads = vars.iter().map(|&v| ctx.grad(v).unwrap()).collect();
    (out.value, grads)
}

/// 在 Dual（前向）上求值；`seed_leaf` 指定哪个叶子带导数种子
fn eval_dual(prog: &[RawInstr], leaves: &[f64], seed_leaf: usize) -> Dual {
    let mut vals: Vec<Dual> = leaves
        .iter()
        .enumerate()
        .map(|(k, &v)| {
            if k == seed_leaf {
                Dual::seed(v)
            } else {
                Dual::constant(v)
            }
        })
        .collect();
    for &ins in prog {
        let i = (ins.a as usize) % vals.len();
        let j = (ins.b as usize) % vals.len();
        let (a, b) = (vals[i], vals[j]);
        let v = match ins.op {
            0 => a + b,
            1 => a - b,
            2 => a * b,
            3 => a / (b.abs() + Dual::constant(0.1)),
            4 => a.sin(),
            5 => a.cos(),
            6 => a.tanh(),
            7 => (a.abs() + Dual::constant(0.1)).sqrt(),
            8 => (a.abs() + Dual::constant(0.1)).ln(),
            _ => a.sigmoid(),
        };
        vals.push(v);
    }
    *vals.last().unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn reverse_matches_dual_oracle(prog in prog_strategy(), leaves in prop::collection::vec(0.5f64..2.0, 3)) {
        let (val_ad, grads) = eval_ad(&prog, &leaves);
        prop_assume!(val_ad.is_finite(), "guarded program should stay finite");

        for k in 0..leaves.len() {
            let d = eval_dual(&prog, &leaves, k);
            prop_assert!(d.re.is_finite());
            // 终值一致性（同一程序、同一数值语义）
            prop_assert!((val_ad - d.re).abs() <= 1e-10 * (1.0 + val_ad.abs()), "value mismatch leaf {}: {} vs {}", k, val_ad, d.re);
            // 梯度一致性
            let g = grads[k];
            prop_assert!(
                (g - d.du).abs() <= 1e-9 * (1.0 + g.abs() + d.du.abs()),
                "grad mismatch leaf {}: ad {} vs dual {} (prog {:?})",
                k, g, d.du, prog
            );
        }
    }
}
