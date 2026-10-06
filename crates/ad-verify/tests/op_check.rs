//! `validate_custom_op` 自验证：正确算子通过；三类故意破坏（槽位错路由、
//! gins 长度错误、forward 不确定）全部被抓住。槽位错路由复刻 §12.3 第 28b
//! 条的潜伏 bug（zip 按位置配对）——验证器必须能抓住这一类。

use ad_core::{Context, CustomOp};
use ad_verify::op_check::validate_custom_op;
use ad_verify::Rng;
use smallvec::{smallvec, SmallVec};
use std::rc::Rc;

/// 3 输入 2 输出的正确算子：out = [a·b + c, sin(a) − b²]，
/// 全部非线性、c 只进输出 0（部分依赖形态）。
#[derive(Clone, Copy)]
struct GoodOp;

impl CustomOp<f64> for GoodOp {
    fn num_inputs(&self) -> usize {
        3
    }
    fn num_outputs(&self) -> usize {
        2
    }
    fn forward(&self, i: &[f64]) -> (SmallVec<[f64; 8]>, SmallVec<[f64; 8]>) {
        (
            smallvec![i[0] * i[1] + i[2], i[0].sin() - i[1] * i[1]],
            i.iter().copied().collect(),
        )
    }
    fn backward(&self, r: &[f64], go: &[f64]) -> SmallVec<[f64; 8]> {
        // out0 = ab+c：∂/∂a = b，∂/∂b = a，∂/∂c = 1
        // out1 = sin a − b²：∂/∂a = cos a，∂/∂b = −2b，∂/∂c = 0
        smallvec![
            go[0] * r[1] + go[1] * r[0].cos(),
            go[0] * r[0] + go[1] * (-2.0 * r[1]),
            go[0],
        ]
    }
    fn name(&self) -> &'static str {
        "good_op"
    }
}

#[test]
fn correct_op_passes_all_shapes() {
    let report = validate_custom_op(
        Rc::new(GoodOp),
        &[vec![0.7, -1.3, 0.4], vec![-0.2, 0.9, 1.1]],
        1e-6, 1e-5,
    );
    assert!(report.passed, "{}", report);
    assert_eq!(report.shapes.len(), 4, "3 输入应有 4 种追踪形态");
    assert!(report.coords_checked >= 3 * 3, "每点至少 3 个被追踪坐标");
}

#[test]
fn correct_op_auto_points_pass() {
    // 空点集 → 确定性自动点
    let r1 = validate_custom_op(Rc::new(GoodOp), &[], 1e-6, 1e-5);
    let r2 = validate_custom_op(Rc::new(GoodOp), &[], 1e-6, 1e-5);
    assert!(r1.passed, "{}", r1);
    assert_eq!(r1.max_rel_error.to_bits(), r2.max_rel_error.to_bits());
}

/// 坏算子 #1：VJP 把梯度按"追踪子集顺序"返回（复刻 zip 按位置配对 bug）——
/// 当输入含常量时（部分追踪形态），gins 顺序与原始槽位错位。
#[derive(Clone, Copy)]
struct MisroutedOp;

impl CustomOp<f64> for MisroutedOp {
    fn num_inputs(&self) -> usize {
        3
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn forward(&self, i: &[f64]) -> (SmallVec<[f64; 8]>, SmallVec<[f64; 8]>) {
        (smallvec![i[0] * i[1] + i[2]], i.iter().copied().collect())
    }
    fn backward(&self, r: &[f64], go: &[f64]) -> SmallVec<[f64; 8]> {
        // 正确应为 [b·g, a·g, g]；这里输出 [a·g, g, b·g]——
        // 全追踪时首坐标也错（a·g vs b·g），部分追踪时错位更隐蔽
        smallvec![r[0] * go[0], go[0], r[1] * go[0]]
    }
    fn name(&self) -> &'static str {
        "misrouted_op"
    }
}

#[test]
fn misrouted_vjp_is_caught() {
    let report = validate_custom_op(Rc::new(MisroutedOp), &[vec![0.7, -1.3, 0.4]], 1e-6, 1e-5);
    assert!(!report.passed, "错路由 VJP 必须被抓：{}", report);
    // 至少一个形态报告了坐标级失败（非契约失败）
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.detail.is_empty() && f.coord != usize::MAX),
        "应有坐标级失败记录"
    );
}

/// 坏算子 #2：backward 返回长度错误（少了常量槽位）。
#[derive(Clone, Copy)]
struct ShortGradOp;

impl CustomOp<f64> for ShortGradOp {
    fn num_inputs(&self) -> usize {
        3
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn forward(&self, i: &[f64]) -> (SmallVec<[f64; 8]>, SmallVec<[f64; 8]>) {
        (smallvec![i[0] + i[1] + i[2]], i.iter().copied().collect())
    }
    fn backward(&self, _r: &[f64], go: &[f64]) -> SmallVec<[f64; 8]> {
        smallvec![go[0], go[0]] // 少一个槽位
    }
    fn name(&self) -> &'static str {
        "short_grad_op"
    }
}

#[test]
fn wrong_gins_length_is_caught() {
    let report = validate_custom_op(Rc::new(ShortGradOp), &[vec![1.0, 2.0, 3.0]], 1e-6, 1e-5);
    assert!(!report.passed, "{}", report);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.detail.contains("backward returned")),
        "应有 gins 长度契约失败"
    );
}

/// 坏算子 #3：forward 不确定（内部可变计数器污染输出）。
struct NondetOp {
    tick: std::cell::Cell<f64>,
}

impl CustomOp<f64> for NondetOp {
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn forward(&self, i: &[f64]) -> (SmallVec<[f64; 8]>, SmallVec<[f64; 8]>) {
        self.tick.set(self.tick.get() + 1.0);
        (smallvec![i[0] + self.tick.get() * 1e-15], smallvec![])
    }
    fn backward(&self, _r: &[f64], go: &[f64]) -> SmallVec<[f64; 8]> {
        smallvec![go[0]]
    }
    fn name(&self) -> &'static str {
        "nondet_op"
    }
}

#[test]
fn nondeterministic_forward_is_caught() {
    let report = validate_custom_op(Rc::new(NondetOp { tick: std::cell::Cell::new(0.0) }), &[], 1e-6, 1e-5);
    assert!(!report.passed, "{}", report);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.detail.contains("not deterministic")),
        "应有确定性契约失败"
    );
}

/// RNG 与报告展示：确定性自动点（用于文档示例）
#[test]
fn auto_points_are_reproducible_across_ops() {
    let mut rng = Rng::new(0x9E3779B97F4A7C15);
    let a = rng.next_f64();
    let mut rng = Rng::new(0x9E3779B97F4A7C15);
    let b = rng.next_f64();
    assert_eq!(a.to_bits(), b.to_bits());
    let _ = Context::<f64>::new(); // 保持 ad_core 引用（与其它测试一致）
}
