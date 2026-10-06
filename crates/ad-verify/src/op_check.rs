//! `CustomOp` 契约验证器（设计文档 §4.3.1 + §12.3 第 33 条）。
//!
//! 把各测试代码里的"单步逐坐标 FD 隔离器"提升为**公开 API**——物理引擎
//! 作者手写 VJP 后一行调用即可验证，无需复制测试模式。检查三层：
//!
//! 1. **前向契约**：输出数 == `num_outputs`；forward 两次调用逐位一致
//!    （checkpoint 确定性重算的前提，§4.4.5）；
//! 2. **VJP 契约**：`backward` 返回长度 == `num_inputs`（此前是
//!    `debug_assert`，release 下消失）；
//! 3. **逐坐标 FD 对拍**：混合全部输出的标量损失（线性 + 二次 + 首尾交叉，
//!    覆盖每个输出通道），AD 反向 vs `loss(forward(x±h))` 中心差分。
//!
//! 并在四种**追踪形态**下重复对拍——常量输入占非尾部槽位的形态曾暴露
//! 梯度路由潜伏 bug（§12.3 第 28b 条），故隔位/前常量形态是标准检查项。

use crate::Rng;
use ad_core::{Context, CustomOp, AD};
use std::rc::Rc;

/// 混合全部输出的标量损失（固定伪随机权重，确定性）。
/// 与各 FD 隔离器测试（ops_fd.rs / chain_fd.rs / contact_fd.rs）同约定。
pub fn mixed_output_loss(out: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, &o) in out.iter().enumerate() {
        s += (0.3 + 0.11 * i as f64) * o + 0.2 * o * o;
    }
    if out.len() >= 2 {
        s += 0.15 * out[0] * out[out.len() - 1];
    }
    s
}

/// 单条验证失败记录。
#[derive(Clone, Debug)]
pub struct OpFailure {
    /// 追踪形态名（"all_tracked" / "const_head" / "const_tail" / "every_other"）
    pub shape: String,
    /// 输入点序号（传入 `points` 的下标）
    pub point: usize,
    /// 失败的输入坐标（原始槽位）
    pub coord: usize,
    pub ad_grad: f64,
    pub fd_grad: f64,
    /// 契约类失败的描述（FD 类失败为空）
    pub detail: String,
}

/// 验证报告。`passed == false` 时 `failures` 给出全部证据。
#[derive(Clone, Debug)]
pub struct OpValidationReport {
    pub op_name: String,
    pub passed: bool,
    /// 实际检查过的追踪形态名
    pub shapes: Vec<String>,
    pub points: usize,
    /// 对拍过的（形态 × 点 × 坐标）总数
    pub coords_checked: usize,
    pub max_rel_error: f64,
    pub failures: Vec<OpFailure>,
}

impl std::fmt::Display for OpValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "op '{}' validation: {} ({} shapes, {} points, {} coords, max_rel_err {:.2e})",
            self.op_name,
            if self.passed { "PASS" } else { "FAIL" },
            self.shapes.len(),
            self.points,
            self.coords_checked,
            self.max_rel_error
        )?;
        for fail in &self.failures {
            if fail.detail.is_empty() {
                writeln!(
                    f,
                    "  [{}] point {} coord {}: ad {:.10} vs fd {:.10}",
                    fail.shape, fail.point, fail.coord, fail.ad_grad, fail.fd_grad
                )?;
            } else {
                writeln!(f, "  [{}] point {}: {}", fail.shape, fail.point, fail.detail)?;
            }
        }
        Ok(())
    }
}

/// 四种追踪形态：`mask[i] == true` 表示槽位 i 被追踪（叶子），
/// false 表示以常量进入算子。
fn tracking_masks(n: usize) -> Vec<(String, Vec<bool>)> {
    let mut out = Vec::new();
    if n == 0 {
        return out;
    }
    out.push(("all_tracked".into(), vec![true; n]));
    if n >= 2 {
        let mut head = vec![false; n];
        for m in head.iter_mut().skip(n / 2) {
            *m = true;
        }
        out.push(("const_head".into(), head));
        let mut tail = vec![true; n];
        for m in tail.iter_mut().skip(n / 2) {
            *m = false;
        }
        out.push(("const_tail".into(), tail));
        let alt: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        out.push(("every_other".into(), alt));
    }
    out
}

/// 在 `points`（每个长度 == `num_inputs`）上验证算子。`points` 为空时用
/// 确定性 Rng 生成 3 个 [0.5, 1.5] 的点（注意：算子定义域未知时自动点
/// 可能落在域外——建议始终传入定义域内的点）。`tol` 为 FD 对拍的相对容差
/// （f64 推荐 1e-5–1e-6；非线性强的算子可放宽）。
pub fn validate_custom_op(
    op: Rc<dyn CustomOp<f64>>,
    points: &[Vec<f64>],
    tol: f64,
) -> OpValidationReport {
    let n = op.num_inputs();
    let n_out = op.num_outputs();
    let name = op.name().to_string();

    // 空输入时生成确定性默认点（值域 [0.5, 1.5]，规避常见对数/除法定义域）
    let owned: Vec<Vec<f64>>;
    let points: &[Vec<f64>] = if points.is_empty() {
        let mut rng = Rng::new(0x9E3779B97F4A7C15);
        owned = (0..3)
            .map(|_| (0..n).map(|_| 0.5 + rng.next_f64()).collect())
            .collect();
        &owned
    } else {
        points
    };

    let mut report = OpValidationReport {
        op_name: name,
        passed: true,
        shapes: Vec::new(),
        points: points.len(),
        coords_checked: 0,
        max_rel_error: 0.0,
        failures: Vec::new(),
    };

    // ---- 前向契约：输出数 + 确定性（逐位） + VJP gins 长度（结构性，查一次） ----
    for (pi, p) in points.iter().enumerate() {
        debug_assert_eq!(p.len(), n, "point {pi} length != num_inputs");
        let (outs1, _) = op.forward(p);
        if outs1.len() != n_out {
            report.passed = false;
            report.failures.push(OpFailure {
                shape: "contract".into(),
                point: pi,
                coord: usize::MAX,
                ad_grad: f64::NAN,
                fd_grad: f64::NAN,
                detail: format!(
                    "forward returned {} outputs, num_outputs = {n_out}",
                    outs1.len()
                ),
            });
        }
        let (outs2, _) = op.forward(p);
        if outs1.len() == outs2.len()
            && outs1
                .iter()
                .zip(outs2.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            report.passed = false;
            report.failures.push(OpFailure {
                shape: "contract".into(),
                point: pi,
                coord: usize::MAX,
                ad_grad: f64::NAN,
                fd_grad: f64::NAN,
                detail: "forward is not deterministic (two calls differ bitwise)".into(),
            });
        }
    }

    // gins 长度是结构性契约：在 AD 对拍**之前**检查（backward_seeds 的
    // debug_assert 会在 debug 构建下抢先 panic——验证器必须先给出干净报告）
    let mut gins_len_ok = true;
    if !points.is_empty() {
        let (p0_outs, p0_res) = op.forward(&points[0]);
        if p0_outs.len() == n_out {
            let gout: Vec<f64> = (0..n_out).map(|k| if k == 0 { 1.0 } else { 0.3 }).collect();
            let gins = op.backward(&p0_res, &gout);
            if gins.len() != n {
                gins_len_ok = false;
                report.passed = false;
                report.failures.push(OpFailure {
                    shape: "contract".into(),
                    point: 0,
                    coord: usize::MAX,
                    ad_grad: f64::NAN,
                    fd_grad: f64::NAN,
                    detail: format!(
                        "backward returned {} grads, num_inputs = {n}",
                        gins.len()
                    ),
                });
            }
        }
    }

    // ---- VJP FD 对拍（四种追踪形态；gins 长度不合法时跳过——已报告） ----
    let h = 1e-6;
    for (shape_name, mask) in tracking_masks(n) {
        report.shapes.push(shape_name.clone());
        if !gins_len_ok {
            continue;
        }
        for (pi, p) in points.iter().enumerate() {
            // AD 路径：按形态构造追踪/常量输入
            let mut ctx = Context::<f64>::new();
            let mut ads: Vec<AD<f64>> = Vec::with_capacity(n);
            let mut vars: Vec<(usize, ad_core::Variable)> = Vec::new();
            for (i, &v) in p.iter().enumerate() {
                if mask[i] {
                    let (ad, var) = ctx.var(v);
                    ads.push(ad);
                    vars.push((i, var));
                } else {
                    ads.push(AD::constant(v));
                }
            }
            let outs = ctx.call_custom_dyn(Rc::clone(&op), "op_under_test", &ads);
            let m = outs.len();
            let mut loss: Option<AD<f64>> = None;
            if m > 0 {
                let lin = ctx.mul(AD::constant(0.3 + 0.0), outs[0]);
                let sq = ctx.mul(outs[0], outs[0]);
                let quad = ctx.mul(AD::constant(0.2), sq);
                loss = Some(ctx.add(lin, quad));
                for i in 1..m {
                    let lin = ctx.mul(AD::constant(0.3 + 0.11 * i as f64), outs[i]);
                    let sq = ctx.mul(outs[i], outs[i]);
                    let quad = ctx.mul(AD::constant(0.2), sq);
                    let term = ctx.add(lin, quad);
                    loss = Some(ctx.add(loss.unwrap(), term));
                }
                if m >= 2 {
                    let cross_in = ctx.mul(outs[0], outs[m - 1]);
                    let cross = ctx.mul(AD::constant(0.15), cross_in);
                    loss = Some(ctx.add(loss.unwrap(), cross));
                }
            }
            let mut ad_grads: Vec<f64> = Vec::new();
            if let Some(l) = loss {
                ctx.backward(l);
                for (_, var) in &vars {
                    ad_grads.push(ctx.grad(*var).unwrap_or(f64::NAN));
                }
            }

            // FD 对拍（仅追踪坐标——常量坐标无梯度可验）
            for (k, &(coord, _)) in vars.iter().enumerate() {
                let mut tp = p.clone();
                tp[coord] += h;
                let mut tm = p.clone();
                tm[coord] -= h;
                let fp = mixed_output_loss(&op.forward(&tp).0);
                let fm = mixed_output_loss(&op.forward(&tm).0);
                let fd = (fp - fm) / (2.0 * h);
                let ad = ad_grads[k];
                let rel = (ad - fd).abs() / (1.0 + ad.abs() + fd.abs());
                report.coords_checked += 1;
                if rel > report.max_rel_error {
                    report.max_rel_error = rel;
                }
                if rel > tol {
                    report.passed = false;
                    report.failures.push(OpFailure {
                        shape: shape_name.clone(),
                        point: pi,
                        coord,
                        ad_grad: ad,
                        fd_grad: fd,
                        detail: String::new(),
                    });
                }
            }
        }
    }
    report
}
