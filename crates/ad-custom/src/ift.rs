//! IFT（隐函数定理）隐式求解模式（设计文档 §4.3.3）。
//!
//! 把整个 `solve(θ) → x*`（满足 `r(x*; θ) = 0`）封装为**一个**自定义算子：
//! - forward：Newton 迭代求解（普通 f64，不入带；迭代次数固定 → 确定性）；
//! - backward：解一次线性伴随系统 `J_xᵀ λ = ḡ`，返回 `grad_θ = -J_θᵀ λ`。
//!
//! 收益：内存与迭代次数无关（O(1)）、无迭代截断偏差、反向成本 = 一次线性求解。
//!
//! `J_x`、`J_θ` 默认用前向双数逐列构造（开发期最不易错的路径）；
//! 物理引擎如有解析 Jacobian，应自己实现 `CustomOp` 覆盖此路径。
//!
//! 系统形态（§12.3 第 10/29 条）：
//! - **方阵**（nr = nx，默认）：Newton 用 `solve_linear`，伴随 `λ = J_x⁻ᵀ ḡ`；
//! - **超定**（nr > nx）：要求 J_x 满列秩且系统相容（解流形存在），forward 走
//!   Gauss–Newton 正规方程 `(J_xᵀJ_x + μI)Δx = -J_xᵀ r`，伴随 `λ = J_x(J_xᵀJ_x)⁻¹ḡ`
//!   （最小范数伴随；方阵时与 `J_x⁻ᵀ` 数学等价）；μ 为 Tikhonov 阻尼（默认 0，
//!   病态时配合 `ad_verify::condition_number_inf` 选取）；
//! - **欠定**（nr < nx）：解不唯一，IFT 不适用——构造时 panic。
//!
//! Warm-start：[`ImplicitSolve::with_warm_start`] 把初值 x0 作为**显式输入槽位**
//! （inputs = [θ..., x0...]）——残差不依赖 x0（收敛解与初值无关），梯度恰为 0，
//! 但迭代路径由 x0 决定，因此 checkpoint 重算必须保存并重放 x0（§4.4.5）。
//!
//! `∂r/∂x` 病态（高刚度接触）时伴随解不可靠——用
//! `ad_verify::condition_number_inf` 检查（κ∞ ≳ 1e12 伴随解不可信）。

use ad_core::dual::Dual;
use ad_core::CustomOp;
use num_traits::Num;
use smallvec::{smallvec, SmallVec};
use std::rc::Rc;

use crate::linear_solve::{solve_linear, solve_linear_transposed};

/// 隐式残差系统 `r(x; θ) = 0`。
///
/// `N` 为数值抽象：前向求解用 `f64`，Jacobian 构造用 [`Dual`]。
/// 残差长度 nr 由 [`Residual::nr`] 给出（默认 = [`Residual::nx`]，方阵）；
/// nr > nx 为超定系统（要求满列秩、相容），nr < nx 不支持。
pub trait Residual: 'static {
    /// 未知量个数 x（输出个数）
    fn nx(&self) -> usize;
    /// 参数个数 θ（输入个数）
    fn ntheta(&self) -> usize;
    /// 残差条数（默认 = nx，方阵系统）
    fn nr(&self) -> usize {
        self.nx()
    }
    /// 计算残差 r(x; θ)，写入 `r`（长度 [`Residual::nr`]）
    fn residual<N: Num + Copy>(&self, x: &[N], theta: &[N], r: &mut [N]);
}

/// Newton 求解配置。迭代次数固定以保证重算确定性（设计文档 §4.4.5）。
#[derive(Clone, Copy, Debug)]
pub struct ImplicitSolveCfg {
    /// 固定 Newton 迭代次数（线性系统 2-3 次即收敛，多余迭代为确定性空转）
    pub max_iters: usize,
    /// 超定正规方程的 Tikhonov 阻尼 μ（对角加 (1+μ)·tr/JᵀJ 尺度的 μ·I；
    /// 默认 0 = 无阻尼，梯度精确；病态超定系统建议 1e-8–1e-4）
    pub damping: f64,
}

impl Default for ImplicitSolveCfg {
    fn default() -> Self {
        ImplicitSolveCfg {
            max_iters: 8,
            damping: 0.0,
        }
    }
}

/// 隐式求解算子：inputs = θ（warm-start 时 = [θ..., x0...]），outputs = x*。
pub struct ImplicitSolve<R: Residual> {
    resid: Rc<R>,
    cfg: ImplicitSolveCfg,
    warm_start: bool,
}

impl<R: Residual> ImplicitSolve<R> {
    /// 方阵/超定系统，零初值 Newton。
    pub fn new(resid: R) -> Self {
        ImplicitSolve {
            resid: Rc::new(resid),
            cfg: ImplicitSolveCfg::default(),
            warm_start: false,
        }
    }

    pub fn with_cfg(resid: R, cfg: ImplicitSolveCfg) -> Self {
        ImplicitSolve {
            resid: Rc::new(resid),
            cfg,
            warm_start: false,
        }
    }

    /// warm-start 形态：inputs = [θ..., x0...]，x0 作为 Newton 初值。
    /// 收敛解与 x0 无关（∂r/∂x0 = 0），x0 槽位梯度为 0；但迭代路径依赖 x0，
    /// checkpoint 重算必须保存并重放（§4.4.5）。
    pub fn with_warm_start(resid: R, cfg: ImplicitSolveCfg) -> Self {
        ImplicitSolve {
            resid: Rc::new(resid),
            cfg,
            warm_start: true,
        }
    }

    pub fn shared(resid: Rc<R>, cfg: ImplicitSolveCfg) -> Self {
        ImplicitSolve {
            resid,
            cfg,
            warm_start: false,
        }
    }

    // 逐列构造：外层 j 是 Jacobian 列号（双数种子位置），下标循环即最清晰写法
    #[allow(clippy::needless_range_loop)]
    pub fn jacobian_x(&self, x: &[f64], theta: &[f64]) -> Vec<Vec<f64>> {
        // 列式：col[j][i] = ∂r_i/∂x_j（i < nr；公开供条件数探针检查）
        let (nx, nr) = (self.resid.nx(), self.resid.nr());
        let mut cols = vec![vec![0.0; nr]; nx];
        for j in 0..nx {
            let xd: Vec<Dual> = x
                .iter()
                .enumerate()
                .map(|(i, &v)| Dual::new(v, if i == j { 1.0 } else { 0.0 }))
                .collect();
            let td: Vec<Dual> = theta.iter().map(|&v| Dual::constant(v)).collect();
            let mut r = vec![Dual::constant(0.0); nr];
            self.resid.residual(&xd, &td, &mut r);
            for i in 0..nr {
                cols[j][i] = r[i].du;
            }
        }
        cols
    }

    #[allow(clippy::needless_range_loop)]
    fn jacobian_theta(&self, x: &[f64], theta: &[f64]) -> Vec<Vec<f64>> {
        // 列式：col[j][i] = ∂r_i/∂θ_j
        let (nt, nr) = (self.resid.ntheta(), self.resid.nr());
        let mut cols = vec![vec![0.0; nr]; nt];
        for j in 0..nt {
            let xd: Vec<Dual> = x.iter().map(|&v| Dual::constant(v)).collect();
            let td: Vec<Dual> = theta
                .iter()
                .enumerate()
                .map(|(i, &v)| Dual::new(v, if i == j { 1.0 } else { 0.0 }))
                .collect();
            let mut r = vec![Dual::constant(0.0); nr];
            self.resid.residual(&xd, &td, &mut r);
            for i in 0..nr {
                cols[j][i] = r[i].du;
            }
        }
        cols
    }

    /// 正规方程矩阵 A = JᵀJ + μI（nx×nx 对称）。
    fn normal_matrix(&self, cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let nx = self.resid.nx();
        let mut a = vec![vec![0.0; nx]; nx];
        for j in 0..nx {
            for k in j..nx {
                let acc: f64 = (0..self.resid.nr()).map(|p| cols[j][p] * cols[k][p]).sum();
                a[j][k] = acc;
                a[k][j] = acc;
            }
            a[j][j] += self.cfg.damping;
        }
        a
    }

    /// 正规方程矩阵 A 与右端 b = -Jᵀr。
    fn normal_equations(&self, cols: &[Vec<f64>], r: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
        let nx = self.resid.nx();
        let a = self.normal_matrix(cols);
        let mut b = vec![0.0; nx];
        for j in 0..nx {
            b[j] = -(0..self.resid.nr()).map(|p| cols[j][p] * r[p]).sum::<f64>();
        }
        (a, b)
    }
}

impl<R: Residual> CustomOp<f64> for ImplicitSolve<R> {
    fn num_inputs(&self) -> usize {
        self.resid.ntheta() + if self.warm_start { self.resid.nx() } else { 0 }
    }

    fn num_outputs(&self) -> usize {
        self.resid.nx()
    }

    fn forward(&self, input: &[f64]) -> (SmallVec<[f64; 8]>, SmallVec<[f64; 8]>) {
        let (nx, nr) = (self.resid.nx(), self.resid.nr());
        assert!(
            nr >= nx,
            "ImplicitSolve: underdetermined residual system (nr = {nr} < nx = {nx}): \
             the solution is not locally unique and the IFT adjoint does not apply"
        );
        let nt = self.resid.ntheta();
        let nw = if self.warm_start { nx } else { 0 };
        debug_assert_eq!(input.len(), nt + nw, "input count mismatch");
        let theta = &input[..nt];
        let mut x: Vec<f64> = if nw > 0 {
            input[nt..nt + nx].to_vec()
        } else {
            vec![0.0; nx]
        };

        let mut r = vec![0.0; nr];
        for _ in 0..self.cfg.max_iters {
            self.resid.residual::<f64>(&x, theta, &mut r);
            let cols = self.jacobian_x(&x, theta);
            let dx: Vec<f64> = if nr == nx {
                // 方阵路径：J 行主序 J[i][j] = cols[j][i]，Δx = -J⁻¹r
                let j: Vec<Vec<f64>> = (0..nx)
                    .map(|i| (0..nx).map(|j| cols[j][i]).collect())
                    .collect();
                let neg_r: Vec<f64> = r.iter().map(|v| -v).collect();
                solve_linear(&j, &neg_r)
            } else {
                // 超定路径：Gauss–Newton 正规方程
                let (a, b) = self.normal_equations(&cols, &r);
                solve_linear(&a, &b)
            };
            for i in 0..nx {
                x[i] += dx[i];
            }
        }
        let mut residual = SmallVec::new();
        residual.extend_from_slice(theta);
        if nw > 0 {
            residual.extend_from_slice(&input[nt..nt + nx]);
        }
        residual.extend_from_slice(&x);
        (x.iter().copied().collect(), residual)
    }

    fn backward(&self, residual: &[f64], grad_output: &[f64]) -> SmallVec<[f64; 8]> {
        let (nt, nx, nr) = (self.resid.ntheta(), self.resid.nx(), self.resid.nr());
        let nw = if self.warm_start { nx } else { 0 };
        let theta = &residual[..nt];
        let x = &residual[nt + nw..nt + nw + nx];

        let cols = self.jacobian_x(x, theta);
        // 伴随 λ：方阵 λ = J⁻ᵀḡ；超定 λ = J(JᵀJ)⁻¹ḡ（最小范数，方阵时数学等价）
        let lambda: Vec<f64> = if nr == nx {
            let j: Vec<Vec<f64>> = (0..nx)
                .map(|i| (0..nx).map(|j| cols[j][i]).collect())
                .collect();
            solve_linear_transposed(&j, grad_output)
        } else {
            let a = self.normal_matrix(&cols);
            let z = solve_linear(&a, grad_output);
            (0..nr).map(|p| (0..nx).map(|j| cols[j][p] * z[j]).sum()).collect()
        };

        // grad_θ_j = -(∂r/∂θ_j)ᵀ λ；warm-start 槽位梯度为 0（∂r/∂x0 = 0）
        let jt_cols = self.jacobian_theta(x, theta);
        let mut grads = smallvec![];
        for col in &jt_cols {
            let g: f64 = col.iter().zip(&lambda).map(|(a, b)| a * b).sum();
            grads.push(-g);
        }
        for _ in 0..nw {
            grads.push(0.0);
        }
        grads
    }

    fn name(&self) -> &'static str {
        "implicit_solve"
    }
}
