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
//! 约束：残差系统为**方阵**（nr = nx）；`∂r/∂x` 病态（高刚度接触）时伴随解
//! 不可靠——用 `ad-verify` 的条件数探针检查。

use ad_core::dual::Dual;
use ad_core::CustomOp;
use num_traits::Num;
use smallvec::{smallvec, SmallVec};
use std::rc::Rc;

use crate::linear_solve::{solve_linear, solve_linear_transposed};

/// 隐式残差系统 `r(x; θ) = 0`。
///
/// `N` 为数值抽象：前向求解用 `f64`，Jacobian 构造用 [`Dual`]。
/// 要求方阵：`r.len() == nx()`。
pub trait Residual: 'static {
    /// 未知量个数 x（输出个数）
    fn nx(&self) -> usize;
    /// 参数个数 θ（输入个数）
    fn ntheta(&self) -> usize;
    /// 计算残差 r(x; θ)，写入 `r`（长度 nx()）
    fn residual<N: Num + Copy>(&self, x: &[N], theta: &[N], r: &mut [N]);
}

/// Newton 求解配置。迭代次数固定以保证重算确定性（设计文档 §4.4.5）。
#[derive(Clone, Copy, Debug)]
pub struct ImplicitSolveCfg {
    /// 固定 Newton 迭代次数（线性系统 2-3 次即收敛，多余迭代为确定性空转）
    pub max_iters: usize,
}

impl Default for ImplicitSolveCfg {
    fn default() -> Self {
        ImplicitSolveCfg { max_iters: 8 }
    }
}

/// 隐式求解算子：inputs = θ，outputs = x*。
pub struct ImplicitSolve<R: Residual> {
    resid: Rc<R>,
    cfg: ImplicitSolveCfg,
}

impl<R: Residual> ImplicitSolve<R> {
    pub fn new(resid: R) -> Self {
        ImplicitSolve {
            resid: Rc::new(resid),
            cfg: ImplicitSolveCfg::default(),
        }
    }

    pub fn with_cfg(resid: R, cfg: ImplicitSolveCfg) -> Self {
        ImplicitSolve {
            resid: Rc::new(resid),
            cfg,
        }
    }

    pub fn shared(resid: Rc<R>, cfg: ImplicitSolveCfg) -> Self {
        ImplicitSolve { resid, cfg }
    }

    // 逐列构造：外层 j 是 Jacobian 列号（双数种子位置），下标循环即最清晰写法
    #[allow(clippy::needless_range_loop)]
    fn jacobian_x(&self, x: &[f64], theta: &[f64]) -> Vec<Vec<f64>> {
        // 列式：col[j][i] = ∂r_i/∂x_j
        let nx = self.resid.nx();
        let mut cols = vec![vec![0.0; nx]; nx];
        for j in 0..nx {
            let xd: Vec<Dual> = x
                .iter()
                .enumerate()
                .map(|(i, &v)| Dual::new(v, if i == j { 1.0 } else { 0.0 }))
                .collect();
            let td: Vec<Dual> = theta.iter().map(|&v| Dual::constant(v)).collect();
            let mut r = vec![Dual::constant(0.0); nx];
            self.resid.residual(&xd, &td, &mut r);
            for i in 0..nx {
                cols[j][i] = r[i].du;
            }
        }
        cols
    }

    #[allow(clippy::needless_range_loop)]
    fn jacobian_theta(&self, x: &[f64], theta: &[f64]) -> Vec<Vec<f64>> {
        // 列式：col[j][i] = ∂r_i/∂θ_j
        let (nx, nt) = (self.resid.nx(), self.resid.ntheta());
        let mut cols = vec![vec![0.0; nx]; nt];
        for j in 0..nt {
            let xd: Vec<Dual> = x.iter().map(|&v| Dual::constant(v)).collect();
            let td: Vec<Dual> = theta
                .iter()
                .enumerate()
                .map(|(i, &v)| Dual::new(v, if i == j { 1.0 } else { 0.0 }))
                .collect();
            let mut r = vec![Dual::constant(0.0); nx];
            self.resid.residual(&xd, &td, &mut r);
            for i in 0..nx {
                cols[j][i] = r[i].du;
            }
        }
        cols
    }
}

impl<R: Residual> CustomOp<f64> for ImplicitSolve<R> {
    fn num_inputs(&self) -> usize {
        self.resid.ntheta()
    }

    fn num_outputs(&self) -> usize {
        self.resid.nx()
    }

    fn forward(&self, theta: &[f64]) -> (SmallVec<[f64; 4]>, SmallVec<[f64; 8]>) {
        let nx = self.resid.nx();
        let mut x = vec![0.0; nx];
        for _ in 0..self.cfg.max_iters {
            let mut r = vec![0.0; nx];
            self.resid.residual::<f64>(&x, theta, &mut r);
            let cols = self.jacobian_x(&x, theta);
            // 行主序 J：J[i][j] = cols[j][i]
            let j: Vec<Vec<f64>> = (0..nx)
                .map(|i| (0..nx).map(|j| cols[j][i]).collect())
                .collect();
            let neg_r: Vec<f64> = r.iter().map(|v| -v).collect();
            let dx = solve_linear(&j, &neg_r);
            for i in 0..nx {
                x[i] += dx[i];
            }
        }
        let mut residual = SmallVec::new();
        residual.extend_from_slice(theta);
        residual.extend_from_slice(&x);
        (x.iter().copied().collect(), residual)
    }

    fn backward(&self, residual: &[f64], grad_output: &[f64]) -> SmallVec<[f64; 4]> {
        let nt = self.resid.ntheta();
        let nx = self.resid.nx();
        let theta = &residual[..nt];
        let x = &residual[nt..nt + nx];

        // 解 J_xᵀ λ = ḡ
        let cols = self.jacobian_x(x, theta);
        let j: Vec<Vec<f64>> = (0..nx)
            .map(|i| (0..nx).map(|j| cols[j][i]).collect())
            .collect();
        let lambda = solve_linear_transposed(&j, grad_output);

        // grad_θ_j = -(∂r/∂θ_j)ᵀ λ
        let jt_cols = self.jacobian_theta(x, theta);
        let mut grads = smallvec![];
        for col in &jt_cols {
            let g: f64 = col.iter().zip(&lambda).map(|(a, b)| a * b).sum();
            grads.push(-g);
        }
        grads
    }

    fn name(&self) -> &'static str {
        "implicit_solve"
    }
}
