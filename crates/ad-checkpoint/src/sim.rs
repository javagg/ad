//! 参考实现：单摆仿真（`Recomputable` + `CustomOp` 的完整接入示例）。
//!
//! 状态 = [θ, ω]；动力学（显式欧拉，逐时间步封装为**一个** CustomOp）：
//! θ' = θ + dt·ω；ω' = ω - dt·(g/L)·sin(θ)
//! 前向残差保存 [sin θ, cos θ, ω, dt, g, L]，反向手工推导（f_fwd 风格）。

use crate::manager::Recomputable;
use ad_core::{Context, CustomOp, AD};
use smallvec::{smallvec, SmallVec};

/// 单摆单步算子：inputs = [θ, ω, g, L, dt]，outputs = [θ', ω']。
pub struct PendulumStep;

impl CustomOp<f64> for PendulumStep {
    fn num_inputs(&self) -> usize {
        5
    }

    fn num_outputs(&self) -> usize {
        2
    }

    fn forward(&self, inputs: &[f64]) -> (SmallVec<[f64; 8]>, SmallVec<[f64; 8]>) {
        let (th, om, g, len, dt) = (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
        let th1 = th + dt * om;
        let om1 = om - dt * (g / len) * th.sin();
        let residual = smallvec![th.sin(), th.cos(), om, dt, g, len];
        (smallvec![th1, om1], residual)
    }

    fn backward(&self, residual: &[f64], grad_output: &[f64]) -> SmallVec<[f64; 8]> {
        // λ1 = ∂L/∂θ'，λ2 = ∂L/∂ω'
        let (lam1, lam2) = (grad_output[0], grad_output[1]);
        let (sin_th, cos_th, om, dt, g, len) = (
            residual[0],
            residual[1],
            residual[2],
            residual[3],
            residual[4],
            residual[5],
        );
        smallvec![
            lam1 + lam2 * (-dt * (g / len) * cos_th), // ∂/∂θ
            lam1 * dt + lam2,                         // ∂/∂ω
            lam2 * (-dt * sin_th / len),              // ∂/∂g
            lam2 * (dt * g * sin_th / (len * len)),   // ∂/∂L
            lam1 * om + lam2 * (-(g / len) * sin_th), // ∂/∂dt
        ]
    }

    fn name(&self) -> &'static str {
        "pendulum_step"
    }
}

/// 单摆仿真。可微参数 g、L 由构造时的叶子变量提供。
pub struct PendulumSim {
    theta: f64,
    omega: f64,
    state_ad: SmallVec<[AD<f64>; 4]>,
    g: AD<f64>,
    len: AD<f64>,
    dt: f64,
    /// 算子句柄缓存：step 每步走 call_custom_dyn，避免 Rc::new 堆分配
    op: std::rc::Rc<dyn CustomOp<f64>>,
    /// `step` 被调用的总次数（含 no_grad 重算）——监测嵌套反转的重算量
    pub steps_executed: usize,
}

impl PendulumSim {
    pub fn new(
        ctx: &mut Context<f64>,
        theta: f64,
        omega: f64,
        g: AD<f64>,
        len: AD<f64>,
        dt: f64,
    ) -> Self {
        let mut sim = PendulumSim {
            theta,
            omega,
            state_ad: SmallVec::new(),
            g,
            len,
            dt,
            op: std::rc::Rc::new(PendulumStep),
            steps_executed: 0,
        };
        sim.bind_state(ctx);
        sim
    }
}

impl Recomputable for PendulumSim {
    type State = (f64, f64);

    fn save_state(&self) -> Self::State {
        (self.theta, self.omega)
    }

    fn load_state(&mut self, state: &Self::State) {
        self.theta = state.0;
        self.omega = state.1;
    }

    fn bind_state(&mut self, ctx: &mut Context<f64>) -> Vec<AD<f64>> {
        let (th, _) = ctx.var(self.theta);
        let (om, _) = ctx.var(self.omega);
        self.state_ad = smallvec![th, om];
        self.state_ad.to_vec()
    }

    fn state(&self) -> &[AD<f64>] {
        &self.state_ad
    }

    fn step(&mut self, ctx: &mut Context<f64>) {
        self.steps_executed += 1;
        let inputs = [
            self.state_ad[0],
            self.state_ad[1],
            self.g,
            self.len,
            AD::constant(self.dt),
        ];
        let outs = ctx.call_custom_dyn(self.op.clone(), "pendulum_step", &inputs);
        self.theta = outs[0].value;
        self.omega = outs[1].value;
        self.state_ad = outs;
    }
}
