//! `ad` 库的交互式 web 演示（Yew + trunk，wasm32-unknown-unknown）。
//!
//! 三个面板对应库的三大能力，每个面板都有**实时有限差分验证徽章**：
//! 1. 标量表达式：`y = sin(a·b + c)` 的反向梯度；
//! 2. 单摆 1000 步 rollout：整步动力学 CustomOp + checkpoint 分段反向 + SVG 轨迹；
//! 3. IFT 隐式求解：隐式欧拉弹簧-阻尼 `solve(θ) → x*` 的伴随梯度。
//!
//! 所有计算在浏览器主线程同步完成（合计 < 5 ms），拖动滑块即时更新。

use ad::prelude::*;
// `Context` 在 yew::prelude 与 ad::prelude 中都有（glob 冲突），显式导入覆盖
use ad::{
    CheckpointManager, CheckpointStrategy, Context, GradientChecker, ImplicitSolve, PendulumSim,
    Residual, AD,
};
use num_traits::Num;
use web_sys::HtmlInputElement;
use yew::prelude::*;

// ============================================================ 通用滑块

#[derive(Properties, PartialEq)]
struct SliderProps {
    label: &'static str,
    value: f64,
    min: f64,
    max: f64,
    step: f64,
    digits: usize,
    on_change: Callback<f64>,
}

#[function_component(Slider)]
fn slider(p: &SliderProps) -> Html {
    let oninput = {
        let cb = p.on_change.clone();
        let fallback = p.value;
        Callback::from(move |e: InputEvent| {
            let v = e
                .target_dyn_into::<HtmlInputElement>()
                .and_then(|el| el.value().parse::<f64>().ok())
                .unwrap_or(fallback);
            cb.emit(v);
        })
    };
    html! {
        <div class="row">
            <label>
                <span class="name">{p.label}</span>
                <span class="val">{format!("{:.*}", p.digits, p.value)}</span>
            </label>
            <input
                type="range"
                min={p.min.to_string()}
                max={p.max.to_string()}
                step={p.step.to_string()}
                value={p.value.to_string()}
                {oninput}
            />
        </div>
    }
}

/// `UseStateHandle` → `Callback`：滑块回填用（Yew 0.21 的 setter 不直接实现 IntoPropValue）
fn state_cb(handle: &UseStateHandle<f64>) -> Callback<f64> {
    let h = handle.clone();
    Callback::from(move |v| h.set(v))
}

fn badge(max_rel: f64) -> Html {
    if max_rel.is_finite() && max_rel < 1e-6 {
        html! { <span class="badge ok">{"✅ 与有限差分一致"}</span> }
    } else {
        html! { <span class="badge warn">{format!("⚠️ 与有限差分偏差 {:.1e}", max_rel)}</span> }
    }
}

fn metric(key: &str, value: String) -> Html {
    html! {
        <div class="metric"><span class="k">{key}</span><span class="v">{value}</span></div>
    }
}

// ============================================================ 面板 1：标量表达式

struct ScalarOut {
    y: f64,
    da: f64,
    db: f64,
    dc: f64,
    fd_rel: f64,
}

fn compute_scalar(a: f64, b: f64, c: f64) -> ScalarOut {
    let guard = Context::<f64>::new().enter();
    let (x, vx) = guard.var(a);
    let (y_, vy) = guard.var(b);
    let (z, vz) = guard.var(c);

    let y = ad::sin(x * y_ + z);
    guard.backward(y);

    let g: Vec<f64> = [vx, vy, vz]
        .iter()
        .map(|&v| guard.grad(v).unwrap())
        .collect();

    // 有限差分 oracle（同一公式的纯 f64 版本）
    let f = |x: &[f64]| (x[0] * x[1] + x[2]).sin();
    let check = GradientChecker::default().check_scalar(f, &[a, b, c], &g);
    ScalarOut {
        y: y.value,
        da: g[0],
        db: g[1],
        dc: g[2],
        fd_rel: check.max_rel_error,
    }
}

#[function_component(ScalarPanel)]
fn scalar_panel() -> Html {
    let a = use_state_eq(|| 1.0f64);
    let b = use_state_eq(|| 2.0f64);
    let c = use_state_eq(|| 3.0f64);
    let out = compute_scalar(*a, *b, *c);

    html! {
        <section class="card">
            <h2>{"1 · 标量反向模式"}</h2>
            <p class="desc">{"y = sin(a·b + c)，tape 记录 + 逆序伴随传播，三个叶子梯度一次反向全部得到。"}</p>
            <div class="cols">
                <div class="left">
                    <Slider label="a" value={*a} min={-3.0} max={3.0} step={0.05} digits={2} on_change={state_cb(&a)} />
                    <Slider label="b" value={*b} min={-3.0} max={3.0} step={0.05} digits={2} on_change={state_cb(&b)} />
                    <Slider label="c" value={*c} min={-3.0} max={3.0} step={0.05} digits={2} on_change={state_cb(&c)} />
                </div>
                <div class="right">
                    {badge(out.fd_rel)}
                    <div class="metrics">
                        {metric("y", format!("{:.6}", out.y))}
                        {metric("∂y/∂a = cos(·)·b", format!("{:.6}", out.da))}
                        {metric("∂y/∂b = cos(·)·a", format!("{:.6}", out.db))}
                        {metric("∂y/∂c", format!("{:.6}", out.dc))}
                    </div>
                </div>
            </div>
        </section>
    }
}

// ============================================================ 面板 2：单摆 + checkpoint

const T: usize = 1000;
const DT: f64 = 0.01;

struct PendulumOut {
    traj: Vec<f64>,
    dg: f64,
    dl: f64,
    dth0: f64,
    fd_rel: f64,
    snapshots: usize,
}

fn compute_pendulum(g: f64, len: f64, th0: f64, target: f64) -> PendulumOut {
    // 前向（no_grad）+ 快照；整步动力学 = 一个 CustomOp（PendulumStep）
    let mut ctx = Context::<f64>::new();
    let (g_ad, vg) = ctx.var(g);
    let (l_ad, vl) = ctx.var(len);
    let mut sim = PendulumSim::new(&mut ctx, th0, 0.0, g_ad, l_ad, DT);
    let init = sim.bind_state(&mut ctx);
    let mut ckpt = CheckpointManager::new(CheckpointStrategy::Uniform { interval: 50 }, &sim);

    let mut traj = Vec::with_capacity(T + 1);
    traj.push(th0);
    for t in 0..T {
        ckpt.forward_step(&mut ctx, &mut sim, t);
        traj.push(sim.state()[0].value);
    }
    let snapshots = ckpt.num_snapshots();

    // 分段反向（边界伴随传递）：loss = (θ_T − target)²
    let loss = |ctx: &mut Context<f64>, sim: &PendulumSim| {
        let th = sim.state()[0];
        let d = ctx.sub(th, AD::constant(target));
        ctx.mul(d, d)
    };
    let init_adj = ckpt.backward(&mut ctx, &mut sim, &loss);
    let dg = ctx.grad(vg).unwrap();
    let dl = ctx.grad(vl).unwrap();
    let dth0 = init_adj[0];

    // 有限差分验证：4 次纯前向 rollout（无 AD）
    let rollout_loss = |gp: f64, lp: f64| -> f64 {
        let mut ctx = Context::<f64>::new();
        let mut sim = PendulumSim::new(&mut ctx, th0, 0.0, AD::constant(gp), AD::constant(lp), DT);
        for _ in 0..T {
            sim.step(&mut ctx);
        }
        let th = sim.state()[0];
        let d = ctx.sub(th, AD::constant(target));
        ctx.mul(d, d).value
    };
    let h = 1e-5;
    let fd_g = (rollout_loss(g + h, len) - rollout_loss(g - h, len)) / (2.0 * h);
    let fd_l = (rollout_loss(g, len + h) - rollout_loss(g, len - h)) / (2.0 * h);
    let fd_rel =
        ((fd_g - dg).abs() / (1.0 + fd_g.abs())).max((fd_l - dl).abs() / (1.0 + fd_l.abs()));
    let _ = init;

    PendulumOut {
        traj,
        dg,
        dl,
        dth0,
        fd_rel,
        snapshots,
    }
}

#[function_component(PendulumPanel)]
fn pendulum_panel() -> Html {
    let g = use_state_eq(|| 9.81f64);
    let len = use_state_eq(|| 1.0f64);
    let th0 = use_state_eq(|| 0.6f64);
    let target = use_state_eq(|| 1.5f64);
    let out = compute_pendulum(*g, *len, *th0, *target);

    // θ(t) 轨迹 → SVG polyline（降采样 ~240 点，显示范围 ±3.2 rad）
    const W: f64 = 640.0;
    const H: f64 = 220.0;
    const SCALE: f64 = 3.2;
    let n = out.traj.len();
    let step = (n / 240).max(1);
    let pts: String = out
        .traj
        .iter()
        .enumerate()
        .step_by(step)
        .map(|(i, th)| {
            let x = 8.0 + (i as f64 / (n - 1).max(1) as f64) * (W - 16.0);
            let y = H / 2.0 - (th.clamp(-SCALE, SCALE) / SCALE) * (H / 2.0 - 10.0);
            format!("{x:.1},{y:.1}")
        })
        .collect::<Vec<_>>()
        .join(" ");
    let ty = H / 2.0 - (target.clamp(-SCALE, SCALE) / SCALE) * (H / 2.0 - 10.0);
    let theta_t = out.traj[n - 1];

    html! {
        <section class="card">
            <h2>{"2 · 单摆长轨迹 + checkpoint"}</h2>
            <p class="desc">
                {format!("T = {T} 步，每步动力学封装为一个 CustomOp；no_grad 前向存 {} 个快照，分段反向 + 边界伴随传递。拖动 g / L / θ₀ 改变轨迹。", out.snapshots)}
            </p>
            <div class="cols">
                <div class="left">
                    <Slider label="g 重力" value={*g} min={1.0} max={20.0} step={0.1} digits={2} on_change={state_cb(&g)} />
                    <Slider label="L 摆长" value={*len} min={0.3} max={3.0} step={0.05} digits={2} on_change={state_cb(&len)} />
                    <Slider label="θ₀ 初始角" value={*th0} min={-3.0} max={3.0} step={0.05} digits={2} on_change={state_cb(&th0)} />
                    <Slider label="目标角" value={*target} min={-3.0} max={3.0} step={0.05} digits={2} on_change={state_cb(&target)} />
                </div>
                <div class="right">
                    {badge(out.fd_rel)}
                    <svg class="plot" viewBox={format!("0 0 {W} {H}")} preserveAspectRatio="none">
                        <line class="zero" x1="8" y1={(H / 2.0).to_string()} x2={(W - 8.0).to_string()} y2={(H / 2.0).to_string()} />
                        <line class="target" x1="8" y1={ty.to_string()} x2={(W - 8.0).to_string()} y2={ty.to_string()} />
                        <polyline class="curve" points={pts} />
                    </svg>
                    <div class="metrics">
                        {metric("θ(T)", format!("{:.3} rad", theta_t))}
                        {metric("∂L/∂θ₀", format!("{:.3e}", out.dth0))}
                        {metric("∂L/∂g", format!("{:.3e}", out.dg))}
                        {metric("∂L/∂L", format!("{:.3e}", out.dl))}
                    </div>
                </div>
            </div>
        </section>
    }
}

// ============================================================ 面板 3：IFT 隐式求解

/// 隐式欧拉弹簧-阻尼残差（与 ad-custom 测试一致）
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

struct IftOut {
    q1: f64,
    v1: f64,
    grads: Vec<f64>,
    fd_rel: f64,
}

fn compute_ift(q0: f64, v0: f64, h: f64, k: f64, c: f64) -> IftOut {
    let op = ImplicitSolve::new(SpringDamper);
    let loss_of = |th: &[f64]| -> f64 {
        // forward = Newton 求解（确定性），loss = q1² + 2·v1
        let (x, _) = op.forward(th);
        x[0] * x[0] + 2.0 * x[1]
    };

    let mut ctx = Context::<f64>::new();
    let mut vars = Vec::new();
    let mut theta = Vec::new();
    for v in [q0, v0, h, k, c] {
        let (ad, var) = ctx.var(v);
        theta.push(ad);
        vars.push(var);
    }
    let out = ctx.call_custom(ImplicitSolve::new(SpringDamper), &theta);
    let q1 = out[0].value;
    let v1 = out[1].value;

    let q1s = ctx.mul(out[0], out[0]);
    let two_v1 = ctx.mul(AD::constant(2.0), out[1]);
    let loss_ad = ctx.add(q1s, two_v1);
    ctx.backward(loss_ad);
    let grads: Vec<f64> = vars.iter().map(|&v| ctx.grad(v).unwrap()).collect();

    // 有限差分 oracle：直接对 forward 求值路径做中心差分
    let th = [q0, v0, h, k, c];
    let eps = 1e-5;
    let mut fd_rel = 0.0f64;
    for j in 0..5 {
        let mut tp = th;
        tp[j] += eps;
        let mut tm = th;
        tm[j] -= eps;
        let fd = (loss_of(&tp) - loss_of(&tm)) / (2.0 * eps);
        fd_rel = fd_rel.max((fd - grads[j]).abs() / (1.0 + fd.abs()));
    }

    IftOut {
        q1,
        v1,
        grads,
        fd_rel,
    }
}

#[function_component(IftPanel)]
fn ift_panel() -> Html {
    let q0 = use_state_eq(|| 1.0f64);
    let v0 = use_state_eq(|| 0.5f64);
    let h = use_state_eq(|| 0.1f64);
    let k = use_state_eq(|| 3.0f64);
    let c = use_state_eq(|| 0.4f64);
    let out = compute_ift(*q0, *v0, *h, *k, *c);

    html! {
        <section class="card">
            <h2>{"3 · 隐式求解器可微（IFT）"}</h2>
            <p class="desc">
                {"隐式欧拉弹簧-阻尼：求解器整体封装为一个 CustomOp，反向只解一次线性伴随系统（内存 O(1)、无迭代截断偏差）。"}
            </p>
            <div class="cols">
                <div class="left">
                    <Slider label="q₀ 初位移" value={*q0} min={-2.0} max={2.0} step={0.05} digits={2} on_change={state_cb(&q0)} />
                    <Slider label="v₀ 初速度" value={*v0} min={-2.0} max={2.0} step={0.05} digits={2} on_change={state_cb(&v0)} />
                    <Slider label="h 步长" value={*h} min={0.01} max={0.2} step={0.01} digits={2} on_change={state_cb(&h)} />
                    <Slider label="k 刚度" value={*k} min={0.0} max={20.0} step={0.1} digits={1} on_change={state_cb(&k)} />
                    <Slider label="c 阻尼" value={*c} min={0.0} max={3.0} step={0.05} digits={2} on_change={state_cb(&c)} />
                </div>
                <div class="right">
                    {badge(out.fd_rel)}
                    <div class="metrics">
                        {metric("q₁（解）", format!("{:.6}", out.q1))}
                        {metric("v₁（解）", format!("{:.6}", out.v1))}
                        {metric("∂L/∂k", format!("{:.3e}", out.grads[3]))}
                        {metric("∂L/∂c", format!("{:.3e}", out.grads[4]))}
                        {metric("∂L/∂h", format!("{:.3e}", out.grads[2]))}
                        {metric("∂L/∂q₀", format!("{:.3e}", out.grads[0]))}
                        {metric("∂L/∂v₀", format!("{:.3e}", out.grads[1]))}
                    </div>
                </div>
            </div>
        </section>
    }
}

// ============================================================ App

#[function_component(App)]
fn app() -> Html {
    html! {
        <main>
            <header>
                <h1><em>{"ad"}</em>{" · 面向物理仿真的反向模式自动微分"}</h1>
                <p>{"纯 Rust · tape-based · 编译到 wasm 运行于浏览器。三个面板分别演示标量求导、长轨迹 checkpoint 分段反向、隐式求解器 IFT 伴随——绿色徽章表示梯度已通过实时有限差分交叉验证。"}</p>
            </header>
            <ScalarPanel />
            <PendulumPanel />
            <IftPanel />
            <footer>{"运行于 wasm32-unknown-unknown · 用 trunk serve 启动 · 源码见 crates/"}</footer>
        </main>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
