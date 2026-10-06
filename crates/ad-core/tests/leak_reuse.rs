//! M5 验收（设计文档 §5.5）：长稳与零泄漏——**复用同一 Context** 的路径。
//!
//! 用计数全局分配器直接度量"当前存活字节数"：warmup 后循环
//! build → backward → clear_tape，存活字节数必须回到基线（容量复用使分配
//! 停止增长；任何泄漏都会表现为单调增长）。
//!
//! 注意：每个 tests/*.rs 是独立测试二进制。凡是带 `#[global_allocator]` 的
//! 测试文件内**只能有一个测试**（测试默认在同进程内并行，多个测试会互相
//! 污染测量窗口——这正是本文件独立成文的原因）。

use ad_core::{Context, CustomOp, AD};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static LIVE: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            if new_size >= layout.size() {
                LIVE.fetch_add(new_size - layout.size(), Ordering::Relaxed);
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        }
        p
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

struct ScaleOp;

impl CustomOp<f64> for ScaleOp {
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        2
    }
    fn forward(
        &self,
        inputs: &[f64],
    ) -> (smallvec::SmallVec<[f64; 8]>, smallvec::SmallVec<[f64; 8]>) {
        let x = inputs[0];
        (
            smallvec::smallvec![x * 2.0, x * 3.0],
            smallvec::smallvec![x],
        )
    }
    fn backward(&self, _residual: &[f64], grad_output: &[f64]) -> smallvec::SmallVec<[f64; 8]> {
        smallvec::smallvec![grad_output[0] * 2.0 + grad_output[1] * 3.0]
    }
    fn name(&self) -> &'static str {
        "scale"
    }
}

fn one_cycle(ctx: &mut Context<f64>, x: AD<f64>) {
    // 混合 Native + 常量折叠 + Custom(Rc 动态分发) 三类记录
    let y = ctx.mul(x, x);
    let z = ctx.mul(x, AD::constant(2.0));
    let s = ctx.add(y, z);
    let outs = ctx.call_custom(ScaleOp, &[s]);
    let loss = ctx.add(outs[0], outs[1]);
    ctx.backward(loss);
    ctx.zero_grads();
    ctx.clear_tape();
}

#[test]
fn reuse_context_loop_is_leak_free() {
    let mut ctx = Context::<f64>::new();
    let (x, _vx) = ctx.var(2.0);

    for _ in 0..500 {
        one_cycle(&mut ctx, x);
    }
    // 容量类增长（tape/伴随 Vec、HashMap）在 warmup 内达到稳态
    let baseline = LIVE.load(Ordering::Relaxed);

    for _ in 0..50_000 {
        one_cycle(&mut ctx, x);
    }
    let after = LIVE.load(Ordering::Relaxed);

    assert!(
        after <= baseline + 8192,
        "memory leak suspected: baseline {baseline} B -> {after} B after 50k cycles"
    );
    assert_eq!(ctx.tape_len(), 0);
}
