//! M5 验收：长稳与零泄漏——**每轮丢弃整个 Context**（线程局部路径，
//! 批量 rollout 场景）。独立测试二进制，原理见 leak_reuse.rs 的说明。

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

fn fresh_cycle() {
    let guard = ad_core::Context::<f64>::new().enter();
    let (x, _vx) = guard.var(3.0);
    let y = x * x + x;
    guard.backward(y);
    let _ = y;
}

#[test]
fn fresh_context_loop_is_leak_free() {
    for _ in 0..500 {
        fresh_cycle();
    }
    let baseline = LIVE.load(Ordering::Relaxed);

    for _ in 0..20_000 {
        fresh_cycle();
    }
    let after = LIVE.load(Ordering::Relaxed);

    assert!(
        after <= baseline + 8192,
        "memory leak suspected: baseline {baseline} B -> {after} B after 20k cycles"
    );
}
