use std::fmt;

/// 计算图节点标识。节点 ID 只对创建它的线程上的 `Context` 有意义。
///
/// 内部表示为 `u32`（§12.3 第 42 条 tape 瘦身）：AD 句柄与 tape 记录的
/// 输入数组随之减半；`clear_tape` 后节点计数回到叶子水位之上，单代
/// 记录数远低于 u32 上限（溢出由 debug_assert 守护）。
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(u32);

impl NodeId {
    pub(crate) fn new(index: usize) -> Self {
        debug_assert!(
            index <= u32::MAX as usize,
            "NodeId overflow: {} exceeds u32 (clear_tape resets the counter; \
             a single tape generation should not exceed 2^32 records)",
            index
        );
        NodeId(index as u32)
    }

    /// 节点在 `Context` 内部数组（伴随数组等）中的下标。
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// 叶子变量句柄：轻量级引用，不持有数据。
///
/// 梯度通过 [`Context::grad`](crate::Context::grad) 按此句柄读取。
/// 携带 `PhantomData<*mut ()>`，不可跨线程（`!Send + !Sync`）。
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Variable {
    pub node: NodeId,
    _not_send: std::marker::PhantomData<*mut ()>,
}

impl Variable {
    pub(crate) fn new(node: NodeId) -> Self {
        Variable {
            node,
            _not_send: std::marker::PhantomData,
        }
    }

    pub fn node(&self) -> NodeId {
        self.node
    }
}

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "var({:?})", self.node)
    }
}
