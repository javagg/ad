use crate::node::NodeId;
use crate::scalar::Scalar;
use smallvec::SmallVec;

/// 一条 tape 记录。SSA 风格：每条记录的输出节点由本记录独占产生，
/// 记录按追加顺序即拓扑序，逆序遍历即拓扑逆序（设计文档 §4.1.3）。
pub(crate) enum OpRecord<S: Scalar> {
    /// 基础算子：局部 Jacobian 已在前向时算好
    Native {
        op: &'static str,
        output: NodeId,
        inputs: SmallVec<[NodeId; 4]>,
        /// ∂output/∂input_i，与 inputs 一一对应
        jacobians: SmallVec<[S; 4]>,
    },
    /// 自定义算子：多输入多输出，反向调用 VJP。
    /// `inputs` 携带该输入在算子原始输入列表中的槽位（常量输入不占节点但占槽位）。
    /// 输出节点由 [`NodeId`](crate::NodeId) 单调分配保证**连续**，存 base 即可
    /// （反向时输出伴随 = `adjoints[base..base+n]` 的连续切片，无需收集）。
    /// 算子本体经 `Context` 的**注册表**共享（`op_id` 索引）——记录不持有
    /// `Rc`，省去每记录引用计数与 8 字节指针（§12.3 第 42 条）。
    Custom {
        name: &'static str,
        op_id: u32,
        inputs: SmallVec<[(u32, NodeId); 8]>,
        /// 首个输出节点（输出 n = op.num_outputs() 连续排布）
        output_base: NodeId,
        /// 前向保存的残差数据（f_fwd 风格），backward 时原样传回
        residual: SmallVec<[S; 8]>,
    },
}

/// Wengert List。
pub(crate) struct Tape<S: Scalar> {
    records: Vec<OpRecord<S>>,
}

impl<S: Scalar> Tape<S> {
    pub(crate) fn new() -> Self {
        Tape {
            records: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, record: OpRecord<S>) {
        self.records.push(record);
    }

    pub(crate) fn len(&self) -> usize {
        self.records.len()
    }

    pub(crate) fn records(&self) -> &[OpRecord<S>] {
        &self.records
    }

    pub(crate) fn clear(&mut self) {
        self.records.clear();
    }
}
