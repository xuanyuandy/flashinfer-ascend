# T8 · 建立 torch.library 注册层与 meta 实现

| | |
|---|---|
| **阶段** | P3 · 图模式 aclgraph |
| **执行序号** | 11 / 20 |
| **依赖** | T1 |
| **规模** | 中 |
| **状态** | `已完成` |

> 索引与全局约束见 [`../TASKS.md`](../TASKS.md)。
> 架构背景与论证见配套的架构说明文档。
> **动手前先读 `../TASKS.md` 的「全局约束」一节。**

## 进度

- [x] 完成并通过验收

---

## 阶段背景（P3 · 图模式 aclgraph）

> **路线已定**：按 aclgraph 做，torchair 最多留兼容。
> 依据：vllm-ascend 已把 aclgraph 作为默认图模式，torchair 标记废弃、v0.12.0rc1 移除、
> 最晚 2026 Q1。该仓 aclgraph 覆盖 20+ 文件，torchair 只剩一段 `ImportError` 兜底。
>
> **核心原则：本仓不捕获图，只负责让自己「可以被捕获」。** 与上游一致。

### 责任划分（动手前先读这张表）

| 动作 | 上游 GPU | 本仓昇腾 |
|---|---|---|
| 调 capture / replay | vLLM | **vllm-ascend，不下沉** |
| 决定捕获哪些 batch size、分桶 | vLLM | **vllm-ascend，不下沉** |
| 分配并持有固定 buffer | vLLM 分配后传入 | **vllm-ascend，不下沉** |
| plan/run 划分、冻结校验 | **FlashInfer** | **本仓**（照搬上游） |
| 最大 workspace 查询 | 不需要 | **本仓**（plan 期，新增） |
| 重放前改写 tiling 参数 | 不需要 | **本仓**（新增，GPU 无对应物） |

**捕获的范围是整个模型 forward**（embedding + N 层 + norm + 采样），
本仓只看得见一个算子，**结构上不可能也不应该**去捕获。

**但最后两行必须下沉**，因为它们是**算子级知识**：哪个算子、第几个参数位、
装的是什么 tiling、每步怎么变。今天它待在 vllm-ascend 的 attention 后端里，
**只是因为算子调用点在那儿**。若调用点下沉而这段不跟着下沉，
就变成「本仓发起调用、vllm-ascend 却要知道它的参数布局才能去补」——
一个算子的知识被劈在两个仓里，比现状更糟。

---

---

## 任务正文

**依赖**：T1

**为什么要做**：本仓**全仓 0 处** `torch.library` / `register_custom_op`。
对比：
- 上游有完整脚手架：`register_custom_op` / `register_fake_op` 定义在 `flashinfer/utils.py`，
  **194 个装饰点、128 个 fake op**（当前出于开销考虑短路成 `lambda x: x`，
  但结构完整，打开只是一行的事）
- vllm-ascend 有 **1,899 行** `torch_binding_meta.cpp` 专门写 meta 实现

**后果**：没有注册和 meta 实现，本仓的算子**无法被 torch.compile 追踪**，
在 PIECEWISE 图模式下会造成 graph break；也无法参与任何需要 fake tensor 推导的流程。

**做什么**：
1. 在 `flashinfer/utils.py` 建立与上游同名的 `register_custom_op` / `register_fake_op`
   （对标：同名同签名，便于后续代码直接对拍）。
2. 给已接入的算子加装饰与 meta（只推形状/dtype，不做计算、不分配设备内存）。
3. 验证 `torch.compile` 下不产生 graph break。

**验收**：
```bash
python3 -c "
import torch, flashinfer
# 对一个已接入算子做 torch.compile，检查无 graph break
"
TORCH_LOGS=graph_breaks python3 <你的最小复现脚本>
```

---

## 完成时填写

- **实际改动**：建立与上游同名同签名的 register_custom_op / register_fake_op；helper 被调用时真实委托 torch.library，builtin 默认仍 eager，需显式 enable_torch_library()；首批给 rmsnorm / fused_add_rmsnorm 加装饰与 meta（均返回 void，meta 只推形状/dtype）；flashinfer/utils/ 空占位包改为 utils.py 模块
- **验收结果**：npu + inductor + fullgraph=True 下 graph_count=1 graph_break_count=0；正对照（故意插入 graph_break）确实打出 [__graph_breaks] 日志，证明该 0 可被证伪；FakeTensorMode 下连调 50 次设备内存恒为 (0,0,0,0)；1592 passed / 3 failed；79/79 选型不变；import 后 torch_npu/triton 均未加载
- **遗留问题**（不要顺手改，记在这里）：上游 helper 当前为 no-op，本仓真实注册属有意差异，已登记账本；其余算子的装饰与 meta 未铺开
- **提交**：`64e748b`
