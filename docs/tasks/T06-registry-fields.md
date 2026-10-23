# T6 · 给 `KernelDescriptor` / `CapabilitySpec` 加字段

| | |
|---|---|
| **阶段** | P2 · 地基扩展 |
| **执行序号** | 6 / 20 |
| **依赖** | T1 |
| **规模** | 大 |
| **状态** | `已完成` |

> 索引与全局约束见 [`../TASKS.md`](../TASKS.md)。
> 架构背景与论证见配套的架构说明文档。
> **动手前先读 `../TASKS.md` 的「全局约束」一节。**

## 进度

- [ ] **T6.1** 执行体字段 — 解锁 AOT/JIT 与 csrc 下沉
- [ ] **T6.2** 「外部仓」身份类别 — 解锁外部高性能实现路由
- [ ] **T6.3** shape / 规模区间字段
- [ ] **T6.4** 昇腾物理属性
- [ ] **T6.5** 入图能力声明 — 解锁图模式

---

## 阶段背景（P2 · 地基扩展）

> **四件事里有三件卡在同一个地方**：登记表能描述的东西太少。
> 这不是某个功能的缺陷，是公共地基的缺陷。

---

## 验收粒度

**本任务按子项逐个验收**（T6.1 … T6.5）。
**索引里 T6 的勾选框，要等五个子项全部通过之后才打勾。**

子项内部也有顺序约束，见正文——其中 **T6.3（规模区间）一条挡三条**
（两级选择的第一级、性能数据索引、图模式按 bucket 固化），优先级要靠前。

---

## 任务正文

**依赖**：T1

**文件**：`flashinfer/runtime/schema.py`、`flashinfer/runtime/hardware.py`、
`flashinfer/runtime/registry.py`、`flashinfer/runtime/dispatch.py`

**现状**：
```python
@dataclass(frozen=True)
class KernelDescriptor:
    kernel_id: str
    backend: Backend
    capability: CapabilitySpec
    compile_mode: CompileMode = CompileMode.NONE
    priority: int = 0
    runner: Optional[KernelRunner] = None     # ← 只能是 Python 函数
```
```python
@dataclass(frozen=True)
class CapabilitySpec:
    op: str
    dtypes / layouts / soc_models / min_aicore / specializations   # ← 无任何 shape 字段
```

**按以下顺序加**（顺序重要，前者是后者的前提）：

#### T6.1 · 执行体字段 — 解锁 AOT/JIT 与 csrc 下沉

加 `artifact: Optional[ArtifactRef]`（产物路径 + 入口符号 + 格式）
与 `source: Optional[SourceRef]`（语言 + 源码路径 + 哈希）。
同时需要 **SOC 型号作为编译维度**（同一 kernel 在 910B / 910_93 / 310P 上是不同产物）。

没有这个字段，**自定义算子编出来也无处登记**，vllm-ascend 的 `csrc/` 也搬不过来（T13）。

#### T6.2 · 「外部仓」身份类别 — 解锁外部高性能实现路由

**问题**：现在 `Backend` 只有 5 个值，把实现二分为「内置（默认参与自动选择）」和
「用户注册（默认**不**参与）」。而**外部高性能仓两边都不是**——
它不随本仓发布，但也不是用户临时塞进来的，是本仓主动选择、验证过、认为更快才对接的。

今天只能二选一，两条都不对：
- 登记成 `cann` → 语义造假，且被排序规则钉在固定档位
- 登记成 `custom` → **默认不参与自动选择**（`FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS` 默认 `"0"`），
  一个一等产品功能默认是关闭的

**做什么**：新增 `Backend.EXTERNAL`（或等价身份字段），**默认参与 auto 候选池**，
与 `custom`（用户临时注册、需 opt-in）区分。
`register_custom()` 对 `custom/` 前缀的强制保留给 `custom`，外部仓用自己的命名空间。

**注意**：设计文档 §10.1 的接入示例用的是 `mycompany/...` 前缀，
而代码强制字面 `custom/` —— 照文档抄会抛错。此任务要顺便修正文档。

#### T6.3 · shape / 规模区间字段

现在**任何 head_dim、page_size 都能通过筛选**，然后在底层算子内部失败，
违反了设计文档自己定的「所有降级必须可见」。

三件事同时建立在这个字段上：按规模分档调优、图模式按 bucket 固化选择、选择结果的正确性。

#### T6.4 · 昇腾物理属性

- `CapabilitySpec` 加**物理排布**字段（NZ 分形 / ND）——现在没地方说明
- `HardwareSignature` 加 **UB（统一缓冲区）容量**——写 Ascend C 算子时决定 tiling 的首要约束

#### T6.5 · 入图能力声明 — 解锁图模式

加 `graph_capability` 字段，三选一：

| 值 | 含义 |
|---|---|
| `capture_safe` | 捕获即可用，重放不需要任何干预 |
| `needs_param_update` | 需要在重放前改写已录制任务的参数（tiling 随 seq_len 变） |
| `not_capturable` | 不能入图，该层必须留在图外（对应 PIECEWISE 模式） |

对 `needs_param_update`，记录自己要交代**「改哪些参数、怎么改」**——
由框架在重放前统一执行。

> 参考：vllm-ascend 今天是用 `use_updatable_graph(attn_backend)` 做这个判断，
> 实现是 `issubclass(attn_backend, AscendAttentionBackend)` —— **靠类型判断猜能力**。
> 这个字段就是要取代它。

**T6 验收**：
```bash
python3 -c "
from flashinfer.runtime.schema import KernelDescriptor, CapabilitySpec
import dataclasses
print([f.name for f in dataclasses.fields(KernelDescriptor)])
print([f.name for f in dataclasses.fields(CapabilitySpec)])
"
pytest tests/ -q
```

---

## 完成时填写

- **实际改动**：T6.1 执行体字段（ArtifactRef.entry_point / SourceRef / target_soc，to_dict 保旧指纹）；T6.2 外部仓身份与 register_external；T6.3 ShapeRange 规模区间（闭区间）；T6.4 physical_layouts 只认 ND(2)/NZ(29)、HardwareSignature.ub_size_bytes；T6.5 GraphCapability 与 graph_param_updates
- **验收结果**：逐子项验收通过；79 条记录 auto 选型逐条不变；A5 实测 UB=221184（ACL 属性 204）
- **遗留问题**（不要顺手改，记在这里）：UB 探测在部分上下文返回 0，已由 unknown 不作通配兜底（见 T20B）
- **提交**：`a12742d` 等 T6.1~T6.5 系列提交
