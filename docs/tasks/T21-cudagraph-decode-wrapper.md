# T21 · 补齐图模式 decode wrapper 的接口对标缺口

> 依赖：T9（图契约与冻结语义）、T10、T11 已完成并合入。
> 执行顺序：**排在 T15 之前**。理由见下「为什么现在做」。

---

## 背景（已核实的事实，不要重新推导）

### 已在本地核实

1. 本仓 `flashinfer/decode.py` 只有 `BatchDecodeWithPagedKVCacheWrapper` 一个
   decode wrapper。图能力是做进通用 wrapper 的：`__init__` 带
   `use_cuda_graph: bool = False` 和三个**可选**的
   `paged_kv_indptr_buffer` / `paged_kv_indices_buffer` /
   `paged_kv_last_page_len_buffer`（`flashinfer/decode.py:84-97`）。

2. **三个 buffer 的必填约束，父类已经实现了。**
   `flashinfer/attention/public_batch.py:63-65`：

   ```python
   if graph:
       if any(value is None for value in buffers):
           raise SchemaError("graph mode requires all fixed metadata buffers")
   ```

   `_init_facade` 在 `use_cuda_graph=True` 时就会逐个校验三个 buffer
   （`:57` 取 decode 的三个名字，`:66-71` 校验 dtype/device/可写，
   `:72-74` 推 `fixed_batch_size`，`:81` 校验 last_page_len 长度，
   `:106-109` 校验不与 workspace 别名）。

   **所以子类不需要新写任何校验逻辑。** 它要做的只是把签名收窄
   （三个 buffer 从可选变位置必填、`use_cuda_graph` 固定为 True），
   让错误在 Python 签名层就发生、并对上上游的调用习惯。
   **不要在子类里重复一遍 `_init_facade` 已经做的检查。**

3. 账本里 `flashinfer.decode.CUDAGraphBatchDecodeWithPagedKVCacheWrapper` 的状态是
   `semantic_status: unassessed` / `implementation_status: missing`。

4. **这条是自动兜底生成的，从没被人工评估过。**
   `tools/update_parity_inventory.py:328-332` 的 `refresh()` 对任何在
   `upstream.exported_symbols()` 里、但账本没有的符号，一律
   `setdefault` 成 `unassessed / missing / P2`。所以「missing」在这里
   **不代表「评估后决定不做」，而是「还没看过」**。

5. 同文件 `:23-27` 的 `ATTENTION_CLASSES` 只有三个类——两个 prefill wrapper
   加一个 `flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper`，
   **不含图模式 wrapper**。所以 `annotate_attention_workspace()` 给三个类
   挂的 `use_max_workspace` 说明（`:176-183`）也没覆盖它。

### 需要你在服务器上复核（我这边没有上游检出）

上游检出在 `/home/dy/projects/flashinfer-ori`（见
`tools/update_parity_inventory.py:348` 的默认值）。请自己核对后再动手：

- `flashinfer/decode.py` 里 `class CUDAGraphBatchDecodeWithPagedKVCacheWrapper`
  的完整定义：它继承谁、`__init__` 签名、哪些参数从可选变必填、
  有没有覆写 `plan` / `run`、docstring 怎么说。
- 它的 docstring 里关于「可能不如通用 wrapper 高效」的说法原文是什么、
  给的理由是什么。

**把这两段原文贴在回复靠前的位置。** 下面的设计要求建立在它们之上，
如果你核对出来的事实和本文档不符，**以你核对的为准，并在回复里指出来**。

---

## 为什么现在做

下游把本仓当上游替身用时，`from flashinfer.decode import
CUDAGraphBatchDecodeWithPagedKVCacheWrapper` 会直接 ImportError。
这是接口面的缺口，不是功能缺口——T9 的冻结语义已经覆盖了同样的约束。
T15/T16 下沉算子之后接口面只会更大，现在补最便宜。

---

## 要做什么

### 1. 实现子类

在 `flashinfer/decode.py` 加 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`，
继承 `BatchDecodeWithPagedKVCacheWrapper`。

**它只做一件事：把通用 wrapper 的「可选图资源」收紧成「必填」，
并强制 `use_cuda_graph=True`。** 执行路径完全复用父类。

- 三个 buffer 做成**位置必填参数**，顺序与参数名按你核对到的上游签名来。
  运行期校验交给父类（见背景第 2 条），**子类里不要再抄一份**。
- 不接受 `use_cuda_graph=False`（显式传 False 要报错，不要静默改写）。
- 默认值按上游签名来。

这是本任务唯一的实现产物，规模应当很小（数十行量级）。
**如果你写出了上百行，大概率是在重复父类已有的东西，停下来重看背景第 2 条。**

### 2. 不要做的事（这条最重要）

- **不要新增任何捕获逻辑。** 本仓不捕获图，捕获由调用方（vllm-ascend 的
  aclgraph）做。本仓只负责「可以被捕获」。子类同样只是约束入口，
  不碰 capture/replay。
- **不要重做或绕开 T9 的冻结检查。** batch_size、后端解析、
  q_len_per_req、block table 宽度这四个维度的冻结必须在子类上照常生效，
  走的就是父类那套，不要复制一份。
- **不要照抄上游 docstring 里「不如通用 wrapper 高效」那句话当事实。**
  那是 CUDA 侧按 batch/seq_len 分发到不同 kernel 的理由，在 NPU 的 FIA
  上不一定成立。要么实测给数据，要么在 docstring 里写清「上游如此说明，
  NPU 侧未实测」，**不要假装验证过**。
- 不要改 dispatch 的过滤与排序（T6/T7）。
- 不要改 capability 谓词。
- 不要动 `docs/tasks/` 和 `docs/TASKS.md`。
- 不要 git commit / push，改完留在工作区。

### 3. 登记账本

- `flashinfer/data/api_parity.json` 里该符号不能再是 `unassessed`。
- 判断它该不该进 `tools/update_parity_inventory.py` 的 `ATTENTION_CLASSES`。
  **先说理由再改**：进了会连带挂上 `use_max_workspace` 的说明和
  `npu_fused_infer_attention_score` 的 CANN 覆盖标记（`:59-65`、`:176-183`），
  要确认这些对子类成立才加。如果不该加，在回复里说清为什么。
- 跑 `tools/update_parity_inventory.py` 确认幂等（跑两遍结果一致）。

---

## 必须给出的可证伪对照

光说「实现了」不算，下面每条都要有实际命令和输出：

1. **必填校验是真的，而且走的是父类那条路**：
   - 子类构造时缺任意一个 buffer → 报错（贴出三种缺法各自的错误原文）。
   - 注释掉 `public_batch.py:63-65` 那个 `SchemaError` → 子类的对应测试
     **必须跟着失败**（贴失败数）。失败了才证明子类真的复用父类校验；
     **如果子类测试照常通过，说明你在子类里又抄了一份检查，那是本任务
     明确禁止的，要改掉**。
2. **T9 冻结在子类上照常生效**：构造后 plan 两次、第二次改 batch_size
   → 按 T9 报错，不是静默接受。block table 宽度同理。
3. **数值与通用 wrapper 逐位一致**：子类 vs
   `BatchDecodeWithPagedKVCacheWrapper(use_cuda_graph=True, <同样三个 buffer>)`，
   同输入下输出 `torch.equal` 为 True。不一致说明子类偷偷改了路径。
4. **buffer 地址不变**：plan 前后三个 buffer 的 `data_ptr()` 不变
   （父类已有此保证，子类不能破坏）。

---

## 验收判据

- 全量测试不差于基线：**1760 passed / 3 failed**
- 79 条记录 auto 选型**逐条不变**（`tools/t19_selection_audit.py`）
- `import flashinfer` 纯净：不触发编译、不探测编译器
- 上述四条可证伪对照全部给出实际输出
- 账本该条目不再是 `unassessed`，且 `update_parity_inventory.py` 幂等
- 环境：`/root/miniforge3/envs/flashinfer-a5/bin/python`

---

## 回复里要说清

1. 上游 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper` 的定义原文与
   docstring 原文（核对结果，和本文档不符的地方要指出来）
2. 子类签名逐参数对上游的对照表
3. 改了哪些文件，每个改了什么、为什么
4. 四条可证伪对照的实际命令与输出
5. `ATTENTION_CLASSES` 加不加，理由
6. 测试数字原样贴（不要只写「通过」）
7. 还有哪些没做完、哪些需要人确认
