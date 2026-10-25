# FlashInfer 框架：设计、实现现状与对标路线图

> 文档状态：Framework design + implementation baseline，专家评审版。
> 本仓基线：`infer` @ `1c9c750`；日期：2026-10-25。
> 上游只读基线：`da992ed99`；上游文件统一写作 `flashinfer-ori: path`。
> 实测环境：8× Ascend950PR，CANN 9.2.0-beta.2，torch 2.10.0+cpu，torch_npu 2.10.0.post7。
> 指定解释器：`/root/miniforge3/envs/flashinfer-a5/bin/python`。
> 本次是文档更新；测试数字引用已核实基线，没有重新执行设备或性能测试。

**阅读图例**：`[V]` 有限定范围的运行证据 · `[P]` 部分接通 · `[G]` 明确缺口。
状态标记不表示全量 dtype、shape、SoC、版本或并发组合均获验证。
`[V]` 也不等于生产成熟、性能领先或与上游完全等价。
每节先写当前状态；“目标设计”“后续验收”表达待达成要求。
细分状态并列时，分别限定已验证机制和未交付范围，不能合并读作完成。

**评审主结论**：分层、内容寻址和显式资源契约值得保留。
当前绝大多数接口依托 CANN aclnn，生产 Ascend C 自研只有两条有限范围路径。
自动 AOT 消费和 single-decode 公共图资源接口已接通；全库交付、batch 入图与扩域仍有明显缺口。
没有做过同口径性能测量，本文不给性能排名。

**编号说明**：保留原 §1–§16 的主干编号。
原 §3.4 已有图模式子节，本版将其升为新增独立节 **§3A**，插在 §3 与 §4 之间。
§3.4 留作迁移入口；原 §3.4.1–§3.4.7 的内容按机制归并到 §3A。
其余主节不顺延；原 §4 的子节空号补为 §4.2；附录 A/B 放在末尾。

| 快速入口 | 读者需要判断的问题 |
| --- | --- |
| §1–§2 | 项目做什么，架构边界及与上游的差异是什么？ |
| §3A、§7、§8、§10 | 图模式、AOT、JIT、自定义算子四条轴线到哪一步？ |
| §12、§14、附录 A | 错误语义、证据强度和当前风险是否可信？ |
| §15、附录 B | 两个月能交付什么，依赖和退出条件是什么？ |

## 1. 概述

当前状态 `[P]`：已有可调用接口与有限真机闭环，尚未形成覆盖上游全部能力域的生产交付。

### 1.1 项目定位

当前状态 `[P]`：本仓对标 FlashInfer 的 Python 接口与数学语义，底座为昇腾 NPU。

接口名称相同不代表参数组合、图能力、数值误差或运行性能相同。
本仓不是 CUDA 二进制兼容层，也不是完整 serving 引擎。
模型调度、分桶、图区域选择和图生命周期由调用方负责。

- **接口对标**：尽量保留函数名、参数语义和 wrapper 生命周期，差异必须明确。
- **多个 backend**：按输入契约、目标和策略选择已登记实现；不是每个接口都已有多个 NPU backend。
- **当前执行主线**：绝大多数接口走 CANN aclnn；REFERENCE 是 host 参考路径。
- **自研实现**：Ascend C 的 native v2 与 catlass v2 各有一条生产登记；Triton 有有限生产入口。
- **AOT/JIT**：对已登记并绑定 `AotKernelRunner` 的 Ascend C 算子，自动发现、命中、绑定和受控回落已接通。
- **交付目标**：逐步让选定生产组合在无编译器环境命中 AOT；尚不能称“全库默认 AOT 交付”。

当前登记口径固定采用已核实的 **81 条 descriptor / 43 个 op**。
backend 分布为 **ascendc 2 / CANN 40 / REFERENCE 38 / TRITON 1**。
生产自研算子为 `matmul_bias_silu` native v2、`flash_attention` catlass v2。
此前 `flashinfer/data/kernels/registry.json` 的 `kernels` 为 `[]`；该历史空值已不代表当前状态。

| 统计对象 | 基线及含义 |
| --- | --- |
| runtime descriptor | 81 条；候选声明，不等于独立算子数或逐组合实测数。 |
| op 接口面 | 43 个；按已核实能力域清单口径引用，不据内部别名重新统计。 |
| 生产 Ascend C | 2 条；不是全库两个接口，也不是两个通用 family 已全覆盖。 |
| 可入图声明 | 1/81，唯一 `NEEDS_PARAM_UPDATE`；`CAPTURE_SAFE` 为 0。 |
| 全量测试 | 1975 passed / 0 failed；不是 1975 个 NPU kernel。 |
| `-m real_compile` | 80 passed / 0 skipped；真机环境见文首。 |

### 1.2 设计目标

当前状态 `[P]`：确定性调度、构建完整性已有证据；接口全集、无分配热路径和广泛部署仍是目标。

1. 对选定推理 workload 提供可度量的接口、数值、异常和资源语义对齐。
2. 同一输入、registry、配置及 tuning 快照下，选择结果确定且可解释。
3. 选定预编译矩阵可在无编译器部署中运行；未覆盖组合按明确策略处理。
4. 构建身份覆盖影响二进制语义的全部已声明输入，完整性优先于命中率。
5. 将发现、编译、调优、资源准备移出可复用执行路径，按入口分别验收。
6. 图能力按 descriptor、输入范围和真实 capture/replay 证据授予。

“热路径无分配、无同步、只 launch”保留为优化目标。
当前 CANN composite 含循环、gather、临时 Tensor；生产 Ascend C adapter 仍同步 eager 执行。
不能把目标句直接用于描述所有 `run()`。

### 1.3 非目标

当前状态 `[P]`：范围已界定；这些选择不抵消范围内未补齐的能力。

- 不模拟 CUDA runtime，不承诺 CUDA binary 直接运行，不逐行翻译 CUDA kernel。
- 不在首期覆盖上游全部实验后端、模型专用路径或训练/autograd。
- 不提供模型级 FULL/PIECEWISE 切图、请求调度或服务端 KV 分配器。
- 不复刻 NVIDIA 专有 MNNVL、NVSHMEM、CUDA IPC 与 PDL 实现。

通信的数学语义和 HCCL 适配仍是尚未补齐项，不能随 NVIDIA 专有实现一起排除。
MLA、稀疏 attention、量化、linear attention 等也不能以硬件不同为理由改名为非目标。
两个月范围选择见附录 B；延期不等于永久放弃。

### 1.4 术语与证据口径

当前状态 `[P]`：下列对象有实现或契约，但各自覆盖范围不同。

| 术语 | 含义 |
| --- | --- |
| interface / op | 前者是公共 Python 入口，后者是内部语义标识，两者不保证一一对应。 |
| backend / kernel | backend 表示来源或实现类别；kernel descriptor 标识一个具体候选。 |
| descriptor / runner | descriptor 声明身份和能力；runner 是实际执行 callable。 |
| recipe / JitSpec | recipe 声明构建输入与 ABI；JitSpec 是冻结的构建规格及内容身份。 |
| AOT / JIT | 提前或按需准备产物，与 native、catlass 等实现来源正交。 |
| op-package | CANN vendor 布局的 host 库、设备 binary、metadata 与验证记录集合。 |
| plan / run | 准备状态与执行的职责分离，不自动承诺所有实现均无同步或无分配。 |
| torch.compile / aclgraph | 分别是程序追踪编译和设备任务捕获重放，必须分别验证。 |
| envelope | 明确支持的 dtype、shape、layout、选项、资源和执行模式范围。 |

事实优先级为：本任务给定最新基线 → 后续交付报告的增量机制 → 早期架构报告。
早期报告用来保留设计判断和风险背景，其历史计数、阻塞状态不覆盖后续交付。
冲突裁决集中列于附录 A；本次不改写任何历史报告。

## 2. 总体架构

当前状态 `[P]`：执行主线、构建支线及两条生产接线已存在，Attention generator 与通用部署仍有断点。

### 2.1 架构分层

当前状态 `[P]`：职责已分离，具体调用不一定逐层经过自研编译链。

```text
调用方：模型 / serving / graph owner
  ↓
L4  公共 API、wrapper：参数、资源、plan/run
  ↓
L3  runtime：signature → registry → dispatch → decision
  ↓
L2  已绑定 runner
  ├─ CANN aclnn / composite → torch_npu / ACL
  ├─ Triton 原生执行路径 → triton-ascend
  └─ Ascend C lazy runner → AOT discovery / 受控 JIT
       ↓
L3' 构建支线：recipe → JitSpec → store → compiler → verified module
       ↓
L1  设备源码与 host OpDef/tiling → op-package / host_shared
       ↓
底座：CANN / ACL Runtime / Ascend AI Core
```

`flashinfer/runtime` 是本仓的调度代码；ACL Runtime 是外部设备运行库。
调度层选择候选，runner 编排实际执行；编译不属于每次 dispatch 的必经步骤。
AOT/JIT 是获取执行 module 的方式，不是与 CANN、Ascend C 并列的数学实现。

```text
常见 eager：公共 API → signature → dispatch → 已绑定 CANN runner → NPU
生产自研：公共 API → dispatch → AotKernelRunner.resolve（首次）
                         → AOT hit，或 missing 且允许 JIT
                         → adapter → ACLNN 两段执行
重复自研调用：已解析 runner → adapter → 执行
batch attention：plan 保存身份/资源；run 仍逐请求组合 FIA
```

图是当前职责示意，不用旧 `architecture_call_path.svg` 证明自动交付或热路径性质。
具体证据见 [分层报告](reports/call-stack-layering.md) §2，增量见 §7、§10。

### 2.2 层次职责

当前状态 `[P]`：职责边界是约束；现有 composite 与独立 Attention 契约尚未完全归一。

| 层 | 当前落点 | 目标职责与实际边界 |
| --- | --- | --- |
| API / wrapper | `flashinfer/decode.py`、`prefill.py`、各域门面 | 校验、资源与生命周期；不能只凭同名入口宣布语义对齐。 |
| 调度 | `flashinfer/runtime/` | 能力过滤与可解释选择；不负责扫描产物或执行失败重试。 |
| 构建 | `flashinfer/jit/`、`flashinfer/aot/` | 身份、构建、验证、导出、发现；冷路径须先于 capture 完成。 |
| 执行 | `flashinfer/cann/`、`triton/`、`ascendc.py` | 真实 callable、stream/workspace/错误边界；同步程度逐路径说明。 |
| 源码与 host | `csrc/kernels/ascendc/`、`csrc/op_package/` | 数学实现、OpDef、infer、tiling；源码不会因注册自动生成。 |
| 设备环境 | `runtime/hardware.py` 与外部运行库 | 当前快照能力有限，不能替代逐设备和软件兼容治理。 |

通用 `Plan` 与 Attention 的 workspace/session 契约并存。
通用 `Plan.workspace_bytes` 的零占位不能解释成所有算子零 workspace。
同样不能把 Attention 的局部资源规划说成全库统一 scheduler。

### 2.3 仓库目录结构

当前状态 `[P]`：以下列真实职责位置；原稿的理想目录不是当前部署布局。

```text
flashinfer/
  decode.py、prefill.py、gemm.py、norm.py 等       公共 API
  attention/                                    plan、metadata、graph wrapper
  runtime/                                      registry、dispatch、ABI、graph 参数
  cann/                                         CANN 适配与 composite
  triton/                                       生产 Triton 路线
  ascendc.py                                    生产 descriptor 与 lazy runner
  jit/                                          recipe、spec、store、loader、adapter
  aot/                                          构建/export；运行期发现见 jit/discovery.py
  tuning/                                       采集、存储、查询
  data/kernels/                                 descriptor 与冻结 spec JSON
  data/op_packages/<spec-sha>/                   生成/安装产物布局，非默认已有 binary
csrc/
  kernels/ascendc/                               device translation units
  op_package/                                   CANN host OpDef / infer / tiling
  include/flashinfer_ascend/                      host/device 共用协议
  recipes/                                      构建 recipe
  launch/                                       host_shared launcher
```

`csrc/` 当前同时包含 device 与 host 文件，不能写成“只放可选胶水”。
当前 wheel 可以附带构建输入及配置选择的 op-package，并未实现独立 SoC provider 体系。
上游也不是“所有 kernel 都塞进主包”：其主包、shim、provider、cubin 分工见下一节。

### 2.4 与上游的同形分层对比

当前状态 `[P]`：四层职责可比较，但 generator、ABI 和覆盖范围不能互相替代。

下表是并排的调用分层图；向下箭头表示准备依赖，不表示每次 run 都重新编译。
上游以 FA2 batch decode 为代表，不外推到所有 CUDA/DSL 路线。

| 层次 ↓ | 上游 CUDA 代表链 | 本仓 Ascend 当前链 |
| --- | --- | --- |
| wrapper ↓ | 参数/layout/workspace → module.plan → plan vector；run 复用 | 公共 facade → Attention session/资源；CANN batch 仍为逐请求 composite `[P]` |
| generator ↓ | dtype/head_dim/variant → Jinja 源码 → compilation unit | Ascend C recipe/spec 可构建；Attention variant 只形成契约 spec，源码闭环缺失 `[P]/[G]` |
| JIT / AOT ↓ | URI + Ninja；主包或 provider 发现 → module；miss 依策略 JIT | `JitSpec.fingerprint` → store；两条登记的 lazy runner 自动发现 AOT、missing 受控 JIT `[V]`（限范围） |
| typed export / 调用 ABI ↓ | `TVM_FFI_DLL_EXPORT_TYPED_FUNC` → `tvm_ffi.load_module` → `.plan/.run` | CANN host codegen → `ctypes.CDLL` → 变参 Tensor adapter → ACLNN 两段；不是同一 typed export 实现 `[P]` |
| 复用执行 | module + planner offsets + workspace | CANN runner 或已解析 package runner；生产 Ascend C 当前为同步 eager |

```text
本仓公共 CANN 主线：wrapper → 已绑定 CANN runner → FIA / aclnn
本仓生产 Ascend C：wrapper → 冻结 recipe/spec → lazy AOT/JIT → ACLNN adapter
本仓 Attention 扩展：wrapper ──缺公共 batch 注入──> variant spec
                                      └─缺 compilation unit──> build
```

上游锚点：`flashinfer-ori: flashinfer/decode.py`、
`flashinfer-ori: flashinfer/jit/attention/modules.py`、
`flashinfer-ori: flashinfer/jit/core.py`、
`flashinfer-ori: csrc/batch_decode_jit_binding.cu`。
本仓锚点：`flashinfer/attention/public_batch.py`、`flashinfer/jit/attention/modules.py`、
`flashinfer/jit/build.py`、`flashinfer/jit/runtime.py`、`flashinfer/jit/tensor_adapter.py`。

| 差异性质 | 具体边界 | 处置与代价 |
| --- | --- | --- |
| **必然差异** | CUDA/NVCC/cubin 与 Ascend C/bisheng/NPU ELF/CANN package | 重建本平台编译与装载链；仍须保留完整性、ABI、错误和生命周期契约。 |
| **必然差异** | CUDA Graph 与 aclgraph 的参数固化/更新协议 | 用 ACL task update 等价处理已声明动态项，不能照搬 Python replan。 |
| **有意不对标** | CUDA runtime 模拟、二进制兼容、PDL、NVIDIA 专有通信实现 | 对外说明迁移差异；通用通信语义另立缺口。 |
| **有意不对标（实现形式）** | 不照搬 TVM-FFI、URI/Ninja、shim/provider 的具体组织 | 采用 recipe、内容寻址、ctypes/ACLNN；必须承担 adapter 与交付治理成本。 |
| **尚未补齐** | Attention generator 到 compilation unit、公共 batch 定制扩展 | 需源码生成与执行契约闭环；已有 FA 独立 eager 路线不能抵销。 |
| **尚未补齐** | fused batch、batch replay、异步 adapter、通用 scalar/attribute ABI | 需实现和独立验收，不能用硬件差异解释。 |
| **尚未补齐** | 多 package identity、运行 CANN 兼容校验、安装后生产验收 | 自动发现已接通，部署治理仍不能宣布完成。 |

实现形式不对标不代表功能等价已经达成，也不预先否决未来 provider 拆包。
上游通用 torch.library helper 在该基线是 identity decorator，但部分 op 直接真实注册。
本仓有限入口真实注册不等于全库 compile 或 graph 能力领先。
完整依据见 [分层报告](reports/call-stack-layering.md) §2–4 和
[上游机制基准](reports/competitor-survey.md) §3–5。

## 3. 对外 API 层

当前状态 `[P]`：已有九个能力域的接口与 runner，逐参数语义、公开扩展和图能力仍不完整。

### 3.1 命名对标规则

当前状态 `[P]`：同名 facade 已有，兼容性必须落实为逐入口的支持范围。

| 能力域 | 已核实 op 数 | 当前边界提示 |
| --- | ---: | --- |
| 标准 attention | 8 | CANN 主线；batch 图与自定义 generator 未闭环。 |
| 采样 | 10 | 有 runner；随机性与全部参数等价仍需按 workload 验收。 |
| RoPE | 6 | Llama 3.1 缩放变体未补齐。 |
| GEMM | 5 | 接口面计数沿既定口径；生产融合算子另按 descriptor 记证据。 |
| 归一化 | 5 | CANN 与有限 Triton 路线，compile/graph 不能整体外推。 |
| 激活 | 3 | 已有前向接口；不承诺训练 backward。 |
| MoE | 3 | 计算入口不等于 EP 通信分发。 |
| cascade | 2 | merge 接口不等于完整多层 wrapper。 |
| page | 1 | KV append 不等于 page allocator 或 MLA KV 管理。 |

**8 个域为空或仅有壳**：MLA、稀疏 attention、通信、量化算子、
GDN-KDA-Mamba、训练辅助、trace、autotuner。
量化的 `QuantSpec`、内部协议 trace、tuning 子包均不能抵作相应公开算子域已实现。
细分接口以 [接口面报告](reports/api-surface-by-category.md) §0–3 为准；历史计数见附录 A 裁决。

目标设计要求名称、默认值、shape、layout、mutation、异常语义都进入兼容矩阵。
NVIDIA backend 名称不能静默解释为所有参数等价的 NPU 实现。
显式不支持的配置须在执行或昂贵分配之前拒绝。
旧 parity 账本是追踪工具，不是当前全域兼容证明。

### 3.2 backend 与 kernel_id

当前状态 `[V]`（限定调度契约）：支持显式 backend 过滤、kernel_id 锁定与可解释拒绝。

- `auto` 在已登记且获准的候选中选择，不表示全库任意 backend 都存在。
- 显式 backend 缩小候选集，仍做组内过滤和排序；不可用时不静默换 backend。
- 显式 `kernel_id` 锁定一个实现；仍受 capability、target 和执行 ABI 检查。
- REFERENCE 是 correctness/debug 路线，生产自动准入受限制。
- custom 参与 auto 需 opt-in；explicit 路由不是免除能力检查。

不同 API 的公开参数签名仍以真实 facade 为准。
不能把一段带 `backend/kernel_id` 的设计伪码当成所有公共函数的统一签名。
`backend=None` 才读取全局默认策略的入口语义见 `runtime/dispatch.py`。

### 3.3 plan / run 生命周期

当前状态 `[P]`：plan/run 有真实状态契约，但 batch run 仍有逐请求工作。

```text
目标：plan → 选型、资源规划、预热 → 保存执行身份
      run  → 校验并消费已准备状态
现状：batch plan 保存 metadata/decision/workspace contract
      batch run 仍 gather/slice → FIA → concat/out copy
```

plan 失败不能发布半成品状态；固定资源路径先校验容量与身份再 copy。
输入超出计划范围须拒绝或显式重建 plan，不能静默替换已捕获 kernel。
函数式入口可能在首次调用做延迟解析；不应统称所有编译发生在 `plan()`。
运行时数据内容不自动属于可重规划项，尤其不能借 shape 契约代替数据依赖 tiling。

证据：`flashinfer/attention/planner.py`、`workspace.py`、`public_batch.py`；
[分层报告](reports/call-stack-layering.md) §2.1。

### 3.4 图模式支持（迁移入口）

当前状态 `[P]`：本小节的原机制移至独立 §3A；当前只有一条窄 single-decode 公共闭环。

原 §3.4.1–§3.4.3 的路线与能力声明见 §3A.1、§3A.4。
原 T9/T10/T11 内容分别并入 §3A.2–§3A.5。
原 §3.4.7 的覆盖边界更新为 §3A.6；主节 §4–§16 不改编号。

## 3A. 图模式：aclgraph 资源、捕获与参数更新（新增独立节）

当前状态 `[P]`：single-decode 有限定真机闭环 `[V]`，覆盖仍为 1/81；生产自研与 batch 入图为 `[G]`。

### 3A.1 路线与责任边界

当前状态 `[P]`：采用 aclgraph capture/replay，未实现旧设想中的 GE 每步 host 回调。

设备图 replay 不重新执行 Python forward、plan、dispatch 或任意 host tiling。
需要变化的 host 参数在 replay **之前**通过声明的 task update 协议更新。
`torch.compile(fullgraph=True)` 的通过不能作为 aclgraph 可捕获证据。

| 动作 | 调用方 | 本仓 |
| --- | --- | --- |
| 创建图、选择区域、FULL/PIECEWISE、分桶 | 负责 | 不提供模型级图管理器。 |
| 固定输入存储、原位更新内容并建立 producer stream 顺序 | 负责 | 校验并持有输入绑定。 |
| plan、workspace/output/LSE 准备 | 捕获外调用；可提供 buffer | 查询/分配或校验资源，冻结身份。 |
| warmup | 捕获外调用并准备其它节点 | 预热本 wrapper 的已准备路径。 |
| capture | 负责外层 `torch.npu.graph` | `recording(graph)` 记录本节点参数和资源。 |
| 参数更新与 replay | 每次显式调用 | `before_replay` 更新；`replay()` 校验凭证后委托图重放。 |
| 错误恢复与释放 | 同步后弃图/重建 | 拒绝非法状态，不自动修补模型图。 |

“本仓不管理图生命周期”与公共 wrapper 代理 `graph.replay()` 不矛盾。
图对象及捕获区域仍归调用方；wrapper 只约束自己持有的执行资源。

### 3A.2 三段机制

当前状态 `[V]`（限定 single-decode）：准备、捕获、更新后重放已连通，不能扩大为通用 batch。

```text
捕获外
  固定 Q/K/V → plan → 最大 workspace / out / LSE / BSND views
             → 冻结 kernel 与存储身份 → warmup
捕获内（调用方拥有图）
  torch.npu.graph(..., auto_dispatch_capture=False)
    → wrapper.recording(graph) → 一次 wrapper.run()
每次 replay 前（捕获外）
  原位更新输入 → before_replay(kv_seq_len, stream=update_stream)
               → task update + event → wrapper.replay()
```

长度变化不替换 Q/K/V 存储；有效 KV 长度与固定容量分开。
输入 shape、stride、dtype、device、地址和 kernel 身份变化需新 wrapper 与新图。
输出和 LSE 会覆盖原 buffer，保留历史结果由调用方另行复制。
同一 wrapper、图及其资源须串行使用，活到最后一次异步执行完成。

### 3A.3 公共资源接口

当前状态 `[V]`（限定路径）：`flashinfer.decode.SingleDecodeWithKVCacheWrapper` 已交付。

| 接口 | 当前动作 | 边界 |
| --- | --- | --- |
| 构造 `workspace_buffer=None, kv_layout="NHD"` | 建立独立资源生命周期 | 不创建图，不自动完成 plan。 |
| `plan(q,k,v,kv_seq_len=...,out=...,lse=...)` | 校验范围、查询最大 workspace、准备固定资源与 views | 捕获外执行，首次成功冻结身份。 |
| `warmup(iterations=3)` | 执行并预热准备好的 run | 不是全库 JIT/AOT/autotune 预热器。 |
| `run(...,return_lse=False)` | 复用输入、FIA `.out`、workspace 与 LSE 转换 | 不重新 dispatch，不扩容量。 |
| `recording(graph)` | 在 active capture 中记录一次本节点 run | 单图单 wrapper，不代替外层 capture。 |
| `before_replay(kv_seq_len,stream=...)` | 校验身份、更新长度、建立 event/stream 顺序 | 每次重放均须成功调用。 |
| `replay()` | 消费一次更新凭证后委托图 replay | 无新凭证拒绝。 |
| `workspace_buffer/output/lse` | 暴露已持有资源 | 不允许 capture 后 resize/rebind。 |
| `kernel_id/kv_capacity` | 暴露冻结选择与容量 | 不扩展支持范围。 |

调用范例见 [single-decode 示例](../examples/single_decode_graph.py)。
底层公共扩展还有 `flashinfer.npu.graph_requirements`、
`maximum_workspace_size` 与 `ReplayParameterUpdates`。
`maximum_workspace_size` 委托已有查询方法；不能据此虚构 ragged 的独立查询接口。

原无状态 `single_decode_with_kv_cache` 不是新增的固定图资源入口。
新 wrapper 补上了旧报告指出的公共资源缺口；它没有提升其它 descriptor。

### 3A.4 GraphCapability 与证据门禁

当前状态 `[P]`：声明/查询/录制门禁已有实现，覆盖扩大必须另取真实证据。

| 枚举 | 含义 | 当前范围 |
| --- | --- | --- |
| `CAPTURE_SAFE` | 捕获后重放前无需改写 host 参数，仍需稳定资源与验证范围 | 0。 |
| `NEEDS_PARAM_UPDATE` | 必须按声明更新参数后 replay | 唯一 single-decode 条目。 |
| `NOT_CAPTURABLE` | 未获验证或当前执行不支持入图 | 两条生产自研及 batch 均维持此声明。 |

`graph_param_updates` 是原生命名参数列表。
`GraphReplaySpec` 绑定逻辑字段映射、必需/固定参数、Tensor metadata、范围、测试 selector 和环境。
当前逻辑映射为 `kv_seq_lengths → actual_seq_lengths_kv`。
图能力不参与普通 eager 过滤排序；eager 的宽范围也不能扩大 graph 范围。

schema 可表达声明，但 NPU 查询/录制会拒绝缺有效 replay 证据的非默认声明。
`validated_environment` 是证据记录，不是运行版本兼容探测器。
新增算子默认 `NOT_CAPTURABLE`；取得对应执行路径证据后才提升。

### 3A.5 固定资源与 replay update

当前状态 `[P]`：single 闭环与 batch T9/T10 资源机制分别存在，不能拼成已支持 batch replay。

batch `use_cuda_graph=True` 要求固定 int32 metadata、容量和不重叠资源。
plan 原位 `copy_`，冻结 batch/backend/Q 长度、block-table 宽度、kernel 与 descriptor 身份。
`use_max_workspace=True` 是额外选择，不等于单开图开关就已绑定最大 workspace。
FIA v1 最大查询可供特定 native batch 路径使用；共享 workspace 取顺序请求需求的最大值。
容量不足在 plan 发布前拒绝，不自动扩容。

single recorder 保存 operation、kwargs、Tensor 引用、task handle 与 `ExternalEvent`。
更新前先检查所有参数及冻结绑定，再进入设备更新；不是通用 tiling binary 改写器。

```text
capture：event.wait → event.reset → task group begin → operation → group end
update ：update stream 等待当前 replay stream
         → task_update_begin → operation(new kwargs) → task_update_end
         → event.record
caller ：同一 replay stream 下 wrapper.replay()
```

漏更新、重复未消费 update、换绑资源及错误 stream 均属于生命周期错误。
设备更新失败后须弃用对应 wrapper 和图；不能继续重放旧任务。
直接调用原生 `graph.replay()` 会绕过 wrapper 凭证检查，不受本接口保护。
调用方还必须传入实际捕获图，不得 reset/re-capture 已绑定图。
当前无法靠公开 current-graph ID 自动核对所有调用方行为。

### 3A.6 覆盖面与缺口

当前状态 `[P]`：**1/81 可入图**是 descriptor 声明口径，不是算子通过率。

| 唯一已验证路线 | 精确范围 |
| --- | --- |
| kernel | `cann/attention/single_decode_fia_v1`，`NEEDS_PARAM_UPDATE`。 |
| 公共输入 | NHD；Q `(8,64)`，K/V `(1024,2,64)`，连续 NPU Tensor。 |
| 原生 views | BSND；Q `(1,1,8,64)`，K/V `(1,1024,2,64)`。 |
| dtype / 选项 | 同为 FP16 或 BF16；heads=8/2、D=64、scale=0.125，无 causal/window/custom mask。 |
| 动态项 | 仅逻辑 `kv_seq_len`，范围 1..1024，固定容量不变。 |
| 输出 | 固定 output；公共 FP32 LSE 为 log2 口径，内部 FIA natural-log LSE 做固定转换。 |
| 捕获形态 | 一图一 wrapper、recording 内一次 run、串行使用、`auto_dispatch_capture=False`。 |

batch 路径仍逐请求 gather/slice/FIA、创建临时与输出 Tensor 并拼接。
固定 CSR/mask/block table 不会自动更新已捕获 Python tuple 或切片边界。
`_batch_run()` 在 recording 下明确拒绝；不能靠删除 guard 完成支持。
外部 provider 的 batch 图资源绑定也未接通。

后续 batch 交付须同时提供固定任务规模、全部中间资源、逐任务更新协议、容量门禁和数值反例。
生产 `matmul_bias_silu`、`flash_attention` 仍未入图；同步 adapter 是需改造的一处边界。
完整模型、MoE piecewise、多 stream 重入、多卡图、其它容量/heads/mask 均未验证。

证据：[图模式交付](reports/graph-mode-delivery.md) §1、§3–§9、§11；
`flashinfer/attention/single_decode_graph.py`、`runtime/_graph_params.py`；
`tests/test_single_decode_graph_real.py`、`test_graph_capability.py`、`test_attention_graph_plan.py`。

## 4. 硬件感知与环境探测

当前状态 `[P]`：有缓存的硬件探测，逐 Tensor 设备选择和完整软件兼容门禁尚未建立。

### 4.1 HardwareSignature（硬件签名）

当前状态 `[V]`（限定 host 契约）：真实字段为 SoC、AI Core 数和 UB 容量。

```python
@dataclass(frozen=True)
class HardwareSignature:
    soc_model: str = "unknown"
    aicore_count: int = 0
    ub_size_bytes: int = 0
```

`rank_num` 不是当前字段；可见设备数不能直接当作编译特化维度。
`ub_size_bytes=0` 表示未知，不能用 HBM 或 L2 容量替代。
SoC 系列归一化可服务过滤，但 PR/DT 或其它实际目标兼容性仍需独立验证。

### 4.2 设备可见性与目标

当前状态 `[P]`：环境可见性已有，异构多设备的身份与路由仍有限。

当前 `current_hardware()` 使用 device 0 构建进程缓存。
它不等同于根据每次 Tensor device 选择最佳 AOT module。
`ASCEND_RT_VISIBLE_DEVICES` 影响可见设备，不进入上述签名。
8 张卡可见不等于全部设备、并发或多卡通信均做过验证。

目标设计应把构建 target、当前执行 device、可见设备集合分开。
跨 SoC 的编译或名称映射只说明产物目标，不证明目标设备数值执行。

### 4.3 探测怎么做

当前状态 `[P]`：torch_npu 与可选 ACL 查询已实现，没有原稿声称的通用 npu-smi 兜底链。

首次访问通过 `torch.npu` 读取设备名及属性，缓存只读硬件对象。
UB 通过可选 `acl.rt.get_device_info` 查询；失败保留未知值。
缺依赖、设备不可用或探测失败不应伪造实机能力。
unknown 快照服务 host 契约测试，不能作为 NPU 算子可运行证据。

### 4.4 环境一致性校验

当前状态 `[G]`：当前 CANN runtime 与构建版本的兼容校验尚未接通。

构建环境、工具链和依赖身份会进入 JitSpec/recipe。
这只能约束产物是怎样构建的，不能证明当前运行库兼容。
不能再写“首次使用已按 support matrix 拒绝所有未验证软件栈”。

目标设计应声明经过验证的 runtime/build 组合及拒绝策略。
兼容检查应先于执行并保留清晰原因，不能只比字符串后假定 ABI 等价。
同时要定义 aarch64 host、不同 driver/CANN 和逐设备 target 的验证矩阵。

证据：`flashinfer/runtime/hardware.py`、`tests/test_hardware.py`；
[AOT 发现说明](aot-runtime-discovery.md)“范围与已知缺口”。

## 5. Dispatcher：多 backend 与 kernel 选择

当前状态 `[V]`（限定选择契约）：过滤、来源栈与组内选择、诊断和缓存已有证据，产物可用性由 runner 负责。

### 5.1 op、backend 与 kernel 的关系

当前状态 `[V]`（限定 schema/dispatch）：语义、来源和具体实现分别建模。

| 对象 | 回答的问题 | 当前例子 |
| --- | --- | --- |
| op | 做什么 | `attention.single_decode`。 |
| backend / stack | 来自哪类实现 | CANN、Ascend C、Triton、custom、external、reference。 |
| kernel_id | 选中哪条执行契约 | `cann/attention/single_decode_fia_v1`。 |
| compile_mode | 该候选如何声明交付 | AOT/JIT metadata，不代表文件已经存在。 |

一个 op 可以有多个候选，同一 backend 可以有多个 variant。
现有 dispatcher 先逐实现过滤，以来源栈组织报告，再按相同排序键选栈与栈内实现。
它不是旧稿所述“根本不存在来源栈选择”的单层表示。
两级展示不增加无实测依据的新排名规则。

### 5.2 backend 类型

当前状态 `[P]`：内置来源有实际 runner，external/custom 是扩展契约，不能据入口推断已有生产集成。

| backend | 当前含义 | 默认与限制 |
| --- | --- | --- |
| `cann` / ACLNN 类别 | 预装算子或 composite | 主要 NPU 执行来源。 |
| `ascendc` | 本仓 recipe 驱动的自研 kernel | 两条生产登记；自动 AOT/JIT 见 §7。 |
| `triton` | triton-ascend 实现 | 生产 norm 走原生 `@triton.jit`，非全库 AOT。 |
| `external` | 本仓认可的外部实现来源 | 注册与命名契约不等于特定外部库已经验收。 |
| `custom` | 用户显式注册实现 | `custom/` 前缀，auto 需 opt-in。 |
| `reference` | host 数值参考 | 不作为 NPU 加速覆盖。 |

`register_external_kernel` 与 `register_custom_kernel` 在内置 registry 冻结后仍可新增。
同名覆盖拒绝；新增改变 revision，使后续决策不能沿用旧候选集缓存。
实现技术与信任来源正交：用户 Ascend C 实现仍可属于 custom。

### 5.3 选择算法

当前状态 `[V]`（限定测试范围）：默认顺序是 `external > cann > aot > jit > reference`。

```text
signature + hardware + policy
  → 显式 kernel_id：锁定后检查 capability/target
  → 否则：按 backend 与 custom/reference/JIT 策略过滤
  → 逐实现检查 dtype/shape/layout/特化参数和 target
  → 查询有效 tuning winner；winner 必须仍在可用候选中
  → 无有效 winner：按配置的 tier、priority、kernel_id 排序
  → 选来源栈及其首选实现；返回 decision、reason、候选报告
```

`tier=aot` 不扫描磁盘，不验证包是否存在，也不表示实测最快。
顺序可配置；当前 CANN 优先于自研 AOT，原稿反过来的默认顺序已更正。
能力过滤与 loader 的 ABI/完整性校验属于不同门禁。
`fallback_chain` 是决策信息，不是 launch 失败后的自动重试引擎。
运行失败不能自行改走 CANN/reference，掩盖错误或重复有 mutation 的调用。

### 5.3.1 显式指定 backend 后的组内选择

当前状态 `[V]`（限定调度契约）：显式 backend 只缩候选集，不改变排序算法。

组内多个 variant 先各自通过 capability 与 target，再使用有效 tuning 或确定性顺序。
同 tier 下 `priority` 高者优先，最终按 `kernel_id` 字典序稳定裁决。
全部被拒绝时返回原因，不跨 backend 静默降级。
显式 kernel_id 的锁定规则优先，但执行前仍不能跳过 loader 或 adapter 校验。

### 5.3.2 dispatch 决策缓存

当前状态 `[V]`（限定 host 测试）：缓存按 registry 实例隔离，并包含 revision 与 tuning token。

```text
决策身份包含：
  signature、hardware、policy、kernel_id
  allow_reference、allow_custom、JIT 配置、dispatch_order
  registry.revision、tuning_cache_token
缓存外层：registry 实例隔离
```

这是进程内 decision 缓存，不是持有已加载 binary 的 JIT loader LRU。
当前实现为按 registry 管理的映射；不再将其描述为已实现容量上限的 LRU。
注册新 custom/external、tuning 变更或策略变化必须反映到下一次选择。
原“候选不可变所以永不失效”与当前扩展入口不符。

| 缓存 | 索引与内容 | 失效边界 |
| --- | --- | --- |
| dispatch | 选择输入 → decision | registry/config/tuning token 变化。 |
| JIT store | `JitSpec.fingerprint` → verified generation | 已声明构建内容变化形成新身份。 |
| loader | 已校验产物 → module | 进程内加载生命周期，不能代替磁盘验证。 |
| lazy runner | 冻结 descriptor/spec → 已绑定 callable | 成功绑定后复用；换包需符合进程身份规则。 |

### 5.4 Tuning 记录与确定性启发式

当前状态 `[P]`：查询与采集设施存在，发布 tuning 库为空，不能声称默认选择来自实测冠军。

有效本地记录优先于发布记录；所有记录仍需匹配硬件和候选身份。
命中记录不能让不支持输入的 kernel 越过过滤。
没有记录时使用确定性启发式，并记录原因。

旧稿的 exact → bucket → 相邻桶借用 → heuristic 是目标查询设计。
当前不可将相邻桶借用和桶内冠军稳定性描述为全库已交付行为。
生产 CANN/Triton 的源码及软件栈身份治理仍不足，不能假定升级后旧 winner 总会失效。
完整采集、存储和治理边界见 §9。

### 5.5 选择流程图

当前状态 `[P]`：选择与产物解析均已实现于各自层，但只对登记的 runner 相连。

```text
输入 → policy/capability/target 过滤 → tuning 或 heuristic → decision
  ↓ 已绑定 runner
  ├─ CANN/reference/Triton：进入各自执行路径
  └─ AotKernelRunner：首次 resolve
       ├─ AOT hit → verify/load/adapter → execute
       ├─ missing → 按 policy 决定 JIT 或拒绝
       └─ corrupt/load error → 拒绝
```

证据：`flashinfer/runtime/dispatch.py`、`registry.py`；
`tests/test_two_level_dispatch.py`、`test_dispatch_ordering.py`、`test_dispatch_cache.py`。
调度通过证明选择契约，不证明每条候选在所有输入下数值正确。

## 6. 算子实现层

当前状态 `[P]`：CANN 为主，Triton 有有限入口，Ascend C 已有两条生产子集。

### 6.1 CANN backend（`flashinfer/cann`）

当前状态 `[V]`（限定既有用例）：CANN runner 已用于多个能力域，性能与全部组合未验证。

CANN backend 封装参数、layout、stream 与 aclnn/composite 调用。
descriptor 必须准确声明能力；厂商有某 kernel 不等于本仓已经接入。
REFERENCE runner 不能当作 CANN 的替代加速实现。
batch attention 当前整理为 dense KV 后执行 FIA，不能说是完整原生 paged fused scheduler。

### 6.2 Ascend C 算子

当前状态 `[V]`（限定 eager envelope）：两条生产实现可经公共入口执行，图与性能仍未验证。

| 路径 | 当前支持范围 | 不可外推 |
| --- | --- | --- |
| `matmul_bias_silu` native v2 | BF16 连续 ND；`silu(A@B+bias)`；M/N/K 为 16 的倍数且在 `[16,256]`，bias 为 `[N]` | 任意转置、广播、dtype、异步执行或 graph。 |
| `flash_attention` catlass v2 | FP16、batch=1、连续 NHD、D=128、heads 1..8；S/T 为 128/256/384/512，无 mask | BF16、GQA/MQA、paged/ragged、LSE、变长 batch 或 graph。 |

FA 的有效长度就是完整稠密 Tensor shape，不接收独立实际长度 Tensor。
两者证明 shape 驱动 host tiling，不证明读取 Tensor 内容的 tiling。
生产 runner、冻结身份与数值证据见 §10 及两份交付报告。

### 6.3 Triton 算子

当前状态 `[P]`：生产 `gemma_fused_add_rmsnorm` 有真机数值证据，统一 store 与生产路径并未归一。

生产入口直接使用原生 `@triton.jit`。
统一 Triton backend 另有 source/request 冻结、worker 编译、产物校验与恢复路线。
不能用生产 norm 的数值测试替代统一 store 的缓存测试。
也不能用统一 store 的禁编译策略宣称控制所有原生 Triton 编译。

### 6.4 kernel 注册（registry）

当前状态 `[P]`：声明与 runner 接线已存在，运行目录、生产 manifest、Attention provider 是不同集合。

`KernelDescriptor` 包含 op/backend、compile_mode、capability、source/artifact、ABI、图声明等身份。
不在文中维护一份容易漂移的缩略 dataclass 作为完整 schema。
真实字段与验证以 `flashinfer/runtime/schema.py` 为准。

内置 Python catalog 与随包生产 manifest 合入 runtime registry。
`registry.json` 不代表全库所有 runtime 条目；CLI 展示也不能代替 runtime inventory。
Attention 的 provider/variant 契约是另一条连接，独立 ACLNN FA 不等于已补齐该 provider。
注册只有在绑定了符合参数契约的 runner 后才构成可执行路径。

## 7. AOT 模式

当前状态 `[P]`：自动发现→命中→加载绑定 runner→受控 JIT 回落已接通；仅限已登记 Ascend C 路径。

### 7.1 AOT 编译粒度

当前状态 `[V]`（限定已交付 recipe）：产物身份是完整 JitSpec，不按每个运行 shape 单独打包。

逻辑粒度可写为 recipe × target × 编译期特化 × 构建/ABI 身份，最终归入 spec fingerprint。
同一源码可以因目标、dtype、静态选项或依赖变化生成多个产物。
v2 的 host_dynamic tiling 可在声明范围内处理不同 shape，不必逐 shape 生成 binary。

原稿“所有高性能维度都必须静态化，否则性能不可接受”没有测量依据，予以删除。
哪些维度静态化是实现与包体积之间的取舍，须由正确性和性能测量共同决定。
声明支持的运行范围不是 AOT 已覆盖的发布矩阵，两者应分开列。

### 7.2 构建、导出与分发

当前状态 `[P]`：复用 JIT 构建/store/export，主 wheel 可附产物；独立 provider 与通用 family 枚举未完成。

```text
发布作者：recipe/source/依赖 → resolve → freeze JitSpec
       → 真实构建 → verify → immutable generation
       → export_op_package → release → 主 wheel 或显式搜索目录
应用：冻结 descriptor/spec → lazy runner → 自动发现已导出 release
```

通用 AOT CLI 仍有 `matmul_probe` family 限制。
生产路径使用 `tools/build_matmul_bias_silu.py`、`tools/build_flash_attention.py`。
它们复用 `export_op_package()`，没有另建 loader 或另一套缓存。
默认 pure wheel 包含 Python 与必要构建输入，不自动表示其中已带全部 AOT binary。
配置构建或 PREBUILT 复制可将产物放入同一主 wheel。

上游主包、shim、架构 provider、cubin 分工是可借鉴的交付职责，不要求照搬包名。
当前尚无本仓独立 SoC provider 发现与版本生态；多 target 附包也不等于异构设备路由。
真正构建并安装生产 AOT wheel 后的端到端运行，仍需独立验收。

### 7.3 op-package 结构与 manifest

当前状态 `[V]`（限定真实导出路径）：以下是生成/安装布局，不宣称源码树默认带有这些 binary。

```text
flashinfer/data/op_packages/<spec-sha>/
  manifest.json
  ascendc-recipe.json
  op-package.json
  op-package.run
  packages/vendors/<vendor>/
    op_api/lib/libcust_opapi.so
    op_impl/ai_core/tbe/kernel/<compute-unit>/...
      NPU ELF 与 CANN JSON
```

| 产物 | 职责 |
| --- | --- |
| `manifest.json` | 完整 spec、构建 record、文件 hash/size inventory。 |
| `ascendc-recipe.json` | recipe、输入快照、工具链与命令 receipt。 |
| `op-package.json` | vendor、API 库、compute unit、recipe 等装载 metadata。 |
| `op-package.run` | 携带的 package 产物；loader 不执行安装器。 |
| host `.so` | ACLNN GetWorkspaceSize 与 launch 导出。 |
| NPU ELF / JSON | 设备代码与 CANN metadata，受同一完整性链约束。 |

不再使用旧稿的简化 manifest JSON 代替真实 inventory。
摘要校验说明内容一致性，不等于发布者签名认证或运行库兼容认证。

### 7.4 自动发现、匹配与 runner 绑定

当前状态 `[V]`（限定登记路径）：首调自动解析已实现，dispatch 排序本身仍不搜索磁盘。

作者通过 `bind_aot_kernel(descriptor,spec,runner_factory,jit_backend_factory)` 绑定 lazy runner。
绑定时不读文件、不发现编译器或设备；运行期只消费发布时冻结的可信契约。
它不从任意搜索到的 binary 反向推断应信任哪个 descriptor。

```text
公共 API → dispatch → descriptor.runner
  → AotPackageIndex.find
     target_soc + 完整 spec fingerprint + descriptor/ABI 绑定
  → 完整校验 → load_op_package → runner_factory
  → 保存 module/callable/provenance → 执行
```

默认搜索安装位置 `flashinfer/data/op_packages/`。
`FLASHINFER_AOT_PATHS` 按 `os.pathsep` 追加根目录或单个 release；默认目录始终参与。
索引只枚举一层，进程缓存目录清单和已验证 manifest metadata。
目录/manifest 的 stat 身份变化会触发刷新，也可显式 `invalidate_op_package_index()`。

索引不是完整校验替代品：命中候选在 find/load 时继续验证 inventory 与 package。
首次绑定成功后复用 module 与 runner，不在每次执行中扫描或 hash 全包。
刷新索引不卸载已绑定 module；磁盘替换不自动改变旧 runner 的来源。
`runner.provenance` 的 source、directory、spec_fingerprint 才说明实际使用 AOT 还是 JIT。

### 7.5 四类错误与受控回落

当前状态 `[V]`（限定现有测试）：四类分流已有实现，corrupt 即使 JIT 开启也拒绝。

| 分类 | 当前行为 | JIT 回落 |
| --- | --- | --- |
| missing | `AotMissingError`，携带搜索路径、目标、spec 与 policy | 仅 `ENABLED` 且登记 backend factory 才允许。 |
| corrupt | `AotCorruptError`，保留 digest/size/manifest 等失败项 | **绝不回落**，不能当普通 miss。 |
| incompatible | 保存各候选 `AotIncompatibleCandidate` 原因，继续尝试其它候选 | 全部不兼容最终为带原因的 missing，再按 missing policy。 |
| load error | `AotLoadError`；dlopen、缺符号、adapter、package identity 等失败保留 cause | **绝不回落**。 |

执行阶段 CANN/launch 错误保留原异常和 traceback，不触发其它 backend 重试。
`DISABLED` 与 `CACHE_ONLY` 在本 lazy AOT 入口遇 missing 都拒绝。
即使本地 JIT store 有缓存，也不能在该入口冒充随包 AOT 命中。
这与直接 `build_and_load` 的 cache-only 行为不同，见 §8.3。

### 7.6 装载边界与待补项

当前状态 `[P]`：loader 已实跑，多 package identity 与当前 CANN 兼容校验仍为明确缺口。

`load_op_package` 校验后激活 vendor 路径，加载 `libcust_opapi.so` 并设置 ABI。
CANN 含点路径问题通过只读目录 FD 与 `/proc/<pid>/fd/<fd>` 别名处理。
FD 随活动 module 保活，验证仍针对原 generation；这引入 Linux `/proc` 依赖。
激活要求在首次 custom-op 执行之前，不再沿用“必须早于 ACL 初始化”的旧说法。

单进程第二个不同 package identity 明确拒绝，换包须新进程。
这意味着两条生产算子分别跑通，不等于可以在同一模型进程同时使用两包。
当前 loader 不比较运行 CANN 版本与构建版本的兼容关系。
真实安装 wheel、其它 SoC/版本、capture 中首次绑定、已激活 module 跨 fork 均未验证。

证据：[AOT 运行期发现](aot-runtime-discovery.md)；
`flashinfer/jit/discovery.py`、`runtime.py`、`op_package.py`；
`tests/test_aot_runtime.py`、`test_aot_discovery_index.py`、`test_aot_loader_paths.py`；
生产验收见 `tests/test_matmul_bias_silu_real.py`、`test_flash_attention_real.py`。

## 8. JIT 模式与 kernel 缓存（核心设计）

当前状态 `[V]`（限定构建/缓存链）：内容寻址、并发去重、原子发布与加载有证据，全库统一治理仍为 `[P]`。

### 8.1 真实 cache 身份

当前状态 `[V]`：真实身份是 `JitSpec.fingerprint`，不是顶层 legacy `build_cache_key`。

```text
JitSpec.to_dict()
  → canonical JSON
  → SHA-256
  → JitSpec.fingerprint
```

JitSpec 绑定 generator、target、environment、specialization、inputs、source digest、
flags、entry、output path 等声明；Ascend C 再绑定 recipe、构建文件和依赖闭包。
完整字段与 canonical 序列化以 `flashinfer/jit/core.py` 为准。
顶层 `build_cache_key` 是构建链之外的 legacy 死代码，非测试调用为零。
`toolchain_fingerprint` 也是 legacy helper；二者均不参与真实构建身份。
`tests/test_jit_build.py` 将 legacy helper 替换为 forbidden，检查真实路径仍可工作。

| 身份层 | 用途 | 不能替代什么 |
| --- | --- | --- |
| kernel_id | 稳定路由名 | 不能表示源码或 ABI 内容未变。 |
| descriptor fingerprint | 能力与 ABI 契约 | 不能替代完整源与工具链闭包。 |
| recipe fingerprint | schema、源摘要、依赖、参数、host/device 约束 | 不能证明实际 C++ 数学正确性。 |
| JitSpec fingerprint | 冻结的完整构建规格 | 不自动追踪任意未声明 include 或运行库兼容性。 |
| 文件 inventory | 已发布 bytes 的 hash/size | 不能替代 spec 对目标与调用契约的绑定。 |

输入身份完整性由声明者承担。
改源码/头/依赖/flags/descriptor 后必须更新摘要、重新 freeze spec。
旧 spec 的 cache hit 验证已发布 bytes；不会因为原工作树变化自动重写身份。
miss 时实际源与声明 digest 不符会拒绝编译，不能拿旧 spec 构建新内容。

动态 shape 是否进入编译特化取决于 recipe，不存在全库统一的“batch/seq 默认 -1”保证。
shape 驱动 tiling 的范围仍须由 shape_constraints、host 和 dispatch 共同限定。
构建环境进入身份也不代表运行 CANN 兼容检查已经完成。

### 8.2 缓存目录与不可变 generation

当前状态 `[V]`（限定 POSIX store）：稳定 key 指向已校验 generation，锁位于发布指针之外。

```text
<cache>/jit/
  <sha前两位>/<完整sha> → .objects/<sha>-<generation>
  .objects/<sha>-<generation>/
    manifest.json
    build.log 与实际产物
  .locks/
  .staging/
  user/<namespace>/             显式选择 namespace 时的独立 store
```

这是运行时布局模式；具体 artifact 路径由 spec 决定，不固定都叫 `kernel.so`。
已发布 generation 不原地覆盖，旧读者持有的已验证目录仍可使用。
构造 store 不触发构建；staging、cache 与只读安装目录分开。
namespace 是显式 store 选项，不是注册 custom 后自动建立的完整安全沙箱。

### 8.3 冷路径、热路径与并发去重

当前状态 `[V]`（限定已有测试）：per-spec flock、锁内复查和发布协议防止同身份首用重复构建。

```text
冷路径：declare spec
  → store.lookup + manifest/inventory 验证
  ├─ hit：verified loader → module
  └─ miss：检查 compilation policy
       → per-spec flock
       → 锁内再次 lookup
       → private staging 中 backend.compile
       → 校验产物与 record
       → 写完整 manifest / 持久化 generation
       → 原子替换 key symlink
       → verified loader → module/runner
热路径：调用方持有 module/runner → 执行
```

发布完整性发生在读者可见之前，不能采用“先发布目录，后写 manifest”的旧流程。
失败保留构建诊断，不将半成品作为 hit，不覆盖旧已发布 generation。
并发去重以 spec 身份为单位；不同 spec 的编译资源预算另行治理。

loader 有进程内 LRU，但重复调用 `build_and_load` 仍涉及磁盘验证。
真正热执行应持有 module/runner，不能声称所有二次 `build_and_load` 都是 O(1) 内存命中。
构建完成不代表执行必然异步；当前生产 Tensor adapter 仍有同步。

| 入口 | DISABLED / CACHE_ONLY 的有效缓存 | miss 行为 |
| --- | --- | --- |
| 直接 `build_and_load` | 可消费有效本地 JIT store | 拒绝编译。 |
| `AotKernelRunner` | 先要求找到随包/搜索目录的 AOT | missing 直接拒绝，不查询 JIT cache 兜底。 |

lazy runner 仅在 missing 且 ENABLED 时调用 backend factory。
因此 AOT hit 不需要发现工具链或解析 catlass checkout。
这条已接通的生产路径仍不能推广为每个无 runner descriptor 都会自动编译。

### 8.4 失效、恢复与治理

当前状态 `[P]`：内容失效与损坏恢复有机制，批量预热、磁盘配额和全库 capture 门禁尚不完整。

已声明内容变化产生新 fingerprint；旧产物不会以新身份复用。
直接 JIT store 的损坏记录可视作 miss，并在策略允许时构建新 generation。
AOT release 损坏则按 §7.5 明确拒绝，两种入口不得混写。
旧 generation 清理需考虑活动读者；不能把磁盘清理当作已加载 module 卸载。

Ascend C 用 CMake/bisheng；native 与 catlass 共用 backend 和验证链。
统一 Triton 有自己的冻结 request 与恢复 ABI；生产原生 Triton 还保留独立编译缓存。
Attention variant spec 尚不是可构建 compilation unit，公共 batch 也拒绝对应注入扩展。

后续需按清单预热并限制编译资源，明确失败聚合和诊断。
全库 active capture 中首次 JIT、AOT 延迟绑定、autotune 的统一拒绝门禁仍缺失。
不能以“调用方可手动预热”替代生产入口保证。

证据：[架构评审](reports/ascend-architecture-review.md) §5；
`flashinfer/jit/core.py`、`cache.py`、`build.py`、`loading.py`；
`tests/test_jit_build.py` 及 Ascend C/Triton 真实编译测试。

## 9. Tuning 机制（进阶特性，后期启用）

当前状态 `[P]`：采集、存储、查询及默认排序已有机制，公开 autotuner 与发布性能库未完成。

本节保留原主标题以便交叉引用；“后期启用”指完整生产 tuning 体系。
不能再写“初版完全没有 tuning 查询”：dispatcher 已消费有效记录。
也不能从查询实现推出已有同口径性能结果。

### 9.1 三种 tuning 模式

当前状态 `[P]`：本地显式采集与查询存在，自动发布流水和在线探索未形成全库闭环。

| 模式 | 目标职责 | 当前边界 |
| --- | --- | --- |
| 发布期离线 tuning | 同环境比较候选，随版本交付可信 winner | 发布 tuning 库为空。 |
| 本地调优 | 对自有 workload 测量并持久化 | 有采集/store 设施；需真实 candidate 与正确性证据。 |
| 在线探索 | 显式 opt-in、受预算约束，服务热路径不测量 | 配置入口不能证明生产自动探索已接通。 |

原则保持不变：先 correctness，再测量；普通 dispatch 只查询，不即时计时竞争。
本地记录优先并不意味着不同环境、不同输入的两条记录可公平比较。

### 9.2 发布期离线 tuning 流水

当前状态 `[G]`：尚无覆盖选定生产矩阵的完整发布流水与数据。

目标流程如下，属于后续验收要求：

```text
冻结 workload / 设备 / 软件栈 / 候选身份
  → 准备输入与已加载 runner
  → correctness 对照
  → warmup 与受控采样
  → 记录分布、噪声、测量顺序、环境与失败
  → 桶内边界复核
  → 发布经过校验的记录
  → 运行期匹配与拒绝失效记录
```

不能只保存一个最快 kernel_id 而丢失环境、误差和统计依据。
一次测量结果不能宣称同桶全部 shape 都最优。
生产没有同口径数据时，heuristic 只是确定性默认策略。

### 9.3 存储与身份

当前状态 `[P]`：本地 SQLite 与发布 manifest 查询已存在，软件身份治理仍不完整。

当前存储实现见 `flashinfer/tuning/store.py`，不是原稿 JSON 目录示例的完整落地。
记录校验 descriptor/capability fingerprint、硬件和采样依据。
发布数据通过 manifest 定位；当前发布库为空。
本地变更通过 tuning token 影响 dispatch 缓存。

目标身份还需稳定绑定生产源码、编译栈与测量政策。
尤其 CANN 与原生 Triton 的升级不能只靠相同 shape 和 kernel_id 复用旧 winner。
环境身份缺口没有因 JitSpec 内容寻址而自动消失：两者是不同缓存。

### 9.4 在线探索与 capture

当前状态 `[G]`：全库在线探索与 active-capture profiling 门禁未交付。

未来若启用探索，应在显式准备阶段完成，并限制候选、采样量及资源占用。
捕获内只允许已准备且已验证的路径，不启动 JIT 或首次 profiling。
结果写本地库，不回写官方发布库；失败应保留确定性选择和诊断。
配置值可读取不等于这些执行行为已经成立。

### 9.5 失效与治理

当前状态 `[P]`：已有记录过滤和重载机制，完整运维 CLI/配额/发布门禁仍待补齐。

旧稿的 `tuning benchmark/validate/clear-cache --tuning` 命令序列不作为已交付 CLI 承诺。
当前可用命令必须按 `flashinfer/__main__.py` 核对，不在此给出虚构操作教程。
目标治理需要覆盖：身份失配、坏记录、噪声拒绝、重新采集、来源展示与清理隔离。

证据：`flashinfer/tuning/benchmark.py`、`store.py`；
`tests/test_tuning.py`、`test_tuning_release.py`；
[架构评审](reports/ascend-architecture-review.md) §9。
公开 autotuner 对标范围另见附录 B，不能用本节已有设施升格整个空域。

## 10. 自定义 kernel 接入

当前状态 `[P]`：native v2 与 catlass v2 已有生产范本，任意算子仍须完成数学、构建、ABI、公开入口和验收。

本节面向“新增一个可交付算子”，不是只新增一条 registry JSON。
两条生产路径证明共享基础设施可复用，不证明扩展成本只剩写 device kernel。

### 10.1 接入方式与契约版本

当前状态 `[V]`（限定 v1/v2 契约与两条范本）：版本分流明确，旧 v1 护栏保留。

| 维度 | v1 | v2 |
| --- | --- | --- |
| 产物 | `host_shared` 或 `op_package` | 仅 `op_package`。 |
| task / sync | AIC-only、compiler_default | 可声明 `aic_aiv` 与 `cross_core_flags`，须配套 metadata 校验。 |
| recipe workspace | 0 | 非负固定值或 `dynamic`，动态需具名 host 声明。 |
| tiling | host_shared 为 none；op_package 为 fixed、schema 1 | host_dynamic、tiling schema ≥2，绑定 host 入口。 |
| Tensor 参数 | 既有三 Tensor probe 契约 | 有序 `tensor_parameters`，任意正数量的受支持 Tensor。 |
| shape | 探针的固定约束 | 符号、有限范围、倍数、rank、跨 Tensor 一致性。 |
| 执行证明 | 已有 probe | 两条生产 envelope；不支持任意 Tensor 内容依赖。 |

`BuildRecipe.schema_version`、recipe 自身 `version`、`output_adapter` 容器版本分别管理。
v2 继续使用 `ascendc.op_package.v1` 容器名，不表示它退回 v1 recipe。
v1 序列化排除 v2 新默认字段，既有身份不应因升级无意变化。

**数据内容驱动的 tiling 是明确缺口。**
`shape_constraints` 绑定 shape 符号，不表达 `actual_seq_lengths` Tensor 中的值。
若 host tiling 必须读这些值，需要另扩数据协议、同步/拷贝、生命周期和 graph 约束。
不得硬编码长度或把内容伪装成 shape 绕过契约。

### 10.2 通用变参 ABI 与执行生命周期

当前状态 `[V]`（限定纯 Tensor eager）：loader 按声明生成参数 ABI，非 Tensor scalar/attribute 与异步生命周期未扩展。

```text
ACLNN 第一段：
  GetWorkspaceSize(aclTensor* arg0, ..., aclTensor* argN,
                   uint64_t* workspace_bytes, aclOpExecutor** executor)
ACLNN 第二段：
  launch(void* workspace, uint64_t bytes,
         aclOpExecutor* executor, aclrtStream stream)
device 参数顺序：
  recipe 中有序 tensors → workspace → tiling
```

“变参”指由 recipe 决定参数表长度，不是无类型 C varargs。
`get_workspace_size(*tensors)` 支持声明顺序的位置参数或具名参数。
个数/名称错误、混用方式错误在进入 CANN 前拒绝。
loader 根据参数数量设置 ctypes ABI，没有按生产算子名字另写专用加载器。

workspace/tiling 不伪装成额外用户 Tensor。
CANN executor 绑定本次 host tiling，launch 必须使用 GetWorkspaceSize 返回的需求。
两段均检查非零 CANN status，第一段还拒绝空 executor。
用户 scratch 与 CANN 系统 workspace 是两件事；不能只分配 recipe 中可见的用户 scratch。
host callback 入口字段是声明与身份，实际执行由 CANN `SetTiling` 注册完成。

`package_tensor_runner()` 负责 torch Tensor → aclTensor、dtype/rank/shape、连续 ND、同设备与输出别名检查。
参数、维度数组、aclTensor、workspace 保活到当前 stream 同步完成，再释放资源。
**当前是同步 eager adapter**；这保证的寿命边界不能冒充无同步热路径或图安全。

### 10.3 native 与 catlass 两条路径

当前状态 `[V]`（限定生产子集）：两者共用 compiler、store、loader 和 AOT 发现，不是两套 runtime。

| 项目 | native v2 | catlass v2 |
| --- | --- | --- |
| 范本 | `matmul_bias_silu` | `flash_attention`。 |
| device 源 | 本仓实现 | 迁移选定模板实例，保留来源及许可。 |
| 依赖 | 可为空依赖集 | 固定 Git revision、include root 与实际 header tree digest。 |
| 构建 | CMake/bisheng，冻结 recipe/spec | 同链路，额外解析外部头与目标模板。 |
| AOT hit | 直接消费已验证 package | 同样不需要构建时 catlass checkout。 |
| JIT miss | 解析本仓源闭包 | 还需匹配 catlass root；缺失在编译前明确失败。 |

`FLASHINFER_CATLASS_ROOT` 提供外部 checkout；wheel 不自带完整 catlass 头树。
改变 revision 或实际头文件内容须改变构建身份。
catlass 失败不能自动换成 native 而保留同一性能或数值承诺。
FA 的定长子集既受所迁例子限制，也受本次主动收窄的 batch/layout 范围限制。
这条独立 ACLNN eager 入口没有补齐 Attention POD provider、公共 batch generator 或 graph。

### 10.4 端到端交付件与注册边界

当前状态 `[P]`：完整范本和清单已交付，脚手架、任意 ABI 与统一发布自动化仍待补齐。

**完整交付件只引用一处**：
[自定义算子接入实录 §4：native v2 / catlass v2 的 33 项交付清单](reports/custom-operator-onboarding.md#4-新增一个自定义算子的可执行交付清单)。
本文不复制该表；实际操作顺序见 [新增算子指南](adding-a-custom-operator.md) §1–§5。

```text
数学 / envelope / reference
  → device TU + host OpDef/infer/tiling + 共用 POD
  → recipe / 源摘要 / 依赖 / shape / ABI
  → resolve + freeze + 真实构建 + verify/export
  → descriptor + runner + 公共 facade
  → host 反例 + 真机数值 + AOT/JIT 错误路径
  → 安装与独立执行模式验收
```

外部 custom 与内置生产登记不能混用同一承诺：

| 入口 | 作者责任 | 限制 |
| --- | --- | --- |
| `register_custom_kernel` | 先提供可调用 runner 与能力契约 | 不自动生成 API、编译器或 lazy AOT 接线。 |
| 内置 `bind_aot_kernel` 后登记 | 冻结 descriptor/spec、runner_factory、可选 JIT factory | 要求相应 Ascend C AOT 类别；不接受任意 CUSTOM descriptor。 |
| `AscendCOp` / torch.library | schema、mutation、device_impl、meta_impl | 与 registry 独立接线，不能因 eager 成功自动获得。 |

custom 必须用 `custom/` 命名空间，禁止同名覆盖内置实现。
auto opt-in 与显式路由分开，显式指定仍需满足能力契约。
自定义 store namespace 与本地 tuning 可分离，但不是自动执行不可信代码的安全隔离系统。

每次扩大 envelope 必须同步 recipe、descriptor、host 实现和数值/拒绝测试。
新增登记要更新精确集合验收，不能把断言放宽为“至少一条”。
图声明默认保持 `NOT_CAPTURABLE`，compile、fake、opcheck、aclgraph 分别验收。

证据：[v2 契约](reports/ascendc-contract-v2.md) §2–§6；
[接入实录](reports/custom-operator-onboarding.md) §1–§6、§8；
`flashinfer/jit/recipe.py`、`op_package.py`、`tensor_adapter.py`、`flashinfer/ascendc.py`。

## 11. 关键对象汇总

当前状态 `[P]`：关键对象存在且分工清楚，但不能从 schema 完整性推导全路径已接通。

| 对象 | 位置 | 当前用途与边界 |
| --- | --- | --- |
| `InputSignature` | `runtime/signature.py` | dispatch 输入身份；编译特化通过 spec 显式声明。 |
| `HardwareSignature` | `runtime/hardware.py` | SoC、AI Core、UB；不含 rank_num 或完整软件兼容结论。 |
| `KernelDescriptor` | `runtime/schema.py` | 候选能力、source/ABI、图声明；需绑定真实 runner。 |
| `DispatchDecision` | `runtime/dispatch.py` | 选择、来源栈、逐候选原因；fallback_chain 不是重试执行器。 |
| `Plan` | `runtime/plan.py` | 通用计划；workspace 零占位不代表全库需求为零。 |
| Attention session/workspace | `attention/planner.py`、`workspace.py` | 独立准备/提交、容量与资源契约；尚无统一 batch scheduler。 |
| `BuildRecipe` | `jit/recipe.py` | v1/v2 构建与 ABI 契约、源/依赖闭包。 |
| `JitSpec` | `jit/core.py` | 内容寻址构建身份；Attention 声明 spec 未必是 compilation unit。 |
| `DiskJitBuildStore` | `jit/cache.py` | 校验、锁、staging、不可变 generation 和原子指针。 |
| `AotPackageIndex` | `jit/discovery.py` | 搜索/匹配索引；不代替 payload 验证。 |
| `AotKernelRunner` | `jit/runtime.py` | 延迟解析、绑定和 provenance；仅 missing 受控 JIT。 |
| `GraphReplaySpec` | `runtime/schema.py` | 参数更新范围与证据；不是任意环境自动兼容认证。 |
| `ReplayParameterUpdates` | `npu.py`、`runtime/_graph_params.py` | task、引用、event、更新顺序；单独不管理模型图。 |
| tuning record | `tuning/` | 测量和查询依据；发布库为空，环境身份仍需完善。 |

Attention binary/wire ABI 与 ACLNN package ABI 是不同契约。
前者的序列化指纹不会生成后者的 Tensor adapter。
上游 TVM-FFI typed export 也不等于本仓 torch.library 注册。

## 12. 错误处理与降级

当前状态 `[P]`：关键拒绝路径有测试，运行 CANN 兼容检查与统一模式门禁仍有缺口。

| 场景 | 当前行为或明确缺口 |
| --- | --- |
| 显式 backend/kernel 不可用 | 拒绝并给候选原因，不静默跨 backend。 |
| AOT missing | 仅 ENABLED 且有 factory 才回落 JIT；其它策略拒绝。 |
| AOT corrupt | 即使 JIT 开启也拒绝，附完整性失败项。 |
| AOT incompatible | 尝试其它候选；全部不匹配归带原因的 missing。 |
| AOT load error | 保留 cause 并拒绝，不改走 JIT。 |
| 执行/launch 错误 | 保留原始错误，不按 fallback_chain 自动重试。 |
| 直接 JIT cache 损坏 | 不执行损坏产物；允许编译时发布新 generation。 |
| 源摘要/冻结身份失配 | 构建前拒绝，要求重新声明与冻结。 |
| plan 容量/身份变化 | 拒绝或显式重建；已捕获资源不能静默换绑。 |
| graph 未准备/漏更新 | 公共 wrapper 拒绝；设备更新失败后弃图重建。 |
| custom 同名或不合法命名 | 注册拒绝；auto 未 opt-in 时不参与。 |
| 第二个不同 package identity | 当前明确拒绝，要求新进程。 |
| 当前运行 CANN 版本不兼容 | 尚无完整预检；不能承诺已提前拦截。 |

目标诊断应把 policy、候选、spec、来源、完整性、加载与执行阶段分开。
“fallback”不能作为忽略所有错误的总开关。
对 mutation 算子，执行后换 backend 重试尤其需要独立设计，当前没有这种保证。

## 13. 配置项

当前状态 `[P]`：下列配置有读取或对应机制，配置存在不等于全库行为已经统一。

| 配置 | 默认/作用 | 当前限定 |
| --- | --- | --- |
| `FLASHINFER_NPU_CACHE_DIR` | `~/.cache/flashinfer`，统一 store 根 | 原生 Triton 还有自己的缓存归属。 |
| `FLASHINFER_NPU_BACKEND` | `auto` | dispatch 的 `backend=None` 才取全局默认。 |
| `FLASHINFER_NPU_DISPATCH_ORDER` | `external,cann,aot,jit,reference` | 完整排列；不是性能排名或磁盘命中查询。 |
| `FLASHINFER_NPU_JIT_DISABLE` | `0`；为 1 时禁止相应 miss 编译 | AOT runner 与直接 build_and_load 的 cache 行为见 §8.3。 |
| `FLASHINFER_NPU_JIT_CACHE_SIZE` | 1024，相关进程内 LRU 上限 | 不表示磁盘配额或 dispatch 映射上限。 |
| `FLASHINFER_NPU_DISABLE_DISPATCH_CACHE` | `0` | 禁用 decision 缓存不改变选择语义。 |
| `FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS` | `0` | 控制 auto 候选准入。 |
| `FLASHINFER_NPU_TUNING_DIR` | `<cache>/tuning` | 本地记录目录，不产生实测数据。 |
| `FLASHINFER_NPU_ONLINE_TUNING` | `0` | 配置入口；不承诺自动探索闭环。 |
| `FLASHINFER_AOT_PATHS` | 未设置 | 追加搜索位置，默认随包目录仍参与。 |
| `FLASHINFER_CATLASS_ROOT` | 外部依赖 root | 构建/JIT miss 使用，AOT hit 无需 checkout。 |
| `ASCEND_RT_VISIBLE_DEVICES` | 由运行环境控制 | 可见卡数不是 HardwareSignature 字段。 |

配置更改不自动卸载已解析 runner、已加载 module 或已捕获图。
重复运行应记录关键配置、软件版本、设备和产物身份，而非仅记录函数名。
证据：`flashinfer/config.py`、`jit/runtime.py`、`jit/discovery.py`、`ascendc.py`。

## 14. 测试策略

当前状态 `[V]`（给定基线范围）：全量 1975 passed / 0 failed，real_compile 80 passed / 0 skipped。

这些是本任务引用的已核实结果，本次没有重新跑 pytest、编译或真机执行。
环境为文首的 8× Ascend950PR；可见设备数量不说明做过八卡并行验收。
不存在同口径上游 GPU/NPU 性能测量，不能声明追平或领先。

### 14.1 证据分层

当前状态 `[P]`：不同层已有测试，不能把一个层的通过外推到另一层。

| 验证层 | 应证明什么 | 不能替代什么 |
| --- | --- | --- |
| host schema/unit | 序列化、身份、过滤、错误和状态机 | NPU 数值或真实编译。 |
| real_compile | 真工具链、真实产物、设备执行及指定负例 | 全部合法组合、并发、跨版本。 |
| 数值/reference | 指定 shape/seed/误差口径 | 性能或所有极端输入。 |
| fake/opcheck/fullgraph | Torch schema、mutation、追踪边界 | aclgraph 稳定资源和 task update。 |
| capture/replay | 固定资源、动态项、更新顺序及 eager 对照 | 完整模型或所有 backend 图支持。 |
| package/安装 | wheel 内容、发现、禁编译运行 | 跨 SoC/CANN 或多包共存。 |
| benchmark | 冷/热分离的同口径测量 | 由测试通过数推断性能。 |

### 14.2 四条轴线的验收入口

当前状态 `[P]`：以下入口有证据或测试载体，新增组合必须补充对应验证。

| 轴线 | 验收关注点 | 入口 |
| --- | --- | --- |
| JIT | 身份变更、锁内复查、并发去重、原子发布、损坏不执行 | `tests/test_jit_build.py`。 |
| AOT | 无编译器命中、missing 开/关 JIT、corrupt 拒绝、不兼容原因、load error | `tests/test_aot_runtime.py`、两个生产 `*_real.py`。 |
| 图模式 | plan/warmup、固定资源、长度变化、多次 replay、漏更新/越界/换绑拒绝 | `tests/test_single_decode_graph_real.py`。 |
| 自定义算子 | v1 身份保留、v2 guard、参数 ABI、shape 拒绝、数值与依赖缺失 | `tests/test_ascendc_contract_v2.py`、两个 production registration/real 文件。 |

生产数值与 golden 的口径见交付报告。
catlass FA golden 来自原生成器、先舍入 FP16 后写 FP32 文件，不是未舍入 FP64 reference。
skip 不能算新设备证据，构建/导入成功也不能算数值通过。
后续新增算子应保留原基线用例，记录新增证据与实际总量，不把快照计数永远钉在旧值。

### 14.3 ABI 冻结与兼容性

当前状态 `[V]`（限定当前自洽契约）：重冻决策已记录，不能声称恢复了原始定义或证明新旧等价。

[ABI 重冻记录](reports/abi-refreeze-2026-10-25.md) 说明旧指纹无活产物依赖，采纳当前重建为权威契约。
结构尺寸未变不足以证明语义兼容；原始 error ABI 完整定义仍未恢复。
若将来获得原定义或含旧契约的真实 manifest，须重新评估决定。
本版不重新修改冻结值，也不以更新 golden 代替兼容性说明。

## 15. 实施路线

当前状态 `[P]`：原里程碑已有局部交付，后续投资以附录 B 的依赖与验收门禁为准。

| 原阶段 | 当前进展 | 后续验收方向 |
| --- | --- | --- |
| M0 框架骨架 | `[V]` 选择/注册/签名已有证据；API 全语义仍 `[P]` | 冻结并覆盖推理关键路径矩阵。 |
| M1 双 backend 闭环 | `[P]` CANN 主线和生产 Ascend C 各有入口 | 不宣称原建议 norm 已有同算子三 backend 对照。 |
| M2 AOT/JIT | `[P]` 两条生产已连自动 AOT 与受控 JIT | 安装、版本、多包及批量预热治理。 |
| M3 Triton | `[P]` 有生产 norm，统一 store 另有路线 | 明确缓存归属和升级/禁编译语义。 |
| M4 attention | `[P]` 标准接口已有，batch/generator/量化扩展不足 | 完善既有范围，按需求选择扩域。 |
| M5 图模式 | `[P]` aclgraph single-decode 窄闭环 | 生产自研与 batch 分别验收；删除旧 GE 回调承诺。 |
| M6 Tuning | `[P]` 查询/采集存在，发布库为空 | 软件身份、真实数据、capture 门禁。 |
| M7 自定义接入 | `[P]` v2 与双范本、33 项清单 | 降低手工冻结/发布负担，扩展前先补契约。 |
| M8 生态 | `[G]` 完整模型及 serving 验收仍缺 | 指定 consumer/workload 的可复现验证。 |

优先修复会阻断部署与组合使用的边界，再扩算子数量。
两条生产路径各自通过不抵销多 package identity 的组合阻塞。
新增目录、类名或配置开关不能作为里程碑完成条件。

## 16. 设计约束清单（摘要）

当前状态 `[P]`：以下区分已执行的机制与待实现约束，不能作为全库完成声明。

1. `[V]` 调度必须可解释；默认 tier 不是性能排名，runner 才执行。
2. `[P]` 已登记 Ascend C 路径自动 AOT/受控 JIT；全库 AOT-first 仍是目标。
3. `[V]` 真实构建身份为 JitSpec fingerprint，不能回退到 legacy helper 公式。
4. `[V]` 命中验证 manifest/inventory，先完整发布 generation 再原子换指针。
5. `[V]` 同身份冷构建由锁内复查去重，旧 generation 不原地覆盖。
6. `[V]` AOT corrupt/load error 不回落，launch 失败不按候选链自动重试。
7. `[P]` 热执行应持有 runner；无分配/无同步需逐入口验收。
8. `[P]` tuning 的性能结论只能来自实测；软件身份与发布数据仍需完善。
9. `[P]` custom 命名/准入与执行契约分开；namespace 不构成安全沙箱。
10. `[V]` v1 护栏保留；v2 的 Tensor/shape/同步扩展带独立校验。
11. `[G]` 数据内容驱动 tiling 不由 shape_dynamic 自动提供。
12. `[P]` aclgraph 的准备、捕获与重放前更新分离，能力逐条附证据。
13. `[G]` 生产自研和 batch 入图未完成，不能靠 class 名或 flag 升格。
14. `[G]` 多 package identity、运行 CANN 兼容检查仍是部署缺口。
15. `[P]` torch.library、compile、aclgraph、安装及性能必须分别验收。
16. `[P]` 上游机制、当前实现和未来路线分开记账；时点变化由增量证据解释。

## 附录 A. 当前实现进展与证据

当前状态 `[P]`：本表是基线快照；每行的 `[V]` 只覆盖列出的机制与已有运行范围。

### A.1 逐组件状态

当前状态 `[P]`：原报告中的部分接通或缺口，只在后续交付有证据时更新对应子项。

| 组件 | 状态 | 当前进展与保留缺口 | 证据指向 |
| --- | --- | --- | --- |
| API 层 | `[P]` | 43 op、九域有入口/runner；八域空或仅壳；不等于逐参数对齐 | [接口面](reports/api-surface-by-category.md) §0–3；`flashinfer/decode.py`、`prefill.py`。 |
| dispatcher | `[V]` 限选择契约 | 过滤、两级报告、确定性排序、显式锁定、revision 缓存；不执行失败重试 | `runtime/dispatch.py`；`tests/test_two_level_dispatch.py`、`test_dispatch_cache.py`。 |
| CANN backend | `[V]` 限既有用例 | 40 descriptor；FIA/composite 等可运行，全部组合与性能未验证 | `flashinfer/cann/`；接口面 §2；[架构评审](reports/ascend-architecture-review.md) §2、§3。 |
| Triton backend | `[P]` | 1 descriptor，生产 norm 有数值；统一 store 与原生路径并存 | `flashinfer/triton/`；架构评审 §5.3；接口面 §2.5。 |
| Ascend C 自研 | `[P]`；两条 envelope `[V]` | native v2 与 catlass v2 已接公共 eager；异步/图/全域未完成 | [v2](reports/ascendc-contract-v2.md) §4–7；[接入实录](reports/custom-operator-onboarding.md) §1–3、§7。 |
| JIT | `[V]` 限缓存/构建链 | spec 内容寻址、锁、原子 generation、真实编译；全库治理 `[P]` | `jit/core.py`、`jit/cache.py`；`tests/test_jit_build.py`；架构评审 §5。 |
| AOT | `[P]`；登记路径 `[V]` | 自动发现/命中/绑定/受控 JIT 已接通；安装矩阵、provider 与多包仍缺 | [发现说明](aot-runtime-discovery.md)；`tests/test_aot_runtime.py`、两个生产 real 文件。 |
| 图模式 | `[P]`；single 窄路径 `[V]` | 1/81，CAPTURE_SAFE 0；生产自研与 batch `[G]` | [图交付](reports/graph-mode-delivery.md) §1、§4、§9、§11；`tests/test_single_decode_graph_real.py`。 |
| 自定义算子接入 | `[P]` | v1/v2、通用 Tensor ABI、双范本；scalar/attr、数据内容 tiling 未补 | [33 项清单](reports/custom-operator-onboarding.md#4-新增一个自定义算子的可执行交付清单)；[操作指南](adding-a-custom-operator.md)。 |
| tuning | `[P]` | 采集/store/query 已有，发布库空；软件身份不足，无公开 autotuner | `flashinfer/tuning/`；`tests/test_tuning.py`、`test_tuning_release.py`；接口面 §3。 |
| ABI 契约 | `[V]` 限当前冻结 | 当前契约自洽；原定义未恢复，新旧语义等价未证明 | [重冻决策](reports/abi-refreeze-2026-10-25.md) §三–五；`tests/test_runtime_abi.py`、`test_attention_metadata_wire.py`。 |
| Attention generator/provider | `[P]` / `[G]` | variant 身份有契约；编译单元、公共 batch 注入和默认 provider 闭环未补 | [分层报告](reports/call-stack-layering.md) §2.2；`jit/attention/modules.py`、`attention/public_batch.py`。 |
| 多 package identity | `[G]` | 同进程第二个不同 identity 明确拒绝 | `jit/op_package.py`；发现说明“范围与已知缺口”。 |
| CANN runtime 兼容 | `[G]` | 构建身份存在，当前运行版本比较未实现 | 发现说明“范围与已知缺口”；v2 报告 §9。 |
| 数据内容驱动 tiling | `[G]` | shape 驱动不能替代 Tensor 内容读取契约 | 接入实录 §3.3、§8；操作指南 §7。 |
| 测试基线 | `[V]` 限给定环境 | 1975 passed / 0 failed；real_compile 80 passed / 0 skipped | 本任务已核实基线；测试载体见 §14；历史运行记录不冒充当前总数。 |
| 性能/完整模型 | `[G]` 证据缺口 | 无同口径性能、完整 serving/模型图验收 | 架构评审 §10；图交付 §11；接入实录 §8。 |

上表短代码路径均相对 `flashinfer/`，`tests/` 路径相对仓库根。
表内两个 `[V]` 子集不能相加推导为多包同进程或多卡组合已经可用。

### A.2 历史报告的时态裁决

当前状态 `[P]`：以下差异由增量交付解释，未发现需要推翻上游静态机制基准的证据。

| 历史说法或口径 | 本版采信与理由 |
| --- | --- |
| 架构评审/简报写无自动 AOT 发现、生产自研登记为空 | 由 P0-2 发现机制、T2′ native v2 和 T4 catlass v2 更新；当前两条生产接线。 |
| AOT 发现说明、分层报告仍写仅 probe/空生产表 | 保留其错误策略与索引机制；生产范围以两份后续交付报告为准。 |
| 分层报告写固定三 Tensor、动态 tiling/混合核不可接 | 作为 v1 历史限制保留；v2 已放开纯 Tensor、shape 驱动及混合核，不能扩到数据内容驱动。 |
| 旧图报告计数及“公共 single 缺资源入口” | 新 wrapper 已补公共入口；T4 新条目不可入图，本版统一 1/81。 |
| 架构评审/简报仍列 ABI 冻结失败 | 采信重冻决策：当前门禁自洽，但原契约不可恢复及语义等价未证仍保留。 |
| v2/T4 报告保留当时登记快照断言失败和旧测试总量 | 属交付过程时点；当前数字直接采用任务给定 1975/80，不重算、不改历史报告。 |
| “当前文档缺图模式一节” | 文件已有 §3.4 子节；本次按要求新增独立 §3A，并修正旧 §15/§16 的 GE 设想。 |
| 旧稿称上游 kernel 全合主包、本仓独立 artifact wheel | 与上游机制基准及本仓交付不符；已改为上游多角色、本仓主 wheel 可附产物。 |

数据口径必须同时注明对象与时点。
本版九域 op 数沿用户已核实清单；不把新增内部域名、别名或 production descriptor 另算一遍。
历史报告的测试总量不是本次重新执行记录，也不是能力百分比。

### A.3 本版仍未验证的事项

当前状态 `[G]`：以下不能从已有单点证据推导，须独立验收。

- 全量 dtype/shape/layout、极端数值、所有合法 envelope 组合。
- 其它 SoC、CANN/driver/torch_npu 版本、aarch64 host 和异构设备。
- 多线程、多 stream 重入、多卡并行、跨 fork module 与长期压力。
- 生产 AOT wheel 安装后的端到端部署矩阵；pure wheel 内容检查不足以证明运行。
- 两条生产自研的 torch.library/fake/fullgraph、aclgraph 与 batch replay。
- 同进程多 package identity、数据内容驱动 tiling、当前 CANN 运行版本兼容。
- 全库首次 JIT/AOT/autotune capture 门禁、完整模型 FULL/PIECEWISE 及下游 consumer。
- 上游 CUDA 真机执行、同口径吞吐/延迟/显存对比；没有性能优势结论。

本次文档验证只检查结构、链接、状态、计数口径和修改范围。
上述缺口中的“机制未实现”与“实现存在但未验证”已分别在正文说明。

## 附录 B. 两个月对标路线图

当前状态 `[P]`：以下是建议目标和验收门禁，不是完成承诺，也不预设人力或精确工期。

### B.1 可行性判断与“对标”的定义

当前状态 `[G]`（全量差距）：**两个月内完成与上游的全量对标不可能。**

上游有 **42 个直接子包、2656 个 `.py` 文件**。
本仓 **9 个能力域有 op，8 个域为空或仅有壳**。
差距不仅是数量，还包括状态生命周期、交付、扩展、通信、编译路线与验证矩阵。
这些数字解释范围，不能换算实现覆盖率、人月或准确工期。

两个月可以追求的目标是：**选定推理关键路径在明确支持矩阵内，具备可对照验收的接口、语义、生命周期和部署行为。**
这要求先冻结 workload、设备/软件栈及关键入口，限制同时新增的能力域。
没有这一步，“做成竞品的样子”容易退化为同名 API、空壳目录或不可复现的演示。

| 对标层 | 可操作定义 | 不算达成 |
| --- | --- | --- |
| 接口 | 参数、返回值、dtype/layout、out/alias、异常边界有矩阵 | 仅 import 成功或函数同名。 |
| 执行语义 | 承诺输入有数值/状态更新证据，不支持的语义明确拒绝 | 静默丢 mask、量化选项或广播差异。 |
| 生命周期 | plan/run、预热、workspace、capture/replay、失效规则可复现 | 仅提供 plan 或 use_cuda_graph 开关。 |
| 交付 | 安装后公共入口可运行，AOT/JIT/禁编译/完整性行为清楚 | 开发机显式加载一个目录成功。 |
| 性能证据 | 相同数学、shape、误差、模式与测量政策下形成可复验数据 | 未测先承诺追平、低延迟或加速比。 |

P0 是本周期应承诺的最低验收集合；未通过就不能宣布周期目标完成。
P1 是 P0 不受侵蚀时争取的增量；最多补齐 1～2 个空域的受限子集。
P2 明确不列入两个月交付，不通过修改名称把它们算作已对齐。

### B.2 P0 必达：关键路径语义与交付可信度

当前状态 `[P]`：已有机制可复用，主要工作是收敛支持范围、补边界和把验证推进到部署。

表中每项均为“缺口 → 依赖 → 验收 → 风险”，规模是相对判断。
不同项可能共用工作，不能把规模相加推出精确工期。

| 条目 / 相对规模 | 缺口描述 | 依赖 | 验收标准 | 风险 |
| --- | --- | --- | --- | --- |
| P0-A 关键路径矩阵 / 中 | 九域有接口但不代表逐参数等价；RoPE/cascade/page 等边界缺失 | 固定上游 commit、目标推理 workload、dtype/layout/选项清单 | 每个承诺入口有数值、mutation/alias、错误和容量正反例；缺失组合可查询并拒绝 | 组合膨胀；CANN 与上游语义不同，需显式适配或收窄。 |
| P0-B 安装后生产 AOT/JIT / 中 | 自动机制已接通，生产 wheel 安装验证与通用交付仍弱 | 当前两个 recipe/spec、构建/export，分别准备无编译器 runtime-only 与具备匹配工具链的环境 | 安装后公共 API 在无编译器环境验证 AOT hit/禁 JIT，在具备工具链环境验证真实 JIT 回落；missing、corrupt、incompatible、load error 各有诊断 | 误把源码目录命中当安装验收；依赖与源码身份漂移。 |
| P0-C 运行版本门禁 / 中 | build 身份不证明当前 CANN runtime 兼容 | 明确支持的软件栈与可采集版本，定义兼容规则 | 已支持组合通过；不兼容或未知组合按发布策略提前拒绝/明确诊断；不虚构跨版本兼容 | CANN ABI 变化不只由版本号决定，验证矩阵受设备环境约束。 |
| P0-D 图模式公共边界 / 中 | single 窄闭环易被误读为全库 graph；首次冷路径统一门禁不足 | §3A 公共资源、预热、状态机和 graph evidence | 已承诺路径捕获前准备完成；漏更新/换绑/越界/cold capture 明确拒绝；不支持路径保持保守声明 | 延迟加载、隐式同步与第三方编译路径可能绕过门禁。 |
| P0-E 生产最小图闭环 / 大 | 两条生产自研均不可入图 | 先选一个 kernel/envelope；完成异步资源寿命与固定 buffer 方案；依赖 P0-D | 所选生产路径真实 capture/replay 与 eager 对照，动态项、容量、失败恢复有正反例；通过后仅提升该声明 | ACLNN executor/host tiling、同步 adapter 改造可能比 kernel 数学更重；未过门禁须记作 P0 未达。 |
| P0-F tuning 身份与诊断 / 中 | 发布库空、软件身份不足；缓存和选择来源不够统一 | 稳定候选、源码/编译栈身份、代表 workload | 升级拒绝旧 winner；坏记录受控；可解释选择与 AOT/JIT provenance；冷/热路径分开报告 | 噪声导致错误 winner，原生 Triton 与统一 store 政策不一致。 |
| P0-G 基线与独立复验 / 小至中 | 总测试数不能对应所有承诺组合 | 上述支持矩阵、可复现环境与安装产物 | 保留当前 1975/80 基线覆盖；新增项映射到接口/设备/模式；全量零失败、real_compile 零跳过，并保留安装证据 | 以导入率、case 数或 skip 掩盖未覆盖语义。 |

P0-A 的“补齐边界”包括明确拒绝与修复承诺范围内错误，不等于补齐九域所有上游入口。
缺少的 Llama 3.1 RoPE、cascade in-place、page 索引辅助等，按目标 workload 决定是否进入承诺清单。
一旦纳入清单，不能靠写“不支持”算作功能交付；未纳入的仍列明差异。

P0-B 必须说明当前单进程单 package identity 的限制。
若选定模型必须同时调用两条不同 package，解除该限制就是该 workload 的前置阻塞，须优先处理并压缩 P1。
若采用已声明的进程隔离部署，只能验收该部署范围，不能宣称多包共存问题解决。

P0-F 只补最小测量与身份底座，不宣称完成公开 autotuner 或完整 trace 域。
性能采集用于发现瓶颈；结论可以是持平或更慢，不把速度门槛凭空设成承诺。

### B.3 P1 争取：有限扩域与 batch 图试点

当前状态 `[G]`（新增范围）：建议新域优先考虑 MLA 与基础量化，是否交付取决于 P0 和依赖收敛。

MLA 与基础量化是候选，不是未经 workload 选择就同时启动的固定承诺。
若实际模型更依赖稀疏或通信，可替换一个候选，不直接累加第三个域。
batch 入图是既有域深化，不计作新增能力域。

| 条目 / 相对规模 | 缺口描述 | 依赖 | 验收标准 | 风险 |
| --- | --- | --- | --- | --- |
| P1-A 受限 MLA / 大 | 无 MLA 模块，MLA page append 等也缺 | 选定 MLA 数学与 KV layout、metadata/workspace、可用 CANN 或自研实现，P0 交付门禁 | 一个明确 decode 或 prefill 子集贯穿 public API→runner→reference→安装；其它布局/长度明确拒绝 | 普通 MHA 不能代替 MLA；数据布局及模型变体可能扩大任务。 |
| P1-B 基础量化算子 / 中至大 | QuantSpec 仅契约，缺 packbits 等实际算子 | 选择真实消费场景、bit/scale layout、舍入规则、目标设备能力 | 最小 pack/unpack 或 quant/dequant 子集含值/scale、尾部/边界误差与下游消费验证 | FP4/FP8/MX 格式和舍入不等价；只做转换而无消费者可能价值有限。 |
| P1-C batch 图试点 / 大 | 固定 metadata 尚未连接可捕获 batch 执行器 | P0-D 门禁，batch 固定任务规模、中间/output/workspace 与逐任务更新协议；P0-E 经验可复用但非硬前置 | 一个固定容量 batch wrapper 在长度/页表变化后 replay 正确；越界、换绑、漏更新拒绝；仅提升该 descriptor | Python loop/gather/cat 需结构改造，不能删除 recording guard 冒充支持。 |
| P1-D 接入减负 / 中 | 手工摘要、spec 冻结、验收 harness 重复 | 两个稳定范本、33 项清单、P0-B | 脚手架生成可审查骨架；漏摘要/依赖/ABI 能提前拒绝；新算子可复用真实七场景验证 | 自动生成可能掩盖数学与 host 约束，必须默认 NOT_CAPTURABLE。 |

P1-A/B 的完成只意味着该子集从空域变为有证据的入口。
不能写成“MLA 已全面对标”或“量化域全覆盖”。
P1-C 未通过时保留 batch `[G]`，不影响已验证 single 的窄声明，也不扩大它。

### B.4 八个空域逐项处置

当前状态 `[G]` / `[P]`：空模块与契约壳分开记录；只有明确产品/平台非目标归“有意不对标”。

| 能力域 | 差异性质与当前缺口 | 依赖 → 两个月验收 / 处置 | 风险与相对规模 |
| --- | --- | --- | --- |
| MLA | **尚未补齐**，无模块 | MLA layout/metadata/backend → 按 P1-A 受限闭环；未入选则保持缺口 | 模型与 KV 契约复杂，大。 |
| 稀疏 attention | **尚未补齐**，无模块 | BSR/索引/mask、backend、reference → 默认 P2；若替换 P1，验收一个精确稀疏子集 | 生态有 kernel 不等于本仓接通；索引与 block 组合多，大。 |
| 通信 | **拆分**：MNNVL/NVSHMEM/CUDA IPC 有意不对标；HCCL/通用 collective/EP 尚未补齐 | communicator/bootstrap、stream/所有权、多卡测试 → 默认 P2；平台专有项记录排除，通用语义保持待办 | 不能把 HCCL 缺口伪装成硬件非目标；死锁与并发寿命，大。 |
| 量化算子 | **尚未补齐**，只有 QuantSpec | 格式/scale/舍入/消费者 → P1-B 基础子集；NVIDIA 专用指令不逐实现复制 | 数学与布局须逐项裁决，不能因名称有 NVIDIA 就排除通用需求，中至大。 |
| GDN-KDA-Mamba | **尚未补齐**，无模块 | recurrent state、chunk/prefill/decode、状态池与数值 reference → 默认 P2 | 状态写入/回滚/并发寿命需新体系，不是普通 attention 别名，大。 |
| 训练辅助 | **有意不对标**，推理定位下无模块 | 若改变定位须重新定义 autograd/backward → 本周期验收范围说明与显式不支持 | 与推理目标不匹配，另立大项目；不是已实现。 |
| trace | **尚未补齐**，仅内部协议记录 | 公共 trace schema、dump/replay、替换契约 → P0 仅最小诊断，完整域 P2 | 内部日志不能冒充公开替换/回放系统，中至大。 |
| autotuner | **尚未补齐**，有 tuning 子包但无公开 autotuner | runner/tactic/profile、持久化、环境身份、capture guard → P0 底座，完整域 P2 | 空发布库与接口壳不构成调优能力，中至大。 |

“有意不对标”表示不承诺复制某种平台机制或不属于产品定位。
“尚未补齐”表示通用需求存在，只是当前未交付；不能用延期替换这个判断。
通信可能采用 HCCL 等不同实现，这是底座的必然差异，不消除接口与生命周期需求。

### B.5 P2 不在两个月内

当前状态 `[G]`：以下工作量、依赖或产品范围与本周期不匹配，须另立验收矩阵。

| 条目 / 相对规模 | 缺口描述 | 依赖 | 后续验收标准 | 风险 / 本周期排除原因 |
| --- | --- | --- | --- | --- |
| 完整稀疏家族 / 大 | block sparse、DSA、scores/selection 等未接 | 索引结构、布局、多个数学 kernel 与目标 workload | 分 family 验证 mask/索引/数值/资源，不以单 kernel 代域 | 多阶段管线和组合远超受限 P1。 |
| HCCL/EP 与计算通信重叠 / 大 | 无公共通信和专家分发链 | bootstrap、collective、持久资源、多卡环境 | 通信正确性、生命周期、错误/退出、多 stream 与 EP dispatch/combine | 依赖独立通信体系，非接口改名可得。 |
| GDN/KDA/Mamba 全族 / 大 | 无 recurrent/chunk 状态计算 | 目标模型、状态 layout、forward 更新与回滚 | 分 prefill/decode/state 管理验证，容量/并发/长序列分别验 | 算法与状态契约跨度大。 |
| 数据内容驱动 tiling / 大 | shape-only v2 无法表达所需内容读取 | 可读输入名单、host/device 数据协议、同步、版本与图策略 | 内容变化影响正确 tiling；越界/生命周期/capture 反例可验证 | 扩契约牵连编译、执行与图，不能只放开字段。 |
| 通用多 package / 异构多 SoC / 大 | 单 identity、device 0 快照、跨目标治理不足 | 包装载隔离、设备选择、版本矩阵、目标真机 | 同进程多包无串用；逐 Tensor 选目标；各平台真实执行 | 组合矩阵大；仅在 P0 workload 必需时前移最小子集。 |
| 完整公开 trace/autotuner / 中至大 | 公共回放/替换、tactic/profiling/持久化体系缺 | P0 诊断与测量身份、稳定 runner/schema | 公开 API、跨进程复用、失效、capture 边界、替换 correctness | 不能用最小日志或一个 winner 文件充数。 |
| 全域 provider/预编译矩阵 / 大 | 独立分发、版本发现、多 family 枚举不足 | 安装验收、产物体积数据、各 target 构建与验证 | 可枚举覆盖、离线禁编译、升级回滚和兼容诊断 | 先完成少数生产交付；无规模数据不先复制上游全部包形态。 |
| 全库 graph/模型 FULL / 大 | 当前只有 single 窄路径 | 各算子捕获能力、batch scheduler、consumer 集成 | 按模型/模式实测，不能由算子区域外推全模型 | 生命周期和模式组合多，依赖 P0/P1 图工作。 |
| 训练/autograd 与 NVIDIA 专有实现 | 产品/平台非目标 | 若产品定位变化重新决策 | 本周期只要求准确范围声明；不设虚构功能验收 | 训练偏离推理；专有 runtime/ISA 不作逐实现兼容。 |

生产算子入图、batch 入图、多 package、CANN 兼容、数据内容 tiling 在路线图中均保留名字。
排到后续不降低其风险，也不改变当前 `[G]` 判断。

### B.6 依赖顺序、交付门禁与专家评审点

当前状态 `[P]`：可按依赖推进，未掌握人力与完整 workload 前不提供精确周计划。

```text
冻结目标 workload 与支持矩阵
  → ABI / 版本 / 来源 / 错误边界明确
  → 安装后公共 API 验收与冷/热准备可控
  → 关键路径语义和最小生产图闭环
  → 有余量时选择 P1 的 1～2 个新域
  → 独立复验、整理未达项、保留 P2 边界
```

相互独立的语义边界修复、安装验证、诊断可并行。
依赖契约未定的域不应先批量生成 facade，再把未接线算作进度。
若 P0-E 或多包前置阻塞超出容量，应削减 P1，而非降低证据门禁。

| 门禁 | 通过条件 | 未通过时的处置 |
| --- | --- | --- |
| 范围门禁 | workload、dtype/shape/layout、设备/版本、模式清单固定 | 不承诺泛化覆盖，先收敛输入。 |
| 语义门禁 | 数值、mutation、错误与不支持行为一致 | 修复或从交付清单明确移除，记录未达目标。 |
| 部署门禁 | 安装后 AOT/JIT policy、完整性与版本诊断成立 | 继续作为开发路径，不宣称无编译器交付完成。 |
| 图门禁 | 固定资源、动态项、stream、失败生命周期与真机 replay 证据 | 保持 NOT_CAPTURABLE，不以名称替代。 |
| 发布门禁 | 原基线与新增测试通过，复现材料可交外部复验 | 不以 targeted pass 代替全量回归。 |

建议专家优先裁决四件事：

- 选定推理关键路径是否足以支撑目标模型，P1 新域优先级是否正确。
- 单 package 限制是否阻断目标部署，是否需要前移最小共存方案。
- 同步 eager adapter 向生产图路径演进的资源所有权方案是否可行。
- 是否接受两个月的有限支持矩阵，以及哪些差异必须保留为显式不支持。

这些评审决定影响范围与投入顺序，不改变当前证据。
路线依据：[架构评审](reports/ascend-architecture-review.md) §4、§9–10；
[接口面](reports/api-surface-by-category.md) §1–3；
[上游 45 条机制基准](reports/competitor-survey.md) §5。
后续工作应按 JI/AO/RT/EX 机制和具体能力域交叉验收，不用目录数或新增代码量代替完成度。
