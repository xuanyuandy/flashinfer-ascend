# Wrapper、原生准备复用与 NPU 图模式设计

面向推理框架集成者、后端开发者和框架维护者：本文说明 wrapper、后端准备结果和 NPU 图的职责，给出扩展执行链路所需的资源契约与验收条件。

状态：拟实施方案。本文中的新增接口、目录、构建目标及验收用例均未实现；现有能力以本节基线为准。
第 2–13 节描述目标设计，其中新增类型、准备模式与调用关系不作为当前公共 API。
当前可调用接口见 [图模式接口与范围](../npu_graph_extensions.md)，分层术语沿用 [架构与术语](../concepts/architecture.md)。本页不扩大任何 kernel 的支持声明。

## 1. 目标与当前基线

交付三个可以独立验证、也可以组合使用的能力：

1. 保持上游公共调用方式：直接函数、run-only wrapper、plan/run wrapper 各自沿用原契约。
2. 为可控制的后端实现准备结果复用，减少重复执行中的原生准备工作。
3. 将满足捕获约束的执行路径接入 NPU 图模式；调用方拥有图和模型捕获区域。

按实现查看图能力可运行 `python -m flashinfer list-kernels --all`。目前只有 `cann/attention/single_decode_fia_v1` 声明 `NEEDS_PARAM_UPDATE`，其余默认实现均声明 `NOT_CAPTURABLE`。
这些数字按完整源码发现统计；分芯片 wheel 的目录会裁剪，不能把目录条数视为当前设备可运行条数。
上游公共 API 基线由 `tools/metadata/api_parity.json` 与 `attention_api_parity.json` 固定为 `2895577720741ddfedff118202c89584996b8875`。

当前 AOT 已完成编译、产物验证、加载和函数入口绑定。`flashinfer/aclnn/_native.py::prepare` 缓存的是模块调用绑定；
`csrc/aclnn_torch_adapter/adapter.cpp` 每次执行仍构造 ACL Tensor、调用 `GetWorkspaceSize`、申请所需 workspace 并同步调用。
查询可能命中 CANN 内部缓存，不能把查询次数等同于完整 tiling 计算次数。当前没有通用的跨调用原生计划复用链路。

| 能力 | 当前边界 | 实现依据 |
| --- | --- | --- |
| 图声明与查询 | 按 `kernel_id` 查询；非默认声明必须绑定验证范围和证据 | `runtime/schema.py::GraphReplaySpec`、`npu.py::graph_requirements` |
| 通用 Host 参数更新 | 在调用方 capture 中按调用顺序录制任务；支持同一 kernel 多次出现；本身不创建或重放图 | `runtime/_graph_params.py::_ReplayParameterUpdates`、`npu.py::ReplayParameterUpdates` |
| 受限 single-decode wrapper | `plan/warmup/recording/before_replay/replay`；一张图只绑定一个此类 wrapper，录制恰好一次 run | `attention/single_decode_graph.py::SingleDecodeWithKVCacheWrapper` |
| batch wrapper 图资源合同 | 固定 metadata、容量和执行身份检查；Host `capture_record` 不代表真实设备图 | `attention/execution_identity.py::AttentionGraphResourceContract`、`attention/batch.py::_execute_reference` |
| 自带 Ascend C | AOT 加载与普通执行；没有 prepared 异步桥或已登记图候选 | `aclnn/_native.py::prepare`、`csrc/aclnn_torch_adapter/adapter.cpp` |
| 通用 prepared 与图会话 | `PreparedExecution`、`BackendPrepared`、`GraphSession` 是本文设计，当前没有对应通用实现 | 下文职责与协议设计，不是可导入 API |

表中 Python 路径均相对 `flashinfer/`。受限 single-decode 的已声明范围是 Ascend950、FP16/BF16、NHD，公共 Q 为 `[8,64]`、K/V 为 `[1024,2,64]`；底层使用 BSND `.out`、固定 workspace/out、无 mask、8/2 heads、scale=0.125，并生成 LSE。
只有逻辑 KV 长度在 `1..1024` 内更新，其他存储和执行规格保持冻结。调用方创建 `NPUGraph` 并使用 `auto_dispatch_capture=False`；每次通过 wrapper 重放前都要提交一次 `before_replay`，不能绕过护栏直接调用底层图。
这项限定声明不能外推到 A2/A3、自带 AOT、其他 shape 或通用 batch 图执行；声明中的环境与真实验证入口见 [实现依据与验证边界](#14-实现依据与验证边界)。
当前普通 `dispatch` 不以 `graph_capability` 改变 eager 候选排序，也没有 prepared 选择模式；图查询和特定录制入口的校验不等于自动图路由。后续准备/捕获准入需要显式接入，现有选择行为由 `tests/framework/test_graph_capability.py` 保持约束。

## 2. 对标边界与分层

上游由公共 wrapper 持有计划和资源，原生 planner 生成调度数据，原生 run/kernel 消费这些数据。
参考 flashinfer-ori: `flashinfer/decode.py`、`csrc/batch_decode_jit_binding.cu`、`csrc/batch_decode.cu`、
`include/flashinfer/attention/scheduler.cuh`。外部库适配参考 flashinfer-ori: `flashinfer/cudnn/decode.py`。

本仓采用同样的职责划分，增加多后端、AOT 身份校验和 NPU 参数更新的适配。
下列是目标位置，标注“新增”的文件尚不存在；现有公共导入和已审核 kernel 身份继续兼容。
现有 Ascend C 源码位于 `csrc/kernels/<domain>/<op>/`，按实际实现提供 `arch22`（A2/A3）或 `arch35`（A5）分支；下面只把新增执行组件作为设计扩展。

```text
flashinfer/
├── decode.py / prefill.py / gemm.py             # 原公共函数、Wrapper 名称及签名
├── operators/<domain>/<op>/
│   ├── operator.py                            # 公共语义、校验、输出与前端绑定
│   ├── wrapper.py                             # 算子专属生命周期（按需新增）
│   └── cases.py                               # 独立数学样例和边界
├── runtime/
│   ├── plan.py                                # 现有选型结果；保持其原含义
│   ├── prepared.py                            # 新增：通用执行会话与资源管理
│   ├── graph_session.py                       # 新增：图绑定、更新与重放协调
│   ├── registry.py / dispatch.py               # 公共登记、完整准入与候选选择
│   └── _graph_params.py                       # 复用已有 Host 参数更新机制
├── ascendc/<domain>/<op>/
│   ├── implementation.py                      # 实现声明、调用/准备候选
│   └── runner.py                              # 原生 prepare/update/launch 适配
├── ascendc/_prepared.py                        # 新增：已验证 prepared 产物绑定
├── triton/<domain>/<op>/
│   ├── implementation.py / runner.py           # 编译准备、启动与私有资源
│   └── kernels.py                             # Triton 设备实现
├── cann/<domain>/<op>/
│   └── implementation.py / runner.py           # 原生库接口或 prepared 接口适配
└── jit/                                       # 通用构建、冻结、缓存、验证与加载
csrc/
├── prepared_torch_adapter/                     # 新增：统一异步准备执行桥
└── kernels/<domain>/<op>/
    ├── CMakeLists.txt                         # 原生及 prepared 构建配置
    ├── op_host/plan.*                         # 新增/抽取：可复用规划函数
    ├── op_host/{arch22,arch35}/                # 实际提供的架构 tiling；非每算子必有两套
    ├── op_host/*                              # 既有 OpDef/tiling 兼容适配
    ├── op_kernel/<entry>.cpp                  # 已有各算子显式命名的设备入口
    └── op_kernel/{arch22,arch35}/              # 实际计算分支；按需增加动态元数据输入
```

| 组件 | 负责 | 边界 |
| --- | --- | --- |
| `operator.py` | 公共语义、输入输出、别名和数值约束 | 不嵌入某一后端的原生 ABI |
| 算子 wrapper | 解释 plan 元数据、保留上游生命周期、确定需要重规划的变化 | 不实现具体设备计算或复制后端加载器 |
| 通用 prepared 层 | 选型、会话身份、资源引用、状态、更新提交、关闭 | 不感知 attention、GEMM 等数学类别 |
| 后端 runner | 后端私有准备、更新验证、启动和必要的设备状态重置 | 准备结果可复用范围必须明确 |
| GraphSession | 绑定调用方图、记录调用点、组织更新/重放顺序、持有图资源 | 不负责模型切图、业务分桶或图缓存策略 |
| 编译/加载层 | 构建并验证模块、设备镜像、planner 与 ABI | 不把编译缓存视为执行计划 |

本文把 native prepare 称为“原生准备”，指产生可被后续 run 消费的 tiling、调度和资源状态；`runtime.plan.Plan` 仍只是选型结果。
“准备结果复用”与“图模式”分别验收；“图模式”指调用方拥有的 ACL Graph/TorchNPU NPU 图的实际 capture/replay，不包括 Host trace 重放或结构身份检查。
`prepared-eager` 指复用原生准备结果的普通执行，`graph replay` 指重放已捕获的设备任务；二者都不是 AOT 编译缓存的别名。

公共入口按现有类名和签名延续，`use_cuda_graph` 等上游参数名保持兼容，在 NPU 上解释为图资源模式：

| 公共入口 | 对标的调用契约 | 设计扩展的执行能力 |
| --- | --- | --- |
| `decode.BatchDecodeWithPagedKVCacheWrapper` | `plan/run`、paged KV、跨层复用计划 | 接真实后端计划与批量 runner，按范围支持图 |
| `prefill.BatchPrefillWithPagedKVCacheWrapper` | `plan/run`、paged KV 与 query 元数据 | 后端规划、固定资源、兼容更新 |
| `prefill.BatchPrefillWithRaggedKVCacheWrapper` | `plan/run`、ragged 元数据 | 后端规划、固定资源、兼容更新 |
| `decode.CUDAGraphBatchDecodeWithPagedKVCacheWrapper` | 固定缓冲的兼容构造入口 | 复用上述能力；类名本身不构成图证据 |
| `gemm.SegmentGEMMWrapper` | 上游 `run-only` | 保留原契约；有可控资源时再按后端验证优化 |
| 本仓 `decode.SingleDecodeWithKVCacheWrapper` | 已有限定的图参数更新契约 | 保留既有边界，作为独立适配案例 |

已有 Attention 层继续承担数学元数据解释。`AttentionFrameworkSession`、`AttentionOperatorWrapperSession`、
`AttentionPreparedOperatorPlan`、workspace/执行身份和 `storage_lease.py` 中的校验与寿命规则通过适配接入通用层，
不新增一套并行 Attention 调度器。已有 opaque plan 或 Host 状态机不自动构成 AOT 原生规划与设备事件证据；
补齐的是真实后端准备、异步事件与图资源绑定。原有协议版本和公共前端保留兼容。

## 3. 三项能力分开声明和选择

“有公共 wrapper”“有原生 prepared 执行”“已验证可入图”分别记录。
公共 wrapper 可以复用框架元数据而仍调用一步式 CANN 接口；这类实现不记作原生计划优化。
某个一步式原生接口也可能独立具备图兼容能力，须单独验证，不能仅凭是否提供 `prepare` 判断。

| 实现类型 | 普通调用 | 原生计划复用 | NPU 图模式 |
| --- | --- | --- | --- |
| CANN 直接路由 | 按现有能力 | 仅能复用外部暴露的状态及框架准备结果 | 按具体接口证据；不能整类拒绝或放行 |
| 自带 Ascend C 现有 op-package | 已有有限验证，按算子/SoC 区分 | 当前未完成 | 当前均为 `NOT_CAPTURABLE` |
| 自带 Ascend C prepared 候选 | 拟新增 | 拟支持 | 完成真实验证后再登记 |
| Triton 实现 | 按既有入口 | 可准备编译模块、配置和辅助资源 | 需独立验证启动路径、同步与资源约束 |

默认直接函数继续选择原有调用候选，准备模式由 wrapper 显式请求；不能因为安装了 prepared 实现就改变所有直接函数的默认行为。
wrapper 请求准备模式时，只在完整准入检查通过的候选间选择。外部原生实现没有准备能力时，普通 wrapper 可以保留兼容路径，
诊断须注明“完整原生调用”；要求原生准备或要求图模式的请求不能静默降级为该路径。

为避免旧同步 runner 被新图声明错误放行，新增 prepared 执行路径使用独立 `kernel_id`，仍绑定同一个逻辑 `op`。
构建输入和设备实现可共享，但新候选的产物/spec、执行 ABI、运行身份及图证据独立绑定；不复用旧候选的冻结结果冒充新执行契约。
调用类型由有默认值的类型化声明及生成工具统一处理，不让作者在多处手写候选过滤条件。
旧入口继续走现有 `run_selected`；prepared 候选只经通用会话执行，缺少准备状态时拒绝调用。

新增生产候选及 ABI 扩展实施时，先提供精确登记集合、冻结协议与兼容测试补丁供审核；本方案不修改现有断言或冻结常量。
图声明只在对应候选、SoC、ABI 和输入范围取得证据后升级；原同步候选继续保持原声明。

## 4. 执行会话与后端协议

现有 `runtime.plan.Plan` 是选型结果，不能直接把它当作已经持有原生计划的对象。
新增 `PreparedExecution` 组合选型结果、后端私有状态、资源集合及执行约束，保留模块和设备镜像的强引用。
登记表中的 runner/工厂只持有可共享绑定；每个 wrapper 的 `prepare` 创建独立执行状态。
不得把可变的“最近一次计划”写回全局 descriptor，避免不同 wrapper 相互覆盖。
以下是协议语义，具体 Python/C++ 签名在实施时按冻结规范落地，不代表当前可调用 API：

```python
backend.prepare(request) -> BackendPrepared
BackendPrepared.validate_update(metadata) -> StagedUpdate
BackendPrepared.commit_update(staged, stream) -> Completion
BackendPrepared.run(inputs, outputs, stream) -> Completion
BackendPrepared.release_after(completion)
```

- `request` 包含已校验的完整输入规格、数学属性、必要元数据、选定实现、硬件和调用方资源要求。
- 准备状态记录 kernel/产物/ABI 身份、dtype/layout/stride、容量与必要地址约束、metadata 布局、workspace、tiling/调度结果及所属设备。
- 普通执行按后端契约允许新 Tensor 绑定；只能固定地址执行的后端须明确拒绝换绑，不能默认缓存带旧指针的 executor。
- 模块与不可变编译结果可以共享；可变 workspace、页表和计划缓冲按会话归属，默认单写者、串行使用。
- 先完整校验更新，再发布到设备；已发生部分设备更新而失败时，会话失效，不能沿用半更新状态。
- 不把 shape 相同当作 metadata 内容相同。带输入值约束的算子必须在合法更新入口校验数据；不能靠跳过既有校验获得较低耗时。
- `run` 保留本次绑定与资源检查、必要的设备计数器/临时状态重置以及启动工作。需要每次重置的数据不能只在 `prepare` 初始化一次。

## 5. 普通 wrapper 工作流

构造只保存配置和调用方资源。`plan` 完成公共校验、完整候选选择、模块准备和原生规划；`run` 复用会话执行。
某些规格只有拿到真实 Tensor 才能确定时，显式准备/预热阶段补全，不能在 active capture 中首次完成这些工作。

```text
构造 → plan(本步结构) → run(第1层数据) → run(第2层数据) → …
     → plan(更新后的结构) → run(第1层新数据) → … → 安全关闭
```

同一步中各层只有在 dtype、数学选项和调度契约一致时共享计划；不同结构或并发执行使用独立会话。
普通模式重规划可选择新候选并调整资源；旧资源在所有引用它的执行完成后释放。
没有可复用准备结果的普通算子仍使用直接函数。保留 `SegmentGEMMWrapper` 的 run-only 契约，不增加上游不存在的强制 `plan`。

## 6. 自带 Ascend C AOT 的具体改造路线

### 6.1 抽取可复用 planner

把现有 `gert::TilingContext` 回调中的可复用算法抽取为独立规划函数，输入规格、属性和 SDK 平台信息，输出：
设备入口/tiling key、block 配置、tiling 字节、workspace/系统空间需求、辅助数组布局及每次执行所需的初始化说明。
既有 OpDef/ACLNN tiling 回调和新增 prepared 接口调用同一规划实现，防止两套算法长期分叉。
数据相关规划明确其输入来源与更新规则；不假定所有 tiling 都只依赖 shape。
A2/A3 复用 arch22 源码与 runner，但准备状态仍绑定实际 SoC、独立产物和资源；不能将 910B/950 wheel family 当作执行身份，也不能跨设备复用原生 handle。

### 6.2 新增可校验的 prepared AOT 交付目标

保留旧 `op_package` 及其既有 schema/加载规则；另建带独立版本的 prepared 交付适配器，承载 planner、
设备镜像、真实入口与参数布局、tiling schema、AIC/AIV/MIX 启动属性以及依赖身份。
现有包内已包含二进制不构成“可直接启动”的证据；须验证完整启动 ABI 和计算等价性后才能复用该镜像。
不能放宽旧验证器来接受任意 `.so`、路径或函数指针。

准备执行的标准启动路线采用 CANN 的独立二进制加载方式：验证产物后加载 binary、解析 function handle，
随后按已审核 ABI 在当前 NPU stream 上启动。SDK 提供 `aclrtBinaryLoadFromFile`、`aclrtBinaryGetFunction`、
`aclrtLaunchKernelWithHostArgs` 等接口；具体镜像、参数打包和混合核启动配置须先完成实机原型验证。
实施时逐一核对目标 SDK 的声明、动态符号与版本约束；接口存在不等于设备镜像、驱动组合或捕获内启动已经验证。
原理参考 [CANN 自定义 kernel 加载与启动](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/others/acldevg/runtime_doc_dev_0019.html)。

优先产出满足独立启动契约的镜像；复用 op-package 内镜像作为后续可选适配，二者都经过同一 prepared 产物验证。
GMM 的 MIX AIC/AIV、跨核同步、系统 workspace 和入口组合单独验收，不套用 AIC-only 启动假设。
原型未通过的算子保留旧 ACLNN 路径，不计为 prepared 完成。

### 6.3 原生准备与执行桥

新增受验证产物约束的 native handle，拥有加载模块、planner 输出、参数布局、tiling 设备存储和资源引用。
wrapper 调 `prepare` 后，`run` 绑定当前 Tensor，在合法 stream 上提交必要重置和 kernel；
该路径不逐次走旧 `GetWorkspaceSize` 调用链，也不重新生成已经冻结的 tiling。
不得把现有 `aclOpExecutor*` 直接缓存为这个 handle；其复用规则未验证。

旧桥的同步与临时资源释放成套保留。新桥单独实现异步调用的资源租约、完成事件和异常清理；
不能只把旧 `RunOpApi(..., true)` 改成异步而立即释放描述符、workspace 或镜像。
现有桥显式使用 `RunOpApi(..., true)`，参见 [ACLNN 桥](../../csrc/aclnn_torch_adapter/adapter.cpp)；
新桥必须单独验证捕获内启动以及捕获外事件管理，不沿用旧桥可入图的假设。
编译、反序列化、SDK 初始化和 binary 装载在预热完成前结束；捕获期间不做冷加载。

## 7. CANN 与 Triton 接入

| 后端路径 | 适配策略 | 不得外推的结论 |
| --- | --- | --- |
| 一步式 TorchNPU/CANN | 保留 callable；wrapper 复用可控的页表/参数适配；完整原生调用仍在 run | 不声明原生 tiling/执行器已经前移 |
| 外部提供 prepared API | runner 封装其准备对象、资源、更新和失效约束 | 不假定能重新绑定任意地址或跨 stream 并发 |
| 外部提供已验证图接口 | 在固定 out/workspace 及限定参数更新契约下提供 capture adapter | 图可用不等于普通 run 已消除原生准备 |
| Triton | 在捕获外完成 JIT/配置选择/预热，持有模块、启动配置和辅助存储；执行使用当前 stream | kernel 编译成功不等于启动器可捕获；逐次同步和临时准备仍须检查 |

现有 CANN FIA 单 decode 的 Host 参数更新复用 `ReplayParameterUpdates`，保持其有限环境、shape 与一次更新凭证规则。
现有公共 single-decode wrapper 的单图/单次录制护栏保持；通用多调用点能力通过新增协调器及专项测试建立，不能直接删除旧护栏。
batch CANN 当前逐请求 gather/FIA/concat；需要单独改造真实 batch runner/固定资源执行，不能仅增加 wrapper 就计为已对齐原生批量调度。

## 8. NPU 图工作流、元数据与寿命

图兼容性是选定执行路径和资源契约的能力。AOT、prepared 或公开 wrapper 都不能单独证明它。
沿用现有 `GraphCapability`：无证据为 `NOT_CAPTURABLE`；无需改写 Host 任务参数的有限范围可验证为 `CAPTURE_SAFE`；
需要已录制任务参数更新的路径使用 `NEEDS_PARAM_UPDATE`。设备 metadata 内容更新不自动归为 Host 参数更新。
现有 `GraphReplaySpec` 的 `none/host_kwargs` 策略保持语义，设备资源更新由 prepared 资源契约描述；如需扩展序列化协议，新增版本并保留旧验证强度。

### 8.1 捕获与重放

以下为拟提供的调用关系，省略具体 API 签名：

```text
调用方创建图兼容 wrapper
捕获外：plan → 绑定输入/输出资源 → 预热
调用方开启 torch.npu.graph
    GraphSession 记录各调用点；执行各层 wrapper.run
调用方结束 capture

每个生成步：
    更新固定输入中的数据
    wrapper.plan(本步 metadata)       # 捕获外，仅提交兼容更新
    GraphSession.before_replay       # 建立更新与重放顺序，处理必要任务参数更新
    调用方 graph.replay              # 提交原图；不重新执行各层 Python run
```

图捕获后，同一 wrapper 的 `plan` 只允许已验证的兼容更新，复用原存储与执行路径；不重新选 backend、换 kernel、
重建 module 或扩大资源。改变被冻结的规格、tiling key、block 配置、地址/stride 或容量时，明确要求另建会话和图。
上面的更新与协调流程是目标协议。当前受限 `SingleDecodeWithKVCacheWrapper` 在捕获后禁止再次 `plan`，只允许通过它的 `before_replay` 和 `replay` 更新长度并重放；不能直接套用这里的未来调用序列。
规划结果写入设备内存能够支持变化的前提，是 kernel 已按该数据处理动态工作量；不能仅增加一个长度 buffer 就声明动态 shape 支持。

### 8.2 固定存储与更新提交

- 框架或调用方持有固定 metadata、tiling、workspace、输出等存储；约定所有权、别名与图中使用范围。
- `plan` 在捕获外校验并生成新调度数据，原位更新持久设备缓冲，保持图已绑定的地址与布局。
- CPU/pinned staging 也按异步拷贝完成事件管理，不能在传输完成前重写或释放。
- 同一流利用执行顺序；跨流使用事件：更新流等待前次使用完成，重放流等待本次更新完成。
- 更新全部数据后才发布新 generation；失败使会话/图失效。不得在重放中途并发覆盖其正在读取的 metadata。
- 静态图在输入与 metadata 均未变时可以重复 replay，无需强制造一次更新；已有 Host 参数更新路径仍要求原契约的逐次凭证。
- 保留必要的设备状态重置任务。图中临时计数器、同步标志和 scratch 状态若需每次初始化，其重置必须随每次 run/replay 执行。

### 8.3 多层、多个 wrapper 和退出

GraphSession 以“捕获调用点”记录任务，不能只按 kernel_id 或 wrapper 对象去重：一个 wrapper 可能跨层调用多次。
同一逻辑计划的 metadata 可以一次更新供多层使用；Host 任务参数则按录制顺序逐任务更新，并校验其输入绑定。
共享 workspace 仅允许已验证的串行使用或显式资源租约，默认不支持并发重入。

图级状态按“可准备 → 已准备/预热 → 录制中 → 已捕获 → 执行中 → 可再次重放”管理。
`plan/run` 的普通会话与“被图引用”状态分别管理，异步未完成或仍被图引用的资源不能释放。
关闭时在捕获外等待有关完成事件，并确保调用方已弃用/reset 相应图，再释放引用；模块卸载晚于最后一次设备使用。
调用方绕过协调器直接使用原生图对象时，本库无法拦截全部生命周期误用，这属于明确的使用契约。

## 9. 首批算子范围

| 算子 | prepared 首批范围 | 图首批范围与后续条件 |
| --- | --- | --- |
| 自带 matmul | 固定 M/N/K、dtype/layout；复用 shape 驱动 tiling；保持已有能力边界 | 在匹配 Ascend950 上先验证固定规格；动态规格另建图或单独开发容量/边界契约 |
| matmul_bias_silu | 固定数学属性、规格与资源，验证 bias/输出别名 | 固定规格、已声明目标平台；不可由普通 matmul 的结果外推 |
| grouped_matmul_swiglu_quant | 已登记 A2/A3，共用 arch22；按对应 SoC 验证固定 M/N/K、专家数量、量化模式和属性；复用 tiling 与非零 workspace | 先验证固定规格；group_list 值更新经合法 plan 校验并写入自有缓冲，仍须满足前缀单调、末项等于 M；保留 A8W8/A8W4 和 MIX 约束 |
| CATLASS single prefill | 现有 batch=1、dense、无 mask 的支持范围，复用规划与系统空间 | 先固定 shape；现有 kernel 没有通用运行期有效长度输入，动态长度/mask 要扩展设备语义并重新验证 |
| Paged/ragged batch attention | 统一 wrapper 和后端计划契约，接真实批量执行或经验证的固定资源组合 | 每条真实 runner 分别验证 metadata 更新与图执行；不沿用当前 composite 的未验证声明 |
| Sparse FA | 当前只有 A2/A3 arch22 Ascend C 注册；先定义这条实现的准备与复用边界 | CANN/Triton 是后续等价实现目标；每条实现的 eager、prepared、graph 分开验收，依赖可用和编译成功均不能代替图证据 |

GMM 的当前公共值检查包含 `group_list.cpu().tolist()`。有状态入口把检查移到合法准备/更新阶段，使用受管理的 metadata 副本；
普通直接接口保留原校验。不得仅凭 Tensor 地址、shape 或 Python 对象相同而跳过内容校验。
框架不承诺自动感知外部 DMA 或任意原位写入；metadata 的修改必须遵守会话更新契约。

按目标平台分别验收：GMM、Sparse FA 和 CATLASS prefill 各有 A2/A3 变体；matmul、matmul_bias_silu 及另一个 CATLASS prefill 变体面向 A5。
910B 与 950 wheel 的交付分组不改变上述范围。同一份 arch22 源码不能替代 A3 验收，A2 编译或运行结果也不能替代 A5 图验证。

## 10. Tuning、选型与身份

`plan` 使用当前完整准入规则和有效调优记录选择执行候选，普通运行不隐式开始性能采样。
call、prepared-eager、graph replay 三种边界的记录分别保存；更换执行 ABI、候选、软件或资源模式后旧记录失效。
其中 call 采用现有 [显式离线调优](../tuning.md) 机制；prepared-eager 与 graph replay 的专用计时、记录和失效绑定属于本设计扩展，当前没有对应交付。
首次准备、编译、输入恢复、metadata 更新和采样边界明确记录，不能把设备计时与同步 wall time 混为一项。

图捕获后冻结候选与 tactic，新调优结果不自动替换旧图的 kernel。需要应用新结果时，在捕获外新建计划和图。
准备状态绑定全部编译输入、host planner、设备镜像、启动属性、descriptor/能力和 ABI；摘要由框架生成，作者不手填。
生成的索引和证据元数据进入构建目录/安装包；独立审核的生产预期与 ABI fixture 继续留在测试侧，不由生成器覆盖。

## 11. 作者职责与通用框架职责

| 维护者 | 需要提供 | 框架承担 |
| --- | --- | --- |
| 逻辑算子作者 | 定义、独立样例；状态化算子的元数据含义、复用条件、更新规则及必要 wrapper 语义 | 公共导出、通用检查、会话和资源状态管理 |
| Ascend C 实现作者 | 设备源码、共享 planner、必要启动适配、自定义 CMake 选项 | 编译/冻结/产物验证、装载、通用执行桥与图协调 |
| Triton/CANN 适配作者 | 自身 runner、可选准备对象适配、依赖和有效能力范围 | 相同逻辑定义下的发现、准入、选型、诊断与测试入口 |
| 框架维护者 | 通用协议、生成器、加载器、桥、能力证据与护栏 | 新算子无需重复编写这些公共文件 |

数学规划和动态 kernel 语义无法仅从 schema 自动生成。现有算子的共享 planner 抽取由框架整改完成，后续接入使用同一模板。
保持普通用户聚焦核心代码，但不隐藏必须由实现方明确的复用和并发边界。

## 12. 实施顺序与退出条件

| 阶段 | 主要改动 | 退出条件 |
| --- | --- | --- |
| 契约冻结 | 本方案、每算子的执行/资源矩阵、独立 ABI 和生产清单补丁 | 原有接口与能力无隐式变化；必要协议/清单扩展按仓库规则审核 |
| 原生启动原型 | prepared 构建适配器、共享 planner、native 异步桥；先做匹配设备的 GMM | 原生启动与旧 ACLNN 数值一致；MIX、workspace、重置和流寿命通过；不预先承诺图支持 |
| 普通 wrapper | 通用 PreparedExecution、算子 wrapper、完整准入和候选隔离 | 一次 plan、多次真实 run；兼容更新和失效正确；稳态不再调用旧准备链 |
| 固定资源图 | GraphSession、预热、固定资源与捕获准入 | 真实一次 capture、多次 replay；无捕获内编译/装载或不兼容同步；数值与资源正确 |
| 动态 metadata 与组合 | 兼容 plan 更新、跨流排序、多层多调用点和资源租约 | 页号、长度/分段、输入内容变化读取新值；多 wrapper 顺序和越界拒绝通过 |
| 后端与算子扩展 | 950 原生算子、Triton、CANN 原生 prepared/图适配、batch 和 Sparse FA | 按各自平台矩阵独立验收；未满足者保持保守能力声明 |
| 发布与场景验收 | 干净安装、普通/图两类场景、同口径测量 | 约定矩阵全通过、0 skipped，源码/包/证据一致；框架与算子提交分开审查 |

这些是后续实现的依赖顺序，不表示当前已交付；各阶段单独提交可审核的实现、测试与平台证据。

## 13. 验收：同时检查机制、正确性与效果

1. **准备结果确实被消费**：原生计数显示一次 prepare 后多次 run 使用同一有效计划；实际设备执行、输出和独立 oracle 对齐。
   prepared 稳态中旧 `GetWorkspaceSize`、模块加载和编译调用数为 0；不把这等同于普通 eager 的所有 Host 操作归零。
2. **更新确实生效**：更换数据内容、页号和有效长度，重放原图得到新结果；覆盖稀疏/分组等相应值约束。
   负例验证漏更新、超容量、错误 dtype/layout、别名、换址、错 kernel/ABI 和已释放资源；保留必要校验强度。
3. **图确实重放**：每个承诺组合 capture 一次、至少 1000 次 replay，覆盖多次 metadata 变化；用调用计数确认重放没有重新执行各层 Python run。
   每个 SoC/backend/dtype/shape/数学选项组合单独计量，无设备/依赖/产物时明确未验证，skip 不计成功。
4. **资源寿命与并发**：验证更新/执行 stream 排序、同一流连续提交、跨流更新、共享 workspace 串行复用、并发拒绝、关闭/失败清理及稳定期资源占用。
   分配次数分别统计框架分配请求与实际设备分配，避免把 allocator 缓存命中误当成没有分配请求。
5. **调用链效果**：同一计算、输入和环境比较现有直接/eager、prepared wrapper、多层 prepared、graph replay。
   分别报告冷准备、兼容更新、稳态 Host 准备/提交（包括实际 Host 工作线程）、设备耗时和同步端到端耗时。
   动态 decode 的端到端统计包含每步 plan、输入更新、必要参数更新及 replay；计划跨层复用范围明确记录。
6. **发布可靠性**：完整回归一次覆盖 real_compile，分平台报告 0 failed/0 errors/0 skipped；编译子集从同次结果提取。
   干净 venv 安装后断言加载位置，离开源码运行自检与 prepared/graph 用例；坏包、目标错配和 JIT 开关行为保持护栏。

验收报告逐项列出哪些 Host 工作已经移出重复路径，以及移出前后的计数和耗时。
只有接口存在、缓存对象存在或固定地址不变均不足以证明优化完成；没有同口径实测前不声明性能优势或固定收益比例。

## 14. 实现依据与验证边界

- 上游生命周期：flashinfer-ori: `flashinfer/decode.py`、`flashinfer/prefill.py` 与 `include/flashinfer/attention/scheduler.cuh`；
  [FlashInfer 论文 Listing 1 与附录 D.1](https://homes.cs.washington.edu/~arvind/papers/flashinfer.pdf#page=8)。
  动态 decode 可以每步在捕获外更新计划，再重放固定图；计划和图并不互相替代。
- 图运行时语义：[CANN ACL Graph 概述](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/acldevg/runtime_doc_dev_0045.html)。
  运行时产品/版本的实际能力仍以目标 SDK、驱动和设备验证为准。

| 核对项 | 代码或测试入口 | 能证明的边界 |
| --- | --- | --- |
| 已选实现与 AOT 加载 | [Plan](../../flashinfer/runtime/plan.py)、[原生绑定缓存](../../flashinfer/aclnn/_native.py)、[ACLNN 桥](../../csrc/aclnn_torch_adapter/adapter.cpp) | 选型、绑定与每次普通执行；不证明原生准备复用 |
| 图声明、证据和参数映射 | [声明约束](../../flashinfer/runtime/schema.py)、[能力测试](../../tests/framework/test_graph_capability.py)、[更新协议测试](../../tests/framework/test_replay_param_runtime.py) | schema 与模拟更新顺序；不证明真实设备 capture/replay |
| 现有受限 FIA 图路径 | [范围声明](../../flashinfer/cann/attention/_shared.py)、[底层真机测试](../../tests/framework/test_replay_param_npu.py)、[公共 wrapper 真机测试](../../tests/framework/test_single_decode_graph_real.py) | 只覆盖声明内的 Ascend950 组合；在匹配设备与依赖环境执行后才形成运行证据 |
| 固定 metadata 与 Host 身份 | [图资源](../../flashinfer/attention/execution_identity.py)、[Tensor 计划测试](../../tests/framework/test_attention_graph_plan.py)、[身份测试](../../tests/framework/test_attention_execution_identity.py) | 资源冻结、更新和结构兼容；不等价于真实 batch 图 |
| 资源租约 | [存储租约](../../flashinfer/attention/storage_lease.py)、[租约测试](../../tests/framework/test_attention_workspace.py) | Host 所有权与状态规则；不等价于已完成异步设备寿命验证 |

上游、SDK 文档和现有 Host 测试提供设计依据，不替代目标实现的设备证据。本文不记录本机路径、一次测试耗时或历史通过数字。
通用 prepared、异步原生桥、多 wrapper 图协调和性能效果均未验证；实现时按上述验收条件重新产生证据，不复用历史全量测试数字。

相关文档：[特性设计](README.md)、[算子接入](../development/adding-an-operator.md)、[编译与交付](../ascendc_aot.md)、[规划与差距](../roadmap.md)。
