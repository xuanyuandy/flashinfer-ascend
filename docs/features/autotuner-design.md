# Ascend AutoTuner 设计

面向推理框架集成者、后端开发者和框架维护者：本文说明自动调优的候选、测量、身份与执行交接，给出从现有离线调优扩展到 kernel/tactic 选择的约束和验收条件。

状态：拟实施方案。本文定义新增 AutoTuner 的目标链路，不代表接口、tactic 搜索、prepared 或图调优已实现。
第 2–11 节描述目标设计，其中新增类型、上下文参数与执行模式不是当前公共 API。现有接口见 [离线调优指南](../tuning.md)，术语沿用 [架构](../concepts/architecture.md)。

## 1. 目标与已有基础

AutoTuner 在显式预热阶段，比较同一逻辑算子的合法 `(kernel_id, tactic)`，保存有适用范围的测量结论。
正常执行复用选择；没有有效证据时使用明确的默认策略，不把默认优先级称为性能最优。
一次预热调用可以触发多个候选的重复测量，不是每个实现只执行一次。
`kernel_id` 标识已登记实现，tactic 是同一实现内部可控且合法的配置；两者都参与目标选择身份。编译缓存、测量记录、运行期准备对象分别管理，不能互相替代。
本文的“v1/版本 1”指本仓现有 `schema_version=1` 调优记录；“上游 v1/v2”指参考实现的两条接口路径。

| 当前基础 | 保留并复用 | 需要补齐 |
| --- | --- | --- |
| `operators`、后端实现目录与 `KernelRegistry` | 公共语义、自动发现、独立实现身份、惰性 runner | 可选的调优适配协议；不再为每个算子手写候选列表 |
| `runtime/dispatch.py` | 完整准入、显式覆盖、来源栈与确定性排序 | 提取共享候选快照，供选型与调优共同使用 |
| `tuning/benchmark.py` | 显式采集、原始样本、噪声与显著性判断 | 多 tactic、执行模式、正确性门禁和预算编排 |
| `tuning/store.py`、`identity.py` | 本地优先、只读查询、身份失效及无冠军证据 | 完整软件/执行身份、新版本 tactic 记录、强制重调 |
| 公共执行与 wrapper | 现有调用签名及生命周期 | 同时消费 kernel 和 tactic；连接未来 prepared 执行链 |

当前采集器由调用方提供 `BenchmarkCandidate`，没有通用的上下文触发与内部 tactic 搜索。
发布调优清单为空；`FLASHINFER_NPU_ONLINE_TUNING` 有配置读取，但未接通自动搜索。
源码全局有 93 个 Descriptor、49 个逻辑 op；11 条生产 Ascend C 均为 `NOT_CAPTURABLE`。只有受限 CANN single-decode 声明 `NEEDS_PARAM_UPDATE`，不能外推为通用图调优。
现有机制见 [测量与存储](../tuning.md)；原生准备和图生命周期见 [Wrapper 与 NPU 图模式设计](wrapper-and-aclgraph-design.md)。源码全量目录与分芯片 wheel 的裁剪目录分开统计，登记不等于设备验证。

| 核对项 | 当前行为 | 设计扩展 |
| --- | --- | --- |
| 候选准入 | 采集器检查显式候选的 capability/target；不执行 dispatcher 的 backend、JIT、custom/reference 策略及 admission | 从同一准入快照构造测量集合，避免测量与实际选型不同 |
| 选择与执行 | `query_tuning_record` 返回冠军 ID 或 `None`，`DispatchDecision` 与 `run_selected` 没有 tactic 字段 | 新执行交接完整消费 `Selection`；保持旧接口契约 |
| 比较集合失效 | v1 校验记录中的每个比较者；新增候选不会自动使旧结论失效 | 绑定完整准入候选快照，集合变化后重新比较 |
| 软件与产物身份 | `runner.tuning_identity` 是可选钩子；发布导出校验 Descriptor/Capability，没有调用该钩子统一重算执行身份 | 在准备阶段生成完整软件、runner、产物和 tactic 身份，并统一查询/导出验证 |
| 损坏与无冠军 | v1 拒绝损坏或失配行后继续查较早有效记录或随包层；最新有效 `winner=null` 抑制旧冠军 | 新协议单独定义拒绝、strict 与默认选择，不改写 v1 读取语义 |

当前调用链是显式提供候选 → `collect_tuning_record` → `save_tuning_record` → dispatcher 查询 → 执行一个 runner。
正确性、输入恢复和完整候选筛选由采集方先完成；采集器不是统一 AutoTuner，也不自动发现同一实现的可调配置。代码与测试依据见 [验证边界](#12-代码依据与验证边界)。

## 2. 架构与文件职责

AutoTuner 位于后端之外。公共层定义计算，registry 提供实现，runner 解释配置，测量服务比较实际调用。
以下为目标文件位置；标注“新增”的文件当前不存在。原生 kernel、CMake 和后端 runner 保持现有组织。

```text
flashinfer/
├── __init__.py                              # 导出拟新增的 autotune 上下文
├── gemm.py / decode.py / prefill.py          # 原公共函数、wrapper；统一执行交接
├── operators/
│   ├── runtime.py                          # 通用接口接入调优执行交接
│   └── <domain>/<op>/
│       ├── operator.py                     # 公共语义与必要的负载特征规则
│       ├── cases.py                        # 独立数值样例、调优输入及恢复规则
│       └── wrapper.py                      # 按需接入；原有 Attention 前端继续复用
├── runtime/
│   ├── registry.py / dispatch.py           # 唯一登记表、共享准入和默认选择
│   └── runner.py                           # 保留旧 run_selected 交接
├── tuning/
│   ├── autotuner.py                        # 新增：上下文、搜索、预算与结果发布
│   ├── protocol.py                         # 新增：类型化请求、tactic、选择与适配协议
│   ├── candidates.py                       # 新增：从 registry 构造候选快照
│   ├── runtime.py                          # 新增：选择 kernel+tactic 并交给执行适配
│   ├── measurement.py                      # 新增：NPU 测量会话与三种执行模式
│   ├── autotune_store.py                   # 新增：独立版本的 tactic 测量记录
│   ├── benchmark.py / identity.py          # 复用统计分析，扩展自动身份生成
│   └── store.py / _release.py              # 保留已有版本记录及发布契约
├── ascendc/<domain>/<op>/runner.py          # 已构建变体、可选准备对象与启动
├── triton/<domain>/<op>/runner.py           # 合法配置、编译准备与启动
└── cann/<domain>/<op>/runner.py             # 默认原生调用或公开的可调配置
tools/
└── tune_flash_attention.py                 # 改为调用通用引擎的场景驱动
```

| 边界 | 职责 | 不能代替的工作 |
| --- | --- | --- |
| dispatcher | 判断实现是否准入，提供默认候选与解释 | 不执行编译、正确性实验或计时 |
| AutoTuner | 在准入集合内搜索、复用记录并返回选择 | 不修改数学语义、能力或图证据 |
| 测量会话 | 私有输入/资源、准备、计时、恢复与完成事件 | 不把调用方模型的 capture 用作试验场 |
| 后端适配器 | 枚举合法 tactic，准备并执行指定配置 | 不各自复制缓存、统计和搜索循环 |
| 执行交接 | 验证并消费完整选择 | 不能只执行 kernel_id 而丢掉 tactic |

## 3. 公共使用方式与触发规则

下面是目标用法，当前不可调用。启停形式参考上游 `autotune(True/False)`；`execution_mode`、`force`、`strict`、`persistent` 是本仓设计参数，不承诺与上游同名参数兼容。执行模式默认 `eager`。

```python
with flashinfer.autotune(True, execution_mode="eager"):
    output = flashinfer.gemm.matmul(a, b)  # 预热：未命中时执行搜索

with flashinfer.autotune(False, execution_mode="eager"):
    output = flashinfer.gemm.matmul(a, b)  # 推理：仅复用记录或按规定回退
```

高级控制集中在通用上下文/策略对象，不加入每个算子的公开参数列表：

| 控制 | 目标语义 |
| --- | --- |
| `True` | 有兼容记录则复用；无记录才调优；显式调用即授权预热，不再要求额外环境开关 |
| `False` | 不测量；命中则复用，未命中按默认策略选已准备的可执行路径；否则提示先预热 |
| `force=True` | 仅在调优模式可用；绕过对应新协议的内存、本地与发布结果，完整重测一次 |
| `strict=True` | 仅复用模式下缺记录、记录失效或未准备时明确失败 |
| `persistent=False` | 本上下文不读写调优磁盘记录，只使用兼容的内存结果；编译缓存仍按既有策略管理 |
| `execution_mode` | `eager`、`prepared`、`aclgraph`；不可用模式明确拒绝，不换计时口径 |
| 测量策略 | 采样、实际收益门槛、资源预算、缓存冷热和适用范围由公共策略集中管理 |

没有上下文时保持现有调用行为和旧记录查询；不新增隐式测量，也不关闭原有允许的惰性加载。
新上下文的复用阶段不因 miss 触发冷编译；普通推理不自动发现并纠正误测。
缓存只保存配置选择，不保存可直接恢复的 executor 或原生指针。命中记录仍须检查当前执行是否已准备；
调优模式可在捕获外准备所选组合，复用模式和 capture 对未准备状态明确拒绝。
上下文状态按线程/异步任务隔离，嵌套退出恢复原状态；同一进程同一设备的测量串行进行。
进入上下文或显式加载/预热阶段完成磁盘记录装载、身份快照和覆盖预检；稳定执行只使用内存选择。`eager` 的已准备仅表示所需加载、JIT 与预热完成，不表示移除了每次原生调用的准备工作。
不在算子导入、登记、Fake/Meta 执行、编译 tracing 或 active capture 中启动搜索。
环境变量不单独在 dispatch 内启动后台或独立流试跑；旧设计中的在线探索是远期目标，不是本方案首期行为。

## 4. 完整调用链

### 4.1 首次预热与搜索

1. **公共调用**：保持原签名，完成语义/值检查和输入输出约定，形成调用请求。
2. **共享准入**：按 `op` 从同一个 registry 获取候选，应用 backend、capability、target、admission、
   JIT、custom/reference 及执行模式限制。显式 kernel 覆盖仍遵守现有分支契约。
3. **构造身份**：生成负载、执行模式、资源政策、环境和候选集合身份，先查询兼容缓存。
   发现与查询只读元数据，不先编译全部候选，不在每次调用查询编译器或重算文件摘要。
4. **测量准备**：miss 或 force 时创建私有样例，加载/编译允许的产物，枚举合法 tactic，检查资源及数值。
5. **重复测量**：默认候选也参与比较；每个合法组合预热并采样，交替/轮换顺序，保存全部样本。
6. **形成结论**：有显著且稳定优势才给出冠军；无显著差异使用确定性默认选择。
7. **原子发布**：记录通过完整校验后更新新版本 store、内存与决策 generation；失败不发布半成品。
8. **真实执行**：使用调用方真实输入输出执行已选组合；不把测量用的私有输出作为返回结果。

测量通过固定候选的执行适配绕过搜索，避免重入 AutoTuner。固定候选只关闭选型，不能免掉该调用形式实际需要的
公共检查、绑定、输出分配和后端转换。公共选择的共有开销单列测量，并在最终场景中验证。
`functional` 与 `out` 的分配行为不同，分别进入测量身份；不得通过预分配 out 美化 functional 成绩。

### 4.2 复用执行与显式控制

命中流程为：公共校验、共享准入、兼容记录/内存选择、执行交接、真实 runner。
稳定期不计时、不写库、不遍历 tactic、不生成测试输入，不读文件或启动编译；wrapper 可直接复用其已绑定选择。

- `backend=` 缩小候选集合，不越界选择其他后端。
- `kernel_id=` 锁定实现；显式调优时只搜索其合法 tactic，不能换 kernel。
- 两者同时指定时沿用现有显式 ID 分支：backend 仍做值合法性解析，不对已锁定 ID 重复按 backend 过滤；capability/target/admission 仍检查，编译策略继续由 runner 执行。
- `Selection` 在交接时校验身份、tactic 合法性、资源和模式；运行异常保持原异常，不偷偷换实现。
- 从 `operators/runtime.py` 的公共交接和已有域前端的执行交接接入同一服务，不让每个算子单独调用计时器。
- `dispatch()`、旧 `DispatchDecision` 和 `run_selected()` 的现有调用/返回契约保持；新路径通过
  `tuning/runtime.py` 传递完整选择。不得把 tactic 塞入原 `kernel_id` 字符串或写回全局 descriptor。

新版本记录不能直接回灌旧的“只返回冠军 ID”查询路径，否则配置会丢失。
既有旧记录继续按原语义消费；没有执行模式、软件身份或 tactic 身份的旧证据不能自动升级为新记录。
新上下文缺记录时使用共享的确定性排序，不借旧记录中的跨口径冠军作为默认选择。
覆盖只使用既有 API 已声明的参数；batch 固定 CANN、single Attention 对 auto 的解释等前端策略仍按 [路由契约](../concepts/backend-routing.md) 处理，不向所有接口隐式增加关键词。
共享准入必须复用实际入口的分支，不能用 manifest 的 `KernelRegistry.select()` 替代普通 `dispatch()` 的筛选结果；两者消费的输入与约束并不相同。

## 5. 实现协议与作者职责

`KernelDescriptor` 表示实现身份、ABI 与能力；`Tactic` 表示该实现内部的合法配置。
数学语义、ABI 或能力边界变化仍应建模为独立实现；tactic 不能用来绕过这些边界。

以下仅为协议语义，不是已冻结的 Python 签名：

```python
adapter.describe_tactics(request) -> tuple[Tactic, ...]
adapter.bind_execution(request, tactic, resources) -> BoundExecution
BoundExecution.run(invocation) -> outputs
adapter.prepare_trial(request, case, tactic, execution_mode) -> PreparedTrial
PreparedTrial.validate(reference) -> CorrectnessEvidence
PreparedTrial.run() -> Completion
PreparedTrial.close_after(completion)
```

生产交接用 `bind_execution` 绑定选择，再以真实 invocation 调用 `BoundExecution.run`。
测量会话在私有案例上使用同一执行适配，避免“测的是一个入口，执行的是另一个入口”。
普通绑定可以只是 callable 与配置引用，不据此声明原生计划复用；prepared 状态仍按独立协议提供。

无内部调参的 callable 自动获得 `default` tactic 和通用 eager 适配；不强制实现新的 runner 类。
默认适配不等于已通过数值验证；缺少独立样例或必要恢复规则时明确提示尚未接入调优，普通调用仍按原契约执行。
特殊算子通过可选适配器提供准备/恢复语义，不能把 opaque executor、函数指针或 callable 序列化进 JSON。
适配绑定由实现工厂统一收集，摘要和索引由框架生成。具体协议若影响冻结 ABI，实施时单独提供兼容补丁审核。

| 维护者 | 必须提供 | 不需要手写 |
| --- | --- | --- |
| 逻辑算子作者 | 公共定义、独立数值样例/参考；必要的分桶及数据恢复规则 | 候选登记副本、计时循环、SQL、手工 digest |
| 后端作者 | 真实 runner；有配置时提供合法 tactic 及指定配置的执行；必要的 CMake/设备实现 | 跨后端选择、统计判断、通用缓存和上下文 |
| 框架 | 通用发现、默认适配、案例协议、搜索、身份、存储、诊断与回归入口 | 不自动推导任意算法的 oracle、tiling 或动态语义 |

### 5.1 三后端的接入差异

| 后端 | tactic 来源 | 限制 |
| --- | --- | --- |
| CANN 原生 | 默认完整调用；公开接口确有可调配置时再枚举 | 不把无法控制的内部 tiling/执行器冒充可复用计划 |
| Ascend C AOT | 已生成、已验证的编译变体或合法运行配置 | 先构建再测量；不能用 tactic 任意修改已有二进制宏、入口或 ABI |
| Triton | runner 允许的 tile、流水等组合 | 在计时外完成 JIT；每个配置仍须通过能力和正确性检查 |

tactic 采用稳定结构化值或不随 shape/枚举顺序变化的固定表项，不存临时计划列表下标。
运行时可变计划/工作区归测量会话或 wrapper 所有；共享 runner 只保存可共享绑定，不能以“最近一次 tactic”串用不同调用。
只有一个合法组合时标记 `single_candidate`，执行该组合，不声称它击败了其他实现；旧采集器的至少两个候选护栏不放宽。

## 6. 请求、选择与适用范围

| 框架对象 | 关键内容 | 用途 |
| --- | --- | --- |
| `TuningRequest` | 逻辑输入语义、完整 shape/stride/物理布局、数学选项、设备、调用形式、执行与资源模式 | 定义可比较的负载 |
| 候选快照 | 当前准入实现集合、各自能力/执行身份、tactic 协议版本 | 防止记录漏比较新增或变化的实现 |
| `Selection` | kernel_id、结构化 tactic、适用身份、选择原因、记录来源与 generation | 被执行层完整消费 |
| 测量记录 | 请求/环境/候选、正确性证据、策略、原始样本、失败原因、统计和结论 | 审核、复用与重调依据 |

首期采用精确规格；只有声明并验收了代表性、边界与合法范围的轴才分桶。
能力范围仅说明“能运行”，不能自动成为“同一配置最快”的性能桶；不默认借用相邻桶。
新增桶不能扩大已有 capability。桶内所选组合不一致时拆桶，或退回精确规格。

shape 相同不一定是相同负载。GMM 的专家负载、稀疏 Attention 的长度/索引分布等，
由算子规则提供必要的结构事实或分类；在 plan/预热阶段生成，不在 run 中遍历整个设备数据或做 D2H 哈希。
未知特征只能保留保守范围。不得用全零、单一热点等方便构造的输入替代真实代表分布。
普通输入数据指针不进入跨进程性能 key；图内绑定的地址与寿命另由图资源契约校验。

## 7. 测量、误测与失败处理

### 7.1 计时口径

| 模式 | 排序目标及必须计入的工作 | 单独报告 |
| --- | --- | --- |
| `eager` | 重复公共调用的同步端到端耗时；包含实际 Host 准备、转换、分配和启动 | 冷编译/加载、设备事件耗时、公共选型开销 |
| `prepared` | 已准备 wrapper 的重复 run；包含每次必须的绑定和状态重置 | 初次 plan、兼容更新、跨层复用与摊销次数 |
| `aclgraph` | 私有真实图的重复执行事务，包含必要逐次参数更新、提交及完成等待 | capture、设备 replay 耗时、每步 plan/输入更新及共享次数 |

这里的 `prepared` 对应图模式设计中的 `prepared-eager`；`aclgraph` 表示 NPU 图模式。三种模式是目标计时边界，当前离线采集器并不识别这些模式，也没有通用 NPU event 或图测量会话。

目标口径、缓存冷热、重复窗口与资源模式进入记录身份，不以设备事件值替代同步端到端成绩。
prepared/图的动态生成场景另做“每步更新计划 + 多层执行”的端到端验收，避免把共享 plan 误计多次或完全漏计。
NPU event、stream、同步和图计时须独立实现、实机验证；不机械替换 CUDA API、延迟 kernel 或 L2 清理手段。

测量会话拥有私有输入、输出与工作区；预备数据/输入恢复在计时外完成，但实际调用每次需要的临时状态重置必须计入。
原位、别名、随机状态、缓存写入等由案例适配保证各样本等价，不能反复修改业务数据来测速。
编译、固定权重打包和资源准备可排除；每次输入相关打包/布局转换仍须计入。

### 7.2 统计结论与预算

保留现有 `BenchmarkConfig` 的最低轮数、实际收益门槛、稳健噪声估计、配对显著性及执行顺序检查，不降低已有护栏。
分析中的组合键由框架生成，映射到 `(kernel_id, tactic)`，不新增生产 kernel 来凑候选。
默认合法组合必须参加比较，全部原始样本、候选顺序、负载观察及失败原因均保留。

| 结果 | 选择行为 | 证据含义 |
| --- | --- | --- |
| `winner` | 使用显著且稳定的冠军 | 仅针对本次完整候选集合及测量范围 |
| `no_significant_difference` | 使用确定性默认组合 | 有效的无冠军证据，不回捞旧冠军或下层发布冠军 |
| `single_candidate` | 使用唯一合法组合 | 未进行跨组合性能证明 |
| 预算耗尽或比较不完整 | 仅可使用已验证并准备的默认组合，否则失败；不发布新冠军 | 不能把未测候选当作已经输掉 |
| 数值、ABI、产物或不可恢复运行失败 | 明确失败，不偷偷回退 | 不产生成功调优记录 |

按 op/负载限制候选数量、准备资源和调优总预算，在安全边界检查停止；不承诺抢占已经启动的设备 kernel。
对接近结果可在统一预算内做预先规定的换序确认，但不重试到出现期望冠军。
发现数值、ABI 或产物校验失败即中止；设备、必需依赖缺失必须明确失败，不能静默少测后宣称完整选优。
资源不满足的候选记录拒绝原因；运行错误按是否已验证可恢复处理，设备上下文损坏时停止继续测量。

### 7.3 如何处理误选

重复采样不能消除错误的计时边界和非代表输入。缓存合法也不能证明今天仍最快；首期不做后台持续探索。
`force=True` 完整重测并发布新 generation，即使旧记录仍然兼容也不复用；中途失败保留旧证据并返回失败，不伪造新结果。
最新有效无冠军结果压过同身份旧冠军。已有 wrapper/图保持其绑定选择，应用新结果需重新准备，必要时重新捕获。
清理分为内存、指定本地记录和调用方持有的计划；仅清内存可能重新读到旧磁盘记录，诊断须明确提示。

## 8. 缓存、身份与数据交付

新记录采用独立版本的本地 SQLite store，复用现有事务和只读查询经验；不直接改写版本 1 的记录、返回类型或校验。
新协议使用独立文件或命名空间，不能直接修改现有 `tuning.sqlite3` 的 `PRAGMA user_version` 使旧读取器失效；路径与 schema 由框架生成，不新增手写 JSON 登记表。
不需要为模仿上游把 SQLite 改成 JSON；需要借鉴的是身份隔离、原子发布、内存命中和失效语义。

查询顺序为兼容内存、有效本地证据、有效随包证据、确定性默认选择。
本地最新有效无冠军记录抑制旧冠军；损坏记录拒绝并解释，不视为无冠军。外部进程更新需显式 reload 或重启。
写入成功后统一失效新协议的查询/决策缓存；旧版本 store 保持原行为。capture 内不读取冷磁盘记录。

身份由框架从以下来源生成，作者不填摘要：

- 逻辑语义、精确规格/已验证桶、dtype/layout/stride、量化、mask、确定性等数学属性；
- SoC/核数/UB 与已验证设备范围，CANN、驱动、TorchNPU、Triton、框架和编译工具身份；
- 完整准入候选集合、tactic schema、设备源码/产物、host planner、runner/桥与启动 ABI；
- 执行模式、functional/out、计时目标、资源限制、分桶版本和有效测量策略。

任何必要事实未知时不能复用为性能证据；允许显式诊断留档，但不驱动选型。
新增/移除候选或改变任何比较参与者都使旧比较失效；只校验冠军不够。
兼容性需要的字段在准备时生成快照，稳定期不 hash 文件、探测编译器或轮询设备状态。
进程内替换包或产物须重启/显式重新准备，不承诺自动感知未登记的二进制替换。
A2/A3 可以共用 arch22 源码和 910B wheel，性能记录仍按 Ascend910B、Ascend910C、Ascend950 的实际硬件与产物分别匹配，不按 wheel 名或源码架构合并。测量与资源绑定还须核对实际执行设备；当前 `current_hardware()` 的设备 0 缓存不是通用的逐输入设备身份服务。

用户缓存与样本写到源码树外。代码仓保存协议、生成工具、测试 fixture 和证据要求，不新增人工 JSON 登记表。
随包记录仅由审核后的真实测量在构建阶段生成；普通运行不写安装目录，用户缓存不混入 wheel。
既有精确登记集合、冻结 ABI 及发布导出护栏保持，涉及新协议的补丁在实施阶段单独评审。

## 9. Wrapper 与 NPU 图模式的衔接

`plan` 完成可确定的准入和资源准备；若测量需要真实 Q/KV 等数据，则显式预热 run 补齐选择，不能假定 plan 已经测速。
普通 wrapper 将 `Selection` 和后端准备对象一起持有；run 不重搜或把新选择写进全局 runner。
原位更新必须仍满足该选择的范围。改变负载范围、候选或 tactic 时重新 plan；保留原有 run-only 接口。

图模式先完成选择、准备和预热，再由调用方 capture；图引用 kernel/tactic、输入输出和计划资源。
调优使用独立私有 capture，不嵌入调用方正在捕获的模型；`NEEDS_PARAM_UPDATE` 遵守既有更新凭证与逐调用点规则。
图内只允许已准备、内存可用且通过资源校验的选择；miss、force、编译、重规划和查询冷磁盘都明确拒绝。
真正 `graph.replay()` 不再执行 Python 选型。生成新调优记录不能替换已捕获任务；需要新计划/新图才能采用。

候选必须先有匹配范围的 graph 证据才能参与图调优；调优不能自动升级 `graph_capability`。
证据须覆盖实际 tactic 和启动配置，某个默认配置的证据不能自动覆盖同 kernel 下的全部新 tactic。
在 prepared/图基础设施落地前，只交付 eager 调优；其他模式明确标记未支持，不用 eager 数据冒充。

## 10. 诊断、并发与特殊算子

每次选择可查询：来源、kernel/tactic、范围、测量模式、候选拒绝原因、记录失效原因、采样统计、预算和选择 generation。
区分性能冠军、唯一可用、无显著差异和缺记录默认选择。命中率统计只说明复用了记录，不证明真实执行 kernel 或性能收益。
日志与统计在调用方请求后读取，稳定执行不逐次打印或导出记录。

框架管理进程内发布锁和每设备测量锁；SQLite 事务保证记录完整，不把跨进程并发测量误认为互不干扰。
多进程启动阶段由应用协调设备测量窗口；首期不提供跨节点一致选择。
含 collective 的实现只有接入专门 group 协议后才能调优，所有参与 rank 要对齐候选、缓存命中与测量顺序，
通过 HCCL/目标环境验证后再声明支持；不能让一个 rank 试跑而其他 rank 使用缓存。
特殊输入/执行模型可接专用调优协议，但仍遵守公共启停、force、存储、计时与失败规则，不允许自行隐式测速。

## 11. 实施顺序与验收

| 阶段 | 交付 | 验收重点 |
| --- | --- | --- |
| 协议与兼容 | 类型化请求/选择、上下文、共享准入、新 store；旧入口保留 | 精确集合、旧 ABI、旧查询与默认选型不变；新字段/模式有拒绝测试 |
| eager 引擎 | 默认 callable 适配、案例协议、NPU 测量、保守统计和 force | 一次预热真实运行多候选，多次复用零采样；完整选择确实交给对应 runner |
| 首批多后端 | 匹配平台的 Matmul；框架案例支持单后端多配置 | 正确性先通过；不同选择由实测产生，不要求人为凑齐各 backend 当冠军 |
| prepared 接入 | 连接准备会话、plan/run 与三种成本报告 | 准备成本与重复运行区分；旧一步式 CANN 不冒充原生准备复用 |
| 图调优 | 私有 capture/replay、必要更新与固定资源、调用方捕获准入 | 一次 capture 至少 1000 次 replay；结果正确、选择固定、寿命和更新顺序通过 |
| 发布与场景 | 已审核记录导出、包外安装验证、完整回归与真实负载报告 | 同口径证据可复验、0 skipped；未验证平台和依赖逐项列明 |

三后端 Matmul 的当前共同登记范围是 Ascend950、BF16、ND、连续 Tensor，M/N/K 为 16..256 且为 16 的倍数；必须在共同合法范围与匹配设备上验收。该范围不是三后端性能或 AutoTuner 已验证的结论。
当前 GMM 可作为匹配平台的单实现/原生准备案例；没有第二个已准入实现或 tactic 时不声称完成跨后端选优。

验收至少覆盖：

1. 候选发现、完整准入、backend/kernel 覆盖和默认选择一致；新候选不会被旧缓存漏比较。
2. 多 tactic、单候选、无显著差异、测量失败和预算停止；原始样本可重新计算结论。
3. 缓存命中不计时、不编译、不访问文件/SQL；force 真正重测；内存/磁盘/计划清理互不混淆。
4. 软件、硬件、数学属性、产物、候选、模式、调用形式、桶变化使记录失效；坏记录明确解释。
5. 私有输入与原位/别名恢复、每次状态重置和异步资源寿命；错误不会污染调用方业务 Tensor。
6. wrapper 和捕获后固定选择；图中拒绝 miss/force；图参数更新读取新内容且数值正确。
7. 统计单测的合成时钟只验证规则，真实性能实验另存命令、环境、原始样本和端到端结果，不承诺固定收益。
8. 实施跨层改造后执行一次 full，real_compile 从同次结果提取；干净 wheel 安装后离开源码验证。

首批不含自动后台重调、任意桶外外推、未验证图模式或多机调优；扩展时沿用相同准入和证据要求。

## 12. 代码依据与验证边界

上游的 runner/tactic 协议、分桶、测量和缓存分工参考 flashinfer-ori: `flashinfer/autotuner/autotuner.py`；
托管缓存、执行模式与稳定 tactic 身份参考 flashinfer-ori: `flashinfer/autotune_cache.py`、`docs/design_docs/autotuner_v2.md`。
使用方式见 [官方 AutoTuning 文档](https://docs.flashinfer.ai/autotuning.html)，
v2 的环境、计时与执行契约见 [上游设计](https://github.com/flashinfer-ai/flashinfer/blob/2895577720741ddfedff118202c89584996b8875/docs/design_docs/autotuner_v2.md)。
借鉴职责与协议，不照搬 CUDA 计时、JSON 格式或未统一的算子特例。
上述固定参考与 `tools/metadata/api_parity.json`、`attention_api_parity.json` 的上游 API 基线一致。上游 v1 的启停形式与 v2 的托管缓存是不同接口；本设计不等于完整移植或 drop-in 兼容。

| 核对项 | 代码与测试入口 | 证据范围 |
| --- | --- | --- |
| 公开离线接口与统计规则 | [接口导出](../../flashinfer/tuning/__init__.py)、[采集器](../../flashinfer/tuning/benchmark.py)、[统计测试](../../tests/framework/test_tuning.py) | 显式候选、同步 wall time、原始样本和保守分析；不是自动搜索或性能优势证据 |
| 候选与显式覆盖 | [dispatcher](../../flashinfer/runtime/dispatch.py)、[选择顺序测试](../../tests/framework/test_dispatch_ordering.py) | 完整普通准入与显式 ID 分支；不提供 tactic 执行交接 |
| 旧记录身份和查询 | [存储](../../flashinfer/tuning/store.py)、[身份钩子](../../flashinfer/tuning/identity.py)、[记录测试](../../tests/packaging/test_tuning_release.py) | 记录内比较者、可选执行身份、无冠军与拒绝行为；不是完整新候选快照 |
| 内存与决策缓存 | [缓存测试](../../tests/framework/test_dispatch_cache.py)、[记录缓存测试](../../tests/packaging/test_tuning_release.py) | 命中无文件/SQL I/O、revision/reload 失效；不是当前最快的证明 |
| 显式 FA 场景 | [FA 脚本](../../tools/tune_flash_attention.py) | 私有 registry、精确 shape、独立数值校验与 api/runner 边界；不能扩大为全算子自动调优 |
| 发布导出与包内资料 | [导出器](../../flashinfer/tuning/_release.py)、[当前清单](../../flashinfer/tuning/data/manifest.json)、[发布测试](../../tests/packaging/test_tuning_release.py) | schema、已有身份与大小/安装协议；清单没有测量 shard，不能证明实际测量来源 |

现有统计与发布单测使用合成时钟、数据或注入身份验证规则，不作为真实 NPU 性能结论。AutoTuner、通用 tactic 交接、prepared/图测量和模型收益仍未验证；实现后按匹配平台重新产生证据，不复用历史通过数字。

相关文档：[特性设计](README.md)、[离线调优](../tuning.md)、[路由](../concepts/backend-routing.md)、[图模式设计](wrapper-and-aclgraph-design.md)、[规划](../roadmap.md)。
