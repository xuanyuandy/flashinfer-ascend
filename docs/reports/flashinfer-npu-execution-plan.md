# flashinfer-npu 执行路径与逐项验收

状态：**按任务版本 `b8ba29b` 持续执行 P0/P1；操作与验证逐步留痕。**

任务与进度的唯一来源为 [`../TASKS.md`](../TASKS.md)
及其链接的 20 份 [`../docs/tasks/`](../docs/tasks/) 工作单。
本文件只细化实施边界与验收路径，不另维护一份任务状态。
最初收到的原文保留为 历史快照（过程产物，未入仓）。
原代码基线证据：reports/pre-t1-baseline.json（过程产物，未入仓）；
此次任务核对：reports/task-reconciliation-b8ba29b.json（过程产物，未入仓）。

本次 pull 只增加任务文档，未改实现和测试。20 份任务正文均可在原始
工作单中逐字找到；新增内容是执行序号、单文件工作单、进度框与验收记录模板。

## 1. 执行与验收约定

用户最新授权：「按照你的这个计划一直执行，每一步修改的操作都要留痕，先把前几个阶段执行一下」。该授权替代此前逐步等待验收的方式。

1. 按索引顺序持续推进 P0/P1，不再逐步暂停请求确认。
2. 每组修改前后由 `tools/audit.py begin/end` 保存差异；
   验证由 `audit.py run` 保存命令、退出码和完整输出。
3. `reports/execution/events.jsonl` 是按时间排列的操作记录；`steps/` 保存补丁和日志。
4. 每个任务独立提交，中文标题带任务号；不混入前序 A5 未提交修改。
5. 缺少来源资料或真实硬件证据的项目明确记录，不编造覆盖关系或通过结果。
6. 进度中区分「代码实现」「自动验收」「用户复核」。持续执行授权不等于用户已逐项验收。

## 2. 当前环境与基线核对

| 项目 | 当前证据 | 对执行的影响 |
| --- | --- | --- |
| 本仓 HEAD | `b8ba29b`，自 `a65a89e` 起只新增 21 份任务文档，另有前序 A5 未提交改动 | 实现基线未被此次 pull 改动；保留既有差异 |
| 运行环境 | `flashinfer-a5`，Python 3.11、torch 2.10.0+cpu、torch_npu 2.10.0.post7、triton-ascend 3.6.0 | 后续验收使用同一环境；NPU 数值验证使用真实 A5 |
| Python 文件 | 包内 104 个；当前测试文件 99 个，HEAD 中为 97 个 | 验收动态枚举文件，以零导入/收集错误为准 |
| 当前导入 | 实测 71 个模块导入失败 | 原单中的「72 个」作为原始快照保留，执行依据实测列表 |
| 当前测试收集 | 211 个测试可收集，87 个模块收集报错 | P0 后首次取得完整测试结果，再按任务归因 |
| 上游 checkout | `/home/dy/projects/flashinfer-ori` 当前 HEAD 为 `da992ed9919604c5eb6dc255c63f5e89a466452c` | 不使用移动 HEAD 对标 |
| 指定上游对象 | `2895577720741ddfedff118202c89584996b8875` 已在本地 | 用 `git show` 或独立只读工作树固定读取该提交，不改上游工作区 |
| 覆盖账本 | 通用 33 条；Attention 34 条，快照日期不同 | T3 先补有证据的差异条目；T5 再导入完整覆盖表并统一基线 |
| 147 条覆盖表、配套架构说明、vllm-ascend 检出 | 当前已检查的位置尚未找到，已请求路径 | 不影响 T1；T5/T3 和依赖 vllm-ascend 的任务需先取得资料 |

原清单中的模块数、行号、接口数和调用点数都是定位线索，实施时按指定
commit 和当前源码重新核实，不通过添加占位条目满足数量阈值。

### 已核实、可用于后续任务的本地 API 信息

- 指定上游提交确有 `RoutingMethodType.TopK=5`、`Unspecified=10`
  及返回 `RoutingMethodType.<成员>` 的 `__repr__`。
- 本机 `npu_top_k_top_p_sample` 的 schema 暴露 `min_ps`、
  `input_is_logits` 和 `generator` 参数。这是接口证据；MinP 的筛选顺序、
  边界与上游是否一致，仍需 T2.1 数值验证。
- `npu_scatter_pa_kv_cache` 暴露 `slot_mapping`、两个可写 cache 和
  `cache_mode="PA_NZ"`。T2.6 必须核对 ND/NZ 与上游页布局映射。
- 本机可见 FIA v1/v2 的最大 workspace 查询；FIA 的 `.out` schema
  另有显式 `workspace`。T10 要核对执行确实使用调用方 workspace，
  不能把「能够查询大小」当作「重放时不会重新分配」。

## 3. 按新版索引执行

以下顺序直接采用 `docs/TASKS.md` 的执行序号，不沿用上一版计划的 P1 重排建议。
各任务及其子任务分别记录自动验收结果，供用户复核。

| 阶段 | 顺序 | 阶段出口 |
| --- | --- | --- |
| P0 | T1 | 全模块可导入、全测试可收集；交付完整失败清单 |
| P1 | T3 → T2.1 → T2.2 → T2.3 → T2.4 → T2.5 → T2.6 → T4.1 → T4.2 → T4.3 → T4.4 → T4.5 → T4.6 → T5 | 先界定差异，再接接口、修既有缺陷，最后完善账本和 CI |
| P2 | T6.1 → T6.2 → T6.3 → T6.4 → T6.5 → T7 | 登记表能描述产物、外部来源、规模、物理限制与图能力 |
| P2+ | T18 → T19 → T20 | 真实编译/加载、两级选择、实测数据路由形成闭环 |
| P3 | T8 → T9 → T10 → T11 → T12 | 编译追踪、固定 buffer、最大 workspace、参数更新和 aclgraph 验证 |
| P4 | T13 → T14 → T15 → T16 | 先构建与注册，再逐个下沉纯张量算子 |
| P5 | T17 | Attention 迁回统一登记和执行链路，保留已验证契约 |

完整父任务顺序为：

```text
T1 → T3 → T2 → T4 → T5 → T6 → T7 → T18 → T19 → T20
   → T8 → T9 → T10 → T11 → T12 → T13 → T14 → T15 → T16 → T17
```

T3 先于 T5：T3 只补齐或标注可由固定上游接口和覆盖表确认的 NVIDIA
绑定条目，并记录 reason；完整 147 条导入、统一基线和 CI 归 T5。
当前通用账本只有 33 条，T3 因而包含新增有证据的条目，不能只改现有项。
若缺少足够的条目依据，报告缺失来源，保留待核实状态；继续不依赖该来源的工作。

## 4. 首步 T1：先还原契约，再解除导入阻塞

### 4.1 先交付契约对照表

动实现前收集 `flashinfer/`、`tests/`、manifest/schema 和 ABI 文档中
每个符号的构造、属性、序列化、校验与指纹使用点，形成：

```text
符号 → 定义位置 → 字段/默认值 → 不可变性 → 校验规则
     → 序列化形式 → ABI/指纹要求 → 使用点与现有测试
```

当前 Git 可达历史中，前一个提交 `5857cc4` 不含这些 runtime 文件，
没有可直接取回的旧实现。因此以调用点、已有测试和规范共同还原，
存在冲突时明确登记，不通过改变冻结断言掩盖差异。

### 4.2 T1 的实际依赖闭包

原单 10 个符号不足以达到「全部可导入/可收集」。已发现还需覆盖：

- `WorkloadSpec`、`DeviceCapability`；
- `CPrimitive`、`CFieldABI`、`CStructABI`；
- `KernelArgumentABI`、`KernelArgumentDirection`、`KernelArgumentPassing`；
- `KernelErrorABI`、`KernelErrorCodeABI`、`ArtifactVerificationError`；
- 已有 `load_kernel_manifest` 的导出，以及测试所用 `build_plan`；
- CLI 所导入但当前缺少的 `format_config`；
- `flashinfer/__main__.py` 当前顶层直接执行 CLI 并抛出 `SystemExit`。
  修复 CLI 导入后，原单的全模块扫描仍会被它终止，需要入口保护。

这些属于已授权的 T1 验收依赖闭包。
建议允许的实现位置以 `runtime/schema.py`、`runtime/__init__.py` 为主，
必要时包含 `runtime/plan.py`、`config.py`、`__main__.py`；若还需其他
文件，记录实际依赖证据及范围。

### 4.3 保留现有执行链路，控制与 T6 的重叠

当前 CANN/Triton descriptor 使用 `capability`、`compile_mode`、`runner`；
旧 manifest/Attention 代码使用 `op`、`constraints`、workspace、artifact
与 ABI 字段。T1 先还原被现有代码实际消费的契约，并明确两种构造形式
如何兼容，保持 A5 已跑通的路径可用。

T1 不引入 `EXTERNAL` 身份、UB 探测、实现栈、调优策略或图能力。
已有调用所必需的历史 artifact/ABI 字段归 T1；T6 再扩展 SourceRef、
统一主登记表消费方式以及新字段。两者边界在契约对照表中列明。

### 4.4 T1 验收与交付

1. 原始清单中的全模块扫描输出 `failing: 0`；导入 CLI 入口不会执行 CLI。
2. `python -m pytest --collect-only -q` 退出码为 0，动态收集全部现有测试文件。
3. schema/量化/ABI 基础测试验证真实校验、序列化与固定布局，不能只靠空类导入通过。
4. 既有 A5 相关测试与 `examples/a5_smoke.py` 保持通过。
5. 跑全量测试，提交结构化失败清单：测试 ID、首个失败、复现命令、
   影响、对应任务号或待排期标记。

T1 允许暴露尚待后续任务处理的行为失败，必须显式列出。新暴露的失败不在 T1 中全部顺手修完。
导入/收集门禁通过并记录全量失败清单后推进 P1；尚未满足的 ABI 冻结断言保留为未解决事项。

## 5. P1 各验收单元

| 单元 | 修改边界 | 验收重点 |
| --- | --- | --- |
| T3 | 逐条差异分类、reason 与账本校验 | CUDA 绑定接口逐条可追溯；每条有 reason；达到原单阈值需有真实条目依据 |
| T2.1 | sampling 门面、CANN runner、reference、注册、账本 | 10 个接口逐个核实；MinP 与 speculative 单独验证；概率边界、非法输入、随机可复现与分布正确性 |
| T2.2 | single prefill/decode 的真实 provider；paged/ragged prefill 门面及注册 | 先确认与 T4.2/T4.4 的修改归属；验证因果对齐、GQA、分页/末页、ragged、LSE 与 plan/run |
| T2.3 | 覆盖表指定的 5 个 RoPE 入口 | 按指定提交核对确切名单；布局、偏移、部分旋转、就地修改与 XPos 等选项如实支持或报错 |
| T2.4 | 覆盖表指定的 4 个 GEMM 入口 | dtype、scale 粒度、转置、输出缓冲与精度；复用已验证 A5 FP8 primitive |
| T2.5 | activation 门面、CANN runner、reference、注册、账本 | SiLU/GELU/tanh GELU 三种语义分开验证；形状拆分、out 与 dtype |
| T2.6 | 按覆盖表逐个接入 cascade/page/sparse/fp4 | merge_states 的状态维与 LSE；页写入的槽位/布局；其余项先确定精确名单后实现 |
| T4.1 | MoE 枚举、导出、账本及相关测试 | 11 个成员和值与指定提交一致；`eval(repr(x))` 往返；新增路由方法在实现前明确报错，不能套用别的方法 |
| T4.2 | 单请求 decode 门面、reference、测试、账本 | 上游无 batch 形状；正反例及 GQA；不以旧测试要求维持错误同名语义 |
| T4.3 | MoE 兼容入口、必要的薄包装、账本 | 精确比较参数名、顺序、默认值；分别验证位置调用、关键字调用、out 和返回值 |
| T4.4 | Decode wrapper 公开签名、分页 metadata 和 workspace reset | 对齐上游 plan；indptr/indices/last_page_len 真正进入计划；reset 有效。图捕获/冻结留给 T9 |
| T4.5 | dispatch 后端解析和错误测试 | 未知值与已知 CUDA 后端给出 `DispatchError`、可用后端和迁移指引 |
| T4.6 | Attention PDL 分支、对应测试、贡献规范、账本 | `None/False/True` 在 NPU 上语义一致；保留硬件参数位；不泛化为忽略一切不支持的计算语义 |
| T5 | 两份账本、parity 解析/CLI、校验工具和 CI | 固定完整上游 SHA；导入 147 行源表及证据；保留新增字段；命令退出码 0；CI 验证账本完整性 |

随机采样不能默认要求 CPU 与 NPU 在同一个 seed 下逐样本相同。
如上游接口允许输入同一随机数，则对拍同一随机数；否则验证分布、候选集合、
边界及各平台自身的可复现性，并登记随机序列的实际兼容边界。

### 跨任务重叠先划清归属

- **T2.2 与 T4.2/T4.4**：T2 要接真实 CANN Attention，T4 要纠正现有
  single decode 形状和 paged decode 的 plan 签名。进入 T2.2 前先交付
  接口/文件/验收归属表。建议 T2 负责新增 prefill 入口与后端计算，
  T4 负责既有公开接口纠偏；如果 T2 的验收必须前置 T4 修改，记录具体
  依赖证据并按任务分别提交，不固化错误 API。
- **T4.4 与 T9**：共用设计，分开验收。P1 先对齐公开 API、分页
  metadata 与 eager workspace reset；P3 再接固定地址与图生命周期。
- **T1 与 T6**：T1 还原已有消费方需要的契约；T6 才引入新的来源、
  编译维度、规模、物理属性和图能力。具体字段归属在 T1 契约表确认。

## 6. P2 与 P2+ 各验收单元

| 单元 | 修改边界 | 验收重点 |
| --- | --- | --- |
| T6.1 | 统一 descriptor 的 artifact/source、目标 SoC 与序列化 | 旧构造兼容；产物目标 SoC 与选择一致；source 哈希能参与身份校验 |
| T6.2 | EXTERNAL 身份、注册入口、命名空间、文档 | 默认进入 auto；CUSTOM 仍需 opt-in；来源与后端技术不混淆 |
| T6.3 | shape/规模字段、输入签名、过滤与 bucket 定义 | 不支持的 head_dim/page_size 在执行前拒绝；边界与区间可解释 |
| T6.4 | ND/NZ 排布、UB 属性和硬件签名 | 属性来源可验证；探测不到时显式 unknown；不同约束影响选择与缓存 |
| T6.5 | graph_capability 和参数更新描述 | 三种能力完整校验；needs_param_update 必须有更新协议；未经捕获验证不声明 capture_safe |
| T7 | 启发式排序、可配置默认顺序、解释与缓存 | 无数据时确定性；外部栈有明确位置；不声称未经测量的性能排序 |
| T18 | 统一编译声明、后端钩子、缓存锁/原子发布、装载与产物查询 | Triton 与最小 Ascend C kernel 真正编译、加载、执行；二次命中、并发、失败清理、哈希/SoC 不匹配都验证 |
| T19 | 实现栈标识、栈内配置、两级筛选与解释 | 多个同 backend 的实现栈可区分；解释栈和配置选择；拒绝原因和缓存身份覆盖两级 |
| T20 | benchmark、记录格式、本地/随包库、dispatch 查询 | 先正确性再计时；NPU 同步、预热与重复测量；查询命中选实测冠军，未命中回确定性策略 |

T18 先用最小 Ascend C kernel 验证共同编译入口，不提前搬运 T13 的整套
vllm-ascend 工程，以免把 T18 反向依赖到 P4。
外部预编译产物允许走验证/装载分支，不伪装成需要本地编译的源码。

## 7. P3、P4、P5 各验收单元

| 单元 | 修改边界 | 验收重点 |
| --- | --- | --- |
| T8 | 现有 `flashinfer/utils/` 包中的注册脚手架、已接算子的 schema/meta | 保留包路径兼容；精确描述 out/就地 alias；FakeTensor 无设备计算；`fullgraph=True` 验证无 graph break，并分开报告真实 NPU 编译执行 |
| T9 | T4.4 的 wrapper、Attention 图身份、固定 metadata buffer | 两次 plan 的地址不变；batch/backend/q_len/block-table 宽度按上游时点冻结；图开关有实际作用 |
| T10 | plan 最大 workspace 查询与调用方 workspace 下发 | 声明 shape 上界、容量与开关；执行使用固定 workspace；不足时报错；报告显存代价 |
| T11 | 通用任务句柄、参数更新协议、事件顺序、NPU 扩展 API | 捕获由外部测试/调用方负责；多 seq_len replay 与 eager 对拍；更新入口与账本声明一致 |
| T12 | framework_design §3.4 及相关图说明 | 与 T9–T11 实现逐项一致；核对指定 vllm-ascend 版本的路线信息 |
| T13 | vllm-ascend 构建框架、CMake/打包、第三方来源与登记 | 先在本机 A5 验证 AOT/JIT；其他 SoC 仅报告实际完成的构建/验证；保留许可证和源提交 |
| T14 | C++/Python 注册和 meta 与 T8 合并 | 无重复注册；PrivateUse1 与 fake 路由正确；最小下游导入和编译追踪可用 |
| T15 | 每个通用算子及 vllm-ascend 薄调用点 | 每个算子单独交付 diff、对拍与下游冒烟；确认新调用可用后再移除旧实现 |
| T16 | 每个含引擎语义算子的纯张量边界和薄适配 | 先确认签名；通用 block_table/slot_mapping 可保留，隔离引擎调度对象与私有 metadata；逐个模型调用验证 |
| T17 | Attention provider 能力上移、单一登记/调度/执行链路 | 保留版本/来源/证据与量化、workspace、ABI 验证；原有 conformance 和下游用例通过后移除重复结构 |

T15/T16 的每个具体算子仍是独立验收单元，提交名形如
`T15 下沉 rms_norm_cast 算子`；不把大批迁移藏在一个无法单独验收的提交中。

## 8. 证据与验收产物的位置

- 当前任务定义与父任务进度：`docs/TASKS.md`。
- 子项进度、简要验收结果、遗留问题与提交：`docs/tasks/T*.md`。
- 历史原文、执行细化与核对报告：`work_items/`。
- 契约对照表、失败清单、阶段报告：`docs/reports/`；执行期日志为过程产物，不入仓。
- 长期架构与接口说明：`docs/` 中对应的设计文档；完整执行日志保存在 reports，
  工作单只记录用户验收所需的简要结果。
- 接口状态与差异原因：`flashinfer/data/*.json`。
- 当前环境只有 A5 真机；A2/A3/310P 的执行支持必须另有相应验证证据。

所有验收命令先在仓库根目录激活现有环境：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate flashinfer-a5
source /usr/local/Ascend/cann/set_env.sh
```

检查退出码时保留原命令的退出码；`tail` 只用于展示结果。
没有 CUDA 设备时不导入上游 CUDA kernel 作为运行前提：先从固定提交
静态提取签名，再用独立 reference 与 A5 实测验证语义。

## 9. 下一步与待确认边界

执行顺序已采用新版索引，当前 20 项均保持「未开始」。
下一步仅为 **T1 契约对照表**：逐条给出符号、字段、构造方式、验证要求
和引用证据，先供用户审阅，再写实现。

当前需要确认的内容是 T1 必要导入闭包及 CLI/入口修复是否纳入本任务，
而非重新决定任务排序。T2/T4 的重叠归属在进入 T2.2 前单独确认。

147 条覆盖表、配套架构说明和 vllm-ascend 检出路径仍待提供；
它们不会阻塞 T1。涉及这些资料的任务开始前记录准确版本，
不推测文件内容和下游调用事实。

**本轮只完成重新梳理；未实施 T1、未勾选完成、未提交任何改动。**
