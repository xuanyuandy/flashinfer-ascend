# 源码资源、量化能力与 JIT 职责核对

日期：2026-11-12。本仓基线 commit 为 `65c13f8`，任务起点已包含设计文档与 Matmul 文档的未提交修改。
上游对照为本地 flashinfer-ori 的 Git HEAD `4cdf2f56f`；部分 JIT 源码未物化到其工作树，以下对照通过 `git show HEAD:<path>` 读取提交内容。

## 量化目录清理

本次删除 `flashinfer/quantization/__init__.py` 及其空目录。该包只从 `runtime.schema` 转导出 `QuantSpec`，没有量化计算实现；仓内 Python 源码没有通过这个包导入类型。

量化模型及能力声明继续保留：

- `flashinfer/runtime/schema.py:120` 定义 `QuantSpec`，`flashinfer.runtime` 继续导出同一类型。
- `InferenceSpec` 表达调用的量化事实，`CapabilityCase/CapabilitySpec` 表达实现支持的组合。
- 算子的后端 `implementation.py` 绑定具体量化能力；如 GMM 的 INT8/INT4、scale、layout、group size 与融合组合。

删除的是只转导出类型的顶层包，没有删除量化语义、计算实现或 kernel 登记。`from flashinfer.quantization import QuantSpec` 这一入口随包删除，统一使用 `from flashinfer.runtime import QuantSpec`。

## JSON 的当前用途与源码归属

“安装或运行时需要元数据”与“必须人工维护一份 JSON 源码”是两件事。当前实现直接读取下列资源，因此不能只删除文件；其位置、生成时机与数量可以在保留契约和验证强度的前提下调整。

| 文件组 | 当前生产者或来源 | 当前消费者及作用 | 源码归属判断 |
| --- | --- | --- | --- |
| `data/operators.json` | `operators/authoring.py` 从逻辑定义派生 | `operators/catalog.py`：接口定义、导出及源码摘要索引 | 生成资源；不应成为第二份手工登记 |
| `data/implementations.json` | 同一生成器，从后端实现声明派生 | `operators/implementation.py`：实现发现、身份及源码摘要验证 | 生成资源；可以在构建阶段生成并随包交付 |
| 四份生产 `csrc/recipes/*.json` | `ascendc/authoring.py` 从 `NativeDefinition` 派生 | freeze 和 AOT 构建：闭包、ABI、依赖及编译配置 | 生成资源；作者维护 Python 声明和实际源码 |
| `csrc/recipes/matmul_probe.json` | 显式维护的探针配方 | 历史兼容与构建探针测试 | 当前仍是配方输入，不属于生产声明自动生成的四份 recipe |
| `data/kernels/registry.json` | 生产 Ascend C 声明派生，freeze 更新 source 身份 | `ascendc/_catalog.py`：四个生产 Descriptor | 交付元数据；不是全部 85 个全局实现的人工登记表 |
| `data/kernels/ascendc_specs.json` | freeze | 同一加载器：可信 recipe/spec、目标和冻结身份 | 发布身份需要保留；文件组织可以调整，不能取消绑定校验 |
| `data/kernels/triton_registry.json` | 实验 Triton attention 声明派生 | 实验声明及校验；不自动进入全局登记 | 生成资源 |
| `data/kernels/manifest.schema.json` | 维护者编写的格式 schema | 格式说明和工具契约 | 可以归入开发工具或文档资源；不等于 kernel 实现 |
| `data/api_parity.json`、`attention_api_parity.json` | 上游接口对照工具与人工状态说明 | `parity.py`、CLI、报告和测试 | 对标诊断资料，不是 kernel 选择和执行必需；迁移时须保留诊断入口的资源加载 |
| `data/attention_operator_operations.json` | 外部 provider API 的结构化契约目录 | `attention/operation_catalog.py`：参数与调用契约 | 领域契约数据，可维护为 Python 或结构化源数据；不是已有可执行 kernel 清单 |
| `data/attention_capabilities.json` | provider 能力记录，目前 profiles 为空 | CLI 和能力契约加载 | 领域能力资料；空记录不构成可用 provider 或设备支持证据 |

产物内部的 `manifest.json`、`op-package.json`、缓存回执和文件摘要又属于另一类：它们绑定编译结果与加载身份。即使减少源码中的 JSON，也必须保留独立的可信身份、ABI、摘要和完整性验证。

建议的单向交付关系是：作者源码与不能派生的契约 → 构建时生成索引/recipe → 发布时冻结 → 安装包资源及产物清单。API 对照资料单独归入诊断资源。当前 loader 和生成器尚未改为这一资源位置安排，本次没有删除这些 JSON。

## flashinfer-ori 的实际做法

上游并非完全不用 JSON。该提交包含后端 `build.json`、部分 kernel manifest 等 JSON 资源；但是没有本仓这套 `flashinfer/data/` 目录，也没有同名 `operators.json`、`implementations.json`、`ascendc_specs.json` 或 `api_parity.json`。

上游对照入口：

- flashinfer-ori: `flashinfer/jit/core.py:292`：`JitSpec` 定义通用缓存、构建和加载生命周期。
- flashinfer-ori: `flashinfer/jit/core.py:739`：`gen_jit_spec` 接收 sources、编译/链接参数、include 路径及加载适配器。
- flashinfer-ori: `flashinfer/jit/attention/modules.py:530`：`gen_single_prefill_module` 根据 dtype、head_dim、位置编码、滑窗等参数选择和生成编译输入。
- flashinfer-ori: `flashinfer/jit/__init__.py:26`：导出 attention 领域生成器，同时导出通用 JIT 核心。
- flashinfer-ori: `flashinfer/aot.py:34`：AOT 复用 JIT spec 与领域生成入口。

因此，可以对齐“Python 声明与生成器驱动通用构建”的机制，不必照搬或额外建立多份人工 JSON 登记。

## 为什么存在 jit/attention

JIT 需要区分基础设施与领域生成适配：

| 层次 | 负责内容 | 是否需要理解 attention 语义 |
| --- | --- | --- |
| 领域或后端生成适配器 | 从 heads、dtype、mask、KV layout 等得到特化身份、源码、编译配置和绑定 | 需要 |
| 通用 JIT 基础设施 | 根据 spec 处理缓存、锁、编译、产物验证和加载 | 不需要解释这些语义；将领域与特化指纹作为身份输入即可 |

把领域生成器组织在 `jit/attention` 是上游采用的一种目录选择，目录存在本身不表示通用缓存或编译器硬编码了 Attention。按本仓已确认的“后端/域/算子”布局，具体 kernel 的生成配置适合放在对应后端算子目录，共有领域计划适配适合归入 Attention 域。

本仓当前 `flashinfer/jit/attention/` 共九个文件，其主要职责如下：

| 文件 | 实际职责 |
| --- | --- |
| `variants.py`、`utils.py` | Attention 特化字段、指纹及模块名 |
| `modules.py` | 将已经选型的 Attention plan/receipt 映射成通用 JitSpec |
| `plan.py` | 计划与缓存解析结果的身份绑定 |
| `artifacts.py`、`loading.py` | 将通用产物验证和加载结果绑定到 Attention 计划 |
| `execution.py`、`planning.py` | 对外部执行器和 plan factory 的身份绑定协议 |
| `__init__.py` | 上述类型与接口导出 |

它不是九种 Attention kernel，也不是完整的实际 kernel 编译器。`modules.py` 生成的历史 Attention spec 主要记录身份与入口；通用 `JitSpec` 明确区分这类协议声明与可执行编译单元。实际 Triton Matmul/Attention 的 spec 和启动逻辑已经分别放在其后端算子 runner 中。

当前三个 Attention 领域模块和四份专项测试仍导入该目录。该目录尚未迁移；后续迁移应保持类型契约、冻结常量、序列化身份、计划/产物/加载校验以及必要的导入兼容，不能只删除目录或以路径迁移代替编译验收。

## 作者接口与仅随安装包交付的数据目录

目标是让 `implementation.py` 声明实现事实，由框架派生完整 Descriptor、recipe 和冻结资源；作者不直接填写其序列化字典。以下为目标接口示意，当前尚未实现：

```python
SUPPORT = Support(
    soc="Ascend950",
    dtype="bfloat16",
    layout="ND",
    dimensions={
        name: DimRange(minimum=16, maximum=256, multiple_of=16)
        for name in ("M", "N", "K")
    },
)

IMPLEMENTATION = TritonImplementation(
    operator=MATMUL,
    name="bf16_nd",
    support=SUPPORT,
    source=PythonKernel(file="kernels.py", entry="_matmul_kernel"),
    runner_factory=TritonMatmulRunner,
)
```

这里的维度符号需要与公共算子定义绑定，框架负责把 M/N/K 映射成各 Tensor 轴上的完整约束，避免作者重复枚举输入和输出的六条 ShapeRange。底层模型可以继续完整表达约束，不要求作者以底层格式填写。

| 类别 | 作者接口处理 |
| --- | --- |
| 逻辑 op、稳定实现名、源码/入口、runner 或工厂 | 明确声明；普通 callable 与 runner 工厂使用不同参数，避免推断其含义 |
| dtype、SoC、布局、尺寸与量化支持组合 | 使用有文档的类型和具名规则；复用公共语义，只填写实现自己的支持限制 |
| 原生 Tensor/属性 ABI、特殊 workspace、混合核任务与同步协议 | 作者或已有原生构建声明提供真实信息；框架检查一致性，不猜测 |
| SHA256 digest、源码闭包、spec 指纹、编译环境身份 | 框架按声明的构建输入计算，作者不填写哈希字符串；发布与加载继续校验 |
| 可确定的后端、默认目录、标准 ABI 元数据、格式版本和空字段 | 框架派生；必要时提供类型明确的覆盖入口 |
| 产物路径、二进制摘要、已解析符号、构建回执 | 构建或发布阶段生成，不进入作者注册声明 |

不同后端可以提供各自的声明类型，共同生成 `ImplementationDefinition` 和 `KernelDescriptor`。改造不能只把同一大字典移到另一个文件。已有 kernel_id 保持稳定；新增实现可按明确的稳定命名约定生成 ID，源码变化通过 spec 指纹区分，不自动重命名已发布 kernel。

源码仓可以完全不包含生成的 `flashinfer/data/` 目录。目标流程为：

- 源码开发：从作者声明建立内存目录，登记不探测工具链、不编译；需要原生产物时读取 `build/generated/` 的冻结构建结果，缺失时明确报告尚未构建。
- 显式构建：在独立 build 目录生成索引、recipe、冻结契约和二进制，不写回源码包目录。
- 打包：在构建输出的 `flashinfer/data/` 中组装安装资源，再进入 wheel；sdist 保留生成资源所需的作者源码和非派生契约。
- 安装运行：只读包内索引和冻结契约，不扫描源码仓、不依赖 checkout 路径，不向安装目录写入缓存，也不在资源缺失或损坏时静默切换到源码模式。

冻结契约应独立于可选 AOT 二进制交付，确保缺包时仍能登记实现并给出准确错误。源码模式也不得将变更后的源码与过期冻结契约混用；重新冻结继续是显式构建操作。这些是目标约束，当前加载器和作者接口仍需按此迁移。

## 本次验证

约定解释器不可执行，实际返回 permission denied、退出码 126；使用现有 conda 解释器与 CANN 环境。设备自检实际输出 `True 8`，设备名为 Ascend910B3。本次没有新的 Ascend950 或 Triton 真机数值执行证据。

实际专项命令：

```bash
source /home/dy/tools/miniconda3/Ascend/ascend-toolkit/set_env.sh
export FLASHINFER_TEST_NPU_DEVICE=4
/home/dy/tools/conda-envs/megamoe-a2/bin/python -B tools/run_regression.py \
  --profile quick --log-dir /tmp/fi-source-resource-audit \
  tests/test_schema.py tests/test_manifests.py tests/test_operator_layout.py \
  tests/test_api_smoke.py tests/test_checkpoint_030_attention_jit_framework.py \
  tests/test_checkpoint_031_attention_jit_active_plan.py \
  tests/test_checkpoint_032_attention_jit_artifact_verification.py \
  tests/test_checkpoint_033_attention_jit_module_loading.py \
  tests/test_release_checks.py tests/test_parity_tracking.py
```

实际结果为 **95 passed、2 个子测试通过、0 skipped**。JUnit 汇总将子测试计入 passed，因此工具 summary 显示 97 passed；二者没有相加重复统计。原始日志、JUnit 和退出码位于上述 log-dir。

静态校验确认：量化空包已移除，`runtime.QuantSpec` 保持原类型，85 个全局 Descriptor 未减少，86 个实现的生成检查通过，主设计文档代码围栏数 64→64。只修改了空导出包和文档，没有修改编译、缓存、设备源码或冻结 spec；本次未重复执行 full、real_compile 或 wheel 构建。


## 作者接口与资源位置的实施更新

原先的目录已按 [作者声明与资源交付](../authoring-and-resources.md) 实施：全部既有后端声明采用类型化作者接口，
生成 JSON 只位于 `build/generated/` 与安装包；非派生契约源位于 `tools/metadata/`。
Attention 编译协议已迁入 `flashinfer/attention/jit/`，历史导入保留薄兼容入口。
上文关于“尚未迁移”的文字记录整改前状态；实际测试证据以本次整改报告为准。
