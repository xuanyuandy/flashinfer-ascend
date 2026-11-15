# 接入算子与新增后端

面向算子开发者：本页说明必须手写的文件、框架生成的内容，以及从公共接口到设备验收的完整工作流。

## 先区分逻辑算子和实现

新增计算语义，新增公共 `operator.py`；给已有算子增加后端，则复用其逻辑 op、参数和输出语义，只新增后端 implementation/runner 与计算实现。公共层不按 backend 复制一份接口，也不把 Ascend C 的原生参数顺序强加给 Triton/CANN。

## 作者文件

| 文件 | 作者职责 |
| --- | --- |
| `flashinfer/operators/<domain>/<op>/operator.py` | OperatorDefinition：公共 schema、输出名、元数据 infer、必要值校验；既有富接口可引用 public_entry |
| 同目录 `cases.py` | 新 schema 直接算子必须提供独立 CPU golden、样例、参数矩阵和逐输出容差；必要时验证某项功能实际生效 |
| `flashinfer/<backend>/<domain>/<op>/implementation.py` | 后端声明、Support、runner/工厂；原生实现还声明 NativeBinding/BuildConfig |
| 同目录 `runner.py` | 选型后将逻辑参数适配到后端，处理真实启动、workspace、stream 和资源寿命 |
| `csrc/kernels/<domain>/<op>/op_kernel/` | Ascend C 设备实现；统一入口与实际提供的 arch22/arch35 分支 |
| 同算子 `op_host/`、按需 `include/` | OpDef、infer、host tiling/workspace、共享布局 |
| 同算子可选 CMake | 扩展 host/device 编译选项，纳入构建身份 |
| Triton 后端目录的 `kernels.py` | 实际 Python 设备 kernel，不放到 csrc |

CANN runner 调用已有原生 API；reference 按相同后端声明协议提供显式参考。源码版权头保留，来源与许可放根目录 `THIRD_PARTY_NOTICES.md`、`LICENSES/`；不在每个算子目录维护重复许可证或手填 provenance JSON。

## 公共定义

带 schema 的定义由框架生成公开函数、Torch functional/out、PrivateUse1 和 FakeTensor 接线。`infer(arguments, layouts)` 只读取元数据，返回 Invocation 的逻辑输入、OutputSpec、属性和 InferenceSpec；FakeTensor 时 layouts 为 None。

`validate_values` 在成功选型后执行必要的值检查。infer 不探测设备或编译器、不 dispatch、不分配真实输出、不启动 kernel。量化 scale/zero-point、融合顺序、辅助输出与 mask 语义必须明确，不从 dtype 名字推断。

常规新增目录不逐项修改 `__init__.py`、领域汇总、Torch 注册器或 selftest 清单。新领域需要把其领域出口接到通用 `install_exports`，已有富接口继续保持原签名与生命周期。

## 实现声明

| 类型 | 作者提供 | 框架派生 |
| --- | --- | --- |
| Support / DimRange | 真实 SoC、dtype、布局、具名维度、组合限制 | CapabilitySpec、维度轴映射与准入 |
| CANNImplementation | operator、support、runner | backend、ID 默认值、Descriptor |
| TritonImplementation / PythonKernel | operator、support、源码、runner 或工厂 | 源摘要、JIT 身份和 Descriptor |
| AscendCImplementation | operator、support、binding、build、必要工厂与 workspace 上界 | 原生 Descriptor、recipe、源闭包与延迟 runner |
| NativeBinding | Tensor/属性有序 ABI、可选性、完整合法 dtype 组合、存储解释 | 标准 Tensor 适配和契约校验 |
| BuildConfig | device/host 源码、OpDef、tiling、可选依赖与编译规则 | 目录默认值、构建摘要和 CMake 接入 |

`runner=` 和 `runner_factory=` 只能选一个。普通 Python 工厂接收 Descriptor；Ascend C 工厂接收已验证模块、spec 和 recipe。作者不填写 digest、冻结指纹或安装产物路径。

带 schema 的直接算子自动生成 `domain="operator"` 能力绑定；它不代替 AttentionPlan 的 profile/POD 协议。图能力默认保持 `NOT_CAPTURABLE`，只有完成匹配设备的捕获/重放证据后才修改声明。

## 多目标与 CMake

ABI 和支持规则相同的多目标原生实现使用 `AscendCImplementation.for_targets(...)`，以规范 SoC 名声明真实目标：Ascend910B、Ascend910C、Ascend950。每个目标拥有独立 Descriptor/recipe/spec/产物，runner 不按 arch22/arch35 分支；原生 ABI 或语义不同则分别声明适配。

设备入口按 `__CCE_AICORE__` 选择已有架构实现，没有分支的架构明确编译失败。A2/A3 目录同为 arch22 不能证明两种产品均已接入；OpDef、tiler 和能力声明都要对应实际产品。

算子 CMake 在 BuildConfig 中显式启用。可提供以下 hook：

```cmake
function(flashinfer_configure_host target)
  target_compile_options(${target} PRIVATE -Wno-deprecated-declarations)
endfunction()

function(flashinfer_configure_kernel target)
  flashinfer_kernel_compile_options(${target} -O2 --cce-auto-sync=on)
endfunction()
```

host hook 接收 CMake target，kernel hook 接收逻辑 handle。使用框架 helper 追加选项；重复直接调用 CANN setter 可能覆盖已有选项。CMake、其本地输入、头和适配源码均进入闭包。

## 生成与验收

1. 定义数学语义、支持域、输出、原生 ABI 和独立 golden。
2. 写公共定义、后端声明、runner、计算实现和必要 CMake。
3. 静态生成、检查、冻结并构建；审阅生成结果，不手改 JSON。
4. 检查拒绝路径、显式/自动路由、out/FakeTensor 和依赖错误。
5. 在目标设备执行数值、非默认流、队列、AOT/缓存与安装验证。

```bash
python -B -m flashinfer.operators generate
python -B -m flashinfer.operators check
python -B -m flashinfer.operators build --kernel-id ascendc/attention/sparse_flash_attention/native/Ascend910B --output-dir build/sfa-aot
```

该 build 会冻结完整原生目录再只编译所选项，外部依赖要求见 [编译与交付](../ascendc_aot.md)。新增生产实现还要独立审核 `tests/production_inventory.py` 的精确预期；不能让生成器自动修改测试答案。

已有通用验收自动发现 cases，复用 out 身份、非默认 stream、队列、缺包/损坏包等场景；新增语义仍要补专门正反例。验收定义不等于真机通过，具体规则见 [测试规范](testing.md)。

## 可直接阅读的实现

| 范本 | 公共定义 | 后端和源码 |
| --- | --- | --- |
| arch22 SFA | [operator/cases](../../flashinfer/operators/attention/sparse_flash_attention/) | [Ascend C 声明与 runner](../../flashinfer/ascendc/attention/sparse_flash_attention/)、[原生源码](../../csrc/kernels/attention/sparse_flash_attention/) |
| GMM/SwiGLU/量化 | [operator/cases](../../flashinfer/operators/gemm/grouped_matmul_swiglu_quant/) | [原生适配](../../flashinfer/ascendc/gemm/grouped_matmul_swiglu_quant/)、[计算源码](../../csrc/kernels/gemm/grouped_matmul_swiglu_quant/) |
| 三后端 Matmul | [逻辑定义](../../flashinfer/operators/gemm/matmul/operator.py) | [CANN](../../flashinfer/cann/gemm/matmul/)、[Ascend C](../../flashinfer/ascendc/gemm/matmul/)、[Triton](../../flashinfer/triton/gemm/matmul/) |

公共协议代码：[definition](../../flashinfer/operators/definition.py)、[作者类型](../../flashinfer/operators/author_api.py)、[原生类型](../../flashinfer/ascendc/definition.py)、[生成](../../flashinfer/operators/authoring.py)、[执行编排](../../flashinfer/operators/runtime.py)。
