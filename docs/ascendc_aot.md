# 编译、产物与交付

面向算子开发者和发布维护者：本页说明声明如何成为可校验产物，以及 AOT、JIT、原生桥和 wheel 的边界。

## 声明与构建身份

作者维护 `OperatorDefinition` 与后端实现声明；Ascend C 使用 `Support`、`NativeBinding`、`BuildConfig`。生成器计算本地源码闭包、CMake 和适配代码摘要，输出 recipe/Descriptor/索引，freeze 再绑定源码、依赖、能力、ABI 和构建环境形成 `JitSpec`。

| 对象 | 回答的问题 | 维护方式 |
| --- | --- | --- |
| OperatorDefinition | 公共调用计算什么、输出是什么 | 作者维护公共语义 |
| Descriptor | 哪个实现支持哪些输入、如何交给 runner | 后端声明派生 |
| v2 recipe | 怎样构建、原生 Tensor/属性顺序是什么 | BuildConfig/NativeBinding 派生 |
| JitSpec | 这次编译单元的完整身份是什么 | 框架冻结 |
| artifact / manifest | 实际交付哪些字节、怎样验证和加载 | 构建与导出生成 |

recipe 和 spec 不替代 capability；编译通过不扩大硬件支持。修改源码、头文件、CMake、能力或 ABI 后重新生成、冻结并构建，不手填 SHA 或改 manifest 消除失配。

## 源码与包内资源

```text
flashinfer/operators/<domain>/<op>/       公共定义、数值 cases
flashinfer/<backend>/<domain>/<op>/       实现声明、runner；Triton 还包含 kernel
csrc/kernels/<domain>/<op>/               Ascend C host/device/tiling
build/generated/csrc/recipes/             生成 recipe
build/generated/flashinfer/data/          生成索引、Descriptor 和冻结 spec
安装包 flashinfer/data/                    只读交付资源、构建输入和可选二进制
```

源码模式发现 Python 声明；安装模式校验随包索引和源码身份，资源损坏不会静默切换到旁边的源码 checkout。缓存和 scratch 位于可写运行/构建目录，不向 site-packages 写产物。

## 生成、冻结和构建

在依赖齐全的源码环境中：

```bash
python -B -m flashinfer.operators generate
python -B -m flashinfer.operators check
python -B -m flashinfer.operators build --kernel-id ascendc/attention/sparse_flash_attention/native/Ascend910B --output-dir build/sfa-aot
```

`generate/check` 不调用设备编译器。`build` 在新 checkout 中冻结完整原生元数据目录，然后只编译并导出所选实现；因此仍需准备目录中声明的外部依赖，包括已有 CATLASS 项。它不把未选项目的二进制一起编译。

批量发布也可使用已有 AOT 入口：

```bash
python -B -m flashinfer.aot --freeze --all
python -B -m flashinfer.aot --all --cache-dir build/aot-cache --output-dir build/aot-release
```

冻结与构建核验依赖根、固定 revision、头文件及环境身份。CATLASS 根通过 `FLASHINFER_CATLASS_ROOT` 或 AOT 的 `--dependency-root catlass=...` 指定；wheel 不携带完整 CATLASS 头树。通用动态依赖根为 `FLASHINFER_<NAME>_ROOT`。

## 原生产物与执行

公开原生构建采用 `schema_version=2`、`op_package`。包包含 ACLNN host API、推形/tiling 库、设备二进制和元数据。旧 v1/raw/host_shared 仅用于内部冻结兼容测试，不是新算子接入路径。

`AotKernelRunner` 优先搜索并验证匹配 AOT，再通过 runner 工厂绑定 Tensor 调用。只有 missing 且实现允许回落、全局策略也允许时才 JIT；corrupt、加载失败和身份冲突不触发重编。所有候选不兼容可能汇总为 missing，之后仍遵守该策略。

原生 runner 复用 TorchNPU/ACLNN 桥：Tensor/属性描述 → GetWorkspaceSize 与 host tiling → workspace/executor → 当前 stream 提交。后端只适配真实 ABI，不悄悄复制输入或替换用户 out。接口的同步、资源与图限制按具体 runner 核对。

首次激活前可准备互不冲突的可信包集合并同进程使用；同名 generation 热替换或激活后追加 vendor 仍要求新进程。缓存字节替换不会改变已经绑定的 runner。

## Triton 与 CANN

Triton 的设备源码位于其后端算子目录。runner 负责特化、模块和 grid，复用适用的编译缓存/验证设施或自身 launcher。`compile_mode=JIT` 的候选过滤与实际编译许可分别检查。

CANN 适配调用已安装的厂商 API 或组合，本仓不重编厂商内置 kernel。`compile_mode=NONE` 仅表示本仓没有构建步骤，不推断第三方 API 内部是否使用编译或缓存。

## 原生桥与 wheel

源码开发可显式构建原生桥：

```bash
python -B -m flashinfer.aclnn.build --output-dir build/aclnn
export FLASHINFER_ACLNN_ADAPTER_PATH="$PWD/build/aclnn"
```

桥需要匹配的 Torch/TorchNPU/CANN 头与库、C++17 编译器和 libffi。它与设备编译是不同构建单元；设备参数不应直接冒充 host 编译选项。

| 构建模式 | 实际行为 |
| --- | --- |
| 默认或 `FLASHINFER_AOT_ALL=1` | 冻结并构建已登记生产 AOT，打入 wheel |
| `FLASHINFER_AOT_CONFIG` | 按配置和可选 `FLASHINFER_AOT_SOCS` 选择构建 |
| `FLASHINFER_AOT_PREBUILT` | 校验并复用已导出的包与原生桥，不调用编译器 |
| `FLASHINFER_AOT_ALL=0` | 源码模式，不提供生产 AOT 二进制 |

ALL、CONFIG、PREBUILT 互斥。所有目标的工具链和依赖必须齐全；构建机不必有对应 NPU，但安装后的运行必须按每个 SoC 分别验证。

```bash
python -B setup.py bdist_wheel
```

完整发布包含源码资源、Python 适配、索引及所选二进制。PREBUILT 子集不能当作全量算子包；装后使用明确 ID 的 selftest，规则见 [开始使用](getting-started.md) 与 [测试规范](development/testing.md)。

代码入口：[作者生成](../flashinfer/operators/authoring.py)、[冻结](../flashinfer/aot/declarations.py)、[共享构建](../flashinfer/jit/build.py)、[延迟绑定](../flashinfer/jit/runtime.py)、[包验证](../flashinfer/jit/op_package.py)、[Torch 适配](../flashinfer/jit/tensor_adapter.py)、[打包](../setup.py)。
