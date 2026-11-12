# T13 分析：统一 Ascend C 构建框架与可选 CATLASS

日期：2026-10-23。本文是分析与设计建议，不是实现完成报告。本次仓库内只新增本文；编译用的小程序、构建目录和日志全部位于临时目录，没有修改任何参考仓库、`flashinfer/` 或任务文档，没有 commit/push。

## 1. 结论与证据范围

**建议采用一个 Ascend C 编译后端，把 native / CATLASS 表达为构建配方中的实现变体与依赖，把 host shared library / op-package 表达为产物和装载方式。** 两种源码的 `SourceRef.language` 都保持 `ascendc`；用户配置选择已声明的变体，解析结果进入完整构建身份。AOT 和 JIT 消费同一配方、同一编译驱动，复用 T18B 的锁、staging 和原子发布。依据是下述源码分析、设备编译、host library 调用和 ACLNN 调用实测，而不是根据 header-only 属性推断 ABI。

核心发现：

1. 上游 CUTLASS 是**标准源码/发行布局中的基础第三方依赖**，通用 nvcc 构建环境默认加入其头文件；但具体 kernel 可以使用自研 CUDA 实现，也可以使用 CUTLASS。它不是所有 kernel 必经的算法层，没有发现全局删除 C++ CUTLASS 依赖的构建开关。Python `nvidia-cutlass-dsl` 的可选 extras 是另一件事。[U10][U11][U12]
2. vllm-ascend 已有“同一 CANN op 构建框架 + 每 op 编译选项”的机制，原生 kernel 与 CATLASS kernel 都能走它。`ENABLE_BUILT_JIT` 控制包是否包含预编设备 binary；`ENABLE_BUILT_CUSTOM` 控制 custom/built-in 包路径，二者都不是 native/CATLASS 选择器。[V1][V2][V3][V7][V8][V9]
3. 本次 A5 最小 GEMM 在统一 C 入口后，两种源码的设备编译命令仅相差 CATLASS include 与 `CATLASS_ARCH=3510`，另有各自源码内部的模板/实现差异。host library 和 op-package 两种封装也都实测编译、装载、计算成功。不能据此推导任意 CATLASS example 原样接入都只需加 include，详见 §4。[E1][E2][E3]
4. `.o` 扩展名不足以决定装载方式：本次直接 `-c` 得到 NPU `ET_REL`；op-package 的 `.o` 是 NPU `ET_EXEC`；host `.so` 则是 x86-64 `ET_DYN`。两类实现内的对应产物格式一致，三种产物之间的加载契约不同。[E4]

本文使用以下路径简称，出处表在 §9，所有行号均针对本地快照：

| 简称 | 绝对路径 / 版本 |
| --- | --- |
| U | `/home/dy/projects/flashinfer-ori`，`da992ed9919604c5eb6dc255c63f5e89a466452c` |
| V | `/home/dy/projects/vllm-ascend`，本地 `csrc/`、`setup.py` 参考快照；不虚构 Git 版本 |
| C | `/home/dy/projects/catlass-ref`，`41bf90da655bba3c66d0acd7e00abe33960ecfd6` |
| L | `/home/dy/projects/flashinfer`，分析开始时 `5566f8b2a7a4700dbb83ab21f4882486eb94236b`，工作区干净 |
| E | `/tmp/flashinfer-t13-analysis-20261023-bS8hFg`，本次实验专用目录 |

V 的三个入口文件 SHA-256：

```text
setup.py             f2f4750db33e9bdc74c8fc6c938e0f1be9d1a0434e5cf73be952511e49f12df9
csrc/build.sh        795766811a91c3ee2ffe709f8148e6aa78679bb6a1bb14abca0dccc89c42b39c
csrc/CMakeLists.txt  b896946ff246e3f23c269386b2e8fbc7ad884024ec47d4358e2616bb25dc077f
```

`E/baseline.json` 记录 Git 基线，`E/reference_manifest.json` 记录所引源码的摘要与行数。用户已确认的 CATLASS header-only 统计、工具链存在性、上一轮标准包产出和 T18B 状态作为输入事实，不重复清点。

## 2. 上游 FlashInfer 对外提供什么

### 2.1 声明、生成、构建与装载

| 接口 | 实际职责与公开面 | 出处 |
| --- | --- | --- |
| `flashinfer.jit.JitSpec` | 抽象生命周期：`try_load()`、`build()`、`load()`、`build_and_load()`、产物路径与锁路径。`build_and_load` 先尝试缓存，再加锁后二次检查，执行 JIT 禁用策略，最后构建和装载 | `U/flashinfer/jit/__init__.py:60–79`；`core.py:259–358` [U1][U2] |
| `JitSpecNvcc` | 一个 nvcc/ninja 模块的 sources、C/CUDA/link flags、include、device linking、load adapter 等；JIT 主产物 `<name>.so` | `U/flashinfer/jit/core.py:445–528` [U3] |
| `gen_jit_spec(name, sources, ...)` | 创建 `JitSpecNvcc`，合入默认编译选项，支持额外 include、按源文件 flags、load adapter，并登记到全局 spec registry；返回 spec，不在这个函数里编译 | `U/flashinfer/jit/core.py:698–815` [U5] |
| 算子级 `gen_*_module` | 把 dtype、head dim、算法变体等映射到 URI、模板实例化源码和 spec；有些直接列静态源码，有些生成 Jinja 配置和实例文件 | activation 的 `134–146`；attention 的 `530–620`、`1773–1942`、`2314–2350` [U8][U9][U13][U20] |
| `spec.build_and_load()` | 公共算子层实际使用的入口。nvcc spec 查找 AOT `.so`，缺失时走 ninja；`load` 用 `tvm_ffi.load_module`，不是把任意设备 `.o` 直接交给 Python | `U/flashinfer/prefill.py:445–449`；`jit/core.py:536–605` [U4][U18] |
| AOT `compile_and_package_modules(...)` | 覆盖配置、重设 source/include/workdir、调用 `gen_all_modules`，再调用 `build_jit_specs`，最后复制产物；CLI 为 `python -m flashinfer.aot` | `U/flashinfer/aot.py:1310–1411,1597–1703` [U6] |
| `build_jit_specs(specs, verbose, skip_prebuilt)` | 合并 nvcc specs 的 ninja 构建；此快照显式拒绝非 `JitSpecNvcc`，不能把它描述成已经覆盖所有 DSL 的 AOT 通用编译器 | `U/flashinfer/jit/core.py:826–862` [U21] |
| `JitSpecCuteDsl` | 同一生命周期抽象下的另一编译器适配：执行 compile callback，导出 `.o`，用 `cute.runtime.load_module(..., enable_tvm_ffi=True)` 重载 | `U/flashinfer/jit/cute_dsl_core.py:91–120,162–180,243–256` [U19] |

这里的“公开”指 `flashinfer.jit` 显式 re-export 或上游实际提供的模块接口，不额外承诺这些开发者接口的跨版本兼容性。[U1]

### 2.2 CUTLASS 的地位：分发层常备，kernel 层按需使用

有三条直接依据：

- `.gitmodules:1–3` 把 `3rdparty/cutlass` 声明为 submodule；`pyproject.toml:71–75,178–181` 把它映射到 `flashinfer.data.cutlass` 并包含 `include/**`、`tools/util/include/**`；`build_backend.py:1110` 在数据目录准备时处理它。[U11][U12]
- `jit/env.py:402–405` 定义两个 CUTLASS include 目录；`jit/cpp_ext.py:135–179` 在通用系统 include 集合里无条件追加它们；AOT 也显式重设这两个路径。[U10][U6]
- `gen_jit_spec` 没有 `use_cutlass` 参数，`JitSpecNvcc` 也没有 CATLASS/CUTLASS 专属编译器分支。具体模块通过 sources、include 和 flags 选择实现。[U3][U5]

代表性 kernel 分层如下，**不是全仓穷举清单**：

| kernel / 路线 | CUTLASS 使用证据 |
| --- | --- |
| BF16 CUTLASS GEMM | `include/flashinfer/gemm/bf16_gemm_cutlass_template.h:23–30` 引用 CUTLASS collective/device adapter；Python 的 `get_gemm_sm100_module_cutlass_bf16` 调对应 generator [U16][U17] |
| Blackwell CUTLASS FMHA | `attention/blackwell/fmha_cutlass_sm100.cuh:22–28` 引用 CuTe/CUTLASS；generator 仍调用 `gen_jit_spec` [U9][U16] |
| Hopper FA3 attention | `attention/hopper/mainloop.cuh:10–18` 引用 CUTLASS/CuTe；即使 backend 名字不是 `cutlass`，也使用了这套头文件 [U15] |
| TGV 的 C++ 模板 | `include/flashinfer/gemm/tgv_gemm_template.h:20–21` 使用 CUTLASS 类型；不能把“backend != cutlass”解释为“无 CUTLASS 依赖” [U22] |
| `act_and_mul` | 自研 CUDA kernel，activation generator 生成绑定并走同一个 `gen_jit_spec`，无需以 CUTLASS GEMM/collective 模板实现 [U13][U14] |
| FA2 prefill | 自有 prefill、mma、cp_async、layout 等源码；`gen_customize_single_prefill_module` 选择自有模板与源码 [U7][U20] |

检索命令实际执行过：

```bash
cd /home/dy/projects/flashinfer-ori
rg -n 'ENABLE_CUTLASS|DISABLE_CUTLASS|USE_CUTLASS|WITH_CUTLASS' \
  flashinfer build_backend.py pyproject.toml csrc include scripts
```

唯一命中是 `flashinfer/experimental/b12x/_lib/runtime_patches.py:25` 的 `B12X_DISABLE_CUTLASS_RUNTIME_PATCHES`，它是局部 runtime patch 开关，不能当成 C++ CUTLASS 的全仓构建开关。结合上述参数表与构建代码，结论是**没有发现标准构建的全局 CUTLASS on/off 机制**；这不等于证明移除头文件后连一个独立 native kernel 都编不了。

另一个容易混淆的点：`pyproject.toml:27–35` 的 `nvidia-cutlass-dsl` extras 是可选 Python DSL 软件包；它与随源/wheel 提供的 C++ CUTLASS headers 是不同依赖层。[U11][U19]

对本仓的借鉴是共享 compiler/spec 生命周期和按 kernel 声明依赖。**不建议照搬其全局 CUTLASS include 注入**：本仓明确要求 native 构建能完全不依赖 CATLASS，§4.5 已做缺失路径验证。

### 2.3 下游自定义 kernel 接触哪些接口

两种入口都存在：

1. **自定义 attention variant**：`gen_customize_single_decode_module`、`gen_customize_single_prefill_module`、batch 对应函数允许指定 URI、dtype、额外 tensor/scalar 参数、`variant_name` 与 C++ `variant_decl`；batch wrapper 的 `backend`、`jit_args`、`jit_kwargs` 接口会接入生成的模块。自定义 prefill 的 `backend="auto"` 被明确拒绝，调用方须确定模板路线。[U1][U18][U20][U23]
2. **独立 CUDA 模块**：提供源码与 TVM FFI 导出，调用 `gen_jit_spec(name, sources, extra_*flags, extra_include_paths, ...)`，然后 `build_and_load()` 并调用导出符号。activation 模板的 `TVM_FFI_DLL_EXPORT_TYPED_FUNC` 和 `gen_act_and_mul_module` 就是完整的小例子；自定义编译器可实现 `JitSpec` 生命周期，CuTe DSL 给出了实例。[U2][U5][U13][U19]

例如下面是对现有接口的用法示意，**本次没有在本机编译 CUDA**：

```python
from flashinfer.jit import gen_jit_spec
spec = gen_jit_spec(
    name="my_kernel_v1",
    sources=["my_kernel.cu", "my_binding.cc"],
    extra_include_paths=["my_include"],
)
module = spec.build_and_load()
# module.my_exported_function(...)
```

不需要把 kernel 改造成 CUTLASS kernel 才能进入这个入口。需要适配的是编译源码、宿主绑定、导出函数和对应调用约定。[U3][U5][U13]

### 2.4 `backend=` 与编译路径不是同一维度

- FA2 / FA3：backend 选择不同模板、源码和架构 flags，最后都落到 `gen_jit_spec` / nvcc；FA3 的 `sm90a_nvcc_flags` 是明确的生成参数。[U8][U20]
- `mm_bf16(backend="cutlass")`：选择 CUTLASS runner/generator；`get_gemm_sm100_module_cutlass_bf16` 调 `.build_and_load()`。backend 不是独立的“CUTLASS 编译器”。[U17]
- CuTe DSL：确实存在不同 compiler/loader adapter，但共享 `JitSpec` 生命周期；它不能与“是否 include C++ CUTLASS”混为一谈。[U19]
- AOT / JIT：在 spec 生命周期中决定。nvcc spec 的 `try_load` 先找可用 AOT；没有时构建 JIT 路径。`backend=` 不直接决定“用 AOT 还是现编”。[U2][U4]

因此本仓也应分别记录实现选择、编译工具链、产物/ABI 与编译时机。

## 3. vllm-ascend 的框架组织

### 3.1 `build.sh` 开关的准确含义

| 输入 / 变量 | 源码中的实际作用 |
| --- | --- |
| `--pkg` | 设置 `ENABLE_BUILD_PKG=TRUE`，默认 built-in；不是 native kernel 的同义词。`build_package` 最终调用 CMake `package` target [V1][V18] |
| `--jit` / `ENABLE_BUILT_JIT` | 参数解析置 TRUE；custom 分支将 `ENABLE_OPS_KERNEL=OFF`，保持 `ENABLE_OPS_HOST=ON`；built-in 包分支也关闭 kernel binary。源文件、动态编译适配脚本和 host 侧仍有安装规则 [V1][V2][V3] |
| `--ops=...` / `ENABLE_BUILT_CUSTOM` | 选择 op 列表，并设置 custom 路径、`ENABLE_BUILT_IN=FALSE`；`--vendor_name` 也会进入 custom 路径。与 CATLASS 无关 [V1][V2] |
| `--soc=a,b` | `process_soc_input` 把逗号转分号；不同包分支按 SoC 配置/构建。custom 分支把 compute-unit 列表交给 CMake [V1][V2][V18] |
| `ENABLE_OPS_HOST / ENABLE_OPS_KERNEL` | CMake 分别控制 host 与 kernel 产出；`custom_build.cmake:802–814` 只有 kernel 开启时才创建各 compute-unit 的 binary targets [V3][V19] |

所以它的 `--jit` 更准确地说是“制作不带预编 kernel binary、保留运行期编译材料的包”。没有证据把它等同于本仓的 `declare_kernel -> JitBuildStore -> build_and_load` Python 调用协议；也不能由这个名字推导 CATLASS/native 的区别。[V2][V3][L4][L6]

`setup.py` 还有一层产品打包编排：`build_ext.run` 在 `COMPILE_CUSTOM_KERNELS` 开启时先执行 `build_aclnn.sh`，再构建 `vllm_ascend.vllm_ascend_C` extension；extension 配置传递 SoC、pybind11、torch-npu 路径。**这两层产品胶水不应被当作通用 Ascend C 编译器接口。**[V12]

### 3.2 op_host / op_kernel、SoC 与产物布局

实际数据流如下：

```text
build.sh：op 列表 / compute units / custom-or-built-in / 是否编 binary
  -> CMake：发现 op 目录、选择 op_host、汇总 host 定义与 tiling
  -> 生成 op 信息与 dynamic Python 编译适配
  -> 按 compute unit 生成并执行 kernel 编译命令
  -> 设备 binary + 每 op JSON + binary_info_config.json
  -> host libraries / 源材料 / vendor package
```

逐项源码依据：

- `func.cmake:88–158` 按目录发现 op，用 `ASCEND_OP_NAME` 过滤；`custom_build.cmake:278–297` 进入 `op_host` 并处理 attention；`attention/CMakeLists.txt:13–28` 也按 op 名发现与筛选。[V3][V5][V20]
- 每 op 的 `op_host/CMakeLists.txt` 登记 op、选择 ACLNN 类型、添加 tiling 源和编译选项；`obj_func.cmake:16–93` 汇总 op_api、infer、tiling；SoC 版本的宏从选中的 arch 目录收集 tiling 文件。[V6][V8]
- `CMakeLists.txt:33–65` 将 `ascend310p/ascend910b/ascend910_93/ascend950/kirinx90` 映射到 `arch22/arch32/arch32/arch35/arch32`，其中 arch32 又加入 arch22；源文件自身还可以按 `__CCE_AICORE__` 选不同实现。这是该快照的过滤规则，不代表每个 op 真正支持全部芯片。[V4][V10]
- `situ_mx_quant/op_host/CMakeLists.txt:11–13` 对非 ascend950 直接 return，是明确的 op 支持边界。[V17]
- `func.cmake:486–497,595–608` 将 custom binary 安装到 `packages/vendors/${VENDOR_NAME}_transformer/op_impl/ai_core/tbe/kernel/<compute_unit>/...`；`766–794` 生成/安装 `binary_info_config.json` 和 relocatable 配置信息。[V15]

### 3.3 同一框架能否编不同风格 kernel

**能，现有表达方式主要是每 op 的 sources、include 和 compile options，而非一个显式的 kernel-style enum。**

- CATLASS 示例：`chunk_fwd_o_vllm/op_host/CMakeLists.txt:16–23` 用 `add_ops_compile_options` 注入 CATLASS include、`--cce-auto-sync=off` 等，host tiling target 也可单独拿到 include；kernel 入口仍是 `extern "C" __global__ __aicore__`，内部调用 Catlass 模板。[V8][V10]
- 原生示例：`kda_layout_swap12` 使用同一登记/编译选项框架，也设置 `--cce-auto-sync=off`，入口同样带 workspace、tiling；内部是原生 `TPipe` 和 vector kernel。[V9][V11]
- `add_ops_compile_options(OP_NAME, COMPUTE_UNIT, OPTIONS)` 的实现统一把选项交给生成器/配置文件，后面的 binary 编译器没有按 CATLASS 另起分支。[V7][V15]

这同时说明同步选项不能仅按库名设全局默认：本例原生 kernel 因为手动同步而关闭 auto-sync。CATLASS 的 A2 examples 另有架构特定 LLVM flags；A5 example 的分支不是同一组 flags。[V9][C2]

### 3.4 值得借鉴与不应照搬的部分

| 值得借鉴 | 在本仓如何采用 |
| --- | --- |
| op、SoC、编译选项、产物收集分层 | 配方声明、目标表、统一 driver、打包 adapter 分开；不让 API wrapper 拼 shell 命令 [V3][V7][V15] |
| 每 op 依赖/include/同步策略 | 变成可验证的 recipe 字段，支持 native 与 CATLASS，保留 per-target 范围 [V8][V9] |
| host/tiling 与 device 的明确契约 | 稳定 ABI、tiling schema、workspace、blockDim 一起记录；有需要才生成 ACLNN host [V6][V10][V11] |
| binary 与源材料分离 | AOT 出 binary；需要现编时才携带 source/header 依赖，二者共用配方 [V2][V3] |

| 不应照搬 | 原因与实证 |
| --- | --- |
| `setup.py` 的 torch-npu / pybind11 / `vllm_ascend_C` 产品路径 | 绑定框架与发行产品；本仓的核心 C ABI 不必依赖它们 [V12][V14] |
| `build_aclnn.sh` 的硬编码大 op 列表、按 SoC 自动拉 CATLASS、全局 `CPATH` | 它会 `git config --global`、submodule sync/update、fetch/checkout 并设置 CPATH；违背本仓“native 不需要 CATLASS、构建可离线”的目标 [V13] |
| `_transformer` vendor、vLLM 符号/类型和产品条件编译 | `torch_binding.cpp` 引入 torch_npu、`namespace vllm_ascend`、`VLLM_ENABLE_ATB_AND_DIRECT_KERNELS`；`func.cmake:506–509` 甚至有特定 QLI 命名映射 [V14][V15] |
| op 私有 CATLASS 快照、全局兼容头与复制到 common 的依赖组织 | `msa_index_score` 同时处理私有快照与 submodule；顶层注入 `cann_compat.h`；不应未经需求验证就成为本仓全局行为 [V4][V16] |
| 复制它的独立缓存、安装脚本和全部 CMake 历史分支 | 该快照已有 `vllm_ascend_build_cache_command`；本仓 T18B 已有缓存协议与原子发布，应保留一个缓存所有者 [V15][L6] |

任务说明中的旧统计“CATLASS 只在 9 个文件、attention/mc2 两个子目录使用”不能用于这份 V 快照：`csrc/moe/chunk_fwd_o_vllm` 已有明确反例。本次不重做全仓计数，也不修改任务文档。[V8][V10]

## 4. 两类 kernel 的实际编译与运行

### 4.1 实验定义与可复核文件

所有实验使用本机 CANN `/usr/local/Ascend/cann-9.2.0-beta.2`、A5 / `dav-3510`；运行使用 NPU 0。命令继承本机已配置的 CANN 环境，其中 `ASCEND_HOME_PATH` 指向上述目录，`CMAKE_PREFIX_PATH` 包含其 `lib64/cmake`；重放时须先配置同类 CANN 环境。CATLASS 侧实际实例化 `BasicMatmulTla + BlockMmadTla + MmadPingpong<Arch::Ascend950>`，改写自 C 的 `43_ascend950_basic_matmul`，不是只 include 一个头文件的空 kernel。native 侧使用原生 Ascend C 的标量循环 GEMM。两边均计算固定 FP32 `16×16×16`，数据和期望结果相同；本实验不比较性能。[C3][E1][E2]

共同设备入口：

```cpp
extern "C" __global__ __aicore__ void t13_probe(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling);
```

共同 host launcher 是 `extern "C" void t13_launch(void *stream, uint8_t *a, uint8_t *b, uint8_t *c)`，用 `<<<1, nullptr, stream>>>` 调设备入口。direct 模式 workspace、tiling 为 null；op-package 模式增加同样的 `REGISTER_TILING_DEFAULT`、`GET_TILING_DATA` 检查，两边共用 host tiling，blockDim=1、workspace=0。[E2][E3]

实验源码完整列于附录 A；本次实际文件及驱动为：

```text
E/src/{native.cpp,catlass.cpp,launcher.inc,CMakeLists.txt}
E/device_compile.py       # 两种设备编译命令和退出码
E/build_direct.py         # 同一工程选择两种 profile
E/launch_direct.py        # ctypes + ACL，创建 stream、拷贝、调用、同步、比较
E/opproject/             # msopgen 生成并在临时目录适配的标准算子工程
E/build_package.py       # 同一 op 工程用 T13_PROFILE 选择实现
E/launch_package.py      # 设置临时 vendor 路径，直接调用生成的 ACLNN 接口
```

### 4.2 实验一：相同设备编译器、相同入口

下面是实际执行参数的可复制写法；只用变量压缩绝对路径，没有省略 compiler flags：

```bash
E=/tmp/flashinfer-t13-analysis-20261023-bS8hFg
CANN=/usr/local/Ascend/cann-9.2.0-beta.2

"$CANN/bin/bisheng" -xcce --cce-aicore-arch=dav-c310-cube \
  --cce-aicore-only -std=c++17 -O2 \
  -I"$CANN/compiler/tikcpp/tikcfw" \
  -I"$CANN/compiler/tikcpp/tikcfw/interface" \
  -I"$CANN/compiler/tikcpp/tikcfw/impl" \
  -c "$E/src/native.cpp" -o "$E/native.device.o"

"$CANN/bin/bisheng" -xcce --cce-aicore-arch=dav-c310-cube \
  --cce-aicore-only -std=c++17 -O2 \
  -I"$CANN/compiler/tikcpp/tikcfw" \
  -I"$CANN/compiler/tikcpp/tikcfw/interface" \
  -I"$CANN/compiler/tikcpp/tikcfw/impl" \
  -I/home/dy/projects/catlass-ref/include -DCATLASS_ARCH=3510 \
  -c "$E/src/catlass.cpp" -o "$E/catlass.device.o"
```

实际输出/检查：[E1][E4]

```text
native:  EXIT_CODE=0
catlass: 3 warnings generated.  EXIT_CODE=0
native.device.o:  Type REL, Machine 0x1029, GLOBAL FUNC t13_probe
catlass.device.o: Type REL, Machine 0x1029, GLOBAL FUNC t13_probe
```

CATLASS 的三条 warning 来自 `tla/numeric/integral_constant.hpp:105` 的 inline/host/aicore 属性；**本次成功编译没有添加 suppress flags、专用链接库或 CATLASS 编译器**。`K_MAX_SHAPE_DIM=0` 在 CATLASS 源码内部设置，来自官方 example 的优化约定；它不在 native 源码里。因而精确说法是“此例命令只差 include/架构宏，源码另有模板与局部设置”，不能把这些源码差异抹掉。[C3][E1]

保留了一次有意义的初始失败：把 layout 构造参数写成 `16` 得到有符号 layout，与模板要求的 `uint32_t` layout 不匹配；改成 `16u` 后成功。日志为 `E/catlass.device.initial.log`，错误是 `no matching constructor ... Kernel::Params`。这是模板类型契约，不是换编译器或改变入口 ABI 的要求。

### 4.3 实验二：同一 CMake 生成 host `.so`，同一接口运行

共用 `find_package(ASC)`、`LANGUAGES ASC CXX`、`add_library(... SHARED ...)`。仅 `PROFILE` 决定源码和可选 CATLASS include/macro，完整 CMake 见附录 A。实际执行：[E2]

```bash
cmake -S "$E/src" -B "$E/build-native" -DPROFILE=native -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$E/build-native" --verbose -j2
cmake -S "$E/src" -B "$E/build-catlass" -DPROFILE=catlass -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$E/build-catlass" --verbose -j2
```

verbose 日志中真正的编译命令如下，路径以 E/CANN 等价展开表示，保留实际参数；编译与链接的工作目录分别是对应的 `build-native`、`build-catlass`：

```bash
# native.build.log:13
"$CANN/bin/bisheng" -DT13_WITH_HOST -Dt13_probe_EXPORTS \
  -std=c++17 -fPIC --npu-arch=dav-3510 -O2 \
  -MD -MT CMakeFiles/t13_probe.dir/native.cpp.o \
  -MF CMakeFiles/t13_probe.dir/native.cpp.o.d \
  -o CMakeFiles/t13_probe.dir/native.cpp.o -c --asc-aicore-lang "$E/src/native.cpp"

# catlass.build.log:13
"$CANN/bin/bisheng" -DCATLASS_ARCH=3510 -DT13_WITH_HOST -Dt13_probe_EXPORTS \
  -I/home/dy/projects/catlass-ref/include \
  -std=c++17 -fPIC --npu-arch=dav-3510 -O2 \
  -MD -MT CMakeFiles/t13_probe.dir/catlass.cpp.o \
  -MF CMakeFiles/t13_probe.dir/catlass.cpp.o.d \
  -o CMakeFiles/t13_probe.dir/catlass.cpp.o -c --asc-aicore-lang "$E/src/catlass.cpp"

# 两者使用同一链接规则，仅输入对象文件不同
"$CANN/bin/bisheng" -shared -Wl,-soname,libt13_probe.so \
  -o libt13_probe.so CMakeFiles/t13_probe.dir/native.cpp.o
"$CANN/bin/bisheng" -shared -Wl,-soname,libt13_probe.so \
  -o libt13_probe.so CMakeFiles/t13_probe.dir/catlass.cpp.o
```

四个 configure/build 均退出 0。此 CANN 的 ASC 规则确实调用 bisheng，见工具链 `asc_modules/CMakeASCInformation.cmake:121–150`。本机这一规则没有产生请求的 `compile_commands.json`，因此本文依据的是 `--verbose` 输出，没有声称拿到了该 JSON。

实际运行：

```bash
python3 "$E/launch_direct.py" native
python3 "$E/launch_direct.py" catlass
```

```text
LOAD_OK=native
{"profile":"native","shape":[16,16,16],"device":0,"stream_sync":0,"max_abs_error":0.0}
LOAD_OK=catlass
{"profile":"catlass","shape":[16,16,16],"device":0,"stream_sync":0,"max_abs_error":0.0}
```

日志中的完整 JSON 还包含相同的前八个输出值 `[-1,9,-1,4,-11,-1,9,-1]`。两者都以 `ctypes.CDLL` 装载，查找 `t13_launch`，复用同一个 ACL stream 调用协议；没有链接任何 `libcatlass`。调用脚本为每种实现单独启动进程，**这里没有验证同进程热切换/同名注册**。[E2]

### 4.4 实验三：同一标准 op-package 工程编译并经 ACLNN 运行

生成工程的实际命令为：

```bash
"$CANN/bin/msopgen" gen -i "$E/t13_probe.json" \
  -c ai_core-Ascend950PR_9579 -lan cpp -out "$E/opproject"
```

`t13_probe.json` 声明 `T13Probe`，两个 float/ND 输入 a/b 和一个 float/ND 输出 c。生成后仅在 E 内适配：host 的 AddConfig 改为 `ascend950`、blockDim 改为 1；kernel 两种实现都注册/读取同一个 `T13ProbeTilingData{uint32_t size;}`，检查 size=256；`T13_PROFILE` 决定包含 native/CATLASS 实现。其余生成的 codegen、host/tiling libraries、kernel library、package 机制共用。[E3]

每种 profile 执行下面一套命令，分别将 `PROFILE` 设为 `native`、`catlass`：

```bash
cmake -S "$E/opproject" -B "$E/pkg-$PROFILE" \
  -DASCEND_COMPUTE_UNIT=ascend950 \
  -DENABLE_SOURCE_PACKAGE=ON -DENABLE_BINARY_PACKAGE=ON \
  -Dvendor_name=t13_analysis \
  -DASCEND_CANN_PACKAGE_PATH="$CANN" \
  -DASCEND_PYTHON_EXECUTABLE=/root/miniforge3/envs/flashinfer-a5/bin/python \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$E/pkg-$PROFILE" \
  -DT13_PROFILE="$PROFILE"
cmake --build "$E/pkg-$PROFILE" --verbose -j2
cmake --build "$E/pkg-$PROFILE" --target binary package --verbose -j2
```

**需要显式构建 `binary package`**：本次生成模板的默认 `all` target 先构建 host；只看 `all` 成功不能宣称已编设备 kernel。生成模板 `E/opproject/build.sh:40` 本身也使用这两个 target。

生成的 op_kernel CMake 统一使用：

```cmake
npu_op_kernel_sources(ascendc_kernels KERNEL_DIR ./)
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/ TILING_LIBRARY cust_optiling)
npu_op_package_add(${package_name} LIBRARY ascendc_kernels)
if(T13_PROFILE STREQUAL "catlass")
    npu_op_kernel_options(ascendc_kernels ALL OPTIONS
        -DT13_USE_CATLASS -I/home/dy/projects/catlass-ref/include -DCATLASS_ARCH=3510)
endif()
```

两边 `*.package.binary.log:53–58` 记录同一个 CANN `ascendc_compile_kernel.py` 驱动，再执行生成的 `T13Probe-t13_probe-0.sh`。CATLASS 额外选项记录在 `pkg-catlass/autogen/custom_compile_options.ini:1`：

```text
ALL,,-DT13_USE_CATLASS;-I/home/dy/projects/catlass-ref/include;-DCATLASS_ARCH=3510
```

这里 `T13_USE_CATLASS` 是实验源码选择宏，不是 CANN 或 CATLASS 自带开关。native 对应的自定义选项文件为空。没有抓取该 Python 驱动内部所有 cc1 子命令；关于 compiler/flags 的直接对比由 §4.2、§4.3 的实际 bisheng 命令提供，不把 CANN 内部参数猜成已经观察到的事实。

两边实际产物中，标准 vendor 树均包含：

```text
packages/vendors/t13_analysis/
  op_api/lib/libcust_opapi.so
  op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so
  op_impl/ai_core/tbe/kernel/ascend950/t13_probe/
    T13Probe_f16d3a4cd378b0cd3f75dcb1a6a2f931.o
    T13Probe_f16d3a4cd378b0cd3f75dcb1a6a2f931.json
  op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json
```

该树位于各 `pkg-*/_CPack_Packages/Linux/External/custom_opp_ubuntu_x86_64.run/` 内；各自还生成了 `pkg-*/custom_opp_ubuntu_x86_64.run`。**没有运行安装器或安装到系统**。运行脚本把 `ASCEND_CUSTOM_OPP_PATH` 指向临时 vendor 树，通过生成的 `libcust_opapi.so` 调用相同的两段 ACLNN API：[E3]

```bash
python3 "$E/launch_package.py" native
python3 "$E/launch_package.py" catlass
```

```text
LOAD_OK=native
GetWorkspaceSize=0 bytes=0
aclnnT13Probe=0
{"profile":"native","shape":[16,16,16],"device":0,"stream_sync":0,"max_abs_error":0.0}
LOAD_OK=catlass
GetWorkspaceSize=0 bytes=0
aclnnT13Probe=0
{"profile":"catlass","shape":[16,16,16],"device":0,"stream_sync":0,"max_abs_error":0.0}
```

两者逐项得到相同的 `binFileSuffix=.o`、`core_type=AIC`、`magic=RT_DEV_BINARY_MAGIC_ELF_AICUBE`，配置指向相同形式的 binPath/jsonPath；payload SHA-256 不同。注意两种源码产生的 CANN 文件名哈希**相同**，不能拿这个文件名充当本仓完整构建缓存 key。[E3][E4]

### 4.5 产物与依赖边界的实测

实际检查命令包括以下调用（绝对路径以 E 等价表示）：

```bash
readelf -h "$E/native.device.o" "$E/catlass.device.o"
readelf -Ws "$E/native.device.o" "$E/catlass.device.o"
readelf -h "$E/build-native/libt13_probe.so" "$E/build-catlass/libt13_probe.so"
nm -D --defined-only "$E/build-native/libt13_probe.so" "$E/build-catlass/libt13_probe.so"
readelf -h "$E"/pkg-native/op_kernel/ascendc_kernels/binary/ascend950/t13_probe/*.o \
  "$E"/pkg-catlass/op_kernel/ascendc_kernels/binary/ascend950/t13_probe/*.o
```

结合文件摘要得到：[E4]

| 产物 | native 字节数 | CATLASS 字节数 | ELF / 入口 / 装载 |
| --- | ---: | ---: | --- |
| `*.device.o`，直接 `-c` | 34,896 | 39,328 | NPU `0x1029`、`ET_REL`；`t13_probe`；本次未把它直接提交 runtime |
| `build-*/libt13_probe.so` | 535,408 | 543,840 | x86-64 `0x3e`、`ET_DYN`；host 导出 `t13_launch`；两边均实际 CDLL + ACL 调用 |
| op-package 的 `T13Probe_*.o` | 6,184 | 9,912 | NPU `0x1029`、`ET_EXEC`；入口 `T13Probe_f16d3a4cd378b0cd3f75dcb1a6a2f931_0`；两边均实际经 ACLNN 调用 |

标准包设备 binary 摘要：

```text
native   048fb66e6680532a313dd5bd110b925e38a35ce20ca36837c8baf1fb85ed009e
catlass  94637b3c5789c690f5272b84b534e5daf9c166bda611b9adb4ffc45a8ec2cfd8
```

字节数只描述本次源码、路径与工具链，不保证不同构建路径下逐字节复现。体积差也不能当成性能结论。

另将临时 CMake 改为按需读取 `CATLASS_ROOT`，在清除 CPATH/CPLUS_INCLUDE_PATH 后实际执行：

```bash
env -u CPATH -u CPLUS_INCLUDE_PATH cmake -S "$E/src" -B "$E/native-no-catlass" \
  -DPROFILE=native -DCATLASS_ROOT=/tmp/t13-deliberately-missing-catlass
env -u CPATH -u CPLUS_INCLUDE_PATH cmake --build "$E/native-no-catlass" --verbose -j2
env -u CPATH -u CPLUS_INCLUDE_PATH cmake -S "$E/src" -B "$E/catlass-missing" \
  -DPROFILE=catlass -DCATLASS_ROOT=/tmp/t13-deliberately-missing-catlass
```

结果：native configure/build 都退出 0；CATLASS configure 退出 1，明确报 `CATLASS headers required only by PROFILE=catlass`。未通过移动/重命名参考克隆模拟缺失。这验证了按 target/profile 声明依赖可以让 native 构建与 CATLASS 解耦。[E5]

### 4.6 哪些差异来自库，哪些来自封装

| 维度 | CATLASS 相对 native 的固有要求 | 与库无关、应由框架单独表达的要求 |
| --- | --- | --- |
| 编译器 | 此例没有替代编译器，两者是 Ascend C / bisheng | CANN ASC 直编、CANN op compiler 都可承载两者 [E1][E2][E3] |
| include / flags | CATLASS 需要模板 headers、匹配的 arch 宏；本例有源码内 `K_MAX_SHAPE_DIM=0` | 同步、优化、调试、AIC/AIV/mixed 等按实际 kernel/目标配置；不可全部按库名继承 [C2][V9] |
| 入口 | 原始 `DeviceGemm` 使用 `KernelAdapter<Kernel>(Params)`，可额外带 fftsAddr | op-package 常用 GM_ADDR + workspace/tiling；可以用薄 wrapper 统一，而不是强迫 CATLASS API 成为公共 ABI [C4][C5][C6] |
| tiling | CATLASS `Initialize` 把 Arguments 转 Params，模板层有自己的 tile/layout 类型 | 也可以像 CATLASS 官方 ACLNN example 那样，由 host TilingContext 产 tiling buffer，再在设备入口构造模板 Params；native 亦可用相同协议 [C4][C6][C7] |
| 产物 / 装载 | 没有 `catlass-only` 文件格式或加载器要求的实证 | 相同 output kind + ABI 下已验证共同装载；不同 output kind 必须明确 loader [E2][E3][E4] |

因此“不强绑定”的**编译框架成本较低**，但仍要做好 wrapper、tiling、依赖闭包与缓存身份。不能把官方 example 43 的 `main + DeviceGemm` 整个 `.cpp` 当成标准 op_kernel 直接搬进去：原 example 的宿主调用和 `Params` 值传参契约已经明确不同；官方 `advanced/basic_matmul_aclnn` 正好示范了适配层。[C3][C4][C5][C6][C8]

## 5. 建议：用配方表达路径，用配置选择变体

以下接口名与配置是**设计提案**，目前仓库没有这些新配置/API。

### 5.1 不改变 `SourceRef.language` 的含义

`L/flashinfer/runtime/schema.py:376–401` 把它定义为源码语言；`jit/backends.py:76–79` 当前还校验 language 与 compiler backend 匹配。给 CATLASS 源码填 `language="catlass"` 会让现有 `AscendCBackend` 声明失败，并人为制造第二个编译器后端。[L1][L2]

建议的归属：

| 信息 | 推荐位置 |
| --- | --- |
| 源码语言 | `SourceRef.language="ascendc"`，两种实现相同 |
| 实现与模板依赖 | 独立、不可变 `BuildRecipe` manifest：`implementation=native/catlass`、sources、header dependencies、definitions/options |
| 用户要选哪个实现 | 构建配置的全局默认与 per-kernel override；解析时选择已存在的 recipe/descriptor |
| 编译器 / 目标 | recipe 的 toolchain=`ascendc`，显式 CANN/SoC 支持表；resolved recipe 记录具体 compiler identity 与 architecture |
| 产物 / 加载协议 | recipe 的 `output_kind=host_shared/op_package`、adapter id；descriptor 继续记录 `ArtifactRef`、`KernelLaunchABI`、`KernelBinaryABI`、tiling schema |
| AOT / JIT | 现有 compile mode / 调用策略；不与 native/CATLASS 合并成一个开关 |

第一版不必扩展 T6.1 `SourceRef` 或给 descriptor 增加 `language` 值。**建议把 manifest 用 `kernel_id + source.digest + target_soc` 明确绑定到 descriptor，解析后的 recipe digest 强制写进 `JitSpec.input_fingerprints`。** 现有 JitSpec 已能记录有名输入摘要、compile options 和环境身份；但现有 `_SourceBackend.declare` 只填 kernel/package，T13 需要补齐 recipe/依赖绑定，不能误称现状已经包含了这些内容。[L2][L3][L8]

同一逻辑 op 的 native 与 CATLASS 实现应有不同的具体 kernel_id/SourceRef，且都归 `ascendc` backend。配置改变时解析到另一条声明；进入编译后不再覆盖已选 descriptor 的 source/ABI。若某 op 只有 CATLASS 实现，选 native 时必须报告“不支持该变体”；配置不会自动把 CATLASS 算法改写成原生算法。该边界由两个实际不同的实验源码及 descriptor 的单一 source 结构直接体现。[L8][E1]

### 5.2 一套框架的控制流

```text
用户构建配置 + recipe manifests + descriptors
  -> 显式解析：选择变体，检查 SoC / source / ABI / dependencies
  -> Frozen resolved recipe -> JitSpec（完整身份）
  -> 统一 AscendC compiler driver
       source closure + 可选 CATLASS include/defines
       + output adapter：host_shared 或 op_package
  -> 校验 artifact 与依赖文件 -> 同一个 store 发布
  -> AOT 导出/打包，或 JIT loader 装载
```

配置示意：

```toml
[ascendc]
implementation = "native"
output_kind = "host_shared"
target_soc = "Ascend950PR_9579"

[ascendc.overrides.matmul_probe]
implementation = "catlass"

[dependencies.catlass]
root = "3rdparty/catlass"
revision = "41bf90da655bba3c66d0acd7e00abe33960ecfd6"
```

切换只改 `implementation`；仅当产物契约也要切换时另改 `output_kind`。库风格和封装形式的四种组合在本次实验都已验证，不能把 `catlass` 固定映射到 direct，把 `native` 固定映射到 ACLNN。[E2][E3]

recipe 的最少字段建议为：

```text
id/version, kernel_id, source_digest, toolchain
implementation, target constraints
sources / header roots / dependency identities
compile definitions / options / sync policy
device entry / host entry / launch ABI id / kernel task type
tiling mode + schema version + host source or generator identity
output kind + adapter version
```

其中 `tiling_mode` 可以区分 none/固定参数、GM buffer、host Params 值传递；不把 CATLASS 的 C++ Params 类型直接序列化到 Python 公共 ABI。原始 CATLASS DeviceGemm 与其 ACLNN example 已证明这些是独立选择。[C4][C5][C6][C7]

compiler adapter 可以从**显式传入的冻结 recipe 集合**按 spec 中的 digest 取回配方，核验后编译；不在 compile hook 内再次读取环境变量来偷偷切换实现。manifest 的读取和依赖解析在显式构建/声明流程进行，保留现有“import/静态 descriptor 构造不探测编译器”的边界。[L1][L2][L3]

### 5.3 AOT/JIT 共用驱动与正确缓存身份

建议把“构建并验证产物”作为共同操作，JIT 沿用现有 `build_and_load` 的 store 编排，AOT 用 build-only 操作调用同一个 compiler driver 后导出 wheel/vendor tree。AOT 不能通过必须 load 到当前设备的操作来强制交叉构建机拥有 NPU。[L4][L6]

缓存至少绑定：主 SourceRef、所有额外 source/header 的内容摘要、recipe 版本、实现变体、CATLASS commit 与实际 header 内容、CANN/compiler/host ABI 身份、SoC/架构、flags、生成器、tiling/launch ABI、output adapter。绝对 include 路径可作为定位配置，不能代替内容身份；只依赖 Git commit 也无法发现脏 headers。[L3][C9][E4]

原生配方的 dependency closure 不含 CATLASS，CATLASS 版本变化不使原生缓存失效；CATLASS 配方必须因所依赖 headers 改动而失效。§4.4 中“同文件名、不同 payload”就是不能只用 CANN binary 文件名缓存的直接反例。[E4]

本仓现有 verifier 对非 Triton 编译单元验证 host `ET_DYN` 和动态导出符号，默认 loader 对 shared library 使用 CDLL。因此建议 **T13 首先以 `host_shared` 为 AOT/JIT 共同默认产物**，它与现有接口最接近；`op_package` 作为同框架内的显式 output adapter，不能把 NPU `.o` 冒充该 host `.so`。[L4][L5][E4]

标准 op-package 涉及多个库、JSON、设备 binary 和 vendor 搜索路径；现有单个 `ArtifactRef` 加默认 CDLL 不会自动完成这一切。后续需定义 package manifest 与 loader/launcher adapter，并保留全部文件校验。`DiskJitBuildStore` 禁止发布 symlink，CMake/CANN 中间树也不适合原样发布；只将整理后的最终文件集放进 store，必要的兼容链接需物化或显式转换。[L4][L5][L6]

### 5.4 CATLASS 可选依赖的集成方式

建议使用 `3rdparty/catlass` 的固定 submodule commit，但**仅在解析出的构建目标需要它时**才创建本仓的 headers INTERFACE target、检查 headers、注入 include 和相应架构宏。不要 `add_subdirectory(catlass)` 来构建模板库：C 的顶层 CMake 会继续加入 examples/3rdparty，带来示例工程及宿主运行时依赖。[C1][C2]

依赖规则建议如下，§4.5 的不存在路径实验已经验证该控制方式可行：[E5]

- native-only：完全不要求 submodule 初始化，不扫描 CATLASS 版本、不拉网络、不注入全局 CPATH。
- 显式 catlass：缺失 headers 或不支持目标时，在编译前给出目标/依赖位置/固定版本的错误；提供手动 submodule 初始化提示，不自动 fetch/checkout。
- auto：如提供此模式，应在解析阶段按已声明支持矩阵与可用依赖选定并记录具体变体；运行状态中显示 resolved implementation，不能在编译失败后静默换算法。
- 发 AOT-only wheel：不必携带 CATLASS headers；声明支持 CATLASS JIT 的源码发行物须显式携带/定位所需 headers 和许可证。header-only 不消除生成代码对 CANN runtime 的依赖。[C2][U11]

本次临时 op-package 的 source 部分仍包含实验用绝对 include 配置，**没有验证把该源包移到无参考克隆机器后的运行期重编**。生产源码包需解决可重定位的依赖打包；本次验证的是当前目录下的 binary 编译和 binary 调用。

## 6. 本仓目录与命名建议

`docs/framework_design.md` §2.3 的设备源码目录现已统一为 `csrc/kernels/`，探针与生产算子按算子分目录；构建工程、launcher 与可选 op-package host 层同属 `csrc/`。位置参考 vLLM-Ascend 的 `csrc/kernels/`，源码选择仍由显式 recipe 决定。[L7]

设备源码位置已按下列目录收敛；其余编译工程细分仍为本分析的建议：

```text
3rdparty/catlass/                     # 可选、固定版本的 submodule
csrc/kernels/<op>/
  native.cpp                         # 原生实现；只声明实际存在的变体
  catlass.cpp                        # 可选 CATLASS 实现
  tiling.h                           # op 私有 host/device 契约
  arch35/...                         # 仅在确有架构差异时拆分
csrc/
  CMakeLists.txt                     # 统一入口；显式 source 路径，不递归拉大仓
  cmake/
    AscendToolchain.cmake
    KernelTargets.cmake
    OptionalCatlass.cmake
    OpPackage.cmake
  recipes/<op>.json                  # 两种实现、目标、依赖和 ABI 的唯一配方
  include/flashinfer_ascend/launch_abi.h
  launch/<op>_launch.cpp              # raw pointer / stream 的稳定 C ABI
  op_package/<op>/op_host/            # 仅 ACLNN 封装所需
    op_def.cpp
    tiling.cpp
tools/                               # AOT build-only / 打包入口
```

`npu_op_kernel_sources(KERNEL_DIR ...)` 与 `npu_op_kernel_library(SRC_BASE ...)` 可以显式指定源码目录，本次生成工程也使用这两个参数。因此不必依赖 `op_host/../op_kernel` 的固定邻接路径才能组织构建。安装包的 vendor 布局仍按 CANN 产物要求生成，源码目录与 vendor 产物目录分开。[E3]

与 V 的具体差别：不把 host 文件是否存在当作 native/CATLASS 分类；不强制 direct kernel 也创建 GE OpDef；不以 glob 扫描结果决定依赖；公共 target/namespace 用 `flashinfer_ascend`、`fi_ascend_*`，vendor 建议 `flashinfer`，稳定入口按 `fi_<op>_<abi_version>` 或带 recipe 区分的唯一符号命名。原有特定 vLLM 类型、`*_transformer` 包名和 op 私有 CATLASS 副本均无须带入。[V5][V8][V14][V15][V16]

打包时将统一的 `csrc/` 构建输入映射到 `flashinfer/data/build/csrc/`，SourceRef 保持相对于构建输入根 `flashinfer/data/build/` 的 locator；源码只维护一份。当前由 `setup.py` 的 `BuildPy` 显式复制并登记输出，schema 与运行时声明则在 `flashinfer/data/kernels/`。该资源映射遵守本仓 SourceRef 的相对路径约束。[U11][L1]

若决定改为所有 device 源码集中到 `csrc/ops/<op>/op_kernel`，须同步更改上述已存在的目录职责文档；本分析没有替用户执行这类文档或目录迁移。

## 7. 建议分五步落地及验收

以下均为后续实施步骤，**不把本次临时实验当成本仓钩子已经实现**。

| 步骤 | 工作 | 验收点 |
| --- | --- | --- |
| 1. 配方与配置契约 | 定义 manifest、descriptor 绑定、实现/产物/SoC 三个独立维度；扩展 AscendC 声明流程，把 recipe 与依赖摘要放入 spec | 两种实现 language 都是 ascendc；缺变体/缺依赖/ABI 不匹配明确报错；旧 descriptor 的序列化与指纹不变；切换实现、flags、依赖内容会产生不同构建身份 [L1][L2][L3] |
| 2. 原生链路与共同构建操作 | native 最小算子 + C ABI host_shared；接通 AscendC compiler hook；AOT build-only 与 JIT 共用 driver，沿用 store | A5 真机数值正确；冷构建、跨进程锁、warm hit 不调编译器；JIT disabled 命中成功/缺失失败；错误 SoC/导出符号/损坏文件被拒绝；不能拿 device ET_REL 冒充 host library [L4][L5][L6][E2] |
| 3. 可选 CATLASS | 引入固定 submodule 与按目标创建的 headers target；加入等价 CATLASS 变体 | 只改配置即可选两类源码，保持同一个 driver/store/loader；未初始化 submodule 的 native 构建成功；显式 CATLASS 缺 headers 明确失败；CATLASS headers 改动只失效相关缓存 [C1][E1][E2][E5] |
| 4. 发行与 op-package adapter | 用同一配方生成 AOT 文件、可选标准 vendor 包；补齐安装后资源/依赖 locator、package manifest、对应 loader/launcher | 两种实现的包都能在干净进程中从声明产物调用；所有库/JSON/binary 参与校验；AOT 客户端不现编；声明支持 JIT 的源码包能脱离参考克隆重编；不触发隐式网络、不携带 vLLM 产品命名 [E3][L4][L5][L6] |
| 5. 支持矩阵与交接 | 扩展 A2/A3/310P、AIC/AIV/mixed、动态 tiling；补齐多模块与 graph 边界；将真实 ArtifactRef/ABI 交给后续注册层 | 每项区分“交叉编译成功/真机 eager 成功/图验证成功”；没有设备证据的组合不标 runtime 支持；不在 T13 顺手搬 T15/T16 算子 [V4][V17][L8] |

第 2 步不等第 3 步才能完成，这是可选依赖真正成立的验收条件。若产品要求第一阶段的主发行物就是标准 op-package，可以提前第 4 步的 adapter，但仍保持两种源码使用一个 AscendC driver。

## 8. 已确定、未验证与需要产品决定的事项

已确定：本机 A5 上，真实 CATLASS GEMM 与 native GEMM 可以共享设备编译器和 C 入口，也能分别使用同一种 direct `.so` 封装、同一种标准 ACLNN op-package 封装完成真实计算。用户给出的 header-only/toolchain 前提没有被本次实验推翻。[E1][E2][E3][E4]

不能扩大结论的边界：

- 只验证了 CANN 9.2 beta.2、A5、固定 FP32 GEMM、AIC-only；未验证 A2/A3/310P、mixed AIC/AIV、动态 shape、量化、复杂 workspace、性能或 graph capture。CANN 老版本是否支持本次 `find_package(ASC)`/`npu_op_*` 组合，需要兼容性实验，不据此宣布支持全部版本。[C1][C2][E1][E2][E3]
- 标准包和 direct library 都以**每种实现单独进程**运行；同进程两份同名 op/符号、多个 vendor 的注册隔离、卸载生命周期尚未测。生产框架应使用唯一构建身份/符号并另做隔离验证。[E2][E3]
- 未验证源码 wheel/sdist 的异机重编、native-only wheel 缺 CATLASS 的完整安装流程；当前只证明 CMake native 目标在 CATLASS root 不存在时能编译。[E5]
- 没有在上游 U 执行 CUDA 构建，也没有运行 V 的整套 `setup.py`/大算子包；对它们的结论是源码快照分析，不是本机运行验收。

需要产品决定、本文给出默认建议而不阻塞分析交付的两点：

1. T13 首个稳定发行物是否必须标准 op-package？推荐先以 host_shared 接现有 JIT verifier/loader，再提供同框架的 op-package adapter；若 ACLNN 生态互操作是首要交付，则提前后者。两种方式都已有本次最小真机证据。[L4][L5][E2][E3]
2. `auto` 是否需要自动选 native/CATLASS？推荐第一版默认 native、显式 override，待各 op 的支持/性能矩阵有证据后再加自动选择。显式 native 缺变体不能静默变成 CATLASS，反之亦然。

## 9. 出处索引

下表的 U/V/C/L 根路径定义见 §1；点击编号可打开对应文件起始行。多段行号在同一文件内。

| 编号 | 文件与行号 |
| --- | --- |
| [U1] | `U/flashinfer/jit/__init__.py:24–79` |
| [U2] | `U/flashinfer/jit/core.py:259–358` |
| [U3] | `U/flashinfer/jit/core.py:445–528` |
| [U4] | `U/flashinfer/jit/core.py:536–605` |
| [U5] | `U/flashinfer/jit/core.py:698–815` |
| [U6] | `U/flashinfer/aot.py:1310–1411,1430–1448,1597–1703` |
| [U7] | `U/include/flashinfer/attention/prefill.cuh:19–42` |
| [U8] | `U/flashinfer/jit/attention/modules.py:530–620` |
| [U9] | `U/flashinfer/jit/attention/modules.py:2314–2350` |
| [U10] | `U/flashinfer/jit/cpp_ext.py:135–179`；另见 `U/flashinfer/jit/env.py:402–405` |
| [U11] | `U/pyproject.toml:27–35,71–75,178–181`；submodule 见 `U/.gitmodules:1–3` |
| [U12] | `U/build_backend.py:1102–1114` |
| [U13] | `U/flashinfer/jit/activation.py:25–97,134–146` |
| [U14] | `U/include/flashinfer/activation.cuh:20–55` |
| [U15] | `U/include/flashinfer/attention/hopper/mainloop.cuh:10–18` |
| [U16] | `U/include/flashinfer/gemm/bf16_gemm_cutlass_template.h:23–30`；FMHA 见 `U/include/flashinfer/attention/blackwell/fmha_cutlass_sm100.cuh:22–28` |
| [U17] | `U/flashinfer/gemm/gemm_base.py:677–694,808–837,1956–1976` |
| [U18] | `U/flashinfer/prefill.py:445–449,1626–1636,2821–2834,2952–2964` |
| [U19] | `U/flashinfer/jit/cute_dsl_core.py:91–120,162–180,243–256` |
| [U20] | `U/flashinfer/jit/attention/modules.py:1773–1942` |
| [U21] | `U/flashinfer/jit/core.py:826–862` |
| [U22] | `U/include/flashinfer/gemm/tgv_gemm_template.h:20–21` |
| [U23] | `U/flashinfer/jit/attention/modules.py:1683–1699` |
| [V1] | `V/csrc/build.sh:42–47,95–99,888–919,1082–1086` |
| [V2] | `V/csrc/build.sh:1493–1505,1580–1622` |
| [V3] | `V/csrc/cmake/custom_build.cmake:278–297,718–814` |
| [V4] | `V/csrc/CMakeLists.txt:33–65,101–107,289–313` |
| [V5] | `V/csrc/cmake/func.cmake:31–43,88–158` |
| [V6] | `V/csrc/cmake/obj_func.cmake:16–93,135–215` |
| [V7] | `V/csrc/cmake/func.cmake:289–312` |
| [V8] | `V/csrc/moe/chunk_fwd_o_vllm/op_host/CMakeLists.txt:1–31` |
| [V9] | `V/csrc/attention/kda_layout_swap12/op_host/CMakeLists.txt:14–20` |
| [V10] | `V/csrc/moe/chunk_fwd_o_vllm/op_kernel/chunk_fwd_o_vllm.cpp:16–39` |
| [V11] | `V/csrc/attention/kda_layout_swap12/op_kernel/kda_layout_swap12.cpp:170–191` |
| [V12] | `V/setup.py:196–203,282–324,399–424` |
| [V13] | `V/csrc/build_aclnn.sh:16–31,194–231,258–280` |
| [V14] | `V/csrc/torch_binding.cpp:17–37,84,2745–2749` |
| [V15] | `V/csrc/cmake/func.cmake:486–510,595–608,724–742,766–794` |
| [V16] | `V/csrc/attention/msa_index_score/op_host/CMakeLists.txt:19–39` |
| [V17] | `V/csrc/moe/situ_mx_quant/op_host/CMakeLists.txt:11–31` |
| [V18] | `V/csrc/build.sh:392–408,543–552,682–685` |
| [V19] | `V/csrc/cmake/config.cmake:38–44` |
| [V20] | `V/csrc/attention/CMakeLists.txt:13–28` |
| [C1] | `C/CMakeLists.txt:12–15,47–57,71–73` |
| [C2] | `C/examples/CMakeLists.txt:15–31,78–96,149–162` |
| [C3] | `C/examples/43_ascend950_basic_matmul/basic_matmul_tla.cpp:11–15,107–171` |
| [C4] | `C/include/catlass/gemm/device/device_gemm.hpp:65–84` |
| [C5] | `C/include/catlass/detail/kernel_adapter.hpp:17–30` |
| [C6] | `C/examples/advanced/basic_matmul_aclnn/op_kernel/catlass_basic_matmul.cpp:24–69` |
| [C7] | `C/examples/advanced/basic_matmul_aclnn/op_host/catlass_basic_matmul.cpp:18–74` |
| [C8] | `C/examples/advanced/basic_matmul_aclnn/CMakeLists.txt:30–48` |
| [C9] | `C/include/catlass/gemm/kernel/basic_matmul_tla.hpp:51–98` |
| [L1] | `L/flashinfer/runtime/schema.py:376–401` |
| [L2] | `L/flashinfer/jit/backends.py:76–110,197–210` |
| [L3] | `L/flashinfer/jit/core.py:56–85,118–127,210–228` |
| [L4] | `L/flashinfer/jit/build.py:27–49,52–94` |
| [L5] | `L/flashinfer/jit/loading.py:376–390` |
| [L6] | `L/flashinfer/jit/cache.py:212–218,242–252,303–348` |
| [L7] | `L/docs/framework_design.md` §2.3 |
| [L8] | `L/flashinfer/runtime/schema.py:918–950` |
| [E1] | `E/native.device.log`、`E/catlass.device.log`，命令从第 1 行起、末行退出码；`E/device_commands.json` |
| [E2] | `E/{native,catlass}.build.log:13`；`E/{native,catlass}.run.log:1–2`；`E/direct_commands.json` |
| [E3] | `E/opgen.log:1`；`E/{native,catlass}.package.binary.log:53–58`；`E/{native,catlass}.package.run.log:1–4`；`E/package_binary_commands.json` |
| [E4] | `E/final_artifacts.json`、`E/elf-inspection.log`；实际 readelf/nm 命令见 §4.5 |
| [E5] | `E/native-missing-catlass-{configure,build}.log`；`E/catlass-missing-headers-configure.log:12–18`；`E/optional_dependency_commands.json` |

## 附录 A：本次最小 direct 工程源码

以下文件全部在 E 下，不是对仓库源码的修改。它们让 §4.2、§4.3 的编译命令可复核；完整运行与标准包实验驱动保留在 E 中。

`src/native.cpp`：

```cpp
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void t13_probe(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    auto pa = reinterpret_cast<__gm__ float *>(a);
    auto pb = reinterpret_cast<__gm__ float *>(b);
    auto pc = reinterpret_cast<__gm__ float *>(c);
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 16; ++n) {
            float v = 0;
            for (int k = 0; k < 16; ++k) v += pa[m * 16 + k] * pb[k * 16 + n];
            pc[m * 16 + n] = v;
        }
    }
}
#include "launcher.inc"
```

`src/catlass.cpp`：

```cpp
// Analysis-only wrapper adapted from CATLASS 43_ascend950_basic_matmul.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "catlass/gemm/kernel/basic_matmul_tla.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
using namespace Catlass;
using namespace tla;
using DispatchPolicy = Gemm::MmadPingpong<Arch::Ascend950, true, false, 1, false, 2, 2, 2, 2>;
using L1TileShape = Shape<Int<256>, Int<256>, Int<128>>;
using L0TileShape = Shape<Int<256>, Int<256>, Int<32>>;
using Tag = layout::RowMajor;
using TileCopy = Gemm::Tile::PackedTileCopyTla<Arch::Ascend950, float, Tag, float, Tag, float, Tag, void>;
using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, float, float, float, void, TileCopy>;
using Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
using Kernel = Gemm::Kernel::BasicMatmulTla<BlockMmad, void, Scheduler>;
extern "C" __global__ __aicore__ void t13_probe(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    auto la = tla::MakeLayout<float, Tag>(16u, 16u);
    auto lb = tla::MakeLayout<float, Tag>(16u, 16u);
    auto lc = tla::MakeLayout<float, Tag>(16u, 16u);
    Kernel::Params params{GemmCoord{16,16,16}, a, la, b, lb, c, lc};
    Kernel kernel;
    kernel(params);
}
#include "launcher.inc"
```

`src/launcher.inc`：

```cpp
#ifdef T13_WITH_HOST
extern "C" void t13_launch(void *stream, uint8_t *a, uint8_t *b, uint8_t *c) {
    t13_probe<<<1, nullptr, stream>>>(a, b, c, nullptr, nullptr);
}
#endif
```

`src/CMakeLists.txt`（完成缺失依赖实验后的最终版）：

```cmake
cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(t13_analysis LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
set(PROFILE "native" CACHE STRING "native or catlass")
set(CATLASS_ROOT "/home/dy/projects/catlass-ref" CACHE PATH "Optional CATLASS root")
set_source_files_properties(${PROFILE}.cpp PROPERTIES LANGUAGE ASC)
add_library(t13_probe SHARED ${PROFILE}.cpp)
target_compile_definitions(t13_probe PRIVATE T13_WITH_HOST)
target_compile_options(t13_probe PRIVATE --npu-arch=dav-3510 -O2)
if(PROFILE STREQUAL "catlass")
  if(NOT EXISTS "${CATLASS_ROOT}/include/catlass/catlass.hpp")
    message(FATAL_ERROR "CATLASS headers required only by PROFILE=catlass")
  endif()
  target_include_directories(t13_probe PRIVATE ${CATLASS_ROOT}/include)
  target_compile_definitions(t13_probe PRIVATE CATLASS_ARCH=3510)
endif()
```

[U1]: /home/dy/projects/flashinfer-ori/flashinfer/jit/__init__.py:24
[U2]: /home/dy/projects/flashinfer-ori/flashinfer/jit/core.py:259
[U3]: /home/dy/projects/flashinfer-ori/flashinfer/jit/core.py:445
[U4]: /home/dy/projects/flashinfer-ori/flashinfer/jit/core.py:536
[U5]: /home/dy/projects/flashinfer-ori/flashinfer/jit/core.py:698
[U6]: /home/dy/projects/flashinfer-ori/flashinfer/aot.py:1310
[U7]: /home/dy/projects/flashinfer-ori/include/flashinfer/attention/prefill.cuh:19
[U8]: /home/dy/projects/flashinfer-ori/flashinfer/jit/attention/modules.py:530
[U9]: /home/dy/projects/flashinfer-ori/flashinfer/jit/attention/modules.py:2314
[U10]: /home/dy/projects/flashinfer-ori/flashinfer/jit/cpp_ext.py:135
[U11]: /home/dy/projects/flashinfer-ori/pyproject.toml:71
[U12]: /home/dy/projects/flashinfer-ori/build_backend.py:1102
[U13]: /home/dy/projects/flashinfer-ori/flashinfer/jit/activation.py:25
[U14]: /home/dy/projects/flashinfer-ori/include/flashinfer/activation.cuh:20
[U15]: /home/dy/projects/flashinfer-ori/include/flashinfer/attention/hopper/mainloop.cuh:10
[U16]: /home/dy/projects/flashinfer-ori/include/flashinfer/gemm/bf16_gemm_cutlass_template.h:23
[U17]: /home/dy/projects/flashinfer-ori/flashinfer/gemm/gemm_base.py:677
[U18]: /home/dy/projects/flashinfer-ori/flashinfer/prefill.py:445
[U19]: /home/dy/projects/flashinfer-ori/flashinfer/jit/cute_dsl_core.py:91
[U20]: /home/dy/projects/flashinfer-ori/flashinfer/jit/attention/modules.py:1773
[U21]: /home/dy/projects/flashinfer-ori/flashinfer/jit/core.py:826
[U22]: /home/dy/projects/flashinfer-ori/include/flashinfer/gemm/tgv_gemm_template.h:20
[U23]: /home/dy/projects/flashinfer-ori/flashinfer/jit/attention/modules.py:1683
[V1]: /home/dy/projects/vllm-ascend/csrc/build.sh:42
[V2]: /home/dy/projects/vllm-ascend/csrc/build.sh:1493
[V3]: /home/dy/projects/vllm-ascend/csrc/cmake/custom_build.cmake:278
[V4]: /home/dy/projects/vllm-ascend/csrc/CMakeLists.txt:33
[V5]: /home/dy/projects/vllm-ascend/csrc/cmake/func.cmake:31
[V6]: /home/dy/projects/vllm-ascend/csrc/cmake/obj_func.cmake:16
[V7]: /home/dy/projects/vllm-ascend/csrc/cmake/func.cmake:289
[V8]: /home/dy/projects/vllm-ascend/csrc/moe/chunk_fwd_o_vllm/op_host/CMakeLists.txt:1
[V9]: /home/dy/projects/vllm-ascend/csrc/attention/kda_layout_swap12/op_host/CMakeLists.txt:14
[V10]: /home/dy/projects/vllm-ascend/csrc/moe/chunk_fwd_o_vllm/op_kernel/chunk_fwd_o_vllm.cpp:16
[V11]: /home/dy/projects/vllm-ascend/csrc/attention/kda_layout_swap12/op_kernel/kda_layout_swap12.cpp:170
[V12]: /home/dy/projects/vllm-ascend/setup.py:196
[V13]: /home/dy/projects/vllm-ascend/csrc/build_aclnn.sh:16
[V14]: /home/dy/projects/vllm-ascend/csrc/torch_binding.cpp:17
[V15]: /home/dy/projects/vllm-ascend/csrc/cmake/func.cmake:486
[V16]: /home/dy/projects/vllm-ascend/csrc/attention/msa_index_score/op_host/CMakeLists.txt:19
[V17]: /home/dy/projects/vllm-ascend/csrc/moe/situ_mx_quant/op_host/CMakeLists.txt:11
[V18]: /home/dy/projects/vllm-ascend/csrc/build.sh:392
[V19]: /home/dy/projects/vllm-ascend/csrc/cmake/config.cmake:38
[V20]: /home/dy/projects/vllm-ascend/csrc/attention/CMakeLists.txt:13
[C1]: /home/dy/projects/catlass-ref/CMakeLists.txt:12
[C2]: /home/dy/projects/catlass-ref/examples/CMakeLists.txt:15
[C3]: /home/dy/projects/catlass-ref/examples/43_ascend950_basic_matmul/basic_matmul_tla.cpp:107
[C4]: /home/dy/projects/catlass-ref/include/catlass/gemm/device/device_gemm.hpp:65
[C5]: /home/dy/projects/catlass-ref/include/catlass/detail/kernel_adapter.hpp:17
[C6]: /home/dy/projects/catlass-ref/examples/advanced/basic_matmul_aclnn/op_kernel/catlass_basic_matmul.cpp:24
[C7]: /home/dy/projects/catlass-ref/examples/advanced/basic_matmul_aclnn/op_host/catlass_basic_matmul.cpp:18
[C8]: /home/dy/projects/catlass-ref/examples/advanced/basic_matmul_aclnn/CMakeLists.txt:30
[C9]: /home/dy/projects/catlass-ref/include/catlass/gemm/kernel/basic_matmul_tla.hpp:51
[L1]: /home/dy/projects/flashinfer/flashinfer/runtime/schema.py:376
[L2]: /home/dy/projects/flashinfer/flashinfer/jit/backends.py:76
[L3]: /home/dy/projects/flashinfer/flashinfer/jit/core.py:56
[L4]: /home/dy/projects/flashinfer/flashinfer/jit/build.py:27
[L5]: /home/dy/projects/flashinfer/flashinfer/jit/loading.py:376
[L6]: /home/dy/projects/flashinfer/flashinfer/jit/cache.py:212
[L7]: /home/dy/projects/flashinfer/docs/framework_design.md
[L8]: /home/dy/projects/flashinfer/flashinfer/runtime/schema.py:918
[E1]: /tmp/flashinfer-t13-analysis-20261023-bS8hFg/device_commands.json
[E2]: /tmp/flashinfer-t13-analysis-20261023-bS8hFg/native.build.log:13
[E3]: /tmp/flashinfer-t13-analysis-20261023-bS8hFg/native.package.binary.log:53
[E4]: /tmp/flashinfer-t13-analysis-20261023-bS8hFg/final_artifacts.json
[E5]: /tmp/flashinfer-t13-analysis-20261023-bS8hFg/optional_dependency_commands.json
