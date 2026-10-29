# Ascend C 编译选项与自定义算子接入

本文以本仓 `117db20`、CANN 9.2.0、Ascend950、Python 3.12 的实际构建为依据；对照本机 vLLM-Ascend checkout `96a69d793457917d5d8fdb6b3b4e18f6f372297f`。选项分别来自 recipe、公共驱动和 CANN SDK，不能只看一处 `-O2` 就推断全部编译阶段。实际 argv、冷构建记录及验证范围见 [分层与编译配置核查](reports/layering-build-options-2026-10-27.md)。

## 1. 本仓的配置入口

新增生产算子采用完整的 `BuildRecipe` 契约，即 `schema_version=2`。配置写在 `csrc/recipes/<op>.json`，经冻结声明进入生产构建；AOT、冷 JIT 共用 `flashinfer/jit/ascendc.py`，没有按算子另开编译器流程。

| 配置 | 当前作用与落点 |
| --- | --- |
| `sources` / `headers` / `host_sources` | 明确设备 TU、头文件、host OpDef/tiling TU，并校验每个 SourceRef 的 SHA-256；不靠递归发现决定参与编译的算子 |
| `include_dirs` | 相对源码根的本仓 include 目录；驱动复制源码闭包后换成私有构建路径 |
| `dependencies` | CATLASS 固定 revision、include 目录及实际头树摘要；native 不读隐式 CATLASS 目录 |
| `definitions` | 算子自己的宏，如 `CATLASS_ARCH=3510`；`FI_` 前缀由框架保留，不能在 recipe 中重定义 |
| `compile_options` | 与 `JitSpec.compile_options` 顺序连接；`op_package` 路径传给设备 target，**不直接传给 host target** |
| `link_options` | `host_shared` 的链接选项；`op_package` 的 codegen 与 host api/proto/tiling 库链接选项。当前两条生产 recipe 均为空 |
| `target_soc` / `npu_arch` | 产品目标及架构契约；驱动检查二者匹配，不能只改一个字符串冒充新 SoC 支持 |
| `output_kind` / `output_adapter` | 新生产算子使用 `op_package` / `ascendc.op_package.v1`；后者是产物适配格式标识，不是 recipe v1 |
| `tensor_parameters` / `shape_constraints` | 输入输出的有序 Tensor ABI、dtype、shape 符号及有限支持域；不是设备编译器选项 |
| `tiling_mode` / `tiling_host_entry` | `host_dynamic` 与实际 host callback；在调用时按输入 shape 计算 tiling，不为每个 shape 重编 kernel |
| `workspace_bytes` / `workspace_host_entry` | 非负固定值或 `dynamic`；动态值由 ACLNN 第一段返回实际大小 |
| `task_type` / `sync_policy` | 混合核与同步的声明和校验；**不会自动变成 `--cce-auto-sync=off`**，实际 task 宏及同步逻辑仍须在源码中实现 |

实现位置：[recipe 校验与指纹](../flashinfer/jit/recipe.py)、[驱动组装](../flashinfer/jit/ascendc.py)、[设备选项传递](../csrc/cmake/op_package/kernel/CMakeLists.txt)、[host 生成与链接](../csrc/cmake/op_package/host/CMakeLists.txt)。

## 2. 当前两条生产算子的明确配置

| 项目 | MatmulBiasSiLU | FlashAttention |
| --- | --- | --- |
| recipe | `csrc/recipes/matmul_bias_silu.json` | `csrc/recipes/flash_attention.json` |
| 实现 / dtype | native / BF16 | CATLASS / FP16 |
| 目标 / recipe 架构 | `Ascend950` / `dav-3510` | `Ascend950` / `dav-3510` |
| 算子编译选项 | `-O2` | `-O2` |
| 算子宏 | 无额外宏 | `CATLASS_ARCH=3510` |
| 算子 include | `csrc/include` | `csrc/kernels/ascendc/flash_attention` |
| 外部依赖 | 无 | CATLASS `41bf90da655bba3c66d0acd7e00abe33960ecfd6` 的 include 树 |
| spec 额外编译选项 / recipe 链接选项 | 均为空 | 均为空 |
| 产物与契约 | `op_package`、recipe v2、tiling schema 2 | 同左 |
| task / 同步声明 | `aic_aiv` / `cross_core_flags` | 同左 |
| host tiling / workspace callback | `FiMatmulBiasSiluTiling` | `FiFlashAttentionTiling` |
| device entry | `fi_matmul_bias_silu` | `fi_flash_attention` |
| host entry | `aclnnFiMatmulBiasSilu` | `aclnnFiFlashAttention` |
| package op type | `FiMatmulBiasSilu` | `FiFlashAttention` |

两条源码都显式使用 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`。recipe 的 `block_dim=1` 是构建声明中的值；完整契约的实际 launch block 数仍由 host callback 的 `SetBlockDim()` 设置。FA 按 heads、S 及设备核数决定 block 数，不能从 `FI_BLOCK_DIM=1` 推断所有输入都只启动一个 block。

公开探针 `matmul_probe` 已统一为 recipe v2，native/CATLASS 的显式优化选项仍均为 `-O2`，CATLASS 多 `CATLASS_ARCH=3510`。它的多 SoC recipe 是编译/ABI 探针声明，不能作为两条生产算子的跨 SoC 支持证据。旧 v1 与裸指针源码仅在内部冻结测试设施保留。

## 3. 公共驱动注入哪些参数

驱动先形成：

```text
设备编译选项 = recipe.compile_options + spec.compile_options
公共宏 = FI_DEVICE_ENTRY + FI_HOST_ENTRY + FI_BLOCK_DIM
op_package 另加 FI_OP_PACKAGE=1 + FI_PACKAGE_OP_TYPE + FI_COMPUTE_UNIT
```

`FI_COMPUTE_UNIT` 为归一化 compute unit，当前生产为 `ascend950`。`host_shared` 才直接在选项首部添加 `--npu-arch=<recipe.npu_arch>`；`op_package` 通过 CANN 的 `ASCEND_COMPUTE_UNIT` 选择目标，再由 SDK 生成具体编译命令。

当前 CMake 配置参数为：

| 公共配置 | 当前值 |
| --- | --- |
| ASC 编译器 | 当前 CANN 的 `bin/bisheng` |
| host 编译器 | 发现并校验的 `c++`；实际位置及二进制摘要计入工具链身份 |
| C++ 标准 / 构建模式 | C++17 / `CMAKE_BUILD_TYPE=Release` |
| SDK / 目标 | `ASCEND_CANN_PACKAGE_PATH=<CANN>`、`ASCEND_COMPUTE_UNIT=ascend950` |
| Python | `ASCEND_PYTHON_EXECUTABLE=<当前构建解释器>` |
| 包类型 | `ENABLE_SOURCE_PACKAGE=OFF`、`ENABLE_BINARY_PACKAGE=ON` |
| 失败策略 | `ASCEND_SKIP_FAILED_COMPUTE_UNIT=OFF`，不能把目标编译失败当作跳过成功 |
| vendor | recipe 的 `package_vendor` |
| 编译目标 / 并行度 | 普通 build，随后 `binary package`；均 `--verbose -j2` |

host 侧由 `npu_op_code_gen` 生成 ACLNN API 和注册代码，再构建 `cust_opapi`、`cust_op_proto`、`cust_optiling`；本仓显式加 `-fvisibility=hidden`，其他基础选项由 CANN target 决定。设备侧通过 `npu_op_kernel_options` 接收上述设备选项、宏和 include。配置为 Release 不足以推断最终 host 一定用 `-O3`。

这些 host target 是 CANN 的调用/注册/tiling 库；完整 wheel 提供本仓 Torch/ACLNN C++ 适配库，两条生产接口统一通过 Torch 注册及该桥调用它们；Python 函数只做自动注册与参数转发。vLLM-Ascend 另有 Torch/pybind 的 `vllm_ascend_C` 扩展，其 Torch include、链接库和 C++ ABI 选项不能直接当作本仓设备 target 的默认配置。

驱动清除 `CPATH`、`CPLUS_INCLUDE_PATH`、`C_INCLUDE_PATH`、`LIBRARY_PATH`、`CXXFLAGS`、`CFLAGS`、`LDFLAGS`、`ASCEND_CUSTOM_OPP_PATH`，避免外部隐式选项或 vendor 污染构建。临时 `export CXXFLAGS=...` 不是本仓为算子配置选项的入口。

## 4. CANN 9.2.0 实际追加的参数

本次以全新缓存构建两条生产算子，并通过 `strace` 记录子进程 `execve`。这是当前 SDK/目标下的观察值，升级 SDK 后须重新核对，不将这些选项全部硬编码进公共驱动。

| 阶段 | 观察到的参数 |
| --- | --- |
| 设备预编译及主核编译 | 参数中先有 SDK 的 `-O3`，随后为 recipe 的 `-O2`；还有 `-std=c++17`、dtype/format 宏及公共 `FI_` 宏 |
| cube / vector | 分别使用 `--cce-aicore-arch=dav-c310-cube`、`--cce-aicore-arch=dav-c310-vec`，并带 `--cce-aicore-only` |
| 同步 | 实际命令带 `--cce-auto-sync`，本仓未显式改成 vLLM 示例中的 `--cce-auto-sync=off` |
| 混合核实例 | SDK 追加 `__MIX_CORE_MACRO__=1`、`WORKSPACE_PARAM_OFFSET=4`、tiling key 和自动生成的 kernel 符号宏 |
| 模板 tiling-key 提取辅助步骤 | 单独观察到 `-O3 -std=c++17 -DTILING_KEY_VAR_0=0UL`；不能把该步骤与主 kernel 编译视为同一条命令 |
| host 库 | 本机实际使用 `-std=gnu++17 -O2 -fPIC -fvisibility=hidden`，并有 SDK 的 ABI、栈保护、FORTIFY 等参数 |

`dav-3510`、`CATLASS_ARCH=3510`、SDK argv 的 `dav-c310-cube/vec` 属于不同层的目标表达，不能将三者混为一个命令行开关。`HIGH_PERFORMANCE` 等 SDK 宏名也不构成本仓的性能测量证据。

完整 argv 见核查记录；未验证其他 CANN 版本产生相同选项，也未测量替换优化级别或同步选项后的数值、性能及图模式。

## 5. 对照 vLLM-Ascend 怎样接入一个自定义算子

本机参考路径均相对于仓库根目录：

| 接入点 | vLLM-Ascend 的实际做法 | 本仓对应做法 |
| --- | --- | --- |
| 源码组织 | `../vllm-ascend/csrc/<域>/<op>/op_host`、`op_kernel`，算子目录 CMake 加入编译列表 | 设备放 `csrc/kernels/ascendc/<op>/`，host 放 `csrc/op_package/<op>.cpp`；由 recipe 显式列出 |
| SoC 选择 | `csrc/build_aclnn.sh` 的对应 SoC 分支选 `CUSTOM_OPS_ARRAY`，再以 `build.sh --pkg --ops=... --soc=...` 构建 | recipe 的目标/依赖/支持域与冻结声明共同决定构建；新生产条目进入登记后被普通 wheel 的全量生产构建包含 |
| 算子级编译选项 | `add_ops_compile_options(OP_NAME ... COMPUTE_UNIT ... OPTIONS ...)`，有全局选项和按 SoC 分支的特例 | `definitions`、`include_dirs`、`compile_options` 写入该算子 recipe；同一驱动传递并计入身份 |
| host tiling | OpDef 与 host callback、tiling 结构、workspace 配置共同交付 | v2 的 callback 字段必须与实际 host 源码及设备 tiling 结构一致 |
| Python 调用边界 | `torch_binding.cpp` 注册 schema 和 `PrivateUse1` 实现，经 adapter 调 ACLNN | Torch schema 的 PrivateUse1 实现统一负责选型；已验证的 `package_tensor_runner` 经 Torch C++/OpCommand 适配到 ACLNN；不是自动生成同一套 C++ typed export |
| Meta / 图 | `torch_binding_meta.cpp` 提供 Meta/FakeTensor 形状推导 | Meta、动态 tiling 与图 capture/replay 分别验收；Meta 注册成功本身不证明可入图 |

具体例子：

- `../vllm-ascend/csrc/moe/rms_norm_cast/op_host/CMakeLists.txt` 为 `RmsNormCast` 加 `--cce-auto-sync=off -Wno-deprecated-declarations`；其当前构建列表位于 A2/A3 分支，不能直接当作 A5 选择示例。
- `../vllm-ascend/csrc/attention/attn_res_fwd/op_host/CMakeLists.txt` 对 A5 的 `AttnResFwd` 另外加 `-mllvm -cce-aicore-dcci-before-kernel-end=false`，其他目标不加。该算子位于当前 A5 列表。这说明特例须限定到具体算子/目标，不能复制成所有算子的默认值。
- vLLM-Ascend 根 `CMakeLists.txt` 的直接 `csrc/kernels/*.cpp` 库在 `ascend950` 分支会跳过；它与 `build_aclnn.sh` 的自定义 ACLNN 包是不同构建入口，不能用前者的 glob 推断后者已接入某个算子。

因此，在本仓新增生产算子需要完成以下落点，不能只放入一个 `.cpp`：

1. 明确 dtype、shape/envelope、数学语义与数值参考，交付设备 TU、host OpDef/动态 tiling/workspace、共享 tiling 结构及必要 include。
2. 新增完整 v2 recipe，列齐源码闭包和 SHA-256；填写编译选项、target、宏、依赖、Tensor 顺序、shape 约束及 callback。`aic_aiv` 必须在源码落实对应同步。
3. 按本仓已有发布 helper 组织 `tools/build_<op>.py`，冻结 descriptor/recipe/spec；修改生产精确集合断言前按仓库规则取得明确授权，保持断言强度。
4. 接入公共 API/runner，确认 CANN API 名、参数顺序、dtype、out、workspace、executor 和 stream；这一步相当于竞品补齐 binding/adapter，不能由编译成功代替。
5. 冷编译、AOT 禁编译命中、允许/禁止 JIT 的缺失、损坏拒绝、越界拒绝、依赖缺失全部验收；再按普通 wheel 流程构建并隔离安装 selftest。图模式另做 capture/replay 后才更新能力声明。

完整交付顺序见 [新增算子指南](adding-a-custom-operator.md)。本次没有新增生产 kernel、改动设备 recipe 的编译参数或放宽冻结契约。

适配竞品算子时还要逐个核对 scalar、attribute、optional Tensor、Tensor list 等参数。例如其 RMSNorm binding 有 `epsilon` 标量；本仓当前完整契约落实的是有序 Tensor ABI，尚不能据此宣称任意新增标量/属性 ABI 都已通用支持。超出已表达范围的参数需要单独扩展声明和校验，不能靠编译选项或手写默认常量绕开。

## 6. 调整编译选项时怎样核对

### 独立的 Torch/ACLNN host 适配库

本仓参考竞品 `csrc/aclnn_torch_adapter/op_api_common.h`，在 `csrc/aclnn_torch_adapter/adapter.cpp` 接通 Tensor 转换、两段式 API、workspace、当前 stream 和 `OpCommand`。这属于 Torch host 调用层，构建配置与上述设备编译分开：

| 配置 | 当前值 |
| --- | --- |
| 编译入口 | `flashinfer/aclnn/build.py`；普通 wheel / CONFIG 构建调用 |
| 编译器与选项 | `c++ -std=c++17 -O2 -g0 -fPIC -shared` |
| C++ ABI | `_GLIBCXX_USE_CXX11_ABI` 取当前 Torch 构建值 |
| 头文件 | Torch、TorchNPU、CANN、系统 libffi 开发头文件 |
| 链接 | `torch`、`torch_cpu`、`c10`、`torch_npu`、`ascendcl`、`ffi`、`dl` |
| 产物 | `flashinfer/data/aclnn/libflashinfer_aclnn.so` 和 SHA-256 / Torch 版本 manifest |
| 非编译模式 | 源码模式需另行构建桥；PREBUILT 验证并复制预编译桥，不调用编译器 |

适配库不复制竞品的私有 NPU storage 布局，使用 TorchNPU 格式 API；保留同步 eager 语义。任意 scalar/attribute 转换、异步返回及图 capture/replay 未验证，详细边界与调用方式见 [Torch/ACLNN 接入](aclnn-torch-adapter.md)。

### 算子 recipe 选项核对

业务选项变更应进入 recipe/spec 身份，不能只手改生成目录、SDK 文件或环境变量。生产配方、源码或选项变化后，按仓库规则审核并重新冻结声明；现有匹配门禁不能绕过。

查看随包的 `ascendc-recipe.json` 时，`resolved.recipe.compile_options` 才是 recipe 的选项；receipt 顶层的 `compile_options` 只记录 spec 额外选项。当前顶层为 `[]` 不表示没有应用 `-O2`，也不列出 SDK 自动追加的全部参数。`commands` 记录 CMake 调用，设备编译器的完整展开参数应从实际 argv 核对。

冷构建可以保存原始 argv，以下与本次核查使用相同方式：

```bash
mkdir -p ../../var/flashinfer
FI_BUILD_EVIDENCE="$(mktemp -d ../../var/flashinfer/options-XXXXXXXX)"
strace -f -qq -e trace=execve -s 65535 -o "$FI_BUILD_EVIDENCE/execve.log" \
  python3 -B -m flashinfer.aot --all \
  --cache-dir "$FI_BUILD_EVIDENCE/cache" --output-dir "$FI_BUILD_EVIDENCE/release" \
  > "$FI_BUILD_EVIDENCE/build.log" 2>&1
FI_BUILD_RC=$?
printf '%s\n' "$FI_BUILD_RC" > "$FI_BUILD_EVIDENCE/build.exitcode"
```

执行前激活匹配的构建环境、加载 CANN，并设置固定的 `FLASHINFER_CATLASS_ROOT`；确认没有禁编译策略。分别检查 CMake 配置、host、设备预编译、主核编译、提取辅助步骤与链接命令，保留参数先后顺序。缓存命中时没有新编译 argv，不能据此判断新选项已生效。
