# T13-II：BuildRecipe 驱动的标准 op-package

本实现提供独立注册的固定 FP32 `16×16×16` GEMM 正确性探针，不把它加入现有
79 条生产 descriptor，也不改变 auto、图或 attention 语义。native 与 CATLASS
继续只通过配置选择；CATLASS 探针实例化 `Arch::Ascend950`，不声明其它架构支持。

## 配方、构建与缓存

`csrc/recipes/matmul_probe.json` 中的同一种 `BuildRecipe` 同时声明
`host_shared / ascendc.host_shared.v1` 和
`op_package / ascendc.op_package.v1`。配置的 `output_kind` 决定产物形态，
`implementation` 决定 native/CATLASS；没有第二种 AOT recipe 或构建缓存。

op-package 另绑定 `host_sources`、`package_vendor`、`package_op_type` 和 ACLNN
`host_entry`，采用 `tiling_mode=fixed`，共享四字节 `FiMatmulProbeTilingData`。
host tiling 检查两个输入和输出均为 `16×16`，设置 blockDim=1、workspace=0；
device 核验 tiling 中的 size=256。direct host launcher 仍不消费 tiling。
所有源码、header、CMake 输入、工具链及选中依赖的内容摘要进入冻结配方与 JitSpec。

`flashinfer.aot.build.compile_and_package()` 与 JIT `build_and_load()` 共用
`_build_record()`：同一个 `AscendCBackend`、CANN 驱动、`DiskJitBuildStore`
锁、二次查询、staging、校验和原子发布。AOT build-only 不初始化 ACL 或要求构建机有 NPU。
source/include 快照与 CMake scratch 在发布前清除；native 不读取 CATLASS，也不运行 Git。

CANN `npu_op_*` API 生成 host 注册/tiling/ACLNN 库，显式执行 `binary package`
target。使用 `ENABLE_BINARY_PACKAGE=ON`、`ENABLE_SOURCE_PACKAGE=OFF`，不提供运行期
重编材料。收集 CPack 已 staging 的标准 `packages/vendors/<vendor>/...` 和 `.run`，
不执行安装器。对外的 `op-package.json` 是 `ArtifactFormat.ASCENDC_OP_PACKAGE`
主产物；store 原有的 `manifest.json` 对整个文件集作大小与 SHA-256 校验。

release 导出保留该 store manifest，去掉其本来就不参与校验的 `build.log`。
这是发行副本，不是另一个缓存；运行端直接只读校验发行目录，不要求构建缓存存在。
同一 spec 的发行目标不可被不同字节静默覆盖，已损坏的发行目录直接报错。

## 装载与调用

```python
from flashinfer.jit.op_package import load_op_package

# fresh process，必须在 aclInit / 其它 CANN 算子初始化之前调用
module = load_op_package(release_directory, target_soc="ascend950")
# 调用方初始化 ACL、选择 device、创建 stream 与三个 aclTensor*。
workspace_bytes, executor = module.get_workspace_size(a_tensor, b_tensor, c_tensor)
# 调用方分配 workspace（bytes=0 可为空），第二阶段消费 executor：
module.launch(workspace, workspace_bytes, executor, stream)
# 调用方同步、读取并检验结果、销毁 tensor/stream/workspace。
```

此过程只需要匹配的 CANN runtime/driver 及其共享库搜索路径，不需要编译器、
CATLASS、源树、JIT 缓存或 `.run` 安装。`target_soc` 必须等于发行 manifest 的完整
目标值；它是发行身份校验，实际设备执行还受 CANN 的设备与二进制兼容检查约束。
本轮只验证本机 CANN 9.2.0-beta.2、Linux x86-64，不承诺跨 CANN 版本 ABI。

装载器先完整校验文件 inventory、recipe/spec/目标、host ELF 的两个 ACLNN 导出、
`binary_info_config.json` 引用和每个 device ELF/JSON 的对应关系。随后设置当前进程
`ASCEND_CUSTOM_OPP_PATH` 到 package 内 vendor 根，显式 dlopen `libcust_opapi.so`。
两阶段 API 均检查非零状态和有效 executor，不吞掉错误；dlopen 成功仅表示装载成功。

`host_shared` 的入口是 `(stream, a_device, b_device, c_device)`；ACLNN 的第一阶段
接受 `aclTensor*`，返回 workspace 字节数和 `aclOpExecutor*`，第二阶段入口是
`(workspace, workspace_bytes, executor, stream)`。调用方拥有 tensor/stream/workspace
及其生命周期，不能把两套 ABI 的参数互传。探针要求输入输出为连续行主序 FP32 ND、固定尺寸；这是构建与
调用协议的样例，不是一般 GEMM API。

host library 必须是当前 host machine 的 ELF `ET_DYN`，package device `.o` 必须
是 NPU machine `0x1029` 的 `ET_EXEC`；直接 `-c` 的 `ET_REL` 会被拒绝。
CANN 会把源码入口重命名成 `<OpType>_<hash>_<tiling-key>`，验证器对照 CANN JSON 的
`kernelList` 和 ELF 实际定义符号，并复核其 SHA-256；不按扩展名或文件名 hash 选择装载器。
310P 的 unified AiCore JSON 没有 `core_type` 字段，使用其明确的
`coreType=AiCore / magic=RT_DEV_BINARY_MAGIC_ELF` 契约；A2/A3/A5 使用
`core_type=AIC / magic=RT_DEV_BINARY_MAGIC_ELF_AICUBE`。

CANN 的 vendor 注册与环境读取属于进程状态。首次装载前不得已经初始化其它 ACLNN
路径；本版主动拒绝在同进程切换另一个 package generation，避免假装支持热切换。
每个对照实验使用独立进程。

需要 PyTorch 注册接口时，可用 `flashinfer.jit.torch_library.matmul_probe_op()`
将上述已装载的 op-package，或 host_shared 的 `JitLoadedModule` 与 `JitSpec`，
转换为注册声明，再显式调用 `enable_torch_library(ascendc_ops=(op,))`。
它暴露 `torch.ops.flashinfer.matmul_probe(a, b, out)`，device kernel 使用
PrivateUse1，独立 meta 只校验输出元信息；注册不构建或装载产物。
新算子接入步骤、两条桥接契约和实际验证见 [T14 报告](reports/T14-report.md)。

## 发版与 SOC

```bash
PY=/root/miniforge3/envs/flashinfer-a5/bin/python
# 发版期编译；--soc 可重复，每个完整 SOC 有独立 spec 和发行目录。
$PY -m flashinfer.aot --config examples/ascendc_aot_native.json \
  --soc ascend950 --cache-dir /tmp/flashinfer-t13ii-cache \
  --output-dir /tmp/flashinfer-t13ii-release

# setup.py 直接接入相同 API，build 输出应放在仓库外。
FLASHINFER_AOT_CONFIG=examples/ascendc_aot_native.json \
FLASHINFER_AOT_SOCS=ascend950 \
FLASHINFER_AOT_CACHE_DIR=/tmp/flashinfer-t13ii-cache \
  $PY setup.py build --build-base /tmp/fi-build \
  bdist_wheel --dist-dir /tmp/fi-wheel

# 也可在没有编译器的打包机器上收集已经导出的包。
FLASHINFER_AOT_PREBUILT=/tmp/flashinfer-t13ii-release \
  $PY setup.py build --build-base /tmp/fi-prebuilt-build \
  bdist_wheel --dist-dir /tmp/fi-prebuilt-wheel
```

不设置 AOT 环境变量时，wheel 保持 `py3-none-any`，仅带原有 Python/数据和
`flashinfer/data/build/{csrc,kernels/ascendc}` 构建输入，不探测编译器。
AOT wheel 使用 `py3-none-linux_x86_64`（在相应 host 构建时采用其平台 tag），
把每个 verified generation 放在 `flashinfer/data/op_packages/<spec_sha256>/`，
包含标准 vendor 树、完整 manifests、receipt 和 `.run`。CANN runtime/driver 不随 wheel 分发。
选中 CATLASS 的构建机器须显式提供固定 submodule 或 `FLASHINFER_CATLASS_ROOT`；
wheel/sdist 均不复制 CATLASS 头文件。

sdist 是可在发版机器上构建的源码包：包括本仓 CMake、host/device 源码、JSON 配方和
示例；明确排除 `.so/.o/.run/.log`、op_packages 和 3rdparty 内容。仅制作 sdist
不编译。可在 sdist 解包后设置同样的 AOT 变量制作 wheel。

| requested SOC | CANN compute unit | npu_arch | 本轮验证边界 |
| --- | --- | --- | --- |
| ascend910b1 (A2) | ascend910b | dav-2201 | native 交叉编译；无设备执行证据 |
| ascend910_9391 (A3) | ascend910_93 | dav-2201 | native 交叉编译；无设备执行证据 |
| ascend310p1 | ascend310p | dav-2002 | native 交叉编译；无设备执行证据 |
| ascend950* (A5) | ascend950 | dav-3510 | native/CATLASS 在本机执行；产品子型号不作穷举保证 |

完整 requested SOC 留在 recipe/spec/cache/release 身份中，不因为 compute-unit 家族映射
而合并缓存。未知 SOC、无 CATLASS 配方的非 A5 目标直接报错，不自动换实现。

## 安装权限

本轮完全没有执行 `.run`。CANN 的生成安装脚本默认写 `ASCEND_OPP_PATH`
（本机是 `/usr/local/Ascend/.../opp`），写该系统目录需要相应权限；脚本支持
`--install-path=<absolute-directory>`，但本轮未测试安装模式，也不以安装作为装载前提。
直接使用 CPack 的 vendor 树和进程局部搜索路径，无需 root、不写系统 OPP、不修改 shell 配置。

## 实测与限制

环境为 `/root/miniforge3/envs/flashinfer-a5/bin/python`、本机 CANN
`/usr/local/Ascend/cann-9.2.0-beta.2`、A5 NPU 0。基线实测为 `3 failed, 1695 passed, 4 warnings in 44.17s`，三个失败均为既有
attention/quant/metadata ABI 冻结指纹；本任务不修改这些邻接协议。

`examples/ascendc_aot_a5.py --deny-compilers` 将 PATH 改为 `/usr/bin:/bin`，
以无特权 seccomp 规则令进程及子进程的 **所有 execve/execveat 返回 EPERM**，
先实测绝对路径 `bisheng --version` 确实被拒绝，再禁止 Python 的 subprocess、
`AscendCToolchain.discover` 和 `AscendCBackend.compile`。这是进程局部的实测屏蔽，
没有删除或重命名系统 SDK。随后完整执行 ACLNN、stream 同步和 256 个输出比较，
输出先填 NaN，防止空 kernel 或静默 no-op 被当成成功。


### 构建与无编译器运行

以下是实际执行的命令。所有目录均在仓外，且原生构建没有 CATLASS 根：

```bash
PY=/root/miniforge3/envs/flashinfer-a5/bin/python
$PY -m flashinfer.aot --config examples/ascendc_aot_native.json \
  --soc ascend950 --soc ascend950pr_9579 --soc ascend910b1 \
  --soc ascend910_9391 --soc ascend310p1 \
  --cache-dir /tmp/flashinfer-t13ii-cache --output-dir /tmp/flashinfer-t13ii-final-native
$PY -m flashinfer.aot --config examples/ascendc_aot_catlass.json \
  --soc ascend950 --cache-dir /tmp/flashinfer-t13ii-cache \
  --output-dir /tmp/flashinfer-t13ii-final-catlass \
  --catlass-root /home/dy/projects/catlass-ref
```

这六个构建均退出 0，均为 `disk_cache_hit=false`：

| SOC | 实现 | 完整发行目录字节数（含 .run） | 执行验证 |
| --- | --- | ---: | --- |
| ascend950 | native | 792127 | 本机通过 |
| ascend950pr_9579 | native | 792162 | 编译并校验；不另声明子型号运行证据 |
| ascend910b1 | native | 795660 | 无对应硬件，仅交叉编译和包校验 |
| ascend910_9391 | native | 795665 | 无对应硬件，仅交叉编译和包校验 |
| ascend310p1 | native | 795520 | 无对应硬件，仅交叉编译和包校验 |
| ascend950 | catlass | 798716 | 本机通过 |

```bash
N=/tmp/flashinfer-t13ii-final-native/8110eee484a23e34554e7618c1ec0137bec58455a20a169b02fe6ef9b60aa8ed
C=/tmp/flashinfer-t13ii-final-catlass/2b66e4f9a8b6276972527cc2ae3c6fd5a18972900f3107e6b3516d73ee3d0bed
$PY examples/ascendc_aot_a5.py --package "$N" --soc ascend950 --deny-compilers
$PY examples/ascendc_aot_a5.py --package "$C" --soc ascend950 --deny-compilers
```

native 实际输出：

```text
COMPILER_BLOCKED: absolute bisheng --version -> PermissionError errno=1
COMPILER_GUARD: PATH bisheng=None ccec=None; execve/execveat=EPERM; discovery/JIT=forbidden
LOAD_OK: ascendc_op_package entry=aclnnFiMatmulProbeNative
GetWorkspaceSize=0 bytes=0
aclnnFiMatmulProbeNative=0
PASS {"compilers_blocked": true, "device": 0, "first8": [-1.0, 9.0, -1.0, 4.0, -11.0, -1.0, 9.0, -1.0], "implementation": "native", "max_abs_error": 0.0, "shape": [16, 16, 16], "stream_sync": 0}
```

CATLASS 实际输出：

```text
COMPILER_BLOCKED: absolute bisheng --version -> PermissionError errno=1
COMPILER_GUARD: PATH bisheng=None ccec=None; execve/execveat=EPERM; discovery/JIT=forbidden
LOAD_OK: ascendc_op_package entry=aclnnFiMatmulProbeCatlass
GetWorkspaceSize=0 bytes=0
aclnnFiMatmulProbeCatlass=0
PASS {"compilers_blocked": true, "device": 0, "first8": [-1.0, 9.0, -1.0, 4.0, -11.0, -1.0, 9.0, -1.0], "implementation": "catlass", "max_abs_error": 0.0, "shape": [16, 16, 16], "stream_sync": 0}
```

还将两个 A5 wheel 分别通过 `pip install --no-deps --no-index --target` 安装到
`/tmp/flashinfer-t13ii-installed` 和 `/tmp/flashinfer-t13ii-installed-catlass`，
删除安装目录里的 `flashinfer/data/build`。在 `/tmp` 运行，以 `PYTHONPATH` 只指向对应
安装目录，设置 `FLASHINFER_NPU_CACHE_DIR=/tmp/flashinfer-t13ii-no-runtime-cache`。
用相同示例及 `--deny-compilers` 再次执行。两个进程均输出：

```text
PASS installed wheel; build_sources=absent; runtime_cache=absent
```

两个进程的 `RUNTIME_PACKAGE` 分别为上述安装目录内的 `flashinfer/__init__.py`，
输出仍为相同的 first8、`GetWorkspaceSize=0 bytes=0`、ACLNN status=0、
`stream_sync=0` 和 `max_abs_error=0.0`。检查确认未创建运行缓存。
实际完整输出留在仓外 `/tmp/flashinfer-t13ii-wheel-{native,catlass}-run.log`。

### wheel/sdist 产物检查

实际在 `/tmp/flashinfer-t13ii-packaging/src` 的源码快照执行；未在工作树内生成
build、wheel、sdist 或 egg-info：

```bash
$PY setup.py build --build-base /tmp/flashinfer-t13ii-packaging/pure-build \
  bdist_wheel --bdist-dir /tmp/flashinfer-t13ii-packaging/pure-bdist \
  --dist-dir /tmp/flashinfer-t13ii-packaging/pure-wheel \
  sdist --dist-dir /tmp/flashinfer-t13ii-packaging/sdist

FLASHINFER_AOT_CONFIG=examples/ascendc_aot_native.json \
FLASHINFER_AOT_SOCS=ascend950 \
FLASHINFER_AOT_CACHE_DIR=/tmp/flashinfer-t13ii-cache \
  $PY setup.py build --build-base /tmp/flashinfer-t13ii-packaging/aot-build \
  bdist_wheel --bdist-dir /tmp/flashinfer-t13ii-packaging/aot-bdist \
  --dist-dir /tmp/flashinfer-t13ii-packaging/aot-wheel

FLASHINFER_AOT_CONFIG=examples/ascendc_aot_native.json \
FLASHINFER_AOT_SOCS=ascend910b1,ascend910_9391,ascend310p1,ascend950 \
FLASHINFER_AOT_CACHE_DIR=/tmp/flashinfer-t13ii-cache \
  $PY setup.py build --build-base /tmp/flashinfer-t13ii-packaging/multisoc-build \
  bdist_wheel --bdist-dir /tmp/flashinfer-t13ii-packaging/multisoc-bdist \
  --dist-dir /tmp/flashinfer-t13ii-packaging/multisoc-wheel
```

| 产物 | 压缩字节数 | 解包文件字节数 |
| --- | ---: | ---: |
| 默认 `py3-none-any` wheel | 485054 | 2183682 |
| A5 native `py3-none-linux_x86_64` wheel | 909352 | 2979693 |
| A5 CATLASS prebuilt 同平台 wheel | 914009 | 2986301 |
| A2+A3+310P+A5 native 同平台 wheel | 2181126 | 5378187 |

A5 native `.run` 为 191294 字节、设备 ET_EXEC 为 6216 字节；CATLASS 对应为
193477 / 9936 字节。发行副本同时含解开的 vendor 树和压缩 `.run`，故会重复携带
一部分二进制。多 SOC 包按独立 generation 累加体积；不把 CANN runtime/driver 打入 wheel。
源码 sdist 约 0.7 MB，具体字节随说明文档更新而变化；最终产物在仓外 sdist 目录。

CATLASS prebuilt wheel 通过 `FLASHINFER_AOT_PREBUILT` 制作；实际把 PATH 限制到
`/usr/bin:/bin`、`ASCEND_HOME_PATH` 指到不存在的目录，并让 `subprocess.Popen`、
`shutil.which`、`AscendCToolchain.discover` 抛错，打包仍成功，输出：

```text
PASS prebuilt wheel packaging; compiler_probes=0 subprocesses=0
```

另将 sdist 解包，在其源码目录调用同样的 `setup.py` AOT native 构建；复用同一个
store，`disk_cache_hit=true`，所得 wheel 为 909352 字节，与上表 native wheel 一致。
ZIP/tar inventory 已检查：默认 wheel 和 sdist 没有 `.so/.o/.run/.log`；AOT wheel
有包内二进制及 `.run`，没有构建日志或 CATLASS headers。

### JIT 与回归

两组 JIT fresh build 及 cache-only 均实际退出 0：

```bash
$PY examples/ascendc_jit_a5.py --config examples/ascendc_native.json \
  --cache-dir /tmp/flashinfer-t13ii-final-jit-native \
  --catlass-root /tmp/flashinfer-t13ii-absent-catlass
$PY examples/ascendc_jit_a5.py --config examples/ascendc_catlass.json \
  --cache-dir /tmp/flashinfer-t13ii-final-jit-catlass \
  --catlass-root /home/dy/projects/catlass-ref
$PY examples/ascendc_jit_a5.py --config examples/ascendc_native.json \
  --cache-dir /tmp/flashinfer-t13ii-final-jit-native --cache-only
$PY examples/ascendc_jit_a5.py --config examples/ascendc_catlass.json \
  --cache-dir /tmp/flashinfer-t13ii-final-jit-catlass --cache-only
```

fresh build 输出节选（两次 `disk_cache_hit: False`；cache-only 则均为 True）：

```text
recipe: native language: ascendc output: host_shared dependencies: []
disk_cache_hit: False
LOAD_OK: shared_library entry: fi_matmul_probe_v1
PASS {"device": 0, "first8": [-1.0, 9.0, -1.0, 4.0, -11.0, -1.0, 9.0, -1.0], "kernel": "ascendc/t13/matmul_probe/native/Ascend950", "max_abs_error": 0.0, "shape": [16, 16, 16], "stream_sync": 0}
recipe: catlass language: ascendc output: host_shared dependencies: ['catlass']
disk_cache_hit: False
LOAD_OK: shared_library entry: fi_matmul_probe_v1
PASS {"device": 0, "first8": [-1.0, 9.0, -1.0, 4.0, -11.0, -1.0, 9.0, -1.0], "kernel": "ascendc/t13/matmul_probe/catlass/Ascend950", "max_abs_error": 0.0, "shape": [16, 16, 16], "stream_sync": 0}
```

```bash
$PY -m pytest -q
$PY -m pytest -q tests/test_ascendc_recipe.py tests/test_ascendc_op_package.py tests/test_jit_build.py
$PY tools/check_imports.py
$PY tools/t19_selection_audit.py --write /tmp/flashinfer-t13ii-selection-before.json
$PY tools/t19_selection_audit.py --compare /tmp/flashinfer-t13ii-selection-before.json
```

选型 baseline 在代码改动前生成。实际结果：

```text
baseline: 3 failed, 1695 passed, 4 warnings in 44.17s
current:  3 failed, 1723 passed, 4 warnings in 24.69s
65 passed in 3.35s
all modules: {"modules": 136, "failing": 0, "failures": []}
PASS all failure assertion outputs identical to baseline
PASS 79/79 auto selections unchanged; 79/79 descriptor fingerprints unchanged
PASS hot-path selections, fallback order and per-candidate filters unchanged
```

三个失败名称保持原样，失败断言输出也逐行相同：

- `tests/test_attention_metadata_wire.py::AttentionMetadataWireTests::test_config_directory_layout_and_canonical_blob_are_frozen`
- `tests/test_attention_quant_launch_abi.py::AttentionQuantizedKVLaunchABITests::test_v1_is_frozen_and_v2_has_an_independent_binary_identity`
- `tests/test_runtime_abi.py::KernelBinaryABITests::test_attention_binary_abi_matches_logical_order_mutation_and_stream`

import 纯净性另在独立进程实测，不能用“所有模块可导入”代替这个检查：
PATH 限制到 `/usr/bin:/bin`，CANN/CATLASS 根指到不存在的目录；MetaPathFinder
拒绝 `torch_npu`、`triton` 导入，`subprocess.Popen/run/check_output` 与 `shutil.which`
全部禁止。之后导入 flashinfer、jit、aot.build、op_package，并完成 79 条 auto 选型：

```text
PASS import flashinfer + jit + aot + op_package; torch_npu=False triton=False compiler_probes=0
PASS missing CANN/CATLASS; auto selection records=79
```

最终仓内扫描 `.so/.o/.a/.run/.log/.npubin` 均为空；
`3rdparty/catlass` 仍只有 mode `160000` 的 gitlink
`41bf90da655bba3c66d0acd7e00abe33960ecfd6`。`docs/tasks/`、`docs/TASKS.md`
未修改，没有 commit/push。

### 79 行实际选型对照

| record | before auto | after auto |
|---|---|---|
| cann/activation/gelu_and_mul_v1 | cann/activation/gelu_and_mul_v1 | cann/activation/gelu_and_mul_v1 |
| cann/activation/gelu_tanh_and_mul_v1 | cann/activation/gelu_tanh_and_mul_v1 | cann/activation/gelu_tanh_and_mul_v1 |
| cann/activation/silu_and_mul_v1 | cann/activation/silu_and_mul_v1 | cann/activation/silu_and_mul_v1 |
| cann/attention/batch_decode_paged_fia_v1 | cann/attention/batch_decode_paged_fia_v1 | cann/attention/batch_decode_paged_fia_v1 |
| cann/attention/batch_prefill_paged_fia_v1 | cann/attention/batch_prefill_paged_fia_v1 | cann/attention/batch_prefill_paged_fia_v1 |
| cann/attention/batch_prefill_ragged_fia_v1 | cann/attention/batch_prefill_ragged_fia_v1 | cann/attention/batch_prefill_ragged_fia_v1 |
| cann/attention/single_decode_fia_v1 | cann/attention/single_decode_fia_v1 | cann/attention/single_decode_fia_v1 |
| cann/attention/single_prefill_fia_v1 | cann/attention/single_prefill_fia_v1 | cann/attention/single_prefill_fia_v1 |
| cann/cascade/merge_state_attention_update_v1 | cann/cascade/merge_state_attention_update_v1 | cann/cascade/merge_state_attention_update_v1 |
| cann/cascade/merge_states_v1 | cann/cascade/merge_states_v1 | cann/cascade/merge_states_v1 |
| cann/decode/cudnn_batch_decode_fia_v1 | cann/decode/cudnn_batch_decode_fia_v1 | cann/decode/cudnn_batch_decode_fia_v1 |
| cann/fused_moe/cutlass_fused_moe_gmm_v1 | cann/fused_moe/cutlass_fused_moe_gmm_v1 | cann/fused_moe/cutlass_fused_moe_gmm_v1 |
| cann/fused_moe/fp8_block_scale_moe_cascade_v1 | cann/fused_moe/fp8_block_scale_moe_cascade_v1 | cann/fused_moe/fp8_block_scale_moe_cascade_v1 |
| cann/fused_moe/fp8_per_tensor_scale_moe_cascade_v1 | cann/fused_moe/fp8_per_tensor_scale_moe_cascade_v1 | cann/fused_moe/fp8_per_tensor_scale_moe_cascade_v1 |
| cann/gemm/bmm_bf16_v1 | cann/gemm/bmm_bf16_v1 | cann/gemm/bmm_bf16_v1 |
| cann/gemm/bmm_fp8_grouped_v1 | cann/gemm/bmm_fp8_grouped_v1 | cann/gemm/bmm_fp8_grouped_v1 |
| cann/gemm/gemm_fp8_nt_groupwise_block_v1 | cann/gemm/gemm_fp8_nt_groupwise_block_v1 | cann/gemm/gemm_fp8_nt_groupwise_block_v1 |
| cann/gemm/mm_bf16_v1 | cann/gemm/mm_bf16_v1 | cann/gemm/mm_bf16_v1 |
| cann/gemm/segment_gemm_v1 | cann/gemm/segment_gemm_v1 | cann/gemm/segment_gemm_v1 |
| cann/norm/fused_add_rmsnorm_aclnn_v1 | cann/norm/fused_add_rmsnorm_aclnn_v1 | cann/norm/fused_add_rmsnorm_aclnn_v1 |
| cann/norm/gemma_rmsnorm_aclnn_v1 | cann/norm/gemma_rmsnorm_aclnn_v1 | cann/norm/gemma_rmsnorm_aclnn_v1 |
| cann/norm/layernorm_aclnn_v1 | cann/norm/layernorm_aclnn_v1 | cann/norm/layernorm_aclnn_v1 |
| cann/norm/rmsnorm_aclnn_v1 | cann/norm/rmsnorm_aclnn_v1 | cann/norm/rmsnorm_aclnn_v1 |
| cann/page/append_paged_kv_cache_v1 | cann/page/append_paged_kv_cache_v1 | cann/page/append_paged_kv_cache_v1 |
| cann/rope/apply_rope_cos_sin_cache_aclnn_v1 | cann/rope/apply_rope_cos_sin_cache_aclnn_v1 | cann/rope/apply_rope_cos_sin_cache_aclnn_v1 |
| cann/rope/apply_rope_cos_sin_cache_out_v1 | cann/rope/apply_rope_cos_sin_cache_out_v1 | cann/rope/apply_rope_cos_sin_cache_out_v1 |
| cann/rope/apply_rope_inplace_v1 | cann/rope/apply_rope_inplace_v1 | cann/rope/apply_rope_inplace_v1 |
| cann/rope/apply_rope_pos_ids_inplace_v1 | cann/rope/apply_rope_pos_ids_inplace_v1 | cann/rope/apply_rope_pos_ids_inplace_v1 |
| cann/rope/apply_rope_pos_ids_v1 | cann/rope/apply_rope_pos_ids_v1 | cann/rope/apply_rope_pos_ids_v1 |
| cann/rope/apply_rope_v1 | cann/rope/apply_rope_v1 | cann/rope/apply_rope_v1 |
| cann/sampling/chain_speculative_sampling_v1 | cann/sampling/chain_speculative_sampling_v1 | cann/sampling/chain_speculative_sampling_v1 |
| cann/sampling/min_p_sampling_from_probs_v1 | cann/sampling/min_p_sampling_from_probs_v1 | cann/sampling/min_p_sampling_from_probs_v1 |
| cann/sampling/sampling_from_logits_v1 | cann/sampling/sampling_from_logits_v1 | cann/sampling/sampling_from_logits_v1 |
| cann/sampling/sampling_from_probs_v1 | cann/sampling/sampling_from_probs_v1 | cann/sampling/sampling_from_probs_v1 |
| cann/sampling/top_k_mask_logits_v1 | cann/sampling/top_k_mask_logits_v1 | cann/sampling/top_k_mask_logits_v1 |
| cann/sampling/top_k_renorm_probs_v1 | cann/sampling/top_k_renorm_probs_v1 | cann/sampling/top_k_renorm_probs_v1 |
| cann/sampling/top_k_sampling_from_probs_v1 | cann/sampling/top_k_sampling_from_probs_v1 | cann/sampling/top_k_sampling_from_probs_v1 |
| cann/sampling/top_k_top_p_sampling_from_probs_v1 | cann/sampling/top_k_top_p_sampling_from_probs_v1 | cann/sampling/top_k_top_p_sampling_from_probs_v1 |
| cann/sampling/top_p_renorm_probs_v1 | cann/sampling/top_p_renorm_probs_v1 | cann/sampling/top_p_renorm_probs_v1 |
| cann/sampling/top_p_sampling_from_probs_v1 | cann/sampling/top_p_sampling_from_probs_v1 | cann/sampling/top_p_sampling_from_probs_v1 |
| reference/activation/gelu_and_mul_v1 | reference/activation/gelu_and_mul_v1 | reference/activation/gelu_and_mul_v1 |
| reference/activation/gelu_tanh_and_mul_v1 | reference/activation/gelu_tanh_and_mul_v1 | reference/activation/gelu_tanh_and_mul_v1 |
| reference/activation/silu_and_mul_v1 | reference/activation/silu_and_mul_v1 | reference/activation/silu_and_mul_v1 |
| reference/attention/decode_v1 | reference/attention/decode_v1 | reference/attention/decode_v1 |
| reference/attention/prefill_ragged_v1 | reference/attention/prefill_ragged_v1 | reference/attention/prefill_ragged_v1 |
| reference/cascade/merge_state_v1 | cann/cascade/merge_state_attention_update_v1 | cann/cascade/merge_state_attention_update_v1 |
| reference/cascade/merge_states_v1 | reference/cascade/merge_states_v1 | reference/cascade/merge_states_v1 |
| reference/decode/cudnn_batch_decode_v1 | cann/decode/cudnn_batch_decode_fia_v1 | cann/decode/cudnn_batch_decode_fia_v1 |
| reference/fused_moe/cutlass_fused_moe_v1 | cann/fused_moe/cutlass_fused_moe_gmm_v1 | cann/fused_moe/cutlass_fused_moe_gmm_v1 |
| reference/fused_moe/trtllm_fp8_block_scale_moe_v1 | reference/fused_moe/trtllm_fp8_block_scale_moe_v1 | reference/fused_moe/trtllm_fp8_block_scale_moe_v1 |
| reference/fused_moe/trtllm_fp8_per_tensor_scale_moe_v1 | reference/fused_moe/trtllm_fp8_per_tensor_scale_moe_v1 | reference/fused_moe/trtllm_fp8_per_tensor_scale_moe_v1 |
| reference/gemm/bmm_bf16_v1 | cann/gemm/bmm_bf16_v1 | cann/gemm/bmm_bf16_v1 |
| reference/gemm/bmm_fp8_v1 | reference/gemm/bmm_fp8_v1 | reference/gemm/bmm_fp8_v1 |
| reference/gemm/gemm_fp8_nt_groupwise_v1 | reference/gemm/gemm_fp8_nt_groupwise_v1 | reference/gemm/gemm_fp8_nt_groupwise_v1 |
| reference/gemm/mm_bf16_v1 | cann/gemm/mm_bf16_v1 | cann/gemm/mm_bf16_v1 |
| reference/gemm/segment_gemm_v1 | cann/gemm/segment_gemm_v1 | cann/gemm/segment_gemm_v1 |
| reference/norm/fused_add_rmsnorm_v1 | cann/norm/fused_add_rmsnorm_aclnn_v1 | cann/norm/fused_add_rmsnorm_aclnn_v1 |
| reference/norm/gemma_fused_add_rmsnorm_v1 | triton/norm/gemma_fused_add_rmsnorm_v1 | triton/norm/gemma_fused_add_rmsnorm_v1 |
| reference/norm/gemma_rmsnorm_v1 | cann/norm/gemma_rmsnorm_aclnn_v1 | cann/norm/gemma_rmsnorm_aclnn_v1 |
| reference/norm/layernorm_v1 | cann/norm/layernorm_aclnn_v1 | cann/norm/layernorm_aclnn_v1 |
| reference/norm/rmsnorm_v1 | cann/norm/rmsnorm_aclnn_v1 | cann/norm/rmsnorm_aclnn_v1 |
| reference/page/append_paged_kv_cache_v1 | reference/page/append_paged_kv_cache_v1 | reference/page/append_paged_kv_cache_v1 |
| reference/rope/apply_rope_inplace_v1 | cann/rope/apply_rope_inplace_v1 | cann/rope/apply_rope_inplace_v1 |
| reference/rope/apply_rope_pos_ids_inplace_v1 | cann/rope/apply_rope_pos_ids_inplace_v1 | cann/rope/apply_rope_pos_ids_inplace_v1 |
| reference/rope/apply_rope_pos_ids_v1 | cann/rope/apply_rope_pos_ids_v1 | cann/rope/apply_rope_pos_ids_v1 |
| reference/rope/apply_rope_v1 | cann/rope/apply_rope_v1 | cann/rope/apply_rope_v1 |
| reference/rope/apply_rope_with_cos_sin_cache_out_v1 | cann/rope/apply_rope_cos_sin_cache_out_v1 | cann/rope/apply_rope_cos_sin_cache_out_v1 |
| reference/rope/apply_rope_with_cos_sin_cache_v1 | cann/rope/apply_rope_cos_sin_cache_aclnn_v1 | cann/rope/apply_rope_cos_sin_cache_aclnn_v1 |
| reference/sampling/chain_speculative_sampling_v1 | reference/sampling/chain_speculative_sampling_v1 | reference/sampling/chain_speculative_sampling_v1 |
| reference/sampling/min_p_sampling_from_probs_v1 | reference/sampling/min_p_sampling_from_probs_v1 | reference/sampling/min_p_sampling_from_probs_v1 |
| reference/sampling/sampling_from_logits_v1 | reference/sampling/sampling_from_logits_v1 | reference/sampling/sampling_from_logits_v1 |
| reference/sampling/sampling_from_probs_v1 | reference/sampling/sampling_from_probs_v1 | reference/sampling/sampling_from_probs_v1 |
| reference/sampling/top_k_mask_logits_v1 | reference/sampling/top_k_mask_logits_v1 | reference/sampling/top_k_mask_logits_v1 |
| reference/sampling/top_k_renorm_probs_v1 | reference/sampling/top_k_renorm_probs_v1 | reference/sampling/top_k_renorm_probs_v1 |
| reference/sampling/top_k_sampling_from_probs_v1 | reference/sampling/top_k_sampling_from_probs_v1 | reference/sampling/top_k_sampling_from_probs_v1 |
| reference/sampling/top_k_top_p_sampling_from_probs_v1 | reference/sampling/top_k_top_p_sampling_from_probs_v1 | reference/sampling/top_k_top_p_sampling_from_probs_v1 |
| reference/sampling/top_p_renorm_probs_v1 | reference/sampling/top_p_renorm_probs_v1 | reference/sampling/top_p_renorm_probs_v1 |
| reference/sampling/top_p_sampling_from_probs_v1 | reference/sampling/top_p_sampling_from_probs_v1 | reference/sampling/top_p_sampling_from_probs_v1 |
| triton/norm/gemma_fused_add_rmsnorm_v1 | triton/norm/gemma_fused_add_rmsnorm_v1 | triton/norm/gemma_fused_add_rmsnorm_v1 |
