# 作者声明与构建资源整改验收

基线：`65c13f8`，任务起点已含设计文档和 Matmul 文档的未提交扩展，予以保留。
本报告覆盖框架整改；稀疏 Attention 算子的迁移应单独提交，
便于从 Git 记录区分框架维护和算子作者所需文件。

## 环境与设备

约定的解释器实际返回 `Permission denied`，退出码 126。使用已有 Torch 环境，
并在独立临时 venv 安装 Triton Ascend 与缺少的测试依赖；没有改写原环境的依赖。
设备自检输出为 **`True 8`**，八张卡实际都是 **Ascend910B3**，并非约定的 Ascend950PR。
CANN 实际版本为 9.2.0；Torch 为 2.9.0+cpu，TorchNPU 为 2.9.0.post9。

```bash
source /home/dy/tools/miniconda3/Ascend/ascend-toolkit/set_env.sh
/home/dy/tools/conda-envs/megamoe-a2/bin/python -B -c \
  'import torch,torch_npu;print(torch.npu.is_available(),torch.npu.device_count());print(torch.npu.get_device_name(4))'
# True 8
# Ascend910B3
```

临时测试解释器是 `/tmp/fi-author-test-env/bin/python`，安装了
`triton-ascend==3.6.0.post2`、`ml_dtypes` 等依赖；Triton 实测目标为
`GPUTarget(backend='npu', arch='Ascend910B3', warp_size=0)`。

## 改动与职责

| 入口 | 作用 |
| --- | --- |
| `flashinfer/operators/author_api.py:43` | `Support/DimRange` 和 Python 后端作者接口；自动计算源码摘要，省去序列化默认字段 |
| `flashinfer/ascendc/definition.py:67` | `NativeBinding/AscendCImplementation`：真实原生 ABI 与构建声明；具名维度复用公共定义 |
| `flashinfer/operators/authoring.py:124` | 发现作者目录，在 build 输出生成索引、Descriptor 与 recipe，不改测试预期 |
| `flashinfer/operators/catalog.py:14`、`flashinfer/operators/implementation.py:52` | 源码读取 Python 声明；安装读取并严格校验随包资源 |
| `flashinfer/_resources.py:28` | 源码、显式项目及安装包的统一资源位置规则 |
| `flashinfer/ascendc/_runtime.py:130` | 静态登记与冻结执行身份分离；首次使用时核对冻结资源，不在登记阶段探测 SDK |
| `setup.py:90` | 生成并组装包内 data；编译模式冻结当前环境，源码模式不编译 |
| `csrc/aclnn_torch_adapter/adapter.cpp:444` | 单独的字符串属性入口，保留原有 Scalar 入口与冻结 ABI 常量 |
| `flashinfer/attention/jit/`、`flashinfer/jit/attention.py` | 领域编译协议归入 Attention；历史导入使用同一组类型对象 |

86 个实现声明已经类型化，其中 85 个进入全局登记、1 个仍为显式实验实现。
47 个逻辑定义保持原值。整改前后逐项比较完整 Descriptor：除了四个原生源码 locator
去掉多余 `ascendc/` 层级，其余字段、capability、ABI、源码 digest 和 kernel ID 全部一致。
历史 v1 夹具的冻结字节保持不变。`QuantSpec` 继续位于 runtime，删除空的量化转导出包。

生成 JSON 不再进入源码 `flashinfer/data/` 或 Git。生产 recipe 输出到
`build/generated/csrc/recipes/`；显式探针仍为 `csrc/recipes/matmul_probe.json`。
不可派生的对标资料和外部契约移至 `tools/metadata/`，打包时复制。
作者职责与文件树见 [作者声明与资源交付](../authoring-and-resources.md)。

## 已执行验证

目录生成和精确元数据检查输出：

```text
Exact descriptor comparison: 85/85 unchanged except the declared native directory relocation; source digests unchanged
Logical operators: 47
Implementation declarations: 86
Generated package data absent from source tree
```

GMM 使用整改后的声明重新冷构建；并非借用旧 spec 的产物：

```bash
P=/home/dy/tools/conda-envs/megamoe-a2/bin/python
export FLASHINFER_CATLASS_ROOT=/tmp/flashinfer-catlass-ref
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11:/usr/include/c++/11/backward:/home/dy/tools/conda-envs/megamoe-a2/include
export LIBRARY_PATH=/home/dy/tools/conda-envs/megamoe-a2/lib
$P -B -m flashinfer.operators build \
  --kernel-id ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B \
  --cache-dir /tmp/fi-author-aot-cache --output-dir /tmp/fi-author-release
# built=1, failed=0, skipped=0; disk_cache_hit=false
```

由该产物构建 wheel 后，在干净 venv 强制安装，随后用纯路径依赖文件提供运行依赖。
已断言 `flashinfer.__file__` 位于安装 venv；安装登记数为 85。装后命令及实际输出：

```bash
/tmp/fi-author-wheel-env/bin/python -B -m flashinfer selftest \
  --kernel-id ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B
# PASS gemm.grouped_matmul_swiglu_quant: bundled AOT hit; compilation=disabled
# CPU reference matched; max_abs=[1.0, 7.275957614183426e-12]
# Summary: 1 PASS, 0 FAIL, 0 SKIP (total=1)
```

安装包的 `operators.json` 被故意损坏时，新的独立解释器明确拒绝
`invalid installed logical operator catalog`，恢复后继续可用；没有回退到源码扫描。
另从不含生成目录的 sdist 构建纯源码 wheel，拦截 SDK 探测和编译入口，构建成功。
该 wheel 在没有 Torch/TorchNPU 的干净 venv 中可登记全部 85 个实现；不依赖冻结 spec。
这验证了登记元数据与原生执行产物的边界，不代表未携带的二进制可以执行。

全量回归实际入口如下，包含 116 个 `real_compile` 用例；不另行重复同一编译矩阵：

```bash
P=/tmp/fi-author-test-env/bin/python
export FLASHINFER_TEST_NPU_DEVICE=4
export FLASHINFER_OPERATOR_TEST_ADAPTER=/tmp/fi-author-adapter
export FLASHINFER_OPERATOR_TEST_RELEASE=/tmp/fi-author-release
export FLASHINFER_AOT_CACHE_DIR=/tmp/fi-author-aot-cache
$P -B tools/run_regression.py --profile full --log-dir /tmp/fi-author-full-01
```

实际结果为 **2062 passed / 47 failed / 63 errors / 142 skipped**，pytest 与入口退出码均为 1。
从同一次 JUnit 结果按收集时的 marker 划分，`real_compile` 是
**38 passed / 44 failed / 34 errors / 0 skipped**，合计 116；没有再次运行相同编译矩阵。
这不是全量通过，也不能用零编译用例 skip 抵消失败。

其中三个非编译失败由本次资源边界调整引起，已经修复：默认 AOT 路径的精确预期、
CLI 的参考实现注入点，以及 tuning 交付测试绕过本仓构建入口而未生成安装索引的问题。
修复保留原有精确目录/输出断言和“安装查表不探测 provider/编译器”的断言。
生产 recipe 与显式 probe 分别取自生成目录和源码目录，继续比较完整精确配方集合。

```bash
$P -B tools/run_regression.py --profile quick --log-dir /tmp/fi-author-corrections-01 \
  tests/test_aot_discovery_index.py tests/test_matmul_bias_silu_registration.py \
  tests/test_tuning_release.py tests/test_ascendc_v2_public.py tests/test_author_api.py \
  tests/test_operator_authoring.py tests/test_operator_layout.py \
  tests/test_release_checks.py tests/test_cli.py
# 137 passed, 5 deselected, 0 skipped; exit_code=0
```

这次专项不代替全量。全量的上述原始数字保留，不能把修复后的专项结果拼成新的全量数字。
新增字符串属性 ABI 的队列 0/1/2 用例在全量中全部通过，实际跨 C++ 边界核对了字符串、
bool、int64、float64 的顺序、数值及存储生命周期；它是桥接验证，不是新 Attention 数值验证。
独立的干净源码副本从无冻结文件开始，单 kernel `build` 已成功生成完整的四条冻结元数据，
仅编译和导出所选 GMM，结果为 `built=1 / failed=0 / skipped=0`。

## 仍未验证与已知限制

- **Ascend950 的数值、图捕获及多后端矩阵未验证**：当前硬件为 Ascend910B3。
  失败包括目标 SoC 拒绝、950 设备夹具断言，以及固定为 950 的 probe 被 CANN 拒绝。
  没有扩大 capability、伪造硬件或把这些失败改为通过。
- 一项历史 v1 `host_shared` 真编译仍报 CANN 内部 eager-link stub 的
  `fatal error: 'cstdint' file not found`；未修改冻结 v1 构建常量来绕过它。
- 142 个 skip 不计成功：16 个要求 Ascend950 NPU，118 个要求已验证的 Ascend950，
  8 个要求已验证的 Ascend950 FIA 路径；原始原因保留在全量 JUnit 与日志中。
- **性能未测**；没有新增性能优势或图能力结论，既有图证据未扩大。
- 稀疏 Attention 的三后端接入不属于本次框架提交；精确登记补丁仍待按仓库约定确认。
  指定 transformer 仓库的 CANN 包装器限制为 Ascend950/960，当前设备无法给出其数值证据。

整改前“安装依赖源码中的生成 data 副本”的约束已不再成立：干净 sdist、纯源码 wheel、
严格安装加载及 GMM AOT 装后验证形成了独立证据。历史报告的其他硬件基线不因此被改写。
主设计文档相对任务起点的代码围栏数为 64→64，相对基线提交为 62→64；
接入指南为 26→28，没有删除既有代码块。
