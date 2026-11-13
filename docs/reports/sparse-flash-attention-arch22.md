# SparseFlashAttention arch22 移植与 A2 验证

工作区基线为 `f493adb`。上游为 vllm-ascend 的
`csrc/attention/sparse_flash_attention/`，revision
`535ae0bb911cb378819f52fcf2e7c8b67fde3f51`；移植前该路径没有未提交改动。
本次只登记 Ascend910B 的 Ascend C AOT 实现。移植验收时保留工作区供审核，
随后根据用户授权提交代码。

## 公共接口与文件职责

公共入口由框架从 `OPERATOR` 自动导出，两个 Python 名称指向同一函数：

```python
flashinfer.sparse_flash_attention(
    query, key, value, sparse_indices, scale_value,
    block_table=None, actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
    query_rope=None, key_rope=None, sparse_block_size=1,
    layout_query="BSND", layout_kv="BSND", sparse_mode=3,
    pre_tokens=9223372036854775807, next_tokens=9223372036854775807,
    attention_mode=2, return_softmax_lse=False,
    out=None, backend="ascendc", kernel_id=None,
)
# 同一入口：flashinfer.attention.sparse_flash_attention
# 返回：(attention_out, softmax_max, softmax_sum)
```

固定 MLA absorb 语义：Q/K/V 的 D=512、KV head=1，可选 RoPE 分量 Dr=64；
Q/K/V 使用相同 FP16 或 BF16 类型。mode=0 为非因果，mode=3 为右下角对齐的因果遮罩。
索引按 KV block 编号，`-1` 必须集中在有效索引之后；块、长度和页表越界在原生调用前拒绝。
BSND 长度为各 batch 的有效长度，TND 长度为前缀和；分页 KV 长度为各 batch 的有效长度。

默认只计算 attention 输出，后两个公共输出为 FP32 `(0,)`；启用辅助输出时，BSND 为
`[B,1,Sq,H]`，TND 为 `[1,Tq,H]`。输入及自供 out 必须连续且物理格式为 ND；
四维 TorchNPU 张量可能默认标记为 NCHW，调用方须显式用
`torch_npu.npu_format_cast(tensor, 2)` 准备输入。框架分配 ND 输出，保留自供 out 的身份及存储。

| 文件 | 作用与作者责任 |
| --- | --- |
| `flashinfer/operators/attention/sparse_flash_attention/operator.py:151` | 手写公共 schema、输出推导和语义校验；不注册特定后端 |
| 同目录 `cases.py` | 手写独立 CPU golden 和 14 组样例，提供框架通用验收及装后 selftest |
| `flashinfer/ascendc/attention/sparse_flash_attention/implementation.py:17` | 手写 Ascend910B Capability、12 个原生 Tensor 参数和 9 个属性、编译输入及 runner 工厂 |
| 同目录 `runner.py` | 手写关闭辅助输出时的两个内部占位缓冲适配；复用通用 ACLNN runner |
| `csrc/kernels/attention/sparse_flash_attention/op_kernel/sparse_flash_attention.cpp:16` | 统一入口和 arch22 编译护栏，原生入口 `fi_sparse_flash_attention` |
| 同目录 `op_kernel/arch22/` | 移入上游 MLA/Cube/Vector 三个实现头；修复实际选中 token 的工作量计算 |
| 同目录 `op_host/sparse_flash_attention_def.cpp` | 适配独立 `FiSparseFlashAttention` OpDef，生成 `aclnnFiSparseFlashAttention` 原生 API |
| 同目录 `op_host/sparse_flash_attention_infershape.cpp` | 移入并绑定输出推导和输出类型 |
| 同目录 `op_host/arch22/` | 移入平台 tiler，查询核数、计算动态 workspace、选择 C 模板 |
| 同目录 `op_kernel/sparse_flash_attention_common.h`、模板 tiling key 头 | 上游共享类型及模板组合，后者只编译本次启用的 C 模板与有效布局组合 |
| 同目录 `op_host/fi_host_checks.h`、`CMakeLists.txt` | 本地诊断宏及此算子的编译选项，不依赖 vLLM runtime |
| `tests/test_sparse_flash_attention.py` | 接口/FakeTensor/拒绝路径/runner 单测，以及额外的 A2 地址与空行用例 |

框架自动发现并生成公共导出、Torch functional/out/FakeTensor 接线、Descriptor、源摘要、
recipe、冻结 spec 和随包目录索引。生成文件在 `build/generated/`，wheel 内在
`flashinfer/data/`；这些 JSON 不作为本次新增源码维护。没有修改中央 dispatcher、
registry、领域 `__init__.py` 或 selftest 登记表。版权保留在移入源码，来源登记在
`THIRD_PARTY_NOTICES.md`，许可使用根目录的统一副本。

## 原生适配与框架补齐

上游 V 聚合分支复用 K 缓存计算 P@V，不能据此声称接受独立 K/V。本次统一使用原有 C
模板，不启用 V 聚合分支；runner 无需探测架构或添加额外的公共/私有属性。
`op_host/arch22/sparse_flash_attention_tiling.cpp:284` 是模板选择位置。

`op_kernel/arch22/sparse_flash_attention_kernel_mla.h:343` 按每行实际有效且经过因果遮罩的
选中 token 数计算 S2 工作量，避免全 `-1` 行产生 NaN、尾部填充覆盖有效辅助统计；
`:973` 扫描完整索引行，避免把因果阈值误当成有效索引槽数量。
迁移中观测到的长序列辅助最大值约 `-2e38` 和全空行 NaN 均由这些边界用例暴露。
没有放宽数值容差来消除失败。

新增算子暴露了四处通用框架缺口，修改原因如下：

| 框架文件 | 修改原因 |
| --- | --- |
| `flashinfer/jit/op_package.py:172` | 已定义的 string 属性补齐 CANN `str` 验包映射、ctypes `c_char_p` 及 UTF-8 编码；继续精确检查属性名称、顺序与类型 |
| `flashinfer/jit/recipe.py:393` | 空 optional Tensor 没有 dtype，使用所有有值 Tensor 共同匹配一条已声明的合法组合；不增加虚构的可执行 dtype 组合 |
| `flashinfer/operators/runtime.py:13` | 四维新输出显式分配为 ND；不转换输入，也不替换调用方 out |
| `flashinfer/operators/testing.py` | 验收 out 同样分配为 ND；空输出先核对精确形状，再以误差 0 记录，保留全部数值与身份断言 |

全量回归进一步暴露了直接算子与 AttentionPlan 能力绑定的边界。
`flashinfer/operators/capability.py` 从公共 schema、backend、输出名及 `CapabilitySpec` 派生
`domain="operator"` 的绑定，关联具体实现 ID；Ascend C 与 Python 后端作者入口自动添加该绑定。
`flashinfer/attention/capability.py` 验证这种直接绑定后，再按原路径处理 AttentionPlan profile。
原有 POD ABI、profile 指纹、规则覆盖和缺失绑定的拒绝校验保持不变；新增测试同样拒绝
直接绑定的错误指纹、错误接口及缺失绑定。作者不填写 profile digest，也不维护额外 JSON。

`tests/test_ascendc_contract_v2.py` 补充 string 类型/名称/顺序损坏拒绝、UTF-8 和空 optional
dtype 组合拒绝测试。已有 ABI 冻结常量、v1 断言及正尺寸原生校验均未修改。

## 环境和已执行的命令

约定解释器无法执行，返回 `Permission denied`、退出码 126。实际使用下列环境：

```bash
source /home/dy/tools/miniconda3/Ascend/ascend-toolkit/set_env.sh
P=/tmp/fi-author-test-env/bin/python
$P -B -c 'import torch,torch_npu;print(torch.npu.is_available(),torch.npu.device_count());print([torch.npu.get_device_name(i) for i in range(torch.npu.device_count())])'
# True 8
# 八张卡均为 Ascend910B3；验证使用 device 4
export FLASHINFER_TEST_NPU_DEVICE=4
export FLASHINFER_CATLASS_ROOT=/tmp/flashinfer-catlass-ref
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11:/usr/include/c++/11/backward:/home/dy/tools/conda-envs/megamoe-a2/include
export LIBRARY_PATH=/home/dy/tools/conda-envs/megamoe-a2/lib
$P -B -m flashinfer.aot --freeze \
  --kernel-id ascendc/attention/sparse_flash_attention/native/Ascend910B \
  --output-dir /tmp/fi-sfa22-release --cache-dir /tmp/fi-sfa22-cache
# built=1, failed=0, requested=1, skipped=0
$P -B tools/run_regression.py --profile quick --log-dir /tmp/fi-sfa22-unit-09 \
  tests/test_sparse_flash_attention.py tests/test_ascendc_contract_v2.py tests/test_ascendc_op_package.py
# 153 passed / 0 skipped
FLASHINFER_AOT_PATHS=/tmp/fi-sfa22-release \
FLASHINFER_ACLNN_ADAPTER_PATH=/tmp/fi-author-adapter FLASHINFER_NPU_JIT_DISABLE=1 \
  $P -B -m flashinfer.operators.testing sparse_flash_attention matrix
# PASS；14 组样例，compiler/toolchain/jit 调用均为 0
```

提交前再次执行相同设备自检，仍为 `True 8`，八张卡均为 Ascend910B3。

最终 spec 为 `e70a71a3c2a42de0f32e0374e6b2bc5e1f7796d6a861367e1fcab630f784330b`，
recipe 为 `7cc9e6caa83765329305643ccb1a2fe0862850ad0e982da6671bd7424ada1000`。
补齐能力绑定后执行 `python -B -m flashinfer.aot --freeze --all`，随后用
`--kernel-id ascendc/attention/sparse_flash_attention/native/Ascend910B`
和 `--kernel-id ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B`
选择重建两份产物，输出到 `/tmp/fi-sfa22-binding-release`，结果为
`built=2, failed=0, requested=2, skipped=0`；其余三份生产 spec 保持原值。
AOT 包为 2,415,185 字节，包含 16 个 Vector 和 16 个 Cube 编译单元；不是运行期编译。
14 组观测最大绝对误差为 attention `0.001953125`、softmax_max 约 `4.47e-8`、
softmax_sum 约 `6.10e-5`。含 FP16/BF16、BSND/TND、分页 KV、RoPE 有/无、
heads=4/64/128、block=1/2/4/16/128、长序列 1100、尾部 `-1` 和全空行。
此列表是实际样例范围，不能替代声明域中其他组合的验证。

## wheel 安装验收

在隔离源码目录从最终 AOT 包打 wheel，使用新建 venv，安装后才添加纯路径依赖，
没有使用 `--system-site-packages`。包内含 arch22 源码、一个匹配的 AOT 包和原生桥；
没有此算子的 arch35 文件或算子目录下的手写 JSON。

```bash
# 隔离源码目录中执行
FLASHINFER_AOT_PREBUILT=/tmp/fi-sfa22-final-release \
FLASHINFER_ACLNN_ADAPTER_PATH=/tmp/fi-author-adapter \
  /tmp/fi-author-test-env/bin/python -B setup.py bdist_wheel --dist-dir /tmp/fi-sfa22-wheelhouse
/tmp/fi-sfa22-wheel-env/bin/python -B -m pip install --no-deps --force-reinstall \
  /tmp/fi-sfa22-wheelhouse/flashinfer-0.1.0.dev0-py3-none-linux_aarch64.whl
# 源码树外运行；bootstrap 校验安装来源并禁止工具链发现、kernel/bridge 编译
PYTHONPATH=/tmp/fi-sfa22-installed-bootstrap FLASHINFER_TEST_NPU_DEVICE=4 FLASHINFER_NPU_JIT_DISABLE=1 \
  /tmp/fi-sfa22-wheel-env/bin/python -B -m flashinfer.operators.testing sparse_flash_attention installed
# PASS；全部 14 组样例，source=aot，compiler/toolchain/jit=0
PYTHONPATH=/tmp/fi-sfa22-installed-bootstrap FLASHINFER_TEST_NPU_DEVICE=4 FLASHINFER_NPU_JIT_DISABLE=1 \
  /tmp/fi-sfa22-wheel-env/bin/python -B -m flashinfer selftest \
  --kernel-id ascendc/attention/sparse_flash_attention/native/Ascend910B
# Summary: 1 PASS, 0 FAIL, 0 SKIP (total=1)
```

父子进程的 `flashinfer.__file__` 均在该 venv 内，禁止编译入口无调用记录。
另外断言 `data_root()` 和原生桥目录也在该 venv 中，外部
`FLASHINFER_AOT_PATHS`、`FLASHINFER_ACLNN_ADAPTER_PATH` 均为未设置，输出 `BUNDLED_PATHS_PASS`。
源码生成资源检查 `python -B -m flashinfer.operators check` 返回 `result=PASS`。
错误架构预处理检查使用 `c++ -E -D__CCE_AICORE__=310`，明确返回
`SparseFlashAttention requires arch22 (A2)`，没有因 SDK 头缺失而意外失败。

能力绑定补齐后，从 `/tmp/fi-sfa22-binding-release` 再构建一次 wheel 至
`/tmp/fi-sfa22-binding-wheelhouse`，包内只收集到 SFA 的最终 spec `e70a71a3c2a4`。
新建 `/tmp/fi-sfa22-binding-wheel-env` 并强制安装，使用
`/tmp/fi-sfa22-binding-installed-bootstrap` 的同等编译禁止与安装来源检查，
重复执行上述 `installed` 矩阵及定向 selftest：14 组全部 PASS、编译调用均为 0，
selftest 为 `1 PASS / 0 FAIL / 0 SKIP`。这是最终绑定版本的安装结果。

## 完整回归与审核状态

全量回归日志为 `build/sparse-flash-attention-arch22/full/`，实际命令为
`python -B tools/run_regression.py --profile full --log-dir /tmp/fi-sfa22-full-02`。
该次全量在最后的能力绑定补齐及清单授权应用前执行，结果如下：

| 范围 | 实际结果 |
| --- | --- |
| 整仓 full | 2142 passed / 64 failed / 63 errors / 142 skipped，共 2411，退出码 1 |
| 同次 full 的 real_compile | 55 passed / 44 failed / 34 errors / 0 skipped，共 133 |
| 同次 full 的 SFA 单测及真机专项 | 65 passed / 0 failed / 0 errors / 0 skipped |

对照此前保留的完整回归失败节点，44 个失败与 63 个错误均已有记录。
20 个新增失败节点中，19 个来自尚未更新的精确登记预期，另一个是上述 CLI 能力绑定缺口。
142 个 skip 全部因为没有对应 Ascend950 设备：16 个 `requires an Ascend950 NPU`、
118 个 `requires verified Ascend950`、8 个 `requires the verified Ascend950 FIA path`，均不计成功。
CLI 绑定修复后，在禁止工具链、编译和原生加载的条件下成功列出 5 个实现；
同一原测试随后仅因旧的 4 项数量预期失败。

精确生产登记预期补丁保存在 `build/sparse-flash-attention-arch22/inventory-approval.patch`，
涉及 AOT 生产 kernel 4→5、全局 Descriptor 85→86、逻辑 op 47→48。
按 `AGENTS.md` 第 4 节取得本轮明确授权后已应用；旧 85 项 capability 快照保持原值，
测试没有被改成包含关系或数量下界。
授权应用后，对全部新增失败所在的清单/导入/登记/发布检查执行 `--profile quick` 复验，
日志为 `build/sparse-flash-attention-arch22/inventory-followup/`：
`191 passed / 0 failed / 0 errors / 0 skipped / 5 real_compile deselected`，退出码 0。
这不是再次完整回归；此前整仓的 142 个 skip 仍不计通过。

能力绑定补齐、重新冻结构建和授权清单更新之后，执行最终受影响的完整档专项：

```bash
/tmp/fi-author-test-env/bin/python -B tools/run_regression.py --profile full \
  --log-dir /tmp/fi-sfa22-binding-focused-01 \
  tests/test_sparse_flash_attention.py tests/test_operator_packages_real.py \
  tests/test_attention_capability.py tests/test_attention_quant_launch_abi.py \
  tests/test_ascendc_contract_v2.py
# 175 passed / 0 failed / 0 errors / 0 skipped，退出码 0
```

该次专项含 17 个 real_compile 用例：SFA 的队列 0/1/2 数值、out 身份和非默认流、
缺包/损坏拒绝、5 组索引/空行/分页统计边界，以及 GMM 原有六项验收全部通过。
其中最终 SFA 相关用例为 66 项，旧 AttentionPlan 与量化 POD ABI 护栏也通过复验。
日志位于 `build/sparse-flash-attention-arch22/focused/`。
最后的绑定修复后没有重复整仓 2411 项 full，不能把受影响专项数字改写成整仓通过。

`build/sparse-flash-attention-arch22/review.patch` 是提交前生成的完整改动快照，包含新增文件；
`upstream-adaptation.diff` 单列相对上游的原生适配。

`NativeBinding` 中的 `Q0/Q1`、`K0/K1` 表示不同布局下的轴位置：BSND 时为
batch/sequence，TND 时首轴为总 token，分页 K/V 时为物理块数/块容量。
`H` 是 query heads，`SP` 是每行索引槽数，`B/NB` 是长度/页表中的 batch/最大块数。
512/64 是此 MLA 版本的固定 D/Dr；其余维度上下限是本实现的接纳边界，
不表示每个边界组合均已实测。128 MiB 是动态 workspace 的拒绝上限，单位为字节。

A3/A5 真机运行、V 聚合分支、capture/replay 和性能均未验证。图能力保持
`NOT_CAPTURABLE`，也不声称本次未运行的其他 dtype/shape/架构组合已验收。
本次未为该 op 新增 CANN 或 Triton 实现。
