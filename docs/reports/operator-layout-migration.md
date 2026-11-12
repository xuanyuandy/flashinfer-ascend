# 多后端算子目录迁移与验收记录

日期：2026-11-12。基线 commit：`822af8ce54a9072a52c54991b8aa9f3781a30831`。
本次任务起点包含此前尚未提交的 GMM 接入及框架整改，比较依据是任务起始文件快照，不能把全部工作区 diff 归为此次新增。

目录重构已落地，受影响的接口、登记及生成检查通过，GMM 的设备验收与安装验收通过。**full 未通过，不能作为全量验收完成的结论。** 当前机器提供八张 Ascend910B3，历史 Ascend950 路径及部分编译依赖仍阻塞验收。

## 架构与源码职责

[组织规范](../operator-layout.md) 是当前目录约定，[接入指南](../adding-a-custom-operator.md#063-已实现的目录接入入口) 说明新增流程。

| 层次 | 核心位置及作用 |
| --- | --- |
| 公共接口 | `flashinfer/gemm.py:399` 等领域门面及 `flashinfer/__init__.py:146` 自动收集新接口；既有函数、类和 plan/run 协议继续保留 |
| 公共定义 | `flashinfer/operators/<domain>/<op>/operator.py` 唯一定义逻辑身份；`definition.py:60` 区分公共 schema/infer 与既有 `public_entry` |
| 实现声明 | `flashinfer/<backend>/<domain>/<op>/implementation.py` 声明 kernel_id、Capability、依赖和 runner 工厂；`operators/implementation.py:31` 检查绑定身份 |
| 原生构建声明 | `flashinfer/ascendc/definition.py:35` 的 `NativeDefinition` 关联 ABI、源码目录和可选 CMake；`ascendc/authoring.py:12` 派生摘要、recipe、Descriptor |
| 实现本体 | Ascend C 在 `csrc/kernels/ascendc/<domain>/<source-op>/`；Triton 的 `kernels.py` 与 runner 同目录；CANN/reference 的执行代码在各算子 runner 或同域共有模块中 |
| 框架收集 | `operators/authoring.py:75` 发现并生成交付索引；`operators/implementation.py:60` 校验随包源码及实现声明 |
| 统一登记 | `runtime/registry.py:284` 汇总非 Ascend C 实现，Ascend C 通过 `ascendc/__init__.py:13` 绑定冻结声明；均加入同一个 Registry |
| 选型和执行 | `operators/runtime.py:70` 完成公共检查、选型、输出与执行；`runtime/dispatch.py:214`、`:284` 在强制选型与自动选型中分别检查候选 admission；`runtime/runner.py` 交接到选中 runner |
| AOT/JIT | `ascendc/_runtime.py:77` 复用已有验证加载、JIT 策略和 Torch/ACLNN 桥；没有另建算子缓存或加载器 |

共有原语可放在后端域内 `_family.py`、`_shared.py`；旧私有模块只保留兼容导出。既有富接口通过 `public_entry` 引用已有前端，其签名、公共校验及资源协议继续由该前端维护。新增无状态接口可以直接使用声明式 schema/infer 和通用运行器。

| 文件类型 | 作者是否维护 | 是否由当前生成器生成 |
| --- | --- | --- |
| 逻辑 `operator.py`、语义输出推导和值检查 | 是 | 否 |
| 后端 `implementation.py`、Capability、必要 runner | 是 | 否 |
| Ascend C host/device/tiling、Triton kernel、必要 CMake | 是 | 否 |
| 独立 golden、样例和容差 `cases.py` | 是 | 否 |
| `data/operators.json`、`data/implementations.json` | 只审阅 | 是，`operators generate` |
| 原生 `csrc/recipes/*.json`、登记 manifest 中的原生记录 | 只审阅 | 是，由原生实现声明派生 |
| `data/kernels/ascendc_specs.json` | 只审阅 | 是，freeze |
| 编译包、产物 manifest、wheel | 运行交付工具 | 是，既有构建/导出流程 |
| Registry、dispatch、Torch 接线、打包与通用验收工具 | 框架维护者维护 | 是共有框架源码，不是每算子的生成文件 |
| 独立生产预期与能力快照测试 | 框架维护者审核 | 生成器不修改 |

此次批量迁移得到的 Python 定义和实现文件是后续人工维护的权威源码；不能因为由编码工具写出，就声称框架会重新生成它们。

## GMM 调用链与作者文件

作者维护的核心文件为：

1. `flashinfer/operators/gemm/grouped_matmul_swiglu_quant/operator.py:16`：公共量化语义、输出推导、前缀值检查及 `OPERATOR`；公共参数中没有原生三项属性 ABI。
2. `flashinfer/operators/gemm/grouped_matmul_swiglu_quant/cases.py`：七组独立数值样例及容差。
3. `flashinfer/ascendc/gemm/grouped_matmul_swiglu_quant/implementation.py:15`：Descriptor/runner 绑定、Capability、原生 ABI、源码和构建配置。
4. 同目录 `runner.py:8`、`:37`：该候选的原生 admission、参数映射与执行适配。
5. `csrc/kernels/ascendc/gemm/grouped_matmul_swiglu_quant/`：host、device、头文件、CMake 及来源许可。

框架从 `gemm.py` 暴露 `grouped_matmul_swiglu_quant`，从逻辑定义收集 Torch functional/out/FakeTensor 接线，先构造逻辑签名，再选择实现。Ascend C runner 在被选中后把逻辑参数映射到原生 ABI，复用验证过的 AOT 加载器和 ACLNN 桥。

为同一算子追加 Triton/CANN 实现时，新增对应后端目录和实现声明；不向公共 `operator.py` 填入另一份后端 Descriptor。后端适配不会让公共签名提前依赖 Ascend C recipe。

## 覆盖与兼容性证据

| 范围 | 数量 |
| --- | ---: |
| 公共逻辑定义 | 47 |
| 全局 Descriptor | 85 |
| Ascend C 生产实现 | 4 |
| CANN 全局实现 | 41 |
| reference 全局实现 | 38 |
| Triton 全局实现 | 2 |
| 显式启用的 Triton attention 实验实现 | 1，未加入全局登记 |

完整逐算子文件映射保存在 [目录清单](../../build/architecture-refactor/layout-inventory.json)，包含每个逻辑定义、kernel_id、scope、implementation、runner 和原生源码位置。

- 85 个 kernel_id、全部 Capability 及除 `source` 外的静态 Descriptor 字段与任务起点完全一致。文件路径变化导致相应源码身份和冻结 spec 变化。
- `tests/production_inventory.py` 没有因本次目录迁移改变；精确集合及数量断言保留。新增 `tests/fixtures/operator_layout/capabilities.json` 固定迁移前的完整能力内容，生成工具不写该预期。
- 五份原生 recipe 的版本、Tensor/属性 ABI、dtype 联立约束、shape 约束、入口、目标和 tiling schema 与任务起点相同。原生及辅助文件共 34 份逐字节相同。
- `tests/fixtures/legacy_ascendc/` 的冻结文件逐字节相同。未放宽 SoC、ABI、图能力或产物校验以适应当前设备。
- `framework_design.md` 代码围栏数 62→62，接入指南 26→26；其他受影响文档也没有减少代码块。已有 CRLF 文档保留原换行，用 `git -c core.whitespace=cr-at-eol diff --check` 检查。

## 实测环境和命令

约定解释器 `/root/miniforge3/envs/flashinfer-a5/bin/python` 在当前会话不可执行，实际返回 permission denied、退出码 126。没有沿用约定中的 Ascend950 环境假设。

实际使用 `/home/dy/tools/conda-envs/megamoe-a2/bin/python`，Torch `2.9.0+cpu`、TorchNPU `2.9.0.post9`；CANN 为 `cann-9.2.0`。设备自检输出：

```text
True 8
['Ascend910B3', 'Ascend910B3', 'Ascend910B3', 'Ascend910B3',
 'Ascend910B3', 'Ascend910B3', 'Ascend910B3', 'Ascend910B3']
```

编译依赖使用官方 CATLASS checkout，commit `41bf90da655bba3c66d0acd7e00abe33960ecfd6`。回归通过设备入口指定 NPU 4。实际执行的环境与 full 命令：

```bash
P=/home/dy/tools/conda-envs/megamoe-a2/bin/python
source /home/dy/tools/miniconda3/Ascend/ascend-toolkit/set_env.sh
export FLASHINFER_CATLASS_ROOT=/tmp/flashinfer-catlass-ref
export CPLUS_INCLUDE_PATH=/home/dy/tools/conda-envs/megamoe-a2/include
export LIBRARY_PATH=/home/dy/tools/conda-envs/megamoe-a2/lib
export FLASHINFER_TEST_NPU_DEVICE=4
export FLASHINFER_NPU_CACHE_DIR=/tmp/fi-architecture-runtime-cache
$P -B tools/run_regression.py --profile full --log-dir /tmp/fi-architecture-tests-full
```

| 执行范围 | 实际 pytest 结果 | 结论 |
| --- | --- | --- |
| full，包含 real_compile | 2041 passed、46 failed、63 errors、148 skipped，另有 404 个子测试通过 | 退出码 1，未通过 |
| 同一次 full 内的 113 个 real_compile 用例 | 27 passed、46 failed、34 errors、6 skipped | 从收集 nodeid 与 JUnit 对应统计，没有重复运行同一编译矩阵 |
| full 中的 GMM 通用真机验收 | 6 passed、0 skipped | 队列 0/1/2 的数值/out/非默认流，缺包两种 JIT 策略及损坏拒绝通过 |
| 最终接口补测 | 104 passed、1 deselected、0 skipped | 通过；不称为 full |
| 最终生成/登记/构建声明补测 | 73 passed、2 个子测试通过、3 deselected、0 skipped | 通过；不称为 real_compile |
| 输出别名及定义边界专项 | 38 passed、0 skipped | 通过 |
| 最终 Triton 解耦与分派补测 | 100 passed、0 skipped | 通过；包括不读取 Ascend C recipe 的 runner 参数校验 |

回归工具的 JUnit 汇总将 404 个通过子测试加进 passed，故 full 的 `summary.json` 显示 2445 passed；上表使用 pytest 用例数 2041，并单列子测试，避免混淆。

最终接口补测实际命令：

```bash
$P -B tools/run_regression.py --profile quick --log-dir /tmp/fi-architecture-tests-final-metadata \
  tests/test_operator_layout.py tests/test_api_smoke.py tests/test_matmul.py \
  tests/test_torch_library.py tests/test_operator_authoring.py tests/test_release_checks.py
$P -B tools/run_regression.py --profile quick --log-dir /tmp/fi-architecture-tests-final-static \
  tests/test_operator_layout.py tests/test_operator_authoring.py tests/test_ascendc_framework.py \
  tests/test_ascendc_v2_public.py tests/test_manifests.py tests/test_aot_multi_operator.py \
  tests/test_aot_inventory_diagnostics.py
/usr/bin/python3 -B -m flashinfer.operators check
```

最终检查还移除了实验 Triton attention runner 对 Ascend C recipe 的旧依赖，其 dtype、shape、连续性、布局及别名护栏在自身实现中保留。实际补测命令为 `$P -B tools/run_regression.py --profile quick --log-dir /tmp/fi-architecture-tests-final-triton tests/test_triton_attention.py tests/test_matmul.py tests/test_operator_layout.py tests/test_two_level_dispatch.py`。该补测不构成真实 Triton 设备执行证据。

生成器检查实际输出 `result: PASS`。原生生产四项在 full 的新缓存中全部冷构建成功，日志均为 `disk_cache_hit: false`、`built: 4`、`failed: 0`、`skipped: 0`。末次清理 FlashAttention 声明中不再使用的重复 source 字段后，重新生成、冻结并专项冷构建该项，实际 `built: 1`、`failed: 0`、`skipped: 0`；其余三项身份不变。最终四项都通过 `verify_build_directory` 和 `verify_op_package`，详见 [产物校验](../../build/architecture-refactor/verified-final-builds.json)。编译成功不代表当前 910B 设备验证了三个 Ascend950 kernel 的数值。

最终 wheel 使用四个验证产物和原生桥构建，再在不启用 system-site-packages 的干净 venv 中强制安装；安装后才写纯路径依赖 `.pth`。实际命令：

```bash
FLASHINFER_AOT_PREBUILT=/tmp/fi-architecture-prebuilt \
FLASHINFER_ACLNN_ADAPTER_PATH=/tmp/fi-architecture-adapter \
$P -B setup.py build --build-base /tmp/fi-architecture-wheel-build bdist_wheel \
  --dist-dir /tmp/fi-architecture-dist --bdist-dir /tmp/fi-architecture-bdist
/tmp/fi-architecture-venv/bin/python -B -m pip install --force-reinstall --no-deps \
  /tmp/fi-architecture-dist/flashinfer-0.1.0.dev0-py3-none-linux_aarch64.whl
cd /tmp
unset PYTHONPATH FLASHINFER_AOT_PATHS FLASHINFER_ACLNN_ADAPTER_PATH
export FLASHINFER_NPU_JIT_DISABLE=1
export FLASHINFER_NPU_CACHE_DIR=/tmp/fi-architecture-installed-cache
/tmp/fi-architecture-venv/bin/python -B -m flashinfer.operators.testing grouped_matmul_swiglu_quant installed
/tmp/fi-architecture-venv/bin/python -B -m flashinfer selftest \
  --kernel-id ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B
```

实际加载位置为 `/tmp/fi-architecture-venv/lib/python3.11/site-packages/flashinfer/__init__.py`。七组安装样例全部通过，最大 INT8 绝对差 1、scale 最大绝对差约 `4.0291e-7`，均满足作者声明的容差；compiler/JIT/toolchain 调用均为 0，provenance 为 `aot`。装后 selftest 输出 `1 PASS, 0 FAIL, 0 SKIP`，验证范围只包含指定 GMM kernel。

wheel 包含 47 份逻辑定义、86 个实现声明记录、四个 AOT 包及 ACLNN 桥，没有遗留 `_generated` 执行副本。SHA256 为 `297af4c70d90c60c16b0afe9eee4fbd079d9c559b06cac2673cce9776b0d888d`。

## 未通过与未验证

| 已观察原因 | full 数量 | 当前边界 |
| --- | ---: | --- |
| kernel/recipe 目标 Ascend950 与设备 Ascend910B 不匹配 | 35 failed、4 errors | 保留目标护栏，不扩写未经验证的支持集 |
| fixture 明确要求已验证的 Ascend950 | 59 errors | attention 29、decode graph 30 |
| 缺少 `ml_dtypes` | 6 failed | CATLASS golden 生成依赖未满足 |
| 当前 Torch/扩展没有注册 NPU Inductor | 4 failed | 原错误为 `Device npu not supported` |
| host_shared 编译找不到 C++ 标准头 | 1 failed | 原错误为 `'cstdint' file not found` |
| Ascend950 硬件限定的 skip | 142 skipped | 原因包括 requires verified Ascend950、requires an Ascend950 NPU、verified Ascend950 FIA path |
| 缺少 Triton | 6 skipped | 原因 `No module named 'triton'`；不计通过 |

仍未完成：在匹配 Ascend950、Triton、golden 及 NPU Inductor 编译环境中完成全量零 skip 验收；三个 Ascend950 AOT kernel 的装后数值与完整 selftest；此次目录迁移后的相关图 capture/replay 实测。GMM 保持 `NOT_CAPTURABLE`。没有性能测量，也不作性能结论。

曾将初期 FP8 CPU mock 测试失败归因于环境；后续定位为迁移后私有 patch 目标不再覆盖函数实际 globals，已修正测试 patch，数值实现未改。该初期归因撤回，不能把它列入最终环境阻塞。

原始 full 日志、JUnit、真实退出码、最终补测日志、失败逐项归因、目录映射及安装验证输出位于 [验收证据目录](../../build/architecture-refactor/)。本报告不把历史时点报告中的“已验证”状态自动迁移到当前设备。

## 逐算子源码索引

每个后端链接指向其 `implementation.py`；`runner.py` 在同目录，Triton 的设备实现为同目录 `kernels.py`。Ascend C 原生源码由该声明的 `NATIVE.source_dir` 指定。

| 逻辑 op 与公共定义 | 后端实现声明 |
| --- | --- |
| [`activation.gelu_and_mul`](../../flashinfer/operators/activation/gelu_and_mul/operator.py) | [cann](../../flashinfer/cann/activation/gelu_and_mul/implementation.py), [reference](../../flashinfer/reference/activation/gelu_and_mul/implementation.py) |
| [`activation.gelu_tanh_and_mul`](../../flashinfer/operators/activation/gelu_tanh_and_mul/operator.py) | [cann](../../flashinfer/cann/activation/gelu_tanh_and_mul/implementation.py), [reference](../../flashinfer/reference/activation/gelu_tanh_and_mul/implementation.py) |
| [`activation.silu_and_mul`](../../flashinfer/operators/activation/silu_and_mul/operator.py) | [cann](../../flashinfer/cann/activation/silu_and_mul/implementation.py), [reference](../../flashinfer/reference/activation/silu_and_mul/implementation.py) |
| [`attention.batch_decode_paged`](../../flashinfer/operators/attention/batch_decode_paged/operator.py) | [cann](../../flashinfer/cann/attention/batch_decode_paged/implementation.py) |
| [`attention.batch_prefill_paged`](../../flashinfer/operators/attention/batch_prefill_paged/operator.py) | [cann](../../flashinfer/cann/attention/batch_prefill_paged/implementation.py) |
| [`attention.batch_prefill_ragged`](../../flashinfer/operators/attention/batch_prefill_ragged/operator.py) | [cann](../../flashinfer/cann/attention/batch_prefill_ragged/implementation.py) |
| [`attention.decode`](../../flashinfer/operators/attention/decode/operator.py) | [reference](../../flashinfer/reference/attention/decode/implementation.py) |
| [`attention.prefill`](../../flashinfer/operators/attention/prefill/operator.py) | [reference](../../flashinfer/reference/attention/prefill/implementation.py) |
| [`attention.single_decode`](../../flashinfer/operators/attention/single_decode/operator.py) | [cann](../../flashinfer/cann/attention/single_decode/implementation.py) |
| [`attention.single_prefill`](../../flashinfer/operators/attention/single_prefill/operator.py) | [cann](../../flashinfer/cann/attention/single_prefill/implementation.py) |
| [`ascendc.single_prefill_catlass`](../../flashinfer/operators/attention/single_prefill_catlass/operator.py) | [ascendc](../../flashinfer/ascendc/attention/single_prefill_catlass/implementation.py), [triton（实验）](../../flashinfer/triton/attention/single_prefill_catlass/implementation.py) |
| [`cascade.merge_state`](../../flashinfer/operators/cascade/merge_state/operator.py) | [cann](../../flashinfer/cann/cascade/merge_state/implementation.py), [reference](../../flashinfer/reference/cascade/merge_state/implementation.py) |
| [`cascade.merge_states`](../../flashinfer/operators/cascade/merge_states/operator.py) | [cann](../../flashinfer/cann/cascade/merge_states/implementation.py), [reference](../../flashinfer/reference/cascade/merge_states/implementation.py) |
| [`decode.cudnn_batch_decode_with_kv_cache`](../../flashinfer/operators/decode/cudnn_batch_decode_with_kv_cache/operator.py) | [cann](../../flashinfer/cann/decode/cudnn_batch_decode_with_kv_cache/implementation.py), [reference](../../flashinfer/reference/decode/cudnn_batch_decode_with_kv_cache/implementation.py) |
| [`fused_moe.cutlass_fused_moe`](../../flashinfer/operators/fused_moe/cutlass_fused_moe/operator.py) | [cann](../../flashinfer/cann/fused_moe/cutlass_fused_moe/implementation.py), [reference](../../flashinfer/reference/fused_moe/cutlass_fused_moe/implementation.py) |
| [`fused_moe.trtllm_fp8_block_scale_moe`](../../flashinfer/operators/fused_moe/trtllm_fp8_block_scale_moe/operator.py) | [cann](../../flashinfer/cann/fused_moe/trtllm_fp8_block_scale_moe/implementation.py), [reference](../../flashinfer/reference/fused_moe/trtllm_fp8_block_scale_moe/implementation.py) |
| [`fused_moe.trtllm_fp8_per_tensor_scale_moe`](../../flashinfer/operators/fused_moe/trtllm_fp8_per_tensor_scale_moe/operator.py) | [cann](../../flashinfer/cann/fused_moe/trtllm_fp8_per_tensor_scale_moe/implementation.py), [reference](../../flashinfer/reference/fused_moe/trtllm_fp8_per_tensor_scale_moe/implementation.py) |
| [`gemm.bmm_bf16`](../../flashinfer/operators/gemm/bmm_bf16/operator.py) | [cann](../../flashinfer/cann/gemm/bmm_bf16/implementation.py), [reference](../../flashinfer/reference/gemm/bmm_bf16/implementation.py) |
| [`gemm.bmm_fp8`](../../flashinfer/operators/gemm/bmm_fp8/operator.py) | [cann](../../flashinfer/cann/gemm/bmm_fp8/implementation.py), [reference](../../flashinfer/reference/gemm/bmm_fp8/implementation.py) |
| [`gemm.gemm_fp8_nt_groupwise`](../../flashinfer/operators/gemm/gemm_fp8_nt_groupwise/operator.py) | [cann](../../flashinfer/cann/gemm/gemm_fp8_nt_groupwise/implementation.py), [reference](../../flashinfer/reference/gemm/gemm_fp8_nt_groupwise/implementation.py) |
| [`gemm.grouped_matmul_swiglu_quant`](../../flashinfer/operators/gemm/grouped_matmul_swiglu_quant/operator.py) | [ascendc](../../flashinfer/ascendc/gemm/grouped_matmul_swiglu_quant/implementation.py) |
| [`gemm.matmul`](../../flashinfer/operators/gemm/matmul/operator.py) | [ascendc](../../flashinfer/ascendc/gemm/matmul/implementation.py), [cann](../../flashinfer/cann/gemm/matmul/implementation.py), [triton](../../flashinfer/triton/gemm/matmul/implementation.py) |
| [`gemm.matmul_bias_silu`](../../flashinfer/operators/gemm/matmul_bias_silu/operator.py) | [ascendc](../../flashinfer/ascendc/gemm/matmul_bias_silu/implementation.py) |
| [`gemm.mm_bf16`](../../flashinfer/operators/gemm/mm_bf16/operator.py) | [cann](../../flashinfer/cann/gemm/mm_bf16/implementation.py), [reference](../../flashinfer/reference/gemm/mm_bf16/implementation.py) |
| [`gemm.segment_gemm`](../../flashinfer/operators/gemm/segment_gemm/operator.py) | [cann](../../flashinfer/cann/gemm/segment_gemm/implementation.py), [reference](../../flashinfer/reference/gemm/segment_gemm/implementation.py) |
| [`norm.fused_add_rmsnorm`](../../flashinfer/operators/norm/fused_add_rmsnorm/operator.py) | [cann](../../flashinfer/cann/norm/fused_add_rmsnorm/implementation.py), [reference](../../flashinfer/reference/norm/fused_add_rmsnorm/implementation.py) |
| [`norm.gemma_fused_add_rmsnorm`](../../flashinfer/operators/norm/gemma_fused_add_rmsnorm/operator.py) | [reference](../../flashinfer/reference/norm/gemma_fused_add_rmsnorm/implementation.py), [triton](../../flashinfer/triton/norm/gemma_fused_add_rmsnorm/implementation.py) |
| [`norm.gemma_rmsnorm`](../../flashinfer/operators/norm/gemma_rmsnorm/operator.py) | [cann](../../flashinfer/cann/norm/gemma_rmsnorm/implementation.py), [reference](../../flashinfer/reference/norm/gemma_rmsnorm/implementation.py) |
| [`norm.layernorm`](../../flashinfer/operators/norm/layernorm/operator.py) | [cann](../../flashinfer/cann/norm/layernorm/implementation.py), [reference](../../flashinfer/reference/norm/layernorm/implementation.py) |
| [`norm.rmsnorm`](../../flashinfer/operators/norm/rmsnorm/operator.py) | [cann](../../flashinfer/cann/norm/rmsnorm/implementation.py), [reference](../../flashinfer/reference/norm/rmsnorm/implementation.py) |
| [`page.append_paged_kv_cache`](../../flashinfer/operators/page/append_paged_kv_cache/operator.py) | [cann](../../flashinfer/cann/page/append_paged_kv_cache/implementation.py), [reference](../../flashinfer/reference/page/append_paged_kv_cache/implementation.py) |
| [`rope.apply_rope`](../../flashinfer/operators/rope/apply_rope/operator.py) | [cann](../../flashinfer/cann/rope/apply_rope/implementation.py), [reference](../../flashinfer/reference/rope/apply_rope/implementation.py) |
| [`rope.apply_rope_inplace`](../../flashinfer/operators/rope/apply_rope_inplace/operator.py) | [cann](../../flashinfer/cann/rope/apply_rope_inplace/implementation.py), [reference](../../flashinfer/reference/rope/apply_rope_inplace/implementation.py) |
| [`rope.apply_rope_pos_ids`](../../flashinfer/operators/rope/apply_rope_pos_ids/operator.py) | [cann](../../flashinfer/cann/rope/apply_rope_pos_ids/implementation.py), [reference](../../flashinfer/reference/rope/apply_rope_pos_ids/implementation.py) |
| [`rope.apply_rope_pos_ids_inplace`](../../flashinfer/operators/rope/apply_rope_pos_ids_inplace/operator.py) | [cann](../../flashinfer/cann/rope/apply_rope_pos_ids_inplace/implementation.py), [reference](../../flashinfer/reference/rope/apply_rope_pos_ids_inplace/implementation.py) |
| [`rope.apply_rope_with_cos_sin_cache`](../../flashinfer/operators/rope/apply_rope_with_cos_sin_cache/operator.py) | [cann](../../flashinfer/cann/rope/apply_rope_with_cos_sin_cache/implementation.py), [reference](../../flashinfer/reference/rope/apply_rope_with_cos_sin_cache/implementation.py) |
| [`rope.apply_rope_with_cos_sin_cache_inplace`](../../flashinfer/operators/rope/apply_rope_with_cos_sin_cache_inplace/operator.py) | [cann](../../flashinfer/cann/rope/apply_rope_with_cos_sin_cache_inplace/implementation.py), [reference](../../flashinfer/reference/rope/apply_rope_with_cos_sin_cache_inplace/implementation.py) |
| [`sampling.chain_speculative_sampling`](../../flashinfer/operators/sampling/chain_speculative_sampling/operator.py) | [cann](../../flashinfer/cann/sampling/chain_speculative_sampling/implementation.py), [reference](../../flashinfer/reference/sampling/chain_speculative_sampling/implementation.py) |
| [`sampling.min_p_sampling_from_probs`](../../flashinfer/operators/sampling/min_p_sampling_from_probs/operator.py) | [cann](../../flashinfer/cann/sampling/min_p_sampling_from_probs/implementation.py), [reference](../../flashinfer/reference/sampling/min_p_sampling_from_probs/implementation.py) |
| [`sampling.sampling_from_logits`](../../flashinfer/operators/sampling/sampling_from_logits/operator.py) | [cann](../../flashinfer/cann/sampling/sampling_from_logits/implementation.py), [reference](../../flashinfer/reference/sampling/sampling_from_logits/implementation.py) |
| [`sampling.sampling_from_probs`](../../flashinfer/operators/sampling/sampling_from_probs/operator.py) | [cann](../../flashinfer/cann/sampling/sampling_from_probs/implementation.py), [reference](../../flashinfer/reference/sampling/sampling_from_probs/implementation.py) |
| [`sampling.top_k_mask_logits`](../../flashinfer/operators/sampling/top_k_mask_logits/operator.py) | [cann](../../flashinfer/cann/sampling/top_k_mask_logits/implementation.py), [reference](../../flashinfer/reference/sampling/top_k_mask_logits/implementation.py) |
| [`sampling.top_k_renorm_probs`](../../flashinfer/operators/sampling/top_k_renorm_probs/operator.py) | [cann](../../flashinfer/cann/sampling/top_k_renorm_probs/implementation.py), [reference](../../flashinfer/reference/sampling/top_k_renorm_probs/implementation.py) |
| [`sampling.top_k_sampling_from_probs`](../../flashinfer/operators/sampling/top_k_sampling_from_probs/operator.py) | [cann](../../flashinfer/cann/sampling/top_k_sampling_from_probs/implementation.py), [reference](../../flashinfer/reference/sampling/top_k_sampling_from_probs/implementation.py) |
| [`sampling.top_k_top_p_sampling_from_probs`](../../flashinfer/operators/sampling/top_k_top_p_sampling_from_probs/operator.py) | [cann](../../flashinfer/cann/sampling/top_k_top_p_sampling_from_probs/implementation.py), [reference](../../flashinfer/reference/sampling/top_k_top_p_sampling_from_probs/implementation.py) |
| [`sampling.top_p_renorm_probs`](../../flashinfer/operators/sampling/top_p_renorm_probs/operator.py) | [cann](../../flashinfer/cann/sampling/top_p_renorm_probs/implementation.py), [reference](../../flashinfer/reference/sampling/top_p_renorm_probs/implementation.py) |
| [`sampling.top_p_sampling_from_probs`](../../flashinfer/operators/sampling/top_p_sampling_from_probs/operator.py) | [cann](../../flashinfer/cann/sampling/top_p_sampling_from_probs/implementation.py), [reference](../../flashinfer/reference/sampling/top_p_sampling_from_probs/implementation.py) |
