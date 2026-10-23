T8 交付报告 · 2026-10-23

本机实测 `torch.compile + torch_npu` 可运行。环境为 `/root/miniforge3/envs/flashinfer-a5/bin/python`，torch `2.10.0+cpu`、torch_npu `2.10.0.post7`，8 张可用 NPU。NPU 和 CPU 的普通加法均由默认 Inductor、`fullgraph=True` 编译执行成功。故意插入 graph break 的正对照能收到 `TORCH_LOGS=graph_breaks` 输出；首批真实算子的编译运行无 graph break。

原始环境与日志探测命令（此命令故意产生一个 graph break，仅用于证明日志有效）：

```bash
TORCH_LOGS=graph_breaks /root/miniforge3/envs/flashinfer-a5/bin/python -u tools/t08_torch_compile_smoke.py --probe
```

```text
torch: 2.10.0+cpu
torch_npu: 2.10.0.post7
npu available: True
npu count: 8
/root/miniforge3/envs/flashinfer-a5/lib/python3.11/site-packages/triton/language/extra/cann/extension/math_ops.py:13: FutureWarning: triton.language.extra.cann.extension.atan2 is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.atan2 instead.
  @_deprecated(replacement="triton.language.extra.cann.libdevice.atan2")
/root/miniforge3/envs/flashinfer-a5/lib/python3.11/site-packages/triton/language/extra/cann/extension/math_ops.py:35: FutureWarning: triton.language.extra.cann.extension.isfinited is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.isfinited instead.
  @_deprecated(replacement="triton.language.extra.cann.libdevice.isfinited")
/root/miniforge3/envs/flashinfer-a5/lib/python3.11/site-packages/triton/language/extra/cann/extension/math_ops.py:49: FutureWarning: triton.language.extra.cann.extension.finitef is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.finitef instead.
  @_deprecated(replacement="triton.language.extra.cann.libdevice.finitef")
npu compile result: tensor([2., 2., 2., 2.])
cpu compile result: tensor([2., 2., 2., 2.])
GRAPH_BREAK_LOG_POSITIVE_CONTROL_BEGIN
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks] Graph break in user code at /home/dy/projects/flashinfer/tools/t08_torch_compile_smoke.py:28
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks] Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks] 
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks] Call to `torch._dynamo.graph_break()`
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks]   Explanation: User-inserted graph break. Message: None
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks]   Hint: Remove the `torch._dynamo.graph_break()` call.
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks] 
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks]   Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks] 
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks]  For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks] User code traceback:
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks]   File "/home/dy/projects/flashinfer/tools/t08_torch_compile_smoke.py", line 103, in <module>
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks]     main()
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks]   File "/home/dy/projects/flashinfer/tools/t08_torch_compile_smoke.py", line 97, in main
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks]     probe(torch)
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks]   File "/home/dy/projects/flashinfer/tools/t08_torch_compile_smoke.py", line 32, in probe
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks]     torch.compile(deliberate_break, backend="eager")(torch.ones(4))
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks]   File "/home/dy/projects/flashinfer/tools/t08_torch_compile_smoke.py", line 28, in deliberate_break
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks]     torch._dynamo.graph_break()
V1023 04:38:37.835000 193530 site-packages/torch/_dynamo/symbolic_convert.py:4275] [1/0] [__graph_breaks] 
GRAPH_BREAK_LOG_POSITIVE_CONTROL_END
```

指定的 `/root/flashinfer-ori` 不存在；实际独立核查了 `/home/dy/projects/flashinfer-ori`。当前检出为 `da992ed9919604c5eb6dc255c63f5e89a466452c`；同时用 git show 检查了账本固定的 `2895577720741ddfedff118202c89584996b8875`，没有导入 CUDA 上游包。

```bash
sed -n '395,450p' /home/dy/projects/flashinfer-ori/flashinfer/utils.py
git -C /home/dy/projects/flashinfer-ori show 2895577720741ddfedff118202c89584996b8875:flashinfer/utils.py
```

两个版本的 torch < 2.4 和 >= 2.4 分支里，register_custom_op / register_fake_op 均返回 `lambda x: x`；>= 2.4 分支中的真实 torch.library 调用被注释，自定义算子处明确注明 overhead 原因。AST 逐文件统计为：

```text
HEAD {'register_custom_op': 184, 'register_fake_op': 132}
2895577720741ddfedff118202c89584996b8875 {'register_custom_op': 194, 'register_fake_op': 128}
```

因此任务书的 194/128 对应固定提交。恒等装饰器自身不建立 Dynamo 的算子边界，不能据此保证无 graph break；其中一些原本可追踪的纯 torch 代码是否能编译，是另一个问题。

方案为接口对齐与显式启用：两个 helper 在调用时真实委托 torch.library，支持装饰器和直接传 fn 两种形式；默认导入与 builtin eager 调用不发生注册。应用初始化时先执行以下调用，再开始 torch.compile。入口幂等、进程级生效；不在 tracing/capture 内调用，不提供运行中的开关。

```python
import torch
import torch_npu
import flashinfer
from flashinfer.utils import enable_torch_library

enable_torch_library()
compiled = torch.compile(flashinfer.rmsnorm, fullgraph=True)
```

这样默认保留上游选择避免注册开销的行为，也保留本仓 torch-free 导入；需要编译时得到真正的算子边界，不用依赖一次隐式注册或先调用一次算子。torch < 2.4 在显式注册时明确报错，默认导入仍可用。helper 行为和新增启用入口均在 `flashinfer/data/api_parity.json` 登记为有意差异。独立抽取上游两个版本分支的函数定义后，inspect.signature 与本地比较结果：

```text
PASS identical signature in both upstream version branches: register_custom_op (name: str, fn: Optional[Callable] = None, /, *, mutates_args: Union[str, Iterable[str]], device_types: Union[str, Sequence[str], NoneType] = None, schema: Optional[str] = None) -> Callable
PASS identical signature in both upstream version branches: register_fake_op (name: str, fn: Optional[Callable] = None) -> Callable
```

首批范围只选已接入且有 CANN/reference 数值基线的两个 norm 算子，覆盖输出 buffer 和双输入原地修改，避免同时引入 attention 的 plan/run 或采样的数据依赖输出形状问题。

| 公开入口 | torch.library 名称 | mutation | 返回语义 |
|---|---|---|---|
| flashinfer.rmsnorm | flashinfer::rmsnorm | out | 注册 op 返回 void；公开入口返回原 out，未传 out 时由门面 empty_like(input) |
| flashinfer.fused_add_rmsnorm | flashinfer::fused_add_rmsnorm | input、residual | void / None |

`flashinfer/utils.py` 替代原本只有空 __init__.py 的 utils 包，避免同名包遮蔽模块；导入路径仍为 flashinfer.utils。`flashinfer/_torch_library.py` 在显式启用时才创建带装饰器的算子和 fake 注册，直接 import 这个模块也不加载 torch。`flashinfer/norm.py` 保留原有 build_signature → hardware → dispatch → runner 路径，启用后在 opaque custom op 内调用同一条路径；没有修改 runner、数值表达式或输出的对象身份。

两个注册 op 都返回 void，fake/meta 只需声明调用者已有 buffer 的元数据保持不变，直接 return None 是正确实现，与固定上游这两个算子的 fake 形式相同。形状/dtype/strides/device 由 out 或门面的 empty_like(input) 给定；FakeTensor/meta 下 empty_like 不产生真实设备存储。没有把 NPU tensor 临时转 CPU、读取值或运行一次 kernel 的路径。

不分配设备内存的检查覆盖三层：直接执行 fake 函数时，参数的任何属性访问、任何 torch dispatcher 张量操作和常用工厂函数均触发断言；通过注册 op 执行 Meta/FakeTensor 时，禁用 fallback，并将硬件查询、dispatch 和真实 impl 全部替换为抛错；最后在真 NPU 上对 float16/bfloat16/float32 的 (1024,4096) FakeTensor 调用两个公开 API，比较当前与峰值 allocated/reserved。另以 torch.library.opcheck 验证两个 mutation schema 和 FakeTensor 契约。

实际导入与内存检查命令：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest -q -s tests/test_torch_library.py::test_import_is_pure_and_registration_is_opt_in tests/test_jit_trigger_layer.py::test_import_selection_and_declarations_with_no_optional_dependencies_or_compiler tests/test_torch_library.py::test_fake_npu_allocates_zero_device_memory
```

关键输出（原始警告未改变结论）：

```text
{"provider_import_attempts": [], "compiler_probes": [], "registration_enabled": false}
.{"records": 79, "optional_import_attempts": [], "compiler_probe_attempts": [], "compiler_paths": {"bisheng": null, "ccec": null, "ascendc": null}, "selected_gemma": "triton/norm/gemma_fused_add_rmsnorm_v1"}
.NPU memory (allocated, reserved, peak_allocated, peak_reserved): before=(0, 0, 0, 0) after=(0, 0, 0, 0)
3 passed, 3 warnings in 2.15s
```

import 检查在 python -S 的新进程中阻断 torch/torch_npu/triton/acl 的任何导入尝试，同时将 shutil.which 和 subprocess.Popen 设为抛错。它包含 import flashinfer、flashinfer.utils、flashinfer._torch_library；T18 既有检查还遍历 79 条记录选型。结果均没有导入或编译器探测尝试。

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python tools/check_imports.py
```

```text
{"modules": 129, "failing": 0, "failures": []}
```

这条工具逐个导入所有模块，和上述专门验证 import flashinfer 纯净性的检查用途不同。

实际 NPU 端到端编译命令和完整输出如下。测试先取得未启用注册的 eager 结果，再显式启用，默认 Inductor fullgraph 执行两次（无 out / 传 out），逐个比较 y、input、residual，并校验传入 out 的身份不变。explain 图中必须同时保留两个 flashinfer 自定义算子节点。

```bash
TORCH_LOGS=graph_breaks /root/miniforge3/envs/flashinfer-a5/bin/python -u tools/t08_torch_compile_smoke.py
```

```text
[W1023 04:39:28.183334216 TensorFactories.cpp:437] Warning: Cannot create tensor with interal format while allow_internel_format=False, tensor will be created with base format. (function operator())
/root/miniforge3/envs/flashinfer-a5/lib/python3.11/site-packages/triton/language/extra/cann/extension/math_ops.py:13: FutureWarning: triton.language.extra.cann.extension.atan2 is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.atan2 instead.
  @_deprecated(replacement="triton.language.extra.cann.libdevice.atan2")
/root/miniforge3/envs/flashinfer-a5/lib/python3.11/site-packages/triton/language/extra/cann/extension/math_ops.py:35: FutureWarning: triton.language.extra.cann.extension.isfinited is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.isfinited instead.
  @_deprecated(replacement="triton.language.extra.cann.libdevice.isfinited")
/root/miniforge3/envs/flashinfer-a5/lib/python3.11/site-packages/triton/language/extra/cann/extension/math_ops.py:49: FutureWarning: triton.language.extra.cann.extension.finitef is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.finitef instead.
  @_deprecated(replacement="triton.language.extra.cann.libdevice.finitef")
PASS device=npu backend=inductor fullgraph=True out=False: graph_count=1 graph_break_count=0 break_reasons=[]; numerics and mutations match eager
graph targets: ['<built-in method empty_like of type object at 0x7990dad83640>', 'flashinfer.rmsnorm.default', 'flashinfer.fused_add_rmsnorm.default']
PASS device=npu backend=inductor fullgraph=True out=True: graph_count=1 graph_break_count=0 break_reasons=[]; numerics and mutations match eager
graph targets: ['flashinfer.rmsnorm.default', 'flashinfer.fused_add_rmsnorm.default']
```

该进程没有 graph-break 日志；同时 fullgraph=True 成功执行，explain 的 break_reasons 为空。结论限定于本次验证的两个算子及这些调用方式，未宣称所有算子均支持图模式。CPU 的默认 Inductor 也单独实测：

```bash
TORCH_LOGS=graph_breaks /root/miniforge3/envs/flashinfer-a5/bin/python -u tools/t08_torch_compile_smoke.py --device cpu
```

```text
PASS device=cpu backend=inductor fullgraph=True out=False: graph_count=1 graph_break_count=0 break_reasons=[]; numerics and mutations match eager
PASS device=cpu backend=inductor fullgraph=True out=True: graph_count=1 graph_break_count=0 break_reasons=[]; numerics and mutations match eager
```

测试还覆盖 CPU fullgraph + dynamic=True 的 batch 4 → 7 复用同一图、Backend 枚举参数、Meta 的 2D/3D/空 batch、FakeTensor 的非连续 strides，以及 helper 的 device_types/schema 参数转发和不支持 torch 版本的明确报错。新增的 NPU Inductor 测试标记 real_compile。

实际测试命令（全量在修改前、最终修改后各执行）：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest -q
/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest -q -s tests/test_torch_library.py
```

原样结果：

```text
3 failed, 1568 passed, 4 warnings in 13.64s
3 failed, 1592 passed, 4 warnings in 23.43s
24 passed, 3 warnings in 12.71s
```

三个既有失败的完整用例名以及实际/预期 ABI 差值经脚本逐项比较完全一致，没有修复或扩大失败范围：

```text
FAILED tests/test_attention_metadata_wire.py::AttentionMetadataWireTests::test_config_directory_layout_and_canonical_blob_are_frozen
FAILED tests/test_attention_quant_launch_abi.py::AttentionQuantizedKVLaunchABITests::test_v1_is_frozen_and_v2_has_an_independent_binary_identity
FAILED tests/test_runtime_abi.py::KernelBinaryABITests::test_attention_binary_abi_matches_logical_order_mutation_and_stream
E       AssertionError: '4fe29f10a3bc5582d41bbf066246302d79c31110e30bfb4df063068c44231c98' != 'f04f7b3f0f1a8ce0550e265c4d563a6da0a53260d1baab8eacbfb56af6068352'
E       AssertionError: 'b1120f7bfd2f436baa17d7365bac8e1e8287526f87d567056a5914e5f0b3229c' != '3087d1b9867361fbb9beac7a116ca60d15964be575943e0258481656b4f721ac'
E       AssertionError: '21d0e29e9c9aca68650c7cdc0a916da3540d11dffb55306f1e1318e3721b1b8e' != 'cb5285127b5926a81fcd28d2e27997ce5fc556d6c1a4a797f5d24a0060821b7c'
```

79 条 auto 选型对照使用同一环境下修改前保存的原始快照，比较 selected、hot_path_selected、fingerprint、signature、hardware、fallback 和逐候选 filters。默认与显式启用注册后分别执行，两个完整 compare 输出逐字相同。原始 baseline 位于 `/tmp/t08-selection-before.json`，其 SHA256 为 `0c39bb1c5c6f2461ab917708424a9233f8325ac23e971e4dfb75befba2b593ea`。

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t19_selection_audit.py --write /tmp/t08-selection-before.json
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t19_selection_audit.py --compare /tmp/t08-selection-before.json
/root/miniforge3/envs/flashinfer-a5/bin/python - <<'AUDIT'
import sys
from flashinfer.utils import enable_torch_library
enable_torch_library()
from tools.t19_selection_audit import main
sys.argv = ['t19_selection_audit.py', '--compare', '/tmp/t08-selection-before.json']
main()
AUDIT
```

修改前 write 输出：

```text
records: 79
stack: cann records: 40
stack: reference records: 38
stack: triton records: 1
multi_config_op_source_groups: 0 []
baseline: /tmp/t08-selection-before.json
```

修改后默认/启用两种模式的完整逐条表（相同，因此只列一份）：

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

```text
PASS 79/79 auto selections unchanged; 79/79 descriptor fingerprints unchanged
PASS hot-path selections, fallback order and per-candidate filters unchanged
```

GraphCapability / graph_param_updates、T6/T7 的过滤排序、T19 两级结构和 T20 tuning 判据均未修改。能够被 torch.compile 追踪不等同于 kernel 已经验证 CAPTURE_SAFE；既有 descriptor 指纹和保守 NOT_CAPTURABLE 声明保持原样。没有写 capture/replay，没有分配固定图 buffer；这些不属于 T8。

账本新增的三个入口均可由刷新器保持元数据不变。为了支持上游 helper 位于版本 if/else 内的定义，只给 `tools/update_parity_inventory.py` 增加同一词法作用域下的条件声明查找，并登记 enable_torch_library 的扩展归属；不进入函数内部，也不导入上游包。增加相应测试。针对 HEAD 基线和工作区账本，在内存中逐条调用 annotate，没有写回文件，输出：

```text
baseline refresh failures: 11
PASS unchanged refresh metadata: flashinfer.utils.register_custom_op
PASS unchanged refresh metadata: flashinfer.utils.register_fake_op
PASS unchanged refresh metadata: flashinfer.utils.enable_torch_library
T8 refresh failures: 11
PASS T8 adds no refresh failure; 11 existing unresolved symbols unchanged
unresolved ledger symbol: flashinfer.runtime.registry.KernelRegistry.register_external
unresolved ledger symbol: flashinfer.runtime.registry.register_external_kernel
unresolved ledger symbol: flashinfer.runtime.schema.Backend
unresolved ledger symbol: flashinfer.config.dispatch_order
unresolved ledger symbol: flashinfer.tuning.BenchmarkConfig
unresolved ledger symbol: flashinfer.tuning.BenchmarkCandidate
unresolved ledger symbol: flashinfer.tuning.collect_tuning_record
unresolved ledger symbol: flashinfer.tuning.query_tuning_record
unresolved ledger symbol: flashinfer.tuning.read_tuning_record
unresolved ledger symbol: flashinfer.tuning.save_tuning_record
unresolved ledger symbol: flashinfer.tuning.reload_tuning_records
```

相邻问题保留：上述刷新工具在基线已无法解析 11 个 T6/T7/T20 扩展，本轮未修；当前账本完整性验证仍通过（all 304 条、attention_core 36 条），全量刷新仍需另行处理这 11 个扩展。三个 ABI 冻结失败保留。设备运行时曾报告 AclrtGetDeviceInfo 507899 并回退到 aclrtGetMemInfo，另有已有的内部格式/弃用警告；本轮未调整驱动、CANN 或算子数学行为。plan/run 冻结、workspace 与 graph 参数更新仍属 T9–T11。

所有改动留在工作区，没有 commit/push，没有修改 docs/tasks/ 或 docs/TASKS.md。`git diff --check` 通过。
