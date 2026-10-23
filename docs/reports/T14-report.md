# T14 Ascend C 注册层与 meta 验收

2026-10-23。注册层沿用 T8 的显式启用方式，新增 `AscendCOp` 声明及 T13 探针的
两种产物桥接。builtin 默认仍走 eager；传入声明并调用 `enable_torch_library()` 后，
才向 dispatcher 注册。只为本仓已有的固定 FP32 `16×16×16` GEMM 探针增加
`flashinfer::matmul_probe`，未引入新的生产 descriptor。

环境：`/root/miniforge3/envs/flashinfer-a5/bin/python`，torch `2.10.0+cpu`，
torch_npu `2.10.0.post7`，8 张可用 NPU；实际执行使用 NPU 0 和本机 CANN
`/usr/local/Ascend/cann-9.2.0-beta.2`。native/CATLASS、host_shared/op_package
四种组合均实际执行了注册、FakeTensor、Inductor fullgraph、opcheck 和 graph break 正对照。

## 注册接口和新算子接入

`flashinfer.jit.torch_library.AscendCOp` 是不可变声明，包含 `name`、`schema`、
`device_impl`、`meta_impl` 和 `mutates_args`。构造声明不导入 torch、不做 I/O。
`enable_torch_library(ascendc_ops=(op, ...))` 先校验整个 batch 的名称、schema、
mutation、返回别名及重复注册，再调用 T8 的真实注册 helper。device kernel 挂
`PrivateUse1`，fake/meta kernel 独立注册；兼容尚未被 torch_npu 重命名的
`privateuseone` 和已重命名的 `npu`。同一声明对象重复启用是幂等的，拒绝替换已启用的实现。
torch 缺少 2.4 引入的 custom_op/register_fake API 时，包括重复 enable，均显式报错。

一个新 Ascend C 算子接入时需要：

1. 按 T13 契约准备并装载产物，写清 Python tensor 到该算子 host ABI 的转换。
2. 定义 `flashinfer::` schema 和 mutation；在对应算子族的 meta 模块中实现形状、
   dtype、stride 等规则，支持符号维度，避免读 tensor 值或调用设备。
3. 构造 `AscendCOp`，在 tracing 和并发执行之前显式 enable；补充该 op 的
   fullgraph/graph-break 正对照、FakeTensor 内存检查和 ABI 数值检查，以及对标账本。

通用声明示例：

```python
from flashinfer.jit.torch_library import AscendCOp
from flashinfer.utils import enable_torch_library

op = AscendCOp(
    name="flashinfer::my_op",
    schema="(Tensor x, Tensor(a!) out) -> ()",
    device_impl=prepared_device_runner,
    meta_impl=metadata_only_rule,
    mutates_args=("out",),
)
enable_torch_library(ascendc_ops=(op,))
# torch.ops.flashinfer.my_op(x, out)
```

本次探针与 T8 一样使用 caller-owned output：

```text
flashinfer::matmul_probe(Tensor a, Tensor b, Tensor(a!) out) -> ()
```

`flashinfer/_meta/ascendc.py` 校验三个 tensor 均为连续 FP32 `16×16` 且 device 一致。
输出的 shape/dtype/stride 已由调用方给定，因此 meta 不需要创建 tensor。
测试禁止所有 tensor dispatch、常用分配函数、data_ptr/item/storage 等数据访问和
动态库装载，直接调用该 meta 100 次仍通过。非法 rank/shape/dtype/stride/device 均被拒绝。

## 与 T13 两条产物路径对接

`matmul_probe_op()` 只适配已准备的 module，不编译、不装载、不注册。四种组合使用
相同 op 名、schema、mutation 和 meta；native/CATLASS 继续由原有配置选择。

host_shared 接收 T13 的 `JitLoadedModule` 和对应 `JitSpec`，核验 receipt/spec、
recipe digest、kernel/source/target 和 host entry，再桥接
`(torch current stream, a.data_ptr(), b.data_ptr(), out.data_ptr())`。
host_shared 原契约没有额外 output_adapter fingerprint，仍通过冻结 recipe 绑定 adapter，
没有为 T14 修改旧 JitSpec 或配方指纹。

```python
from flashinfer.jit.torch_library import matmul_probe_op
from flashinfer.utils import enable_torch_library

# recipe、spec、loaded 来自原有配置选择及 T13 build_and_load 流程。
op = matmul_probe_op(recipe, loaded, spec=spec)
enable_torch_library(ascendc_ops=(op,))
```

op_package 保留 T13-II 的完整 inventory/ELF/JSON 校验、完整 target_soc 身份、
进程级 package 注册和首次 ACL 初始化前装载的顺序：

```python
from flashinfer.jit.op_package import load_op_package
from flashinfer.jit.torch_library import matmul_probe_op
from flashinfer.utils import enable_torch_library

# 在 torch_npu/ACL 初始化之前执行；directory 是已验证的发行 generation。
module = load_op_package(directory, target_soc="ascend950")
op = matmul_probe_op(module.recipe, module)
enable_torch_library(ascendc_ops=(op,))
```

device bridge 从三个 ND tensor 创建 `aclTensor*`，调用原 module 的
`get_workspace_size(a, b, out)` 和 `launch(workspace, bytes, executor, stream)`。
需要时用 torch allocator 管理 workspace，使用 tensor 所属设备的 current stream。
两个 bridge 都检查 ND 存储及输出不与输入重叠，并保留 module 生命周期。
探针同步后释放描述符/workspace，包括失败路径；这只是正确性探针的同步调用，
不声明 aclgraph capture/replay 安全，不改 T9/T10/T11、T11B 或 descriptor graph 元数据。

## 三条可证伪验证

独立进程在已有 T13 产物上执行，且禁止 Ascend C discovery、source/dependency resolution
和 compilation。该 guard 保留 torch.compile 自己的图编译功能。实际命令：

```bash
PY=/root/miniforge3/envs/flashinfer-a5/bin/python
TORCH_LOGS=graph_breaks $PY tools/t14_torch_library_smoke.py \
  --config examples/ascendc_native.json \
  --cache-dir /tmp/flashinfer-t13ii-final-jit-native --cache-only
TORCH_LOGS=graph_breaks $PY tools/t14_torch_library_smoke.py \
  --config examples/ascendc_catlass.json \
  --cache-dir /tmp/flashinfer-t13ii-final-jit-catlass --cache-only
TORCH_LOGS=graph_breaks $PY tools/t14_torch_library_smoke.py \
  --package /tmp/flashinfer-t13ii-final-native/8110eee484a23e34554e7618c1ec0137bec58455a20a169b02fe6ef9b60aa8ed \
  --soc ascend950
TORCH_LOGS=graph_breaks $PY tools/t14_torch_library_smoke.py \
  --package /tmp/flashinfer-t13ii-final-catlass/2b66e4f9a8b6276972527cc2ae3c6fd5a18972900f3107e6b3516d73ee3d0bed \
  --soc ascend950
```

上述缓存/发行目录为 T13 已有产物，未修改。新增的两条 real_compile 集成测试另用
pytest 临时目录从当前配方冷构建 native host_shared 与 op_package，再运行同一验证工具；
不依赖这些历史临时目录。无 torch_npu/NPU/CANN 编译器时相应真机测试跳过。

四种组合的实际 fullgraph 输出：

```text
COMPILE PASS output=host_shared implementation=native backend=inductor fullgraph=True graph_count=1 graph_break_count=0 break_reasons=[] max_abs_error=0.0 mutation_identity=True
COMPILE PASS output=host_shared implementation=catlass backend=inductor fullgraph=True graph_count=1 graph_break_count=0 break_reasons=[] max_abs_error=0.0 mutation_identity=True
COMPILE PASS output=op_package implementation=native backend=inductor fullgraph=True graph_count=1 graph_break_count=0 break_reasons=[] max_abs_error=0.0 mutation_identity=True
COMPILE PASS output=op_package implementation=catlass backend=inductor fullgraph=True graph_count=1 graph_break_count=0 break_reasons=[] max_abs_error=0.0 mutation_identity=True
```

每个进程还输出以下共同结果：

```text
BUILD_GUARD PASS Ascend C discovery/resolution/compilation=forbidden
NAMESPACE PASS names=["flashinfer::fused_add_rmsnorm", "flashinfer::matmul_probe", "flashinfer::rmsnorm"]; PrivateUse1=True Meta=True CPU=False
META PASS FakeTensorMode calls=100 before=(0, 0, 0, 0) after=(0, 0, 0, 0)
graph targets: ['flashinfer.matmul_probe']
OPCHECK PASS {"test_faketensor": "SUCCESS", "test_schema": "SUCCESS"}
POSITIVE_CONTROL PASS graph_count=2 graph_break_count=1 break_reasons=1 fullgraph_rejected=True
```

四个内存计数依次为 allocated、reserved、peak_allocated、peak_reserved；每次 fake 调用
后都检查四项为 0，且把 device runner 替换为抛异常的 guard。不会把调用前已有内存的
“增量为零”写成“绝对为零”。

命名空间检查比较显式 enable 前后完整 dispatcher inventory 的差集，要求恰好为上述
三个 `flashinfer::` op，并检查当前 dispatcher 没有 `_C_ascend::`；不是只对过滤后的
flashinfer 列表做检查。名字包含 vllm/_c_ascend 或其它 namespace 的声明在注册前被拒绝。

正对照在同一个探针 op 后插入 `torch._dynamo.graph_break()`。
`TORCH_LOGS=graph_breaks` 实际报告 `Graph break in user code` 和
`User-inserted graph break`；explain 计数为 1，fullgraph 再次追踪时抛 Unsupported。
stderr 的正对照前后有明确标记，测试要求普通运行区间没有 graph break，正对照区间确有报告。

完整原始输出留在仓外 `/tmp/flashinfer-t14-{jit,aot}-{native,catlass}.log`。

## 导入纯净性和无设备环境

纯净性检查在 `python -S` 子进程拒绝 torch/torch_npu/triton/acl 导入，禁止
`shutil.which`、subprocess 和 ctypes.CDLL；覆盖 import flashinfer、注册模块、
新声明类型及 meta 模块的导入与声明构造。另一个子进程先导入 torch，然后拒绝
所有可选 provider 和编译器探测，真实 enable/PrivateUse1 注册仍成功。

实际命令和输出见下方测试记录。无设备依赖模拟将
`TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，并通过
MetaPathFinder 对 torch_npu/triton/acl 抛 ModuleNotFoundError；另外用
`torch.npu.is_available = lambda: False` 单独模拟无 NPU。两种情况都运行本次测试与
T8 测试，设备项目跳过，host/meta 项目仍执行。

## 全量测试与检查

实际执行：

```bash
PY=/root/miniforge3/envs/flashinfer-a5/bin/python
$PY -m pytest -q
$PY -m pytest -q tests/test_ascendc_torch_library.py tests/test_norm.py
$PY -m pytest -q -s \
  tests/test_ascendc_torch_library.py::test_import_and_declaration_are_torch_free \
  tests/test_ascendc_torch_library.py::test_setup_without_optional_providers_or_compilers \
  tests/test_torch_library.py::test_import_is_pure_and_registration_is_opt_in
$PY tools/check_imports.py
git diff --check
```

输出数字原样保留：

```text
baseline: 3 failed, 1723 passed, 4 warnings in 23.15s
final:    3 failed, 1760 passed, 4 warnings in 51.12s
85 passed, 3 warnings in 33.33s
3 passed in 2.83s
```

导入检查的原始有效载荷：

```text
{"provider_import_attempts": [], "compiler_or_library_probes": [], "registration_enabled": false}
PASS explicit setup; PrivateUse1=True torch_npu=False triton=False compiler_probes=0
{"provider_import_attempts": [], "compiler_probes": [], "registration_enabled": false}
```

全模块可导入检查另输出 `{"modules": 139, "failing": 0, "failures": []}`；
它与上面的 import 纯净性检查分别执行。`git diff --check` 无输出、退出 0。

无可选 provider 模拟的实际命令：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY - <<'PYTEST'
import importlib.abc
import sys
class NoProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch_npu', 'triton', 'acl'}:
            raise ModuleNotFoundError('simulated absent optional provider: ' + fullname, name=fullname)
sys.meta_path.insert(0, NoProviders())
import pytest
raise SystemExit(pytest.main(['-q', 'tests/test_ascendc_torch_library.py', 'tests/test_torch_library.py']))
PYTEST
```

```text
57 passed, 4 skipped in 2.82s
```

无 NPU 模拟另执行：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 $PY - <<'PYTEST'
import torch
import torch_npu
torch.npu.is_available = lambda: False
import pytest
raise SystemExit(pytest.main(['-q', 'tests/test_ascendc_torch_library.py', 'tests/test_torch_library.py']))
PYTEST
```

```text
57 passed, 4 skipped, 3 warnings in 1.89s
```

基线和最终日志中以 `FAILED ` 开头的失败名称列表、以 `E ` 开头的全部失败断言
逐行比较，输出：

```text
PASS 3/3 existing ABI failure names unchanged; no added failures
PASS all failure assertion outputs identical to baseline
FAILED tests/test_attention_metadata_wire.py::AttentionMetadataWireTests::test_config_directory_layout_and_canonical_blob_are_frozen
FAILED tests/test_attention_quant_launch_abi.py::AttentionQuantizedKVLaunchABITests::test_v1_is_frozen_and_v2_has_an_independent_binary_identity
FAILED tests/test_runtime_abi.py::KernelBinaryABITests::test_attention_binary_abi_matches_logical_order_mutation_and_stream
3 failed, 1723 passed, 4 warnings in 23.15s
3 failed, 1760 passed, 4 warnings in 51.12s
```

原始测试输出位于仓外 `/tmp/flashinfer-t14-{baseline,final,regression}-tests.log`、
`/tmp/flashinfer-t14-purity.log` 和 `/tmp/flashinfer-t14-no-{providers,npu}-final-tests.log`。

## 参考组织方式和保留的相邻问题

只读查看 `/home/dy/projects/vllm-ascend/csrc/torch_binding_meta.cpp` 的 1,899 行。
它在一个文件内按算子族放元信息函数、用共享输出形状 helper 减少重复，尾部集中挂
Meta impl，并有平台条件分组。本次只借鉴“形状规则与绑定分开、同族共享规则”的结构，
将本仓 Ascend C meta 放在独立族模块。未复制函数体、形状公式、常量、注册表、平台
条件、vLLM op 名或 `_C_ascend` namespace；也没有从它的 meta 注释推导 aclgraph 安全。

以下相邻问题保持原样：

- 三个既有 attention metadata / quant / launch ABI 冻结指纹失败；未修改协议或冻结值。
- `tools/update_parity_inventory.py` 原有 T13-I 分支的说明文字仍把 AOT/op-package 写成
  留给 13-II；本次只增加 T14 新接口的刷新分支，没有重写 T13 历史说明。
- T13-II 仍要求首次 ACL 初始化前装载 package，并拒绝进程内切换另一 generation。
  新 bridge 遵循该装载契约。固定探针仍不提供通用 GEMM、性能或 aclgraph 支持承诺。

全部产物、算子包和日志位于仓外。`3rdparty/catlass` 保持原 gitlink；
`docs/tasks/`、`docs/TASKS.md` 未修改，没有 commit/push。


## 79 条 auto 选型逐行对照

baseline 在改代码前保存，SHA-256 为
`7bd7f5abe410e6823fd7def504f75e1f2fa85b10c2837b2b2b6d8bec8222384b`。

```bash
$PY tools/t19_selection_audit.py --write /tmp/flashinfer-t14-selection-before.json
$PY tools/t19_selection_audit.py --compare /tmp/flashinfer-t14-selection-before.json
```

另一个进程以 T13 cache-only 装载 native host_shared，构造并 enable 真实
`matmul_probe_op` 后，再执行同一个 `tools.t19_selection_audit.main()` compare。
默认与 enable 后的完整 compare 输出经 cmp 检查，逐字相同；比较包括 selected、
完整 descriptor fingerprint、hot_path_selected、fallback、逐候选 filters、signature
和 hardware，而不只比较 op 名。

显式 enable 后的实际命令：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 $PY - <<'AUDIT'
import json
import sys
from pathlib import Path
from flashinfer.jit import AscendCBackend, DiskJitBuildStore, JitCompilationPolicy, JitSpec, build_and_load
from flashinfer.jit.recipe import BuildRecipe
from flashinfer.jit.torch_library import matmul_probe_op
from flashinfer.utils import enable_torch_library
cache=DiskJitBuildStore('/tmp/flashinfer-t13ii-final-jit-native')
spec=JitSpec.from_dict(json.loads(Path('/tmp/flashinfer-t13ii-final-jit-native/native-spec.json').read_text()))
loaded=build_and_load(spec,backend=AscendCBackend(),store=cache,policy=JitCompilationPolicy.CACHE_ONLY)
receipt=json.loads((cache.directory(cache.lookup(spec))/'ascendc-recipe.json').read_text())
recipe=BuildRecipe.from_dict(receipt['resolved']['recipe'])
op=matmul_probe_op(recipe,loaded,spec=spec)
enable_torch_library(ascendc_ops=(op,))
from tools.t19_selection_audit import main
sys.argv=['t19_selection_audit.py','--compare','/tmp/flashinfer-t14-selection-before.json']
main()
AUDIT
cmp /tmp/flashinfer-t14-selection-default.log /tmp/flashinfer-t14-selection-enabled.log
```

```text
records: 79
stack: cann records: 40
stack: reference records: 38
stack: triton records: 1
multi_config_op_source_groups: 0 []
PASS 79/79 auto selections unchanged; 79/79 descriptor fingerprints unchanged
PASS hot-path selections, fallback order and per-candidate filters unchanged
```

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
