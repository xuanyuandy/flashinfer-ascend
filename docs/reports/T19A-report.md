# T19A · 来源栈、两级 dispatch 与真实多配置验收

日期：2026-10-23。改动保留在工作区，未 commit / push。
Python 均为 `/root/miniforge3/envs/flashinfer-a5/bin/python`。

## 1. 修改前独立复核

直接遍历 `get_global_registry().descriptors()`，按来源分组，再数每组内的 op，
没有采用上游给出的组合数作为输入。结果：

| 来源栈 | 记录数 | 实际成员 |
|---|---:|---|
| `cann` | 40 | 下方完整对照表内全部 `cann/` 记录 |
| `reference` | 38 | 下方完整对照表内全部 `reference/` 记录 |
| `triton` | 1 | `triton/norm/gemma_fused_add_rmsnorm_v1` |
| `ascendc` / 具体外部仓 / `custom` | 0 | 无登记记录 |

同一 op、同一来源有多份实现的组合数为 **0**，所以原始 79 条记录没有任何栈内
多配置选择数据。多实现 op 有 36 个：35 个为 CANN + reference，另 1 个为
Triton + reference（`norm.gemma_fused_add_rmsnorm`）。因此“全部都是 CANN +
reference”不成立，但不影响第二级原本为空的结论。

请求的 `docs/tasks/T19-two-level-selection.md` 不存在，实际阅读的是
`docs/tasks/T19-two-level-dispatch.md`，以及 TASKS 全局约束、T18/T18B 报告、
dispatch/schema/registry 和对应测试。没有修改 `docs/tasks/` 或 `docs/TASKS.md`。

## 2. 最终数据形状与两级控制流

```python
@dataclass(frozen=True)
class ImplementationStack:
    stack_id: str
    descriptors: Tuple[KernelDescriptor, ...]
```

`KernelRegistry.stacks(op=None)` 返回稳定排序的只读快照，引用原 descriptor；
`stack_id_for(kernel_id)` 查询来源身份。栈是登记表的派生视图，不新增一份可漂移的
descriptor、capability 或排序事实源，也不修改 79 条记录的序列化与指纹。

来源身份跨 op 稳定。CANN 与历史 ACLNN 归 `cann`；Ascend C 的普通、历史 AOT/JIT
backend 都归 `ascendc`；本仓 Triton 为 `triton`。具体外部仓的身份为
`external:<T6.2 kernel_id 仓库命名空间>`，例如 `external:repo-a` 与
`external:repo-b`，两者 Backend 均为 EXTERNAL，仍是两个栈。custom 保留现有用户
自定义入口池的身份与 opt-in 语义。编译方式和具体配置不改变来源栈。

普通 dispatch 的控制流为：

1. 获取同一 op 的一份栈快照，按原 kernel_id 顺序遍历其中的实现。
2. 保持原有 policy、custom/reference opt-in、capability、target SoC、JIT 过滤。
3. 按栈收集通过者；没有通过者的栈淘汰，保留每条实现的全部原因。
4. **第一级选栈**：以栈内最优可用实现的既有 T7 键
   `(tier, -priority, kernel_id)` 比较。现有 tuning 冠军只有通过过滤才可锁定所在栈。
5. **第二级选实现**：仅在选中栈的通过者中应用同一个 T7 键或现有 tuning 冠军。
6. 后备链仍按原来的全局逐 kernel 顺序生成，避免把同栈的低档配置提前。

启发式下 `min_stack(min_impl(T7_key)) == min_all_impl(T7_key)`，所以分层不改
原选型。栈不能简单固定一个 tier：同一来源可能同时有 AOT/JIT，且最优配置可能被
capability 过滤。现有 tuning 查询仍是原来的 stub，没有新增调优协议或采样。
过滤期间的新登记留给下一次 revision 决策，避免新记录混入本次未评估的报告。

显式 kernel_id 保持原契约：只检查指定实现的 capability/target，跳过 auto opt-in、
JIT 禁用和排序；解释中明确其他栈/配置未评估，不虚构比较结果。

`DispatchDecision` 追加 `stack_id` 和 `stack_reports`。
`StackReport` 含 `stack_id / accepted / selected / representative_kernel_id /
reasons / implementations`：区分“栈可用”与“栈被选中”，说明栈间比较的具体键；
嵌套 CandidateReport 保留逐实现拒绝原因，并为通过者解释唯一候选、tier、priority、
kernel_id 字典序或既有 tuning 冠军。无候选错误也输出两级原因。
原 `decision.reports` 继续只表示过滤，原 `reason` 和 fallback 契约保持。

## 3. 两级解释的真实输出

以下来自最终示例的新进程缓存装载运行，未改写解释文本。

```text
dispatch norm.gemma_fused_add_rmsnorm → triton/t19a/gemma/block128（backend=triton）
  选型依据：确定性启发式 tier 3（自研 JIT / 未声明 AOT）
  第一级：选择实现栈
    [reject] reference：栈不可用：栈内全部 1 个实现被过滤；reference/norm/gemma_fused_add_rmsnorm_v1：reference 需显式 opt-in（backend='reference' 或 allow_reference=True）
    [select] triton：栈内 2/2 个实现通过过滤；比较边界：tier=3（自研 JIT / 未声明 AOT）, priority=0, kernel_id=triton/t19a/gemma/block128；栈间 T7 排序第一
  第二级：各栈内的实现（含逐条过滤与排序原因）
    栈 reference，首选=无
      [reject] reference/norm/gemma_fused_add_rmsnorm_v1 (reference)：reference 需显式 opt-in（backend='reference' 或 allow_reference=True）
    栈 triton，首选=triton/t19a/gemma/block128（最终选中栈）
      [accept] triton/t19a/gemma/block128 (triton)：通过全部候选过滤；tier=3（自研 JIT / 未声明 AOT）, priority=0, kernel_id=triton/t19a/gemma/block128；栈内 T7 排序第一
      [accept] triton/t19a/gemma/block256 (triton)：通过全部候选过滤；tier=3（自研 JIT / 未声明 AOT）, priority=0, kernel_id=triton/t19a/gemma/block256；栈内排序落后于 triton/t19a/gemma/block128：同 tier / priority，kernel_id 字典序 triton/t19a/gemma/block128 < triton/t19a/gemma/block256
  fallback：triton/t19a/gemma/block256

dispatch norm.gemma_fused_add_rmsnorm → triton/t19a/gemma/block256（backend=triton）
  选型依据：确定性启发式 tier 3（自研 JIT / 未声明 AOT）
  第一级：选择实现栈
    [reject] reference：栈不可用：栈内全部 1 个实现被过滤；reference/norm/gemma_fused_add_rmsnorm_v1：reference 需显式 opt-in（backend='reference' 或 allow_reference=True）
    [select] triton：栈内 1/2 个实现通过过滤；比较边界：tier=3（自研 JIT / 未声明 AOT）, priority=0, kernel_id=triton/t19a/gemma/block256；栈间 T7 排序第一
  第二级：各栈内的实现（含逐条过滤与排序原因）
    栈 reference，首选=无
      [reject] reference/norm/gemma_fused_add_rmsnorm_v1 (reference)：reference 需显式 opt-in（backend='reference' 或 allow_reference=True）
    栈 triton，首选=triton/t19a/gemma/block256（最终选中栈）
      [reject] triton/t19a/gemma/block128 (triton)：hidden 256 不在支持区间 [1, 128]
      [accept] triton/t19a/gemma/block256 (triton)：通过全部候选过滤；tier=3（自研 JIT / 未声明 AOT）, priority=0, kernel_id=triton/t19a/gemma/block256；栈内唯一通过过滤的实现
```

另对内置 `norm.rmsnorm` 使用 `allow_reference=True`，可以看到“可用但栈间排序
落后”的真实来源比较；reference 不会因为通过过滤就被误报为选中。

```text
dispatch norm.rmsnorm → cann/norm/rmsnorm_aclnn_v1（backend=cann）
  选型依据：确定性启发式 tier 1（CANN / ACLNN 算子）
  第一级：选择实现栈
    [select] cann：栈内 1/1 个实现通过过滤；比较边界：tier=1（CANN / ACLNN 算子）, priority=0, kernel_id=cann/norm/rmsnorm_aclnn_v1；栈间 T7 排序第一
    [accept] reference：栈内 1/1 个实现通过过滤；比较边界：tier=4（reference）, priority=0, kernel_id=reference/norm/rmsnorm_v1；栈可用但未选中：落后于 cann：tier 1 < 4
  第二级：各栈内的实现（含逐条过滤与排序原因）
    栈 cann，首选=cann/norm/rmsnorm_aclnn_v1（最终选中栈）
      [accept] cann/norm/rmsnorm_aclnn_v1 (cann)：通过全部候选过滤；tier=1（CANN / ACLNN 算子）, priority=0, kernel_id=cann/norm/rmsnorm_aclnn_v1；栈内唯一通过过滤的实现
    栈 reference，首选=reference/norm/rmsnorm_v1
      [accept] reference/norm/rmsnorm_v1 (reference)：通过全部候选过滤；tier=4（reference）, priority=0, kernel_id=reference/norm/rmsnorm_v1；栈内唯一通过过滤的实现
  fallback：reference/norm/rmsnorm_v1
```

## 4. 真实多配置栈

`examples/two_level_gemma.py` 显式建立独立 KernelRegistry：一个 Triton 栈内有
`triton/t19a/gemma/block128` 和 `triton/t19a/gemma/block256`，并登记真实 reference
来源以演示第一级过滤。生产全局登记表仍为原始 79 条。这是调用真实 dispatch、
真实统一 JIT 构建/装载入口的可复现案例，不是默认生产策略变更。

两份配置使用 **T18B 原样的 `triton/_norm_kernels.py` 和同一入口**
`_gemma_fused_add_rmsnorm_kernel`；区别是 JitSpec 中真实的 constexpr
`BLOCK=128` / `BLOCK=256`。hidden/eps 仍是运行时参数。配置没有通用新 schema：
具体身份仍是 kernel_id、SourceRef、JitSpec 编译选项和完整 spec 指纹。

能力范围分别声明 hidden ∈ [1,128] / [1,256]，是实际单 tile 容量：BLOCK=128
只能处理前 128 列，不能正确执行 hidden=256。使用原 ShapeRange 谓词，无谓词修改。
两配置均验证 float32；不外推到其他 dtype。两个都可用时按既有 kernel_id 字典序选
block128，这只是确定性规则，没有声称更快。hidden=256 时 block128 真正不支持，
所以自动选择 block256。两次 auto 决策都真实编译、装载并执行所选实现。

| 配置 | npubin 字节 | npubin SHA-256 | 可执行代码字节 | 代码 SHA-256 |
|---|---:|---|---:|---|
| BLOCK=128 | 22816 | `a81be8a44b6bfadd32bb05aed63476663db020803ef91160d999195a584f30f4` | 2880 | `d6742b6197c6e9af86a04d54995770562a51d687effb7da0c203708917c27262` |
| BLOCK=256 | 22808 | `96bf8a8cfbc182b8e43fd670eed4c89ef152fe26c7af572fdefa74715611e2e1` | 2872 | `d59dcfa23cf1d100d31022f6c76b2bf277337cca3076007953427cf3507d6ac0` |

还验证了 ELF64 NPU machine=0x1029、真实入口符号；比较可执行 PROGBITS 段，排除
仅路径、文件头或元数据不同的情况。完整产物和编译日志全部在
`/tmp/flashinfer-t19a-gemma/`，没有加入仓库。

实编命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 /root/miniforge3/envs/flashinfer-a5/bin/python examples/two_level_gemma.py --cache-dir /tmp/flashinfer-t19a-gemma
```

实际输出摘录（两份的首次 `disk_cache_hit` 都是 False）：

```text
PASS auto kernel=triton/t19a/gemma/block128 float32 shape=(4,128) max_abs_error=0.0 residual_exact=True
PASS auto kernel=triton/t19a/gemma/block256 float32 shape=(4,256) max_abs_error=9.5367431640625e-07 residual_exact=True
PASS 2 distinct auto selections; 2 distinct real NPU ELF artifacts; device code differs
PASS overlap kernel=triton/t19a/gemma/block256 float32 shape=(4,128) max_abs_error=4.76837158203125e-07 residual_exact=True
```

新进程复验命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 /root/miniforge3/envs/flashinfer-a5/bin/python examples/two_level_gemma.py --cache-dir /tmp/flashinfer-t19a-gemma --cache-only
```

两份 `disk_cache_hit: True`，数值结果与上面相同，且输出：

```text
PASS fresh-process cache-only run with all subprocesses forbidden
```

## 5. 测试、导入前提与逐条选择对照

修改前后同一命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 /root/miniforge3/envs/flashinfer-a5/bin/python -m pytest tests/ -q
```

修改前实际结果：

```text
FAILED tests/test_attention_metadata_wire.py::AttentionMetadataWireTests::test_config_directory_layout_and_canonical_blob_are_frozen
FAILED tests/test_attention_quant_launch_abi.py::AttentionQuantizedKVLaunchABITests::test_v1_is_frozen_and_v2_has_an_independent_binary_identity
FAILED tests/test_runtime_abi.py::KernelBinaryABITests::test_attention_binary_abi_matches_logical_order_mutation_and_stream
3 failed, 1453 passed, 4 warnings in 12.57s
```

最终实际结果：

```text
FAILED tests/test_attention_metadata_wire.py::AttentionMetadataWireTests::test_config_directory_layout_and_canonical_blob_are_frozen
FAILED tests/test_attention_quant_launch_abi.py::AttentionQuantizedKVLaunchABITests::test_v1_is_frozen_and_v2_has_an_independent_binary_identity
FAILED tests/test_runtime_abi.py::KernelBinaryABITests::test_attention_binary_abi_matches_logical_order_mutation_and_stream
3 failed, 1481 passed, 4 warnings in 12.74s
```

失败名称和实际/预期指纹差异逐项比对完全相同。新增 28 个通过用例，覆盖栈的完备
划分、历史 backend 合栈、两个外部仓不混栈、冻结后注册、同一决策快照、两级结构与
解释、三层 T7 比较、部分配置淘汰、无候选错误、显式覆盖、既有 tuning 过滤边界、
真实案例 tile 容量边界和新接口的账本往返。

最终相关测试命令：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest tests/test_two_level_dispatch.py tests/test_dispatch.py tests/test_dispatch_ordering.py tests/test_physical_layouts.py tests/test_shape_ranges.py -q
```

```text
246 passed in 0.16s
```

导入、选型、声明的无依赖/无编译器审计（独立 `python -S`、PATH 为空、禁止可选
包、shutil.which 和所有 subprocess）：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest tests/test_jit_trigger_layer.py::test_import_selection_and_declarations_with_no_optional_dependencies_or_compiler -q -s
```

```text
{"records": 79, "optional_import_attempts": [], "compiler_probe_attempts": [], "compiler_paths": {"bisheng": null, "ccec": null, "ascendc": null}, "selected_gemma": "triton/norm/gemma_fused_add_rmsnorm_v1"}
1 passed in 0.14s
```

新接口已更新 `api_parity.json` 和可重生成的 annotate 规则：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -m flashinfer parity-check
git diff --check
```

```text
Parity ledger integrity: passed
```

修改前保存并在修改后逐项重放的命令：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t19_selection_audit.py --write /tmp/flashinfer-t19a-selection-before.json
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t19_selection_audit.py --compare /tmp/flashinfer-t19a-selection-before.json
```

每条记录构造一个有效代表请求：dtype/layout/SoC 取能力声明的首项（未限定则分别
取 float16 / 空逻辑 layout / Ascend910B），shape=(4,128)，AI Core=32，特化项
取各允许集首项。使用 backend=auto、allow_reference=True 覆盖 reference-only op；
另比对默认不允许 reference 的全部 79 个热路径结果（包含无候选）。这不是跨任意
workload 的穷举，完整请求和修改前结果保存在上述 JSON，生成器已随代码提供。
同时复放了 T18B 留下的原始 `/tmp/flashinfer-t18b-selection-baseline.json`，其
79/79 auto 选择和 descriptor 指纹也不变。

以下是独立保存的本次 79 条记录逐条前后对照；记录前缀同时给出实际栈成员。

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

## 6. 必须留待 19B 的结论

- 共同支持区间内哪个 BLOCK 更快、是否应按 shape/dtype/SKU 切换，缺少实测数据。
  本次字典序和 tile 容量都不能作为性能优劣证据。
- 需要搜索哪些 tiling/block/并行配置、搜索预算和有效候选集，必须由真实 kernel
  的可编译性、数值结果和测量成本决定；这两份配置不足以设计通用搜索空间。
- 调优记录应怎样按完整 workload/硬件/环境/产物版本区分、何时过期，以及跨栈与
  栈内测量结果怎样组合，需用真实调优数据验证。现有 tuning stub 保持原状，没有
  先造数据库、通用选择器、性能采样或自动选优策略。

## 7. 遇到但没有修改的相邻事项

- 三个 Attention/ABI 冻结指纹失败保持原样，没有修改 ABI 或冻结断言。
- 既有决策缓存 key 含 `registry.revision`，却没有 registry 身份。复现：先用一个
  revision=1 的 registry 选择 `triton/cache-audit`，再对同一签名/硬件使用另一个
  revision=1、只登记 `cann/cache-audit` 的 registry，第二次错误命中前者的缓存。
  该 key 在本次 diff 中未变；本次不扩展到缓存隔离修复。真实示例是单一独立 registry，
  测试沿用每例清空缓存的 fixture。
- `KernelRegistry.select()` 仍是独立的 manifest WorkloadSpec/DeviceCapability
  入口，保留 T6 的额外 KernelConstraints 与 T7 排序。本次修改的是题目指定的
  InputSignature/HardwareSignature dispatch；没有从残缺 manifest 输入反推 tensor
  shape，也没有顺手改造 select 的接口。
- T18B 的 Ascend C host wrapper/launch ABI 仍未实现；本次真实案例仅验证 Triton。

所有改动均在工作区；没有编译产物或构建日志入仓，没有修改任务文件或新增
work_items 类目录。
