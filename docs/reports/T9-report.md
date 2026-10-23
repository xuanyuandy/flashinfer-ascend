T9 交付报告 · 2026-10-23

本次完成 P3 第二步的 plan 固定资源绑定与冻结校验。首次成功的 graph-enabled plan 即建立冻结状态，与上游生命周期一致；本仓无法获知调用方何时开始捕获。没有实现或调用 aclgraph capture/replay，没有改变公开构造函数和 plan 签名。

独立核对任务书四项，结果如下。核对依据为当前工作区修改前源码，以及 `/home/dy/projects/flashinfer-ori` 的 `da992ed9919604c5eb6dc255c63f5e89a466452c`；未导入 CUDA 上游包。

| 任务书项目 | 修改前真实状态 | 本次处理 |
|---|---|---|
| 1. 构造时接收固定 buffer | 本地 BatchDecode 构造函数已有 9 个参数，含 use_cuda_graph、三个 paged KV 固定 buffer、backend 和 jit_args；上游普通 wrapper 同样为 9 个。任务书“本地只有 2 个参数”已过时。ReferenceTensor 路径已有资源描述；真实 tensor 开图仍直接报未实现。 | 保持公开签名，接通真实 tensor 的固定资源校验与持有；补齐 workspace 同设备、连续性和不重叠约束。 |
| 2. plan 只写固定 buffer | 原先只检查 ReferenceTensor buffer 容量/形状，未 copy_ 元数据；真实 tensor 图路径在构造时被拒绝。 | decode、paged prefill、ragged prefill 均写入调用方固定 buffer，不替换设备存储；可选 mask 同样原地更新。 |
| 3. 冻结维度 | AttentionFrameworkSession 已有 batch size 和 q_len_per_req 冻结，不能算本次新增；真实 tensor session 原先没有 graph 标志。后端解析结果与 block table 宽度尚未冻结。 | 接通既有两维，增加实际解析结果、kernel ID/descriptor 指纹、block table 宽度及有无变化冻结；所有校验先于 buffer 写入和新 plan 发布。 |
| 4. decode 两处误导描述 | 一次性 cudnn_batch_decode_with_kv_cache 的文档与 del 行注释仍称“NPU 无对应机制”。 | 改为准确的资源绑定说明，说明 aclgraph 由调用方负责，当前无状态适配器仍保留兼容参数但没有固定图资源保证。 |

上游检查的精确范围：普通 plan 的 `_plan_impl` 检查 `_cudnn_auto_frozen`、`_q_len_per_req` 和 `_fixed_batch_size`；CUDAGraph 专用 wrapper 把三个调用方 buffer 传给父类并强制 use_cuda_graph=True。`_plan_cudnn` 禁止已分配的自动 block table 在图模式下增长；对调用方 table 的分支验证容量/设备，当前检出没有单独的无条件“宽度必须等于前次”判断。因此任务书“上游所有 block table 宽度变化都报错”的概括也不够精确。本次按用户要求实行更严格的宽度增长、缩小、None 与非 None 切换均拒绝。

修改文件及原因：

| 文件 | 改动原因 |
|---|---|
| flashinfer/attention/execution_identity.py | 在 AttentionGraphResourceContract 上增加私有 plan 校验，进程内 frozen state 与序列化资源 ABI 分开，保持原有 schema/指纹格式。 |
| flashinfer/attention/batch.py | 固定 buffer 校验扩展到真实可写 tensor；ReferenceTensor 的原有 dtype 断言保持严格。沿用 is_graph_enabled / is_cuda_graph_enabled 兼容别名。 |
| flashinfer/attention/planner.py | 保留原有 batch/q_len 断言，仅让错误包含 planned/got 数值。 |
| flashinfer/attention/public_batch.py | 共享门面是真实 plan 实现所在。接通固定资源、两阶段校验/发布和 copy_；首次提供的 block_tables 由 wrapper 持有，后续只拷贝到原表；run 的分页索引读固定存储。拒绝外部 resize 与 native graph workspace 换址。保留外部 provider 图模式的既有未实现错误优先级。 |
| flashinfer/decode.py | 补生命周期说明，修正两处“NPU 无对应机制”的误导文字，公开参数名不变。 |
| flashinfer/prefill.py | 说明共享 plan 图契约和调用方捕获责任。 |
| flashinfer/data/api_parity.json、flashinfer/data/attention_api_parity.json | 同步既有条目的资源/冻结进度与兼容别名状态，保留 T10/T11 限制，不新增公开接口。 |
| tests/test_attention_graph_plan.py | 新增 21 项真实 NPU 验证：三种 wrapper 的地址/内容与设备工厂禁分配、四维冻结、同后端换 kernel、table 增减/有无、固定索引确实参与 run、失败不发布/不写入、外部 resize/workspace 换址、eager 可重规划、prefill packed/unpacked mask 更新与数值。 |
| tools/t09_graph_plan_audit.py | 可重复正对照及内存中关闭冻结的负对照；不改源码、不捕获图。 |
| docs/reports/T9-report.md | 保存本次命令、真实输出与 79 行逐条对照；没有修改任务文档。 |

验证环境输出：

```text
python: /root/miniforge3/envs/flashinfer-a5/bin/python
torch: 2.10.0+cpu
torch_npu: 2.10.0.post7
NPU available: True
NPU count: 8
NPU model: Ascend950PR_9579
```

固定设备地址与冻结检查的命令：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t09_graph_plan_audit.py
```

完整输出如下。地址来自真实 NPU tensor 的 data_ptr()；前两次 plan 更换元数据内容，并验证调用方 buffer 本身与其有效内容，未用假地址。测试还拦截常用 tensor 工厂，禁止 plan 创建新的 NPU 存储；CPU 源数据的临时 staging 允许存在。可选 block table 的首次传入地址也独立验证。

后端变化测试需要说明：当前真实 tensor 门面只解析到 CANN，无法自然切换到第二种实现栈。该项在真实 NPU wrapper 的第二次 plan 中对 dispatch 返回的 backend 注入 cann→ascendc，以验证冻结门禁；没有伪造 buffer 或实际执行 Ascend C。另一个同后端换 kernel 的测试使用独立临时 KernelRegistry 和真实 dispatch 选中更高优先级 CANN kernel；没有修改全局登记表。

```text
decode data_ptr: before=(19792499049472, 19792499049984, 19792499050496) plan1=(19792499049472, 19792499049984, 19792499050496) plan2=(19792499049472, 19792499049984, 19792499050496); no device allocation
.paged_prefill data_ptr: before=(19792499049472, 19792499049984, 19792499050496, 19792499051008) plan1=(19792499049472, 19792499049984, 19792499050496, 19792499051008) plan2=(19792499049472, 19792499049984, 19792499050496, 19792499051008); no device allocation
.ragged_prefill data_ptr: before=(19792499049472, 19792499049984) plan1=(19792499049472, 19792499049984) plan2=(19792499049472, 19792499049984); no device allocation
.batch_size: AttentionStateError: graph-enabled wrapper requires a fixed batch size: planned 2, got 1
.resolved_backend: AttentionStateError: graph-enabled wrapper requires a fixed resolved_backend: planned 'cann', got 'ascendc'; use a separate wrapper
.q_len_per_req: AttentionStateError: graph-enabled wrapper requires a fixed q_len_per_req: planned 1, got 2
.block_table_width: AttentionStateError: graph-enabled wrapper requires a fixed block_table_width: planned 2, got 1; use a separate wrapper
.same backend, different kernel: graph-enabled wrapper requires a fixed kernel_identity: planned ('cann/attention/batch_decode_paged_fia_v1', '5ee76134f5bf3a2ab4be30f20a7e8694883a32eb20dbd4863f82e178fdf3d084'), got ('cann/test/another_graph_kernel', 'b050353a7cf48db6e9986fb0cf4563d5ec4b2b42a26fbf8fa0a0175bf8e3beac'); use a separate wrapper
.block_table_width 2->3: graph-enabled wrapper requires a fixed block_table_width: planned 2, got 3; use a separate wrapper
.block_table_width 3->2: graph-enabled wrapper requires a fixed block_table_width: planned 3, got 2; use a separate wrapper
.block_table_width None->2: graph-enabled wrapper requires a fixed block_table_width: planned None, got 2; use a separate wrapper
.block_table_width 2->None: graph-enabled wrapper requires a fixed block_table_width: planned 2, got None; use a separate wrapper
.block_tables data_ptr: plan1=19792499060736 plan2=19792499060736; run reads fixed storage
.........
=============================== warnings summary ===============================
tests/test_attention_graph_plan.py::test_graph_prefill_custom_mask_copies_and_runs[True-False]
  /home/dy/projects/flashinfer/flashinfer/cann/attention.py:95: UserWarning: Cannot create tensor with interal format while allow_internel_format=False, tensor will be created with base format. (Triggered internally at ../torch_npu/csrc/aten/common/TensorFactories.cpp:437.)
    unpacked = ((packed_custom_mask.flatten()[bits // 8].int() >> (bits % 8)) & 1).bool()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
21 passed, 1 warning in 5.86s
```

地址保证的边界：真实 graph-enabled tensor 路径在 plan 中不替换持有的固定 buffer。ReferenceTensor 是不可变 host 结构模型，既没有 data_ptr，也不执行设备 copy_；这里没有把该模型当作物理地址证据。Eager run 仍保留既有临时 tensor 分配；graph run 的 KV 整理、动态长度和 tiling 尚不能据此宣称已完整支持 aclgraph 重放。

负对照命令：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t09_graph_plan_audit.py --negative-control
```

工具在内存里关闭 AttentionFrameworkSession 的 batch/q_len 冻结和 AttentionGraphResourceContract 的冻结校验，其他 metadata 校验与 copy_ 保留。两个 block table 的宽度为 2→1，均能承载同一组索引；若无冻结，底层 copy_ 可以广播，所以不会以拷贝形状错误冒充冻结错误。四维第二次 plan 全部成功，测试逐条因 DID NOT RAISE 失败。结果汇总为原样输出：

```text
NEGATIVE CONTROL: session batch/q_len and graph contract freezes disabled in memory
E       Failed: DID NOT RAISE <class 'flashinfer.attention.planner.AttentionStateError'>
E       Failed: DID NOT RAISE <class 'flashinfer.attention.planner.AttentionStateError'>
E       Failed: DID NOT RAISE <class 'flashinfer.attention.planner.AttentionStateError'>
E       Failed: DID NOT RAISE <class 'flashinfer.attention.planner.AttentionStateError'>
FAILED tests/test_attention_graph_plan.py::test_graph_plan_freezes_dimension[batch_size]
FAILED tests/test_attention_graph_plan.py::test_graph_plan_freezes_dimension[resolved_backend]
FAILED tests/test_attention_graph_plan.py::test_graph_plan_freezes_dimension[q_len_per_req]
FAILED tests/test_attention_graph_plan.py::test_graph_plan_freezes_dimension[block_table_width]
4 failed in 5.50s
PASS negative control: 4/4 freeze tests failed with guards removed; source files unchanged
```

工具要求恰好上述四项调用阶段失败，出现 skip、收集错误或其他失败集合均不会把负对照判为成功；finally 恢复进程内方法，源码始终未改写。

全量测试命令（修改前、修改后）：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest -q
```

原样数字：

```text
3 failed, 1592 passed, 4 warnings in 22.97s
3 failed, 1613 passed, 4 warnings in 22.29s
```

中间一次全量曾暴露外部 provider 图模式错误优先级和两本账本不同步的问题，现均已修正；当次结果与修复后针对性命令也保留：

```text
8 failed, 1608 passed, 4 warnings in 23.15s
```

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest -q tests/test_checkpoint_038_public_paged_prefill_provider_runtime.py tests/test_checkpoint_039_public_paged_decode_provider_runtime.py tests/test_checkpoint_040_public_ragged_prefill_provider_runtime.py tests/test_parity_tracking.py
```

```text
30 passed in 0.10s
```

基线与最终日志的失败 ID 和实际/预期 ABI 指纹经 Python re.findall 逐项比较，结果：

```text
PASS unchanged 3/3 failing test IDs and 3/3 ABI actual/expected fingerprints
FAILED tests/test_attention_metadata_wire.py::AttentionMetadataWireTests::test_config_directory_layout_and_canonical_blob_are_frozen
FAILED tests/test_attention_quant_launch_abi.py::AttentionQuantizedKVLaunchABITests::test_v1_is_frozen_and_v2_has_an_independent_binary_identity
FAILED tests/test_runtime_abi.py::KernelBinaryABITests::test_attention_binary_abi_matches_logical_order_mutation_and_stream
E       AssertionError: '4fe29f10a3bc5582d41bbf066246302d79c31110e30bfb4df063068c44231c98' != 'f04f7b3f0f1a8ce0550e265c4d563a6da0a53260d1baab8eacbfb56af6068352'
E       AssertionError: 'b1120f7bfd2f436baa17d7365bac8e1e8287526f87d567056a5914e5f0b3229c' != '3087d1b9867361fbb9beac7a116ca60d15964be575943e0258481656b4f721ac'
E       AssertionError: '21d0e29e9c9aca68650c7cdc0a916da3540d11dffb55306f1e1318e3721b1b8e' != 'cb5285127b5926a81fcd28d2e27997ce5fc556d6c1a4a797f5d24a0060821b7c'
3 failed, 1592 passed, 4 warnings in 22.97s
3 failed, 1613 passed, 4 warnings in 22.29s
```

导入纯净性使用已有强门禁，在 python -S 子进程中阻止 torch、torch_npu、triton、acl 的任何导入尝试，并阻止 shutil.which/subprocess.Popen；也遍历 79 条声明和选型。未改变 T8 注册语义。

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest -q -s tests/test_torch_library.py::test_import_is_pure_and_registration_is_opt_in tests/test_jit_trigger_layer.py::test_import_selection_and_declarations_with_no_optional_dependencies_or_compiler
```

```text
{"provider_import_attempts": [], "compiler_probes": [], "registration_enabled": false}
.{"records": 79, "optional_import_attempts": [], "compiler_probe_attempts": [], "compiler_paths": {"bisheng": null, "ccec": null, "ascendc": null}, "selected_gemma": "triton/norm/gemma_fused_add_rmsnorm_v1"}
.
2 passed in 0.27s
```

还实际执行了模块导入门禁和账本校验：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python tools/check_imports.py
/root/miniforge3/envs/flashinfer-a5/bin/python tools/validate_manifests.py
```

check_imports 的结构化输出为 `{"modules": 129, "failing": 0, "failures": []}`。它主动遍历所有模块，输出过已有的 Triton 弃用警告，不能与上面的顶层导入纯净性门禁混同。两份账本的完整校验输出如下：

```text
FlashInfer API parity
  upstream: flashinfer-ai/flashinfer (2895577720741ddfedff118202c89584996b8875)
  scope: all
  inventory: bootstrap
  symbols: 304
  upstream-symbols: 256
  coverage-matrix: missing_source
  intentionally-different: 142
  local-extensions: 47
  pending-gaps: 130
  optimized: 0
  functional: 70
  reference: 10
  framework: 44
  missing:   180
  release-compatible: no
  note: original 147-row coverage matrix is unavailable; CANN/consumer coverage remains unverified where evidence is absent.
FlashInfer API parity
  upstream: flashinfer-ai/flashinfer (2895577720741ddfedff118202c89584996b8875)
  scope: attention_core
  inventory: bootstrap
  symbols: 36
  upstream-symbols: 31
  coverage-matrix: missing_source
  intentionally-different: 24
  local-extensions: 5
  pending-gaps: 9
  optimized: 0
  functional: 22
  reference: 10
  framework: 4
  missing:   0
  attention-surfaces: 6
  host-reference-surfaces: 6
  provider-routed-surfaces: 6
  npu-callable-surfaces: 5
  release-compatible: no
  note: original 147-row coverage matrix is unavailable; CANN/consumer coverage remains unverified where evidence is absent.
Validated kernel descriptors: 0
Validated Attention capability profiles: 0
Validated Attention kernel bindings: 0
```

79 条 auto 选型对照在修改前保存快照、修改后逐条比较；工具同时比较 stack、descriptor 指纹、signature、hardware、selected、fallback、hot_path_selected 和逐候选 filters。

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t19_selection_audit.py --write /tmp/t9-selection-before.json
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t19_selection_audit.py --compare /tmp/t9-selection-before.json
```

基线快照 SHA256：`0c39bb1c5c6f2461ab917708424a9233f8325ac23e971e4dfb75befba2b593ea`。

```text
records: 79
stack: cann records: 40
stack: reference records: 38
stack: triton records: 1
multi_config_op_source_groups: 0 []
baseline: /tmp/t9-selection-before.json
```

以下为工具输出的完整 79 行 before/after 对照：

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

已知相邻问题保留：

- T10 最大 workspace 查询仍未接入 native CANN；T11 动态长度/tiling 重放更新尚未实现。当前 native run 仍有按请求索引收集和形状相关的 FIA 组合逻辑；本次只证明 plan 契约，没有捕获或重放验证，没有将 kernel 声明改成 CAPTURE_SAFE。
- 外部 provider 的图资源绑定仍显式报 NotImplementedError；本次接通内置 CANN，保留 Host ReferenceTensor 的结构契约。
- 一次性 cudnn_batch_decode_with_kv_cache 的 workspace_buffer/is_cuda_graph_compatible 仍是兼容参数；两处说明已修正，不以该布尔值承诺可捕获。
- 三个既有 ABI 指纹失败原样保留；现有内部格式/弃用警告未处理。
- 账本原有 use_tensor_cores 条目套用了图模式说明、缺失的 cascade wrapper 条目也有旧图进度注释，本轮保留这些相邻库存问题。

T6/T7 的过滤排序、T19 两级结构、T20 tuning 判据、T8 注册代码均未修改。执行 `git diff --check` 和 `git diff --exit-code -- docs/tasks/ docs/TASKS.md` 均无输出并成功。所有改动留在工作区，没有 git commit/push。
