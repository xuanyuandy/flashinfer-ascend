# 本仓逻辑 op 清单：原静态审计快照

此表保留原基线的 45 op / 81 Descriptor；当前登记及类型化声明入口见
[当前实现说明](../../operator-coverage.md#本仓当前实现全集)，不要据此遗漏后续 Matmul、GMM 或 SFA。

由原审计时的运行时登记表生成，按 op 而非公开别名或 descriptor 计数。
`npu-runner-bound-limited` 表示存在受限 NPU runner，不表示所有上游参数、产品或图模式已验证。
完整 dtype/shape/SoC 能力字段与 runner 来源见同目录 CSV/JSON。

| op | 顶层公共入口 | backend | 状态 |
| --- | --- | --- | --- |
| `activation.gelu_and_mul` | `flashinfer.gelu_and_mul` | cann;reference | npu-runner-bound-limited |
| `activation.gelu_tanh_and_mul` | `flashinfer.gelu_tanh_and_mul` | cann;reference | npu-runner-bound-limited |
| `activation.silu_and_mul` | `flashinfer.silu_and_mul` | cann;reference | npu-runner-bound-limited |
| `ascendc.single_prefill_catlass` | `flashinfer.single_prefill_with_kv_cache_catlass` | ascendc | npu-runner-bound-limited |
| `attention.batch_decode_paged` | `flashinfer.BatchDecodeWithPagedKVCacheWrapper`<br/>`flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper` | cann | npu-runner-bound-limited |
| `attention.batch_prefill_paged` | `flashinfer.BatchPrefillWithPagedKVCacheWrapper` | cann | npu-runner-bound-limited |
| `attention.batch_prefill_ragged` | `flashinfer.BatchPrefillWithRaggedKVCacheWrapper` | cann | npu-runner-bound-limited |
| `attention.decode` | 内部 reference op | reference | reference-only |
| `attention.prefill` | 内部 reference op | reference | reference-only |
| `attention.single_decode` | `flashinfer.single_decode_with_kv_cache` | cann | npu-runner-bound-limited |
| `attention.single_prefill` | `flashinfer.single_prefill_with_kv_cache` | cann | npu-runner-bound-limited |
| `cascade.merge_state` | `flashinfer.merge_state` | cann;reference | npu-runner-bound-limited |
| `cascade.merge_states` | `flashinfer.merge_states` | cann;reference | npu-runner-bound-limited |
| `decode.cudnn_batch_decode_with_kv_cache` | `flashinfer.cudnn_batch_decode_with_kv_cache` | cann;reference | npu-runner-bound-limited |
| `fused_moe.cutlass_fused_moe` | `flashinfer.fused_moe` | cann;reference | npu-runner-bound-limited |
| `fused_moe.trtllm_fp8_block_scale_moe` | `flashinfer.fp8_block_scale_moe` | cann;reference | npu-runner-bound-limited |
| `fused_moe.trtllm_fp8_per_tensor_scale_moe` | `flashinfer.fp8_per_tensor_scale_moe` | cann;reference | npu-runner-bound-limited |
| `gemm.bmm_bf16` | `flashinfer.bmm_bf16` | cann;reference | npu-runner-bound-limited |
| `gemm.bmm_fp8` | `flashinfer.bmm_fp8` | cann;reference | npu-runner-bound-limited |
| `gemm.gemm_fp8_nt_groupwise` | `flashinfer.gemm_fp8_nt_groupwise` | cann;reference | npu-runner-bound-limited |
| `gemm.matmul_bias_silu` | `flashinfer.matmul_bias_silu` | ascendc | npu-runner-bound-limited |
| `gemm.mm_bf16` | `flashinfer.mm_bf16` | cann;reference | npu-runner-bound-limited |
| `gemm.segment_gemm` | `flashinfer.SegmentGEMMWrapper` | cann;reference | npu-runner-bound-limited |
| `norm.fused_add_rmsnorm` | `flashinfer.fused_add_rmsnorm` | cann;reference | npu-runner-bound-limited |
| `norm.gemma_fused_add_rmsnorm` | `flashinfer.gemma_fused_add_rmsnorm` | reference;triton | npu-runner-bound-limited |
| `norm.gemma_rmsnorm` | `flashinfer.gemma_rmsnorm` | cann;reference | npu-runner-bound-limited |
| `norm.layernorm` | `flashinfer.layernorm` | cann;reference | npu-runner-bound-limited |
| `norm.rmsnorm` | `flashinfer.rmsnorm` | cann;reference | npu-runner-bound-limited |
| `page.append_paged_kv_cache` | `flashinfer.append_paged_kv_cache` | cann;reference | npu-runner-bound-limited |
| `rope.apply_rope` | `flashinfer.apply_rope` | cann;reference | npu-runner-bound-limited |
| `rope.apply_rope_inplace` | `flashinfer.apply_rope_inplace` | cann;reference | npu-runner-bound-limited |
| `rope.apply_rope_pos_ids` | `flashinfer.apply_rope_pos_ids` | cann;reference | npu-runner-bound-limited |
| `rope.apply_rope_pos_ids_inplace` | `flashinfer.apply_rope_pos_ids_inplace` | cann;reference | npu-runner-bound-limited |
| `rope.apply_rope_with_cos_sin_cache` | `flashinfer.apply_rope_with_cos_sin_cache` | cann;reference | npu-runner-bound-limited |
| `rope.apply_rope_with_cos_sin_cache_inplace` | `flashinfer.apply_rope_with_cos_sin_cache_inplace` | cann;reference | npu-runner-bound-limited |
| `sampling.chain_speculative_sampling` | `flashinfer.chain_speculative_sampling` | cann;reference | npu-runner-bound-limited |
| `sampling.min_p_sampling_from_probs` | `flashinfer.min_p_sampling_from_probs` | cann;reference | npu-runner-bound-limited |
| `sampling.sampling_from_logits` | `flashinfer.sampling_from_logits` | cann;reference | npu-runner-bound-limited |
| `sampling.sampling_from_probs` | `flashinfer.sampling_from_probs` | cann;reference | npu-runner-bound-limited |
| `sampling.top_k_mask_logits` | `flashinfer.top_k_mask_logits` | cann;reference | npu-runner-bound-limited |
| `sampling.top_k_renorm_probs` | `flashinfer.top_k_renorm_probs` | cann;reference | npu-runner-bound-limited |
| `sampling.top_k_sampling_from_probs` | `flashinfer.top_k_sampling_from_probs` | cann;reference | npu-runner-bound-limited |
| `sampling.top_k_top_p_sampling_from_probs` | `flashinfer.top_k_top_p_sampling_from_probs` | cann;reference | npu-runner-bound-limited |
| `sampling.top_p_renorm_probs` | `flashinfer.top_p_renorm_probs` | cann;reference | npu-runner-bound-limited |
| `sampling.top_p_sampling_from_probs` | `flashinfer.top_p_sampling_from_probs` | cann;reference | npu-runner-bound-limited |

`BatchAttention` 是另一个 Host/框架门面，未计入上述生产 NPU op；其默认 NPU provider 尚未接通。
同一个 op 的 Python 顶层/子模块、Torch functional/out 或 wrapper 生命周期方法不重复计为新 op。
