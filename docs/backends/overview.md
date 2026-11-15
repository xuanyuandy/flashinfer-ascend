# 后端与算子支持

面向终端用户和后端开发者：本页用于按算子、dtype、SoC 和测试覆盖选择已有实现。支持标签来自源码登记与测试定义，不能当作当前机器的运行验收结果。

## 支持标签

- **Supported**：目标后端有注册、可调用 runner 和对应数值测试定义。
- **Partial**：有实现，但存在显著 envelope/组合限制，或测试没有覆盖完整登记域；具体限制见表。
- **Not supported**：没有该公共语义对应的内建注册或实现。

`dtype` 列是 Descriptor 可接收的 Tensor 存储类型集合，包含索引、scale 和辅助输出类型；它不代表任意 Tensor 可以混用这些类型。完整组合由 CapabilityCase、NativeBinding 和 runner 校验。`none` 表示可选输入缺省，不是数值 dtype。

## 后端摘要

| backend | 全局 Descriptor | 执行方式 | 主要边界 |
| --- | ---: | --- | --- |
| `cann` | 41 | 调用已安装的 Torch/TorchNPU/CANN 算子或组合 | 各算子单独限制 SoC、dtype 和布局 |
| `ascendc` | 5 | 本仓 v2 op-package，ACLNN + TorchNPU 桥 | Ascend950 三项、Ascend910B 两项；各自固定支持域 |
| `triton` | 2 | 后端目录内的 Triton kernel，首次需要时 JIT | Matmul 与 Gemma fused Add RMSNorm |
| `reference` | 38 | 显式参考计算 | 不作为默认 auto 的运行失败兜底 |

`custom`、`external` 是运行时扩展入口，当前没有内建数学算子记录；`aclnn`、`ascendc_aot`、`ascendc_jit` 是代码接受的后端枚举，当前全局目录没有对应记录。枚举存在不等于后端已部署。

## 算子族摘要


| 算子族 | CANN | Ascend C | Triton | reference |
| --- | --- | --- | --- | --- |
| Attention | Partial (6) | Partial (2) | Not supported | Partial (3) |
| GEMM | Partial (6) | Partial (3) | Partial (1) | Partial (5) |
| MoE | Partial (3) | Not supported | Not supported | Partial (3) |
| Norm | Supported (4) | Not supported | Supported (1) | Partial (5) |
| Activation | Supported (3) | Not supported | Not supported | Partial (3) |
| RoPE | Supported (6) | Not supported | Not supported | Partial (6) |
| Sampling | Supported (10) | Not supported | Not supported | Partial (10) |
| KV cache | Partial (1) | Not supported | Not supported | Partial (1) |
| State merge | Supported (2) | Not supported | Not supported | Partial (2) |

族级标签覆盖该族的现有登记项，不能外推为整个上游算子族。以下逐项表列出全部 86 个 Descriptor 对应的 op × backend。

## 算子登记矩阵

尺寸区间为包含端点；`/ n` 表示维度必须是 n 的倍数。SoC 列保留代码声明值；可路由值以硬件归一化和精确匹配为准。

| 逻辑算子 | backend | 存储 dtype | 声明 SoC | 状态与约束 | 测试定义 |
| --- | --- | --- | --- | --- | --- |
| <a id="op-activation-gelu-and-mul"></a>`activation.gelu_and_mul` | `cann` | bfloat16, float16, float32 | Ascend950 | **Supported** — logical=ND | [test_activation.py](../../tests/test_activation.py) |
| `activation.gelu_and_mul` | `reference` | bfloat16, float16, float32 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — logical=ND；显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_activation.py](../../tests/test_activation.py) |
| <a id="op-activation-gelu-tanh-and-mul"></a>`activation.gelu_tanh_and_mul` | `cann` | bfloat16, float16, float32 | Ascend950 | **Supported** — logical=ND | [test_activation.py](../../tests/test_activation.py) |
| `activation.gelu_tanh_and_mul` | `reference` | bfloat16, float16, float32 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — logical=ND；显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_activation.py](../../tests/test_activation.py) |
| <a id="op-activation-silu-and-mul"></a>`activation.silu_and_mul` | `cann` | bfloat16, float16, float32 | Ascend950 | **Supported** — logical=ND | [test_activation.py](../../tests/test_activation.py) |
| `activation.silu_and_mul` | `reference` | bfloat16, float16, float32 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — logical=ND；显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_activation.py](../../tests/test_activation.py) |
| <a id="op-ascendc-single-prefill-catlass"></a>`ascendc.single_prefill_catlass` | `ascendc` | bfloat16, float16 | Ascend950 | **Partial** — S=128..512 / 128; T=128..512 / 128; H=1..8 / 1; D=128..128 / 128；physical=ND；公共 Tensor 契约准入；原生 dtype 组合数=2；missing 后按策略 JIT | [test_flash_attention_real.py](../../tests/test_flash_attention_real.py)、[test_triton_attention.py](../../tests/test_triton_attention.py) |
| <a id="op-attention-batch-decode-paged"></a>`attention.batch_decode_paged` | `cann` | bfloat16, float16 | Ascend950 | **Partial** — logical=NHD,HND；head_dim 为 64 或 128 | [test_attention_cann.py](../../tests/test_attention_cann.py)、[test_attention_reference.py](../../tests/test_attention_reference.py) |
| <a id="op-attention-batch-prefill-paged"></a>`attention.batch_prefill_paged` | `cann` | bfloat16, float16 | Ascend950 | **Partial** — logical=NHD,HND；head_dim 为 64 或 128 | [test_attention_cann.py](../../tests/test_attention_cann.py)、[test_attention_reference.py](../../tests/test_attention_reference.py) |
| <a id="op-attention-batch-prefill-ragged"></a>`attention.batch_prefill_ragged` | `cann` | bfloat16, float16 | Ascend950 | **Partial** — logical=NHD,HND；head_dim 为 64 或 128 | [test_attention_cann.py](../../tests/test_attention_cann.py)、[test_attention_reference.py](../../tests/test_attention_reference.py) |
| <a id="op-attention-decode"></a>`attention.decode` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_attention_cann.py](../../tests/test_attention_cann.py)、[test_attention_reference.py](../../tests/test_attention_reference.py) |
| <a id="op-attention-prefill"></a>`attention.prefill` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_attention_cann.py](../../tests/test_attention_cann.py)、[test_attention_reference.py](../../tests/test_attention_reference.py) |
| <a id="op-attention-single-decode"></a>`attention.single_decode` | `cann` | bfloat16, float16 | Ascend950 | **Partial** — logical=NHD,HND；head_dim 为 64 或 128 | [test_attention_cann.py](../../tests/test_attention_cann.py)、[test_attention_reference.py](../../tests/test_attention_reference.py) |
| <a id="op-attention-single-prefill"></a>`attention.single_prefill` | `cann` | bfloat16, float16 | Ascend950 | **Partial** — logical=NHD,HND；head_dim 为 64 或 128 | [test_attention_cann.py](../../tests/test_attention_cann.py)、[test_attention_reference.py](../../tests/test_attention_reference.py) |
| <a id="op-attention-sparse-flash-attention"></a>`attention.sparse_flash_attention` | `ascendc` | bfloat16, float16, float32, int32 | Ascend910B | **Partial** — 2 组完整能力组合；MLA D=512; KV head=1; 连续 ND；BSND/TND/分页；AOT-only；原生 dtype 组合数=2 | [test_sparse_flash_attention.py](../../tests/test_sparse_flash_attention.py)、[test_operator_packages_real.py](../../tests/test_operator_packages_real.py) |
| <a id="op-cascade-merge-state"></a>`cascade.merge_state` | `cann` | bfloat16, float16, float32 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_cascade.py](../../tests/test_cascade.py)、[test_page_and_states.py](../../tests/test_page_and_states.py) |
| `cascade.merge_state` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_cascade.py](../../tests/test_cascade.py)、[test_page_and_states.py](../../tests/test_page_and_states.py) |
| <a id="op-cascade-merge-states"></a>`cascade.merge_states` | `cann` | bfloat16, float16, float32 | Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_cascade.py](../../tests/test_cascade.py)、[test_page_and_states.py](../../tests/test_page_and_states.py) |
| `cascade.merge_states` | `reference` | bfloat16, float16, float32 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_cascade.py](../../tests/test_cascade.py)、[test_page_and_states.py](../../tests/test_page_and_states.py) |
| <a id="op-decode-cudnn-batch-decode-with-kv-cache"></a>`decode.cudnn_batch_decode_with_kv_cache` | `cann` | bfloat16, float16, int32, int64 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_cudnn_decode.py](../../tests/test_cudnn_decode.py) |
| `decode.cudnn_batch_decode_with_kv_cache` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_cudnn_decode.py](../../tests/test_cudnn_decode.py) |
| <a id="op-fused-moe-cutlass-fused-moe"></a>`fused_moe.cutlass_fused_moe` | `cann` | bfloat16, float16, float32, int32, int64 | Ascend910B, Ascend910_93, Ascend950 | **Partial** — Ascend910_93 与硬件规范化值不匹配，不计入 A3 可路由范围 | [test_fused_moe.py](../../tests/test_fused_moe.py) |
| `fused_moe.cutlass_fused_moe` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_fused_moe.py](../../tests/test_fused_moe.py) |
| <a id="op-fused-moe-trtllm-fp8-block-scale-moe"></a>`fused_moe.trtllm_fp8_block_scale_moe` | `cann` | bfloat16, float16, float32, float8_e4m3fn, int32, int64 | Ascend950 | **Partial** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_fused_moe.py](../../tests/test_fused_moe.py) |
| `fused_moe.trtllm_fp8_block_scale_moe` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_fused_moe.py](../../tests/test_fused_moe.py) |
| <a id="op-fused-moe-trtllm-fp8-per-tensor-scale-moe"></a>`fused_moe.trtllm_fp8_per_tensor_scale_moe` | `cann` | bfloat16, float16, float32, float8_e4m3fn, int32, int64 | Ascend950 | **Partial** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_fused_moe.py](../../tests/test_fused_moe.py) |
| `fused_moe.trtllm_fp8_per_tensor_scale_moe` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_fused_moe.py](../../tests/test_fused_moe.py) |
| <a id="op-gemm-bmm-bf16"></a>`gemm.bmm_bf16` | `cann` | bfloat16, float16, float32 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_gemm_segment.py](../../tests/test_gemm_segment.py)、[test_gemm_fp8.py](../../tests/test_gemm_fp8.py) |
| `gemm.bmm_bf16` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_gemm_segment.py](../../tests/test_gemm_segment.py)、[test_gemm_fp8.py](../../tests/test_gemm_fp8.py) |
| <a id="op-gemm-bmm-fp8"></a>`gemm.bmm_fp8` | `cann` | float32, float8_e4m3fn, float8_e5m2 | Ascend950 | **Partial** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_gemm_segment.py](../../tests/test_gemm_segment.py)、[test_gemm_fp8.py](../../tests/test_gemm_fp8.py) |
| `gemm.bmm_fp8` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_gemm_segment.py](../../tests/test_gemm_segment.py)、[test_gemm_fp8.py](../../tests/test_gemm_fp8.py) |
| <a id="op-gemm-gemm-fp8-nt-groupwise"></a>`gemm.gemm_fp8_nt_groupwise` | `cann` | float32, float8_e4m3fn, float8_e5m2 | Ascend950 | **Partial** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_gemm_segment.py](../../tests/test_gemm_segment.py)、[test_gemm_fp8.py](../../tests/test_gemm_fp8.py) |
| `gemm.gemm_fp8_nt_groupwise` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_gemm_segment.py](../../tests/test_gemm_segment.py)、[test_gemm_fp8.py](../../tests/test_gemm_fp8.py) |
| <a id="op-gemm-grouped-matmul-swiglu-quant"></a>`gemm.grouped_matmul_swiglu_quant` | `ascendc` | bfloat16, float16, float32, int32, int64, int8 | Ascend910B | **Partial** — 7 组完整能力组合；A8W8/A8W4 carrier/scale 组合；ND/NZ 与对齐按 binding；AOT-only；原生 dtype 组合数=4 | [test_grouped_matmul_swiglu_quant.py](../../tests/test_grouped_matmul_swiglu_quant.py)、[test_operator_packages_real.py](../../tests/test_operator_packages_real.py) |
| <a id="op-gemm-matmul"></a>`gemm.matmul` | `ascendc` | bfloat16 | Ascend950 | **Partial** — M=16..256 / 16; N=16..256 / 16; K=16..256 / 16；physical=ND；公共 Tensor 契约准入 | [test_matmul.py](../../tests/test_matmul.py)、[test_matmul_real.py](../../tests/test_matmul_real.py) |
| `gemm.matmul` | `cann` | bfloat16 | Ascend950 | **Partial** — M=16..256 / 16; K=16..256 / 16; N=16..256 / 16；physical=ND；公共 Tensor 契约准入 | [test_matmul.py](../../tests/test_matmul.py)、[test_matmul_real.py](../../tests/test_matmul_real.py) |
| `gemm.matmul` | `triton` | bfloat16 | Ascend950 | **Partial** — M=16..256 / 16; K=16..256 / 16; N=16..256 / 16；physical=ND；公共 Tensor 契约准入 | [test_matmul.py](../../tests/test_matmul.py)、[test_matmul_real.py](../../tests/test_matmul_real.py) |
| <a id="op-gemm-matmul-bias-silu"></a>`gemm.matmul_bias_silu` | `ascendc` | bfloat16 | Ascend950 | **Partial** — physical=ND；公共 Tensor 契约准入；M/N/K 为 16..256 的 16 倍数；missing 后按策略 JIT | [test_matmul_bias_silu_registration.py](../../tests/test_matmul_bias_silu_registration.py)、[test_matmul_bias_silu_real.py](../../tests/test_matmul_bias_silu_real.py) |
| <a id="op-gemm-mm-bf16"></a>`gemm.mm_bf16` | `cann` | bfloat16, float16, float32 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_gemm_segment.py](../../tests/test_gemm_segment.py)、[test_gemm_fp8.py](../../tests/test_gemm_fp8.py) |
| `gemm.mm_bf16` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_gemm_segment.py](../../tests/test_gemm_segment.py)、[test_gemm_fp8.py](../../tests/test_gemm_fp8.py) |
| <a id="op-gemm-segment-gemm"></a>`gemm.segment_gemm` | `cann` | bfloat16, float16, float32, int32, int64 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_gemm_segment.py](../../tests/test_gemm_segment.py)、[test_gemm_fp8.py](../../tests/test_gemm_fp8.py) |
| `gemm.segment_gemm` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_gemm_segment.py](../../tests/test_gemm_segment.py)、[test_gemm_fp8.py](../../tests/test_gemm_fp8.py) |
| <a id="op-norm-fused-add-rmsnorm"></a>`norm.fused_add_rmsnorm` | `cann` | bfloat16, float16, float32 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_norm.py](../../tests/test_norm.py) |
| `norm.fused_add_rmsnorm` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_norm.py](../../tests/test_norm.py) |
| <a id="op-norm-gemma-fused-add-rmsnorm"></a>`norm.gemma_fused_add_rmsnorm` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_norm.py](../../tests/test_norm.py) |
| `norm.gemma_fused_add_rmsnorm` | `triton` | bfloat16, float16, float32 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_norm.py](../../tests/test_norm.py) |
| <a id="op-norm-gemma-rmsnorm"></a>`norm.gemma_rmsnorm` | `cann` | bfloat16, float16, float32 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_norm.py](../../tests/test_norm.py) |
| `norm.gemma_rmsnorm` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_norm.py](../../tests/test_norm.py) |
| <a id="op-norm-layernorm"></a>`norm.layernorm` | `cann` | bfloat16, float16, float32 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_norm.py](../../tests/test_norm.py) |
| `norm.layernorm` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_norm.py](../../tests/test_norm.py) |
| <a id="op-norm-rmsnorm"></a>`norm.rmsnorm` | `cann` | bfloat16, float16, float32 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_norm.py](../../tests/test_norm.py) |
| `norm.rmsnorm` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_norm.py](../../tests/test_norm.py) |
| <a id="op-page-append-paged-kv-cache"></a>`page.append_paged_kv_cache` | `cann` | bfloat16, float16, float32, int32, int64 | Ascend950 | **Partial** — logical=NHD,HND | [test_page_and_states.py](../../tests/test_page_and_states.py) |
| `page.append_paged_kv_cache` | `reference` | bfloat16, float16, float32, int32, int64 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — logical=NHD,HND；显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_page_and_states.py](../../tests/test_page_and_states.py) |
| <a id="op-rope-apply-rope"></a>`rope.apply_rope` | `cann` | bfloat16, float16, float32, int32, int64 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_rope.py](../../tests/test_rope.py)、[test_rope_pos_ids.py](../../tests/test_rope_pos_ids.py) |
| `rope.apply_rope` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_rope.py](../../tests/test_rope.py)、[test_rope_pos_ids.py](../../tests/test_rope_pos_ids.py) |
| <a id="op-rope-apply-rope-inplace"></a>`rope.apply_rope_inplace` | `cann` | bfloat16, float16, float32, int32, int64 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_rope.py](../../tests/test_rope.py)、[test_rope_pos_ids.py](../../tests/test_rope_pos_ids.py) |
| `rope.apply_rope_inplace` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_rope.py](../../tests/test_rope.py)、[test_rope_pos_ids.py](../../tests/test_rope_pos_ids.py) |
| <a id="op-rope-apply-rope-pos-ids"></a>`rope.apply_rope_pos_ids` | `cann` | bfloat16, float16, float32, int32, int64 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_rope.py](../../tests/test_rope.py)、[test_rope_pos_ids.py](../../tests/test_rope_pos_ids.py) |
| `rope.apply_rope_pos_ids` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_rope.py](../../tests/test_rope.py)、[test_rope_pos_ids.py](../../tests/test_rope_pos_ids.py) |
| <a id="op-rope-apply-rope-pos-ids-inplace"></a>`rope.apply_rope_pos_ids_inplace` | `cann` | bfloat16, float16, float32, int32, int64 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_rope.py](../../tests/test_rope.py)、[test_rope_pos_ids.py](../../tests/test_rope_pos_ids.py) |
| `rope.apply_rope_pos_ids_inplace` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_rope.py](../../tests/test_rope.py)、[test_rope_pos_ids.py](../../tests/test_rope_pos_ids.py) |
| <a id="op-rope-apply-rope-with-cos-sin-cache"></a>`rope.apply_rope_with_cos_sin_cache` | `cann` | bfloat16, float16, float32, int32, int64 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_rope.py](../../tests/test_rope.py)、[test_rope_pos_ids.py](../../tests/test_rope_pos_ids.py) |
| `rope.apply_rope_with_cos_sin_cache` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_rope.py](../../tests/test_rope.py)、[test_rope_pos_ids.py](../../tests/test_rope_pos_ids.py) |
| <a id="op-rope-apply-rope-with-cos-sin-cache-inplace"></a>`rope.apply_rope_with_cos_sin_cache_inplace` | `cann` | bfloat16, float16, float32, int32, int64 | Ascend910B, Ascend950 | **Supported** — 使用公共 Tensor 契约及 runner 校验；存在目标后端数值测试定义 | [test_rope.py](../../tests/test_rope.py)、[test_rope_pos_ids.py](../../tests/test_rope_pos_ids.py) |
| `rope.apply_rope_with_cos_sin_cache_inplace` | `reference` | 由完整组合约束 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — 显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_rope.py](../../tests/test_rope.py)、[test_rope_pos_ids.py](../../tests/test_rope_pos_ids.py) |
| <a id="op-sampling-chain-speculative-sampling"></a>`sampling.chain_speculative_sampling` | `cann` | bfloat16, float16, float32 | Ascend950 | **Supported** — logical=ND | [test_sampling.py](../../tests/test_sampling.py) |
| `sampling.chain_speculative_sampling` | `reference` | bfloat16, float16, float32 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — logical=ND；显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_sampling.py](../../tests/test_sampling.py) |
| <a id="op-sampling-min-p-sampling-from-probs"></a>`sampling.min_p_sampling_from_probs` | `cann` | bfloat16, float16, float32 | Ascend950 | **Supported** — logical=ND | [test_sampling.py](../../tests/test_sampling.py) |
| `sampling.min_p_sampling_from_probs` | `reference` | bfloat16, float16, float32 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — logical=ND；显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_sampling.py](../../tests/test_sampling.py) |
| <a id="op-sampling-sampling-from-logits"></a>`sampling.sampling_from_logits` | `cann` | bfloat16, float16, float32 | Ascend950 | **Supported** — logical=ND | [test_sampling.py](../../tests/test_sampling.py) |
| `sampling.sampling_from_logits` | `reference` | bfloat16, float16, float32 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — logical=ND；显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_sampling.py](../../tests/test_sampling.py) |
| <a id="op-sampling-sampling-from-probs"></a>`sampling.sampling_from_probs` | `cann` | bfloat16, float16, float32 | Ascend950 | **Supported** — logical=ND | [test_sampling.py](../../tests/test_sampling.py) |
| `sampling.sampling_from_probs` | `reference` | bfloat16, float16, float32 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — logical=ND；显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_sampling.py](../../tests/test_sampling.py) |
| <a id="op-sampling-top-k-mask-logits"></a>`sampling.top_k_mask_logits` | `cann` | bfloat16, float16, float32 | Ascend950 | **Supported** — logical=ND | [test_sampling.py](../../tests/test_sampling.py) |
| `sampling.top_k_mask_logits` | `reference` | bfloat16, float16, float32 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — logical=ND；显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_sampling.py](../../tests/test_sampling.py) |
| <a id="op-sampling-top-k-renorm-probs"></a>`sampling.top_k_renorm_probs` | `cann` | bfloat16, float16, float32 | Ascend950 | **Supported** — logical=ND | [test_sampling.py](../../tests/test_sampling.py) |
| `sampling.top_k_renorm_probs` | `reference` | bfloat16, float16, float32 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — logical=ND；显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_sampling.py](../../tests/test_sampling.py) |
| <a id="op-sampling-top-k-sampling-from-probs"></a>`sampling.top_k_sampling_from_probs` | `cann` | bfloat16, float16, float32 | Ascend950 | **Supported** — logical=ND | [test_sampling.py](../../tests/test_sampling.py) |
| `sampling.top_k_sampling_from_probs` | `reference` | bfloat16, float16, float32 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — logical=ND；显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_sampling.py](../../tests/test_sampling.py) |
| <a id="op-sampling-top-k-top-p-sampling-from-probs"></a>`sampling.top_k_top_p_sampling_from_probs` | `cann` | bfloat16, float16, float32 | Ascend950 | **Supported** — logical=ND | [test_sampling.py](../../tests/test_sampling.py) |
| `sampling.top_k_top_p_sampling_from_probs` | `reference` | bfloat16, float16, float32 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — logical=ND；显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_sampling.py](../../tests/test_sampling.py) |
| <a id="op-sampling-top-p-renorm-probs"></a>`sampling.top_p_renorm_probs` | `cann` | bfloat16, float16, float32 | Ascend950 | **Supported** — logical=ND | [test_sampling.py](../../tests/test_sampling.py) |
| `sampling.top_p_renorm_probs` | `reference` | bfloat16, float16, float32 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — logical=ND；显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_sampling.py](../../tests/test_sampling.py) |
| <a id="op-sampling-top-p-sampling-from-probs"></a>`sampling.top_p_sampling_from_probs` | `cann` | bfloat16, float16, float32 | Ascend950 | **Supported** — logical=ND | [test_sampling.py](../../tests/test_sampling.py) |
| `sampling.top_p_sampling_from_probs` | `reference` | bfloat16, float16, float32 | 未按 SoC 限制（仍受执行实现约束） | **Partial** — logical=ND；显式参考；Host/数值对照测试不覆盖所有登记组合 | [test_sampling.py](../../tests/test_sampling.py) |

上述限制来自每个后端的 `implementation.py`、Support/NativeBinding 与 runner，而非旧文档计数。入口：[discover_implementations](../../flashinfer/operators/authoring.py#L53)、[get_global_registry](../../flashinfer/runtime/registry.py#L290)；独立精确预期为 [tests/production_inventory.py](../../tests/production_inventory.py)。测试定义存在不表示执行过当前 dtype × shape × 设备矩阵。

## 图与实验实现

图能力枚举中，`cann/attention/single_decode_fia_v1` 为 `needs_param_update`，其余 85 条为 `not_capturable`。类名中的 `CUDAGraph`、固定 buffer 或 FakeTensor 不增加这个范围。5 条生产 Ascend C 均为 `not_capturable`。

实验 Triton FA 的作者记录采用独立 scope；它不进入默认 86 项。其支持域与默认 CATLASS FA 的同名公共入口仍须分别核对。代码：[IMPLEMENTATION](../../flashinfer/triton/attention/single_prefill_catlass/implementation.py)、[tests/test_single_decode_graph_real.py](../../tests/test_single_decode_graph_real.py)。

## 安装与构建边界

包的声明版本为 `0.1.0.dev0`、Python `>=3.9`。`environment-a5.yml` 指定 Python 3.11、Torch `2.10.0+cpu`、torch-npu `2.10.0.post7`、triton-ascend `3.6.0`；它是 A5 环境模板，不构成其他产品或版本组合的验证。

普通 wheel 构建默认全量 AOT。`FLASHINFER_AOT_ALL=1`、`FLASHINFER_AOT_CONFIG`、`FLASHINFER_AOT_PREBUILT` 互斥；`FLASHINFER_AOT_ALL=0` 是源码模式。源码模式不提供 AOT-only 算子所需二进制。PREBUILT 仍校验产物和桥的身份。

构建依赖包括匹配的 CANN/Torch/TorchNPU 头文件与库、编译器及 libffi；CATLASS 声明钉定 revision，头树由构建方提供。安装后执行仍需要相应运行栈，框架不隐式安装依赖。代码：[aot_mode](../../setup.py#L20)、[BuildPy](../../setup.py#L90)、[pyproject.toml](../../pyproject.toml)。

相关说明：[文档导航](../README.md)、[规划与差距](../roadmap.md)。
