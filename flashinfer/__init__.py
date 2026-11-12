"""FlashInfer-style inference primitives for Ascend NPU.

M0：框架骨架——对外 API 门面、输入/硬件签名、registry、dispatcher、
JIT cache key。kernel 产物（CANN 适配、Ascend C/Triton AOT/JIT）随
M1 起接入（实施路线见 docs/framework_design.md §16）。
"""

from .version import __version__
from .activation import silu_and_mul, gelu_and_mul, gelu_tanh_and_mul
from .attention import BatchAttention
from .attention.catlass_prefill import single_prefill_with_kv_cache_catlass
from .decode import (
    BatchDecodeWithPagedKVCacheWrapper,
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    cudnn_batch_decode_with_kv_cache,
    single_decode_with_kv_cache,
    single_decode_with_kv_cache_with_jit_module,
)
from .cascade import merge_state, merge_states
from .page import append_paged_kv_cache
from .gemm import SegmentGEMMWrapper, bmm_bf16, bmm_fp8, mm_bf16, gemm_fp8_nt_groupwise, matmul, matmul_bias_silu
from .fused_moe import (
    RoutingMethodType,
    fp8_block_scale_moe,
    fp8_per_tensor_scale_moe,
    fused_moe,
    reorder_rows_for_gated_act_gemm,
)
from .jit import build_cache_key, toolchain_fingerprint
from .norm import (
    fused_add_rmsnorm,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    layernorm,
    rmsnorm,
)
from .rope import (
    apply_rope,
    apply_rope_inplace,
    apply_rope_pos_ids,
    apply_rope_pos_ids_inplace,
    apply_rope_with_cos_sin_cache,
    apply_rope_with_cos_sin_cache_inplace,
)
from .prefill import (
    single_prefill_with_kv_cache, single_prefill_with_kv_cache_with_jit_module,
    BatchPrefillWithPagedKVCacheWrapper, BatchPrefillWithRaggedKVCacheWrapper,
)
from .sampling import (
    sampling_from_probs, sampling_from_logits, top_p_sampling_from_probs,
    top_k_sampling_from_probs, top_k_top_p_sampling_from_probs,
    min_p_sampling_from_probs, top_p_renorm_probs, top_k_renorm_probs,
    top_k_mask_logits, chain_speculative_sampling,
)
from .runtime import (
    Backend,
    CapabilitySpec,
    CompileMode,
    DispatchDecision,
    DispatchError,
    HardwareSignature,
    InputSignature,
    KernelDescriptor,
    KernelRegistry,
    Plan,
    build_signature,
    clear_dispatch_cache,
    current_hardware,
    dispatch,
    explain_dispatch,
    get_global_registry,
    register_custom_kernel,
    register_external_kernel,
)

__all__ = [
    "__version__",
    "BatchAttention",
    "silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul",
    # 对外 API（对标 flashinfer）
    "single_prefill_with_kv_cache",
    "single_prefill_with_kv_cache_catlass",
    "single_prefill_with_kv_cache_with_jit_module",
    "single_decode_with_kv_cache_with_jit_module",
    "sampling_from_probs", "sampling_from_logits", "top_p_sampling_from_probs",
    "top_k_sampling_from_probs", "top_k_top_p_sampling_from_probs",
    "min_p_sampling_from_probs", "top_p_renorm_probs", "top_k_renorm_probs",
    "top_k_mask_logits", "chain_speculative_sampling",
    "single_decode_with_kv_cache",
    "cudnn_batch_decode_with_kv_cache",
    "SegmentGEMMWrapper",
    "bmm_bf16",
    "matmul_bias_silu",
    "matmul",
    "bmm_fp8",
    "mm_bf16",
    "gemm_fp8_nt_groupwise",
    "RoutingMethodType",
    "reorder_rows_for_gated_act_gemm",
    "fused_moe",
    "fp8_per_tensor_scale_moe",
    "fp8_block_scale_moe",
    "merge_state",
    "merge_states", "append_paged_kv_cache",
    "BatchDecodeWithPagedKVCacheWrapper",
    "CUDAGraphBatchDecodeWithPagedKVCacheWrapper",
    "BatchPrefillWithPagedKVCacheWrapper",
    "BatchPrefillWithRaggedKVCacheWrapper",
    "rmsnorm",
    "fused_add_rmsnorm",
    "gemma_rmsnorm",
    "gemma_fused_add_rmsnorm",
    "layernorm",
    "apply_rope",
    "apply_rope_inplace",
    "apply_rope_pos_ids",
    "apply_rope_pos_ids_inplace",
    "apply_rope_with_cos_sin_cache",
    "apply_rope_with_cos_sin_cache_inplace",
    # runtime 契约
    "Backend",
    "CapabilitySpec",
    "CompileMode",
    "DispatchDecision",
    "DispatchError",
    "HardwareSignature",
    "InputSignature",
    "KernelDescriptor",
    "KernelRegistry",
    "Plan",
    # 调度与注册
    "build_signature",
    "clear_dispatch_cache",
    "current_hardware",
    "dispatch",
    "explain_dispatch",
    "get_global_registry",
    "register_custom_kernel",
    "register_external_kernel",
    # JIT
    "build_cache_key",
    "toolchain_fingerprint",
]


from .operators.catalog import install_exports as _install_operator_exports
_install_operator_exports(globals())
