"""Backend-independent declaration for fused_moe.trtllm_fp8_block_scale_moe."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='trtllm_fp8_block_scale_moe', domain='fused_moe', op='fused_moe.trtllm_fp8_block_scale_moe', public_entry='flashinfer.fused_moe:fp8_block_scale_moe')
