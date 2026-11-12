"""Ascend C backend entry; operator implementations are grouped by interface domain."""
from ._catalog import _contracts
from ._runtime import _dependency_roots, _jit_backend, tensor_contract_specialization
from .gemm.matmul_bias_silu.implementation import KERNEL_ID as MATMUL_BIAS_SILU_KERNEL_ID
from .attention.single_prefill_catlass.implementation import KERNEL_ID as FLASH_ATTENTION_KERNEL_ID


def load_contract(kernel_id=MATMUL_BIAS_SILU_KERNEL_ID):
    from ._catalog import load_contract as read
    return read(kernel_id)


def ascendc_descriptors():
    from flashinfer.operators.implementation import implementations
    from ._runtime import bind_kernel
    declared = {d.kernel_id: d for d in implementations(backend="ascendc")}
    # Explicit recipe/manifest clients remain supported alongside directory authoring.
    return tuple(declared[kernel_id].bind() if kernel_id in declared else bind_kernel(kernel_id)
                 for kernel_id in _contracts())


__all__ = ["MATMUL_BIAS_SILU_KERNEL_ID", "FLASH_ATTENTION_KERNEL_ID", "load_contract", "ascendc_descriptors"]
