"""Common Matmul contract shared by every execution backend; import-safe."""

from types import SimpleNamespace


OP = "gemm.matmul"
AOT_KERNEL_ID = "ascendc/gemm/matmul/native/op_package/Ascend950"
CANN_KERNEL_ID = "cann/gemm/matmul_v1"
TRITON_KERNEL_ID = "triton/gemm/matmul_v1/Ascend950"


def contract():
    from .ascendc import load_contract

    return load_contract(AOT_KERNEL_ID)


def output_metadata(a, b, out=None):
    if out is not None:
        return out
    return SimpleNamespace(
        shape=(a.shape[0] if len(a.shape) else 0,
               b.shape[1] if len(b.shape) >= 2 else 0),
        dtype=a.dtype, device=a.device, is_contiguous=lambda: True,
    )


def validate_tensors(a, b, out):
    """Recheck at the execution boundary without casting or copying inputs."""
    import torch
    import torch_npu
    from .ascendc import tensor_contract_specialization
    from .jit.tensor_adapter import _reject_output_overlap

    tensors = (a, b, out)
    diagnostic = tensor_contract_specialization(AOT_KERNEL_ID, tensors)
    if diagnostic != "supported":
        raise ValueError(diagnostic)
    if a.device.type != torch._C._get_privateuse1_backend_name():
        raise ValueError("Matmul requires NPU tensors")
    if any(torch_npu.get_npu_format(t) != 2 for t in tensors):
        raise ValueError("Matmul requires contiguous ND tensors")
    _reject_output_overlap(contract()[1].tensor_parameters, tensors)
