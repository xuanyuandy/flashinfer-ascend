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


def metadata_reason(a, b, out):
    if len(a.shape) != 2 or len(b.shape) != 2 or len(out.shape) != 2:
        return "Matmul requires rank-2 tensors"
    if a.shape[1] != b.shape[0] or tuple(out.shape) != (a.shape[0], b.shape[1]):
        return "Matmul input/output dimensions are inconsistent"
    if any(int(v) <= 0 for t in (a, b, out) for v in t.shape):
        return "Matmul dimensions must be positive"
    if any(t.dtype != a.dtype or t.device != a.device for t in (b, out)):
        return "Matmul tensors must share dtype and device"
    return "supported"


def overlap_parameters(count=3):
    return tuple(SimpleNamespace(role="output" if i == count - 1 else "input") for i in range(count))


def validate_tensors(a, b, out):
    """Shared execution semantics, independent of any frozen AOT recipe."""
    import torch
    import torch_npu
    from .jit.tensor_adapter import _reject_output_overlap

    reason = metadata_reason(a, b, out)
    if reason != "supported":
        raise ValueError(reason)
    if a.device.type != torch._C._get_privateuse1_backend_name():
        raise ValueError("Matmul requires NPU tensors")
    if any(not t.is_contiguous() or torch_npu.get_npu_format(t) != 2 for t in (a, b, out)):
        raise ValueError("Matmul requires contiguous ND tensors")
    _reject_output_overlap(overlap_parameters(), (a, b, out))
