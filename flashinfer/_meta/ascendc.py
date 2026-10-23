"""Metadata-only contracts for the Ascend C probes (no runtime or allocation)."""


def matmul_probe_meta(a, b, out):
    """Validate the fixed T13 ABI; the caller already owns output metadata.

    Do not broaden this into a general GEMM shape rule: both T13 implementations
    and both output adapters support only contiguous FP32 16x16 matrices.
    Symbolic comparisons stay symbolic until torch's metadata guards resolve
    them. No tensor operation, data access, device query or allocation is needed.
    """
    import torch

    for tensor in (a, b, out):
        torch._check(tensor.dim() == 2, lambda: "matmul_probe requires rank 2")
        torch._check(tensor.shape[0] == 16, lambda: "matmul_probe requires 16x16")
        torch._check(tensor.shape[1] == 16, lambda: "matmul_probe requires 16x16")
        torch._check(tensor.dtype == torch.float32, lambda: "matmul_probe requires float32")
        torch._check(tensor.is_contiguous(), lambda: "matmul_probe requires contiguous ND tensors")
        torch._check(tensor.device == a.device, lambda: "matmul_probe requires one device")
    return None
