"""Backend-independent declaration for gemm.matmul_bias_silu."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='matmul_bias_silu', domain='gemm', op='gemm.matmul_bias_silu', public_entry='flashinfer.gemm:matmul_bias_silu')


from types import SimpleNamespace

def _matmul(a, b, bias, *, backend="auto", kernel_id=None):
    from flashinfer.gemm import _matmul_bias_silu_impl
    return _matmul_bias_silu_impl(a, b, bias, backend=backend, kernel_id=kernel_id)


def _matmul_out(a, b, bias, out, *, backend="auto", kernel_id=None):
    from flashinfer.gemm import _matmul_bias_silu_impl
    _matmul_bias_silu_impl(a, b, bias, out=out, backend=backend, kernel_id=kernel_id)


def _matmul_meta(a, b, bias, *, backend="auto", kernel_id=None):
    import torch

    shape = (a.shape[0] if a.ndim else 0, b.shape[1] if b.ndim >= 2 else 0)
    prospective = SimpleNamespace(shape=shape, dtype=torch.bfloat16, device=a.device,
                                  is_contiguous=lambda: True)
    _matmul_out_meta(a, b, bias, prospective, backend=backend, kernel_id=kernel_id)
    return torch.empty(shape, dtype=torch.bfloat16, device=a.device)


def _matmul_out_meta(a, b, bias, out, *, backend="auto", kernel_id=None):
    from flashinfer._matmul import metadata_reason
    reason = metadata_reason(a, b, out)
    if reason != "supported" or tuple(bias.shape) != (b.shape[1],):
        raise ValueError("invalid MatmulBiasSilu Tensor metadata: " + reason)


def torch_declarations():
    from flashinfer.jit.torch_library import AscendCOp
    return (
        AscendCOp("flashinfer::matmul_bias_silu", '(Tensor a, Tensor b, Tensor bias, *, str? backend="auto", str? kernel_id=None) -> Tensor',
                          _matmul, _matmul_meta),
        AscendCOp("flashinfer::matmul_bias_silu_out",
                          '(Tensor a, Tensor b, Tensor bias, Tensor(a!) out, *, str? backend="auto", str? kernel_id=None) -> ()',
                          _matmul_out, _matmul_out_meta, ("out",)),
    )
