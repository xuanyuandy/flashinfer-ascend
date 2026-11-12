"""Backend-independent declaration for gemm.matmul."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='matmul', domain='gemm', op='gemm.matmul', public_entry='flashinfer.gemm:matmul')


from types import SimpleNamespace

def _plain_matmul(a, b, *, backend=None, kernel_id=None):
    from flashinfer.gemm import _matmul_impl
    return _matmul_impl(a, b, backend=backend, kernel_id=kernel_id)


def _plain_matmul_out(a, b, out, *, backend=None, kernel_id=None):
    from flashinfer.gemm import _matmul_impl
    _matmul_impl(a, b, out, backend=backend, kernel_id=kernel_id)


def _plain_matmul_out_meta(a, b, out, *, backend=None, kernel_id=None):
    from flashinfer._matmul import metadata_reason
    reason = metadata_reason(a, b, out)
    if reason != "supported":
        from flashinfer.runtime import SchemaError
        raise SchemaError(reason)


def _plain_matmul_meta(a, b, *, backend=None, kernel_id=None):
    import torch
    from flashinfer._matmul import output_metadata

    prospective = output_metadata(a, b)
    _plain_matmul_out_meta(a, b, prospective)
    return torch.empty(prospective.shape, dtype=a.dtype, device=a.device)


def torch_declarations():
    from flashinfer.jit.torch_library import AscendCOp
    return (
        AscendCOp("flashinfer::matmul", '(Tensor a, Tensor b, *, str? backend=None, str? kernel_id=None) -> Tensor',
                          _plain_matmul, _plain_matmul_meta),
        AscendCOp("flashinfer::matmul_out",
                          '(Tensor a, Tensor b, Tensor(a!) out, *, str? backend=None, str? kernel_id=None) -> ()',
                          _plain_matmul_out, _plain_matmul_out_meta, ("out",)),
    )
