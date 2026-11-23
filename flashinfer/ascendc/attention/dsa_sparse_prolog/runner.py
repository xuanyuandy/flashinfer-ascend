"""Launch the native DsaSparseProlog ABI after selection."""

from flashinfer.jit.tensor_adapter import package_tensor_runner


def make_runner(module, spec, recipe):
    return package_tensor_runner(module, recipe, spec=spec)
