import pytest

from flashinfer.runtime import (
    DispatchError, HardwareSignature, InputSignature, KernelRegistry,
    dispatch, explain_dispatch,
)


@pytest.mark.parametrize("backend", ["fa2", "fa3", "cudnn", "trtllm-gen", "cutlass", "cute-dsl"])
@pytest.mark.parametrize("operation", [dispatch, explain_dispatch])
def test_cuda_backend_names_explain_the_npu_migration(backend, operation):
    with pytest.raises(DispatchError) as raised:
        operation(InputSignature("unknown"), HardwareSignature("Ascend950", 56),
                  backend=backend, registry=KernelRegistry())
    message = str(raised.value)
    assert backend in message
    assert "CUDA" in message
    assert "backend='auto'" in message and "backend='cann'" in message
    assert "reference" in message and "triton" in message


@pytest.mark.parametrize("backend", ["typo", 42])
def test_unknown_backend_does_not_leak_enum_value_error(backend):
    with pytest.raises(DispatchError, match="支持的 backend 值"):
        dispatch(InputSignature("unknown"), HardwareSignature("Ascend950", 56),
                 backend=backend, registry=KernelRegistry())


def test_environment_policy_gets_the_same_diagnostic(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NPU_BACKEND", "fa3")
    with pytest.raises(DispatchError, match="CUDA"):
        dispatch(InputSignature("unknown"), HardwareSignature("Ascend950", 56),
                 backend=None, registry=KernelRegistry())
