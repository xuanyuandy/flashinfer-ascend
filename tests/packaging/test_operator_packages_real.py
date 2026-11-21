"""One acceptance matrix discovers all declarative operators, without per-op glue."""

import pytest

from flashinfer.operators.catalog import cases_module
from flashinfer.operators.pytest_support import aot_params, damaged_release, execute, prepare_release
from flashinfer.operators.testing import case_parameters

pytestmark = pytest.mark.real_compile


@pytest.fixture(scope="module", params=aot_params())
def operator_release(request, tmp_path_factory):
    impl = request.param
    cases_module(impl.op)
    return impl, prepare_release(impl.op, tmp_path_factory.mktemp("operator-aot"), kernel_id=impl.kernel_id)


@pytest.mark.parametrize("queue", [0, 1, 2])
def test_numeric_matrix_out_identity_and_nondefault_stream(operator_release, tmp_path, queue):
    impl, release = operator_release
    result = execute(impl.op, "matrix", tmp_path, release, kernel_id=impl.kernel_id, TASK_QUEUE_ENABLE=str(queue))
    assert result["target_soc"] == impl.native.target_soc
    assert result["observations"]["source"] == "aot"
    inputs = case_parameters(cases_module(impl.op), impl.kernel_id)
    assert set(result["observations"]) == {"source", "spec", *(str(i) for i in range(len(inputs)))}


@pytest.mark.parametrize("jit_disabled", ["0", "1"])
def test_missing_aot_obeys_implementation_jit_policy(operator_release, tmp_path, jit_disabled):
    impl, release = operator_release
    empty = tmp_path / "empty"
    empty.mkdir()
    mode = "jit" if jit_disabled == "0" and impl.native.allow_jit_fallback else "missing"
    result = execute(impl.op, mode, tmp_path, release, kernel_id=impl.kernel_id,
                     FLASHINFER_AOT_PATHS=str(empty), FLASHINFER_NPU_JIT_DISABLE=jit_disabled,
                     FLASHINFER_NPU_CACHE_DIR=str(tmp_path / "cold-cache"))
    if mode == "jit":
        assert result["observations"]["source"] == "jit"
    else:
        assert result["observations"]["error"] == "AotMissingError"


def test_corrupt_aot_never_falls_back(operator_release, tmp_path):
    impl, release = operator_release
    damaged = damaged_release(impl.op, tmp_path, release, kernel_id=impl.kernel_id)
    result = execute(impl.op, "corrupt", tmp_path, release, kernel_id=impl.kernel_id, FLASHINFER_AOT_PATHS=str(damaged),
                     FLASHINFER_NPU_JIT_DISABLE="0")
    assert result["observations"]["error"] == "AotCorruptError"
