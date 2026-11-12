"""One acceptance matrix discovers all declarative operators, without per-op glue."""

import pytest

from flashinfer.operators.catalog import cases_module, entries
from flashinfer.operators.pytest_support import damaged_release, execute, prepare_release

pytestmark = pytest.mark.real_compile
NAMES = tuple(r["name"] for r in entries() if r["cases"])


@pytest.fixture(scope="module", params=NAMES)
def operator_release(request, tmp_path_factory):
    name = request.param
    return name, prepare_release(name, tmp_path_factory.mktemp("operator-aot"))


@pytest.mark.parametrize("queue", [0, 1, 2])
def test_numeric_matrix_out_identity_and_nondefault_stream(operator_release, tmp_path, queue):
    name, release = operator_release
    result = execute(name, "matrix", tmp_path, release, TASK_QUEUE_ENABLE=str(queue))
    assert result["observations"]["source"] == "aot"
    assert len(result["observations"]) == len(cases_module(name).CASE_PARAMETERS) + 2


@pytest.mark.parametrize("jit_disabled", ["0", "1"])
def test_missing_aot_never_compiles(operator_release, tmp_path, jit_disabled):
    name, release = operator_release
    empty = tmp_path / "empty"
    empty.mkdir()
    result = execute(name, "missing", tmp_path, release, FLASHINFER_AOT_PATHS=str(empty),
                     FLASHINFER_NPU_JIT_DISABLE=jit_disabled)
    assert result["observations"]["error"] == "AotMissingError"


def test_corrupt_aot_never_falls_back(operator_release, tmp_path):
    name, release = operator_release
    damaged = damaged_release(name, tmp_path, release)
    result = execute(name, "corrupt", tmp_path, release, FLASHINFER_AOT_PATHS=str(damaged),
                     FLASHINFER_NPU_JIT_DISABLE="0")
    assert result["observations"]["error"] == "AotCorruptError"
