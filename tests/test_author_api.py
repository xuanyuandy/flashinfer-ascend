"""Author declarations and resource loading keep runtime/build boundaries explicit."""

from pathlib import Path
import subprocess
import sys
import shutil
import os

import pytest

from flashinfer.operators import (
    CANNImplementation,
    DimRange,
    OperatorDefinition,
    PythonKernel,
    Support,
)
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.runtime import InputSignature, SchemaError


def test_named_dimensions_bind_all_occurrences_and_enforce_alignment():
    operator = OperatorDefinition(
        "dense",
        "gemm",
        "gemm.dense",
        tensor_shapes=(("M", "K"), ("K", "N"), ("M", "N")),
    )
    implementation = CANNImplementation(
        operator=operator,
        runner=lambda: None,
        support=Support(
            soc="Ascend910B",
            dtype="bfloat16",
            layout="ND",
            dimensions={"M": DimRange(16, 256, 16)},
        ),
    )
    descriptor = implementation.bind()
    assert descriptor.kernel_id == "cann/gemm/dense/default"
    assert descriptor.capability.soc_models == ("Ascend910B",)
    assert {
        (r.tensor_index, r.axis, r.minimum, r.maximum)
        for r in descriptor.capability.shape_ranges
    } == {
        (0, 0, 16, 256),
        (2, 0, 16, 256),
    }
    assert (
        descriptor.admission(
            InputSignature(operator.op, shapes=((32, 16), (16, 64), (32, 64)))
        )
        == ()
    )
    assert (
        "multiple_of=16"
        in descriptor.admission(
            InputSignature(operator.op, shapes=((17, 16), (16, 64), (17, 64)))
        )[0]
    )


@pytest.mark.parametrize("limits", [(0, 2, 1), (2, 1, 1), (1, 2, 0), (True, 2, 1)])
def test_dimension_units_and_bounds_are_not_guessed(limits):
    with pytest.raises(SchemaError):
        DimRange(*limits)


def test_author_typos_and_ambiguous_runner_definitions_fail_explicitly():
    operator = OperatorDefinition("dense", "gemm", "gemm.dense")
    with pytest.raises(SchemaError, match="unknown support"):
        Support(dytpe="bfloat16")
    with pytest.raises(SchemaError, match="both"):
        Support(dtype="bfloat16", dtypes=("float16",))
    with pytest.raises(SchemaError, match="absent"):
        Support(dimensions={"unknown": DimRange(1, 16)}).capability(operator)
    with pytest.raises(SchemaError, match="exactly one"):
        CANNImplementation(
            operator=operator, runner=lambda: None, runner_factory=lambda _: None
        )


def test_python_source_identity_is_computed_and_path_escape_is_rejected(tmp_path):
    directory = tmp_path / "triton/gemm/dense"
    directory.mkdir(parents=True)
    source = directory / "kernels.py"
    source.write_text("# kernel version one\n")
    kernel = PythonKernel()
    before = kernel.source_ref(directory, tmp_path)
    source.write_text("# kernel version two\n")
    assert kernel.source_ref(directory, tmp_path).digest != before.digest
    outside = tmp_path / "external.py"
    outside.write_text("# unrelated source\n")
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(SchemaError, match="regular implementation-local"):
        kernel.source_ref(directory, tmp_path)
    with pytest.raises(SchemaError):
        PythonKernel("../../../external.py").source_ref(directory, tmp_path)


def test_string_attributes_have_a_typed_nul_safe_contract():
    attribute = AttributeParameter("layout", "string", "BSND")
    assert attribute.validate("TND") == "TND"
    for value in (None, 1, True, b"BSND", "BSND\0TND"):
        with pytest.raises(SchemaError):
            attribute.validate(value)
    with pytest.raises(SchemaError, match="numeric bounds"):
        AttributeParameter("layout", "string", "BSND", minimum=1)


def test_source_registry_does_not_read_generated_json_or_probe_a_toolchain():
    root = Path(__file__).resolve().parents[1]
    program = """
from pathlib import Path
from flashinfer._resources import generated_root
from flashinfer.jit import AscendCToolchain
from flashinfer.runtime import get_global_registry
from tests.production_inventory import DEFAULT_DESCRIPTOR_COUNT
def forbidden(*args, **kwargs):
    raise AssertionError("registration probed the build toolchain")
AscendCToolchain.discover = forbidden
read = Path.read_text
def checked(path, *args, **kwargs):
    if path.suffix == ".json" and path.is_relative_to(generated_root()):
        raise AssertionError("registration read generated JSON: " + str(path))
    return read(path, *args, **kwargs)
Path.read_text = checked
descriptors = get_global_registry().descriptors()
assert len(descriptors) == DEFAULT_DESCRIPTOR_COUNT
assert all(d.runner.provenance is None for d in descriptors if d.backend.value == "ascendc")
print("source registry: no generated metadata, SDK probe or compilation")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_fresh_checkout_generates_delivery_resources_without_source_data(tmp_path):
    from flashinfer._resources import data_root, recipe_root
    from flashinfer.operators.authoring import generate
    from flashinfer.runtime.manifest import load_kernel_manifest
    from tests.production_inventory import PRODUCTION_KERNEL_IDS

    root = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    for name in ("flashinfer", "csrc"):
        shutil.copytree(
            root / name,
            project / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copyfile(root / "pyproject.toml", project / "pyproject.toml")
    assert not (project / "flashinfer/data").exists()
    generate(source_root=project)
    assert {
        d.kernel_id
        for d in load_kernel_manifest(data_root(project) / "kernels/registry.json")
    } == PRODUCTION_KERNEL_IDS
    assert recipe_root(project).is_relative_to(project / "build/generated")
    assert not (data_root(project) / "kernels/ascendc_specs.json").exists()
    assert not (project / "flashinfer/data").exists()
    generate(source_root=project, check=True)


@pytest.mark.real_compile
@pytest.mark.parametrize("queue", [0, 1, 2])
def test_native_string_attributes_keep_order_and_owned_storage(
    tmp_path, queue, string_adapter
):
    source = tmp_path / "probe.cpp"
    source.write_text(
        r"""
#include <cstdint>
#include <cstring>
extern "C" {
static int calls = 0;
uint32_t aclnnFiStringProbeGetWorkspaceSize(void*, void*, const char* q, const char* kv,
        bool flag, int64_t integer, double floating, void*, uint64_t* size, void** executor) {
    if (std::strcmp(q, "BSND") || std::strcmp(kv, "TND") || !flag || integer != 17 || floating != 2.5) return 7;
    ++calls; *size = 0; *executor = reinterpret_cast<void*>(1); return 0;
}
uint32_t aclnnFiStringProbe(void*, uint64_t, void*, void*) { return 0; }
int probe_calls() { return calls; }
}
"""
    )
    compiler = shutil.which("c++")
    assert compiler, "native bridge verification requires c++"
    library = tmp_path / "probe.so"
    subprocess.run(
        [compiler, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
    )
    program = r"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import ctypes, os, sys
import torch, torch_npu
from flashinfer.aclnn import _native
from flashinfer.aclnn.build import build_adapter
from flashinfer.ascendc import load_contract
from flashinfer.jit.aclnn_contract import AttributeParameter
torch.npu.set_device(int(os.environ.get("FLASHINFER_TEST_NPU_DEVICE", "0")))
assert torch.npu.is_available(), "missing NPU is a failure"
if not os.environ.get("FLASHINFER_ACLNN_ADAPTER_PATH"):
    adapter = Path(sys.argv[1]).parent / "adapter"
    build_adapter(adapter)
    os.environ["FLASHINFER_ACLNN_ADAPTER_PATH"] = str(adapter)
recipe = load_contract("ascendc/gemm/matmul/native/op_package/Ascend950")[1]
attributes = (AttributeParameter("layout_query", "string", "BSND"),
              AttributeParameter("layout_kv", "string", "TND"),
              AttributeParameter("enabled", "bool", True),
              AttributeParameter("count", "int64", 17),
              AttributeParameter("scale", "float64", 2.5))
recipe = replace(recipe, recipe_id="string_probe", kernel_id="test/string_probe",
                 package_op_type="FiStringProbe", host_entry="aclnnFiStringProbe", attributes=attributes)
api = ctypes.CDLL(sys.argv[1])
run = _native.prepare(SimpleNamespace(api=api), recipe)
tensors = tuple(torch.empty((16, 16), device="npu", dtype=torch.bfloat16) for _ in range(3))
for _ in range(3):
    run(tensors, recipe.validate_attributes())
torch.npu.synchronize()
assert api.probe_calls() == 3
print("native string ABI: three calls, exact attribute order and values")
"""
    environment = dict(
        os.environ,
        TASK_QUEUE_ENABLE=str(queue),
        FLASHINFER_ACLNN_ADAPTER_PATH=str(string_adapter),
    )
    environment.pop("ASCEND_LAUNCH_BLOCKING", None)
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, str(library)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def string_adapter(tmp_path_factory):
    from flashinfer.aclnn.build import build_adapter
    from flashinfer.aclnn._native import verified_adapter

    supplied = os.environ.get("FLASHINFER_OPERATOR_TEST_ADAPTER")
    if supplied:
        verified_adapter(supplied)
        return Path(supplied)
    directory = tmp_path_factory.mktemp("string-attribute-adapter")
    build_adapter(directory)
    return directory
