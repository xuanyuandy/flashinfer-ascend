"""Versioned recipe and generic ACLNN contracts, without an SDK or device.

Synthetic callables test argument forwarding only. Actual mixed-core execution
and package verification belong to the separately marked real_compile tests.
"""

import ctypes as C
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from flashinfer.jit.core import JitSpec, _canonical_hash
from flashinfer.jit.op_package import OP_PACKAGE_ADAPTER_HASH, OpPackageModule, verify_op_package
from flashinfer.jit.recipe import BuildRecipe, ShapeConstraint, TensorParameter, load_build_recipes
from flashinfer.runtime import SchemaError

from tests.test_ascendc_op_package import elf_fixture, write_fixture


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def v1():
    return next(r for r in load_build_recipes(ROOT / "csrc/recipes/matmul_probe.json")
                if r.output_kind == "op_package" and r.implementation == "native")


@pytest.fixture
def v2(v1):
    return replace(v1, schema_version=2, tiling_mode="host_dynamic", tiling_schema_version=2,
        tiling_host_entry="FixtureTiling", workspace_host_entry="FixtureTiling",
        workspace_bytes="dynamic", task_type="aic_aiv", sync_policy="cross_core_flags",
        tensor_parameters=(TensorParameter("a", "input", "bfloat16", ("M", "K")),
                           TensorParameter("b", "input", "bfloat16", ("K", "N")),
                           TensorParameter("bias", "input", "bfloat16", ("N",)),
                           TensorParameter("out", "output", "bfloat16", ("M", "N"))),
        shape_constraints=tuple(ShapeConstraint(n, 16, 256, 16) for n in ("M", "N", "K")))


def metadata(m=16, n=32, k=64):
    return dict(shapes=dict(a=(m, k), b=(k, n), bias=(n,), out=(m, n)),
                dtypes={n: "bfloat16" for n in ("a", "b", "bias", "out")})


def test_v1_serialization_and_fingerprints_remain_exactly_frozen():
    manifest = json.loads((ROOT / "csrc/recipes/matmul_probe.json").read_text())
    for original in manifest:
        recipe = BuildRecipe.from_dict(original)
        assert json.loads(json.dumps(recipe.to_dict())) == original
        assert recipe.fingerprint == _canonical_hash(original)
        assert recipe.tensor_parameter_names == ("a", "b", "c")


@pytest.mark.parametrize("change", [
    dict(tiling_mode="host_dynamic"), dict(tiling_schema_version=2), dict(workspace_bytes=4),
    dict(workspace_bytes="dynamic"), dict(task_type="aic_aiv"),
    dict(sync_policy="cross_core_flags"), dict(tiling_host_entry="Tiling"),
    dict(tensor_parameters=(TensorParameter("x", "input", "float32", (16,)),)),
])
def test_v1_does_not_inherit_v2_capabilities(v1, change):
    with pytest.raises(SchemaError, match="v1"):
        replace(v1, **change)


@pytest.mark.parametrize("change,message", [
    (dict(tiling_mode="fixed"), "host_dynamic"),
    (dict(tiling_schema_version=1), "schema >= 2"),
    (dict(tiling_host_entry=None), "tiling_host_entry"),
    (dict(tiling_host_entry="bad entry"), "tiling_host_entry"),
    (dict(workspace_host_entry=None), "workspace_host_entry"),
    (dict(workspace_host_entry="bad entry"), "workspace_host_entry"),
    (dict(workspace_bytes=-1), "workspace_bytes"),
    (dict(workspace_bytes=True), "workspace_bytes"),
    (dict(workspace_bytes="unknown"), "workspace_bytes"),
    (dict(sync_policy="compiler_default"), "synchronization"),
    (dict(task_type="aiv_only"), "task_type"),
    (dict(tensor_parameters=()), "tensor_parameters"),
    (dict(shape_constraints=()), "every tensor shape symbol"),
    (dict(block_dim=0), "block_dim"),
])
def test_v2_rejects_incomplete_or_unimplemented_protocols(v2, change, message):
    with pytest.raises(SchemaError, match=message):
        replace(v2, **change)


def test_v2_requires_unique_ordered_tensor_arguments_and_exact_symbol_bounds(v2):
    for parameters in (v2.tensor_parameters + v2.tensor_parameters[:1],
                       v2.tensor_parameters[-1:] + v2.tensor_parameters[:-1]):
        with pytest.raises(SchemaError, match="tensor_parameters"):
            replace(v2, tensor_parameters=parameters)
    for constraints in (v2.shape_constraints + v2.shape_constraints[:1],
                        v2.shape_constraints + (ShapeConstraint("Unused", 1, 16),)):
        with pytest.raises(SchemaError, match="shape_constraints"):
            replace(v2, shape_constraints=constraints)


@pytest.mark.parametrize("argument", [
    dict(name="workspace", role="input", dtype="float32", shape=(16,)),
    dict(name="a", role="scalar", dtype="float32", shape=(16,)),
    dict(name="a", role="input", dtype="opaque", shape=(16,)),
    dict(name="a", role="input", dtype="float32", shape=(True,)),
    dict(name="a", role="input", dtype="float32", shape=(0,)),
])
def test_tensor_parameters_are_typed_tensors_not_unchecked_foreign_arguments(argument):
    with pytest.raises(SchemaError):
        TensorParameter(**argument)


@pytest.mark.parametrize("bounds", [(0, 16, 1), (16, 8, 1), (1, 16, 0), (1, 3, 4), (True, 16, 1)])
def test_shape_bounds_must_describe_a_nonempty_positive_integer_range(bounds):
    with pytest.raises(SchemaError, match="range"):
        ShapeConstraint("M", *bounds)


def test_v2_contract_is_round_trippable_and_identity_binds_every_new_field(v1, v2):
    assert BuildRecipe.from_dict(json.loads(json.dumps(v2.to_dict()))) == v2
    assert v1.fingerprint != v2.fingerprint
    assert v2.to_dict()["schema_version"] == 2
    assert v2.device_parameter_order == ("a", "b", "bias", "out", "workspace", "tiling")
    for change in (dict(tiling_schema_version=3), dict(tiling_host_entry="OtherTiling"),
                   dict(workspace_host_entry="OtherWorkspace"),
                   dict(workspace_bytes=4096, workspace_host_entry=None),
                   dict(task_type="aic_only", sync_policy="compiler_default"),
                   dict(tensor_parameters=(replace(v2.tensor_parameters[0], dtype="float32"),)
                        + v2.tensor_parameters[1:]),
                   dict(shape_constraints=(ShapeConstraint("M", 16, 128, 16),)
                        + v2.shape_constraints[1:])):
        assert replace(v2, **change).fingerprint != v2.fingerprint
    assert replace(v2, workspace_bytes=0, workspace_host_entry=None).workspace_bytes == 0


def test_dispatch_metadata_validation_checks_cross_tensor_dimensions_dtype_and_bounds(v2):
    assert v2.validate_tensor_metadata(**metadata()) == dict(M=16, K=64, N=32)
    assert v2.validate_tensor_metadata(**metadata(256, 256, 256))["M"] == 256
    for unsupported in (metadata(17), metadata(272), metadata(0)):
        with pytest.raises(SchemaError, match="dispatch shape"):
            v2.validate_tensor_metadata(**unsupported)
    inconsistent = metadata()
    inconsistent["shapes"]["bias"] = (16,)
    with pytest.raises(SchemaError, match="symbol N is inconsistent"):
        v2.validate_tensor_metadata(**inconsistent)
    rank = metadata()
    rank["shapes"]["bias"] = (1, 32)
    with pytest.raises(SchemaError, match="requires rank 1"):
        v2.validate_tensor_metadata(**rank)
    dtype = metadata()
    dtype["dtypes"]["bias"] = "float32"
    with pytest.raises(SchemaError, match="dtype for bias must be bfloat16"):
        v2.validate_tensor_metadata(**dtype)
    missing = metadata()
    missing["shapes"].pop("bias")
    with pytest.raises(SchemaError, match="match recipe parameters"):
        v2.validate_tensor_metadata(**missing)


@pytest.mark.parametrize("input_count", [0, 1, 3, 5])
def test_generic_aclnn_adapter_forwards_recipe_order_for_arbitrary_tensor_counts(v2, input_count):
    parameters = tuple(TensorParameter("arg%d" % i, "input", "float32", (16,))
                       for i in range(input_count)) + (TensorParameter("result", "output", "float32", (16,)),)
    recipe = replace(v2, tensor_parameters=parameters, shape_constraints=())
    calls = []

    def workspace(*arguments):
        calls.append(arguments[:-2])
        C.cast(arguments[-2], C.POINTER(C.c_uint64))[0] = 4096
        C.cast(arguments[-1], C.POINTER(C.c_void_p))[0] = 1234
        return 0

    def launch(*arguments):
        calls.append(arguments)
        return 0

    api = SimpleNamespace(**{recipe.host_entry + "GetWorkspaceSize": workspace, recipe.host_entry: launch})
    module = OpPackageModule(recipe, Path("/unloaded-synthetic-package"), api)
    arguments = tuple(range(1, len(parameters) + 1))
    size, executor = module.get_workspace_size(*arguments)
    assert (size, executor.value) == (4096, 1234)
    assert calls[-1] == arguments
    # Keyword insertion order is intentionally the reverse of the ABI order.
    module.get_workspace_size(**dict(reversed(tuple(zip(recipe.tensor_parameter_names, arguments)))))
    assert calls[-1] == arguments
    module.launch(99, size, executor, 7)
    assert calls[-1] == (99, 4096, executor, 7)
    for incorrect in (arguments[:-1], arguments + (999,)):
        with pytest.raises(SchemaError, match="tensor arguments"):
            module.get_workspace_size(*incorrect)
    with pytest.raises(SchemaError, match="match recipe order"):
        module.get_workspace_size(wrong=1)
    with pytest.raises(SchemaError, match="match recipe order"):
        module.get_workspace_size(1, **dict(zip(recipe.tensor_parameter_names, arguments)))


@pytest.fixture
def mixed_package(v2, tmp_path):
    spec = JitSpec(name="mixed_fixture", domain="test", generator_id="ascendc", generator_version="1",
        target_soc=v2.target_soc, environment_fingerprint="a" * 64, specialization_fingerprint="b" * 64,
        input_fingerprints=(("build_recipe", v2.fingerprint), ("output_adapter", OP_PACKAGE_ADAPTER_HASH)),
        kernel_id=v2.kernel_id, source=v2.source, output_path="op-package.json", entry_points=(v2.host_entry,))
    paths = write_fixture(tmp_path, v2, spec)
    metadata_path = tmp_path / paths["metadata"]
    metadata = json.loads(metadata_path.read_text())
    entry = metadata["kernelList"][0]["kernelName"]
    binary = elf_fixture((entry + "_mix_aic", entry + "_mix_aiv"), machine=0x1029, kind=2)
    (tmp_path / paths["binary"]).write_bytes(binary)
    metadata.pop("core_type")
    metadata.update(coreType="MIX", magic="RT_DEV_BINARY_MAGIC_ELF", intercoreSync=1,
                    sha256=hashlib.sha256(binary).hexdigest(), workspace=dict(num=1, size=[-1]),
                    supportInfo=dict(opMode="dynamic", inputs=[dict(name=p.name, dtype=p.dtype)
                        for p in v2.tensor_parameters if p.role == "input"],
                        outputs=[dict(name=p.name, dtype=p.dtype)
                        for p in v2.tensor_parameters if p.role == "output"]))
    metadata["kernelList"][0].update(kernelType="MIX_AIC", crossCoreSync=1, taskRation="1:2")
    metadata_path.write_text(json.dumps(metadata))
    return tmp_path, spec, v2, paths, metadata


def test_mixed_package_verifies_both_core_entries_and_declared_abi(mixed_package):
    directory, spec, recipe, _, _ = mixed_package
    assert verify_op_package(directory, spec)[1] == recipe


@pytest.mark.parametrize("change,message", [
    (dict(coreType="AiCore"), "AIC\\+AIV"),
    (dict(magic="RT_DEV_BINARY_MAGIC_ELF_AICUBE"), "AIC\\+AIV"),
    (dict(intercoreSync=0), "synchronized"),
    (dict(workspace=dict(num=1, size=[0])), "dynamic workspace"),
    (dict(supportInfo=dict(opMode="static")), "dynamic host tiling"),
])
def test_mixed_package_rejects_metadata_that_weakens_v2(mixed_package, change, message):
    directory, spec, _, paths, metadata = mixed_package
    metadata.update(change)
    (directory / paths["metadata"]).write_text(json.dumps(metadata))
    with pytest.raises(SchemaError, match=message):
        verify_op_package(directory, spec)


def test_mixed_package_rejects_missing_vector_entry(mixed_package):
    directory, spec, _, paths, metadata = mixed_package
    entry = metadata["kernelList"][0]["kernelName"]
    (directory / paths["binary"]).write_bytes(elf_fixture((entry + "_mix_aic",), machine=0x1029, kind=2))
    with pytest.raises(SchemaError, match="metadata entries"):
        verify_op_package(directory, spec)


@pytest.mark.parametrize("invalid", ["order", "dtype", "cross_core"])
def test_mixed_package_rejects_tensor_abi_or_cross_core_mismatch(mixed_package, invalid):
    directory, spec, _, paths, metadata = mixed_package
    if invalid == "order":
        metadata["supportInfo"]["inputs"].reverse()
    elif invalid == "dtype":
        metadata["supportInfo"]["inputs"][0]["dtype"] = "float32"
    else:
        metadata["kernelList"][0]["crossCoreSync"] = 0
    (directory / paths["metadata"]).write_text(json.dumps(metadata))
    with pytest.raises(SchemaError, match="order/types|synchronized"):
        verify_op_package(directory, spec)
