"""operator authoring regression coverage."""


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
    root = Path(__file__).resolve().parents[2]
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

    root = Path(__file__).resolve().parents[2]
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


from dataclasses import replace
import json


from flashinfer.operators.authoring import discover, discover_implementations, generate
from flashinfer.operators.implementation import ImplementationDefinition, implementation_entries
from flashinfer.runtime import (
    Backend, CapabilityCase, CapabilitySpec, FeatureConstraint, HardwareSignature, InputSignature, KernelDescriptor,
    KernelRegistry, SchemaError, ShapeRange, dispatch, get_global_registry,
)
from flashinfer.runtime.registry import kernel_registry_scope
from flashinfer.runtime.runner import run_selected

ROOT = Path(__file__).resolve().parents[2]


def test_existing_85_and_reviewed_arch22_capabilities_match_exactly():
    baseline = json.loads((ROOT / 'tests/framework/data/production_capabilities.json').read_text())
    actual = {d.kernel_id: d.capability.to_dict() for d in get_global_registry().descriptors()}
    # Keep the original 85-entry fixture frozen. Reviewed additions and SoC
    # changes are explicit below, never copied from implementation discovery.
    expected = dict(baseline)
    expected["ascendc/t4/flash_attention/catlass/op_package/Ascend910B"] = dict(
        baseline["ascendc/t4/flash_attention/catlass/op_package/Ascend950"],
        soc_models=["Ascend910B"],
    )
    expected["ascendc/attention/sparse_flash_attention/native/Ascend910B"] = CapabilitySpec(
        op="attention.sparse_flash_attention",
        soc_models=("Ascend910B",),
        dtypes=("float16", "bfloat16", "int32", "float32", "none"),
        cases=tuple(CapabilityCase(
            name="arch22_mla_lse" if lse else "arch22_mla_no_lse",
            compute_dtypes=("float16", "bfloat16"),
            accumulator_dtypes=("float32",), attention_variants=("mla",),
            kv_cache_types=("dense", "ragged", "paged"), causal_values=(False, True),
            sparse_formats=("block_sparse",),
            auxiliary_outputs=("softmax_max", "softmax_sum") if lse else (),
        ) for lse in (False, True)),
    ).to_dict()
    # Only these reviewed general-purpose backends extend their SoC lists.
    # The frozen fixture remains unchanged; every other capability field is exact.
    for kernel_id in (
        'cann/cascade/merge_state_attention_update_v1',
        'cann/decode/cudnn_batch_decode_fia_v1',
        'cann/fused_moe/cutlass_fused_moe_gmm_v1',
        'cann/gemm/bmm_bf16_v1',
        'cann/gemm/mm_bf16_v1',
        'cann/gemm/segment_gemm_v1',
        'cann/norm/fused_add_rmsnorm_aclnn_v1',
        'cann/norm/gemma_rmsnorm_aclnn_v1',
        'cann/norm/layernorm_aclnn_v1',
        'cann/norm/rmsnorm_aclnn_v1',
        'cann/rope/apply_rope_cos_sin_cache_aclnn_v1',
        'cann/rope/apply_rope_cos_sin_cache_out_v1',
        'cann/rope/apply_rope_inplace_v1',
        'cann/rope/apply_rope_pos_ids_inplace_v1',
        'cann/rope/apply_rope_pos_ids_v1',
        'cann/rope/apply_rope_v1',
        'triton/norm/gemma_fused_add_rmsnorm_v1',
    ):
        expected[kernel_id] = dict(baseline[kernel_id],
                                   soc_models=["Ascend910B", "Ascend910C", "Ascend950"])
    # Reviewed addition: gemm.grouped_matmul_k_dim on arch22. The capability is
    # restated here rather than imported, so widening what the kernel claims still
    # requires editing this fixture.
    kdim_cases = tuple(
        CapabilityCase(
            name="%s_to_%s" % (compute, output),
            dtype_signatures=((compute, compute, "int64", output),),
            compute_dtypes=(compute,),
            accumulator_dtypes=("float32",),
            shape_ranges=(ShapeRange("XB", 0, 1, 1, 65535), ShapeRange("N", 1, 1, 1, 65535)),
            specializations=(("tensor_contract", ("supported",)),),
        )
        for compute in ("float16", "bfloat16")
        for output in (compute, "float32")
    )
    for soc in ("Ascend910B", "Ascend910C"):
        expected["ascendc/gemm/grouped_matmul_k_dim/native/op_package/" + soc] = CapabilitySpec(
            op="gemm.grouped_matmul_k_dim",
            soc_models=(soc,),
            physical_layouts=("ND",),
            cases=kdim_cases,
        ).to_dict()
    # The reference backend is device agnostic and declares no SoC or layout claim.
    expected["reference/gemm/grouped_matmul_k_dim_v1"] = CapabilitySpec(
        op="gemm.grouped_matmul_k_dim",
        cases=kdim_cases,
    ).to_dict()
    for kernel_id in (
        "ascendc/t4/flash_attention/catlass/op_package/Ascend910B",
        "ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B",
        "ascendc/attention/sparse_flash_attention/native/Ascend910B",
    ):
        expected[kernel_id.rsplit("/", 1)[0] + "/Ascend910C"] = dict(
            expected[kernel_id], soc_models=["Ascend910C"])
    assert actual == expected


def test_all_existing_implementations_have_one_authority_and_a_bound_runner():
    declarations = discover_implementations(ROOT)
    production = [d for _, _, d in declarations if d.scope == 'global']
    registry = get_global_registry()
    assert {d.kernel_id for d in production} == {d.kernel_id for d in registry.descriptors()}
    assert len(production) == 93
    assert len(discover(ROOT)) == 49
    assert len([d for _, _, d in declarations if d.scope == 'experimental']) == 1
    for directory, _, declaration in declarations:
        assert directory.parent.parent.name == declaration.backend.value
        assert callable(declaration.bind().runner)
    assert {r['op'] for r in implementation_entries()} == {o.op for _, o in discover(ROOT)}
    generate(check=True)


def test_admission_rejects_only_its_own_backend_before_ranking_and_execution():
    calls = []
    first = KernelDescriptor('cann/example/narrow', op='example.op', backend=Backend.CANN,
        capability=CapabilitySpec('example.op'), runner=lambda value: calls.append(('narrow', value)),
        admission=lambda signature: ('native shape bound exceeded',) if signature.shapes[0][0] > 8 else ())
    second = KernelDescriptor('triton/example/wide', op='example.op', backend=Backend.TRITON,
        capability=CapabilitySpec('example.op'), runner=lambda value: calls.append(('wide', value)))
    registry = KernelRegistry((first, second)); registry.freeze()
    signature = InputSignature('example.op', shapes=((16,),))
    hardware = HardwareSignature('Ascend910B', 20)
    with kernel_registry_scope(registry):
        decision = dispatch(signature, hardware)
        assert decision.kernel_id == second.kernel_id
        run_selected(decision.kernel_id, 'logical argument')
        with pytest.raises(RuntimeError, match='native shape bound exceeded'):
            dispatch(signature, hardware, kernel_id=first.kernel_id)
    assert calls == [('wide', 'logical argument')]


def test_implementation_factory_cannot_change_identity_or_omit_runner():
    descriptor = KernelDescriptor('cann/example/one', op='example.op', backend=Backend.CANN,
                                  runner=lambda: None)
    definition = ImplementationDefinition(descriptor.kernel_id, descriptor.op, descriptor.backend,
                                          lambda: descriptor)
    assert definition.bind() is descriptor
    with pytest.raises(SchemaError, match='identity'):
        replace(definition, factory=lambda: replace(descriptor, op='other.op')).bind()
    with pytest.raises(SchemaError, match='runner'):
        replace(definition, factory=lambda: replace(descriptor, runner=None)).bind()


def test_common_matmul_metadata_never_reads_an_ascendc_contract(monkeypatch):
    from types import SimpleNamespace
    from flashinfer._matmul import metadata_reason
    monkeypatch.setattr('flashinfer.ascendc.load_contract', lambda _: pytest.fail('backend coupling'))
    tensor = lambda shape: SimpleNamespace(shape=shape, dtype='bfloat16', device='meta')
    assert metadata_reason(tensor((16, 32)), tensor((32, 64)), tensor((16, 64))) == 'supported'
    assert metadata_reason(tensor((16, 31)), tensor((32, 64)), tensor((16, 64))) != 'supported'


def test_logical_call_can_use_another_backend_without_loading_a_native_recipe(monkeypatch):
    import torch
    from flashinfer.operators.catalog import definition
    from flashinfer.operators import runtime

    operator = definition('gemm.grouped_matmul_swiglu_quant')
    observed = []
    def runner(x, weight, scale, x_scale, assistance, groups, out, out_scale, *, swiglu_limit):
        observed.append((assistance, groups, swiglu_limit))
        out.zero_()
        out_scale.fill_(1)
    monkeypatch.setattr('flashinfer.ascendc.load_contract', lambda _: pytest.fail('native recipe leak'))
    monkeypatch.setattr(runtime, 'current_hardware', lambda: HardwareSignature('Ascend910B', 20))
    x = torch.ones((2, 16), dtype=torch.int8)
    weight = torch.ones((1, 16, 32), dtype=torch.int8)
    scale = torch.ones((1, 32))
    x_scale = torch.ones(2)
    groups = torch.tensor([2], dtype=torch.int64)
    arguments = operator.bind(x, weight, scale, x_scale, groups, backend='reference', swiglu_limit=.5)
    inference = operator.infer(arguments, {"weight": "ND"}).inference
    capability = CapabilitySpec(operator.op, cases=(CapabilityCase("logical_reference_test",
        compute_dtypes=(inference.compute_dtype,), accumulator_dtypes=(inference.accumulator_dtype,),
        quantization=inference.quantization, fusion_ops=inference.fusion_ops,
        auxiliary_outputs=inference.auxiliary_outputs,
        features=tuple(FeatureConstraint(name, (value,)) for name, value in inference.features)),))
    descriptor = KernelDescriptor('reference/test/grouped', op=operator.op, backend=Backend.REFERENCE,
                                  capability=capability, runner=runner)
    registry = KernelRegistry((descriptor,)); registry.freeze()
    with kernel_registry_scope(registry):
        out, out_scale = runtime.execute(operator.op, arguments)
    assert observed == [(None, groups, .5)]
    assert out.shape == (2, 16) and out_scale.shape == (2,)
    assert bool((out == 0).all()) and bool((out_scale == 1).all())


def test_functional_out_alias_is_rejected_without_loading_a_backend(monkeypatch):
    import torch
    from flashinfer.operators.catalog import definition
    from flashinfer.operators import runtime
    operator = definition('gemm.grouped_matmul_swiglu_quant')
    monkeypatch.setattr('flashinfer.ascendc.load_contract', lambda _: pytest.fail('native recipe leak'))
    x = torch.ones((2, 16), dtype=torch.int8)
    weight = torch.ones((1, 16, 32), dtype=torch.int8)
    scale = torch.ones((1, 32))
    x_scale = torch.ones(2)
    groups = torch.tensor([2], dtype=torch.int64)
    out = torch.empty((2, 16), dtype=torch.int8)
    arguments = operator.bind(x, weight, scale, x_scale, groups, out=(out, x_scale))
    with pytest.raises(ValueError, match='overlap'):
        runtime.metadata(operator.op, arguments, check_layouts=True)


def test_every_public_entry_resolves_and_preserves_its_frontend_signature():
    import importlib
    import inspect
    from flashinfer.operators.catalog import definition, entries
    for record in entries():
        operator = definition(record["op"])
        if operator.public_entry is None:
            continue
        module, attribute = operator.public_entry.split(":", 1)
        frontend = getattr(importlib.import_module(module), attribute)
        assert callable(frontend), operator.op
        assert operator.public_signature == inspect.signature(frontend), operator.op


import hashlib
import inspect
from types import SimpleNamespace


from flashinfer.operators.authoring import discover, generate
from flashinfer.operators.catalog import definition
from flashinfer.operators.definition import schema_arguments
from flashinfer.jit.recipe import BuildRecipe, resolve_build_recipe
from flashinfer.runtime import SchemaError
from flashinfer._resources import data_root, recipe_root

NAME = "grouped_matmul_swiglu_quant"
KERNEL_ID = "ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B"


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    data = root / "flashinfer/data/kernels"
    data.mkdir(parents=True)
    for file in ("registry.json", "ascendc_specs.json"):
        shutil.copyfile(data_root(ROOT) / "kernels" / file, data / file)
    paths = {
        "native": root / "csrc/kernels/gemm/local_probe",
        "logical": root / "flashinfer/operators/gemm/local_probe",
        "implementation": root / "flashinfer/ascendc/gemm/local_probe",
    }
    shutil.copytree(ROOT / "csrc/kernels/gemm" / NAME, paths["native"])
    shutil.copytree(ROOT / "flashinfer/operators/gemm" / NAME, paths["logical"])
    shutil.copytree(ROOT / "flashinfer/ascendc/gemm" / NAME, paths["implementation"])
    for directory in (paths["logical"], paths["implementation"]):
        for source in directory.rglob("*.py"):
            text = source.read_text().replace(
                "gemm.grouped_matmul_swiglu_quant", "gemm.local_probe"
            )
            text = text.replace("gemm/grouped_matmul_swiglu_quant", "gemm/local_probe")
            text = text.replace(
                'name="grouped_matmul_swiglu_quant"', 'name="local_probe"'
            )
            text = text.replace(
                "name='grouped_matmul_swiglu_quant'", "name='local_probe'"
            )
            text = text.replace(
                "recipe_id='grouped_matmul_swiglu_quant'", "recipe_id='local_probe'"
            )
            text = text.replace('recipe_id="grouped_matmul_swiglu_quant"', 'recipe_id="local_probe"')
            text = text.replace(
                "flashinfer.operators.gemm.local_probe.operator",
                "flashinfer.operators.gemm.grouped_matmul_swiglu_quant.operator",
            )
            if source.name == "implementation.py":
                # This fixture authors one explicit B probe and an editable ID
                # for the takeover negative case. Production GMM now expands
                # B/C variants; keep the fixture's original single-unit scope.
                text = text.replace('soc_models=ARCH22_SOCS', 'soc_models=("Ascend910B",)')
                text = text.replace('IMPLEMENTATIONS = AscendCImplementation.for_targets(',
                                    'IMPLEMENTATION = AscendCImplementation(')
                text = text.replace('default_target="Ascend910B",', 'kernel_id=KERNEL_ID,')
                text = text.replace('IMPLEMENTATION = IMPLEMENTATIONS[0]',
                                    'IMPLEMENTATIONS = (IMPLEMENTATION,)')
                text = text.replace(
                    "from .runner import make_runner, admission",
                    "from .runner import make_runner, admission\nfrom dataclasses import replace\n"
                    "OPERATOR = replace(OPERATOR, name='local_probe', op='gemm.local_probe')",
                )
            source.write_text(text)
    (root / "csrc/recipes").mkdir()
    shutil.copyfile(ROOT / "csrc/CMakeLists.txt", root / "csrc/CMakeLists.txt")
    shutil.copytree(ROOT / "csrc/cmake", root / "csrc/cmake")
    return root, paths


def recipe(root):
    return BuildRecipe.from_dict(
        json.loads((root / "csrc/recipes/local_probe.json").read_text())[0]
    )


def test_new_directory_generates_all_glue_without_changing_framework_sources(project):
    root, directory = project
    before = json.loads((root / "flashinfer/data/kernels/registry.json").read_text())
    selected = generate(source_root=root)
    assert selected == ("ascendc/gemm/local_probe/native/op_package/Ascend910B",)
    after = json.loads((root / "flashinfer/data/kernels/registry.json").read_text())
    assert after["kernels"][:-1] == before["kernels"]
    assert after["kernels"][-1]["op"] == "gemm.local_probe"
    catalog = json.loads((root / "flashinfer/data/operators.json").read_text())
    assert [(r["name"], r["domain"]) for r in catalog["operators"]] == [
        ("local_probe", "gemm")
    ]
    assert (directory["logical"] / "operator.py").is_file()
    assert (directory["logical"] / "cases.py").is_file()
    implementations = json.loads(
        (root / "flashinfer/data/implementations.json").read_text()
    )
    assert [r["module"] for r in implementations["implementations"]] == [
        "flashinfer.ascendc.gemm.local_probe.implementation"
    ]
    assert not (root / "flashinfer/operators/_generated").exists()
    generate(source_root=root, check=True)


@pytest.mark.parametrize(
    "owner,path",
    [
        ("native", "CMakeLists.txt"),
        ("native", "settings.cmake"),
        ("logical", "operator.py"),
        ("logical", "cases.py"),
        ("implementation", "runner.py"),
        ("native", "op_kernel/arch22/grouped_matmul_swiglu_quant.h"),
        ("native", "op_kernel/extra.hpp"),
    ],
)
def test_all_author_inputs_invalidate_recipe_and_stale_files_are_detected(
    project, owner, path
):
    root, directories = project
    directory = directories[owner]
    if path == "settings.cmake":
        (directory / path).write_text("# declared local CMake include\n")
    if path == "op_kernel/extra.hpp":
        (directory / path).write_text("// auxiliary template header\n")
    generate(source_root=root)
    original = recipe(root)
    target = directory / path
    target.write_text(
        target.read_text()
        + ("\n// changed\n" if target.suffix == ".h" else "\n# changed\n")
    )
    with pytest.raises(SchemaError, match="stale"):
        generate(source_root=root, check=True)
    with pytest.raises(SchemaError, match="digest mismatch"):
        resolve_build_recipe(original, source_root=root, toolchain_fingerprint="a" * 64)
    generate(source_root=root)
    changed = recipe(root)
    assert changed.fingerprint != original.fingerprint
    assert (
        changed.source == original.source
    )  # A header/CMake/hook change still matters.


def test_generation_is_idempotent_and_does_not_touch_unselected_frozen_specs(project):
    root, _ = project
    frozen = root / "flashinfer/data/kernels/ascendc_specs.json"
    old = frozen.read_bytes()
    generate(source_root=root)
    paths = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in paths}
    generate(source_root=root)
    assert all((p.stat().st_mtime_ns, p.read_bytes()) == before[p] for p in paths)
    assert frozen.read_bytes() == old


def test_missing_case_provider_fails_before_publishing_any_record(project):
    root, directory = project
    registry = root / "flashinfer/data/kernels/registry.json"
    old = registry.read_bytes()
    (directory["logical"] / "cases.py").unlink()
    with pytest.raises(SchemaError, match="cases.py"):
        generate(source_root=root)
    assert registry.read_bytes() == old
    assert not (root / "csrc/recipes/local_probe.json").exists()


def test_removing_the_last_author_directory_cannot_report_a_fresh_catalog(project):
    root, directory = project
    generate(source_root=root)
    (directory["logical"] / "operator.py").unlink()
    with pytest.raises(SchemaError, match="managed operator source was removed"):
        generate(source_root=root, check=True)


def test_symlink_source_is_not_silently_included_in_a_closed_build(project, tmp_path):
    root, directory = project
    external = tmp_path / "external.h"
    external.write_text("// untracked dependency\n")
    (directory["native"] / "op_kernel/external.h").symlink_to(external)
    with pytest.raises(SchemaError, match="in-directory regular"):
        generate(source_root=root)


def test_author_directory_cannot_take_over_an_existing_production_identity(project):
    root, directory = project
    registry = root / "flashinfer/data/kernels/registry.json"
    before = registry.read_bytes()
    source = directory["implementation"] / "implementation.py"
    source.write_text(
        source.read_text().replace(
            "ascendc/gemm/local_probe/native/op_package/Ascend910B",
            "ascendc/gemm/matmul/native/op_package/Ascend950",
        )
    )
    with pytest.raises(SchemaError, match="unrelated production"):
        generate(source_root=root)
    assert registry.read_bytes() == before


def test_optional_cmake_and_nested_python_helpers_are_collected(project):
    root, directory = project
    source = directory["implementation"] / "implementation.py"
    source.write_text(
        source.read_text().replace('cmake="CMakeLists.txt"', "cmake=None")
    )
    (directory["native"] / "CMakeLists.txt").unlink()
    helpers = directory["implementation"] / "helpers"
    helpers.mkdir()
    (helpers / "__init__.py").write_text('"""Author helper package."""\n')
    (helpers / "values.py").write_text("VALUE = 1\n")
    generate(source_root=root)
    assert recipe(root).cmake_entry is None
    author_helper = directory["implementation"] / "helpers/values.py"
    assert author_helper.read_text() == "VALUE = 1\n"
    assert author_helper.relative_to(root).as_posix() in {
        r.locator for r in recipe(root).extra_build_files
    }
    resolved = resolve_build_recipe(
        recipe(root), source_root=root, toolchain_fingerprint="a" * 64
    )
    assert "csrc/cmake/operator/CMakeLists.txt" in dict(resolved.build_files)


def test_public_integration_points_have_no_operator_specific_glue():
    for name in ("__init__.py", "gemm.py", "aclnn/_ops.py", "selftest.py"):
        assert NAME not in (ROOT / "flashinfer" / name).read_text(), name


def test_extra_build_contract_is_opt_in_and_roundtrips_without_changing_legacy_records():
    from tests.support.legacy_ascendc import ROOT as LEGACY_ROOT, load_build_recipes

    raw = json.loads((LEGACY_ROOT / "csrc/recipes/matmul_probe.json").read_text())
    for declaration, original in zip(
        load_build_recipes(LEGACY_ROOT / "csrc/recipes/matmul_probe.json"), raw
    ):
        assert json.loads(json.dumps(declaration.to_dict())) == original
        with pytest.raises(SchemaError, match="hashed extra"):
            replace(declaration, cmake_entry="untracked.cmake")
    current = BuildRecipe.from_dict(
        json.loads((recipe_root(ROOT) / (NAME + ".json")).read_text())[0]
    )
    assert BuildRecipe.from_dict(current.to_dict()) == current
    assert current.cmake_entry.endswith("/CMakeLists.txt")
    assert current.extra_build_files


def test_api_and_torch_schemas_derive_from_the_same_author_declaration():
    import flashinfer

    operator = definition(NAME)
    assert (
        inspect.signature(flashinfer.grouped_matmul_swiglu_quant)
        == operator.public_signature
    )
    assert (
        operator.public_signature.parameters["bias"].kind
        == inspect.Parameter.KEYWORD_ONLY
    )
    schema, _ = operator.torch_schema()
    assert schema.index("Tensor? bias") < schema.index("*")
    out_schema, _ = operator.torch_schema(out=True)
    assert "Tensor(a!) output" in out_schema and "Tensor(b!) output_scale" in out_schema


def test_generic_fake_supports_a_different_operator_with_a_single_output(monkeypatch):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from flashinfer.operators import Invocation, OutputSpec
    from flashinfer.operators import runtime
    from flashinfer.jit.recipe import ShapeConstraint, TensorParameter

    original = definition(NAME)
    tensors = (
        TensorParameter("x", "input", "float32", ("M",)),
        TensorParameter("y", "output", "float32", ("M",)),
    )
    operator = replace(
        original,
        name="single_output_probe",
        schema="(Tensor x) -> Tensor",
        output_names=("y",),
        op="gemm.single_output_probe",
        validate_values=None,
        infer=lambda a, layouts: Invocation(
            (a["x"],),
            (OutputSpec(tuple(a["x"].shape), "float32", a["x"].device),),
            {},
            None,
        ),
    )
    monkeypatch.setattr(runtime, "definition", lambda name: operator)
    monkeypatch.setattr(
        "flashinfer.ascendc.load_contract",
        lambda _: pytest.fail("fake cannot read a native contract"),
    )
    monkeypatch.setattr(
        runtime, "dispatch", lambda *a, **k: pytest.fail("fake cannot dispatch")
    )
    with FakeTensorMode():
        x = torch.empty(7)
        result = runtime.fake(operator.name, operator.bind(x))
        assert result.shape == (7,) and result.dtype == torch.float32
        assert runtime.fake(operator.name, operator.bind(x, out=result)) is None


def test_import_and_source_generation_work_without_torch_or_sdk():
    code = """
import sys
sys.modules['torch'] = None
sys.modules['torch_npu'] = None
import flashinfer
from flashinfer.operators.authoring import generate
assert callable(flashinfer.grouped_matmul_swiglu_quant)
assert generate(check=True)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
