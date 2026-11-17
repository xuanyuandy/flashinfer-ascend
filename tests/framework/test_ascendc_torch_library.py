"""T14 registration, metadata isolation, ABI bridges and real NPU tracing."""

import ctypes as C
from dataclasses import replace
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flashinfer._meta.ascendc import matmul_probe_meta
from flashinfer.jit import JitLoadedModule, JitModuleLoadReceipt, JitResolvedSymbol, JitSpec
from flashinfer.jit.core import _canonical_hash
from flashinfer.jit.op_package import OpPackageModule
from flashinfer.jit.torch_library import AscendCOp
from flashinfer.jit._legacy_torch_library import _MatmulProbeRunner, matmul_probe_op
from flashinfer.runtime import SchemaError
from flashinfer.utils import enable_torch_library
from tests.support.legacy_ascendc import ROOT, BuildRecipe, AscendCBuildConfig, load_build_recipes, export_op_package, resolve_build_recipe


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "(Tensor a, Tensor b, Tensor(a!) out) -> ()"


@pytest.fixture(autouse=True)
def current_package_in_legacy_children(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))


def forbidden(*args, **kwargs):
    raise AssertionError("metadata/setup reached device execution, allocation or compiler")


@lru_cache(maxsize=1)
def declaration():
    return AscendCOp("flashinfer::t14_metadata_probe", SCHEMA, forbidden, matmul_probe_meta, ("out",))


@pytest.fixture
def registered(monkeypatch):
    torch = pytest.importorskip("torch")
    import flashinfer.norm

    # Match T8: registration persists, frontend routing belongs to this test.
    monkeypatch.setattr(flashinfer.norm, "_torch_library_ops", None)
    enable_torch_library(ascendc_ops=(declaration(),))
    return torch, torch.ops.flashinfer.t14_metadata_probe.default


def test_import_and_declaration_are_torch_free():
    code = r'''
import importlib.abc, json, shutil, subprocess, sys
attempts, probes = [], []
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'triton', 'acl'}:
            attempts.append(fullname)
            raise AssertionError(fullname)
def forbidden(*args, **kwargs):
    probes.append(str(args))
    raise AssertionError('compiler/library probe')
sys.meta_path.insert(0, Guard())
shutil.which = forbidden
subprocess.Popen = forbidden
import ctypes
ctypes.CDLL = forbidden
import flashinfer
import flashinfer._torch_library
from flashinfer.jit.torch_library import AscendCOp
from flashinfer._meta.ascendc import matmul_probe_meta
op = AscendCOp('flashinfer::t14_import_probe', '(Tensor x) -> Tensor', forbidden, forbidden)
assert flashinfer.norm._torch_library_ops is None
assert attempts == [] and probes == []
print(json.dumps(dict(provider_import_attempts=attempts, compiler_or_library_probes=probes,
                      registration_enabled=False)))
'''
    result = subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    print(result.stdout.strip())
    assert json.loads(result.stdout) == dict(provider_import_attempts=[], compiler_or_library_probes=[],
                                           registration_enabled=False)


def test_setup_without_optional_providers_or_compilers():
    code = r'''
import torch
import importlib.abc, json, shutil, subprocess, sys
assert 'torch_npu' not in sys.modules and 'triton' not in sys.modules
attempts = []
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch_npu', 'triton', 'acl'}:
            attempts.append(fullname)
            raise AssertionError(fullname)
def forbidden(*args, **kwargs):
    raise AssertionError('setup probed a compiler or called a device implementation')
sys.meta_path.insert(0, Guard())
shutil.which = forbidden
subprocess.Popen = forbidden
from flashinfer.jit.torch_library import AscendCOp
from flashinfer._meta.ascendc import matmul_probe_meta
from flashinfer.utils import enable_torch_library
op = AscendCOp('flashinfer::t14_setup_probe', '(Tensor a, Tensor b, Tensor(a!) out) -> ()',
              forbidden, matmul_probe_meta, ('out',))
enable_torch_library(ascendc_ops=(op,))
assert torch._C._dispatch_has_kernel_for_dispatch_key(op.name, 'PrivateUse1')
assert attempts == [] and 'torch_npu' not in sys.modules and 'triton' not in sys.modules
print('PASS explicit setup; PrivateUse1=True torch_npu=False triton=False compiler_probes=0')
'''
    pytest.importorskip("torch")
    environment = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())


@pytest.mark.parametrize("name", ["_C_ascend::op", "other::op", "flashinfer::op_vllm",
                                 "flashinfer::op__c_ascend", "flashinfer::rmsnorm"])
def test_product_or_reserved_names_are_rejected_before_registration(name):
    with pytest.raises(ValueError):
        AscendCOp(name, SCHEMA, forbidden, matmul_probe_meta, ("out",))


def test_setup_is_idempotent_and_only_installs_privateuse1_device_kernel(registered):
    torch, op = registered
    enable_torch_library(ascendc_ops=(declaration(),))
    assert torch.ops.flashinfer.t14_metadata_probe.default is op
    for key in ("PrivateUse1", "Meta"):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(declaration().name, key)
    for key in ("CPU", "CUDA", "CompositeExplicitAutograd"):
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(declaration().name, key)
    assert op._schema.returns == []
    assert {arg.name for arg in op._schema.arguments if arg.alias_info and arg.alias_info.is_write} == {"out"}
    x = torch.empty(16, 16)
    with pytest.raises(NotImplementedError):
        op(x, x, torch.empty_like(x))
    with pytest.raises(ValueError, match="fresh process"):
        enable_torch_library(ascendc_ops=(replace(declaration(), device_impl=lambda *args: None),))


@pytest.mark.parametrize("attribute", ["custom_op", "register_fake"])
def test_even_repeated_enable_rejects_unsupported_torch(registered, monkeypatch, attribute):
    torch, _ = registered
    monkeypatch.delattr(torch.library, attribute)
    with pytest.raises(RuntimeError, match="torch >= 2.4"):
        enable_torch_library(ascendc_ops=(declaration(),))


@pytest.mark.parametrize("schema,mutations", [
    (SCHEMA, ()), ("(Tensor a) -> Tensor(a)", ()), ("(int x) -> int", ()), ("invalid", ()),
])
def test_bad_schema_batch_does_not_partially_register(schema, mutations):
    torch = pytest.importorskip("torch")
    before = set(torch._C._dispatch_get_all_op_names())
    good = AscendCOp("flashinfer::t14_batch_good", SCHEMA, forbidden, matmul_probe_meta, ("out",))
    with pytest.raises((ValueError, RuntimeError)):
        bad = AscendCOp("flashinfer::t14_batch_bad", schema, forbidden, matmul_probe_meta, mutations)
        enable_torch_library(ascendc_ops=(good, bad))
    assert set(torch._C._dispatch_get_all_op_names()) == before
    with pytest.raises(ValueError, match="duplicate"):
        enable_torch_library(ascendc_ops=(good, good))
    assert set(torch._C._dispatch_get_all_op_names()) == before


def test_meta_has_no_tensor_operations_allocations_or_data_access(registered, monkeypatch):
    torch, _ = registered
    from torch.utils._python_dispatch import TorchDispatchMode

    tensors = [torch.empty(16, 16, device="meta") for _ in range(3)]

    class NoTensorOperations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            raise AssertionError("meta performed tensor computation/allocation: " + str(func))

    for name in ("empty", "empty_like", "zeros", "ones", "tensor", "matmul", "mm"):
        monkeypatch.setattr(torch, name, forbidden)
    for name in ("item", "data_ptr", "cpu", "numpy", "tolist", "storage", "untyped_storage"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    monkeypatch.setattr(C, "CDLL", forbidden)
    with NoTensorOperations():
        for _ in range(100):
            assert matmul_probe_meta(*tensors) is None


@pytest.mark.parametrize("bad", ["shape", "rank", "dtype", "stride", "device"])
def test_fake_meta_rejects_unsupported_metadata(registered, bad):
    torch, op = registered
    x = torch.empty(16, 16, device="meta")
    output = {
        "shape": lambda: torch.empty(16, 32, device="meta"),
        "rank": lambda: torch.empty(256, device="meta"),
        "dtype": lambda: torch.empty(16, 16, device="meta", dtype=torch.float16),
        "stride": lambda: x.t(),
        "device": lambda: torch.empty(16, 16),
    }[bad]()
    with pytest.raises(RuntimeError, match="matmul_probe requires"):
        op(x, x, output)


def test_fake_cpu_preserves_metadata_and_never_runs_privateuse1_kernel(registered):
    torch, op = registered
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(allow_fallback_kernels=False):
        a, b, out = [torch.empty(16, 16) for _ in range(3)]
        for _ in range(100):
            assert op(a, b, out) is None
            assert (out.shape, out.dtype, out.stride(), out.device.type) == (
                (16, 16), torch.float32, (16, 1), "cpu")


@pytest.fixture
def recipes():
    return load_build_recipes(ROOT / "csrc/recipes/matmul_probe.json")


def fixture_module(recipe):
    """Synthetic receipt for setup validation only; never execute this module."""
    bindings = (("build_recipe", recipe.fingerprint),)
    if recipe.output_kind == "op_package":
        bindings += (("output_adapter", _canonical_hash(recipe.output_adapter)),)
    spec = JitSpec("t14.fixture", "testing", "ascendc", "2", recipe.target_soc, "a" * 64, "b" * 64,
        kernel_id=recipe.kernel_id, source=recipe.source,
        output_path="probe.so" if recipe.output_kind == "host_shared" else "op-package.json",
        entry_points=(recipe.host_entry,), input_fingerprints=bindings)
    receipt = JitModuleLoadReceipt(spec.name, spec.fingerprint, "c" * 64, "d" * 64, "e" * 64,
        "test.loader", "1", "fixture", 1, (JitResolvedSymbol(recipe.host_entry, "fixture"),))
    opaque = (SimpleNamespace(**{recipe.host_entry: lambda *args: None}) if recipe.output_kind == "host_shared"
              else OpPackageModule(recipe, Path("/unused"), object()))
    return spec, JitLoadedModule(receipt, opaque)


@pytest.mark.parametrize("implementation,kind", [("native", "host_shared"), ("catlass", "host_shared"),
                                                ("native", "op_package"), ("catlass", "op_package")])
def test_four_recipe_variants_share_one_schema_and_meta(recipes, implementation, kind, monkeypatch):
    recipe = next(r for r in recipes if (r.implementation, r.output_kind) == (implementation, kind))
    spec, loaded = fixture_module(recipe)
    monkeypatch.setattr(C, "CDLL", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    op = matmul_probe_op(recipe, loaded, spec=spec)
    assert op.name == "flashinfer::matmul_probe" and op.schema == SCHEMA
    assert op.meta_impl is matmul_probe_meta and op.mutates_args == ("out",)
    if kind == "host_shared":
        assert op.device_impl.entry.argtypes == [C.c_void_p] * 4
    else:
        assert matmul_probe_op(recipe, loaded.opaque_module).schema == SCHEMA


@pytest.mark.parametrize("mismatch", ["spec", "recipe", "entry", "raw_host", "package"])
def test_module_recipe_and_abi_mismatches_are_rejected(recipes, mismatch):
    recipe = recipes[0]
    spec, loaded = fixture_module(recipe)
    if mismatch == "spec":
        spec = replace(spec, specialization_fingerprint="f" * 64)
    elif mismatch == "recipe":
        recipe = recipes[1]
    elif mismatch == "entry":
        loaded = replace(loaded, receipt=replace(loaded.receipt, symbols=(JitResolvedSymbol("wrong", "fixture"),)))
    elif mismatch == "raw_host":
        loaded = loaded.opaque_module
    else:
        recipe = next(r for r in recipes if r.output_kind == "op_package")
        loaded = OpPackageModule(replace(recipe, version=2), Path("/unused"), object())
    with pytest.raises(SchemaError):
        matmul_probe_op(recipe, loaded, spec=spec)


@pytest.mark.parametrize("failure", [None, "create", "workspace", "launch"])
def test_aclnn_bridge_releases_descriptors_after_sync_and_propagates_failures(monkeypatch, failure):
    pytest.importorskip("torch")
    from flashinfer.jit import _legacy_torch_library as torch_library

    calls = []
    def create(*args):
        calls.append("create")
        return None if failure == "create" and calls.count("create") == 2 else calls.count("create")
    def destroy(descriptor):
        calls.append(("destroy", descriptor))
        return 0
    def workspace(*descriptors):
        calls.append(("workspace", descriptors))
        if failure == "workspace":
            raise RuntimeError("workspace failure")
        return 8, C.c_void_p(123)
    def launch(pointer, size, executor, stream):
        calls.append("launch")
        assert pointer and size == 8 and executor.value == 123 and stream == 7
        if failure == "launch":
            raise RuntimeError("launch failure")
    monkeypatch.setattr(torch_library, "_acl_meta_api", lambda: SimpleNamespace(
        aclCreateTensor=create, aclDestroyTensor=destroy))
    runner = _MatmulProbeRunner(SimpleNamespace(get_workspace_size=workspace, launch=launch), None)
    stream = SimpleNamespace(npu_stream=7, synchronize=lambda: calls.append("sync"))
    if failure:
        with pytest.raises(RuntimeError, match="failed|failure"):
            runner._launch_package([1, 2, 3], "cpu", stream)
    else:
        runner._launch_package([1, 2, 3], "cpu", stream)
    expected = [1] if failure == "create" else [3, 2, 1]
    assert calls[calls.index("sync") + 1:] == [("destroy", d) for d in expected]


def test_new_extensions_have_refreshable_parity_entries():
    from flashinfer.parity import load_packaged_manifest
    from tools.update_parity_inventory import annotate

    entries = {e.upstream: e.to_dict() for e in load_packaged_manifest("all").entries}
    source = SimpleNamespace(resolve=lambda symbol: None)
    for name in ("AscendCOp", "matmul_probe_op"):
        entry = entries["flashinfer.jit.torch_library." + name]
        refreshed = annotate(entry, source)
        assert refreshed["api_kind"] == "npu_extension"
        assert refreshed["semantic_status"] == "intentionally_different"
        assert refreshed["reason"]


@pytest.mark.real_compile
@pytest.mark.parametrize("kind", ["host_shared", "op_package"])
@pytest.mark.requires_soc("Ascend950")
def test_real_native_artifact_compile_fake_memory_namespace_and_positive_control(tmp_path, kind):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires NPU for T14 compile and memory evidence")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/absent"))
    if not (cann / "bin/bisheng").is_file():
        pytest.skip("requires CANN compiler to prepare a fresh native T13 artifact")
    environment = dict(os.environ, TORCH_LOGS="graph_breaks")
    script = [sys.executable, "tools/t14_torch_library_smoke.py"]
    if kind == "host_shared":
        script += ["--config", "examples/ascendc_native.json", "--cache-dir", str(tmp_path / "cache"),
                   "--catlass-root", str(tmp_path / "absent-catlass")]
    else:
        build = subprocess.run([sys.executable, "-m", "tests.support.legacy_ascendc", "--config",
            "examples/ascendc_aot_native.json", "--soc", "ascend950", "--cache-dir", str(tmp_path / "cache"),
            "--output-dir", str(tmp_path / "release")], cwd=ROOT, capture_output=True, text=True, timeout=180)
        assert build.returncode == 0, build.stdout + build.stderr
        releases = list((tmp_path / "release").glob("*/op-package.json"))
        assert len(releases) == 1
        script += ["--package", str(releases[0].parent), "--soc", "ascend950"]
    result = subprocess.run(script, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"COMPILE PASS output={kind}" in result.stdout
    assert "backend=inductor fullgraph=True graph_count=1 graph_break_count=0 break_reasons=[]" in result.stdout
    assert "calls=100 before=(0, 0, 0, 0) after=(0, 0, 0, 0)" in result.stdout
    assert "NAMESPACE PASS" in result.stdout and "PrivateUse1=True Meta=True CPU=False" in result.stdout
    assert "POSITIVE_CONTROL PASS graph_count=2 graph_break_count=1 break_reasons=1 fullgraph_rejected=True" in result.stdout
    # Only the positive-control interval may emit graph-break diagnostics.
    ordinary, control = result.stderr.split("GRAPH_BREAK_POSITIVE_CONTROL_BEGIN\n")
    control, trailing = control.split("GRAPH_BREAK_POSITIVE_CONTROL_END\n")
    assert "Graph break in user code" not in ordinary + trailing
    assert "Graph break in user code" in control
    print(result.stdout.strip())
