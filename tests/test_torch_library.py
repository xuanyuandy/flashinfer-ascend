"""T8 registration, metadata-only execution, and compile contracts."""

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import flashinfer
from flashinfer.utils import enable_torch_library, register_custom_op, register_fake_op


@pytest.fixture
def registered(monkeypatch):
    torch = pytest.importorskip("torch")
    # Registration is process-wide, but frontend routing is scoped to this test.
    monkeypatch.setattr(flashinfer.norm, "_torch_library_ops", None)
    enable_torch_library()
    return torch


def test_upstream_helper_signatures():
    custom = inspect.signature(register_custom_op)
    assert str(custom) == (
        "(name: str, fn: Optional[Callable] = None, /, *, "
        "mutates_args: Union[str, Iterable[str]], "
        "device_types: Union[str, Sequence[str], NoneType] = None, "
        "schema: Optional[str] = None) -> Callable"
    )
    assert str(inspect.signature(register_fake_op)) == (
        "(name: str, fn: Optional[Callable] = None) -> Callable"
    )


def test_parity_refresh_tracks_version_guarded_helpers_and_setup(monkeypatch):
    import ast
    from flashinfer.parity import load_packaged_manifest
    from tools.update_parity_inventory import Upstream, annotate

    # The upstream helpers live under a torch-version if/else. Parse their
    # declarations without importing torch or the upstream CUDA package.
    text = (
        "if old_torch:\n"
        "    def register_custom_op(): pass\n"
        "    def register_fake_op(): pass\n"
        "else:\n"
        "    def register_custom_op(): pass\n"
        "    def register_fake_op(): pass\n"
    )
    source = object.__new__(Upstream)
    monkeypatch.setattr(source, "module", lambda name: (
        ("flashinfer/utils.py", text, ast.parse(text)) if name == "flashinfer.utils" else None
    ))
    entries = {e.upstream: e.to_dict() for e in load_packaged_manifest("all").entries}
    for name, line in [("register_custom_op", 5), ("register_fake_op", 6)]:
        symbol = "flashinfer.utils." + name
        refreshed = annotate(entries[symbol], source)
        assert refreshed["api_kind"] == "upstream"
        assert refreshed["semantic_status"] == "intentionally_different"
        assert refreshed["upstream_source"]["declaration"] == symbol
        assert refreshed["upstream_source"]["line"] == line
    setup = annotate(entries["flashinfer.utils.enable_torch_library"], source)
    assert setup["api_kind"] == "npu_extension"
    assert setup["semantic_status"] == "intentionally_different" and setup["reason"]


def test_import_is_pure_and_registration_is_opt_in():
    code = r'''
import importlib.abc
import json
import shutil
import subprocess
import sys
attempts = []
probes = []
class DenyProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'triton', 'acl'}:
            attempts.append(fullname)
            raise AssertionError('provider import: ' + fullname)
def forbidden(*args, **kwargs):
    probes.append(str(args))
    raise AssertionError('compiler discovery or subprocess')
sys.meta_path.insert(0, DenyProviders())
shutil.which = forbidden
subprocess.Popen = forbidden
import flashinfer
import flashinfer.utils
import flashinfer._torch_library
assert flashinfer.norm._torch_library_ops is None
assert attempts == []
assert probes == []
assert not any(n.split('.')[0] in {'torch', 'torch_npu', 'triton', 'acl'} for n in sys.modules)
print(json.dumps({'provider_import_attempts': attempts, 'compiler_probes': probes,
                  'registration_enabled': False}))
'''
    result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True,
                            text=True, check=True, cwd=Path(__file__).resolve().parents[1])
    print(result.stdout.strip())
    assert json.loads(result.stdout) == {
        "provider_import_attempts": [], "compiler_probes": [],
        "registration_enabled": False,
    }


@pytest.mark.parametrize("form", ["decorator", "direct"])
def test_helpers_really_register_and_forward_options(form):
    torch = pytest.importorskip("torch")
    name = "flashinfer_t08_tests::" + form

    def impl(x):
        return x + 1

    def meta(x):
        return x.new_empty(x.shape)

    options = dict(mutates_args=(), device_types="cpu", schema="(Tensor x) -> Tensor")
    if form == "decorator":
        op = register_custom_op(name, **options)(impl)
        assert register_fake_op(name)(meta) is meta
    else:
        op = register_custom_op(name, impl, **options)
        assert register_fake_op(name=name, fn=meta) is meta
    torch.testing.assert_close(op(torch.ones(2, 4)), torch.full((2, 4), 2.0))
    result = op(torch.empty(2, 4, device="meta"))
    assert result.device.type == "meta" and result.shape == (2, 4)


@pytest.mark.parametrize("helper, attribute, kwargs", [
    (register_custom_op, "custom_op", {"mutates_args": ()}),
    (register_fake_op, "register_fake", {}),
])
def test_unsupported_torch_fails_instead_of_silently_disabling(monkeypatch, helper, attribute, kwargs):
    torch = pytest.importorskip("torch")
    monkeypatch.delattr(torch.library, attribute)
    with pytest.raises(RuntimeError, match="torch >= 2.4"):
        helper("flashinfer_t08_tests::unsupported", **kwargs)


def test_setup_is_idempotent_and_schemas_declare_mutations(registered):
    torch = registered
    first = flashinfer.norm._torch_library_ops
    enable_torch_library()
    assert flashinfer.norm._torch_library_ops is first
    for name, mutated in [("rmsnorm", {"out"}), ("fused_add_rmsnorm", {"input", "residual"})]:
        schema = getattr(torch.ops.flashinfer, name).default._schema
        assert schema.returns == []
        assert {arg.name for arg in schema.arguments
                if arg.alias_info is not None and arg.alias_info.is_write} == mutated


def _forbidden(*args, **kwargs):
    raise AssertionError("meta/fake execution attempted runtime dispatch or tensor computation")


def _forbid_runtime(monkeypatch):
    for name in ("_rmsnorm_impl", "_fused_add_rmsnorm_impl", "build_signature",
                 "current_hardware", "dispatch", "_run_selected", "get_global_registry"):
        monkeypatch.setattr(flashinfer.norm, name, _forbidden)


def test_meta_bodies_do_not_even_access_tensor_arguments(registered, monkeypatch):
    from flashinfer._torch_library import _fused_add_rmsnorm_meta, _rmsnorm_meta
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoTensorAccess:
        def __getattribute__(self, name):
            raise AssertionError("meta accessed tensor data or metadata: " + name)

    class NoTensorOperations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            raise AssertionError("meta executed a tensor operation: " + str(func))

    _forbid_runtime(monkeypatch)
    for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like", "tensor"):
        monkeypatch.setattr(registered, name, _forbidden)
    sentinel = NoTensorAccess()
    with NoTensorOperations():
        assert _rmsnorm_meta(sentinel, sentinel, sentinel, 1e-6, None, None) is None
        assert _fused_add_rmsnorm_meta(sentinel, sentinel, sentinel, 1e-6, None, None) is None


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("shape", [(4, 128), (2, 3, 128), (0, 128)])
def test_public_meta_shapes_dtypes_and_out_identity(registered, monkeypatch, dtype, shape):
    torch = registered
    _forbid_runtime(monkeypatch)
    x = torch.empty(shape, dtype=getattr(torch, dtype), device="meta")
    weight = torch.empty(shape[-1], dtype=x.dtype, device="meta")
    residual = torch.empty_like(x)
    result = flashinfer.rmsnorm(x, weight)
    assert (result.shape, result.dtype, result.device, result.stride()) == (
        x.shape, x.dtype, x.device, x.stride(),
    )
    out = torch.empty_like(x)
    assert flashinfer.rmsnorm(x, weight, out=out) is out
    assert flashinfer.fused_add_rmsnorm(x, residual, weight) is None
    assert residual.shape == x.shape


def test_fake_cpu_preserves_strides_without_running_kernels(registered, monkeypatch):
    torch = registered
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    _forbid_runtime(monkeypatch)
    with FakeTensorMode(allow_fallback_kernels=False):
        x = torch.empty(128, 4).t()
        residual = torch.empty_like(x)
        weight = torch.empty(128)
        output = flashinfer.rmsnorm(x, weight)
        assert isinstance(output, FakeTensor)
        assert output.shape == x.shape and output.stride() == x.stride()
        assert flashinfer.fused_add_rmsnorm(x, residual, weight) is None


def test_fake_npu_allocates_zero_device_memory(registered, monkeypatch):
    torch = registered
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires NPU memory counters")
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    torch.npu.init()
    _forbid_runtime(monkeypatch)

    def counters():
        return (torch.npu.memory_allocated(), torch.npu.memory_reserved(),
                torch.npu.max_memory_allocated(), torch.npu.max_memory_reserved())

    before = counters()
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        with FakeTensorMode(allow_fallback_kernels=False):
            x = torch.empty(1024, 4096, device="npu", dtype=dtype)
            weight = torch.empty(4096, device="npu", dtype=dtype)
            residual = torch.empty_like(x)
            output = flashinfer.rmsnorm(x, weight)
            assert isinstance(output, FakeTensor) and output.device.type == "npu"
            assert flashinfer.rmsnorm(x, weight, out=output) is output
            assert flashinfer.fused_add_rmsnorm(x, residual, weight) is None
    after = counters()
    print(f"NPU memory (allocated, reserved, peak_allocated, peak_reserved): before={before} after={after}")
    assert after == before


@pytest.mark.parametrize("name", ["rmsnorm", "fused_add_rmsnorm"])
def test_opcheck_mutation_and_fake_contract(registered, name):
    torch = registered
    x = torch.randn(4, 128)
    weight = torch.randn(128)
    args = ((torch.empty_like(x), x, weight) if name == "rmsnorm"
            else (x, torch.randn_like(x), weight))
    result = torch.library.opcheck(
        getattr(torch.ops.flashinfer, name).default,
        (*args, 1e-5, "reference", None),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert set(result.values()) == {"SUCCESS"}


def test_compile_fullgraph_dynamic_cpu_preserves_values_mutations_and_out(registered):
    torch = registered
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def forward(x, residual, weight, out):
        y = flashinfer.rmsnorm(x, weight, eps=1e-5, out=out, backend=flashinfer.Backend.REFERENCE)
        flashinfer.fused_add_rmsnorm(x, residual, weight, eps=1e-5, backend="reference")
        return y, x, residual

    compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=True)
    for rows in (4, 7):
        x, residual, weight = torch.randn(rows, 128), torch.randn(rows, 128), torch.randn(128)
        expected = forward(x.clone(), residual.clone(), weight, torch.empty_like(x))
        out = torch.empty_like(x)
        actual = compiled(x.clone(), residual.clone(), weight, out)
        assert actual[0] is out
        for got, wanted in zip(actual, expected):
            torch.testing.assert_close(got, wanted)
    assert len(graphs) == 1
    targets = [str(node.target) for node in graphs[0].graph.nodes]
    assert any("flashinfer.rmsnorm" in target for target in targets)
    assert any("flashinfer.fused_add_rmsnorm" in target for target in targets)


@pytest.mark.real_compile
def test_compile_fullgraph_inductor_npu_with_graph_break_logging(registered):
    torch = registered
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires NPU torch.compile")
    root = Path(__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("FLASHINFER_NPU_")}
    environment["TORCH_LOGS"] = "graph_breaks"
    result = subprocess.run(
        [sys.executable, "tools/t08_torch_compile_smoke.py"], cwd=root,
        capture_output=True, text=True, env=environment, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Graph break in user code" not in result.stderr, result.stderr
    assert result.stdout.count("graph_break_count=0 break_reasons=[]") == 2
    print(result.stdout.strip())
