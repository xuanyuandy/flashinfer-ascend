"""Shared Matmul metadata, allocation, dispatch and compilation-policy contracts."""

from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from flashinfer import gemm
from flashinfer._matmul import AOT_KERNEL_ID, CANN_KERNEL_ID, TRITON_KERNEL_ID, contract
from flashinfer.runtime import (
    Backend, CompileMode, DispatchError, GraphCapability, HardwareSignature,
    SchemaError, get_global_registry,
)


BACKENDS = ("cann", "ascendc", "triton")
IDS = dict(zip(BACKENDS, (CANN_KERNEL_ID, AOT_KERNEL_ID, TRITON_KERNEL_ID)))


def forbidden(*args, **kwargs):
    raise AssertionError("invalid metadata reached allocation, compilation or execution")


@pytest.fixture
def host_dispatch(monkeypatch):
    original = gemm.build_signature

    def signature(op, tensors, **kwargs):
        tensors = tuple(tensors)
        return replace(original(op, tensors, **kwargs), physical_layouts=("ND",) * len(tensors))

    monkeypatch.setattr(gemm, "build_signature", signature)
    monkeypatch.setattr(gemm, "current_hardware", lambda: HardwareSignature("Ascend950", 32, 262144))
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "0")
    monkeypatch.setenv("FLASHINFER_NPU_TUNING_DB", "")


def test_exact_three_backends_share_capability_and_truthful_graph_metadata():
    descriptors = get_global_registry().descriptors("gemm.matmul")
    assert {d.kernel_id for d in descriptors} == set(IDS.values())
    assert {d.backend for d in descriptors} == {Backend.CANN, Backend.ASCENDC, Backend.TRITON}
    assert all(d.capability == contract()[0].capability for d in descriptors)
    assert all(d.graph_capability == GraphCapability.NOT_CAPTURABLE for d in descriptors)
    assert get_global_registry().get(AOT_KERNEL_ID).compile_mode == CompileMode.AOT
    assert get_global_registry().get(TRITON_KERNEL_ID).compile_mode == CompileMode.JIT
    assert get_global_registry().get(CANN_KERNEL_ID).compile_mode == CompileMode.NONE
    assert contract()[1].tensor_parameter_names == ("a", "b", "out")
    assert contract()[1].allow_jit_fallback is False
    assert get_global_registry().get(AOT_KERNEL_ID).runner.jit_backend_factory is None


def test_import_and_registration_need_no_optional_provider_or_compiler():
    code = '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'triton', 'acl'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
import flashinfer
from flashinfer.runtime import get_global_registry
assert len(get_global_registry().descriptors('gemm.matmul')) == 3
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("supply_output", [False, True])
def test_common_allocation_and_original_tensor_objects(host_dispatch, monkeypatch, backend, supply_output):
    import torch

    a = torch.empty((32, 48), dtype=torch.bfloat16)
    b = torch.empty((48, 64), dtype=torch.bfloat16)
    provided = torch.empty((32, 64), dtype=a.dtype) if supply_output else None
    allocations, launches = [], []
    original_empty = torch.empty

    def allocate(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        allocations.append((args, kwargs, result))
        return result

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(gemm, "_run_selected", lambda kid, *args: launches.append((kid, args)))
    result = gemm._matmul_impl(a, b, provided, backend=backend)
    assert len(allocations) == (0 if supply_output else 1)
    assert result.shape == (32, 64) and result.dtype == a.dtype and result.is_contiguous()
    if supply_output:
        assert result is provided
    else:
        assert result is allocations[0][2]
        assert allocations[0][1] == dict(dtype=a.dtype, device=a.device)
    assert launches[0][0] == IDS[backend]
    assert all(actual is expected for actual, expected in zip(launches[0][1], (a, b, result)))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case,diagnostic", [
    ("rank", "rank"), ("contraction", "inconsistent"), ("alignment", "multiple_of=16"),
    ("upper_bound", "M=272"), ("empty", "positive integer dimensions"), ("dtype", "dtype"),
    ("mixed_dtype", "dtype"), ("noncontiguous", "contiguous"),
    ("out_shape", "inconsistent"), ("out_dtype", "dtype for out"),
])
def test_invalid_inputs_fail_before_allocation_or_runner(host_dispatch, monkeypatch, backend, case, diagnostic):
    import torch

    a, b, out = (torch.empty(shape, dtype=torch.bfloat16) for shape in
                 ((32, 48), (48, 64), (32, 64)))
    if case == "rank":
        a = a.unsqueeze(0)
    elif case == "contraction":
        b = torch.empty((32, 64), dtype=b.dtype)
    elif case == "alignment":
        a = a[:17]
    elif case == "upper_bound":
        a = torch.empty((272, 48), dtype=a.dtype)
    elif case == "empty":
        a = a[:0]
    elif case == "dtype":
        a, b = a.float(), b.float()
    elif case == "mixed_dtype":
        b = b.half()
    elif case == "noncontiguous":
        a = a.t()
        b = torch.empty((32, 64), dtype=b.dtype)
    elif case == "out_shape":
        out = torch.empty((32, 32), dtype=out.dtype)
    elif case == "out_dtype":
        out = out.float()
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(gemm, "_run_selected", forbidden)
    with pytest.raises(DispatchError, match=diagnostic):
        gemm._matmul_impl(a, b, out if case.startswith("out_") else None, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_output_overlap_rejected_for_every_backend(host_dispatch, monkeypatch, backend):
    import torch

    storage = torch.empty(1024, dtype=torch.bfloat16)
    a = storage[:256].view(16, 16)
    out = storage[128:384].view(16, 16)
    b = torch.empty_like(a)
    monkeypatch.setattr(gemm, "_run_selected", forbidden)
    with pytest.raises(ValueError, match="must not overlap"):
        gemm._matmul_impl(a, b, out, backend=backend)


def test_auto_order_global_default_and_forced_backend(host_dispatch, monkeypatch):
    import torch

    a = torch.empty(16, 16, dtype=torch.bfloat16)
    b, out = torch.empty_like(a), torch.empty_like(a)
    launches = []
    monkeypatch.setattr(gemm, "_run_selected", lambda kid, *args: launches.append(kid))
    monkeypatch.setenv("FLASHINFER_NPU_BACKEND", "ascendc")
    gemm._matmul_impl(a, b, out)
    gemm._matmul_impl(a, b, out, backend="auto")
    gemm._matmul_impl(a, b, out, backend=Backend.TRITON)
    assert launches == [AOT_KERNEL_ID, CANN_KERNEL_ID, TRITON_KERNEL_ID]
    monkeypatch.setenv("FLASHINFER_NPU_DISPATCH_ORDER", "jit,aot,cann,external,reference")
    gemm._matmul_impl(a, b, out, backend="auto")
    assert launches[-1] == TRITON_KERNEL_ID
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")
    with pytest.raises(DispatchError, match="JIT"):
        gemm._matmul_impl(a, b, out, backend="triton")


def test_aot_only_policy_roundtrip_preserves_existing_recipe_identity():
    from flashinfer.jit import BuildRecipe
    from flashinfer.ascendc import load_contract

    _, recipe, _ = contract()
    assert BuildRecipe.from_dict(recipe.to_dict()) == recipe
    legacy_default = load_contract()[1]
    assert "allow_jit_fallback" not in legacy_default.to_dict()
    assert replace(legacy_default, allow_jit_fallback=True).fingerprint == legacy_default.fingerprint
    with pytest.raises(SchemaError, match="boolean"):
        replace(recipe, allow_jit_fallback=0)


def test_triton_declaration_binds_source_options_and_descriptor():
    runner = get_global_registry().get(TRITON_KERNEL_ID).runner
    descriptor = runner.descriptor
    path = Path(__file__).resolve().parents[2] / "flashinfer" / descriptor.source.locator
    assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor.source.digest
    spec = runner.spec()
    assert spec.kernel_id == TRITON_KERNEL_ID
    assert spec.source == descriptor.source
    assert dict(spec.input_fingerprints)["kernel"] == descriptor.fingerprint
    assert 'binary={"mode":"mix"}' in spec.compile_options


def test_auto_uses_valid_tuning_record_but_forced_backend_stays_authoritative(
        host_dispatch, monkeypatch, tmp_path):
    import torch
    from unittest.mock import patch
    from flashinfer.runtime import build_signature, dispatch
    from flashinfer.tuning import (
        BenchmarkCandidate, BenchmarkConfig, collect_tuning_record, reload_tuning_records,
        save_tuning_record,
    )

    monkeypatch.setenv("FLASHINFER_NPU_TUNING_DB", str(tmp_path / "synthetic-routing.sqlite3"))
    monkeypatch.setenv("FLASHINFER_NPU_DISPATCH_ORDER", "external,cann,aot,jit,reference")
    a, b, out = (torch.empty(shape, dtype=torch.bfloat16) for shape in ((32, 48), (48, 64), (32, 64)))
    signature = replace(build_signature("gemm.matmul", (a, b, out),
                                       specializations={"tensor_contract": "supported"}),
                        physical_layouts=("ND",) * 3)
    hardware = HardwareSignature("Ascend950", 32, 262144)
    registry = get_global_registry()
    current, ticks = [CANN_KERNEL_ID], [0]

    def candidate(key):
        def prepare(iterations):
            current[0] = key
            return lambda: None
        return BenchmarkCandidate(registry.get(key), prepare)

    def clock():
        ticks[0] += 1
        return 0 if ticks[0] % 2 else (50_000 if current[0] == TRITON_KERNEL_ID else 100_000)

    try:
        reload_tuning_records()
        assert dispatch(signature, hardware).kernel_id == CANN_KERNEL_ID
        # Synthetic host clocks validate record consumption only. These are
        # never measurements of any backend's performance.
        with patch("flashinfer.tuning.benchmark.time.perf_counter_ns", clock), patch(
                "flashinfer.tuning.benchmark.time.sleep", lambda _: None):
            record = collect_tuning_record(
                signature, hardware, [candidate(key) for key in IDS.values()],
                synchronize=lambda: None, config=BenchmarkConfig(warmup=1, rounds=25, iterations=1),
                exact_shapes=True,
                metadata=dict(fixture="SYNTHETIC ROUTING UNIT TEST ONLY; NOT PERFORMANCE EVIDENCE"),
            )
        save_tuning_record(record)
        assert record["result"]["winner"] == TRITON_KERNEL_ID
        decision = dispatch(signature, hardware)
        assert decision.kernel_id == TRITON_KERNEL_ID, decision.reason
        launches = []
        monkeypatch.setattr(gemm, "_run_selected", lambda kid, *args: launches.append(kid))
        gemm._matmul_impl(a, b, out, backend="auto")
        gemm._matmul_impl(a, b, out, backend="cann")
        gemm._matmul_impl(a, b, out, backend="ascendc")
        assert launches == [TRITON_KERNEL_ID, CANN_KERNEL_ID, AOT_KERNEL_ID]
    finally:
        reload_tuning_records()


def test_torch_functional_out_and_fake_metadata(monkeypatch):
    import torch
    import flashinfer
    from torch._subclasses.fake_tensor import FakeTensorMode
    from flashinfer.aclnn._ops import ensure_registered

    ensure_registered()
    monkeypatch.setattr(gemm, "_run_selected", forbidden)
    with FakeTensorMode():
        a = torch.empty((32, 48), dtype=torch.bfloat16)
        b = torch.empty((48, 64), dtype=a.dtype)
        out = torch.empty((32, 64), dtype=a.dtype)
        for backend in BACKENDS:
            result = flashinfer.matmul(a, b, backend=backend)
            assert result.shape == out.shape and result.dtype == a.dtype
            assert flashinfer.matmul(a, b, out=out, backend=backend) is out
        with pytest.raises(SchemaError, match="inconsistent"):
            flashinfer.matmul(a, b.t(), backend="triton")
