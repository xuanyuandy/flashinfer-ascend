"""Targeted T0, validation and direct-Torch regressions for rich K mutation."""

import pytest
import torch

import flashinfer
from flashinfer.operators.attention.indexer_prologue_k import cases

pytestmark = [pytest.mark.real_compile,
              pytest.mark.requires_soc("Ascend910B", "Ascend910C"),
              pytest.mark.usefixtures("aot_runtime")]


def test_t0_is_a_noop_without_aot(npu_runtime, monkeypatch):
    _, _ = npu_runtime
    monkeypatch.setenv("FLASHINFER_AOT_PATHS", "/nonexistent-aot")
    monkeypatch.setenv("FLASHINFER_ACLNN_ADAPTER_PATH", "/nonexistent-adapter")
    latent = torch.zeros(0, 512, dtype=torch.bfloat16).npu()
    wk = torch.zeros(128, 512, dtype=torch.bfloat16).npu()
    norm = torch.ones(128, dtype=torch.bfloat16).npu()
    cos = torch.ones(0, 1, 1, 64).npu()
    sin = torch.zeros(0, 1, 1, 64).npu()
    indices = torch.zeros(0, 2, dtype=torch.int32).npu()
    key = torch.zeros(2, 2, 1, 128, dtype=torch.int8).npu()
    scale = torch.zeros(2, 2, 1, 1, dtype=torch.float16).npu()
    before = key.cpu().clone(), scale.cpu().clone()
    assert flashinfer.indexer_prologue_k(latent, wk, norm, cos, sin, indices, key, scale) is None
    assert torch.equal(key.cpu(), before[0]) and torch.equal(scale.cpu(), before[1])


@pytest.mark.parametrize("indices", [(-2, 1), (-1, 0), (0, -1), (99, 0)])
def test_invalid_slots_fail_before_dispatch(npu_runtime, indices):
    _, _ = npu_runtime
    cpu, _ = cases.make_inputs(torch, tokens=1)
    cpu["indices"] = torch.tensor([indices], dtype=torch.int32)
    with pytest.raises(ValueError, match="indices"):
        flashinfer.indexer_prologue_k(
            cpu["latent"].npu(), cpu["wk"].npu(), cpu["norm_weight"].npu(),
            cpu["rope_cos"].npu(), cpu["rope_sin"].npu(), cpu["indices"].npu(),
            cpu["k_cache"].npu(), cpu["k_scale_cache"].npu())


def test_direct_torch_t0_returns_none_without_aot(npu_runtime, monkeypatch):
    _, _ = npu_runtime
    from flashinfer.aclnn._ops import ensure_registered
    ensure_registered()
    monkeypatch.setenv("FLASHINFER_AOT_PATHS", "/nonexistent-direct-aot")
    monkeypatch.setenv("FLASHINFER_ACLNN_ADAPTER_PATH", "/nonexistent-direct-adapter")
    latent = torch.zeros(0, 512, dtype=torch.bfloat16).npu()
    wk = torch.zeros(128, 512, dtype=torch.bfloat16).npu()
    norm = torch.ones(128, dtype=torch.bfloat16).npu()
    cos = torch.ones(0, 1, 1, 64).npu()
    sin = torch.zeros(0, 1, 1, 64).npu()
    indices = torch.zeros(0, 2, dtype=torch.int32).npu()
    key = torch.zeros(2, 2, 1, 128, dtype=torch.int8).npu()
    scale = torch.zeros(2, 2, 1, 1, dtype=torch.float16).npu()
    assert torch.ops.flashinfer.indexer_prologue_k(
        latent, wk, norm, cos, sin, indices, key, scale, 1e-6) is None


@pytest.mark.parametrize("indices", [(-2, 1), (-1, 0), (0, -1), (99, 0)])
def test_direct_torch_rejects_invalid_slots(npu_runtime, indices):
    _, _ = npu_runtime
    from flashinfer.aclnn._ops import ensure_registered
    ensure_registered()
    cpu, _ = cases.make_inputs(torch, tokens=1)
    with pytest.raises(ValueError, match="indices"):
        torch.ops.flashinfer.indexer_prologue_k(
            cpu["latent"].npu(), cpu["wk"].npu(), cpu["norm_weight"].npu(),
            cpu["rope_cos"].npu(), cpu["rope_sin"].npu(),
            torch.tensor([indices], dtype=torch.int32).npu(),
            cpu["k_cache"].npu(), cpu["k_scale_cache"].npu(), 1e-6)


def test_direct_torch_rejects_graph_capture_before_reading_slots(npu_runtime, monkeypatch):
    torch, torch_npu = npu_runtime
    from flashinfer.aclnn._ops import ensure_registered
    ensure_registered()
    cpu, _ = cases.make_inputs(torch, tokens=1)
    monkeypatch.setattr(torch_npu.npu, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="NOT_CAPTURABLE"):
        torch.ops.flashinfer.indexer_prologue_k(
            cpu["latent"].npu(), cpu["wk"].npu(), cpu["norm_weight"].npu(),
            cpu["rope_cos"].npu(), cpu["rope_sin"].npu(), cpu["indices"].npu(),
            cpu["k_cache"].npu(), cpu["k_scale_cache"].npu(), 1e-6)
