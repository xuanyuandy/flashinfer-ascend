"""api compatibility regression coverage."""


import pytest

import flashinfer as finpu
from flashinfer.runtime.schema import DispatchError


def test_top_level_import_surface():
    assert finpu.__version__
    for name in (
        "single_prefill_with_kv_cache",
        "single_decode_with_kv_cache",
        "BatchDecodeWithPagedKVCacheWrapper",
        "dispatch",
        "explain_dispatch",
        "register_custom_kernel",
        "build_cache_key",
    ):
        assert callable(getattr(finpu, name)), name


def test_global_registry_has_builtin_reference_kernels():
    registry = finpu.get_global_registry()
    assert registry.frozen
    kernel_ids = {d.kernel_id for d in registry.descriptors()}
    assert "reference/attention/decode_v1" in kernel_ids
    assert "reference/attention/prefill_ragged_v1" in kernel_ids


def test_decode_auto_dispatch_selects_registered_cann_runner():
    from flashinfer.runtime.signature import InputSignature
    from flashinfer.runtime.hardware import HardwareSignature

    decision = finpu.dispatch(
        InputSignature(op="attention.single_decode", dtypes=("float16",),
                       layouts=("NHD",), specializations=(("head_dim", "64"),)),
        HardwareSignature(soc_model="Ascend950", aicore_count=56),
    )
    assert decision.backend == finpu.Backend.CANN
    assert callable(finpu.get_global_registry().get(decision.kernel_id).runner)


def test_wrapper_requires_plan_before_run():
    from flashinfer.attention import ReferenceTensor

    wrapper = finpu.BatchDecodeWithPagedKVCacheWrapper(
        ReferenceTensor.zeros((128,), dtype="uint8"), backend="reference",
    )
    assert not wrapper.is_planning_done
    with pytest.raises(RuntimeError, match="plan"):
        wrapper.run(q=None, paged_kv_cache=None)


def test_wrapper_plan_selects_reference_only_with_opt_in():
    from flashinfer.attention import ReferenceTensor

    wrapper = finpu.BatchDecodeWithPagedKVCacheWrapper(
        ReferenceTensor.zeros((128,), dtype="uint8"), backend="reference",
    )
    plan = wrapper.plan(
        ReferenceTensor.from_nested([0, 1], dtype="int32"),
        ReferenceTensor.from_nested([0], dtype="int32"),
        ReferenceTensor.from_nested([1], dtype="int32"),
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim=128,
        page_size=16,
        q_data_type="float16",
    )
    assert plan is None
    assert wrapper.plan_selection.backend == "reference"
    assert wrapper.is_planning_done


import json
from pathlib import Path


from flashinfer.parity import ParityEntry
from flashinfer.runtime import SchemaError


def test_nvidia_inventory_has_reasons_and_pinned_declarations():
    root = Path(__file__).parents[2] / "tools/metadata"
    total = 0
    for name in ("api_parity.json", "attention_api_parity.json"):
        entries = json.loads((root / name).read_text())["entries"]
        assert len({e["upstream"] for e in entries}) == len(entries)
        for entry in entries:
            ParityEntry.from_dict(entry)
            if entry["semantic_status"] == "intentionally_different":
                assert entry["reason"].strip()
            if "upstream_source" in entry:
                source = entry["upstream_source"]
                assert source["commit"] == "2895577720741ddfedff118202c89584996b8875"
                assert source["path"].startswith("flashinfer/")
                assert source["line"] > 0
                assert len(source["sha256"]) == 64
                total += 1
    assert total >= 46


def test_intentional_difference_cannot_lose_its_reason():
    with pytest.raises(SchemaError, match="requires reason"):
        ParityEntry.from_dict(
            dict(
                upstream="flashinfer.comm.example",
                local="flashinfer.comm.example",
                upstream_kind="stable",
                semantic_status="intentionally_different",
                implementation_status="missing",
                priority="P3",
            )
        )


from flashinfer import RoutingMethodType
from flashinfer.runtime.reference_kernels import _route_topk
from flashinfer.tllm_enums import RoutingMethodType as CanonicalRoutingMethodType


def test_enum_members_and_eval_repr_match_pinned_upstream():
    assert RoutingMethodType is CanonicalRoutingMethodType
    assert {m.name: m.value for m in RoutingMethodType} == {
        "Default": 0, "Renormalize": 1, "DeepSeekV3": 2, "Llama4": 3,
        "RenormalizeNaive": 4, "TopK": 5, "SigmoidRenorm": 6, "MiniMax2": 7,
        "Sigmoid": 8, "TopKSigmoid": 9, "Unspecified": 10,
    }
    for member in RoutingMethodType:
        assert eval(repr(member), {"RoutingMethodType": RoutingMethodType}) is member
    assert RoutingMethodType(5) is RoutingMethodType.TopK


@pytest.mark.parametrize("member", list(RoutingMethodType)[5:])
def test_valid_but_unimplemented_routes_fail_before_tensor_computation(member):
    with pytest.raises(NotImplementedError, match=member.name):
        _route_topk(None, None, 2, 1, 1, 1.0, member)


import torch

import flashinfer
from flashinfer.attention import ReferenceTensor


def index(values):
    return ReferenceTensor.from_nested(values, dtype="int32")


@pytest.mark.parametrize("enable_pdl", [None, False, True])
@pytest.mark.parametrize("mode", ["paged", "ragged", "decode"])
def test_attention_pdl_is_a_hardware_noop(mode, enable_pdl):
    workspace = ReferenceTensor.zeros((128,), dtype="uint8")
    q = ReferenceTensor.from_nested([[[0.0]]])
    k = ReferenceTensor.from_nested([[[0.0]], [[0.0]]])
    v = ReferenceTensor.from_nested([[[4.0]], [[8.0]]])
    if mode == "ragged":
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, backend="reference")
        wrapper.plan(index([0, 1]), index([0, 2]), 1, 1, 1, q_data_type="float32")
        output = wrapper.run(q, k, v, enable_pdl=enable_pdl)
    else:
        cls = flashinfer.BatchPrefillWithPagedKVCacheWrapper if mode == "paged" else flashinfer.BatchDecodeWithPagedKVCacheWrapper
        wrapper = cls(workspace, backend="reference")
        args = (index([0, 1]), index([0]), index([2]), 1, 1, 1, 2)
        if mode == "paged":
            args = (index([0, 1]), *args)
        wrapper.plan(*args, q_data_type="float32")
        kp = ReferenceTensor((1, 2, 1, 1), k.data, k.dtype)
        vp = ReferenceTensor((1, 2, 1, 1), v.data, v.dtype)
        output = wrapper.run(q, (kp, vp), enable_pdl=enable_pdl)
    assert output.data == (6.0,)


@pytest.mark.parametrize("enable_pdl", [None, False, True])
def test_norm_and_activation_share_pdl_semantics(monkeypatch, enable_pdl):
    monkeypatch.setenv("FLASHINFER_NPU_BACKEND", "reference")
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    weight = torch.ones(4)
    actual = flashinfer.rmsnorm(x, weight, enable_pdl=enable_pdl)
    expected = flashinfer.rmsnorm(x, weight, enable_pdl=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual = flashinfer.silu_and_mul(x, enable_pdl=enable_pdl)
    expected = flashinfer.silu_and_mul(x, enable_pdl=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
