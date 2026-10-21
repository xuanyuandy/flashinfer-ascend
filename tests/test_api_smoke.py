"""顶层 import 与 API 门面冒烟（不依赖 torch / NPU）。"""

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
