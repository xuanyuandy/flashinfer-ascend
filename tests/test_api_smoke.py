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


def test_decode_auto_dispatch_has_no_runner_yet():
    # auto 路径上内置 kernel 只有 reference，而 reference 默认被排除 → 无候选
    from flashinfer.runtime.signature import InputSignature
    from flashinfer.runtime.hardware import HardwareSignature

    with pytest.raises(DispatchError, match="无可用 kernel"):
        finpu.dispatch(
            InputSignature(op="attention.decode", dtypes=("float16",)),
            HardwareSignature(soc_model="Ascend910B", aicore_count=20),
        )


def test_wrapper_requires_plan_before_run():
    wrapper = finpu.BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer=None
    )
    assert not wrapper.is_planning_done
    with pytest.raises(RuntimeError, match="plan"):
        wrapper.run(q=None, kv_cache=None)


def test_wrapper_plan_falls_back_to_reference_with_opt_in():
    wrapper = finpu.BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer=None, backend="reference"
    )
    plan = wrapper.plan(
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim=128,
        page_size=16,
        q_dtype="float16",
    )
    assert plan.kernel_id == "reference/attention/decode_v1"
    assert wrapper.is_planning_done
