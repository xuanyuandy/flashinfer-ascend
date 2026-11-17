"""rope 算子：注册、调度与数值（§3.1 / §5.3 / §12）。

host 单测无需 torch / NPU（§14.1）；数值测试按环境条件跳过，
在 NPU 容器中执行全量。
"""

from __future__ import annotations

import pytest

import flashinfer as finpu
from flashinfer.runtime.hardware import HardwareSignature
from flashinfer.runtime.registry import _reset_global_registry_for_testing
from flashinfer.runtime.schema import (
    Backend,
    CapabilitySpec,
    DispatchError,
    KernelDescriptor,
)
from flashinfer.runtime.signature import InputSignature

NPU_HW = HardwareSignature(soc_model="Ascend910B", aicore_count=20)
HOST_HW = HardwareSignature()  # unknown：host 开发机

ROPE_ID_CANN = "cann/rope/apply_rope_cos_sin_cache_aclnn_v1"
ROPE_ID_REF = "reference/rope/apply_rope_with_cos_sin_cache_v1"


def _rope_sig(q_dtype="float16", pos_dtype="int64") -> InputSignature:
    return InputSignature(
        op="rope.apply_rope_with_cos_sin_cache_inplace",
        dtypes=(pos_dtype, q_dtype, q_dtype, "float32"),
        shapes=((8,), (8, 4 * 128), (8, 2 * 128), (2048, 128)),
        specializations=(("head_size", "128"), ("is_neox", "True")),
    )


class TestRopeRegistration:
    def test_global_registry_has_rope_kernels(self):
        ids = {d.kernel_id for d in finpu.get_global_registry().descriptors()}
        assert {ROPE_ID_CANN, ROPE_ID_REF} <= ids

    def test_cann_descriptor_declares_support_set(self):
        descriptor = finpu.get_global_registry().get(ROPE_ID_CANN)
        assert descriptor.backend == Backend.CANN
        assert descriptor.capability.soc_models == ("Ascend910B", "Ascend910C", "Ascend950")
        assert set(descriptor.capability.dtypes) == {
            "float16",
            "bfloat16",
            "float32",
            "int32",
            "int64",
        }


class TestRopeDispatch:
    def test_auto_prefers_cann_on_npu(self):
        decision = finpu.dispatch(_rope_sig(), NPU_HW)
        assert decision.kernel_id == ROPE_ID_CANN
        assert decision.backend == Backend.CANN

    def test_reference_excluded_from_auto(self):
        decision = finpu.dispatch(_rope_sig(), NPU_HW)
        report = {r.kernel_id: r for r in decision.reports}[ROPE_ID_REF]
        assert not report.accepted
        assert any("opt-in" in reason for reason in report.reasons)

    def test_reference_via_explicit_opt_in(self):
        decision = finpu.dispatch(_rope_sig(), NPU_HW, backend="reference")
        assert decision.kernel_id == ROPE_ID_REF

    def test_explicit_backend_cann(self):
        decision = finpu.dispatch(_rope_sig(), NPU_HW, backend="cann")
        assert decision.kernel_id == ROPE_ID_CANN

    def test_explicit_kernel_id_lock(self):
        decision = finpu.dispatch(_rope_sig(), NPU_HW, kernel_id=ROPE_ID_REF)
        assert decision.kernel_id == ROPE_ID_REF
        assert decision.backend == Backend.REFERENCE

    def test_host_auto_has_no_candidate(self):
        # host（unknown SoC）不在 CANN 声明支持集内，reference 需 opt-in
        with pytest.raises(DispatchError, match="无可用 kernel"):
            finpu.dispatch(_rope_sig(), HOST_HW)

    def test_unsupported_dtype_rejected(self):
        with pytest.raises(DispatchError) as excinfo:
            finpu.dispatch(_rope_sig("float64"), NPU_HW)
        assert "float64" in str(excinfo.value)

    def test_dispatch_cache_hit_on_repeat(self):
        first = finpu.dispatch(_rope_sig(), NPU_HW)
        second = finpu.dispatch(_rope_sig(), NPU_HW)
        assert not first.cache_hit
        assert second.cache_hit
        assert first.kernel_id == second.kernel_id


class _FakeTensor:
    """仅携带 dtype/shape 协议的轻量假张量（§14.1：host 单测不依赖 torch）。"""

    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape


@pytest.fixture
def _clean_global_registry():
    yield
    _reset_global_registry_for_testing()


class TestRopeFacadeFlow:
    """门面全链路：假张量 + custom probe kernel（kernel_id 显式锁定）。"""

    def test_facade_flow(self, monkeypatch, _clean_global_registry):
        calls = []
        finpu.register_custom_kernel(
            KernelDescriptor(
                kernel_id="custom/rope/apply_rope_probe",
                backend=Backend.CUSTOM,
                capability=CapabilitySpec(
                    op="rope.apply_rope_with_cos_sin_cache_inplace"
                ),
                runner=lambda *args: calls.append(args),
            )
        )
        monkeypatch.setattr(
            "flashinfer.rope.current_hardware", lambda: NPU_HW
        )
        pos = _FakeTensor("int64", (8,))
        q = _FakeTensor("float16", (8, 4 * 128))
        k = _FakeTensor("float16", (8, 2 * 128))
        cache = _FakeTensor("float32", (2048, 128))
        result = finpu.apply_rope_with_cos_sin_cache_inplace(
            pos, q, k, 128, cache, is_neox=False,
            kernel_id="custom/rope/apply_rope_probe",
        )
        assert result is None
        (pos_arg, q_arg, k_arg, hs_arg, cache_arg, neox_arg), = calls
        assert pos_arg is pos and q_arg is q and k_arg is k
        assert cache_arg is cache
        assert hs_arg == 128 and neox_arg is False

    def test_cache_dtype_must_be_float32(self, _clean_global_registry):
        pos = _FakeTensor("int64", (8,))
        q = _FakeTensor("float16", (8, 4 * 128))
        k = _FakeTensor("float16", (8, 2 * 128))
        cache = _FakeTensor("float16", (2048, 128))
        with pytest.raises(ValueError, match="float32"):
            finpu.apply_rope_with_cos_sin_cache_inplace(pos, q, k, 128, cache)


class TestRopeDeviceDispatch:
    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")

    def _require_npu(self):
        pytest.importorskip("torch_npu")
        if not self.torch.npu.is_available():
            pytest.skip("无 NPU 环境")

    def test_auto_dispatch_picks_cann_on_npu(self):
        self._require_npu()
        hardware = finpu.current_hardware()
        assert hardware.soc_model in ("Ascend910B", "Ascend910C", "Ascend950")
        assert finpu.dispatch(_rope_sig(), hardware).kernel_id == ROPE_ID_CANN
