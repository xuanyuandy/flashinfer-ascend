"""norm 算子：注册、调度与数值（§3.1 / §5.3 / §12）。

host 单测无需 torch / NPU（§14.1）；数值测试按环境条件跳过，
在 NPU 容器中执行全量。
"""

from __future__ import annotations

import pytest

import flashinfer as finpu
from flashinfer.runtime.hardware import HardwareSignature
from flashinfer.runtime.schema import (
    Backend,
    CapabilitySpec,
    CompileMode,
    DispatchError,
    KernelDescriptor,
)
from flashinfer.runtime.signature import InputSignature

NPU_HW = HardwareSignature(soc_model="Ascend910B", aicore_count=20)
HOST_HW = HardwareSignature()  # unknown：host 开发机

RMSNORM_ID_CANN = "cann/norm/rmsnorm_aclnn_v1"
FUSED_ID_CANN = "cann/norm/fused_add_rmsnorm_aclnn_v1"
RMSNORM_ID_REF = "reference/norm/rmsnorm_v1"
FUSED_ID_REF = "reference/norm/fused_add_rmsnorm_v1"


def _rmsnorm_sig(input_dtype="float16", weight_dtype="float16") -> InputSignature:
    return InputSignature(
        op="norm.rmsnorm",
        dtypes=(input_dtype, weight_dtype),
        shapes=((8, 4096), (4096,)),
        specializations=(("eps", "1e-06"),),
    )


def _fused_sig(dtype="float16") -> InputSignature:
    return InputSignature(
        op="norm.fused_add_rmsnorm",
        dtypes=(dtype, dtype, dtype),
        shapes=((8, 4096), (8, 4096), (4096,)),
        specializations=(("eps", "1e-06"),),
    )


class TestNormRegistration:
    def test_global_registry_has_norm_kernels(self):
        ids = {d.kernel_id for d in finpu.get_global_registry().descriptors()}
        assert {RMSNORM_ID_CANN, FUSED_ID_CANN, RMSNORM_ID_REF, FUSED_ID_REF} <= ids
        assert {
            "cann/norm/gemma_rmsnorm_aclnn_v1",
            "cann/norm/layernorm_aclnn_v1",
            "triton/norm/gemma_fused_add_rmsnorm_v1",
            "reference/norm/gemma_rmsnorm_v1",
            "reference/norm/gemma_fused_add_rmsnorm_v1",
            "reference/norm/layernorm_v1",
        } <= ids

    def test_triton_descriptor_for_cann_missing_op(self):
        registry = finpu.get_global_registry()
        descriptor = registry.get("triton/norm/gemma_fused_add_rmsnorm_v1")
        assert descriptor.backend == Backend.TRITON
        assert descriptor.compile_mode == CompileMode.JIT

    def test_cann_descriptors_declare_support_set(self):
        descriptor = finpu.get_global_registry().get(RMSNORM_ID_CANN)
        assert descriptor.backend == Backend.CANN
        assert descriptor.capability.soc_models == ("Ascend910B", "Ascend910C", "Ascend950")
        assert set(descriptor.capability.dtypes) == {
            "float16",
            "bfloat16",
            "float32",
        }


class TestNormDispatch:
    @pytest.mark.parametrize(
        "sig,expected",
        [(_rmsnorm_sig(), RMSNORM_ID_CANN), (_fused_sig(), FUSED_ID_CANN)],
    )
    def test_auto_prefers_cann_on_npu(self, sig, expected):
        decision = finpu.dispatch(sig, NPU_HW)
        assert decision.kernel_id == expected
        assert decision.backend == Backend.CANN

    def test_reference_excluded_from_auto(self):
        decision = finpu.dispatch(_rmsnorm_sig(), NPU_HW)
        report = {r.kernel_id: r for r in decision.reports}[RMSNORM_ID_REF]
        assert not report.accepted
        assert any("opt-in" in reason for reason in report.reasons)

    def test_reference_via_explicit_opt_in(self):
        for sig, expected in (
            (_rmsnorm_sig(), RMSNORM_ID_REF),
            (_fused_sig(), FUSED_ID_REF),
        ):
            decision = finpu.dispatch(sig, NPU_HW, backend="reference")
            assert decision.kernel_id == expected

    def test_explicit_backend_cann(self):
        decision = finpu.dispatch(_fused_sig(), NPU_HW, backend="cann")
        assert decision.kernel_id == FUSED_ID_CANN

    def test_explicit_kernel_id_lock(self):
        decision = finpu.dispatch(
            _rmsnorm_sig(), NPU_HW, kernel_id=RMSNORM_ID_REF
        )
        assert decision.kernel_id == RMSNORM_ID_REF
        assert decision.backend == Backend.REFERENCE

    def test_host_auto_has_no_candidate(self):
        # host（unknown SoC）不在 CANN 声明支持集内，reference 需 opt-in
        with pytest.raises(DispatchError, match="无可用 kernel"):
            finpu.dispatch(_rmsnorm_sig(), HOST_HW)

    def test_unsupported_dtype_rejected(self):
        with pytest.raises(DispatchError) as excinfo:
            finpu.dispatch(_rmsnorm_sig("float64", "float64"), NPU_HW)
        assert "float64" in str(excinfo.value)

    def test_dispatch_cache_hit_on_repeat(self):
        first = finpu.dispatch(_fused_sig(), NPU_HW)
        second = finpu.dispatch(_fused_sig(), NPU_HW)
        assert not first.cache_hit
        assert second.cache_hit
        assert first.kernel_id == second.kernel_id


class _FakeTensor:
    """仅携带 dtype/shape 协议的轻量假张量（§14.1：host 单测不依赖 torch）。"""

    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape


class TestNormFacadeFlow:
    """门面全链路：假张量 + custom probe kernel（kernel_id 显式锁定）。"""

    def test_rmsnorm_facade_flow(self, monkeypatch, clean_global_registry):
        calls = []
        finpu.register_custom_kernel(
            KernelDescriptor(
                kernel_id="custom/norm/rmsnorm_probe",
                backend=Backend.CUSTOM,
                capability=CapabilitySpec(op="norm.rmsnorm"),
                runner=lambda *args: calls.append(args),
            )
        )
        monkeypatch.setattr(
            "flashinfer.norm.current_hardware", lambda: NPU_HW
        )
        x = _FakeTensor("float16", (8, 4096))
        w = _FakeTensor("float16", (4096,))
        o = _FakeTensor("float16", (8, 4096))
        # enable_pdl 接受并忽略；out 复用并原样返回
        result = finpu.norm.rmsnorm(
            x, w, eps=1e-5, out=o, enable_pdl=True,
            kernel_id="custom/norm/rmsnorm_probe",
        )
        assert result is o
        (out_arg, in_arg, w_arg, eps_arg), = calls
        assert out_arg is o and in_arg is x and w_arg is w
        assert eps_arg == pytest.approx(1e-5)

    def test_fused_add_rmsnorm_facade_flow(
        self, monkeypatch, clean_global_registry
    ):
        calls = []
        finpu.register_custom_kernel(
            KernelDescriptor(
                kernel_id="custom/norm/fused_add_rmsnorm_probe",
                backend=Backend.CUSTOM,
                capability=CapabilitySpec(op="norm.fused_add_rmsnorm"),
                runner=lambda *args: calls.append(args),
            )
        )
        monkeypatch.setattr(
            "flashinfer.norm.current_hardware", lambda: NPU_HW
        )
        x = _FakeTensor("float16", (8, 4096))
        r = _FakeTensor("float16", (8, 4096))
        w = _FakeTensor("float16", (4096,))
        result = finpu.norm.fused_add_rmsnorm(
            x, r, w, enable_pdl=False,
            kernel_id="custom/norm/fused_add_rmsnorm_probe",
        )
        assert result is None
        (in_arg, r_arg, w_arg, eps_arg), = calls
        assert in_arg is x and r_arg is r and w_arg is w
        assert eps_arg == pytest.approx(1e-6)


@pytest.mark.usefixtures("torch_for_class")
class TestNormDeviceDispatch:
    @pytest.mark.usefixtures("npu_device")
    def test_auto_dispatch_picks_cann_on_npu(self):
        hardware = finpu.current_hardware()
        assert hardware.soc_model in ("Ascend910B", "Ascend910C", "Ascend950")
        assert finpu.dispatch(_rmsnorm_sig(), hardware).kernel_id == (
            RMSNORM_ID_CANN
        )
        assert finpu.dispatch(_fused_sig(), hardware).kernel_id == FUSED_ID_CANN


class TestGemmaNormDispatch:
    def _gemma_sig(self):
        return InputSignature(
            op="norm.gemma_rmsnorm",
            dtypes=("float16", "float16"),
            shapes=((8, 4096), (4096,)),
            specializations=(("eps", "1e-06"),),
        )

    def _gemma_fused_sig(self):
        return InputSignature(
            op="norm.gemma_fused_add_rmsnorm",
            dtypes=("float16", "float16", "float16"),
            shapes=((8, 4096), (8, 4096), (4096,)),
            specializations=(("eps", "1e-06"),),
        )

    def _layernorm_sig(self, dtypes=("float16", "float16", "float16")):
        return InputSignature(
            op="norm.layernorm",
            dtypes=dtypes,
            shapes=((8, 4096), (4096,), (4096,)),
            specializations=(("eps", "1e-06"),),
        )

    def test_auto_prefers_cann_for_gemma_rmsnorm(self):
        decision = finpu.dispatch(self._gemma_sig(), NPU_HW)
        assert decision.kernel_id == "cann/norm/gemma_rmsnorm_aclnn_v1"

    def test_auto_prefers_cann_for_layernorm(self):
        decision = finpu.dispatch(self._layernorm_sig(), NPU_HW)
        assert decision.kernel_id == "cann/norm/layernorm_aclnn_v1"

    def test_auto_prefers_triton_for_gemma_fused(self):
        # CANN 无该算子：auto 应落到 Triton 自定义 kernel
        decision = finpu.dispatch(self._gemma_fused_sig(), NPU_HW)
        assert decision.kernel_id == "triton/norm/gemma_fused_add_rmsnorm_v1"
        assert decision.backend == Backend.TRITON

    def test_explicit_backend_selection(self):
        decision = finpu.dispatch(
            self._gemma_fused_sig(), NPU_HW, backend="reference"
        )
        assert decision.kernel_id == "reference/norm/gemma_fused_add_rmsnorm_v1"
        decision = finpu.dispatch(
            self._layernorm_sig(), NPU_HW, backend="cann"
        )
        assert decision.kernel_id == "cann/norm/layernorm_aclnn_v1"

    def test_layernorm_mixed_dtype_sig_accepted(self):
        # 上游约定 gemma/beta 为 float32、input 可为 bfloat16
        sig = self._layernorm_sig(("bfloat16", "float32", "float32"))
        decision = finpu.dispatch(sig, NPU_HW)
        assert decision.kernel_id == "cann/norm/layernorm_aclnn_v1"
