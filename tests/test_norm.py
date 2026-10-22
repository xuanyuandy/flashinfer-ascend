"""norm 算子：注册、调度与数值（§3.1 / §5.3 / §12）。

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

    def test_cann_descriptors_declare_verified_support_set(self):
        descriptor = finpu.get_global_registry().get(RMSNORM_ID_CANN)
        assert descriptor.backend == Backend.CANN
        assert descriptor.capability.soc_models == ("Ascend910B", "Ascend950")
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


@pytest.fixture
def _clean_global_registry():
    yield
    _reset_global_registry_for_testing()


class TestNormFacadeFlow:
    """门面全链路：假张量 + custom probe kernel（kernel_id 显式锁定）。"""

    def test_rmsnorm_facade_flow(self, monkeypatch, _clean_global_registry):
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
        self, monkeypatch, _clean_global_registry
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


class TestNormNumerics:
    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")

    def _require_npu(self):
        pytest.importorskip("torch_npu")
        if not self.torch.npu.is_available():
            pytest.skip("无 NPU 环境")

    def _fp64_rmsnorm(self, x, w, eps):
        xf = x.double()
        return (
            xf
            * self.torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
            * w.double()
        )

    def test_rmsnorm_reference_cpu(self):
        torch = self.torch
        x = torch.randn(8, 256, dtype=torch.float32)
        w = torch.randn(256, dtype=torch.float32)
        out = finpu.rmsnorm(x, w, backend="reference")
        assert torch.allclose(
            out.double(), self._fp64_rmsnorm(x, w, 1e-6), atol=1e-5
        )

    def test_rmsnorm_out_buffer_reused(self):
        torch = self.torch
        x = torch.randn(8, 256, dtype=torch.float32)
        w = torch.randn(256, dtype=torch.float32)
        o = torch.empty_like(x)
        result = finpu.rmsnorm(x, w, out=o, backend="reference")
        assert result is o
        assert torch.allclose(
            o.double(), self._fp64_rmsnorm(x, w, 1e-6), atol=1e-5
        )

    def test_fused_add_rmsnorm_reference_cpu(self):
        torch = self.torch
        x = torch.randn(8, 256, dtype=torch.float32)
        r = torch.randn(8, 256, dtype=torch.float32)
        w = torch.randn(256, dtype=torch.float32)
        x0, r0 = x.clone(), r.clone()
        result = finpu.fused_add_rmsnorm(x, r, w, backend="reference")
        assert result is None
        # Step 1: residual += input
        new_r = x0.double() + r0.double()
        assert torch.allclose(r.double(), new_r, atol=1e-5)
        # Step 2: input = (residual / RMS(residual)) * weight
        expected = (
            new_r
            * torch.rsqrt(new_r.pow(2).mean(-1, keepdim=True) + 1e-6)
            * w.double()
        )
        assert torch.allclose(x.double(), expected, atol=1e-5)

    def test_eps_flows_into_computation(self):
        torch = self.torch
        x = torch.full((2, 64), 3.0)
        w = torch.ones(64)
        out_small = finpu.rmsnorm(x, w, eps=1e-6, backend="reference")
        out_large = finpu.rmsnorm(x, w, eps=1.0, backend="reference")
        assert not torch.allclose(out_small, out_large)

    def test_auto_dispatch_picks_cann_on_npu(self):
        self._require_npu()
        hardware = finpu.current_hardware()
        assert hardware.soc_model in ("Ascend910B", "Ascend950")
        assert finpu.dispatch(_rmsnorm_sig(), hardware).kernel_id == (
            RMSNORM_ID_CANN
        )
        assert finpu.dispatch(_fused_sig(), hardware).kernel_id == FUSED_ID_CANN

    @pytest.mark.parametrize(
        "dtype,atol",
        [("float16", 2e-2), ("bfloat16", 8e-2), ("float32", 1e-4)],
    )
    def test_rmsnorm_npu_vs_fp64(self, dtype, atol):
        self._require_npu()
        torch = self.torch
        dt = getattr(torch, dtype)
        x = torch.randn(8, 4096, dtype=dt, device="npu")
        w = torch.randn(4096, dtype=dt, device="npu")
        out = finpu.rmsnorm(x, w)  # auto → CANN aclnn
        expected = self._fp64_rmsnorm(x.cpu(), w.cpu(), 1e-6)
        assert torch.allclose(out.cpu().double(), expected, atol=atol)

    @pytest.mark.parametrize(
        "dtype,atol",
        [("float16", 2e-2), ("bfloat16", 8e-2), ("float32", 1e-4)],
    )
    def test_fused_add_rmsnorm_npu_vs_fp64(self, dtype, atol):
        self._require_npu()
        torch = self.torch
        dt = getattr(torch, dtype)
        x = torch.randn(8, 4096, dtype=dt, device="npu")
        r = torch.randn(8, 4096, dtype=dt, device="npu")
        w = torch.randn(4096, dtype=dt, device="npu")
        x0, r0 = x.clone().cpu(), r.clone().cpu()
        finpu.fused_add_rmsnorm(x, r, w)  # auto → CANN aclnn
        new_r = x0.double() + r0.double()
        assert torch.allclose(r.cpu().double(), new_r, atol=atol)
        expected = (
            new_r
            * torch.rsqrt(new_r.pow(2).mean(-1, keepdim=True) + 1e-6)
            * w.cpu().double()
        )
        assert torch.allclose(x.cpu().double(), expected, atol=atol)

    @pytest.mark.parametrize(
        "dtype,atol",
        [("float16", 2e-2), ("bfloat16", 8e-2), ("float32", 1e-4)],
    )
    def test_cann_vs_reference_consistency(self, dtype, atol):
        """M1 验收口径：同输入，CANN 与 reference 两 backend 数值一致。"""
        self._require_npu()
        torch = self.torch
        dt = getattr(torch, dtype)
        x = torch.randn(8, 4096, dtype=dt, device="npu")
        r = torch.randn(8, 4096, dtype=dt, device="npu")
        w = torch.randn(4096, dtype=dt, device="npu")

        y_cann = finpu.rmsnorm(x, w, backend="cann")
        y_ref = finpu.rmsnorm(x.cpu(), w.cpu(), backend="reference")
        assert torch.allclose(y_cann.cpu().double(), y_ref.double(), atol=atol)

        x1, r1 = x.clone(), r.clone()
        x2, r2 = x.cpu(), r.cpu()
        finpu.fused_add_rmsnorm(x1, r1, w, backend="cann")
        finpu.fused_add_rmsnorm(x2, r2, w.cpu(), backend="reference")
        assert torch.allclose(x1.cpu().double(), x2.double(), atol=atol)
        assert torch.allclose(r1.cpu().double(), r2.double(), atol=atol)


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


class TestGemmaNormNumerics:
    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")

    def _require_npu(self):
        pytest.importorskip("torch_npu")
        if not self.torch.npu.is_available():
            pytest.skip("无 NPU 环境")

    def _require_triton(self):
        pytest.importorskip("triton")

    def _fp64_gemma_rmsnorm(self, x, w, eps):
        torch = self.torch
        xf = x.double()
        return (
            xf
            * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
            * (w.double() + 1.0)
        )

    def _fp64_layernorm(self, x, w, b, eps):
        torch = self.torch
        xf = x.double()
        mean = xf.mean(-1, keepdim=True)
        var = xf.var(-1, unbiased=False, keepdim=True)
        return (xf - mean) * torch.rsqrt(var + eps) * w.double() + b.double()

    def test_gemma_rmsnorm_reference_cpu(self):
        torch = self.torch
        x = torch.randn(8, 256, dtype=torch.float32)
        w = torch.randn(256, dtype=torch.float32)
        out = finpu.gemma_rmsnorm(x, w, backend="reference")
        assert torch.allclose(
            out.double(), self._fp64_gemma_rmsnorm(x, w, 1e-6), atol=1e-5
        )

    def test_gemma_fused_add_rmsnorm_reference_cpu(self):
        torch = self.torch
        x = torch.randn(8, 256, dtype=torch.float32)
        r = torch.randn(8, 256, dtype=torch.float32)
        w = torch.randn(256, dtype=torch.float32)
        x0, r0 = x.clone(), r.clone()
        result = finpu.gemma_fused_add_rmsnorm(x, r, w, backend="reference")
        assert result is None
        new_r = x0.double() + r0.double()
        assert torch.allclose(r.double(), new_r, atol=1e-5)
        expected = (
            new_r
            * torch.rsqrt(new_r.pow(2).mean(-1, keepdim=True) + 1e-6)
            * (w.double() + 1.0)
        )
        assert torch.allclose(x.double(), expected, atol=1e-5)

    def test_layernorm_reference_cpu(self):
        torch = self.torch
        x = torch.randn(8, 256, dtype=torch.float32)
        w = torch.randn(256, dtype=torch.float32)
        b = torch.randn(256, dtype=torch.float32)
        out = finpu.layernorm(x, w, b, backend="reference")
        assert torch.allclose(
            out.double(), self._fp64_layernorm(x, w, b, 1e-6), atol=1e-5
        )

    @pytest.mark.parametrize(
        "dtype,atol",
        [("float16", 2e-2), ("bfloat16", 8e-2), ("float32", 1e-4)],
    )
    def test_gemma_rmsnorm_npu_vs_fp64(self, dtype, atol):
        self._require_npu()
        torch = self.torch
        dt = getattr(torch, dtype)
        x = torch.randn(8, 4096, dtype=dt, device="npu")
        w = torch.randn(4096, dtype=dt, device="npu")
        out = finpu.gemma_rmsnorm(x, w)  # auto → CANN aclnn
        expected = self._fp64_gemma_rmsnorm(x.cpu(), w.cpu(), 1e-6)
        assert torch.allclose(out.cpu().double(), expected, atol=atol)

    @pytest.mark.parametrize(
        "dtype,atol",
        [("float16", 2e-2), ("bfloat16", 8e-2), ("float32", 1e-4)],
    )
    def test_layernorm_npu_vs_fp64(self, dtype, atol):
        self._require_npu()
        torch = self.torch
        dt = getattr(torch, dtype)
        x = torch.randn(8, 4096, dtype=dt, device="npu")
        w = torch.randn(4096, dtype=dt, device="npu")
        b = torch.randn(4096, dtype=dt, device="npu")
        out = finpu.layernorm(x, w, b)  # auto → CANN aclnn
        expected = self._fp64_layernorm(x.cpu(), w.cpu(), b.cpu(), 1e-6)
        assert torch.allclose(out.cpu().double(), expected, atol=atol)

    @pytest.mark.parametrize(
        "dtype,atol",
        [("float16", 2e-2), ("bfloat16", 8e-2), ("float32", 1e-4)],
    )
    def test_gemma_fused_add_rmsnorm_triton_npu_vs_fp64(self, dtype, atol):
        """M1 验收口径：Triton 自定义 kernel 与 fp64 参考数值一致。"""
        self._require_npu()
        self._require_triton()
        torch = self.torch
        dt = getattr(torch, dtype)
        x = torch.randn(8, 4096, dtype=dt, device="npu")
        r = torch.randn(8, 4096, dtype=dt, device="npu")
        w = torch.randn(4096, dtype=dt, device="npu")
        x0, r0 = x.clone().cpu(), r.clone().cpu()
        finpu.gemma_fused_add_rmsnorm(x, r, w)  # auto → Triton
        new_r = x0.double() + r0.double()
        assert torch.allclose(r.cpu().double(), new_r, atol=atol)
        expected = (
            new_r
            * torch.rsqrt(new_r.pow(2).mean(-1, keepdim=True) + 1e-6)
            * (w.cpu().double() + 1.0)
        )
        assert torch.allclose(x.cpu().double(), expected, atol=atol)

    @pytest.mark.parametrize(
        "dtype,atol",
        [("float16", 2e-2), ("bfloat16", 8e-2), ("float32", 1e-4)],
    )
    def test_gemma_fused_triton_vs_reference_consistency(self, dtype, atol):
        """跨 backend 一致性：Triton 与 reference 同输入结果一致。"""
        self._require_npu()
        self._require_triton()
        torch = self.torch
        dt = getattr(torch, dtype)
        x = torch.randn(8, 4096, dtype=dt, device="npu")
        r = torch.randn(8, 4096, dtype=dt, device="npu")
        w = torch.randn(4096, dtype=dt, device="npu")
        x1, r1 = x.clone(), r.clone()
        x2, r2 = x.cpu(), r.cpu()
        finpu.gemma_fused_add_rmsnorm(x1, r1, w, backend="triton")
        finpu.gemma_fused_add_rmsnorm(x2, r2, w.cpu(), backend="reference")
        assert torch.allclose(x1.cpu().double(), x2.double(), atol=atol)
        assert torch.allclose(r1.cpu().double(), r2.double(), atol=atol)
