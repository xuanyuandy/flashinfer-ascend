"""bmm_fp8 / gemm_fp8_nt_groupwise：注册、调度与数值（§3.1 / §5.3 / §12）。

fp8 GEMM 为 950 代际特性：CANN kernel 走 ``npu_grouped_matmul`` fp8
路径（限定 Ascend950 注册，910B auto 无候选）；reference kernel（CPU
反量化 fp32 计算）需显式 opt-in。host 单测无需 torch / NPU（§14.1），
数值与 CANN 数据流测试在容器 CPU 上执行（fp8 张量无法物化到 NPU；
CANN runner 的 fp8 GEMM 原语以朴素 fp32 matmul 替换后对拍 reference）。
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
NPU_HW_950 = HardwareSignature(soc_model="Ascend950", aicore_count=64)

BMM_ID_REF = "reference/gemm/bmm_fp8_v1"
GW_ID_REF = "reference/gemm/gemm_fp8_nt_groupwise_v1"
BMM_ID_CANN = "cann/gemm/bmm_fp8_grouped_v1"
GW_ID_CANN = "cann/gemm/gemm_fp8_nt_groupwise_block_v1"


def _bmm_sig() -> InputSignature:
    return InputSignature(
        op="gemm.bmm_fp8",
        dtypes=("float8_e4m3fn", "float8_e4m3fn", "float32", "float32"),
        shapes=((4, 16, 32), (4, 32, 24), (1,), (1,)),
        specializations=(("dtype", "bfloat16"),),
    )


def _gw_sig() -> InputSignature:
    return InputSignature(
        op="gemm.gemm_fp8_nt_groupwise",
        dtypes=("float8_e4m3fn", "float8_e4m3fn", "float32", "float32"),
        shapes=((32, 256), (64, 256), (32, 2), (1, 2)),
        specializations=(
            ("scale_granularity_mnk", "(1, 128, 128)"),
            ("scale_major_mode", "K"),
        ),
    )


class TestGemmFp8Registration:
    def test_global_registry_has_gemm_kernels(self):
        ids = {d.kernel_id for d in finpu.get_global_registry().descriptors()}
        assert {BMM_ID_REF, GW_ID_REF, BMM_ID_CANN, GW_ID_CANN} <= ids


class TestGemmFp8Dispatch:
    def test_reference_via_explicit_opt_in(self):
        assert finpu.dispatch(_bmm_sig(), NPU_HW, backend="reference").kernel_id == BMM_ID_REF
        assert finpu.dispatch(_gw_sig(), NPU_HW, backend="reference").kernel_id == GW_ID_REF

    def test_auto_has_no_candidate(self):
        # 910B 无 CANN fp8 GEMM kernel（soc 限定 950），reference 需 opt-in
        # → auto 无候选
        with pytest.raises(DispatchError, match="无可用 kernel"):
            finpu.dispatch(_bmm_sig(), NPU_HW)
        with pytest.raises(DispatchError, match="无可用 kernel"):
            finpu.dispatch(_gw_sig(), NPU_HW)

    def test_auto_prefers_cann_on_950(self):
        assert finpu.dispatch(_bmm_sig(), NPU_HW_950).kernel_id == BMM_ID_CANN
        assert finpu.dispatch(_gw_sig(), NPU_HW_950).kernel_id == GW_ID_CANN


class _FakeTensor:
    """仅携带 dtype/shape 协议的轻量假张量（§14.1：host 单测不依赖 torch）。"""

    def __init__(self, dtype, shape, ndim=None):
        self.dtype = dtype
        self.shape = shape
        self.ndim = ndim if ndim is not None else len(shape)


@pytest.fixture
def _clean_global_registry():
    yield
    _reset_global_registry_for_testing()


class TestGemmFp8FacadeFlow:
    """门面全链路：假张量 + custom probe kernel（kernel_id 显式锁定）。"""

    def test_bmm_fp8_facade_flow(self, monkeypatch, _clean_global_registry):
        calls = []
        finpu.register_custom_kernel(
            KernelDescriptor(
                kernel_id="custom/gemm/bmm_fp8_probe",
                backend=Backend.CUSTOM,
                capability=CapabilitySpec(op="gemm.bmm_fp8"),
                runner=lambda *args: calls.append(args) or args[-1],
            )
        )
        monkeypatch.setattr(
            "flashinfer.gemm.current_hardware", lambda: NPU_HW
        )
        A = _FakeTensor("float8_e4m3fn", (4, 16, 32))
        B = _FakeTensor("float8_e4m3fn", (4, 32, 24))
        s = _FakeTensor("float32", (1,))
        out = _FakeTensor("bfloat16", (4, 16, 24))
        result = finpu.bmm_fp8(
            A, B, s, s, "bfloat16", out,
            kernel_id="custom/gemm/bmm_fp8_probe",
        )
        assert result is out
        (a_, b_, sa_, sb_, out_), = calls
        assert a_ is A and b_ is B and sa_ is s and sb_ is s and out_ is out

    def test_groupwise_facade_flow(self, monkeypatch, _clean_global_registry):
        calls = []
        finpu.register_custom_kernel(
            KernelDescriptor(
                kernel_id="custom/gemm/gw_probe",
                backend=Backend.CUSTOM,
                capability=CapabilitySpec(op="gemm.gemm_fp8_nt_groupwise"),
                runner=lambda *args: calls.append(args) or args[-1],
            )
        )
        monkeypatch.setattr(
            "flashinfer.gemm.current_hardware", lambda: NPU_HW
        )
        a = _FakeTensor("float8_e4m3fn", (32, 256))
        b = _FakeTensor("float8_e4m3fn", (64, 256))
        sa = _FakeTensor("float32", (32, 2))
        sb = _FakeTensor("float32", (1, 2))
        out = _FakeTensor("bfloat16", (32, 64))
        result = finpu.gemm_fp8_nt_groupwise(
            a, b, sa, sb,
            scale_major_mode="K",
            scale_granularity_mnk=(1, 128, 128),
            out=out,
            kernel_id="custom/gemm/gw_probe",
        )
        assert result is out
        (a_, b_, sa_, sb_, mode, gran, out_), = calls
        assert a_ is a and b_ is b and out_ is out
        assert mode == "K" and gran == (1, 128, 128)

    def test_groupwise_shape_validation(self, _clean_global_registry):
        a = _FakeTensor("float8_e4m3fn", (32, 256))
        b3d = _FakeTensor("float8_e4m3fn", (4, 64, 256))
        with pytest.raises(ValueError, match="Shape mismatch"):
            finpu.gemm_fp8_nt_groupwise(a, b3d, a, a)
        bk = _FakeTensor("float8_e4m3fn", (64, 128))
        with pytest.raises(ValueError, match="Shape mismatch"):
            finpu.gemm_fp8_nt_groupwise(a, bk, a, a)
        with pytest.raises(ValueError, match="scale_major_mode"):
            finpu.gemm_fp8_nt_groupwise(a, a, a, a, scale_major_mode="X")




class TestGemmFp8CannCascade:
    """CANN runner（950）数据流：CPU 上将 _fp8_grouped_gemm 替换为 fp32
    朴素分段 matmul，与 reference runner 对拍（不依赖 NPU）。"""

    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")
        from flashinfer.cann import gemm as cann_gemm
        from flashinfer.runtime import reference_kernels as ref

        self.cann_gemm = cann_gemm
        self.ref = ref

    def _naive_grouped_gemm(self, x8, weight_list, group_list):
        """fp32 朴素分段 matmul，替换 _fp8_grouped_gemm（不依赖 NPU）。"""
        torch = self.torch
        outs, start = [], 0
        for e, end in enumerate(group_list):
            if end > start:
                outs.append(x8[start:end].float() @ weight_list[e].float())
            start = end
        if not outs:
            return torch.zeros(0, weight_list[0].shape[1])
        return torch.cat(outs, dim=0)

    def test_bmm_fp8_matches_reference(self, monkeypatch):
        torch = self.torch
        torch.manual_seed(0)
        monkeypatch.setitem(self.cann_gemm._bmm_fp8_cann.__globals__, '_fp8_grouped_gemm', self._naive_grouped_gemm)
        A = (torch.randn(4, 16, 32) * 0.2).to(torch.float8_e4m3fn)
        B = (torch.randn(4, 32, 24) * 0.2).to(torch.float8_e4m3fn)
        sa = torch.rand(1) * 0.02 + 0.01
        sb = torch.rand(1) * 0.02 + 0.01
        out_c = torch.empty(4, 16, 24, dtype=torch.bfloat16)
        out_r = torch.empty(4, 16, 24, dtype=torch.bfloat16)
        self.cann_gemm._bmm_fp8_cann(A, B, sa, sb, out_c)
        self.ref.reference_bmm_fp8(A, B, sa, sb, out_r)
        assert torch.allclose(out_c.float(), out_r.float(), atol=1e-2)

    @pytest.mark.parametrize("mode", ["K", "MN"])
    def test_groupwise_matches_reference(self, monkeypatch, mode):
        torch = self.torch
        torch.manual_seed(0)
        monkeypatch.setitem(self.cann_gemm._gemm_fp8_nt_groupwise_cann.__globals__, '_fp8_grouped_gemm', self._naive_grouped_gemm)
        m, n, k = 32, 128, 256
        gm, gn, gk = 1, 128, 128
        a = (torch.randn(m, k) * 0.2).to(torch.float8_e4m3fn)
        b = (torch.randn(n, k) * 0.2).to(torch.float8_e4m3fn)
        sa = torch.rand(m // gm, k // gk) * 0.01 + 0.005
        sb = torch.rand(n // gn, k // gk) * 0.01 + 0.005
        if mode == "MN":
            sa, sb = sa.t().contiguous(), sb.t().contiguous()
        out_c = torch.empty(m, n, dtype=torch.bfloat16)
        out_r = torch.empty(m, n, dtype=torch.bfloat16)
        self.cann_gemm._gemm_fp8_nt_groupwise_cann(
            a, b, sa, sb, mode, (gm, gn, gk), out_c
        )
        self.ref.reference_gemm_fp8_nt_groupwise(
            a, b, sa, sb, mode, (gm, gn, gk), out_r
        )
        assert torch.allclose(out_c.float(), out_r.float(), atol=1e-2)
