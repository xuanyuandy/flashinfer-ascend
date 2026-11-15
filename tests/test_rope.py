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


class TestRopeNumerics:
    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")

    def _require_npu(self):
        pytest.importorskip("torch_npu")
        if not self.torch.npu.is_available():
            pytest.skip("无 NPU 环境")

    def _fp64_rope(self, t3, cs, is_neox):
        """t3: (nnz, H, D) fp64；cs: (nnz, rotary_dim) fp64；仅旋转前 rd 维。"""
        torch = self.torch
        rd = cs.shape[-1]
        half = rd // 2
        cos = cs[:, None, :half]
        sin = cs[:, None, half:]
        out = t3.clone()
        x = t3[..., :rd]
        if is_neox:
            x1, x2 = x[..., :half], x[..., half:]
            out[..., :rd] = torch.cat(
                [x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1
            )
        else:
            xe, xo = x[..., 0::2], x[..., 1::2]
            out[..., :rd] = torch.stack(
                [xe * cos - xo * sin, xo * cos + xe * sin], dim=-1
            ).flatten(-2)
        return out

    def _make_inputs(self, dtype, device, nnz=8, nq=4, nk=2, hs=128, rd=128):
        torch = self.torch
        torch.manual_seed(0)
        pos = torch.randint(0, 2048, (nnz,), dtype=torch.int64, device=device)
        q = torch.randn(nnz, nq * hs, dtype=dtype, device=device)
        k = torch.randn(nnz, nk * hs, dtype=dtype, device=device)
        cache = torch.randn(2048, rd, dtype=torch.float32, device=device)
        return pos, q, k, cache

    @pytest.mark.parametrize("is_neox", [True, False])
    @pytest.mark.parametrize("rd", [128, 64])  # 全旋转 / 部分旋转（rd < head_size）
    def test_reference_cpu(self, is_neox, rd):
        torch = self.torch
        pos, q, k, cache = self._make_inputs(torch.float32, "cpu", rd=rd)
        q0, k0 = q.clone(), k.clone()
        result = finpu.apply_rope_with_cos_sin_cache_inplace(
            pos, q, k, 128, cache, is_neox=is_neox, backend="reference"
        )
        assert result is None
        cs = cache.double().index_select(0, pos)
        exp_q = self._fp64_rope(q0.double().view(-1, 4, 128), cs, is_neox)
        exp_k = self._fp64_rope(k0.double().view(-1, 2, 128), cs, is_neox)
        assert torch.allclose(q.double().view(-1, 4, 128), exp_q, atol=1e-5)
        assert torch.allclose(k.double().view(-1, 2, 128), exp_k, atol=1e-5)
        if rd < 128:  # 未旋转维保持原值
            assert torch.equal(
                q.view(-1, 4, 128)[..., rd:], q0.view(-1, 4, 128)[..., rd:]
            )

    def test_auto_dispatch_picks_cann_on_npu(self):
        self._require_npu()
        hardware = finpu.current_hardware()
        assert hardware.soc_model in ("Ascend910B", "Ascend910C", "Ascend950")
        assert finpu.dispatch(_rope_sig(), hardware).kernel_id == ROPE_ID_CANN

    @pytest.mark.parametrize("is_neox", [True, False])
    @pytest.mark.parametrize(
        "dtype,atol",
        [("float16", 2e-2), ("bfloat16", 8e-2), ("float32", 1e-4)],
    )
    def test_rope_npu_vs_fp64(self, dtype, atol, is_neox):
        self._require_npu()
        torch = self.torch
        dt = getattr(torch, dtype)
        pos, q, k, cache = self._make_inputs(dt, "npu")
        q0, k0 = q.clone().cpu(), k.clone().cpu()
        finpu.apply_rope_with_cos_sin_cache_inplace(
            pos, q, k, 128, cache, is_neox=is_neox  # auto → CANN
        )
        cs = cache.cpu().double().index_select(0, pos.cpu())
        exp_q = self._fp64_rope(q0.double().view(-1, 4, 128), cs, is_neox)
        exp_k = self._fp64_rope(k0.double().view(-1, 2, 128), cs, is_neox)
        assert torch.allclose(q.cpu().double().view(-1, 4, 128), exp_q, atol=atol)
        assert torch.allclose(k.cpu().double().view(-1, 2, 128), exp_k, atol=atol)

    @pytest.mark.parametrize("is_neox", [True, False])
    def test_rope_partial_rotary_npu(self, is_neox):
        """rotary_dim < head_size：仅旋转前 rd 维，其余维原样保留。"""
        self._require_npu()
        torch = self.torch
        rd = 64
        pos, q, k, cache = self._make_inputs(torch.float16, "npu", rd=rd)
        q0, k0 = q.clone().cpu(), k.clone().cpu()
        finpu.apply_rope_with_cos_sin_cache_inplace(
            pos, q, k, 128, cache, is_neox=is_neox
        )
        cs = cache.cpu().double().index_select(0, pos.cpu())
        exp_q = self._fp64_rope(q0.double().view(-1, 4, 128), cs, is_neox)
        exp_k = self._fp64_rope(k0.double().view(-1, 2, 128), cs, is_neox)
        assert torch.allclose(
            q.cpu().double().view(-1, 4, 128), exp_q, atol=2e-2
        )
        assert torch.allclose(
            k.cpu().double().view(-1, 2, 128), exp_k, atol=2e-2
        )

    def test_cann_vs_reference_consistency(self):
        """M1 验收口径：同输入，CANN 与 reference 两 backend 数值一致。"""
        self._require_npu()
        torch = self.torch
        pos, q, k, cache = self._make_inputs(torch.float16, "npu")
        q1, k1 = q.clone(), k.clone()
        q2, k2 = q.cpu(), k.cpu()
        finpu.apply_rope_with_cos_sin_cache_inplace(
            pos, q1, k1, 128, cache, backend="cann"
        )
        finpu.apply_rope_with_cos_sin_cache_inplace(
            pos.cpu(), q2, k2, 128, cache.cpu(), backend="reference"
        )
        assert torch.allclose(q1.cpu().double(), q2.double(), atol=2e-2)
        assert torch.allclose(k1.cpu().double(), k2.double(), atol=2e-2)

    def test_real_tensor_cache_dtype_validation(self):
        torch = self.torch
        pos, q, k, cache = self._make_inputs(torch.float16, "cpu")
        with pytest.raises(ValueError, match="float32"):
            finpu.apply_rope_with_cos_sin_cache_inplace(
                pos, q, k, 128, cache.half()
            )
