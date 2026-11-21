"""cascade.merge_state：注册、调度、门面链路与数值（§3.1 / §5.3 / §12）。

底层 kernel：cann（npu_attention_update，update_type=1，Ascend910B）/
reference（fp32 计算，-inf lse 安全）。数值语义与上游
flashinfer.cascade.merge_state 一致：
``s_m = log(exp(s_a)+exp(s_b))``，``v_m = v_a·exp(s_a-s_m) + v_b·exp(s_b-s_m)``。
"""

from __future__ import annotations

import pytest

import flashinfer as finpu
from flashinfer.runtime.hardware import HardwareSignature
from flashinfer.runtime.schema import (
    Backend,
    CapabilitySpec,
    KernelDescriptor,
)
from flashinfer.runtime.signature import InputSignature

NPU_HW = HardwareSignature(soc_model="Ascend910B", aicore_count=20)

REF_ID = "reference/cascade/merge_state_v1"
CANN_ID = "cann/cascade/merge_state_attention_update_v1"


def _sig() -> InputSignature:
    return InputSignature(
        op="cascade.merge_state",
        dtypes=("bfloat16", "float32", "bfloat16", "float32"),
        shapes=((16, 4, 32), (16, 4), (16, 4, 32), (16, 4)),
    )


class TestMergeStateRegistration:
    def test_global_registry_has_kernels(self):
        ids = {d.kernel_id for d in finpu.get_global_registry().descriptors()}
        assert {REF_ID, CANN_ID} <= ids


class TestMergeStateDispatch:
    def test_auto_prefers_cann_on_npu(self):
        assert finpu.dispatch(_sig(), NPU_HW).kernel_id == CANN_ID

    def test_reference_via_opt_in(self):
        d = finpu.dispatch(_sig(), NPU_HW, backend="reference")
        assert d.kernel_id == REF_ID


class _FakeTensor:
    """仅携带 dtype/shape 协议的轻量假张量（§14.1：host 单测不依赖 torch）。"""

    def __init__(self, dtype, shape, ndim=None):
        self.dtype = dtype
        self.shape = shape
        self.ndim = ndim if ndim is not None else len(shape)


class TestMergeStateFacadeFlow:
    def test_facade_flow(self, monkeypatch, clean_global_registry):
        calls = []
        finpu.register_custom_kernel(
            KernelDescriptor(
                kernel_id="custom/cascade/merge_state_probe",
                backend=Backend.CUSTOM,
                capability=CapabilitySpec(op="cascade.merge_state"),
                runner=lambda *args: calls.append(args) or ("v", "s"),
            )
        )
        monkeypatch.setattr(
            "flashinfer.cascade.current_hardware", lambda: NPU_HW
        )
        va = _FakeTensor("bfloat16", (16, 4, 32))
        sa = _FakeTensor("float32", (16, 4))
        vb = _FakeTensor("bfloat16", (16, 4, 32))
        sb = _FakeTensor("float32", (16, 4))
        result = finpu.merge_state(
            va, sa, vb, sb, kernel_id="custom/cascade/merge_state_probe"
        )
        assert result == ("v", "s")
        (va_, sa_, vb_, sb_), = calls
        assert va_ is va and sa_ is sa and vb_ is vb and sb_ is sb

    def test_shape_validation(self, clean_global_registry):
        v = _FakeTensor("bfloat16", (16, 4, 32))
        s = _FakeTensor("float32", (16, 4))
        with pytest.raises(ValueError, match="形状须一致"):
            finpu.merge_state(v, s, _FakeTensor("bfloat16", (8, 4, 32)), s)
        with pytest.raises(ValueError, match="3D"):
            finpu.merge_state(
                _FakeTensor("bfloat16", (16, 32)), s,
                _FakeTensor("bfloat16", (16, 32)), s,
            )
        with pytest.raises(ValueError, match="num_heads"):
            finpu.merge_state(v, _FakeTensor("float32", (16, 2)), v, s)


@pytest.mark.usefixtures("torch_for_class")
class TestMergeStateNumerics:
    def _expected(self, va, sa, vb, sb):
        torch = self.torch
        s_m = torch.logaddexp(sa, sb)
        v_m = (
            va.float() * torch.exp(sa - s_m).unsqueeze(-1)
            + vb.float() * torch.exp(sb - s_m).unsqueeze(-1)
        )
        return v_m, s_m

    def test_reference_vs_naive_fp64(self):
        torch = self.torch
        torch.manual_seed(0)
        va = torch.randn(16, 4, 32)
        sa = torch.randn(16, 4)
        vb = torch.randn(16, 4, 32)
        sb = torch.randn(16, 4)
        v_m, s_m = finpu.merge_state(va, sa, vb, sb, backend="reference")
        # fp64 基线
        s_e = torch.logaddexp(sa.double(), sb.double())
        v_e = (
            va.double() * torch.exp(sa.double() - s_e).unsqueeze(-1)
            + vb.double() * torch.exp(sb.double() - s_e).unsqueeze(-1)
        )
        assert v_m.dtype == va.dtype and s_m.dtype == torch.float32
        assert torch.allclose(v_m.double(), v_e, atol=1e-6)
        assert torch.allclose(s_m.double(), s_e, atol=1e-6)

    def test_reference_neg_inf_lse(self):
        """s=-inf（空 KV 段）时输出为另一段本身，不产生 nan。"""
        torch = self.torch
        va = torch.randn(4, 2, 8)
        sa = torch.full((4, 2), float("-inf"))
        vb = torch.randn(4, 2, 8)
        sb = torch.randn(4, 2)
        v_m, s_m = finpu.merge_state(va, sa, vb, sb, backend="reference")
        assert not torch.isnan(v_m).any() and not torch.isnan(s_m).any()
        assert torch.allclose(v_m, vb)
        assert torch.allclose(s_m, sb)

    def test_reference_two_way_merge_equiv_full_attention(self):
        """merge 结果等价于对拼接 KV 直接做注意力（小尺度复算）。"""
        torch = self.torch
        torch.manual_seed(0)
        q = torch.randn(1, 2, 8)  # (1, heads, hd)
        ka, kb = torch.randn(5, 2, 8), torch.randn(3, 2, 8)
        va_, vb_ = torch.randn(5, 2, 8), torch.randn(3, 2, 8)

        def seg_attn(k, v):
            score = torch.einsum("qhd,khd->hqk", q, k) / 8**0.5  # (heads, q, k)
            p = score.softmax(-1)
            out = torch.einsum("hqk,khd->qhd", p, v)
            lse = torch.logsumexp(score, dim=-1).t()  # (q, heads)
            return out, lse

        oa, sa = seg_attn(ka, va_)
        ob, sb = seg_attn(kb, vb_)
        v_m, s_m = finpu.merge_state(
            oa, sa, ob, sb, backend="reference"
        )
        # 拼接 KV 全量注意力
        k_full = torch.cat([ka, kb])
        v_full = torch.cat([va_, vb_])
        score = torch.einsum("qhd,khd->hqk", q, k_full) / 8**0.5
        p = score.softmax(-1)
        o_full = torch.einsum("hqk,khd->qhd", p, v_full)
        s_full = torch.logsumexp(score, dim=-1).t()
        assert torch.allclose(v_m, o_full, atol=1e-5)
        assert torch.allclose(s_m, s_full, atol=1e-5)

    def test_reference_dtype_preserved(self):
        torch = self.torch
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            va = torch.randn(4, 2, 8).to(dtype)
            vb = torch.randn(4, 2, 8).to(dtype)
            sa = torch.randn(4, 2)
            sb = torch.randn(4, 2)
            v_m, s_m = finpu.merge_state(va, sa, vb, sb, backend="reference")
            assert v_m.dtype == dtype and s_m.dtype == torch.float32


@pytest.mark.usefixtures("torch_for_class", "npu_device")
class TestMergeStateCannNpu:
    """CANN kernel（npu_attention_update）NPU 数值验证：与 fp64 基线对拍。"""

    @pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
    def test_cann_vs_fp64(self, dtype_name):
        torch = self.torch
        torch.manual_seed(0)
        dev = torch.device("npu", torch.npu.current_device())
        dtype = getattr(torch, dtype_name)
        va = torch.randn(16, 4, 32, device=dev, dtype=dtype)
        sa = torch.randn(16, 4, device=dev)
        vb = torch.randn(16, 4, 32, device=dev, dtype=dtype)
        sb = torch.randn(16, 4, device=dev)
        v_m, s_m = finpu.merge_state(va, sa, vb, sb, backend="cann")
        # fp64 基线（CPU）
        sad, sbd = sa.cpu().double(), sb.cpu().double()
        s_e = torch.logaddexp(sad, sbd)
        v_e = (
            va.cpu().double() * torch.exp(sad - s_e).unsqueeze(-1)
            + vb.cpu().double() * torch.exp(sbd - s_e).unsqueeze(-1)
        )
        assert v_m.dtype == dtype and s_m.dtype == torch.float32
        assert v_m.shape == va.shape and s_m.shape == sa.shape
        rel_v = (v_m.cpu().double() - v_e).norm() / v_e.norm()
        assert rel_v < 0.01
        assert torch.allclose(s_m.cpu().double(), s_e, atol=1e-4)

    def test_cann_vs_reference(self):
        """同一输入下 cann 与 reference 结果一致。"""
        torch = self.torch
        torch.manual_seed(1)
        dev = torch.device("npu", torch.npu.current_device())
        va = torch.randn(8, 4, 16)
        sa = torch.randn(8, 4)
        vb = torch.randn(8, 4, 16)
        sb = torch.randn(8, 4)
        v_npu, s_npu = finpu.merge_state(
            va.to(dev, torch.bfloat16), sa.to(dev),
            vb.to(dev, torch.bfloat16), sb.to(dev),
            backend="cann",
        )
        v_ref, s_ref = finpu.merge_state(va, sa, vb, sb, backend="reference")
        rel = (v_npu.cpu().float() - v_ref).norm() / v_ref.norm()
        assert rel < 0.01
        assert torch.allclose(s_npu.cpu(), s_ref, atol=1e-4)
