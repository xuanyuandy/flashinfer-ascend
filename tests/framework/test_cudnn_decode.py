"""cudnn_batch_decode_with_kv_cache：注册、调度与数值（§3.1 / §5.3 / §12）。

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
    DispatchError,
    KernelDescriptor,
)
from flashinfer.runtime.signature import InputSignature

NPU_HW = HardwareSignature(soc_model="Ascend910B", aicore_count=20)
HOST_HW = HardwareSignature()  # unknown：host 开发机

DECODE_ID_CANN = "cann/decode/cudnn_batch_decode_fia_v1"
DECODE_ID_REF = "reference/decode/cudnn_batch_decode_v1"


def _decode_sig(q_dtype="float16") -> InputSignature:
    return InputSignature(
        op="decode.cudnn_batch_decode_with_kv_cache",
        dtypes=(q_dtype, q_dtype, q_dtype, "int64", "int32"),
        shapes=((3, 8, 64), (10, 2, 16, 64), (10, 2, 16, 64), (3,), (3, 3)),
        specializations=(("max_sequence_kv", "48"), ("scale", "0.125")),
    )


class TestCudnnDecodeRegistration:
    def test_global_registry_has_decode_kernels(self):
        ids = {d.kernel_id for d in finpu.get_global_registry().descriptors()}
        assert {DECODE_ID_CANN, DECODE_ID_REF} <= ids

    def test_cann_descriptor_declares_support_set(self):
        descriptor = finpu.get_global_registry().get(DECODE_ID_CANN)
        assert descriptor.backend == Backend.CANN
        assert descriptor.capability.soc_models == ("Ascend910B", "Ascend910C", "Ascend950")
        assert set(descriptor.capability.dtypes) == {
            "float16",
            "bfloat16",
            "int32",
            "int64",
        }


class TestCudnnDecodeDispatch:
    def test_auto_prefers_cann_on_npu(self):
        decision = finpu.dispatch(_decode_sig(), NPU_HW)
        assert decision.kernel_id == DECODE_ID_CANN
        assert decision.backend == Backend.CANN

    def test_reference_excluded_from_auto(self):
        decision = finpu.dispatch(_decode_sig(), NPU_HW)
        report = {r.kernel_id: r for r in decision.reports}[DECODE_ID_REF]
        assert not report.accepted
        assert any("opt-in" in reason for reason in report.reasons)

    def test_reference_via_explicit_opt_in(self):
        decision = finpu.dispatch(_decode_sig(), NPU_HW, backend="reference")
        assert decision.kernel_id == DECODE_ID_REF

    def test_explicit_kernel_id_lock(self):
        decision = finpu.dispatch(
            _decode_sig(), NPU_HW, kernel_id=DECODE_ID_REF
        )
        assert decision.kernel_id == DECODE_ID_REF
        assert decision.backend == Backend.REFERENCE

    def test_host_auto_has_no_candidate(self):
        # host（unknown SoC）不在 CANN 声明支持集内，reference 需 opt-in
        with pytest.raises(DispatchError, match="无可用 kernel"):
            finpu.dispatch(_decode_sig(), HOST_HW)

    def test_unsupported_dtype_rejected(self):
        with pytest.raises(DispatchError) as excinfo:
            finpu.dispatch(_decode_sig("float64"), NPU_HW)
        assert "float64" in str(excinfo.value)


class _FakeTensor:
    """仅携带 dtype/shape 协议的轻量假张量（§14.1：host 单测不依赖 torch）。"""

    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape


def _fake_inputs():
    return dict(
        q=_FakeTensor("float16", (3, 8, 64)),
        k=_FakeTensor("float16", (10, 2, 16, 64)),
        v=_FakeTensor("float16", (10, 2, 16, 64)),
        lens=_FakeTensor("int64", (3,)),
        tables=_FakeTensor("int32", (3, 3)),
        out=_FakeTensor("float16", (3, 8, 64)),
    )


class TestCudnnDecodeFacadeFlow:
    """门面全链路：假张量 + custom probe kernel（kernel_id 显式锁定）。"""

    def test_facade_flow(self, monkeypatch, clean_global_registry):
        calls = []
        finpu.register_custom_kernel(
            KernelDescriptor(
                kernel_id="custom/decode/cudnn_batch_decode_probe",
                backend=Backend.CUSTOM,
                capability=CapabilitySpec(
                    op="decode.cudnn_batch_decode_with_kv_cache"
                ),
                runner=lambda *args: calls.append(args) or args[-1],
            )
        )
        monkeypatch.setattr(
            "flashinfer.decode.current_hardware", lambda: NPU_HW
        )
        t = _fake_inputs()
        result = finpu.cudnn_batch_decode_with_kv_cache(
            t["q"], t["k"], t["v"], 0.125, _FakeTensor("uint8", (1024,)),
            max_sequence_kv=48,
            actual_seq_lens_kv=t["lens"],
            block_tables=t["tables"],
            out=t["out"],
            kernel_id="custom/decode/cudnn_batch_decode_probe",
        )
        assert result is t["out"]
        (q, k, v, scale, max_kv, lens, tables, out), = calls
        assert q is t["q"] and k is t["k"] and v is t["v"]
        assert scale == pytest.approx(0.125) and max_kv == 48
        assert lens is t["lens"] and tables is t["tables"] and out is t["out"]

    def test_missing_lens_or_tables_rejected(self, clean_global_registry):
        t = _fake_inputs()
        with pytest.raises(ValueError, match="不能为空"):
            finpu.cudnn_batch_decode_with_kv_cache(
                t["q"], t["k"], t["v"], 0.125, None,
                max_sequence_kv=48, block_tables=t["tables"], out=t["out"],
            )
        with pytest.raises(ValueError, match="不能为空"):
            finpu.cudnn_batch_decode_with_kv_cache(
                t["q"], t["k"], t["v"], 0.125, None,
                max_sequence_kv=48, actual_seq_lens_kv=t["lens"], out=t["out"],
            )

    def test_batch_offsets_rejected(self, clean_global_registry):
        t = _fake_inputs()
        with pytest.raises(NotImplementedError, match="batch_offsets"):
            finpu.cudnn_batch_decode_with_kv_cache(
                t["q"], t["k"], t["v"], 0.125, None,
                max_sequence_kv=48,
                actual_seq_lens_kv=t["lens"],
                block_tables=t["tables"],
                batch_offsets_q=t["lens"],
                out=t["out"],
            )


@pytest.mark.usefixtures("torch_for_class")
class TestCudnnDecodeNumerics:
    def _make_inputs(self, dtype, device):
        """B=3, GQA(8/2), page=16, 末页部分填充；lens [5, 17, 40]。"""
        torch = self.torch
        torch.manual_seed(0)
        B, H_qo, H_kv, D, ps, nps = 3, 8, 2, 64, 16, 3
        total_pages = B * nps + 1
        q = torch.randn(B, H_qo, D, dtype=dtype, device=device)
        k = torch.randn(total_pages, H_kv, ps, D, dtype=dtype, device=device)
        v = torch.randn(total_pages, H_kv, ps, D, dtype=dtype, device=device)
        lens = torch.tensor([5, 17, 40], dtype=torch.int64, device="cpu")
        perm = torch.randperm(total_pages)[: B * nps].reshape(B, nps)
        tables = perm.to(torch.int32).to(device)
        scale = 1.0 / (D ** 0.5)
        return q, k, v, scale, nps * ps, lens, tables

    def _fp64_decode(self, q, k, v, scale, max_kv, lens, tables):
        """与实现无关的 fp64 循环参考（CPU）。"""
        torch = self.torch
        B, H_qo, D = q.shape
        _, H_kv, ps, D_v = v.shape
        group = H_qo // H_kv
        out = torch.empty(B, H_qo, D_v, dtype=torch.float64)
        for b in range(B):
            kv_len = min(int(lens[b]), max_kv)
            kb = k[tables[b].long()].double().permute(0, 2, 1, 3)
            vb = v[tables[b].long()].double().permute(0, 2, 1, 3)
            kb = kb.reshape(-1, H_kv, D)[:kv_len].repeat_interleave(group, 1)
            vb = vb.reshape(-1, H_kv, D_v)[:kv_len].repeat_interleave(group, 1)
            s = torch.einsum("hd,lhd->hl", q[b].double(), kb) * scale
            out[b] = torch.einsum(
                "hl,lhd->hd", torch.softmax(s, -1), vb
            )
        return out

    def test_reference_cpu(self):
        torch = self.torch
        q, k, v, scale, max_kv, lens, tables = self._make_inputs(
            torch.float32, "cpu"
        )
        out = finpu.cudnn_batch_decode_with_kv_cache(
            q, k, v, scale, torch.empty(0),
            max_sequence_kv=max_kv, actual_seq_lens_kv=lens,
            block_tables=tables, backend="reference",
        )
        expected = self._fp64_decode(q, k, v, scale, max_kv, lens, tables)
        assert torch.allclose(out.double(), expected, atol=1e-4)

    @pytest.mark.usefixtures("npu_device")
    def test_auto_dispatch_picks_cann_on_npu(self):
        hardware = finpu.current_hardware()
        assert hardware.soc_model in ("Ascend910B", "Ascend910C", "Ascend950")
        assert finpu.dispatch(_decode_sig(), hardware).kernel_id == DECODE_ID_CANN

    @pytest.mark.usefixtures("npu_device")
    @pytest.mark.parametrize(
        "dtype,atol", [("float16", 2e-2), ("bfloat16", 8e-2)]
    )
    def test_decode_npu_vs_fp64(self, dtype, atol):
        torch = self.torch
        dt = getattr(torch, dtype)
        q, k, v, scale, max_kv, lens, tables = self._make_inputs(dt, "npu")
        out = finpu.cudnn_batch_decode_with_kv_cache(
            q, k, v, scale, torch.empty(0, device="npu"),
            max_sequence_kv=max_kv, actual_seq_lens_kv=lens,
            block_tables=tables,  # auto → CANN FIA
        )
        expected = self._fp64_decode(
            q.cpu(), k.cpu(), v.cpu(), scale, max_kv, lens, tables.cpu()
        )
        assert torch.allclose(out.cpu().double(), expected, atol=atol)

    @pytest.mark.usefixtures("npu_device")
    def test_cann_vs_reference_consistency(self):
        """M1 验收口径：同输入，CANN 与 reference 两 backend 数值一致。"""
        torch = self.torch
        q, k, v, scale, max_kv, lens, tables = self._make_inputs(
            torch.float16, "npu"
        )
        y_cann = finpu.cudnn_batch_decode_with_kv_cache(
            q, k, v, scale, torch.empty(0, device="npu"),
            max_sequence_kv=max_kv, actual_seq_lens_kv=lens,
            block_tables=tables, backend="cann",
        )
        y_ref = finpu.cudnn_batch_decode_with_kv_cache(
            q.cpu(), k.cpu(), v.cpu(), scale, torch.empty(0),
            max_sequence_kv=max_kv, actual_seq_lens_kv=lens,
            block_tables=tables.cpu(), backend="reference",
        )
        assert torch.allclose(y_cann.cpu().double(), y_ref.double(), atol=2e-2)

    @pytest.mark.usefixtures("npu_device")
    def test_out_buffer_reused(self):
        torch = self.torch
        q, k, v, scale, max_kv, lens, tables = self._make_inputs(
            torch.float16, "npu"
        )
        buf = torch.empty(3, 8, 64, dtype=torch.float16, device="npu")
        result = finpu.cudnn_batch_decode_with_kv_cache(
            q, k, v, scale, torch.empty(0, device="npu"),
            max_sequence_kv=max_kv, actual_seq_lens_kv=lens,
            block_tables=tables, out=buf,
        )
        assert result is buf
