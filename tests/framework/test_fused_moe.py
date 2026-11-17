"""fused_moe 系列：注册、调度、门面链路与数值（§3.1 / §5.3 / §12）。

- RoutingMethodType / reorder_rows_for_gated_act_gemm：纯 host 工具；
- cutlass_fused_moe：reference + cann（npu_grouped_matmul，容器内 NPU
  数值验证）；
- trtllm_fp8_*：reference + cann（纯 fp8 小算子级联，仅 950：token 聚簇
  → fp8 grouped GEMM → scale 折算 → swiglu → 中间激活动态量化回 fp8
  → fc2 fp8 GEMM → 反置换 + 路由权重聚合；block_scale 按 K 方向 128
  分块累加）。
"""

from __future__ import annotations

import importlib

import pytest

import flashinfer as finpu

# finpu.fused_moe 顶层属性已被同名函数遮蔽，门面模块经 import_module 获取
_MOE_MOD = importlib.import_module("flashinfer.fused_moe")
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
NPU_HW_950 = HardwareSignature(soc_model="Ascend950", aicore_count=20)

CUTLASS_ID_REF = "reference/fused_moe/cutlass_fused_moe_v1"
CUTLASS_ID_CANN = "cann/fused_moe/cutlass_fused_moe_gmm_v1"
PT_ID_REF = "reference/fused_moe/trtllm_fp8_per_tensor_scale_moe_v1"
BS_ID_REF = "reference/fused_moe/trtllm_fp8_block_scale_moe_v1"
PT_ID_CANN = "cann/fused_moe/fp8_per_tensor_scale_moe_cascade_v1"
BS_ID_CANN = "cann/fused_moe/fp8_block_scale_moe_cascade_v1"


def _cutlass_sig() -> InputSignature:
    return InputSignature(
        op="fused_moe.cutlass_fused_moe",
        dtypes=("bfloat16", "int64", "float32", "bfloat16", "bfloat16"),
        shapes=((8, 32), (8, 2), (8, 2), (4, 32, 32), (4, 32, 16)),
        specializations=(("output_dtype", "bfloat16"),),
    )


def _pt_sig() -> InputSignature:
    return InputSignature(
        op="fused_moe.trtllm_fp8_per_tensor_scale_moe",
        dtypes=("float32", "float8_e4m3fn", "float8_e4m3fn", "float8_e4m3fn"),
        shapes=((8, 8), (8, 32), (8, 32, 32), (8, 32, 16)),
        specializations=(("top_k", 2), ("routing_method_type", 0)),
    )


def _bs_sig() -> InputSignature:
    return InputSignature(
        op="fused_moe.trtllm_fp8_block_scale_moe",
        dtypes=("float32", "float8_e4m3fn", "float8_e4m3fn", "float8_e4m3fn"),
        shapes=((8, 8), (8, 32), (8, 32, 32), (8, 32, 16)),
        specializations=(("top_k", 2), ("routing_method_type", 2)),
    )


class TestUpstreamNameAliases:
    """Incompatible upstream names must not masquerade as drop-in aliases."""

    def test_incompatible_aliases_are_absent(self):
        import importlib

        module = importlib.import_module("flashinfer.fused_moe")
        for name in ("cutlass_fused_moe", "trtllm_fp8_per_tensor_scale_moe",
                     "trtllm_fp8_block_scale_moe"):
            assert not hasattr(finpu, name)
            assert not hasattr(module, name)
        assert callable(finpu.fused_moe)
        assert callable(finpu.fp8_per_tensor_scale_moe)
        assert callable(finpu.fp8_block_scale_moe)


class TestRoutingMethodType:
    def test_enum_values_match_upstream(self):
        R = finpu.RoutingMethodType
        assert (R.Default, R.Renormalize, R.DeepSeekV3) == (0, 1, 2)
        assert (R.Llama4, R.RenormalizeNaive, R.Unspecified) == (3, 4, 10)

    def test_is_int_enum(self):
        assert finpu.RoutingMethodType.DeepSeekV3 == 2
        assert int(finpu.RoutingMethodType.Llama4) == 3


class TestFusedMoeRegistration:
    def test_global_registry_has_moe_kernels(self):
        ids = {d.kernel_id for d in finpu.get_global_registry().descriptors()}
        assert {CUTLASS_ID_REF, CUTLASS_ID_CANN, PT_ID_REF, BS_ID_REF,
                PT_ID_CANN, BS_ID_CANN} <= ids


class TestFusedMoeDispatch:
    def test_cutlass_auto_prefers_cann_on_npu(self):
        assert finpu.dispatch(_cutlass_sig(), NPU_HW).kernel_id == CUTLASS_ID_CANN

    def test_cutlass_reference_via_opt_in(self):
        d = finpu.dispatch(_cutlass_sig(), NPU_HW, backend="reference")
        assert d.kernel_id == CUTLASS_ID_REF

    def test_trtllm_reference_via_opt_in(self):
        d = finpu.dispatch(_pt_sig(), NPU_HW, backend="reference")
        assert d.kernel_id == PT_ID_REF
        d = finpu.dispatch(_bs_sig(), NPU_HW, backend="reference")
        assert d.kernel_id == BS_ID_REF

    def test_trtllm_auto_no_candidate_on_910b(self):
        # fp8 CANN 级联 kernel 仅注册 950；910B 上 auto 无候选
        # （reference 不进热路径，需显式 opt-in）
        with pytest.raises(DispatchError):
            finpu.dispatch(_pt_sig(), NPU_HW)
        with pytest.raises(DispatchError):
            finpu.dispatch(_bs_sig(), NPU_HW)

    def test_trtllm_auto_prefers_cann_on_950(self):
        assert finpu.dispatch(_pt_sig(), NPU_HW_950).kernel_id == PT_ID_CANN
        assert finpu.dispatch(_bs_sig(), NPU_HW_950).kernel_id == BS_ID_CANN


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


class TestCutlassFacadeFlow:
    def test_facade_flow(self, monkeypatch, _clean_global_registry):
        calls = []
        finpu.register_custom_kernel(
            KernelDescriptor(
                kernel_id="custom/fused_moe/cutlass_probe",
                backend=Backend.CUSTOM,
                capability=CapabilitySpec(op="fused_moe.cutlass_fused_moe"),
                runner=lambda *args: calls.append(args) or args[-1],
            )
        )
        monkeypatch.setattr(
            _MOE_MOD, "current_hardware", lambda: NPU_HW
        )
        x = _FakeTensor("bfloat16", (8, 32))
        idx = _FakeTensor("int64", (8, 2))
        w = _FakeTensor("float32", (8, 2))
        w1 = _FakeTensor("bfloat16", (4, 32, 32))
        w2 = _FakeTensor("bfloat16", (4, 32, 16))
        out = _FakeTensor("bfloat16", (8, 32))
        result = finpu.fused_moe(
            x, idx, w, w1, w2, "bfloat16", [],
            output=out, kernel_id="custom/fused_moe/cutlass_probe",
        )
        assert result is out
        (x_, idx_, w_, w1_, w2_, od_, out_, ep_size_, ep_rank_, moe_ctx_), = calls
        assert x_ is x and idx_ is idx and out_ is out and od_ == "bfloat16"
        assert ep_size_ == 1 and ep_rank_ == 0 and moe_ctx_ is None

    def test_unsupported_features_raise(self, _clean_global_registry):
        x = _FakeTensor("bfloat16", (8, 32))
        with pytest.raises(NotImplementedError, match="FP8 Block"):
            finpu.fused_moe(
                x, x, x, x, x, "bfloat16", [], use_fp8_block_scaling=True
            )
        with pytest.raises(NotImplementedError, match="W4A8"):
            finpu.fused_moe(
                x, x, x, x, x, "bfloat16", [], use_w4a8_group_scaling=True
            )
        with pytest.raises(NotImplementedError, match="min latency"):
            finpu.fused_moe(
                x, x, x, x, x, "bfloat16", [], min_latency_mode=True
            )
        with pytest.raises(NotImplementedError, match="单卡"):
            finpu.fused_moe(x, x, x, x, x, "bfloat16", [], tp_size=2)
        x8 = _FakeTensor("float8_e4m3fn", (8, 32))
        with pytest.raises(NotImplementedError, match="fp8"):
            finpu.fused_moe(x8, x, x, x, x, "bfloat16", [])


class _FakeSymmBuffer:
    """duck-typed 通信域对象（与 cann_ops_transformer SymmBuffer 属性对齐）。"""

    def __init__(self, ep_world_size=2, rank_id=0):
        self.ep_world_size = ep_world_size
        self.rank_id = rank_id


class TestFusedMoeEpContext:
    """ep_size > 1：moe_context 校验与透传（不依赖真实多卡环境）。"""

    def test_ep_without_context_raises(self, _clean_global_registry):
        x = _FakeTensor("bfloat16", (8, 32))
        with pytest.raises(ValueError, match="必传 moe_context"):
            finpu.fused_moe(x, x, x, x, x, "bfloat16", [], ep_size=2)

    def test_single_card_context_ignored(self, monkeypatch, _clean_global_registry):
        calls = []
        finpu.register_custom_kernel(
            KernelDescriptor(
                kernel_id="custom/fused_moe/ep1_probe",
                backend=Backend.CUSTOM,
                capability=CapabilitySpec(op="fused_moe.cutlass_fused_moe"),
                runner=lambda *args: calls.append(args) or args[6],
            )
        )
        monkeypatch.setattr(_MOE_MOD, "current_hardware", lambda: NPU_HW)
        x = _FakeTensor("bfloat16", (8, 32))
        out = _FakeTensor("bfloat16", (8, 32))
        result = finpu.fused_moe(
            x, x, x, x, x, "bfloat16", [],
            output=out, moe_context=_FakeSymmBuffer(1),
            kernel_id="custom/fused_moe/ep1_probe",
        )
        assert result is out  # ep_size=1 传 moe_context 不报错、照常执行

    def test_context_ep_mismatch_raises(self, _clean_global_registry):
        x = _FakeTensor("bfloat16", (8, 32))
        with pytest.raises(ValueError, match="ep_world_size"):
            finpu.fused_moe(
                x, x, x, x, x, "bfloat16", [],
                ep_size=4, moe_context=_FakeSymmBuffer(2),
            )

    def test_context_rank_mismatch_raises(self, _clean_global_registry):
        x = _FakeTensor("bfloat16", (8, 32))
        with pytest.raises(ValueError, match="rank_id"):
            finpu.fused_moe(
                x, x, x, x, x, "bfloat16", [],
                ep_size=2, ep_rank=1, moe_context=_FakeSymmBuffer(2, 0),
            )

    def test_ep_context_passed_to_runner(self, monkeypatch, _clean_global_registry):
        calls = []
        finpu.register_custom_kernel(
            KernelDescriptor(
                kernel_id="custom/fused_moe/ep_probe",
                backend=Backend.CUSTOM,
                capability=CapabilitySpec(op="fused_moe.cutlass_fused_moe"),
                runner=lambda *args: calls.append(args) or args[6],
            )
        )
        monkeypatch.setattr(_MOE_MOD, "current_hardware", lambda: NPU_HW)
        x = _FakeTensor("bfloat16", (8, 32))
        out = _FakeTensor("bfloat16", (8, 32))
        ctx = _FakeSymmBuffer(2, 1)
        result = finpu.fused_moe(
            x, x, x, x, x, "bfloat16", [],
            output=out, ep_size=2, ep_rank=1, moe_context=ctx,
            kernel_id="custom/fused_moe/ep_probe",
        )
        assert result is out
        (_, _, _, _, _, _, out_, ep_size_, ep_rank_, moe_ctx_), = calls
        assert out_ is out and ep_size_ == 2 and ep_rank_ == 1 and moe_ctx_ is ctx

    def test_reference_ep_raises(self):
        torch = pytest.importorskip("torch")
        from flashinfer.runtime.reference_kernels import (
            reference_cutlass_fused_moe,
        )

        x = torch.randn(4, 8)
        with pytest.raises(NotImplementedError, match="单卡语义"):
            reference_cutlass_fused_moe(
                x, x, x, x, x, torch.bfloat16, torch.empty(4, 8), ep_size=2,
                moe_context=_FakeSymmBuffer(),
            )


class TestCannMegamoeBranch:
    """CANN runner EP 分支：fake cann_ops_transformer 验证 megaMoe 参数映射。"""

    def test_megamoe_arg_mapping(self, monkeypatch):
        torch = pytest.importorskip("torch")
        import sys
        import types

        calls = {}

        def fake_mega_moe(x, topk_ids, topk_weights, w1_list, w2_list, sym_buffer,
                          *, activation=None):
            calls.update(
                x=x, topk_ids=topk_ids, topk_weights=topk_weights,
                w1_list=w1_list, w2_list=w2_list, sym_buffer=sym_buffer,
                activation=activation,
            )
            return torch.ones(x.shape[0], 8, dtype=torch.float16), torch.zeros(4)

        pkg = types.ModuleType("cann_ops_transformer")
        ops_pkg = types.ModuleType("cann_ops_transformer.ops")
        mm_mod = types.ModuleType("cann_ops_transformer.ops.mega_moe")
        mm_mod.mega_moe = fake_mega_moe
        monkeypatch.setitem(sys.modules, "cann_ops_transformer", pkg)
        monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", ops_pkg)
        monkeypatch.setitem(
            sys.modules, "cann_ops_transformer.ops.mega_moe", mm_mod
        )

        from flashinfer.cann.moe import _cutlass_fused_moe_gmm

        x = torch.randn(4, 8, dtype=torch.bfloat16)
        idx = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int64)
        scales = torch.rand(4, 2)
        w1 = torch.randn(4, 16, 8, dtype=torch.bfloat16)  # (E_local, 2I, H)
        w2 = torch.randn(4, 8, 8, dtype=torch.bfloat16)  # (E_local, H, I)
        out = torch.empty(4, 8, dtype=torch.bfloat16)
        ctx = _FakeSymmBuffer(2, 0)
        result = _cutlass_fused_moe_gmm(
            x, idx, scales, w1, w2, torch.bfloat16, out,
            ep_size=2, ep_rank=0, moe_context=ctx,
        )
        assert result is out
        assert calls["sym_buffer"] is ctx
        assert calls["activation"] == "swiglu"
        assert calls["topk_ids"].dtype == torch.int32
        assert len(calls["w1_list"]) == 4 and len(calls["w2_list"]) == 4
        # megaMoe 要求 (H, 2I)/(I, H) 布局：适配层按专家转置
        assert torch.equal(calls["w1_list"][0], w1[0].t())
        assert torch.equal(calls["w2_list"][0], w2[0].t())
        assert out.dtype == torch.bfloat16
        assert torch.all(out == 1)


class TestTrtllmFacadeFlow:
    def test_per_tensor_facade_flow(self, monkeypatch, _clean_global_registry):
        calls = []
        finpu.register_custom_kernel(
            KernelDescriptor(
                kernel_id="custom/fused_moe/pt_probe",
                backend=Backend.CUSTOM,
                capability=CapabilitySpec(
                    op="fused_moe.trtllm_fp8_per_tensor_scale_moe"
                ),
                runner=lambda *args: calls.append(args),
            )
        )
        monkeypatch.setattr(
            _MOE_MOD, "current_hardware", lambda: NPU_HW
        )
        logits = _FakeTensor("float32", (8, 8))
        bias = _FakeTensor("float32", (8,))
        x8 = _FakeTensor("float8_e4m3fn", (8, 32))
        w8 = _FakeTensor("float8_e4m3fn", (8, 32, 32))
        w2_8 = _FakeTensor("float8_e4m3fn", (8, 32, 16))
        s = _FakeTensor("float32", (8,))
        out = _FakeTensor("bfloat16", (8, 32))
        result = finpu.fp8_per_tensor_scale_moe(
            logits, bias, x8, w8, s, s, w2_8, s, out,
            num_experts=8, top_k=2, n_group=2, topk_group=1,
            intermediate_size=16, local_expert_offset=0, local_num_experts=8,
            routed_scaling_factor=1.0, use_routing_scales_on_input=False,
            routing_method_type=int(finpu.RoutingMethodType.Default),
            kernel_id="custom/fused_moe/pt_probe",
        )
        assert result is None
        args, = calls
        assert args[0] is logits and args[8] is out
        assert args[10] == 2 and args[19] == 0  # top_k / routing_method_type

    def test_block_scale_facade_flow(self, monkeypatch, _clean_global_registry):
        calls = []
        finpu.register_custom_kernel(
            KernelDescriptor(
                kernel_id="custom/fused_moe/bs_probe",
                backend=Backend.CUSTOM,
                capability=CapabilitySpec(
                    op="fused_moe.trtllm_fp8_block_scale_moe"
                ),
                runner=lambda *args: calls.append(args),
            )
        )
        monkeypatch.setattr(
            _MOE_MOD, "current_hardware", lambda: NPU_HW
        )
        logits = _FakeTensor("float32", (8, 8))
        bias = _FakeTensor("float32", (8,))
        x8 = _FakeTensor("float8_e4m3fn", (8, 32))
        xs = _FakeTensor("float32", (1, 8))
        w8 = _FakeTensor("float8_e4m3fn", (8, 32, 32))
        ws_ = _FakeTensor("float32", (8, 1, 1))
        out = _FakeTensor("bfloat16", (8, 32))
        result = finpu.fp8_block_scale_moe(
            logits, bias, x8, xs, w8, ws_, w8, ws_, out,
            num_experts=8, top_k=2, n_groups=2, top_k_groups=1,
            intermediate_size=16, local_expert_offset=0, local_num_experts=8,
            routed_scaling=1.0,
            routing_method_type=int(finpu.RoutingMethodType.DeepSeekV3),
            kernel_id="custom/fused_moe/bs_probe",
        )
        assert result is None
        args, = calls
        assert args[0] is logits and args[8] is out
        assert args[10] == 2 and args[18] == 2  # top_k / routing_method_type




class TestCannFp8Cascade:
    """fp8 CANN 纯 fp8 级联 runner（950）：CPU 上将 _fp8_gmm 替换为 fp32
    朴素分段 matmul（模拟 fp8 MAC 语义），与 reference 对拍。差异仅来自
    中间激活重量化回 fp8（e4m3 相对步长 ~6%），取 fp8 级容差。"""

    def setup_method(self):
        self.torch = pytest.importorskip("torch")
        self.cann_moe = importlib.import_module("flashinfer.cann.moe")
        self.ref = importlib.import_module(
            "flashinfer.runtime.reference_kernels"
        )

    def _fp8(self, x):
        torch = self.torch
        finfo = torch.finfo(torch.float8_e4m3fn)
        amax = x.abs().max().clamp(min=1e-12)
        scale = finfo.max / amax
        return (x * scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)

    def _naive_fp8_gmm(self, x8, weight_list_t, group_list):
        """fp32 朴素分段 matmul，替换 _fp8_gmm（不依赖 NPU）。"""
        torch = self.torch
        outs, start = [], 0
        for e, end in enumerate(group_list):
            if end > start:
                outs.append(x8[start:end].float() @ weight_list_t[e].float())
            start = end
        if not outs:
            return torch.zeros(0, weight_list_t[0].shape[1])
        return torch.cat(outs, dim=0)

    def test_per_tensor_matches_reference(self, monkeypatch):
        torch = self.torch
        torch.manual_seed(0)
        seq, hidden, inter, num_e, top_k = 8, 64, 32, 4, 2
        x8 = self._fp8(torch.randn(seq, hidden))
        w1_8 = self._fp8(torch.randn(num_e, 2 * inter, hidden) * 0.05)
        w2_8 = self._fp8(torch.randn(num_e, hidden, inter) * 0.05)
        logits = torch.randn(seq, num_e)
        bias = torch.zeros(num_e)
        s1 = torch.rand(num_e) * 0.02 + 0.01
        s1g = torch.rand(num_e) * 0.5 + 0.75
        s2 = torch.rand(num_e) * 0.02 + 0.01
        monkeypatch.setitem(self.cann_moe._fp8_per_tensor_scale_moe_cascade.__globals__, '_fp8_gmm', self._naive_fp8_gmm)
        args = (logits, bias, x8, w1_8, s1, s1g, w2_8, s2)
        kw = (num_e, top_k, 1, 1, inter, 0, num_e, 1.0, False, 8, 0, False)
        out_cann = torch.empty(seq, hidden, dtype=torch.bfloat16)
        self.cann_moe._fp8_per_tensor_scale_moe_cascade(*args, out_cann, *kw)
        out_ref = torch.empty(seq, hidden, dtype=torch.bfloat16)
        self.ref.reference_trtllm_fp8_per_tensor_scale_moe(*args, out_ref, *kw)
        # 中间激活再次量化到 FP8；按整体范数约束量化误差，避免接近零的
        # 元素因抵消而产生任意大的逐元素相对误差。
        assert torch.isfinite(out_cann.float()).all()
        rel = (out_cann.float() - out_ref.float()).norm() / out_ref.float().norm()
        assert rel < 0.05

    def test_block_scale_matches_reference(self, monkeypatch):
        torch = self.torch
        torch.manual_seed(0)
        seq, hidden, inter, num_e, top_k = 8, 256, 256, 8, 2
        n_group, topk_group = 2, 1
        x8 = self._fp8(torch.randn(seq, hidden))
        w1_8 = self._fp8(torch.randn(num_e, 2 * inter, hidden) * 0.05)
        w2_8 = self._fp8(torch.randn(num_e, hidden, inter) * 0.05)
        logits = torch.randn(seq, num_e)
        bias = torch.randn(num_e) * 0.1
        xs = torch.rand(hidden // 128, seq) * 0.02 + 0.01  # (H//128, S)
        g1s = torch.rand(num_e, 2 * inter // 128, hidden // 128) * 0.02 + 0.01
        g2s = torch.rand(num_e, hidden // 128, inter // 128) * 0.02 + 0.01
        monkeypatch.setitem(self.cann_moe._fp8_block_gmm.__globals__, '_fp8_gmm', self._naive_fp8_gmm)
        args = (logits, bias, x8, xs, w1_8, g1s, w2_8, g2s)
        kw = (num_e, top_k, n_group, topk_group, inter, 0, num_e, 2.5, 8,
              int(finpu.RoutingMethodType.DeepSeekV3))
        out_cann = torch.empty(seq, hidden, dtype=torch.bfloat16)
        self.cann_moe._fp8_block_scale_moe_cascade(*args, out_cann, *kw)
        out_ref = torch.empty(seq, hidden, dtype=torch.bfloat16)
        self.ref.reference_trtllm_fp8_block_scale_moe(*args, out_ref, *kw)
        assert torch.isfinite(out_cann.float()).all()
        rel = (out_cann.float() - out_ref.float()).norm() / out_ref.float().norm()
        assert rel < 0.05

    def test_facade_cann_backend_flow(self, monkeypatch):
        """门面 backend='cann' 端到端接线：950 上调度命中级联 kernel。"""
        torch = self.torch
        torch.manual_seed(0)
        seq, hidden, inter, num_e, top_k = 8, 64, 32, 4, 2
        x8 = self._fp8(torch.randn(seq, hidden))
        w1_8 = self._fp8(torch.randn(num_e, 2 * inter, hidden) * 0.05)
        w2_8 = self._fp8(torch.randn(num_e, hidden, inter) * 0.05)
        monkeypatch.setattr(_MOE_MOD, "current_hardware", lambda: NPU_HW_950)
        monkeypatch.setitem(self.cann_moe._fp8_per_tensor_scale_moe_cascade.__globals__, '_fp8_gmm', self._naive_fp8_gmm)
        out = torch.empty(seq, hidden, dtype=torch.bfloat16)
        ret = finpu.fp8_per_tensor_scale_moe(
            torch.randn(seq, num_e), torch.zeros(num_e), x8, w1_8,
            torch.ones(num_e), torch.ones(num_e), w2_8, torch.ones(num_e),
            out,
            num_experts=num_e, top_k=top_k, n_group=1, topk_group=1,
            intermediate_size=inter, local_expert_offset=0,
            local_num_experts=num_e, routed_scaling_factor=1.0,
            use_routing_scales_on_input=False,
            routing_method_type=0, backend="cann",
        )
        assert ret is None
        assert torch.isfinite(out.float()).all()

    def test_mask_nonlocal_experts(self):
        torch = self.torch
        idx = torch.tensor([[0, 1], [2, 3]])
        w = torch.ones(2, 2)
        masked = self.cann_moe._mask_nonlocal_experts(idx, w, 2, 2)
        assert masked.tolist() == [[0.0, 0.0], [1.0, 1.0]]
        # 全覆盖时原样返回（无拷贝）
        assert self.cann_moe._mask_nonlocal_experts(idx, w, 0, 4) is w
