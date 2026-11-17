"""Public numerical results against independent reference calculations."""

import pytest

import flashinfer as finpu


class TestFusedMoeNumerics:
    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")

    def _moe_naive(self, x, idx, scales, w1, w2):
        """fp64 朴素专家循环（校验基线）。"""
        torch = self.torch
        seq, hidden = x.shape
        inter = w1.shape[1] // 2
        out = torch.zeros(seq, hidden, dtype=torch.float64)
        for e in range(w1.shape[0]):
            rows, ks = (idx == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            h = x[rows].double() @ w1[e].double().t()
            h = torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]
            y = h @ w2[e].double().t()
            out.index_add_(0, rows, y * scales[rows, ks].double().unsqueeze(-1))
        return out

    def test_reorder_rows(self):
        torch = self.torch
        x = torch.arange(24).reshape(6, 4)
        y = finpu.reorder_rows_for_gated_act_gemm(x)
        # [r0,r1,r2,r3,r4,r5] → [r0,r3,r1,r4,r2,r5]
        expected = x[[0, 3, 1, 4, 2, 5]]
        assert torch.equal(y, expected)
        with pytest.raises(AssertionError, match="even"):
            finpu.reorder_rows_for_gated_act_gemm(torch.randn(5, 4))
        with pytest.raises(AssertionError, match="2D"):
            finpu.reorder_rows_for_gated_act_gemm(torch.randn(2, 4, 4))

    def test_cutlass_reference_vs_naive(self):
        torch = self.torch
        torch.manual_seed(0)
        seq, hidden, inter, num_e, top_k = 8, 32, 16, 4, 2
        x = torch.randn(seq, hidden)
        logits = torch.randn(seq, num_e)
        probs = logits.softmax(-1)
        scales, idx = probs.topk(top_k, dim=-1)
        w1 = torch.randn(num_e, 2 * inter, hidden) * 0.05
        w2 = torch.randn(num_e, hidden, inter) * 0.05
        out = finpu.fused_moe(
            x, idx, scales, w1, w2, torch.float32, [], backend="reference"
        )
        expected = self._moe_naive(x, idx, scales, w1, w2)
        assert out.shape == (seq, hidden)
        assert torch.allclose(out.double(), expected, atol=1e-5)

    def test_cutlass_out_reuse_and_zero_scale(self):
        torch = self.torch
        seq, hidden, inter, num_e = 4, 16, 8, 2
        x = torch.randn(seq, hidden)
        idx = torch.zeros(seq, 1, dtype=torch.long)
        scales = torch.zeros(seq, 1)  # 零权重 → 输出为零
        w1 = torch.randn(num_e, 2 * inter, hidden)
        w2 = torch.randn(num_e, hidden, inter)
        buf = torch.full((seq, hidden), 7.0, dtype=torch.float16)
        result = finpu.fused_moe(
            x, idx, scales, w1, w2, torch.float16, [], output=buf,
            backend="reference",
        )
        assert result is buf
        assert torch.all(buf == 0)

    def _fp8(self, x):
        torch = self.torch
        finfo = torch.finfo(torch.float8_e4m3fn)
        amax = x.abs().max().clamp(min=1e-12)
        scale = finfo.max / amax
        return (x * scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)

    def test_trtllm_per_tensor_default_routing(self):
        torch = self.torch
        torch.manual_seed(0)
        seq, hidden, inter, num_e, top_k = 8, 32, 16, 4, 2
        x = torch.randn(seq, hidden)
        w1 = torch.randn(num_e, 2 * inter, hidden) * 0.05
        w2 = torch.randn(num_e, hidden, inter) * 0.05
        logits = torch.randn(seq, num_e)
        bias = torch.zeros(num_e)
        s1 = torch.rand(num_e) * 0.02 + 0.01
        s1g = torch.rand(num_e) * 0.5 + 0.75
        s2 = torch.rand(num_e) * 0.02 + 0.01
        out = torch.empty(seq, hidden, dtype=torch.bfloat16)
        ret = finpu.fp8_per_tensor_scale_moe(
            logits, bias, self._fp8(x), self._fp8(w1), s1, s1g,
            self._fp8(w2), s2, out,
            num_experts=num_e, top_k=top_k, n_group=1, topk_group=1,
            intermediate_size=inter, local_expert_offset=0,
            local_num_experts=num_e, routed_scaling_factor=1.0,
            use_routing_scales_on_input=False,
            routing_method_type=int(finpu.RoutingMethodType.Default),
            backend="reference",
        )
        assert ret is None
        # Default 路由：softmax → topk；per-tensor scale 折算进权重
        probs = logits.softmax(-1)
        scales, idx = probs.topk(top_k, dim=-1)
        w1_eff = self._fp8(w1).float() * (s1 * s1g).reshape(-1, 1, 1)
        w2_eff = self._fp8(w2).float() * s2.reshape(-1, 1, 1)
        expected = self._moe_naive(
            self._fp8(x).float(), idx, scales, w1_eff, w2_eff
        )
        assert torch.allclose(out.float(), expected.float(), atol=1e-2, rtol=1e-2)

    def test_trtllm_block_scale_deepseek_routing(self):
        torch = self.torch
        torch.manual_seed(0)
        seq, hidden, inter, num_e, top_k = 8, 128, 128, 8, 2
        n_group, topk_group = 2, 1
        x = torch.randn(seq, hidden)
        w1 = torch.randn(num_e, 2 * inter, hidden) * 0.05
        w2 = torch.randn(num_e, hidden, inter) * 0.05
        logits = torch.randn(seq, num_e)
        bias = torch.randn(num_e) * 0.1
        x8, w1_8, w2_8 = self._fp8(x), self._fp8(w1), self._fp8(w2)
        xs = torch.rand(hidden // 128, seq) * 0.02 + 0.01  # (hidden//128, seq)
        g1s = torch.rand(num_e, 2 * inter // 128, hidden // 128) * 0.02 + 0.01
        g2s = torch.rand(num_e, hidden // 128, inter // 128) * 0.02 + 0.01
        out = torch.empty(seq, hidden, dtype=torch.bfloat16)
        ret = finpu.fp8_block_scale_moe(
            logits, bias, x8, xs, w1_8, g1s, w2_8, g2s, out,
            num_experts=num_e, top_k=top_k, n_groups=n_group,
            top_k_groups=topk_group, intermediate_size=inter,
            local_expert_offset=0, local_num_experts=num_e,
            routed_scaling=2.5,
            routing_method_type=int(finpu.RoutingMethodType.DeepSeekV3),
            backend="reference",
        )
        assert ret is None
        # DeepSeekV3 路由（与 reference 同一段数学，独立复算）
        s = torch.sigmoid(logits)
        sb = s + bias
        sb_g = sb.reshape(seq, n_group, num_e // n_group)
        gscore = sb_g.topk(2, dim=-1).values.sum(-1)
        topg = gscore.topk(topk_group, dim=-1).indices
        mask = torch.zeros_like(sb_g, dtype=torch.bool)
        mask.scatter_(1, topg.unsqueeze(-1).expand(-1, -1, num_e // n_group), True)
        sb = sb_g.masked_fill(~mask, float("-inf")).reshape(seq, num_e)
        _, idx = sb.topk(top_k, dim=-1)
        scales = s.gather(-1, idx) * 2.5
        x_eff = x8.float() * xs.float().t().repeat_interleave(128, dim=-1)
        w1_eff = w1_8.float() * g1s.repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)
        w2_eff = w2_8.float() * g2s.repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)
        expected = self._moe_naive(x_eff, idx, scales, w1_eff, w2_eff)
        assert torch.allclose(out.float(), expected.float(), atol=1e-2, rtol=1e-2)

    def test_trtllm_ep_local_expert_masking(self):
        """local_expert_offset/local_num_experts：区间外专家贡献被跳过。"""
        torch = self.torch
        torch.manual_seed(0)
        seq, hidden, inter, num_e, top_k = 4, 16, 8, 4, 2
        x = torch.randn(seq, hidden)
        w1 = torch.randn(num_e, 2 * inter, hidden) * 0.05
        w2 = torch.randn(num_e, hidden, inter) * 0.05
        logits = torch.randn(seq, num_e)
        out = torch.empty(seq, hidden)
        finpu.fp8_per_tensor_scale_moe(
            logits, torch.zeros(num_e), self._fp8(x), self._fp8(w1),
            torch.ones(num_e), torch.ones(num_e), self._fp8(w2),
            torch.ones(num_e), out,
            num_experts=num_e, top_k=top_k, n_group=1, topk_group=1,
            intermediate_size=inter, local_expert_offset=2,
            local_num_experts=2, routed_scaling_factor=1.0,
            use_routing_scales_on_input=False,
            routing_method_type=0,
            backend="reference",
        )
        # 仅专家 2/3 生效
        probs = logits.softmax(-1)
        scales, idx = probs.topk(top_k, dim=-1)
        x8, w1_8, w2_8 = self._fp8(x), self._fp8(w1), self._fp8(w2)
        expected = torch.zeros(seq, hidden)
        for e in (2, 3):
            rows, ks = (idx == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            h = x8.float()[rows] @ w1_8[e].float().t()
            h = torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]
            y = h @ w2_8[e].float().t()
            expected.index_add_(0, rows, y * scales[rows, ks].unsqueeze(-1))
        assert torch.allclose(out, expected, atol=1e-4)



class TestCutlassMoeCannNpu:
    """CANN kernel（npu_grouped_matmul）NPU 数值验证：与 fp64 基线对拍。"""

    @pytest.fixture(autouse=True)
    def _npu(self):
        self.torch = pytest.importorskip("torch")
        pytest.importorskip("torch_npu")
        if not self.torch.npu.is_available():
            pytest.skip("无 NPU 环境")

    def test_cutlass_cann_vs_fp64(self):
        torch = self.torch
        torch.manual_seed(0)
        dev = torch.device("npu", torch.npu.current_device())
        seq, hidden, inter, num_e, top_k = 16, 64, 32, 4, 2
        x = torch.randn(seq, hidden, device=dev, dtype=torch.bfloat16)
        logits = torch.randn(seq, num_e)
        scales, idx = logits.softmax(-1).topk(top_k, dim=-1)
        idx = idx.to(dev)
        scales = scales.to(dev)
        w1 = (torch.randn(num_e, 2 * inter, hidden) * 0.05).to(dev, torch.bfloat16)
        w2 = (torch.randn(num_e, hidden, inter) * 0.05).to(dev, torch.bfloat16)
        out = finpu.fused_moe(
            x, idx, scales, w1, w2, torch.bfloat16, [], backend="cann"
        )
        # fp64 基线（CPU）
        xf = x.cpu().float()
        expected = torch.zeros(seq, hidden, dtype=torch.float64)
        for e in range(num_e):
            rows, ks = (idx.cpu() == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            h = xf[rows].double() @ w1[e].cpu().double().t()
            h = torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]
            y = h @ w2[e].cpu().double().t()
            expected.index_add_(0, rows, y * scales.cpu()[rows, ks].double().unsqueeze(-1))
        rel = (out.cpu().double() - expected).norm() / expected.norm()
        assert rel < 0.02

    def test_cutlass_cann_empty_expert_group(self):
        """存在零 token 专家（空组）时结果仍正确。"""
        torch = self.torch
        torch.manual_seed(1)
        dev = torch.device("npu", torch.npu.current_device())
        seq, hidden, inter, num_e = 8, 32, 16, 4
        x = torch.randn(seq, hidden, device=dev, dtype=torch.bfloat16)
        idx = torch.zeros(seq, 1, dtype=torch.long, device=dev)  # 全部路由到专家 0
        scales = torch.rand(seq, 1, device=dev)
        w1 = (torch.randn(num_e, 2 * inter, hidden) * 0.05).to(dev, torch.bfloat16)
        w2 = (torch.randn(num_e, hidden, inter) * 0.05).to(dev, torch.bfloat16)
        out = finpu.fused_moe(
            x, idx, scales, w1, w2, torch.bfloat16, [], backend="cann"
        )
        h = x.cpu().float() @ w1[0].cpu().float().t()
        h = torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]
        expected = (h @ w2[0].cpu().float().t()) * scales.cpu()
        rel = (out.cpu().float() - expected).norm() / expected.norm()
        assert rel < 0.02

    def test_cutlass_cann_vs_reference(self):
        """同一输入下 cann 与 reference 结果一致（bf16 容差）。"""
        torch = self.torch
        torch.manual_seed(2)
        dev = torch.device("npu", torch.npu.current_device())
        seq, hidden, inter, num_e, top_k = 8, 32, 16, 3, 2
        x = torch.randn(seq, hidden)
        logits = torch.randn(seq, num_e)
        scales, idx = logits.softmax(-1).topk(top_k, dim=-1)
        w1 = torch.randn(num_e, 2 * inter, hidden) * 0.05
        w2 = torch.randn(num_e, hidden, inter) * 0.05
        out_npu = finpu.fused_moe(
            x.to(dev, torch.bfloat16), idx.to(dev), scales.to(dev),
            w1.to(dev, torch.bfloat16), w2.to(dev, torch.bfloat16),
            torch.bfloat16, [], backend="cann",
        )
        out_ref = finpu.fused_moe(
            x, idx, scales, w1, w2, torch.float32, [], backend="reference"
        )
        rel = (out_npu.cpu().float() - out_ref).norm() / out_ref.norm()
        assert rel < 0.02
