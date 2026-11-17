"""Public numerical results against independent reference calculations."""

import pytest

import flashinfer as finpu


class TestGemmFp8Numerics:
    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")

    def _to_float8(self, x, dtype=None):
        """与上游示例一致的 per-tensor 量化（返回 fp8 张量与 scale）。"""
        torch = self.torch
        dt = dtype or torch.float8_e4m3fn
        finfo = torch.finfo(dt)
        min_val, max_val = x.aminmax()
        amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
        scale = finfo.max / amax
        x_q = (x * scale).clamp(min=finfo.min, max=finfo.max)
        return x_q.to(dt), scale.float().reciprocal()

    def test_bmm_fp8_reference(self):
        torch = self.torch
        torch.manual_seed(0)
        x = torch.randn(4, 16, 32)
        w = torch.randn(4, 24, 32).transpose(-2, -1)  # (4, 32, 24) column major
        x8, xs = self._to_float8(x)
        w8, ws = self._to_float8(w)
        out = finpu.bmm_fp8(x8, w8, xs, ws, torch.bfloat16,
                            backend="reference")
        # 与 fp32 反量化计算一致（同一段数学，bf16 舍入后收紧比较）
        expected = (torch.matmul(x8.float(), w8.float()) * xs * ws).to(torch.bfloat16)
        assert out.shape == (4, 16, 24) and out.dtype == torch.bfloat16
        assert torch.allclose(out.float(), expected.float(), atol=1e-2)
        # 与原始未量化输入的整体相对误差在 fp8 量化噪声量级内（Frobenius 范数）
        ref = torch.matmul(x, w)
        rel = (out.float() - ref).norm() / ref.norm()
        assert rel < 0.05

    def test_bmm_fp8_out_reuse(self):
        torch = self.torch
        x8, xs = self._to_float8(torch.randn(2, 8, 16))
        w8, ws = self._to_float8(torch.randn(2, 16, 8))
        buf = torch.empty(2, 8, 8, dtype=torch.float16)
        result = finpu.bmm_fp8(x8, w8, xs, ws, torch.float16, buf,
                               backend="reference")
        assert result is buf

    @pytest.mark.parametrize("mode", ["K", "MN"])
    def test_gemm_fp8_nt_groupwise_reference(self, mode):
        torch = self.torch
        torch.manual_seed(0)
        m, n, k = 32, 128, 256
        gm, gn, gk = 1, 128, 128
        x = torch.randn(m, k)
        w = torch.randn(n, k)
        x8, _ = self._to_float8(x)
        w8, _ = self._to_float8(w)
        # 分组 scale（随机构造，模拟 DeepSeek 风格 blockwise 量化）
        sa = torch.rand(m // gm, k // gk) * 0.01 + 0.005
        sb = torch.rand(n // gn, k // gk) * 0.01 + 0.005
        if mode == "MN":
            sa_in, sb_in = sa.t().contiguous(), sb.t().contiguous()
        else:
            sa_in, sb_in = sa, sb
        out = finpu.gemm_fp8_nt_groupwise(
            x8, w8, sa_in, sb_in,
            scale_major_mode=mode,
            scale_granularity_mnk=(gm, gn, gk),
            backend="reference",
        )
        # 与 fp32 反量化计算一致（bf16 舍入后收紧比较）
        a_f = x8.float() * sa.repeat_interleave(gm, 0).repeat_interleave(gk, 1)
        b_f = w8.float() * sb.repeat_interleave(gn, 0).repeat_interleave(gk, 1)
        expected = (a_f @ b_f.t()).to(torch.bfloat16)
        assert out.shape == (m, n) and out.dtype == torch.bfloat16
        assert torch.allclose(out.float(), expected.float(), atol=1e-2)
