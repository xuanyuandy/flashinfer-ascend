"""Public numerical results against independent reference calculations."""

import pytest

import flashinfer as finpu


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
    @pytest.mark.real_compile
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
    @pytest.mark.real_compile
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
