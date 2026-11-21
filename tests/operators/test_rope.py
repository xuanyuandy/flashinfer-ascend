"""Public numerical results against independent reference calculations."""

import pytest

import flashinfer as finpu


@pytest.mark.usefixtures("torch_for_class")
class TestRopeNumerics:
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


    @pytest.mark.usefixtures("npu_device")
    @pytest.mark.parametrize("is_neox", [True, False])
    @pytest.mark.parametrize(
        "dtype,atol",
        [("float16", 2e-2), ("bfloat16", 8e-2), ("float32", 1e-4)],
    )
    def test_rope_npu_vs_fp64(self, dtype, atol, is_neox):
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

    @pytest.mark.usefixtures("npu_device")
    @pytest.mark.parametrize("is_neox", [True, False])
    def test_rope_partial_rotary_npu(self, is_neox):
        """rotary_dim < head_size：仅旋转前 rd 维，其余维原样保留。"""
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

    @pytest.mark.usefixtures("npu_device")
    def test_cann_vs_reference_consistency(self):
        """M1 验收口径：同输入，CANN 与 reference 两 backend 数值一致。"""
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



import torch
import flashinfer
from flashinfer.runtime import current_hardware

NPU = current_hardware().is_npu_available


requires_npu = pytest.mark.skipif(not NPU, reason="需要 NPU")


DEV = "npu" if NPU else "cpu"


def _qk(nnz=7, heads=4, head_dim=64, dtype=torch.float16):
    g = torch.Generator().manual_seed(1234)
    q = torch.randn(nnz, heads, head_dim, generator=g, dtype=torch.float32)
    k = torch.randn(nnz, heads, head_dim, generator=g, dtype=torch.float32)
    return q.to(dtype).to(DEV), k.to(dtype).to(DEV)


def _pos(nnz=7):
    return torch.arange(nnz, dtype=torch.int32, device=DEV)


def _rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


@requires_npu
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pos_ids_cann_matches_reference(interleave, dtype):
    q, k = _qk(dtype=dtype)
    pos = _pos()
    qr = flashinfer.apply_rope_pos_ids(
        q, k, pos, interleave=interleave, backend="reference")
    qc = flashinfer.apply_rope_pos_ids(
        q, k, pos, interleave=interleave, backend="cann")
    assert _rel_err(qc[0], qr[0]) < 2e-2
    assert _rel_err(qc[1], qr[1]) < 2e-2


@requires_npu
@pytest.mark.parametrize("rotary_dim", [None, 32])
def test_partial_rotary_leaves_tail_untouched(rotary_dim):
    """只旋转前 rotary_dim 维，其余维必须逐位不变。"""
    q, k = _qk()
    pos = _pos()
    out_q, _ = flashinfer.apply_rope_pos_ids(
        q, k, pos, rotary_dim=rotary_dim, backend="cann")
    rd = rotary_dim or q.shape[-1]
    if rd < q.shape[-1]:
        assert torch.equal(out_q[..., rd:], q[..., rd:])


@requires_npu
def test_inplace_modifies_input_and_returns_none():
    q, k = _qk()
    pos = _pos()
    q_before = q.clone()
    ret = flashinfer.apply_rope_pos_ids_inplace(q, k, pos, backend="cann")
    assert ret is None
    assert not torch.equal(q, q_before), "原地接口未修改输入"


@requires_npu
def test_out_of_place_does_not_modify_input():
    q, k = _qk()
    pos = _pos()
    q_before, k_before = q.clone(), k.clone()
    flashinfer.apply_rope_pos_ids(q, k, pos, backend="cann")
    assert torch.equal(q, q_before)
    assert torch.equal(k, k_before)


@requires_npu
def test_inplace_and_out_of_place_agree():
    q, k = _qk()
    pos = _pos()
    out_q, out_k = flashinfer.apply_rope_pos_ids(q, k, pos, backend="cann")
    qi, ki = q.clone(), k.clone()
    flashinfer.apply_rope_pos_ids_inplace(qi, ki, pos, backend="cann")
    assert torch.equal(qi, out_q)
    assert torch.equal(ki, out_k)


@requires_npu
def test_indptr_variant_matches_pos_ids_variant():
    """apply_rope 与等价的 apply_rope_pos_ids 必须给出相同结果。"""
    q, k = _qk(nnz=9)
    indptr = torch.tensor([0, 3, 5, 9], dtype=torch.int32, device=DEV)
    offsets = torch.tensor([10, 100, 0], dtype=torch.int32, device=DEV)
    a_q, a_k = flashinfer.apply_rope(q, k, indptr, offsets, backend="cann")
    pos = torch.tensor([10, 11, 12, 100, 101, 0, 1, 2, 3],
                       dtype=torch.int32, device=DEV)
    b_q, b_k = flashinfer.apply_rope_pos_ids(q, k, pos, backend="cann")
    assert torch.equal(a_q, b_q)
    assert torch.equal(a_k, b_k)


@requires_npu
@pytest.mark.parametrize("rope_scale,rope_theta", [(1, 1e4), (4, 1e4), (1, 5e5)])
def test_scale_and_theta_reach_the_kernel(rope_scale, rope_theta):
    """改变 scale/theta 必须改变结果，且与参考实现一致。"""
    q, k = _qk()
    pos = _pos()
    base, _ = flashinfer.apply_rope_pos_ids(q, k, pos, backend="cann")
    got, _ = flashinfer.apply_rope_pos_ids(
        q, k, pos, rope_scale=rope_scale, rope_theta=rope_theta, backend="cann")
    ref, _ = flashinfer.apply_rope_pos_ids(
        q, k, pos, rope_scale=rope_scale, rope_theta=rope_theta,
        backend="reference")
    assert _rel_err(got, ref) < 2e-2
    if (rope_scale, rope_theta) != (1, 1e4):
        assert not torch.equal(got, base), "参数未传到 kernel"


@requires_npu
def test_cos_sin_cache_out_of_place_matches_inplace():
    nnz, heads, hs = 7, 4, 64
    q = torch.randn(nnz, heads * hs, dtype=torch.float16, device=DEV)
    k = torch.randn(nnz, heads * hs, dtype=torch.float16, device=DEV)
    pos = _pos(nnz).to(torch.int64)
    cache = torch.randn(128, hs, dtype=torch.float32, device=DEV)
    out_q, out_k = flashinfer.apply_rope_with_cos_sin_cache(
        pos, q, k, hs, cache, backend="cann")
    qi, ki = q.clone(), k.clone()
    flashinfer.apply_rope_with_cos_sin_cache_inplace(
        pos, qi, ki, hs, cache, backend="cann")
    assert torch.equal(qi, out_q)
    assert torch.equal(ki, out_k)
