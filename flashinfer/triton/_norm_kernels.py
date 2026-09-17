"""Triton norm 自定义 kernel（仅运行期导入：需要 triton-ascend 与 NPU）。

实现 CANN 未覆盖的 norm 算子。首批：gemma_fused_add_rmsnorm——
CANN 的 ``npu_add_rms_norm`` 缩放因子为 ``gamma``，与 Gemma 约定的
``gamma + 1`` 语义不符，且无 gemma 变体，故用 Triton 自定义实现。
"""

import triton
import triton.language as tl


@triton.jit
def _gemma_fused_add_rmsnorm_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    hidden,
    eps,
    BLOCK: tl.constexpr,
):
    """每 program 处理一行：r += x；x = (r / RMS(r)) * (w + 1)。

    fp32 累计；BLOCK 为 hidden 向上取整的 2 次幂，单 tile 覆盖整行。
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < hidden
    x = tl.load(x_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    s = x + r
    out_ty = x_ptr.dtype.element_ty
    # Step 1：写回新残差（按输入 dtype 舍入）
    tl.store(r_ptr + row * hidden + cols, s.to(out_ty), mask=mask)
    # Step 2：以未舍入的 fp32 残差计算归一化输出
    rstd = 1.0 / tl.sqrt(tl.sum(s * s, axis=0) / hidden + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = s * rstd * (1.0 + w)
    tl.store(x_ptr + row * hidden + cols, y.to(out_ty), mask=mask)


def gemma_fused_add_rmsnorm(input, residual, weight, eps: float) -> None:
    """原地执行 gemma fused add rmsnorm（2D 语义，末维归一化）。"""
    assert input.shape == residual.shape, "input 与 residual 形状须一致"
    assert weight.shape == input.shape[-1:], "weight 形状须为 (hidden_size,)"
    assert input.is_contiguous() and residual.is_contiguous(), "输入须连续"
    assert weight.is_contiguous(), "weight 须连续"
    hidden = input.shape[-1]
    x2d = input.view(-1, hidden)
    r2d = residual.view(-1, hidden)
    BLOCK = triton.next_power_of_2(hidden)
    _gemma_fused_add_rmsnorm_kernel[(x2d.shape[0],)](
        x2d, r2d, weight, hidden, float(eps), BLOCK=BLOCK
    )
