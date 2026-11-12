"""Dense NHD prefill kernel; imported only by the explicit Triton compiler worker."""

import triton
import triton.language as tl


@triton.jit
def _dense_prefill_kernel(
    q, k, v, out, S, T, H,
    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    head = tl.program_id(1)
    dims = tl.arange(0, D)
    columns = tl.arange(0, BLOCK_N)
    query = tl.load(q + rows[:, None] * H * D + head * D + dims[None, :],
                    mask=rows[:, None] < S, other=0)
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, D), tl.float32)
    scale: tl.constexpr = 1.4426950408889634 / (D ** 0.5)
    for start in range(0, T, BLOCK_N):
        positions = start + columns
        key = tl.load(k + positions[None, :] * H * D + head * D + dims[:, None],
                      mask=positions[None, :] < T, other=0)
        scores = tl.dot(query, key) * scale
        scores = tl.where(positions[None, :] < T, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        correction = tl.exp2(maximum - next_maximum)
        probability = tl.exp2(scores - next_maximum[:, None])
        value = tl.load(v + positions[:, None] * H * D + head * D + dims[None, :],
                        mask=positions[:, None] < T, other=0)
        accumulator = accumulator * correction[:, None]
        accumulator = tl.dot(probability.to(q.dtype.element_ty), value, accumulator)
        denominator = denominator * correction + tl.sum(probability, axis=1)
        maximum = next_maximum
    result = accumulator / denominator[:, None]
    tl.store(out + rows[:, None] * H * D + head * D + dims[None, :],
             result, mask=rows[:, None] < S)
