"""BF16 ND Matmul; imported only inside the explicit Triton JIT worker."""

import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(a, b, out, M, N, K,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(0, K, BLOCK_K):
        indices = start + reduction
        left = tl.load(a + rows[:, None] * K + indices[None, :],
                       mask=(rows[:, None] < M) & (indices[None, :] < K), other=0)
        right = tl.load(b + indices[:, None] * N + columns[None, :],
                        mask=(indices[:, None] < K) & (columns[None, :] < N), other=0)
        accumulator = tl.dot(left, right, accumulator)
    tl.store(out + rows[:, None] * N + columns[None, :], accumulator,
             mask=(rows[:, None] < M) & (columns[None, :] < N))
