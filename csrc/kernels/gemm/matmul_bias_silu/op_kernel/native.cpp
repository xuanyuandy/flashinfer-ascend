// Small correctness-first native Ascend C operator. Cube uses real BF16 MMAD;
// two vector subcores consume FP32 scratch after a cross-core FIX-pipe flag.
#include "kernel_operator.h"
#include "flashinfer_ascend/matmul_bias_silu_tiling.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/matmul_bias_silu_kernel.h"
#else
#error "matmul_bias_silu currently requires arch35 (__CCE_AICORE__ == 310)"
#endif

extern "C" __global__ __aicore__ void fi_matmul_bias_silu(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(FiMatmulBiasSiluTilingData);
    GET_TILING_DATA(t, tiling);
    if (t.version != 2 || t.tile_m != 16 || t.tile_n != 16 || t.tile_k != 16 ||
        t.workspace_bytes != 4 * t.m * t.n || workspace == nullptr) return;
    SetSysWorkspace(workspace);
    GM_ADDR scratch = GetUserWorkspace(workspace);
    if ASCEND_IS_AIC { Cube(a, b, scratch, t); }
    if ASCEND_IS_AIV { Vector(bias, out, scratch, t); }
}
