// Native BF16 MMAD with FP32 accumulation and a direct BF16 output store.
#include "kernel_operator.h"
#include "flashinfer_ascend/matmul_tiling.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/matmul_kernel.h"
#else
#error "matmul currently requires arch35 (__CCE_AICORE__ == 310)"
#endif

extern "C" __global__ __aicore__ void FI_DEVICE_ENTRY(
    GM_ADDR a, GM_ADDR b, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    REGISTER_TILING_DEFAULT(FiMatmulTilingData);
    GET_TILING_DATA(t, tiling);
    if (t.version != 2) return;
    Matmul(a, b, out, t);
}
