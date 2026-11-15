// Migrated from catlass example 49 at revision
// 6c0a3242b2effd861ac91aab1999f2226b643614.
// See LICENSES/CANN-Open-Software-License-2.0.txt at the repository root.
// Only dense batch=1, equal heads, FP16/BF16 D=128, unmasked attention is exposed.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/fai_kernel.h"
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "arch22/fai_kernel.h"
#else
#error "flash_attention requires arch22 or arch35"
#endif

extern "C" __global__ __aicore__ void fi_flash_attention(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(FATilingData);
    AscendC::SetSysWorkspace(workspace);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    auto *data = reinterpret_cast<__gm__ FATilingData*>(tiling);
    AscendC::SetSyncBaseAddr(data->hardwareSyncAddr);
    GM_ADDR scratch = AscendC::GetUserWorkspace(workspace);
    GM_ADDR s = scratch;
    GM_ADDR p = s + data->mm1OutSize;
    GM_ADDR temp = p + data->smOnlineOutSize;
    GM_ADDR update = temp + data->mm2OutSize;
    FAInferTla<DTYPE_Q>(q, k, v, q, q, out, nullptr, nullptr, s, p, temp, update, tiling);
#else
    // The Ascend950 implementation keeps algorithm scratch in L1/UB.
    FAInferTla<DTYPE_Q, false, false>(q, k, v, q, q, out, nullptr, nullptr, tiling);
#endif
}
