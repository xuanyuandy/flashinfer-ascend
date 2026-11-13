// Migrated from catlass example 49 at revision
// 41bf90da655bba3c66d0acd7e00abe33960ecfd6.
// See LICENSES/CANN-Open-Software-License-2.0.txt at the repository root.
// Only dense batch=1, equal heads, FP16/BF16 D=128, unmasked attention is exposed.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "kernel_operator.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/fai_kernel.h"
#else
#error "flash_attention currently requires arch35 (__CCE_AICORE__ == 310)"
#endif

extern "C" __global__ __aicore__ void FI_DEVICE_ENTRY(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(FATilingData);
    // CANN owns the system workspace. This Ascend950 kernel exchanges QK/PV
    // tiles through L1/UB; it does not require additional user GM scratch.
    SetSysWorkspace(workspace);
    // Mask and block table views are formed by upstream templates, but never
    // read when both compile-time flags are false. Use a valid device address.
    // CANN compiles a distinct binary for each registered ordered dtype tuple.
    FAInferTla<DTYPE_Q, false, false>(q, k, v, q, q, out, nullptr, nullptr, tiling);
}
