// Instantiate the public CATLASS BasicMatmulTla API with the same probe ABI.
// See CATLASS example 43_ascend950_basic_matmul for the template contract.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "kernel_operator.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/catlass.h"
#else
#error "CATLASS matmul_probe currently requires arch35 (__CCE_AICORE__ == 310)"
#endif

extern "C" __global__ __aicore__ void FI_DEVICE_ENTRY(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    FI_PROBE_CHECK_TILING();
    auto la = tla::MakeLayout<float, Tag>(16u, 16u);
    auto lb = tla::MakeLayout<float, Tag>(16u, 16u);
    auto lc = tla::MakeLayout<float, Tag>(16u, 16u);
    Kernel::Params params{GemmCoord{16, 16, 16}, a, la, b, lb, c, lc};
    Kernel kernel;
    kernel(params);
}
