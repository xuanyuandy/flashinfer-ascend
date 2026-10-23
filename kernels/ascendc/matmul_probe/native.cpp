// Correctness probe for the build/launch contract, not a production GEMM.
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void FI_DEVICE_ENTRY(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    auto pa = reinterpret_cast<__gm__ float *>(a);
    auto pb = reinterpret_cast<__gm__ float *>(b);
    auto pc = reinterpret_cast<__gm__ float *>(c);
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 16; ++n) {
            float value = 0;
            for (int k = 0; k < 16; ++k) value += pa[m * 16 + k] * pb[k * 16 + n];
            pc[m * 16 + n] = value;
        }
    }
}

#include "matmul_probe.inc"
