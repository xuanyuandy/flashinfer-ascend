// Correctness probe for the build/launch contract, not a production GEMM.
#include "kernel_operator.h"
#include "flashinfer_ascend/probe_tiling.h"
#include "common/native.h"

extern "C" __global__ __aicore__ void fi_matmul_probe_native(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    FI_PROBE_CHECK_TILING();
    ProbeMatmul(a, b, c);
}
