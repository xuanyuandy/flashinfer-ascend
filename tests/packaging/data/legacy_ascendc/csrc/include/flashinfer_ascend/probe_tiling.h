#pragma once
#include <cstdint>

// Consumed only by the op-package adapter; the direct launcher passes no tiling.
struct FiMatmulProbeTilingData {
    uint32_t size;
};

#ifdef FI_OP_PACKAGE
#define FI_PROBE_CHECK_TILING() \
    REGISTER_TILING_DEFAULT(FiMatmulProbeTilingData); \
    GET_TILING_DATA(fi_tiling, tiling); \
    if (fi_tiling.size != 256) return
#else
#define FI_PROBE_CHECK_TILING()
#endif
