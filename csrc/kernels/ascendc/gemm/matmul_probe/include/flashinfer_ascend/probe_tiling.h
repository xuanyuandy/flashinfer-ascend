#pragma once
#include <cstdint>

// Public v2 probe tiling. The legacy four-byte layout is a test fixture only.
struct FiMatmulProbeTilingData {
    uint32_t version;
    uint32_t size;
};

#ifdef FI_OP_PACKAGE
#define FI_PROBE_CHECK_TILING() \
    REGISTER_TILING_DEFAULT(FiMatmulProbeTilingData); \
    GET_TILING_DATA(fi_tiling, tiling); \
    if (fi_tiling.version != 2 || fi_tiling.size != 256) return
#else
#define FI_PROBE_CHECK_TILING()
#endif
