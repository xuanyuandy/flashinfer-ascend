#pragma once
#include <cstdint>

struct FiMatmulTilingData {
    uint32_t version;
    uint32_t m;
    uint32_t n;
    uint32_t k;
};
