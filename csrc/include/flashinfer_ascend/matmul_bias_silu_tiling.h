#pragma once
#include <cstdint>

// Versioned, POD device-buffer payload; host and device share this exact layout.
struct FiMatmulBiasSiluTilingData {
    uint32_t version;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t tile_m;
    uint32_t tile_n;
    uint32_t tile_k;
    uint32_t workspace_bytes;
};
