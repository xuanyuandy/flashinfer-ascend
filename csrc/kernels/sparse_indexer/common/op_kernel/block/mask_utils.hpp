/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_MASK_UTILS_HPP
#define FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_MASK_UTILS_HPP

#include <cstdint>
#include "kernel_operator.h"

namespace SparseIndexer::Common::Block {

/// IEEE-754 float -inf bit pattern (shared by Reduce mask writes and TopK filters).
inline constexpr uint32_t kFloatNegInfBits = 0xFF800000U;

CATLASS_DEVICE bool IsNegInfFloat(float v)
{
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = v;
    return bits.u == kFloatNegInfBits;
}

/**
 * LI V2 create_mask visible column count for local Q row ``localQ``.
 * ``qAct`` / ``kAct`` are seqused lengths (or full request lengths when seqused absent).
 */
CATLASS_DEVICE uint32_t Mask3VisibleLen(
    uint32_t localQ, uint32_t qAct, uint32_t kAct, int32_t cmpRatio, int32_t residual)
{
    if (kAct == 0) {
        return 0;
    }
    const int32_t ratio = cmpRatio <= 0 ? 1 : cmpRatio;
    const int32_t tmp = static_cast<int32_t>(kAct) * ratio + residual - static_cast<int32_t>(qAct);
    const int32_t numer = tmp + static_cast<int32_t>(localQ) + 1;
    if (numer < 0) {
        return 0;
    }
    const uint32_t vis = static_cast<uint32_t>(numer / ratio);
    return vis < kAct ? vis : kAct;
}

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_MASK_UTILS_HPP
