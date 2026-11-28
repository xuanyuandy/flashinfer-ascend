/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_indexer_rearrange_tiling_data.h
 * \brief POD tiling payload shared by host tiling and the device entry.
 *
 * Migrated from gitcode ops-transformer experimental/attention/
 * block_indexer_rearrange (REGISTER_TILING_DATA_CLASS payload); the field
 * order mirrors the source TilingData so the kernel math stays identical.
 */

#ifndef BLOCK_INDEXER_REARRANGE_TILING_DATA_H
#define BLOCK_INDEXER_REARRANGE_TILING_DATA_H

#include <cstdint>

struct BlockIndexerRearrangeTilingData {
    // Layout version 1: the migrated payload. The device entry rejects any
    // other version so a stale host/device pair cannot misinterpret it.
    uint32_t version;
    // topk_indices is [T1, N2, blockCount]; one AIV lane per query token.
    int64_t totalSeqlen;
    int64_t headNum;
    int64_t blockCount;
    // Tokens per block. The broadcast buffer materializes 8 token lanes per
    // block index, so only 1..8 is representable.
    int64_t blockSize;
    // Accepted for interface parity with the source operator; the TND token
    // expansion of the kernel already encodes the causal lower-triangle rule.
    int64_t causal;
    // Core split over the flat token index.
    uint32_t usedCoreNum;
    uint32_t formerCoreNum;
    uint32_t formerCoreProcessNums;
    uint32_t remainCoreProcessNums;
};

#endif // BLOCK_INDEXER_REARRANGE_TILING_DATA_H
