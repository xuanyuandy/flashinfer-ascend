/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDEXER_PROLOGUE_QW_FULL_FUSION_TILING_DEVICE_H_
#define INDEXER_PROLOGUE_QW_FULL_FUSION_TILING_DEVICE_H_

#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
namespace tiling {

// Device mirror of the host TILING_DATA_FIELD_DEF layout. Scalar fields have
// the same trailing-underscore names as the host macro, while the two cube
// structs retain their host field names and binary layout.
struct IndexerPrologueQWFullFusionTilingData {
    uint32_t coreNum_;
    uint32_t groupNum_;
    uint32_t totalInGroup_;
    uint32_t k_;
    uint32_t n_;
    uint32_t ubCalSize_;
    uint32_t ubRestBytes_;
    uint32_t parallNum_;
    uint32_t dqReduceTmpBytes_;
    uint32_t dqBroadcastTmpBytes_;
    uint32_t denseLayout_;
    float wScale_;
    TCubeTiling mmTilingData;
    TCubeTiling wTilingData;

    __aicore__ inline uint32_t get_coreNum() const
    {
        return coreNum_;
    }
    __aicore__ inline uint32_t get_groupNum() const
    {
        return groupNum_;
    }
    __aicore__ inline uint32_t get_totalInGroup() const
    {
        return totalInGroup_;
    }
    __aicore__ inline uint32_t get_k() const
    {
        return k_;
    }
    __aicore__ inline uint32_t get_n() const
    {
        return n_;
    }
    __aicore__ inline uint32_t get_ubCalSize() const
    {
        return ubCalSize_;
    }
    __aicore__ inline uint32_t get_ubRestBytes() const
    {
        return ubRestBytes_;
    }
    __aicore__ inline uint32_t get_parallNum() const
    {
        return parallNum_;
    }
    __aicore__ inline uint32_t get_dqReduceTmpBytes() const
    {
        return dqReduceTmpBytes_;
    }
    __aicore__ inline uint32_t get_dqBroadcastTmpBytes() const
    {
        return dqBroadcastTmpBytes_;
    }
    __aicore__ inline uint32_t get_denseLayout() const
    {
        return denseLayout_;
    }
    __aicore__ inline float get_wScale() const
    {
        return wScale_;
    }
};

} // namespace tiling
} // namespace AscendC
#endif // INDEXER_PROLOGUE_QW_FULL_FUSION_TILING_DEVICE_H_
