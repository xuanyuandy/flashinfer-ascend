/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <cstdint>
struct FATilingData {
    uint32_t numHeads = 0;
    uint32_t embeddingSize = 0;
    uint32_t numBlocks = 0;
    uint32_t blockSize = 0;
    uint32_t maxKvSeqlen = 0;
    uint32_t kvHeads = 0;
    uint32_t batch = 0;
    uint32_t maxNumBlocksPerBatch = 0;
    uint32_t firstBatchTaskNum = 0;
    uint32_t totalTaskNum = 0;
    uint32_t maskType = 0;
    uint64_t mm1OutSize = 0;
    uint64_t smOnlineOutSize = 0;
    uint64_t mm2OutSize = 0;
    uint64_t UpdateSize = 0;
    uint64_t workSpaceSize = 0;
    float scaleValue = 0.0;
    uint32_t qSeqlen = 0; // fixed dense query length from host tiling
    uint64_t hardwareSyncAddr = 0;
};
