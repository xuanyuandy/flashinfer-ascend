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
 * \file block_indexer_rearrange_tiling.cpp
 * \brief Host tiling for the arch22 (910B/910C/950) block-index expansion.
 *
 * One AIV lane per query token; the flat token index is split across cores
 * exactly like the source ops-transformer tiler. The payload is the POD
 * struct shared with fi_block_indexer_rearrange.
 */

#include "register/op_impl_registry.h"
#include "fi_host_checks.h"
#include "tiling/platform/platform_ascendc.h"
#include "block_indexer_rearrange_tiling_data.h"

using namespace ge;

namespace optiling {
namespace {
constexpr uint32_t kVersion = 1;
// tensor inputs
constexpr size_t TOPK_INPUT_INDEX = 0;
constexpr size_t SEQ_LENQ_INDEX = 1;
constexpr size_t SEQ_LENKV_INDEX = 2;
// attrs inputs
constexpr size_t CAUSAL_ATTR_INDEX = 0;
constexpr size_t BLOCK_SIZE_ATTR_INDEX = 1;
constexpr int64_t MIN_BLOCK_SIZE = 1;
// The kernel broadcast materializes 8 token lanes per block index.
constexpr int64_t MAX_BLOCK_SIZE = 8;
// The kernel broadcast buffer is 128 KiB of INT32 lanes (blockCount * 8 * 4).
constexpr int64_t MAX_BLOCK_COUNT = 4096;
constexpr int64_t MIN_TOKENS = 1;
} // namespace

struct BlockIndexerRearrangeCompileInfo {
    uint32_t aivNum = 0;
};

ge::graphStatus BlockIndexerRearrangeTilingPrepare(gert::TilingParseContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockIndexerRearrangeTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const gert::StorageShape *topkShape = context->GetInputShape(TOPK_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkShape);
    const gert::StorageShape *seqlenQShape = context->GetInputShape(SEQ_LENQ_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, seqlenQShape);
    const gert::StorageShape *seqlenKvShape = context->GetInputShape(SEQ_LENKV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, seqlenKvShape);

    if (topkShape->GetStorageShape().GetDimNum() != 3) {
        OP_LOGE(context->GetNodeName(), "topk_indices must be rank-3 [T1, N2, blockCount]");
        return ge::GRAPH_FAILED;
    }
    if (seqlenQShape->GetStorageShape().GetDimNum() != 1 ||
        seqlenKvShape->GetStorageShape().GetDimNum() != 1 ||
        seqlenQShape->GetStorageShape().GetDim(0) < 2 ||
        seqlenKvShape->GetStorageShape().GetDim(0) < 2) {
        OP_LOGE(context->GetNodeName(), "acc_seq lengths must be rank-1 cumulative [B+1] with B >= 1");
        return ge::GRAPH_FAILED;
    }
    // Only the varlen (TND) layout is supported; dim0 is the flat token count.
    const int64_t totalSeqlen = topkShape->GetStorageShape().GetDim(0);
    const int64_t headNum = topkShape->GetStorageShape().GetDim(1);
    const int64_t blockCount = topkShape->GetStorageShape().GetDim(2);
    if (totalSeqlen < MIN_TOKENS) {
        OP_LOGE(context->GetNodeName(), "topk_indices requires at least one token");
        return ge::GRAPH_FAILED;
    }
    if (blockCount < 1 || blockCount > MAX_BLOCK_COUNT) {
        OP_LOGE(context->GetNodeName(), "blockCount must be in [1, 4096]");
        return ge::GRAPH_FAILED;
    }

    int64_t causal = 0;
    int64_t blockSize = 1;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool *causalPtr = attrs->GetAttrPointer<bool>(CAUSAL_ATTR_INDEX);
        const int64_t *blockSizePtr = attrs->GetAttrPointer<int64_t>(BLOCK_SIZE_ATTR_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, causalPtr);
        OP_CHECK_NULL_WITH_CONTEXT(context, blockSizePtr);
        causal = *causalPtr ? 1 : 0;
        blockSize = *blockSizePtr;
    }
    if (blockSize < MIN_BLOCK_SIZE || blockSize > MAX_BLOCK_SIZE) {
        OP_LOGE(context->GetNodeName(), "block_size must be in [1, 8]");
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivNum = platform.GetCoreNumAiv();
    const uint32_t aicNum = platform.GetCoreNumAic();
    if (aicNum == 0 || aivNum == 0) {
        OP_LOGE(context->GetNodeName(), "num of core obtained is 0.");
        return ge::GRAPH_FAILED;
    }

    // Flat token split: former cores take one extra token. The device entry
    // launches every AIV lane, lanes beyond usedCoreNum stay idle.
    const int64_t aivNumI64 = static_cast<int64_t>(aivNum);
    const int64_t usedCoreNum = totalSeqlen < aivNumI64 ? totalSeqlen : aivNumI64;
    const int64_t formerCoreProcessNums = (totalSeqlen + aivNumI64 - 1) / aivNumI64;
    const int64_t remainCoreNum = formerCoreProcessNums * aivNumI64 - totalSeqlen;
    const int64_t formerCoreNum = aivNumI64 - remainCoreNum;

    BlockIndexerRearrangeTilingData *data = context->GetTilingData<BlockIndexerRearrangeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, data);
    *data = {};
    data->version = kVersion;
    data->totalSeqlen = totalSeqlen;
    data->headNum = headNum;
    data->blockCount = blockCount;
    data->blockSize = blockSize;
    data->causal = causal;
    data->usedCoreNum = static_cast<uint32_t>(usedCoreNum);
    data->formerCoreNum = static_cast<uint32_t>(formerCoreNum);
    data->formerCoreProcessNums = static_cast<uint32_t>(formerCoreProcessNums);
    data->remainCoreProcessNums = static_cast<uint32_t>(formerCoreProcessNums - 1);

    // Single implementation path (INT32 indices only); entry 0 is the default
    // kernel entry emitted for the unbranched device source.
    if (context->SetBlockDim(aivNum) != ge::GRAPH_SUCCESS ||
        context->SetTilingKey(0) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FiBlockIndexerRearrange)
    .Tiling(BlockIndexerRearrangeTilingFunc)
    .TilingParse<BlockIndexerRearrangeCompileInfo>(BlockIndexerRearrangeTilingPrepare);
} // namespace optiling
