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
 * \file block_indexer_rearrange_infershape.cpp
 * \brief Shape/type inference for the TND block-index expansion.
 *
 * topk_output[T1, N2, blockCount * blockSize]; INT32 in, INT32 out.
 */

#include <register/op_impl_registry.h>
#include "fi_host_checks.h"

using namespace ge;

namespace ops {
constexpr uint32_t TOPK_INPUT_INDEX = 0;
constexpr uint32_t TOPK_OUTPUT_INDEX = 0;
constexpr size_t BLOCK_SIZE_ATTR_INDEX = 1;
constexpr int64_t MIN_BLOCK_SIZE = 1;
// The kernel broadcast materializes 8 token lanes per block index.
constexpr int64_t MAX_BLOCK_SIZE = 8;

ge::graphStatus InferShapeBlockIndexerRearrange(gert::InferShapeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const gert::Shape *topkShape = context->GetInputShape(TOPK_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkShape);
    if (topkShape->GetDimNum() != 3) {
        OP_LOGE(context->GetNodeName(), "topk_indices must be rank-3 [T1, N2, blockCount]");
        return ge::GRAPH_FAILED;
    }
    int64_t blockSize = 1;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *blockSizePtr = attrs->GetAttrPointer<int64_t>(BLOCK_SIZE_ATTR_INDEX);
        if (blockSizePtr != nullptr) {
            blockSize = *blockSizePtr;
        }
    }
    if (blockSize < MIN_BLOCK_SIZE || blockSize > MAX_BLOCK_SIZE) {
        OP_LOGE(context->GetNodeName(), "block_size must be in [1, 8]");
        return ge::GRAPH_FAILED;
    }
    gert::Shape *outShape = context->GetOutputShape(TOPK_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    // topk_output: [T1, N2, blockCount * blockSize]
    outShape->SetDimNum(3);
    outShape->SetDim(0, topkShape->GetDim(0));
    outShape->SetDim(1, topkShape->GetDim(1));
    outShape->SetDim(2, topkShape->GetDim(2) * blockSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeBlockIndexerRearrange(gert::InferDataTypeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    context->SetOutputDataType(TOPK_OUTPUT_INDEX, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FiBlockIndexerRearrange)
    .InferShape(InferShapeBlockIndexerRearrange)
    .InferDataType(InferDataTypeBlockIndexerRearrange);
} // namespace ops
