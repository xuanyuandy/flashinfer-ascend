/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

using namespace ge;
namespace ops {
namespace {
graphStatus InferShape4IndexerPrologueKFullFusion(gert::InferShapeContext* context)
{
    const gert::Shape* cache = context->GetInputShape(6);
    const gert::Shape* scale = context->GetInputShape(7);
    gert::Shape* outCache = context->GetOutputShape(0);
    gert::Shape* outScale = context->GetOutputShape(1);
    if (cache == nullptr || scale == nullptr || outCache == nullptr || outScale == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outCache = *cache;
    *outScale = *scale;
    return ge::GRAPH_SUCCESS;
}

graphStatus InferDtype4IndexerPrologueKFullFusion(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(6));
    context->SetOutputDataType(1, context->GetInputDataType(7));
    return ge::GRAPH_SUCCESS;
}
} // namespace

FI_IMPL_SHAPE(FI_PACKAGE_OP_TYPE)
    .InferShape(InferShape4IndexerPrologueKFullFusion)
    .InferDataType(InferDtype4IndexerPrologueKFullFusion);
} // namespace ops
