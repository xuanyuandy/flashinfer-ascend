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
constexpr int64_t HEAD_NUM = 32;
constexpr int64_t HEAD_DIM = 128;

graphStatus InferShape4IndexerPrologueQWCommon(gert::InferShapeContext* context, size_t qrIndex)
{
    const gert::Shape* qrShape = context->GetInputShape(qrIndex);
    gert::Shape* qShape = context->GetOutputShape(0);
    gert::Shape* qScaleShape = context->GetOutputShape(1);
    gert::Shape* wShape = context->GetOutputShape(2);
    if (qrShape == nullptr || qShape == nullptr || qScaleShape == nullptr || wShape == nullptr ||
        qrShape->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t tokens = qrShape->GetDim(0);
    *qShape = gert::Shape({tokens, HEAD_NUM, HEAD_DIM});
    *qScaleShape = gert::Shape({tokens, HEAD_NUM});
    *wShape = gert::Shape({tokens, HEAD_NUM});
    return ge::GRAPH_SUCCESS;
}

graphStatus InferShape4IndexerPrologueQWFullFusion(gert::InferShapeContext* context)
{
    // The native ABI consumes qb.T[6], which the runner maps to the public T.
    return InferShape4IndexerPrologueQWCommon(context, 6);
}

graphStatus InferDtype4IndexerPrologueQWCommon(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_INT8);
    context->SetOutputDataType(1, ge::DT_FLOAT16);
    context->SetOutputDataType(2, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}
} // namespace

FI_IMPL_SHAPE(FI_PACKAGE_OP_TYPE)
    .InferShape(InferShape4IndexerPrologueQWFullFusion)
    .InferDataType(InferDtype4IndexerPrologueQWCommon);
} // namespace ops
