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
 * \file mla_prolog_infershape.cpp
 * \brief Shape/type inference for the fixed DS4.1 MLA prolog contract.
 */

#include <register/op_impl_registry.h>
#include "fi_host_checks.h"

using namespace ge;

namespace ops {
constexpr size_t X_INPUT_INDEX = 0;
constexpr size_t Q_OUTPUT_INDEX = 0;
constexpr size_t QR_OUTPUT_INDEX = 1;
constexpr uint32_t HEADS = 64;
constexpr uint32_t HEAD_DIM = 512;
constexpr uint32_t Q_LORA = 1280;

ge::graphStatus InferShapeMlaProlog(gert::InferShapeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const gert::Shape *xShape = context->GetInputShape(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    if (xShape->GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(), "x must be rank-2");
        return ge::GRAPH_FAILED;
    }
    gert::Shape *qShape = context->GetOutputShape(Q_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    qShape->SetDimNum(3);
    qShape->SetDim(0, xShape->GetDim(0));
    qShape->SetDim(1, HEADS);
    qShape->SetDim(2, HEAD_DIM);
    gert::Shape *qrShape = context->GetOutputShape(QR_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qrShape);
    qrShape->SetDimNum(2);
    qrShape->SetDim(0, xShape->GetDim(0));
    qrShape->SetDim(1, Q_LORA);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeMlaProlog(gert::InferDataTypeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    context->SetOutputDataType(Q_OUTPUT_INDEX, context->GetInputDataType(X_INPUT_INDEX));
    context->SetOutputDataType(QR_OUTPUT_INDEX, context->GetInputDataType(X_INPUT_INDEX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FiMlaProlog)
    .InferShape(InferShapeMlaProlog)
    .InferDataType(InferDataTypeMlaProlog);
} // namespace ops
