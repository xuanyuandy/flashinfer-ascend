/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"
#include "quant_flash_attn_host_contract.h"

namespace quant_flash_attn_host {
ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* q = context->GetInputShape(Q);
    const auto* v = context->GetInputShape(V);
    auto* output = context->GetOutputShape(0);
    auto* lse = context->GetOutputShape(1);
    const auto* attrs = context->GetAttrs();
    if (q == nullptr || q->GetDimNum() != 3 || v == nullptr || (v->GetDimNum() != 4 && v->GetDimNum() != 5) ||
        output == nullptr || lse == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* returnLse = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE);
    if (returnLse == nullptr) {
        return ge::GRAPH_FAILED;
    }
    output->SetDimNum(3);
    output->SetDim(0, q->GetDim(0));
    output->SetDim(1, q->GetDim(1));
    output->SetDim(2, q->GetDim(2)); // C4A4 requires Dv == D.
    lse->SetDimNum(*returnLse ? 2 : 1);
    lse->SetDim(0, *returnLse ? q->GetDim(1) : 0);
    if (*returnLse) {
        lse->SetDim(1, q->GetDim(0));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
} // namespace quant_flash_attn_host
