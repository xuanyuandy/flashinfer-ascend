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
 * \file grouped_matmul_k_dim_infershape.cpp
 * \brief Shape inference: the K-dim grouped matmul emits one [M, N] tile per group.
 */
#include "register/op_impl_registry.h"
#include "fi_host_checks.h"

namespace ge {

static ge::graphStatus InferShape4GroupedMatmulKDim(gert::InferShapeContext *context)
{
    // (M, K) or (K, M)
    const gert::Shape *xShape = context->GetInputShape(0);
    // (K, N)
    const gert::Shape *weightShape = context->GetInputShape(1);
    // cumulative prefixes over K
    const gert::Shape *groupListShape = context->GetInputShape(2);
    if (xShape->GetDimNum() != 2 || weightShape->GetDimNum() != 2 || groupListShape->GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "Dim of inputs not right");
        return ge::GRAPH_FAILED;
    }

    bool trans_a = *(context->GetAttrs()->GetAttrPointer<bool>(0));

    auto M = trans_a ? xShape->GetDim(1) : xShape->GetDim(0);
    auto N = weightShape->GetDim(1);
    auto groupNum = groupListShape->GetDim(0);

    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = gert::Shape({groupNum, M, N});
    return GRAPH_SUCCESS;
}

FI_IMPL_SHAPE(FI_PACKAGE_OP_TYPE).InferShape(InferShape4GroupedMatmulKDim);
} // namespace ge
