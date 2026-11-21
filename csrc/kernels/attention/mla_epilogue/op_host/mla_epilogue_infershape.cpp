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
 * \file mla_epilogue_infershape.cpp
 * \brief mla_epilogue 形状推导：y = [x.shape[0], 5120]
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

constexpr int64_t MLA_EPILOGUE_OUT_N = 5120;

static ge::graphStatus InferShape4MlaEpilogue(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // x 为 [S, 1, 64, 512]；动态 rank 场景首维未知时填 -1
    int64_t s = -1;
    if (xShape->GetDimNum() == 4U) {
        s = xShape->GetDim(0);
    } else {
        OP_LOGW(context, "x rank is %zu, expected 4, output dim0 set to -1", xShape->GetDimNum());
    }

    yShape->SetDimNum(2U);
    yShape->SetDim(0, s);
    yShape->SetDim(1, MLA_EPILOGUE_OUT_N);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FiMlaEpilogue).InferShape(InferShape4MlaEpilogue);

} // namespace ops
