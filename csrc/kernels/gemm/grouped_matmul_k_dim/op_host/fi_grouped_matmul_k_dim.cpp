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
 * \file fi_grouped_matmul_k_dim.cpp
 * \brief OpDef, tiling and shape inference in one host translation unit.
 *
 * Input order and dtype combinations here are the ACLNN ABI that NativeBinding
 * declares; the generated adapter rejects any drift between the two.
 */
#include "fi_host_checks.h"
#include "arch22/grouped_matmul_k_dim_tiling.cpp"
#include "grouped_matmul_k_dim_infershape.cpp"
#include "register/op_def_registry.h"

#define FI_STRING_INNER(value) #value
#define FI_STRING(value) FI_STRING_INNER(value)
#define FI_REGISTER_OP(op) OP_ADD(op)

namespace ops {
class FI_PACKAGE_OP_TYPE : public OpDef {
public:
    explicit FI_PACKAGE_OP_TYPE(const char *name) : OpDef(name)
    {
        // x: (M, K) or (K, M) when trans_a; weight: (K, N); group_list: cumulative
        // prefixes over K; y: one [M, N] tile per group.
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("group_list")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("trans_a").AttrType(OPTIONAL).Bool(false);

        SetInferShape(InferShape4GroupedMatmulKDim);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "aclnn_only")
            .ExtendCfgInfo("jitCompile.flag", "false");
        AICore().AddConfig(FI_STRING(FI_COMPUTE_UNIT), aicoreConfig);
    }
};
FI_REGISTER_OP(FI_PACKAGE_OP_TYPE);
} // namespace ops
