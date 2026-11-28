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
 * \file indexer_prologue_k_def.cpp
 * \brief Rich mutation-only K prologue: inputs 0..7, aliased cache outputs 8..9.
 *
 * The two cache outputs are declared separately because the ACLNN Tensor ABI
 * requires unique parameter names. The rich runner passes the exact same cache
 * views for input and output; the additive strided-alias bridge verifies the
 * aliases and preserves independent block/slot strides.
 */

#include "fi_host_checks.h"
#include "arch22/indexer_prologue_k_tiling.cpp"
#include "indexer_prologue_k_infershape.cpp"
#include "register/op_def_registry.h"

#define FI_STRING_INNER(x) #x
#define FI_STRING(x) FI_STRING_INNER(x)

namespace ops {
class FI_PACKAGE_OP_TYPE : public OpDef {
public:
    explicit FI_PACKAGE_OP_TYPE(const char* name) : OpDef(name)
    {
        this->Input("latent").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("wk").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("norm_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rope_cos")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("rope_sin")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("k_cache")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("k_scale_cache")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Output("k_cache_out").ParamType(REQUIRED).DataTypeList({ge::DT_INT8}).FormatList({ge::FORMAT_ND});
        this->Output("k_scale_cache_out")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Attr("norm_eps").AttrType(REQUIRED).Float(1.0e-6F);

        SetInferShape(InferShape4IndexerPrologueKFullFusion).SetInferDataType(InferDtype4IndexerPrologueKFullFusion);
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false);
        this->AICore().AddConfig(FI_STRING(FI_COMPUTE_UNIT), aicoreConfig);
    }
};
FI_REGISTER_OP(FI_PACKAGE_OP_TYPE);
} // namespace ops
