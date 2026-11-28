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
 * \file indexer_prologue_qw_def.cpp
 * \brief Shared host declaration for the dense and caller-prequantized packages.
 *
 * Both packages build this directory and differ only in the package-level
 * definition INDEXER_PROLOGUE_QW_PREQUANTIZED, which selects the physical weight
 * contract. The device implementation and source identity are shared; the op
 * type, vendor and ACLNN host entry are derived from FI_PACKAGE_OP_TYPE.
 */

#include "fi_host_checks.h"
#include "arch22/indexer_prologue_qw_tiling.cpp"
#include "indexer_prologue_qw_infershape.cpp"
#include "register/op_def_registry.h"

#define FI_STRING_INNER(x) #x
#define FI_STRING(x) FI_STRING_INNER(x)

namespace ops {
class FI_PACKAGE_OP_TYPE : public OpDef {
public:
    explicit FI_PACKAGE_OP_TYPE(const char* name) : OpDef(name)
    {
        this->Input("qr").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("activation_q")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
#ifdef INDEXER_PROLOGUE_QW_PREQUANTIZED
        this->Input("weight_q")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_FRACTAL_NZ})
            .IgnoreContiguous();
        this->Input("wqb")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_FRACTAL_NZ})
            .IgnoreContiguous();
#else
        this->Input("weight_q")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("wqb").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
#endif
        this->Input("weight_scale")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("activation_scale")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("qb").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("x").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("ww").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("wb").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("rope_sin")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("rope_cos")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Output("q").ParamType(REQUIRED).DataTypeList({ge::DT_INT8}).FormatList({ge::FORMAT_ND});
        this->Output("q_scale").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});
        this->Output("w").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});
        this->Attr("w_scale").AttrType(REQUIRED).Float(1.0F);

        SetInferShape(InferShape4IndexerPrologueQWFullFusion).SetInferDataType(InferDtype4IndexerPrologueQWCommon);
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig(FI_STRING(FI_COMPUTE_UNIT), aicoreConfig);
    }
};
FI_REGISTER_OP(FI_PACKAGE_OP_TYPE);
} // namespace ops
