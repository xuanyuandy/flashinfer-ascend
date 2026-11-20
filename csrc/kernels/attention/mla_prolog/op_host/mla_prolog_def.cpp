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
 * \file mla_prolog_def.cpp
 * \brief OpDef for FiMlaProlog (BF16 only, A3/arch22).
 *
 * Declares the public tensor order and dtypes; the device entry
 * fi_mla_prolog consumes the tiling/workspace contract below.
 */

#include "fi_host_checks.h"
#include "arch22/mla_prolog_tiling.cpp"
#include "mla_prolog_infershape.cpp"
#include "register/op_def_registry.h"

#define FI_STRING_INNER(x) #x
#define FI_STRING(x) FI_STRING_INNER(x)

namespace ops {
class FiMlaProlog : public OpDef {
public:
    explicit FiMlaProlog(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Input("w_qa")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Input("gamma_qa")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Input("w_qb")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_FRACTAL_NZ});
        this->Input("w_qb_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Input("w_kv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Input("gamma_kv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Input("rope_cos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("rope_sin")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("slot_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        // kv_cache is an input whose addressed rows are updated in place.
        // OpDef cannot express input mutation; the ACLNN parameter is an input
        // pointer. The public mutation contract needs maintainer review.
        this->Input("kv_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        // Host tiling parses this attribute into the tiling payload; the
        // kernel ABI forwards Tensor arguments plus workspace/tiling only.
        // 1e-6 preserves the frozen contract, DS4.1 callers may pass 1e-20.
        this->Attr("rms_norm_eps").AttrType(OPTIONAL).Float(1e-6);
        this->Output("q_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        // Required second output: BF16 q_norm(wq_a(x)), [t,1280] ND.
        this->Output("qr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});

        SetInferShape(InferShapeMlaProlog).SetInferDataType(InferDataTypeMlaProlog);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig(FI_STRING(FI_COMPUTE_UNIT), aicore_config);
    }
};
OP_ADD(FiMlaProlog);
} // namespace ops
