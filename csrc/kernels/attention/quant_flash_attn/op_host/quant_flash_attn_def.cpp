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

// Keep a single host translation unit for CANN codegen, as with SFA.
#include "arch35/quant_flash_attn_tiling_mxfp4.cpp"
#include "quant_flash_attn_infershape.cpp"

namespace {
// GE wire values from ops-transformer's op_const_def_transformer.h.
constexpr auto kE8M0 = static_cast<ge::DataType>(37);
constexpr auto kE2M1 = static_cast<ge::DataType>(40);
} // namespace

namespace ops {
class FiQuantFlashAttn : public OpDef {
public:
    explicit FiQuantFlashAttn(const char* name) : OpDef(name)
    {
        Input("q").ParamType(REQUIRED).DataType({kE2M1}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        Input("k").ParamType(REQUIRED).DataType({kE2M1}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        Input("v").ParamType(REQUIRED).DataType({kE2M1}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        Input("q_descale").ParamType(REQUIRED).DataType({kE8M0}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        Input("k_descale").ParamType(REQUIRED).DataType({kE8M0}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        Input("v_descale").ParamType(REQUIRED).DataType({kE8M0}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        Input("block_table").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        // Keep the reserved inputs in the upstream device argument order.
        Input("p_scale").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        Input("cu_seqlens_q").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        Input("cu_seqlens_kv").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("seqused_q").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        Input("seqused_kv").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        Input("sinks").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        Input("attn_mask").ParamType(OPTIONAL).DataTypeList({ge::DT_INT8}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        Input("metadata").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        Output("attn_out").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Output("softmax_lse").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});

        Attr("quant_compute_mode").AttrType(REQUIRED).Int(quant_flash_attn_host::kC4A4);
        Attr("softmax_scale").AttrType(OPTIONAL).Float(1.0F);
        Attr("mask_mode").AttrType(OPTIONAL).Int(quant_flash_attn_host::kNoMask);
        Attr("win_left").AttrType(OPTIONAL).Int(-1);
        Attr("win_right").AttrType(OPTIONAL).Int(-1);
        Attr("max_seqlen_q").AttrType(OPTIONAL).Int(-1);
        Attr("max_seqlen_kv").AttrType(OPTIONAL).Int(-1);
        Attr("layout_q").AttrType(OPTIONAL).String("TND");
        Attr("layout_q_descale").AttrType(OPTIONAL).String("TND");
        Attr("layout_kv").AttrType(OPTIONAL).String("PA_BBND");
        Attr("layout_out").AttrType(OPTIONAL).String("TND");
        Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);
        Attr("quant_schema").AttrType(OPTIONAL).Int(2);

        SetInferShape(quant_flash_attn_host::InferShape).SetInferDataType(quant_flash_attn_host::InferDataType);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("opFile.value", "quant_flash_attn")
            .ExtendCfgInfo("opInterface.value", "fi_quant_flash_attn");
        AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(FiQuantFlashAttn);
} // namespace ops
