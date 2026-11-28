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
 * \file block_indexer_rearrange_def.cpp
 * \brief OpDef for FiBlockIndexerRearrange (TND block-index expansion, arch22).
 *
 * Expands block-level TopK indices to per-token indices for sparse attention
 * pre-processing. Migrated from gitcode ops-transformer
 * experimental/attention/block_indexer_rearrange.
 */

#include "fi_host_checks.h"
#include "arch22/block_indexer_rearrange_tiling.cpp"
#include "block_indexer_rearrange_infershape.cpp"
#include "register/op_def_registry.h"

#define FI_STRING_INNER(x) #x
#define FI_STRING(x) FI_STRING_INNER(x)

namespace ops {
class FiBlockIndexerRearrange : public OpDef {
public:
    explicit FiBlockIndexerRearrange(const char *name) : OpDef(name)
    {
        this->Input("topk_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("acc_seq_qlen")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("acc_seq_kvlen")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("topk_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});

        // Threaded through the tiling payload; the kernel ABI forwards only
        // Tensor arguments plus workspace/tiling.
        this->Attr("causal").AttrType(OPTIONAL).Bool(false);
        this->Attr("block_size").AttrType(OPTIONAL).Int(1);

        SetInferShape(InferShapeBlockIndexerRearrange)
            .SetInferDataType(InferDataTypeBlockIndexerRearrange);
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
OP_ADD(FiBlockIndexerRearrange);
} // namespace ops
