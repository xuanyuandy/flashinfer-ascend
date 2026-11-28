/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDEXER_PROLOGUE_QW_FULL_FUSION_TILING_H_
#define INDEXER_PROLOGUE_QW_FULL_FUSION_TILING_H_

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/soc_spec.h"
#ifdef ASCENDC_ASSERT
#undef ASCENDC_ASSERT
#endif
#include "tiling/tiling_api.h"

namespace optiling {

struct IndexerPrologueQWFullFusionCompileInfo {
    uint64_t coreNum = 0;
    NpuArch npuArch = NpuArch::DAV_2201;
};

__attribute__((visibility("default"))) ge::graphStatus TilingPrepareForIndexerPrologueQWFullFusion(
    gert::TilingParseContext* context);

BEGIN_TILING_DATA_DEF(IndexerPrologueQWFullFusionTilingData)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, totalInGroup);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, n);
TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);
TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);
TILING_DATA_FIELD_DEF(uint32_t, parallNum);
TILING_DATA_FIELD_DEF(uint32_t, dqReduceTmpBytes);
TILING_DATA_FIELD_DEF(uint32_t, dqBroadcastTmpBytes);
TILING_DATA_FIELD_DEF(uint32_t, denseLayout);
TILING_DATA_FIELD_DEF(float, wScale);
TILING_DATA_FIELD_DEF_ARR(int32_t, 50, mmTilingData);
TILING_DATA_FIELD_DEF_ARR(int32_t, 50, wTilingData);
END_TILING_DATA_DEF;

FI_REGISTER_TILING(FI_PACKAGE_OP_TYPE, IndexerPrologueQWFullFusionTilingData)
} // namespace optiling
#endif // INDEXER_PROLOGUE_QW_FULL_FUSION_TILING_H_
