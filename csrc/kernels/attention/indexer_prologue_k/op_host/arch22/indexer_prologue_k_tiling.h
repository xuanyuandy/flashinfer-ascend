/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INDEXER_PROLOGUE_K_FULL_FUSION_TILING_H_
#define INDEXER_PROLOGUE_K_FULL_FUSION_TILING_H_

#ifndef __CCE_AICORE__
#include "register/op_impl_registry.h"
#endif
#include "register/tilingdata_base.h"
#include "platform/soc_spec.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexerPrologueKFullFusionTilingData)
TILING_DATA_FIELD_DEF(int32_t, totalRows);
TILING_DATA_FIELD_DEF(int32_t, numBatch);
TILING_DATA_FIELD_DEF(int32_t, numSlotPerBatch);
TILING_DATA_FIELD_DEF(int32_t, usedCores);
TILING_DATA_FIELD_DEF(int32_t, baseM);
TILING_DATA_FIELD_DEF(uint64_t, kCacheBlockStride);
TILING_DATA_FIELD_DEF(uint64_t, kCacheSlotStride);
TILING_DATA_FIELD_DEF(uint64_t, kScaleBlockStride);
TILING_DATA_FIELD_DEF(uint64_t, kScaleSlotStride);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(int32_t, normIsFp32);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTiling);
END_TILING_DATA_DEF;

#ifndef __CCE_AICORE__
FI_REGISTER_TILING(FI_PACKAGE_OP_TYPE, IndexerPrologueKFullFusionTilingData)
#endif

struct IndexerPrologueKFullFusionCompileInfo {
    uint64_t aicNum = 0;
    uint64_t aivNum = 0;
    NpuArch npuArch = NpuArch::DAV_2201;
};
} // namespace optiling
#endif // INDEXER_PROLOGUE_K_FULL_FUSION_TILING_H_
