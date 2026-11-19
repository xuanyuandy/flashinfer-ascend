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
 * \file grouped_matmul_k_dim_tiling.h
 * \brief Tiling data shared by the host tiler and the device entry.
 */
#ifndef GROUPED_MATMUL_K_DIM_TILING_H_
#define GROUPED_MATMUL_K_DIM_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GroupedMatmulKDimBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, M);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, TotalK);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, trans_a);
TILING_DATA_FIELD_DEF(uint32_t, singleM);
TILING_DATA_FIELD_DEF(uint32_t, singleN);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedMatmulKDimBaseParamsOp, GroupedMatmulKDimBaseParams)

BEGIN_TILING_DATA_DEF(GroupedMatmulKDimTilingData)
TILING_DATA_FIELD_DEF_STRUCT(GroupedMatmulKDimBaseParams, gmmBaseParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

// Registered against the build-injected op type so opbuild can bind the tiling
// struct to the operator and emit a working GET_TILING_DATA for the entry.
FI_REGISTER_TILING(FI_PACKAGE_OP_TYPE, GroupedMatmulKDimTilingData)
} // namespace optiling

#endif
