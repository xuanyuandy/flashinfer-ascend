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
 * \file mla_epilogue.cpp
 * \brief mla_epilogue Kernel 入口（arch22 / Ascend910_93）
 *
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

#include "arch22/mla_epilogue_tiling_data.h"
#include "arch22/mla_epilogue_tiling_key.h"
#include "arch22/mla_epilogue_gemm.h"
#include "arch22/mla_epilogue_gemm_large.h"
#include "arch22/mla_epilogue_rope.h"

using namespace AscendC;

/**
 * @brief mla_epilogue 全功能 Kernel 入口（单 TilingKey）
 *
 * @param x         [S, 1, 64, 512] bf16，行内布局 [8 g][8 h][512 d]
 * @param cos       [S, 1, 1, 64] fp32
 * @param sin       [S, 1, 1, 64] fp32
 * @param w1        [8, 4096, 1024] bf16
 * @param w2        [5120, 8192] bf16
 * @param y         [S, 5120] bf16（直接写用户输出）
 * @param workspace 系统 workspace + 用户 workspace（r / c 中间结果）
 * @param tiling    MlaEpilogueTilingData
 */
template <uint32_t SCH_MODE>
__global__ __aicore__ void fi_mla_epilogue(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR w1, GM_ADDR w2, GM_ADDR y,
                                        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); // AIC:AIV = 1:2

    if constexpr ((SCH_MODE == 0U) || (SCH_MODE == 1U)) {
        REGISTER_TILING_DEFAULT(MlaEpilogueTilingData); // 必须在 GET_TILING_DATA_WITH_STRUCT 之前
        GET_TILING_DATA_WITH_STRUCT(MlaEpilogueTilingData, tilingData, tiling);
        GM_ADDR userWs = GetUserWorkspace(workspace);

        if ASCEND_IS_AIC {
            Catlass::Arch::Resource<NsMlaEpilogue::CatlassArch> resource;
            if constexpr (SCH_MODE == 0U) {
                NsMlaEpilogue::AicContext aic(resource);
                aic.Init(tilingData, userWs, w1, w2, y);
                CrossCoreWaitFlag<2>(0);
                aic.Gemm1(tilingData);
                CrossCoreSetFlag<0, PIPE_FIX>(2);
                CrossCoreWaitFlag<0, PIPE_MTE2>(2);
                aic.Gemm2(tilingData);
            } else {
                NsMlaEpilogue::AicContextLarge aic(resource);
                aic.Init(tilingData, userWs, w1, w2, y);
                CrossCoreWaitFlag<2>(0);
                aic.Gemm1(tilingData);
                CrossCoreSetFlag<0, PIPE_FIX>(2);
                CrossCoreWaitFlag<0, PIPE_MTE2>(2);
                aic.Gemm2(tilingData);
            }
        }

        if ASCEND_IS_AIV {
            TPipe pipe;
            NsMlaEpilogue::RopeContext rope;
            rope.Init(tilingData, &pipe, userWs, x, cos, sin);
            for (uint32_t t = 0U; t < tilingData.pipeTileNum; ++t) {
                rope.Tile(t, tilingData);
            }
            SyncAll<true>();
            CrossCoreSetFlag<2, PIPE_MTE3>(0);
        }
    }
}
