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
 * \file grouped_matmul_k_dim.cpp
 * \brief Device entry of the K-dim grouped matmul (one translation unit).
 *
 * Grouping runs along the shared K axis; every group produces its own [M, N] tile.
 * `group_list` is the cumulative prefix array over K, so the kernel derives each
 * slice length from consecutive entries.
 */
#include "kernel_operator.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "arch22/grouped_matmul_k_dim.h"
#else
#error "grouped_matmul_k_dim currently requires arch22 (__CCE_AICORE__ == 220)"
#endif

using namespace AscendC;
using namespace matmul;

#define EXEC_GMM_KDIM_STEP0 \
    using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type, trans_a>; \
    using weightType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type, false>; \
    using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type>; \
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type>; \
    using matmulType = matmul::MatmulImpl<xType, weightType, yType, biasType, CFG_MDL>; \
    matmulType mm; \
    mm.SetSubBlockIdx(0);

#define EXEC_GMM_KDIM_STEP0_MIXED_OUTPUT(output_type) \
    using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type, trans_a>; \
    using weightType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type, false>; \
    using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, output_type>; \
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, output_type>; \
    using matmulType = matmul::MatmulImpl<xType, weightType, yType, biasType, CFG_MDL>; \
    matmulType mm; \
    mm.SetSubBlockIdx(0);

#define EXEC_GMM_KDIM_STEP1 \
    mm.Init(mmTiling, &tPipe); \
    grouped_matmul_kdim::GMMCompute<decltype(mm), tensor_type, tensor_type, int64_t, trans_a, false> computeOp(mm); \
    computeOp.Init(x, weight, group_list, y); \
    grouped_matmul_kdim::GMMProcess<decltype(computeOp)> op(computeOp, tiling_data); \
    op.Init(); \
    op.Process();

#define EXEC_GMM_KDIM_STEP1_MIXED_OUTPUT(output_type) \
    mm.Init(mmTiling, &tPipe); \
    grouped_matmul_kdim::GMMCompute<decltype(mm), tensor_type, output_type, int64_t, trans_a, false> computeOp(mm); \
    computeOp.Init(x, weight, group_list, y); \
    grouped_matmul_kdim::GMMProcess<decltype(computeOp)> op(computeOp, tiling_data); \
    op.Init(); \
    op.Process();

// Ascend C's generic pipeline needs an explicit preload depth on arch22; the
// matmul instruction stream stalls without it. Kept verbatim from the upstream
// operator rather than re-derived.
#define GMM_KDIM_PRELOAD() PRELOAD(4)

extern "C" __global__ __aicore__ void fi_grouped_matmul_k_dim(GM_ADDR x, GM_ADDR weight, GM_ADDR group_list, GM_ADDR y,
                                                              GM_ADDR workspace, GM_ADDR tiling)
{
    // Only cube cores execute this kernel; the vector cores retire immediately.
    if (g_coreType == AIV) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GET_TILING_DATA(tiling_data, tiling);

    TPipe tPipe;
    // Saturation on FP32 accumulation paths, matching the upstream operator.
    AscendCUtils::SetOverflow(1);

    const TCubeTiling *__restrict mmTiling = &(tiling_data.mmTilingData);

    if (TILING_KEY_IS(0)) {
        constexpr bool trans_a = false;
        using tensor_type = half;
        EXEC_GMM_KDIM_STEP0
        GMM_KDIM_PRELOAD();
        EXEC_GMM_KDIM_STEP1
    } else if (TILING_KEY_IS(1)) {
        constexpr bool trans_a = true;
        using tensor_type = half;
        EXEC_GMM_KDIM_STEP0
        GMM_KDIM_PRELOAD();
        EXEC_GMM_KDIM_STEP1
    } else if (TILING_KEY_IS(2)) {
        constexpr bool trans_a = false;
        using tensor_type = bfloat16_t;
        EXEC_GMM_KDIM_STEP0
        GMM_KDIM_PRELOAD();
        EXEC_GMM_KDIM_STEP1
    } else if (TILING_KEY_IS(3)) {
        constexpr bool trans_a = true;
        using tensor_type = bfloat16_t;
        EXEC_GMM_KDIM_STEP0
        GMM_KDIM_PRELOAD();
        EXEC_GMM_KDIM_STEP1
    } else if (TILING_KEY_IS(4)) {
        constexpr bool trans_a = false;
        using tensor_type = half;
        using output_type = float;
        EXEC_GMM_KDIM_STEP0_MIXED_OUTPUT(output_type)
        GMM_KDIM_PRELOAD();
        EXEC_GMM_KDIM_STEP1_MIXED_OUTPUT(output_type)
    } else if (TILING_KEY_IS(5)) {
        constexpr bool trans_a = true;
        using tensor_type = half;
        using output_type = float;
        EXEC_GMM_KDIM_STEP0_MIXED_OUTPUT(output_type)
        GMM_KDIM_PRELOAD();
        EXEC_GMM_KDIM_STEP1_MIXED_OUTPUT(output_type)
    } else if (TILING_KEY_IS(6)) {
        constexpr bool trans_a = false;
        using tensor_type = bfloat16_t;
        using output_type = float;
        EXEC_GMM_KDIM_STEP0_MIXED_OUTPUT(output_type)
        GMM_KDIM_PRELOAD();
        EXEC_GMM_KDIM_STEP1_MIXED_OUTPUT(output_type)
    } else if (TILING_KEY_IS(7)) {
        constexpr bool trans_a = true;
        using tensor_type = bfloat16_t;
        using output_type = float;
        EXEC_GMM_KDIM_STEP0_MIXED_OUTPUT(output_type)
        GMM_KDIM_PRELOAD();
        EXEC_GMM_KDIM_STEP1_MIXED_OUTPUT(output_type)
    }
}
