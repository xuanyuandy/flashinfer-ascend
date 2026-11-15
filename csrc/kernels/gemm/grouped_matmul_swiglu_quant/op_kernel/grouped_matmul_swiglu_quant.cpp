/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant.cpp
 * \brief
 */
#include "kernel_operator.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "arch22/grouped_matmul_swiglu_quant_kernel.h"
#else
#error "grouped_matmul_swiglu_quant currently requires arch22 (__CCE_AICORE__ == 220)"
#endif

extern "C" __global__ __aicore__ void fi_grouped_matmul_swiglu_quant(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale,
                                                                  GM_ADDR xScale, GM_ADDR weightAssistanceMatrix,
                                                                  GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale,
                                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR user1 = GetUserWorkspace(workspace);
#if defined(GMM_SWIGLU_QUANT_A8W8)
    if (TILING_KEY_IS(0)) { // antiquant msd
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP(GMMSwigluCompute, // computeClass
                         DTYPE_WEIGHT_SCALE,
                         false, // transA
                         false, // transB
                         false  // sync
        );
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP(GMMSwigluSplitWorkSpaceCompute, // computeClass
                         DTYPE_WEIGHT_SCALE,
                         false, // transA
                         false, // transB
                         false  // sync
        );
    }
#elif defined(GMM_SWIGLU_QUANT_A8W4_MSD)
    if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP_A8W4_MSD(GMMSwigluQuantPipelineSchedule, // computeClass
                                  DTYPE_WEIGHT_SCALE,
                                  false, // transA
                                  false, // transB
                                  false  // sync
        );
    }
#endif
}
