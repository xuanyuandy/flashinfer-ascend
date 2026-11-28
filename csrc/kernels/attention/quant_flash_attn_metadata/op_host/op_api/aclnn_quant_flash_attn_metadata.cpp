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
 * \file aclnn_quant_flash_attn_metadata.cpp
 * \brief
 */

#include "aclnn_quant_flash_attn_metadata.h"
#include "l0_quant_flash_attn_metadata.h"
#include "acl/acl_rt.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

#include "../quant_flash_attn_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnFiQuantFlashAttnMetadataGetWorkspaceSize(
    const aclTensor* cuSeqlensQOptional, const aclTensor* cuSeqlensKvOptional, const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKvOptional, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ,
    int64_t numHeadsKv, int64_t headDim, int64_t headDimV, int64_t quantMode, int64_t maskMode, int64_t winLeft,
    int64_t winRight, const char* layoutQ, const char* layoutQDescale, const char* layoutKv, const char* layoutOut,
    bool isGradEnabled, const aclTensor* metaData, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFiQuantFlashAttnMetadata,
                   DFX_IN(cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional, batchSize,
                          maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, headDimV, quantMode, maskMode,
                          winLeft, winRight, layoutQ, layoutQDescale, layoutKv, layoutOut, isGradEnabled),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = FiQuantFlashAttnMetadataCheck::ParamsCheck(
        cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional, batchSize, maxSeqlenQ,
        maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, headDimV, quantMode, maskMode, winLeft, winRight, layoutQ,
        layoutQDescale, layoutKv, layoutOut, isGradEnabled, metaData);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo& npuInfo = op::GetCurrentPlatformInfo();
    // 与主算子 ACLNN tiling 保持一致：优先获取当前线程的有效核数（含 stream 控核）。
    // 查询失败时，与 opbase InitL2Phase1Context 一样回退到硬件平台核数。
    uint32_t aicCoreNum = 0;
    uint32_t aivCoreNum = 0;
    if (aclrtGetResInCurrentThread(ACL_RT_DEV_RES_CUBE_CORE, &aicCoreNum) != ACL_SUCCESS) {
        aicCoreNum = npuInfo.GetCubeCoreNum();
    }
    if (aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &aivCoreNum) != ACL_SUCCESS) {
        aivCoreNum = npuInfo.GetVectorCoreNum();
    }
    // Torch may expose a flattened StorageShape and AICPU may not receive a TensorShape.
    // Validate the logical output shape on the host and pass the row capacity to AICPU.
    const auto& metadataShape = metaData->GetViewShape();
    const int64_t metadataDimNum = metadataShape.GetDimNum();
    CHECK_COND(metadataDimNum == 2 && metadataShape.GetDim(0) == 1 && metadataShape.GetDim(1) > 0,
               ACLNN_ERR_PARAM_INVALID, "metadata output must have shape [1, C]");
    const int64_t metadataRowSize = metadataShape.GetDim(1);

    auto output = l0op::FiQuantFlashAttnMetadata(
        cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional, batchSize, maxSeqlenQ,
        maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, quantMode, maskMode, winLeft, winRight, layoutQ,
        layoutQDescale, layoutKv, layoutOut, aicCoreNum, aivCoreNum, metadataDimNum, metadataRowSize,
        metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnFiQuantFlashAttnMetadata(void* workspace, uint64_t workspaceSize,
                                                                               aclOpExecutor* executor,
                                                                               aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFiQuantFlashAttnMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
