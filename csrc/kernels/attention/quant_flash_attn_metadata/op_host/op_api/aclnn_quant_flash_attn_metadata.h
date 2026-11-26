#ifndef FI_ACLNN_QUANT_FLASH_ATTN_METADATA_H
#define FI_ACLNN_QUANT_FLASH_ATTN_METADATA_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnFiQuantFlashAttnMetadataGetWorkspaceSize(
    const aclTensor* cuSeqlensQ, const aclTensor* cuSeqlensKvOptional, const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKv, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t headDimV, int64_t quantMode,
    int64_t maskMode, int64_t winLeft, int64_t winRight, const char* layoutQ, const char* layoutQDescale,
    const char* layoutKv, const char* layoutOut, bool isGradEnabled, const aclTensor* metadata,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnFiQuantFlashAttnMetadata(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
