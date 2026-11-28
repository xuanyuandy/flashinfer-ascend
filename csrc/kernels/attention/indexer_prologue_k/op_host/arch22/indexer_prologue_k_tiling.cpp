/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indexer_prologue_k_tiling.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "fi_host_checks.h"

using namespace ge;
namespace optiling {
namespace {
constexpr int64_t INPUT_DIM = 512;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t ROPE_DIM = 64;
constexpr int32_t BASE_M = 16;
constexpr int32_t BASE_K = 64;
constexpr int32_t SMALL_BASE_M = 4;
constexpr int32_t SMALL_BASE_K = 32;
constexpr uint64_t TILE_BYTES = BASE_M * HEAD_DIM * sizeof(uint16_t);

int32_t SelectBaseM(int64_t tokens, uint64_t aicNum)
{
    // One AIC wave: use the small-M path only when every tile fits on an AIC.
    // Larger M keeps the clean baseM=16/baseK=64/VECIN path, avoiding extra waves.
    return (tokens + SMALL_BASE_M - 1) / SMALL_BASE_M <= static_cast<int64_t>(aicNum) ? SMALL_BASE_M : BASE_M;
}

ge::graphStatus CheckShapes(gert::TilingContext* context, int64_t& tokens, int64_t& nb, int64_t& ns,
                            uint64_t& keyBlockStride, uint64_t& keySlotStride, uint64_t& scaleBlockStride,
                            uint64_t& scaleSlotStride)
{
    for (size_t i = 0; i < 8; ++i) {
        OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(i));
        OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(i));
    }
    const auto& latent = context->GetInputShape(0)->GetStorageShape();
    const auto& wk = context->GetInputShape(1)->GetStorageShape();
    const auto& norm = context->GetInputShape(2)->GetStorageShape();
    const auto& cos = context->GetInputShape(3)->GetStorageShape();
    const auto& sin = context->GetInputShape(4)->GetStorageShape();
    const auto& indices = context->GetInputShape(5)->GetStorageShape();
    const auto& cache = context->GetInputShape(6)->GetStorageShape();
    const auto& scale = context->GetInputShape(7)->GetStorageShape();
    OP_CHECK_IF(latent.GetDimNum() != 2 || latent.GetDim(1) != INPUT_DIM,
                OP_LOGE(context->GetNodeName(), "latent must be [T,512]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(wk.GetDimNum() != 2 || wk.GetDim(0) != HEAD_DIM || wk.GetDim(1) != INPUT_DIM,
                OP_LOGE(context->GetNodeName(), "wk must be [128,512]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(norm.GetShapeSize() != HEAD_DIM,
                OP_LOGE(context->GetNodeName(), "norm_weight must contain 128 elements"), return ge::GRAPH_FAILED);
    tokens = latent.GetDim(0);
    OP_CHECK_IF(cos.GetDimNum() != 2 || cos.GetDim(0) != tokens || cos.GetDim(1) != ROPE_DIM,
                OP_LOGE(context->GetNodeName(), "rope_cos must be [T,64]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(sin.GetDimNum() != 2 || sin.GetDim(0) != tokens || sin.GetDim(1) != ROPE_DIM,
                OP_LOGE(context->GetNodeName(), "rope_sin must match rope_cos"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(indices.GetDimNum() != 2 || indices.GetDim(0) != tokens || indices.GetDim(1) != 2,
                OP_LOGE(context->GetNodeName(), "indices must be [T,2]"), return ge::GRAPH_FAILED);
    const bool cacheValid = (cache.GetDimNum() == 3 && cache.GetDim(2) == HEAD_DIM) ||
                            (cache.GetDimNum() == 4 && cache.GetDim(2) == 1 && cache.GetDim(3) == HEAD_DIM);
    OP_CHECK_IF(!cacheValid, OP_LOGE(context->GetNodeName(), "k_cache must be [NB,NS,128] or [NB,NS,1,128]"),
                return ge::GRAPH_FAILED);
    nb = cache.GetDim(0);
    ns = cache.GetDim(1);
    const bool scaleValid =
        (scale.GetDimNum() == 3 && scale.GetDim(0) == nb && scale.GetDim(1) == ns && scale.GetDim(2) == 1) ||
        (scale.GetDimNum() == 4 && scale.GetDim(0) == nb && scale.GetDim(1) == ns && scale.GetDim(2) == 1 &&
         scale.GetDim(3) == 1);
    OP_CHECK_IF(!scaleValid, OP_LOGE(context->GetNodeName(), "k_scale_cache must be [NB,NS,1] or [NB,NS,1,1]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tokens <= 0 || nb < 0 || ns < 0 || tokens > std::numeric_limits<int32_t>::max() ||
                    nb > std::numeric_limits<int32_t>::max() || ns > std::numeric_limits<int32_t>::max(),
                OP_LOGE(context->GetNodeName(), "invalid T/NB/NS for current full-fusion tiling"),
                return ge::GRAPH_FAILED);

    // Never infer a contiguous layout when stride metadata is missing: a lost
    // view flag may still carry non-contiguous cache strides. Both caches must
    // expose real metadata; a non-view/contiguous tensor still has the default
    // strides supplied by the ACL tensor descriptor.
    const auto* keyStrides = context->GetInputStride(6);
    const auto* scaleStrides = context->GetInputStride(7);
    OP_CHECK_IF(keyStrides == nullptr || scaleStrides == nullptr,
                OP_LOGE(context->GetNodeName(), "K cache stride metadata is required"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyStrides->GetDimNum() != cache.GetDimNum() ||
                    keyStrides->GetStride(cache.GetDimNum() - 1) != 1 ||
                    keyStrides->GetStride(0) < 0 || keyStrides->GetStride(1) < 0,
                OP_LOGE(context->GetNodeName(), "k_cache requires matching nonnegative strides and contiguous last axis"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(scaleStrides->GetDimNum() != scale.GetDimNum() ||
                    scaleStrides->GetStride(scale.GetDimNum() - 1) != 1 ||
                    scaleStrides->GetStride(0) < 0 || scaleStrides->GetStride(1) < 0,
                OP_LOGE(context->GetNodeName(), "k_scale_cache requires matching nonnegative strides and contiguous last axis"),
                return ge::GRAPH_FAILED);
    keyBlockStride = static_cast<uint64_t>(keyStrides->GetStride(0));
    keySlotStride = static_cast<uint64_t>(keyStrides->GetStride(1));
    scaleBlockStride = static_cast<uint64_t>(scaleStrides->GetStride(0));
    scaleSlotStride = static_cast<uint64_t>(scaleStrides->GetStride(1));
    OP_CHECK_IF((nb > 1 && keyBlockStride == 0) || (ns > 1 && keySlotStride == 0) ||
                    (nb > 1 && scaleBlockStride == 0) || (ns > 1 && scaleSlotStride == 0),
                OP_LOGE(context->GetNodeName(), "cache block and slot strides must be positive"),
                return ge::GRAPH_FAILED);
    const auto normType = context->GetInputDesc(2)->GetDataType();
    OP_CHECK_IF(normType != ge::DT_BF16 && normType != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(), "norm_weight must be BF16 or FP32"), return ge::GRAPH_FAILED);
    const ge::DataType expected[] = {ge::DT_BF16,  ge::DT_BF16,  normType,    ge::DT_FLOAT,
                                     ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8, ge::DT_FLOAT16};
    for (size_t i = 0; i < 8; ++i) {
        OP_CHECK_IF(context->GetInputDesc(i)->GetDataType() != expected[i],
                    OP_LOGE(context->GetNodeName(), "input dtype mismatch at %zu", i), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus TilingPrepareForIndexerPrologueKFullFusion(gert::TilingParseContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto* compileInfo = context->GetCompiledInfo<IndexerPrologueKFullFusionCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    compileInfo->aicNum = platform.GetCoreNumAic();
    compileInfo->aivNum = platform.GetCoreNumAiv();
    compileInfo->npuArch = platform.GetCurNpuArch();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForIndexerPrologueKFullFusion(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint64_t aicNum = platform.GetCoreNumAic();
    const uint64_t aivNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(platform.GetCurNpuArch() != NpuArch::DAV_2201 || aicNum == 0 || aivNum == 0,
                OP_LOGE(context->GetNodeName(), "full fusion requires arch22 AIC/AIV"), return ge::GRAPH_FAILED);

    int64_t tokens = 0;
    int64_t nb = 0;
    int64_t ns = 0;
    uint64_t keyBlockStride = 0;
    uint64_t keySlotStride = 0;
    uint64_t scaleBlockStride = 0;
    uint64_t scaleSlotStride = 0;
    if (CheckShapes(context, tokens, nb, ns, keyBlockStride, keySlotStride, scaleBlockStride, scaleSlotStride) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* eps = attrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, eps);

    const int32_t baseM = SelectBaseM(tokens, aicNum);
    const int32_t baseK = baseM == SMALL_BASE_M ? SMALL_BASE_K : BASE_K;
    const int32_t totalTiles = static_cast<int32_t>((tokens + baseM - 1) / baseM);
    const int32_t usedCores = std::min(totalTiles, static_cast<int32_t>(aicNum));
    OP_CHECK_IF(usedCores <= 0, OP_LOGE(context->GetNodeName(), "invalid zero core count for full fusion"),
                return ge::GRAPH_FAILED);
    matmul_tiling::MultiCoreMatmulTiling mmTilingApi(platform);
    mmTilingApi.SetDim(1);
    mmTilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16,
                         false);
    mmTilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16,
                         true);
    mmTilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
    mmTilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT);
    mmTilingApi.SetOrgShape(baseM, HEAD_DIM, INPUT_DIM);
    mmTilingApi.SetShape(baseM, HEAD_DIM, INPUT_DIM);
    mmTilingApi.EnableBias(false);
    mmTilingApi.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM);
    mmTilingApi.SetFixSplit(baseM, HEAD_DIM, baseK);
    mmTilingApi.SetBufferSpace(-1, -1, -1);

    IndexerPrologueKFullFusionTilingData tilingData;
    OP_CHECK_IF(mmTilingApi.GetTiling(tilingData.mmTiling) == -1,
                OP_LOGE(context->GetNodeName(), "Matmul tiling generation failed"), return ge::GRAPH_FAILED);
    tilingData.mmTiling.set_usedCoreNum(usedCores);
    OP_CHECK_IF(tilingData.mmTiling.get_usedCoreNum() != usedCores || tilingData.mmTiling.get_singleCoreM() != baseM ||
                    tilingData.mmTiling.get_singleCoreN() != HEAD_DIM ||
                    tilingData.mmTiling.get_singleCoreK() != INPUT_DIM,
                OP_LOGE(context->GetNodeName(), "unexpected Matmul tiling for full fusion"), return ge::GRAPH_FAILED);

    tilingData.set_totalRows(static_cast<int32_t>(tokens));
    tilingData.set_numBatch(static_cast<int32_t>(nb));
    tilingData.set_numSlotPerBatch(static_cast<int32_t>(ns));
    tilingData.set_usedCores(usedCores);
    tilingData.set_baseM(baseM);
    tilingData.set_kCacheBlockStride(keyBlockStride);
    tilingData.set_kCacheSlotStride(keySlotStride);
    tilingData.set_kScaleBlockStride(scaleBlockStride);
    tilingData.set_kScaleSlotStride(scaleSlotStride);
    tilingData.set_eps(*eps);
    tilingData.set_normIsFp32(context->GetInputDesc(2)->GetDataType() == ge::DT_FLOAT ? 1 : 0);
    context->SetBlockDim(usedCores);
    context->SetTilingKey(0);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = platform.GetLibApiWorkSpaceSize() + static_cast<size_t>(usedCores) * TILE_BYTES;
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

FI_IMPL_TILING(FI_PACKAGE_OP_TYPE)
    .Tiling(TilingForIndexerPrologueKFullFusion)
    .TilingParse<IndexerPrologueKFullFusionCompileInfo>(TilingPrepareForIndexerPrologueKFullFusion);
} // namespace optiling
