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
 * \file mla_prolog_tiling.cpp
 * \brief Host tiling/workspace for the A3 (arch22) DS4.1 MLA prolog entry.
 *
 * The tiling payload is the POD struct shared with fi_mla_prolog. It
 * carries the standalone MakeWorkspaceLayout offsets and the hardware sync
 * address; the device entry calls GetUserWorkspace before applying offsets.
 */

#include <acl/acl_rt.h>
#include <cmath>
#include "register/op_impl_registry.h"
#include "fi_host_checks.h"
#include "tiling/platform/platform_ascendc.h"
#include "mla_prolog_tiling_data.h"

namespace optiling {
namespace {
constexpr uint32_t kVersion = 4;
constexpr size_t kRmsNormEpsAttrIndex = 0;
constexpr size_t kHeadNumAttrIndex = 1;
constexpr size_t kUpdateKvCacheAttrIndex = 2;
constexpr float kDefaultRmsNormEps = 1.0e-6F;
constexpr int64_t kDefaultHeadNum = 64;
constexpr bool kDefaultUpdateKvCache = true;
constexpr uint32_t kHidden = 5120;
constexpr uint32_t kQa = 1280;
constexpr uint32_t kKv = 512;
constexpr uint32_t kHeadDim = 512;
constexpr uint32_t kMinHeads = 1;
constexpr uint32_t kMaxHeads = 64;
constexpr uint32_t kQbEvery = 80;
constexpr uint32_t kQbBlock = 16;
constexpr uint32_t kQbInner = 32;
constexpr uint32_t kCachePages = 22594;
constexpr uint32_t kCacheRows = 128;
constexpr uint32_t kMaxProjectionSplits = 3;
constexpr uint32_t kProjectionSplitsSmall = 3;
constexpr uint32_t kSplitProjectionMaxT = 128;
constexpr uint32_t kProjectionSplitsMid = 2;
constexpr uint32_t kSplitProjectionMidMaxT = 128;
constexpr uint32_t kNarrowProjectionMaxT = 192;
constexpr uint32_t kWorkspaceStages = 2;
constexpr uint32_t kMaxT = 4096;

inline uint32_t Align512(uint64_t value)
{
    return static_cast<uint32_t>((value + 511U) & ~uint64_t(511U));
}

struct MlaPrologCompileInfo {
    uint32_t aicNum = 0;
};

ge::graphStatus MlaPrologTilingPrepare(gert::TilingParseContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus MlaPrologTiling(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const gert::StorageShape *xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    if (xShape->GetStorageShape().GetDimNum() != 2 ||
        xShape->GetStorageShape().GetDim(1) != kHidden) {
        OP_LOGE(context->GetNodeName(), "x must have shape [T, 5120]");
        return ge::GRAPH_FAILED;
    }
    const int64_t t = xShape->GetStorageShape().GetDim(0);
    if (t < 1 || t > kMaxT) {
        OP_LOGE(context->GetNodeName(), "T must be in [1, 4096]");
        return ge::GRAPH_FAILED;
    }

    // rms_norm_eps, q_head_num and update_kv_cache are OpDef attributes;
    // optional callers get the framework-materialized defaults. Validate them
    // before any launch.
    float rmsNormEps = kDefaultRmsNormEps;
    int64_t headNumAttr = kDefaultHeadNum;
    bool updateKvCache = kDefaultUpdateKvCache;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float *eps = attrs->GetAttrPointer<float>(kRmsNormEpsAttrIndex);
        if (eps != nullptr) {
            rmsNormEps = *eps;
        }
        const int64_t *heads =
            attrs->GetAttrPointer<int64_t>(kHeadNumAttrIndex);
        if (heads != nullptr) {
            headNumAttr = *heads;
        }
        const bool *update =
            attrs->GetAttrPointer<bool>(kUpdateKvCacheAttrIndex);
        if (update != nullptr) {
            updateKvCache = *update;
        }
    }
    if (!std::isfinite(rmsNormEps) || !(rmsNormEps > 0.0F)) {
        OP_LOGE(context->GetNodeName(), "rms_norm_eps must be finite and > 0");
        return ge::GRAPH_FAILED;
    }
    if (headNumAttr < static_cast<int64_t>(kMinHeads) ||
        headNumAttr > static_cast<int64_t>(kMaxHeads)) {
        OP_LOGE(context->GetNodeName(), "q_head_num must be in [1, 64]");
        return ge::GRAPH_FAILED;
    }
    const uint32_t headNum = static_cast<uint32_t>(headNumAttr);
    const uint32_t qbOut = headNum * kHeadDim;

    // w_qb is INT8 FRACTAL_NZ [headNum*16, 80, 16, 32]; w_qb_scale is BF16
    // [headNum*512]. Mismatched local head counts must fail on the host.
    const gert::StorageShape *wqbShape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, wqbShape);
    const auto &wqbDims = wqbShape->GetStorageShape();
    if (wqbDims.GetDimNum() != 4 ||
        wqbDims.GetDim(0) != static_cast<int64_t>(headNum) * 16 ||
        wqbDims.GetDim(1) != static_cast<int64_t>(kQbEvery) ||
        wqbDims.GetDim(2) != static_cast<int64_t>(kQbBlock) ||
        wqbDims.GetDim(3) != static_cast<int64_t>(kQbInner)) {
        OP_LOGE(context->GetNodeName(),
                "w_qb must have shape [q_head_num*16, 80, 16, 32]");
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *wqbScaleShape = context->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, wqbScaleShape);
    const auto &wqbScaleDims = wqbScaleShape->GetStorageShape();
    if (wqbScaleDims.GetDimNum() != 1 ||
        wqbScaleDims.GetDim(0) != static_cast<int64_t>(qbOut)) {
        OP_LOGE(context->GetNodeName(),
                "w_qb_scale must have shape [q_head_num*512]");
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *cacheShape = context->GetInputShape(10);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheShape);
    if (cacheShape->GetStorageShape().GetDimNum() != 3 ||
        cacheShape->GetStorageShape().GetDim(0) != kCachePages ||
        cacheShape->GetStorageShape().GetDim(1) != kCacheRows ||
        cacheShape->GetStorageShape().GetDim(2) != kKv) {
        OP_LOGE(context->GetNodeName(), "kv_cache must have shape [22594, 128, 512]");
        return ge::GRAPH_FAILED;
    }
    // InferShape already derived q_output from the same attribute; re-check it
    // here when the framework exposes the output shape.
    const gert::StorageShape *qShape = context->GetOutputShape(0);
    if (qShape != nullptr) {
        const auto &qDims = qShape->GetStorageShape();
        if (qDims.GetDimNum() != 3 || qDims.GetDim(0) != t ||
            qDims.GetDim(1) != static_cast<int64_t>(headNum) ||
            qDims.GetDim(2) != static_cast<int64_t>(kHeadDim)) {
            OP_LOGE(context->GetNodeName(),
                    "q_output must have shape [T, q_head_num, 512]");
            return ge::GRAPH_FAILED;
        }
    }
    // The kv output is always declared [T, 512]. With update_kv_cache=true the
    // kernel leaves it untouched (no fused-path GM write); with false it is
    // the only KV write target.
    const gert::StorageShape *kvOutShape = context->GetOutputShape(2);
    if (kvOutShape != nullptr) {
        const auto &kvDims = kvOutShape->GetStorageShape();
        if (kvDims.GetDimNum() != 2 || kvDims.GetDim(0) != t ||
            kvDims.GetDim(1) != static_cast<int64_t>(kKv)) {
            OP_LOGE(context->GetNodeName(),
                    "kv must have shape [T, 512]");
            return ge::GRAPH_FAILED;
        }
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aicNum = platform.GetCoreNumAic();
    if (aicNum == 0) {
        OP_LOGE(context->GetNodeName(), "no AI Core is available");
        return ge::GRAPH_FAILED;
    }
    void *hardwareSyncAddr = nullptr;
    if (aclrtGetHardwareSyncAddr(&hardwareSyncAddr) != ACL_SUCCESS ||
        hardwareSyncAddr == nullptr) {
        OP_LOGE(context->GetNodeName(), "hardware sync address is unavailable");
        return ge::GRAPH_FAILED;
    }

    uint32_t projectionSplits = 1;
    uint32_t narrowM = 0;
    if (t <= kSplitProjectionMaxT) {
        projectionSplits = kProjectionSplitsSmall;
    } else if (t <= kNarrowProjectionMaxT) {
        narrowM = 1;
    } else if (t <= kSplitProjectionMidMaxT) {
        projectionSplits = kProjectionSplitsMid;
    }

    MlaPrologTilingData *data = context->GetTilingData<MlaPrologTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, data);
    *data = {};
    data->version = kVersion;
    data->t = static_cast<uint32_t>(t);
    data->aicNum = aicNum;
    data->projectionSplits = projectionSplits;
    data->narrowM = narrowM;
    data->headNum = headNum;
    data->updateKvCache = updateKvCache ? 1U : 0U;
    data->rmsNormEps = rmsNormEps;
    data->qaOffset = 0;
    data->qaBytes = Align512(static_cast<uint64_t>(kMaxProjectionSplits) * t * kQa * sizeof(float));
    data->kvOffset = data->qaOffset + data->qaBytes;
    data->kvBytes = Align512(static_cast<uint64_t>(kMaxProjectionSplits) * t * kKv * sizeof(float));
    data->kvNormOffset = data->kvOffset + data->kvBytes;
    data->kvNormBytes = Align512(static_cast<uint64_t>(t) * kKv * sizeof(uint16_t));
    data->qInt8Offset = data->kvNormOffset + data->kvNormBytes;
    data->qInt8Bytes = Align512(static_cast<uint64_t>(t) * kQa * sizeof(int8_t));
    data->qScaleOffset = data->qInt8Offset + data->qInt8Bytes;
    data->qScaleBytes = Align512(static_cast<uint64_t>(t) * sizeof(float));
    data->wScaleFpOffset = data->qScaleOffset + data->qScaleBytes;
    data->wScaleFpBytes = Align512(static_cast<uint64_t>(qbOut) * sizeof(float));
    data->qmmOffset = data->wScaleFpOffset + data->wScaleFpBytes;
    data->qmmBytes = Align512(static_cast<uint64_t>(aicNum) * kWorkspaceStages * 128U * 256U *
                              sizeof(int32_t));
    data->totalBytes = Align512(static_cast<uint64_t>(data->qmmOffset) + data->qmmBytes);
    data->hardwareSyncAddr = reinterpret_cast<uint64_t>(hardwareSyncAddr);

    if (context->SetBlockDim(aicNum) != ge::GRAPH_SUCCESS ||
        context->SetTilingKey(0) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() + data->totalBytes;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FiMlaProlog)
    .Tiling(MlaPrologTiling)
    .TilingParse<MlaPrologCompileInfo>(MlaPrologTilingPrepare);
} // namespace optiling
