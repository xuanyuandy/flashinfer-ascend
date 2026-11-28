/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indexer_prologue_qw_tiling.h"
#include "tiling/tiling_api.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "tiling/pad/broadcast_tiling.h"
#include "tiling/reduce/reduce_tiling.h"
#include "fi_host_checks.h"

using namespace ge;
namespace optiling {
namespace {
constexpr uint32_t BEST_BASE_M = 128;
constexpr uint32_t BEST_BASE_N = 256;
constexpr uint32_t BEST_BASE_K = 128;
constexpr uint32_t Q_ELEMS = 4096;
constexpr uint32_t K_ELEMS = 1280;
constexpr uint32_t W_ELEMS = 32;
constexpr uint32_t X_ELEMS = 5120;
constexpr uint64_t TILING_KEY = 0;
constexpr int64_t DQ_ROWS_PER_TILE = 4;

bool GenerateQTiling(IndexerPrologueQWFullFusionTilingData& data, bool dense)
{
    optiling::TCubeTiling tilingData;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        return false;
    }
    matmul_tiling::MultiCoreMatmulTiling tilingApi(*platform);
    tilingApi.SetDim(1);
    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8,
                       false);
    if (dense) {
        tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_INT8, true);
    } else {
        tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ,
                           matmul_tiling::DataType::DT_INT8, false);
    }
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    tilingApi.SetBias(false);
    tilingApi.SetOrgShape(BEST_BASE_M, data.get_n(), data.get_k());
    tilingApi.SetShape(BEST_BASE_M, data.get_n(), data.get_k());
    tilingApi.SetFixSplit(BEST_BASE_M, BEST_BASE_N, BEST_BASE_K);
    if (tilingApi.GetTiling(tilingData) == -1) {
        return false;
    }
    tilingData.set_dbL0C(1);
    tilingData.set_stepKa(4);
    tilingData.set_stepKb(4);
    tilingData.set_depthA1(8);
    tilingData.set_depthB1(8);
    tilingData.set_stepM(1);
    tilingData.set_stepN(1);
    tilingData.SaveToBuffer(data.get_mmTilingData(), tilingData.GetDataSize());
    return true;
}

bool GenerateWTiling(IndexerPrologueQWFullFusionTilingData& data, uint32_t wCoreNum)
{
    optiling::TCubeTiling tilingData;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        return false;
    }
    matmul_tiling::MultiCoreMatmulTiling tilingApi(*platform);
    tilingApi.SetDim(wCoreNum);
    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16,
                       false);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16,
                       true);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
    tilingApi.SetBias(false);
    tilingApi.SetOrgShape(data.get_totalInGroup(), W_ELEMS, X_ELEMS);
    tilingApi.SetShape(data.get_totalInGroup(), W_ELEMS, X_ELEMS);
    tilingApi.SetBufferSpace(-1, -1, -1);
    if (tilingApi.GetTiling(tilingData) == -1) {
        return false;
    }
    if (tilingData.get_usedCoreNum() <= 0 || tilingData.get_singleCoreM() <= 0 || tilingData.get_singleCoreN() <= 0) {
        return false;
    }
    tilingData.SaveToBuffer(data.get_wTilingData(), tilingData.GetDataSize());
    return true;
}

bool ShapeEquals(const gert::Shape& actual, const std::vector<int64_t>& expected)
{
    if (actual.GetDimNum() == expected.size()) {
        for (size_t index = 0; index < expected.size(); ++index) {
            if (actual.GetDim(index) != expected[index]) {
                return false;
            }
        }
        return true;
    }
    // The public executor may elide a leading singleton for T=1 while qb
    // retains its explicit [1,4096] storage shape. Accept only that exact
    // normalization; every other rank or dimension mismatch is rejected.
    if (!expected.empty() && expected[0] == 1 && actual.GetDimNum() + 1 == expected.size()) {
        for (size_t index = 0; index < actual.GetDimNum(); ++index) {
            if (actual.GetDim(index) != expected[index + 1]) {
                return false;
            }
        }
        return true;
    }
    return false;
}

std::string ShapeText(const gert::Shape& shape)
{
    std::string text = "[";
    for (size_t index = 0; index < shape.GetDimNum(); ++index) {
        if (index != 0) {
            text += ",";
        }
        text += std::to_string(shape.GetDim(index));
    }
    return text + "]";
}

std::string ExpectedText(const std::vector<int64_t>& dims)
{
    std::string text = "[";
    for (size_t index = 0; index < dims.size(); ++index) {
        if (index != 0) {
            text += ",";
        }
        text += std::to_string(dims[index]);
    }
    return text + "]";
}

bool IsFlatPublicStorageShape(const gert::Shape& actual, const std::vector<int64_t>& expected)
{
    if (actual.GetDimNum() != 1 || expected.empty()) {
        return false;
    }
    int64_t elements = 1;
    for (const auto dim : expected) {
        elements *= dim;
    }
    return actual.GetDim(0) == elements;
}

bool IsWeightStorageShape(const gert::Shape& actual, const std::vector<int64_t>& expected, size_t inputIndex,
                          bool dense)
{
    // The public executor flattens dense non-token weights, while caller-owned
    // Prequantized NZ weights expose their physical multi-dimensional NZ shape.
    if (inputIndex != 2 && inputIndex != 3 && inputIndex != 8) {
        return false;
    }
    if (expected.empty() || (dense && actual.GetDimNum() != 1)) {
        return false;
    }
    int64_t elements = 1;
    for (const auto dim : expected) {
        elements *= dim;
    }
    return actual.GetDimNum() > 0 && actual.GetShapeSize() == elements;
}

bool CheckQWTensorShapes(gert::TilingContext* context, int64_t tokens, bool dense)
{
    const int64_t weightRows = dense ? Q_ELEMS : K_ELEMS;
    const int64_t weightCols = dense ? K_ELEMS : Q_ELEMS;
    struct InputExpectation {
        size_t index;
        const char* name;
        std::vector<int64_t> dims;
    };
    const std::vector<InputExpectation> inputs = {
        {0, "qr", {tokens, K_ELEMS}},
        {1, "activation_q", {tokens, K_ELEMS}},
        {2, "weight_q", {weightRows, weightCols}},
        {3, "wqb", {weightRows, weightCols}},
        {4, "weight_scale", {Q_ELEMS}},
        {5, "activation_scale", {tokens}},
        {6, "qb", {tokens, Q_ELEMS}},
        {7, "x", {tokens, X_ELEMS}},
        {8, "ww", {W_ELEMS, X_ELEMS}},
        {9, "wb", {tokens, W_ELEMS}},
        {10, "rope_sin", {tokens, 64}},
        {11, "rope_cos", {tokens, 64}},
    };
    for (const auto& expected : inputs) {
        const auto* shape = context->GetInputShape(expected.index);
        const bool exactShape = shape != nullptr && ShapeEquals(shape->GetShape(), expected.dims) &&
                                ShapeEquals(shape->GetOriginShape(), expected.dims) &&
                                ShapeEquals(shape->GetStorageShape(), expected.dims);
        // Prequantized NZ weights must carry the exact logical [1280,4096]
        // orientation. Flat/numel-only acceptance belonged to the public
        // executor path and cannot distinguish a transposed NZ backing.
        const bool publicFlat = shape != nullptr && (dense || expected.index == 8) &&
                                (IsFlatPublicStorageShape(shape->GetShape(), expected.dims) ||
                                 IsFlatPublicStorageShape(shape->GetStorageShape(), expected.dims));
        const bool weightStorage = shape != nullptr && (dense || expected.index == 8) &&
                                   IsWeightStorageShape(shape->GetStorageShape(), expected.dims,
                                                        expected.index, dense);
        if (!exactShape && !publicFlat && !weightStorage) {
            OP_LOGE(context->GetNodeName(),
                    "%s shape is inconsistent with qb.T=%lld and the QW contract: logical=%s origin=%s storage=%s "
                    "expected=%s",
                    expected.name, static_cast<long long>(tokens),
                    shape == nullptr ? "null" : ShapeText(shape->GetShape()).c_str(),
                    shape == nullptr ? "null" : ShapeText(shape->GetOriginShape()).c_str(),
                    shape == nullptr ? "null" : ShapeText(shape->GetStorageShape()).c_str(),
                    ExpectedText(expected.dims).c_str());
            return false;
        }
    }
    const std::vector<std::vector<int64_t>> outputs = {{tokens, W_ELEMS, 128}, {tokens, W_ELEMS}, {tokens, W_ELEMS}};
    for (size_t index = 0; index < outputs.size(); ++index) {
        const auto* shape = context->GetOutputShape(index);
        if (shape == nullptr || (!ShapeEquals(shape->GetStorageShape(), outputs[index]) &&
                                 !IsFlatPublicStorageShape(shape->GetStorageShape(), outputs[index]))) {
            OP_LOGE(context->GetNodeName(), "output %zu shape is inconsistent with qb.T=%lld", index,
                    static_cast<long long>(tokens));
            return false;
        }
    }
    return true;
}

ge::graphStatus TilingForQW(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const auto* qbShape = context->GetInputShape(6);
    OP_CHECK_NULL_WITH_CONTEXT(context, qbShape);
    const auto& qb = qbShape->GetStorageShape();
    OP_CHECK_IF(qb.GetDimNum() != 2 || qb.GetDim(1) != Q_ELEMS || qb.GetDim(0) < 1,
                OP_LOGE(context->GetNodeName(), "qb must be [T,4096] with T>=1"), return ge::GRAPH_FAILED);
    const uint32_t tokens = static_cast<uint32_t>(qb.GetDim(0));
    const auto* wqbDesc = context->GetInputDesc(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, wqbDesc);
    const bool dense = wqbDesc->GetDataType() == ge::DT_BF16;
    if (!dense) {
        const auto* weightQDesc = context->GetInputDesc(2);
        OP_CHECK_NULL_WITH_CONTEXT(context, weightQDesc);
        OP_CHECK_IF(weightQDesc->GetStorageFormat() != ge::FORMAT_FRACTAL_NZ ||
                        wqbDesc->GetStorageFormat() != ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(context->GetNodeName(), "Prequantized weight_q/wqb must use physical FRACTAL_NZ"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(!CheckQWTensorShapes(context, tokens, dense),
                OP_LOGE(context->GetNodeName(), "QW input/output shape cross-validation failed"),
                return ge::GRAPH_FAILED);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* wScale = attrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, wScale);

    uint64_t coreNum = 0;
    NpuArch npuArch = NpuArch::DAV_RESV;
    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC platform(platformInfo);
        coreNum = platform.GetCoreNumAic();
        npuArch = platform.GetCurNpuArch();
    } else {
        const auto* compileInfo = context->GetCompileInfo<IndexerPrologueQWFullFusionCompileInfo>();
        OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
        coreNum = compileInfo->coreNum;
        npuArch = compileInfo->npuArch;
    }
    OP_CHECK_IF(npuArch != NpuArch::DAV_2201 || coreNum == 0,
                OP_LOGE(context->GetNodeName(), "QW full fusion requires arch22 and a positive AIC count"),
                return ge::GRAPH_FAILED);

    IndexerPrologueQWFullFusionTilingData data;
    data.set_groupNum(1);
    data.set_totalInGroup(tokens);
    data.set_k(K_ELEMS);
    data.set_n(Q_ELEMS);
    data.set_ubCalSize(24 * BEST_BASE_N);
    data.set_ubRestBytes(118784);
    data.set_parallNum(4);
    data.set_wScale(*wScale);
    const uint32_t taskBlocks =
        ((tokens + BEST_BASE_M - 1) / BEST_BASE_M) * ((Q_ELEMS + BEST_BASE_N - 1) / BEST_BASE_N);
    const uint32_t taskCoreNum = std::min<uint32_t>(taskBlocks, static_cast<uint32_t>(coreNum));
    // P0 dense-only: extra launch cores serve DQ/post on small dense shapes.
    // The 16 Q tiles retain their core-0..15 mapping; W keeps its original width.
    const uint32_t blockDim =
        dense && tokens <= BEST_BASE_M ? static_cast<uint32_t>(coreNum) : taskCoreNum;
    data.set_coreNum(blockDim);

    uint32_t rmMax = 0;
    uint32_t rmMin = 0;
    uint32_t bcMax = 0;
    uint32_t bcMin = 0;
    const ge::Shape reduceShape(std::vector<int64_t>{DQ_ROWS_PER_TILE, K_ELEMS});
    AscendC::GetReduceMaxMaxMinTmpSize(reduceShape, ge::DT_FLOAT, AscendC::ReducePattern::AR, true, false, rmMax,
                                       rmMin);
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    OP_CHECK_NULL_WITH_CONTEXT(context, platform);
    const ge::Shape bcastSrc(std::vector<int64_t>{1, 1});
    const ge::Shape bcastDst(std::vector<int64_t>{1, K_ELEMS});
    AscendC::GetBroadCastMaxMinTmpSize(*platform, bcastSrc, bcastDst, sizeof(float), false, bcMax, bcMin);
    const uint32_t dqWorkBytes = 24 * BEST_BASE_N * sizeof(float);
    OP_CHECK_IF(rmMin == 0 || rmMin > dqWorkBytes || rmMin + bcMin > 2 * dqWorkBytes,
                OP_LOGE(context->GetNodeName(), "invalid DQ temporary sizes %u/%u", rmMin, bcMin),
                return ge::GRAPH_FAILED);
    data.set_dqReduceTmpBytes(rmMin);
    data.set_dqBroadcastTmpBytes(bcMin);
    data.set_denseLayout(dense ? 1 : 0);
    OP_CHECK_IF(!GenerateQTiling(data, dense), OP_LOGE(context->GetNodeName(), "failed to generate Q cube tiling"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!GenerateWTiling(data, taskCoreNum), OP_LOGE(context->GetNodeName(), "failed to generate W cube tiling"),
                return ge::GRAPH_FAILED);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(TILING_KEY);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    const uint64_t userWorkspace =
        static_cast<uint64_t>(data.get_parallNum()) * BEST_BASE_N * BEST_BASE_M * sizeof(int32_t) * blockDim;
    workspaces[0] = userWorkspace + platform->GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForQWFullFusion(gert::TilingParseContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto* compileInfo = context->GetCompiledInfo<IndexerPrologueQWFullFusionCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC platform(platformInfo);
        compileInfo->coreNum = platform.GetCoreNumAic();
        compileInfo->npuArch = platform.GetCurNpuArch();
    } else {
        // GenerateQTiling/GenerateWTiling require concrete platform resources.
        // A null offline context has no authoritative AIC count or NPU arch;
        // reject it explicitly here instead of reaching the same failure later
        // with an uninitialized CompileInfo.
        OP_LOGE(context->GetNodeName(), "QW full fusion does not support null-platform offline tiling; provide "
                                        "platformInfo with arch22 and a positive AIC count");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(compileInfo->npuArch != NpuArch::DAV_2201 || compileInfo->coreNum == 0,
                OP_LOGE(context->GetNodeName(), "QW tiling parse requires arch22 and a positive AIC count"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus TilingPrepareForIndexerPrologueQWFullFusion(gert::TilingParseContext* context)
{
    return TilingPrepareForQWFullFusion(context);
}

ge::graphStatus TilingForIndexerPrologueQWFullFusion(gert::TilingContext* context)
{
    return TilingForQW(context);
}

} // namespace optiling

FI_IMPL_TILING(FI_PACKAGE_OP_TYPE)
    .Tiling(optiling::TilingForIndexerPrologueQWFullFusion)
    .TilingParse<optiling::IndexerPrologueQWFullFusionCompileInfo>(
        optiling::TilingPrepareForIndexerPrologueQWFullFusion);
