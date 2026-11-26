/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../quant_flash_attn_host_contract.h"
#include "../../op_kernel/arch35/quant_flash_attn_template_tiling_key.h"

#undef true
#undef false

namespace quant_flash_attn_host {
namespace {
constexpr int64_t kCausal = 3;
constexpr uint32_t kScheduleBatchMode = 1;
constexpr int64_t kCausalTemplateSize = 2048;
constexpr uint64_t kMixedQueryTile = 128;
constexpr uint64_t kLseAlign = 8;
constexpr int64_t kMaxBatchSize = 65535;
constexpr int64_t kMaxHeadCount = 256;
constexpr int64_t kMaxGqaGroupSize = 64;

// Values stored in QuantFlashAttnPageAttentionParams::paLayoutType.
enum class PagedKvLayout : uint8_t { BNBD = 0, BBND = 1, NZ = 2 };

struct TilingAttributes {
    float softmaxScale;
    int64_t maskMode;
    int64_t maxSeqQ;
    bool returnLse;
    uint8_t quantSchema;
    bool decode;
    PagedKvLayout kvLayout;
};

struct QkvShapes {
    const gert::Shape* key;
    const gert::Shape* value;
    const gert::Shape* kDescale;
    const gert::Shape* vDescale;
    int64_t tokens;
    int64_t headDim;
    int64_t pageSize;
    int64_t kvHeads;
    int64_t groups;
};

struct ScheduleInputs {
    int64_t batch;
    int64_t pagesPerBatch;
    bool hasSequsedQ;
};

bool ValidDimension(int64_t value)
{
    return value > 0 && value <= std::numeric_limits<int32_t>::max();
}

bool ValidHeadDim(int64_t value)
{
    return value == 128 || value == 256;
}

bool ValidPageSize(int64_t size)
{
    return size >= 64 && size <= 1024 && (size & (size - 1)) == 0;
}

bool ShapeIs(const gert::Shape* shape, std::initializer_list<int64_t> dimensions)
{
    if (shape == nullptr || shape->GetDimNum() != dimensions.size()) {
        return false;
    }
    size_t axis = 0;
    for (const int64_t dimension : dimensions) {
        if (shape->GetDim(axis++) != dimension) {
            return false;
        }
    }
    return true;
}

const gert::Shape* InputShape(gert::TilingContext* context, uint32_t index)
{
    const auto* shape = context->GetInputShape(index);
    return shape == nullptr ? nullptr : &shape->GetStorageShape();
}

const gert::Shape* OptionalShape(gert::TilingContext* context, uint32_t index)
{
    const auto* tensor = context->GetOptionalInputTensor(index);
    return tensor == nullptr ? nullptr : &tensor->GetStorageShape();
}

bool ReadPageStrides(gert::TilingContext* context, Input index, const gert::Shape& shape, bool headMajor,
                     StridesParams& result)
{
    if (!context->InputIsView(index)) {
        return true;
    }
    const auto* strides = context->GetInputStride(index);
    const auto rank = shape.GetDimNum();
    if (strides == nullptr || strides->GetDimNum() != rank) {
        return false;
    }
    // NB/NZ views can have independent page and KV-head strides. BBND heads remain interleaved.
    const size_t firstContiguousAxis = headMajor ? 2 : 1;
    int64_t innerElements = 1;
    for (size_t axis = rank; axis-- > firstContiguousAxis;) {
        const auto stride = strides->GetStride(axis);
        if (stride <= 0 || (shape.GetDim(axis) != 1 && stride != innerElements)) {
            return false;
        }
        innerElements *= shape.GetDim(axis);
    }
    const auto pageStride = strides->GetStride(0);
    const auto headStride = strides->GetStride(1);
    // C4 reads payload bytes and E8M0 pairs by dividing the logical outer strides by two.
    if (pageStride < innerElements || pageStride % 2 != 0 ||
        (headMajor && (headStride <= 0 || headStride % 2 != 0 ||
                       (shape.GetDim(1) > 1 && headStride < innerElements)))) {
        return false;
    }
    result.bnStride = static_cast<uint64_t>(pageStride);
    result.n2Stride = static_cast<uint64_t>(headStride);
    return true;
}

ge::graphStatus TilingError(const char* reason)
{
    std::fprintf(stderr, "FiQuantFlashAttn tiling: %s\n", reason);
    return ge::GRAPH_FAILED;
}

ge::graphStatus ParseAttributes(gert::TilingContext* context, TilingAttributes& result)
{
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return TilingError("missing attributes");
    }
    const auto* quantMode = attrs->GetAttrPointer<int64_t>(QUANT_MODE);
    const auto* scale = attrs->GetAttrPointer<float>(SOFTMAX_SCALE);
    const auto* maskMode = attrs->GetAttrPointer<int64_t>(MASK_MODE);
    const auto* winLeft = attrs->GetAttrPointer<int64_t>(WIN_LEFT);
    const auto* winRight = attrs->GetAttrPointer<int64_t>(WIN_RIGHT);
    const auto* maxQ = attrs->GetAttrPointer<int64_t>(MAX_SEQLEN_Q);
    const auto* maxKv = attrs->GetAttrPointer<int64_t>(MAX_SEQLEN_KV);
    const auto* qLayout = attrs->GetAttrPointer<char>(LAYOUT_Q);
    const auto* qScaleLayout = attrs->GetAttrPointer<char>(LAYOUT_Q_DESCALE);
    const auto* kvLayout = attrs->GetAttrPointer<char>(LAYOUT_KV);
    const auto* outLayout = attrs->GetAttrPointer<char>(LAYOUT_OUT);
    const auto* returnLse = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE);
    const auto* quantSchema = attrs->GetAttrPointer<int64_t>(QUANT_SCHEMA);

    if (quantMode == nullptr || *quantMode != kC4A4 ||
        scale == nullptr || !std::isfinite(*scale) || *scale <= 0 ||
        maskMode == nullptr || (*maskMode != kNoMask && *maskMode != kCausal) ||
        winLeft == nullptr || *winLeft != -1 || winRight == nullptr || *winRight != -1 ||
        maxQ == nullptr || (*maxQ != -1 && !ValidDimension(*maxQ)) ||
        maxKv == nullptr || (*maxKv != -1 && !ValidDimension(*maxKv)) ||
        returnLse == nullptr || quantSchema == nullptr || *quantSchema < 0 || *quantSchema > 2) {
        return TilingError("expected C4A4 mode 5, schema 0..2, positive scale and mask mode 0 or 3");
    }
    if (qLayout == nullptr || std::strcmp(qLayout, "TND") != 0 ||
        outLayout == nullptr || std::strcmp(outLayout, "TND") != 0 ||
        kvLayout == nullptr || qScaleLayout == nullptr) {
        return TilingError("expected TND query/output and a paged KV layout");
    }
    const bool decodeLayout = std::strcmp(qScaleLayout, "N2TGD") == 0;
    if (!decodeLayout && std::strcmp(qScaleLayout, "TND") != 0) {
        return TilingError("expected TND or N2TGD query descale");
    }

    const bool nz = std::strcmp(kvLayout, "PA_NZ") == 0;
    const bool bnbd = std::strcmp(kvLayout, "PA_BNBD") == 0;
    if (!nz && !bnbd && std::strcmp(kvLayout, "PA_BBND") != 0) {
        return TilingError("expected PA_BBND, PA_BNBD or PA_NZ KV");
    }
    const auto layout = nz ? PagedKvLayout::NZ : bnbd ? PagedKvLayout::BNBD : PagedKvLayout::BBND;
    result = {*scale, *maskMode, *maxQ, *returnLse, static_cast<uint8_t>(*quantSchema), decodeLayout, layout};
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseQkvShapes(gert::TilingContext* context, const TilingAttributes& attrs, QkvShapes& result)
{
    const bool nz = attrs.kvLayout == PagedKvLayout::NZ;
    const bool bnbd = attrs.kvLayout == PagedKvLayout::BNBD;
    const bool headMajor = nz || bnbd;
    const auto* q = InputShape(context, Q);
    const auto* k = InputShape(context, K);
    const auto* v = InputShape(context, V);
    if (q == nullptr || q->GetDimNum() != 3 || k == nullptr || k->GetDimNum() != (nz ? 5 : 4)) {
        return TilingError("invalid Q or K rank");
    }
    // aclTensor and CANN tiling expose the logical FP4 dimension.
    const auto tokens = q->GetDim(0);
    const auto qHeads = q->GetDim(1);
    const auto d = q->GetDim(2);
    const auto pages = k->GetDim(0);
    const auto pageSize = k->GetDim(nz ? 3 : bnbd ? 2 : 1);
    const auto kvHeads = k->GetDim(headMajor ? 1 : 2);
    if (!ValidDimension(tokens) || !ValidDimension(qHeads) || !ValidHeadDim(d) ||
        !ValidDimension(pages) || !ValidPageSize(pageSize) || !ValidDimension(kvHeads) || qHeads % kvHeads != 0) {
        return TilingError("invalid Q/K/V head dimensions, grouping or page size");
    }
    const auto groups = qHeads / kvHeads;
    if (qHeads > kMaxHeadCount || kvHeads > kMaxHeadCount || groups > kMaxGqaGroupSize) {
        return TilingError("expected at most 256 Q and KV heads and 64 Q heads per KV head");
    }
    const auto* qScale = InputShape(context, Q_DESCALE);
    const auto* kScale = InputShape(context, K_DESCALE);
    const auto* vScale = InputShape(context, V_DESCALE);
    const bool validQScale = attrs.decode ? ShapeIs(qScale, {kvHeads, tokens, groups, d / 64, 2})
                                         : ShapeIs(qScale, {tokens, qHeads, d / 64, 2});
    const bool validKv = nz
        ? ShapeIs(k, {pages, kvHeads, d / 64, pageSize, 64}) && ShapeIs(v, {pages, kvHeads, d / 64, pageSize, 64})
        : bnbd ? ShapeIs(k, {pages, kvHeads, pageSize, d}) && ShapeIs(v, {pages, kvHeads, pageSize, d})
               : ShapeIs(k, {pages, pageSize, kvHeads, d}) && ShapeIs(v, {pages, pageSize, kvHeads, d});
    const bool validKvScale = nz
        ? ShapeIs(kScale, {pages, kvHeads, pageSize / 16, d / 64, 16, 2}) &&
          ShapeIs(vScale, {pages, kvHeads, d / 16, pageSize / 64, 16, 2})
        : bnbd ? ShapeIs(kScale, {pages, kvHeads, pageSize, d / 64, 2}) &&
                 ShapeIs(vScale, {pages, kvHeads, pageSize / 64, d, 2})
               : ShapeIs(kScale, {pages, pageSize, kvHeads, d / 64, 2}) &&
                 ShapeIs(vScale, {pages, pageSize / 64, kvHeads, d, 2});
    if (!validKv || !validQScale || !validKvScale) {
        return TilingError("invalid FP4 payload or E8M0 descale shape for the layout");
    }
    result = {k, v, kScale, vScale, tokens, d, pageSize, kvHeads, groups};
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseScheduleInputs(gert::TilingContext* context, int64_t pageSize, int64_t maskMode,
                                   ScheduleInputs& result)
{
    const auto* table = InputShape(context, BLOCK_TABLE);
    const auto* metadata = InputShape(context, METADATA);
    if (table == nullptr || table->GetDimNum() != 2 || !ValidDimension(table->GetDim(0)) ||
        !ValidDimension(table->GetDim(1)) ||
        !ValidDimension(pageSize * table->GetDim(1)) || metadata == nullptr) {
        return TilingError("invalid page table or missing metadata");
    }
    const auto batch = table->GetDim(0);
    if (batch > kMaxBatchSize) {
        return TilingError("batch size must be less than 65536");
    }
    const auto* usedQ = OptionalShape(context, SEQUSED_Q);
    const auto* mask = OptionalShape(context, ATTN_MASK);
    if (!ShapeIs(InputShape(context, CU_SEQLENS_Q), {batch + 1}) ||
        !ShapeIs(InputShape(context, SEQUSED_KV), {batch}) ||
        (usedQ != nullptr && !ShapeIs(usedQ, {batch}))) {
        return TilingError("expected cu_seqlens_q[B+1] and effective lengths of shape [B]");
    }
    if ((maskMode == kCausal && !ShapeIs(mask, {kCausalTemplateSize, kCausalTemplateSize})) ||
        (maskMode == kNoMask && mask != nullptr)) {
        return TilingError("causal mode requires the 2048 x 2048 mask template");
    }
    if ((metadata->GetDimNum() != 1 && metadata->GetDimNum() != 2) ||
        (metadata->GetDimNum() == 2 && metadata->GetDim(0) != 2) ||
        metadata->GetDim(metadata->GetDimNum() - 1) < METADATA_STRIDE) {
        return TilingError("metadata must be [C] or [2,C] with room for the FA/FD header");
    }
    for (const auto input : {P_SCALE, CU_SEQLENS_KV, SINKS, V_TAIL, BLOCK_TABLE_TAIL, SEQUSED_V_TAIL}) {
        if (OptionalShape(context, input) != nullptr) {
            return TilingError("unsupported optional input for C4A4");
        }
    }
    result = {batch, table->GetDim(1), usedQ != nullptr};
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcNumBlocks(platform_ascendc::PlatformAscendC& platform, uint32_t& aiv, uint32_t& blockDim)
{
    const auto aic = platform.GetCoreNumAic();
    aiv = platform.GetCoreNumAiv();
    if (aic == 0 || aiv / aic != 2 || aiv % aic != 0) {
        return TilingError("expected one AIC per two AIVs");
    }
    blockDim = platform.CalcTschBlockDim(aiv, aic, aiv);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcWorkspaceSize(uint32_t aiv, uint32_t blockDim, int64_t headDim,
                                  QuantFlashAttnWorkspaceParams& params, uint64_t& scratchBytes)
{
    // Match the source host tiler: compile and provision the FD-capable variant,
    // then let the metadata header enable or bypass its FD reduction at run time.
    const uint64_t accumOut = uint64_t{blockDim} * 2 * kMixedQueryTile * headDim;
    const uint64_t logSumExp = uint64_t{blockDim} * 2 * kMixedQueryTile * kLseAlign;
    if (blockDim == 0 || accumOut > UINT32_MAX || logSumExp > UINT32_MAX) {
        return TilingError("invalid block dimension or FP32 partial workspace");
    }
    params = {static_cast<uint32_t>(accumOut), static_cast<uint32_t>(logSumExp)};
    // Two paged matmul descriptors, double-buffered per AIV.
    scratchBytes = uint64_t{aiv} * 2 * 2 * 64 + (accumOut + 2 * logSumExp) * sizeof(float);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillTilingData(gert::TilingContext* context, const TilingAttributes& attrs,
                               const QkvShapes& qkv, const ScheduleInputs& inputs,
                               uint32_t blockDim, QuantFlashAttnTilingData& data)
{
    auto& base = data.baseTiling.quantFlashAttnBaseParams;
    const bool headMajor = attrs.kvLayout != PagedKvLayout::BBND;
    if (!ReadPageStrides(context, K, *qkv.key, headMajor, base.keyStrides) ||
        !ReadPageStrides(context, V, *qkv.value, headMajor, base.valueStrides) ||
        !ReadPageStrides(context, K_DESCALE, *qkv.kDescale, headMajor, base.kDescaleStrides) ||
        !ReadPageStrides(context, V_DESCALE, *qkv.vDescale, headMajor, base.vDescaleStrides)) {
        return TilingError("expected contiguous page interiors, with paired strides on the page and head axes");
    }
    base.bSize = static_cast<uint32_t>(inputs.batch);
    base.t1Size = static_cast<uint32_t>(qkv.tokens);
    base.n2Size = static_cast<uint32_t>(qkv.kvHeads);
    base.gSize = static_cast<uint32_t>(qkv.groups);
    // Preserve the source tiler's unsigned -1 sentinel for an unspecified max Q length.
    base.s1Size = static_cast<uint32_t>(attrs.maxSeqQ);
    base.s2Size = static_cast<uint32_t>(qkv.pageSize * inputs.pagesPerBatch);
    base.dSize = base.dSizeV = static_cast<uint32_t>(qkv.headDim);
    base.cuSeqLensQSize = base.bSize;
    base.seqUsedQSize = inputs.hasSequsedQ ? base.bSize : 0;
    base.seqUsedKvSize = base.bSize;
    base.scaleValue = attrs.softmaxScale;
    base.iscuSeqLengthsKVNull = 1;
    base.isSoftMaxLseEnable = attrs.returnLse;
    base.quantSchema = attrs.quantSchema;
    base.coreNum = blockDim;
    base.outputLayout = InOutLayoutType_TND;

    auto& maskParams = data.baseTiling.quantFlashAttnAttenMaskParams;
    maskParams.sparseMode = static_cast<uint8_t>(attrs.maskMode);
    maskParams.winLefts = maskParams.winRights = INT32_MAX;
    if (attrs.maskMode == kCausal) {
        maskParams.attenMaskS1Size = maskParams.attenMaskS2Size = kCausalTemplateSize;
    }
    auto& page = data.baseTiling.quantFlashAttnPageAttentionParams;
    page.paLayoutType = static_cast<uint8_t>(attrs.kvLayout);
    page.blockSize = static_cast<uint32_t>(qkv.pageSize);
    page.maxBlockNumPerBatch = static_cast<uint32_t>(inputs.pagesPerBatch);
    return ge::GRAPH_SUCCESS;
}

uint64_t GenTilingKey(const TilingAttributes& attrs, int64_t headDim)
{
    const uint64_t config = headDim == 256 ? Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256
                                          : Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128;
    const uint64_t mode = attrs.decode ? QFA_MXFP4_FP32_DECODE : QFA_MXFP4_FP32_PREFILL;
    const uint64_t layout = attrs.kvLayout == PagedKvLayout::NZ ? KvLayoutType_PA_NZ
                            : attrs.kvLayout == PagedKvLayout::BNBD ? KvLayoutType_PA_BNBD
                                                                  : KvLayoutType_PA_BBND;
    return GET_TPL_TILING_KEY(InOutLayoutType_TND_TND, config, mode, attrs.maskMode == kCausal, layout, true, false);
}
} // namespace

ge::graphStatus FiQuantFlashAttnTiling(gert::TilingContext* context)
{
    TilingAttributes attrs{};
    QkvShapes qkv{};
    ScheduleInputs inputs{};
    if (ParseAttributes(context, attrs) != ge::GRAPH_SUCCESS ||
        ParseQkvShapes(context, attrs, qkv) != ge::GRAPH_SUCCESS ||
        ParseScheduleInputs(context, qkv.pageSize, attrs.maskMode, inputs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto* platformInfo = context->GetPlatformInfo();
    auto* tiling = context->GetTilingData<QuantFlashAttnTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (platformInfo == nullptr || tiling == nullptr || workspace == nullptr) {
        return TilingError("missing platform, tiling buffer or workspace slot");
    }

    platform_ascendc::PlatformAscendC platform(platformInfo);
    uint32_t aiv = 0;
    uint32_t blockDim = 0;
    uint64_t scratchBytes = 0;
    QuantFlashAttnTilingData data{};
    if (CalcNumBlocks(platform, aiv, blockDim) != ge::GRAPH_SUCCESS ||
        CalcWorkspaceSize(aiv, blockDim, qkv.headDim, data.baseTiling.quantFlashAttnWorkspaceParams,
                          scratchBytes) != ge::GRAPH_SUCCESS ||
        FillTilingData(context, attrs, qkv, inputs, blockDim, data) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (context->SetScheduleMode(kScheduleBatchMode) != ge::GRAPH_SUCCESS) {
        return TilingError("failed to select batch scheduling");
    }
    *tiling = data;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(GenTilingKey(attrs, qkv.headDim));
    workspace[0] = platform.GetLibApiWorkSpaceSize() + scratchBytes;
    return ge::GRAPH_SUCCESS;
}
} // namespace quant_flash_attn_host

IMPL_OP_OPTILING(FiQuantFlashAttn).Tiling(quant_flash_attn_host::FiQuantFlashAttnTiling);
