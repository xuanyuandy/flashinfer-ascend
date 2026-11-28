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
 * \file quant_flash_attn_common_def.h
 * \brief QuantFlashAttn算子tiling和kernel共用的相关定义
 */

#ifndef QUANT_FLASH_ATTN_COMMON_DEF_H_
#define QUANT_FLASH_ATTN_COMMON_DEF_H_

#include <cstdint>

static constexpr uint32_t QFA_HEAD_NEED_INIT_OUTPUT_INDEX = 15U;

#define ASCENDC_TPL_5_BW 5
#define ASCENDC_TPL_10_BW 10

enum class inferFaLayOutTypeEnum {
    LAYOUT_TND = 4,
};

enum class inferDTemplateType {
    Aligned128 = 128,
    Aligned256 = 256,
    NotAligned = 577,
};

enum class inferS1TemplateType {
    Aligned128 = 128,
};

enum class inferS2TemplateType {
    Aligned128 = 128,
    Aligned256 = 256,
};

#define InOutLayoutType_TND 2
#define InOutLayoutType_TND_TND InOutLayoutType_TND

// The layout key indexes this table directly. Only TND_TND is selected.
static constexpr inferFaLayOutTypeEnum InOutLayoutTypeValue[InOutLayoutType_TND_TND + 1][2] = {
    {},
    {},
    {inferFaLayOutTypeEnum::LAYOUT_TND, inferFaLayOutTypeEnum::LAYOUT_TND},
};

#define KvLayoutType_PA_BBND 1
#define KvLayoutType_PA_BNBD 2
#define KvLayoutType_PA_NZ 3

struct ConfigParams {
    inferS1TemplateType s1;
    inferS2TemplateType s2;
    inferDTemplateType d;
    inferDTemplateType dv;
};

#define Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128 2
#define Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256 5

// Keep the active configurations at their tiling key indices 2 and 5.
static constexpr ConfigParams ConfigValue[] = {
    {},
    {},
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128},
    {},
    {},
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256},
};

#define QFA_MXFP4_FP32_PREFILL 3
#define QFA_MXFP4_FP32_DECODE 4

#define false 0
#define true 1

struct FDparamsX {
    uint32_t fdCoreEnable;
    uint32_t fdBN2Idx;
    uint32_t fdMIdx;
    uint32_t fdS2SplitNum;
    uint32_t mStart;
    uint32_t mLen;
    uint32_t fdWorkspaceIdx;
};

struct StridesConstInfo {
    uint64_t bnStride = 0;
    uint64_t n2Stride = 0;
};

struct CommonConstInfo {
    uint32_t bSize; // batch 数量
    uint64_t t1Size; // Q 的 T 轴尺寸，供 TND、NTD 等布局寻址使用
    uint32_t dSizeV; // V 的逻辑 head dimension
    uint32_t gSize; // 原始 GQA 分组大小，通常为 Q head 数除以 KV head 数
    uint32_t n2Size; // 原始 KV head 数
    uint32_t realGSize; // 实际用于调度及行轴合并的 G，不合轴相当是1，不等于gSize
    uint32_t realN2Size; // 实际用于调度的 head 数，不合轴相当于是q的head数量，合轴就是kv head的数量
    uint64_t s1Size; // tiling 提供的 Q/KV 序列尺寸，供默认长度等逻辑使用；不是每个 batch 的实际有效长度。
    uint64_t s2Size;
    uint64_t cuSeqLensQSize; // Q 累计长度输入的规模信息，就是数组长度
    uint64_t cuSeqLensKVSize; // KV 累计长度输入的对应规模信息
    uint64_t seqUsedQSize;
    uint64_t seqUsedKvSize;

    uint32_t sparseMode; // attention 的稀疏／mask 模式
    uint32_t attenMaskS2Size; // mask 的 KV 方向尺寸
    int64_t preTokens; // 原始左／右窗口配置
    int64_t nextTokens;
    float scaleValue; // attention score 的缩放系数，例如 1 / sqrt(D)

    uint32_t aicIdx; // 当前 AIC 编号；在 AIV 上表示其对应的 AIC 编号
    uint32_t aivIdx; // 当前 AIV 编号
    uint8_t subBlockIdx; // 当前 AIV 在所属 AIC 配对中的子核编号，0或1
    uint32_t coreNum;  // FA 路径参与计算的 AIC 数量

    uint32_t accumOutSize; // FD 部分输出区域的容量
    uint32_t logSumExpSize; // FD 统计量区域的容量

    bool needInitOutput; // 是否需要初始化输出的标记
    bool enableFlashDecode; // 运行时是否启用 FD 的标记

    // 四个平面的 stride
    StridesConstInfo keyStrides;
    StridesConstInfo valueStrides;
    StridesConstInfo kDescaleStrides;
    StridesConstInfo vDescaleStrides;
};

struct PAConstInfo {
    uint32_t blockSize; // page的大小
    uint32_t maxBlockNumPerBatch; // block table 每个 batch 的条目数，用于定位对应行
};

struct LseConstInfo {
    bool isSoftmaxLseEnable; // 是否输出 Softmax LSE
};

struct ConstInfo_t : CommonConstInfo, PAConstInfo, LseConstInfo {};

// Buffer definitions are provided by the device-only buffer headers.
namespace fa_base_matmul {
enum class BufferType;
enum class SyncType;
enum class SyncMode;

template <BufferType bufferType, SyncType syncType, SyncMode syncMode>
class Buffer;
template <BufferType bufferType>
class BufferManager;
} // namespace fa_base_matmul

namespace BaseApi {
constexpr float LN2 = 0.6931471806f;
constexpr float INV_LN2 = 1.4426950409f;

// Values match the public quant_schema attribute. Q/K/V encoding is unchanged.
enum class QuantSchema : uint8_t {
    OCP_QUANT = 0,
    CX_QUANT = 1,
    CX_DELTA_QUANT = 2,
};

using fa_base_matmul::Buffer;
using fa_base_matmul::BufferManager;
using fa_base_matmul::BufferType;
using fa_base_matmul::SyncType;
// Only the row coordinates and task boundaries are shared between Cube and Vector.
struct Mxfp4TaskInfo {
    uint32_t loop;
    uint32_t bIdx;
    uint32_t realN2Idx;
    uint32_t gS1Idx;
    uint32_t actMSize;
    uint32_t actMSizeAlign32;
    uint32_t s2Idx;
    uint32_t actSingleLoopS2Size;
    bool isLastS2Loop;
    bool isGroupLast;
};

struct Mxfp4CubeTask : Mxfp4TaskInfo {
    static constexpr uint32_t KV_PAGE_CAPACITY = 8U;
    uint32_t n2Idx;
    uint32_t kvPageIdx[KV_PAGE_CAPACITY]; // CacheKvPageIndices fills the entries consumed by this task.
};

struct Mxfp4VecTask : Mxfp4TaskInfo {
    int64_t causalOffset;
    bool isFirstS2Loop;
    bool isS2SplitCore;
    uint32_t faTmpOutWsPos;
    uint64_t v1GroupId; // Written by V1 before the delayed P-scale update and V2.
};

struct Mxfp4SubRunInfo {
    uint64_t sequence = 0;     // 连续的 QK 子块编号
    uint32_t subLoop = 0;      // 当前是 task 内的第几个子块
    uint32_t scoreSlot = 0;    // 使用哪个 score 缓冲槽
    uint32_t s2Offset = 0;     // 子块在完整 task 内的 KV 方向偏移
    uint32_t validN = 0;       // 子块实际有效的 KV token 数
    uint32_t alignedN = 0;     // 对齐后的 N 长度
    bool isFirst = false;     // 是否为 task 的首个／最后一个子块
    bool isLast = false;
};

} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_COMMON_DEF_H_
