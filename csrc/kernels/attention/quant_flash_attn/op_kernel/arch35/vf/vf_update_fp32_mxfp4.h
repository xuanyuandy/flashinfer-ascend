/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QFA_VF_UPDATE_FP32_MXFP4_H_
#define QFA_VF_UPDATE_FP32_MXFP4_H_
#include "vf_softmax_fp32_mxfp4.h"
#include "../quant_flash_attn_common_def.h"

namespace Mxfp4Api {
using BaseApi::QuantSchema;
// DN：输入每个 task 为 [TaskN/32,Q64] FP32 指数，输出为 [4,TaskN/64+1,16,2] 字节。
// TaskN256 八路、TaskN128 四路按阶段展开；每路处理一个 S32 小组的全部 Q64。
template <uint32_t TaskN, QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT>
__simd_vf__ inline void UpdatePScaleDnVf(__ubuf__ uint8_t* encodedScale,
    __ubuf__ float* firstTaskExponent, __ubuf__ float* secondTaskExponent,
    __ubuf__ float* groupReference, __ubuf__ float* previousReference,
    __ubuf__ float* historyMultiplier, uint16_t firstTaskCount, uint16_t secondTaskCount, bool firstGroup)
{
    static_assert(TaskN == 128 || TaskN == 256, "MXFP4 Pscale supports TaskN128/256");
    constexpr uint32_t ROW_COUNT = 64;
    constexpr uint32_t GROUPS_PER_TASK = TaskN / 32;
    constexpr uint32_t TASK_EXPONENT_ELEMENTS = ROW_COUNT * GROUPS_PER_TASK;
    constexpr uint32_t TASK_SCALE_BYTES = ROW_COUNT / 16 * (TaskN / 64 + 1) * 32;
    MaskReg allRows = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> groupMaximum, oldReference, mergedReference, referenceDifference, zeroValues;
    RegTensor<float> multiplier;
    MaskReg oldValidRows, newValidRows, validHistory;
    // 每个 lane 对应一行 Q；参考和 alpha 在整个 task 组内共用。
    LoadAlign(groupMaximum, groupReference);
    Duplicate(zeroValues, 0.0F);
    if (firstGroup) {
        Duplicate(oldReference, INVALID_GROUP_EXPONENT);
    } else {
        LoadAlign(oldReference, previousReference);
    }
    Max(mergedReference, groupMaximum, oldReference, allRows);
    Compares<float, CMPMODE::GT>(oldValidRows, oldReference, INVALID_GROUP_EXPONENT, allRows);
    Compares<float, CMPMODE::GT>(newValidRows, mergedReference, INVALID_GROUP_EXPONENT, allRows);
    MaskAnd(validHistory, oldValidRows, newValidRows, allRows);
    Sub(referenceDifference, oldReference, mergedReference, allRows);
    Select(referenceDifference, referenceDifference, zeroValues, validHistory);
    Muls(referenceDifference, referenceDifference, MXFP4_LN2, allRows);
    Exp(multiplier, referenceDifference, allRows);
    Select(multiplier, multiplier, zeroValues, validHistory);
    StoreAlign(historyMultiplier, multiplier, allRows);
    StoreAlign(groupReference, mergedReference, allRows);
    StoreAlign(previousReference, mergedReference, allRows);

    MaskReg allHalfwords = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> compactIndices, scaleOffsets, rowBlocks, rowsInBlock;

    // 每对 S32 的两个 scale 合成 uint16，按 [Q16块,S64块,Q16,2] 散写。
    Arange(reinterpret_cast<RegTensor<int16_t>&>(compactIndices), static_cast<int16_t>(0));
    ShiftRights(rowBlocks, compactIndices, static_cast<int16_t>(4), allHalfwords);
    Muls(rowBlocks, rowBlocks, static_cast<uint16_t>((TaskN / 64 + 1) * 16), allHalfwords);
    ShiftLefts(rowsInBlock, compactIndices, static_cast<int16_t>(12), allHalfwords);
    ShiftRights(rowsInBlock, rowsInBlock, static_cast<int16_t>(12), allHalfwords);
    Add(scaleOffsets, rowBlocks, rowsInBlock, allHalfwords);
    ShiftLefts(compactIndices, compactIndices, static_cast<int16_t>(1), allHalfwords);

    uint32_t validHalfwordCount = 64;
    MaskReg validHalfwords = UpdateMask<uint16_t>(validHalfwordCount);
    RegTensor<uint16_t> paddingPairs;
    Duplicate(paddingPairs, static_cast<uint16_t>(0));
    MaskReg allBytes = CreateMask<uint8_t, MaskPattern::ALL>();
    for (uint16_t segmentIndex = 0; segmentIndex < 2; ++segmentIndex) {
        const uint16_t segmentTaskCount = segmentIndex == 0 ? firstTaskCount : secondTaskCount;
        __ubuf__ float* segmentExponent = segmentIndex == 0 ? firstTaskExponent : secondTaskExponent;
        __ubuf__ uint8_t* segmentScale = encodedScale +
            (segmentIndex == 0 ? 0 : firstTaskCount * TASK_SCALE_BYTES);
        for (uint16_t taskIndex = 0; taskIndex < segmentTaskCount; ++taskIndex) {
            __ubuf__ float* taskGroupExponent = segmentExponent + taskIndex * TASK_EXPONENT_ELEMENTS;
            __ubuf__ uint8_t* taskScale = segmentScale + taskIndex * TASK_SCALE_BYTES;
            // 数字后缀表示 S32 小组编号；各路独立，量化结果原位复用指数寄存器
            RegTensor<float> vreg_scale0;
            RegTensor<float> vreg_scale1;
            RegTensor<float> vreg_scale2;
            RegTensor<float> vreg_scale3;
            RegTensor<float> vreg_scale4;
            RegTensor<float> vreg_scale5;
            RegTensor<float> vreg_scale6;
            RegTensor<float> vreg_scale7;
            MaskReg preg_valid0;
            MaskReg preg_valid1;
            MaskReg preg_valid2;
            MaskReg preg_valid3;
            MaskReg preg_valid4;
            MaskReg preg_valid5;
            MaskReg preg_valid6;
            MaskReg preg_valid7;
            RegTensor<int32_t> vreg_integer0;
            RegTensor<int32_t> vreg_integer1;
            RegTensor<int32_t> vreg_integer2;
            RegTensor<int32_t> vreg_integer3;
            RegTensor<int32_t> vreg_integer4;
            RegTensor<int32_t> vreg_integer5;
            RegTensor<int32_t> vreg_integer6;
            RegTensor<int32_t> vreg_integer7;
            RegTensor<uint8_t> vreg_bytes0;
            RegTensor<uint8_t> vreg_bytes1;
            RegTensor<uint8_t> vreg_bytes2;
            RegTensor<uint8_t> vreg_bytes3;
            RegTensor<uint8_t> vreg_bytes4;
            RegTensor<uint8_t> vreg_bytes5;
            RegTensor<uint8_t> vreg_bytes6;
            RegTensor<uint8_t> vreg_bytes7;
            RegTensor<uint16_t> vreg_pair0;
            RegTensor<uint16_t> vreg_pair1;
            RegTensor<uint16_t> vreg_pair2;
            RegTensor<uint16_t> vreg_pair3;

            // 先加载全部小组，TaskN128 不访问后四个小组
            LoadAlign(vreg_scale0, taskGroupExponent + 0 * ROW_COUNT);
            LoadAlign(vreg_scale1, taskGroupExponent + 1 * ROW_COUNT);
            LoadAlign(vreg_scale2, taskGroupExponent + 2 * ROW_COUNT);
            LoadAlign(vreg_scale3, taskGroupExponent + 3 * ROW_COUNT);
            if constexpr (TaskN == 256) {
                LoadAlign(vreg_scale4, taskGroupExponent + 4 * ROW_COUNT);
                LoadAlign(vreg_scale5, taskGroupExponent + 5 * ROW_COUNT);
                LoadAlign(vreg_scale6, taskGroupExponent + 6 * ROW_COUNT);
                LoadAlign(vreg_scale7, taskGroupExponent + 7 * ROW_COUNT);
            }

            // 在覆盖指数之前保存各小组的有效性，空组保持零编码。
            Compares<float, CMPMODE::GT>(preg_valid0, vreg_scale0, INVALID_GROUP_EXPONENT, allRows);
            Compares<float, CMPMODE::GT>(preg_valid1, vreg_scale1, INVALID_GROUP_EXPONENT, allRows);
            Compares<float, CMPMODE::GT>(preg_valid2, vreg_scale2, INVALID_GROUP_EXPONENT, allRows);
            Compares<float, CMPMODE::GT>(preg_valid3, vreg_scale3, INVALID_GROUP_EXPONENT, allRows);
            if constexpr (TaskN == 256) {
                Compares<float, CMPMODE::GT>(preg_valid4, vreg_scale4, INVALID_GROUP_EXPONENT, allRows);
                Compares<float, CMPMODE::GT>(preg_valid5, vreg_scale5, INVALID_GROUP_EXPONENT, allRows);
                Compares<float, CMPMODE::GT>(preg_valid6, vreg_scale6, INVALID_GROUP_EXPONENT, allRows);
                Compares<float, CMPMODE::GT>(preg_valid7, vreg_scale7, INVALID_GROUP_EXPONENT, allRows);
            }

            // 所有小组统一减去本调度组冻结的参考指数。
            Sub(vreg_scale0, vreg_scale0, mergedReference, allRows);
            Sub(vreg_scale1, vreg_scale1, mergedReference, allRows);
            Sub(vreg_scale2, vreg_scale2, mergedReference, allRows);
            Sub(vreg_scale3, vreg_scale3, mergedReference, allRows);
            if constexpr (TaskN == 256) {
                Sub(vreg_scale4, vreg_scale4, mergedReference, allRows);
                Sub(vreg_scale5, vreg_scale5, mergedReference, allRows);
                Sub(vreg_scale6, vreg_scale6, mergedReference, allRows);
                Sub(vreg_scale7, vreg_scale7, mergedReference, allRows);
            }

            // 普通量化编码为 e-R+125，CX 编码为 e-R+127-ceil(log2(7.25)) 或者 e-R+127-ceil(log2(7.25/1.5)) ，即 e-R+124。
            Adds(vreg_scale0, vreg_scale0, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
            Adds(vreg_scale1, vreg_scale1, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
            Adds(vreg_scale2, vreg_scale2, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
            Adds(vreg_scale3, vreg_scale3, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
            if constexpr (TaskN == 256) {
                Adds(vreg_scale4, vreg_scale4, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
                Adds(vreg_scale5, vreg_scale5, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
                Adds(vreg_scale6, vreg_scale6, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
                Adds(vreg_scale7, vreg_scale7, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
            }

            // 编码下界钳制。
            Maxs(vreg_scale0, vreg_scale0, 0.0F, allRows);
            Maxs(vreg_scale1, vreg_scale1, 0.0F, allRows);
            Maxs(vreg_scale2, vreg_scale2, 0.0F, allRows);
            Maxs(vreg_scale3, vreg_scale3, 0.0F, allRows);
            if constexpr (TaskN == 256) {
                Maxs(vreg_scale4, vreg_scale4, 0.0F, allRows);
                Maxs(vreg_scale5, vreg_scale5, 0.0F, allRows);
                Maxs(vreg_scale6, vreg_scale6, 0.0F, allRows);
                Maxs(vreg_scale7, vreg_scale7, 0.0F, allRows);
            }

            // 编码上界钳制，避免写入保留编码255。
            Mins(vreg_scale0, vreg_scale0, 254.0F, allRows);
            Mins(vreg_scale1, vreg_scale1, 254.0F, allRows);
            Mins(vreg_scale2, vreg_scale2, 254.0F, allRows);
            Mins(vreg_scale3, vreg_scale3, 254.0F, allRows);
            if constexpr (TaskN == 256) {
                Mins(vreg_scale4, vreg_scale4, 254.0F, allRows);
                Mins(vreg_scale5, vreg_scale5, 254.0F, allRows);
                Mins(vreg_scale6, vreg_scale6, 254.0F, allRows);
                Mins(vreg_scale7, vreg_scale7, 254.0F, allRows);
            }

            // 空 S32 小组写零字节编码，对应 P 的清零由 Softmax 完成。
            Select(vreg_scale0, vreg_scale0, zeroValues, preg_valid0);
            Select(vreg_scale1, vreg_scale1, zeroValues, preg_valid1);
            Select(vreg_scale2, vreg_scale2, zeroValues, preg_valid2);
            Select(vreg_scale3, vreg_scale3, zeroValues, preg_valid3);
            if constexpr (TaskN == 256) {
                Select(vreg_scale4, vreg_scale4, zeroValues, preg_valid4);
                Select(vreg_scale5, vreg_scale5, zeroValues, preg_valid5);
                Select(vreg_scale6, vreg_scale6, zeroValues, preg_valid6);
                Select(vreg_scale7, vreg_scale7, zeroValues, preg_valid7);
            }

            // 先交错完成各路 FP32 到 int32 的转换。
            Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer0, vreg_scale0, allRows);
            Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer1, vreg_scale1, allRows);
            Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer2, vreg_scale2, allRows);
            Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer3, vreg_scale3, allRows);
            if constexpr (TaskN == 256) {
                Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer4, vreg_scale4, allRows);
                Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer5, vreg_scale5, allRows);
                Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer6, vreg_scale6, allRows);
                Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer7, vreg_scale7, allRows);
            }

            // 再转换为字节：偶数组放第0字节，奇数组放第1字节。
            Cast<uint8_t, int32_t, MXFP4_CAST_ZERO>(vreg_bytes0, vreg_integer0, allRows);
            Cast<uint8_t, int32_t, MXFP4_CAST_ONE>(vreg_bytes1, vreg_integer1, allRows);
            Cast<uint8_t, int32_t, MXFP4_CAST_ZERO>(vreg_bytes2, vreg_integer2, allRows);
            Cast<uint8_t, int32_t, MXFP4_CAST_ONE>(vreg_bytes3, vreg_integer3, allRows);
            if constexpr (TaskN == 256) {
                Cast<uint8_t, int32_t, MXFP4_CAST_ZERO>(vreg_bytes4, vreg_integer4, allRows);
                Cast<uint8_t, int32_t, MXFP4_CAST_ONE>(vreg_bytes5, vreg_integer5, allRows);
                Cast<uint8_t, int32_t, MXFP4_CAST_ZERO>(vreg_bytes6, vreg_integer6, allRows);
                Cast<uint8_t, int32_t, MXFP4_CAST_ONE>(vreg_bytes7, vreg_integer7, allRows);
            }

            // 相邻两个 S32 小组打包成一个 S64 的 scale 对。
            Or(vreg_bytes0, vreg_bytes0, vreg_bytes1, allBytes);
            Or(vreg_bytes2, vreg_bytes2, vreg_bytes3, allBytes);
            if constexpr (TaskN == 256) {
                Or(vreg_bytes4, vreg_bytes4, vreg_bytes5, allBytes);
                Or(vreg_bytes6, vreg_bytes6, vreg_bytes7, allBytes);
            }

            // 去掉每个32位槽后两个空字节，前64个uint16对应Q64。
            Gather(vreg_pair0, reinterpret_cast<RegTensor<uint16_t>&>(vreg_bytes0), compactIndices);
            Gather(vreg_pair1, reinterpret_cast<RegTensor<uint16_t>&>(vreg_bytes2), compactIndices);
            if constexpr (TaskN == 256) {
                Gather(vreg_pair2, reinterpret_cast<RegTensor<uint16_t>&>(vreg_bytes4), compactIndices);
                Gather(vreg_pair3, reinterpret_cast<RegTensor<uint16_t>&>(vreg_bytes6), compactIndices);
            }

            // 按现有物理布局写出各个S64的scale，输出地址单位为字节。
            Scatter(reinterpret_cast<__ubuf__ uint16_t*>(taskScale + 0 * 32),
                vreg_pair0, scaleOffsets, validHalfwords);
            Scatter(reinterpret_cast<__ubuf__ uint16_t*>(taskScale + 1 * 32),
                vreg_pair1, scaleOffsets, validHalfwords);
            if constexpr (TaskN == 256) {
                Scatter(reinterpret_cast<__ubuf__ uint16_t*>(taskScale + 2 * 32),
                    vreg_pair2, scaleOffsets, validHalfwords);
                Scatter(reinterpret_cast<__ubuf__ uint16_t*>(taskScale + 3 * 32),
                    vreg_pair3, scaleOffsets, validHalfwords);
            }

            // 每个Q16块末尾预留的32B padding只需写一次。
            Scatter(reinterpret_cast<__ubuf__ uint16_t*>(taskScale + TaskN / 64 * 32),
                paddingPairs, scaleOffsets, validHalfwords);
        }
    }
}

// ND：输入每个 task 为 [Q64,TaskN/32] FP32 指数，输出为 [4,TaskN/64+1,16,2] 字节。
// TaskN256 八路、TaskN128 四路按阶段展开；每路连续读取 [Q8,8] 或 [Q16,4]。
template <uint32_t TaskN, QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT, uint32_t RowCapacity = 64>
__simd_vf__ inline void UpdatePScaleNdVf(__ubuf__ uint8_t* encodedScale,
    __ubuf__ float* firstTaskExponent, __ubuf__ float* secondTaskExponent,
    __ubuf__ float* groupReference, __ubuf__ float* previousReference,
    __ubuf__ float* historyMultiplier, uint16_t firstTaskCount, uint16_t secondTaskCount, bool firstGroup)
{
    static_assert(TaskN == 128 || TaskN == 256, "MXFP4 Pscale supports TaskN128/256");
    static_assert(RowCapacity >= 16 && RowCapacity <= 64 && RowCapacity % 16 == 0, "ND Pscale rows must align Q16");
    constexpr uint32_t ROW_COUNT = 64;
    constexpr uint32_t GROUPS_PER_TASK = TaskN / 32;
    constexpr uint32_t TASK_EXPONENT_ELEMENTS = ROW_COUNT * GROUPS_PER_TASK;
    constexpr uint32_t TASK_SCALE_BYTES = ROW_COUNT / 16 * (TaskN / 64 + 1) * 32;
    constexpr uint32_t REGISTER_ELEMENTS = 64;
    constexpr uint32_t ROWS_PER_REGISTER = REGISTER_ELEMENTS / GROUPS_PER_TASK;
    constexpr uint32_t ROW_BLOCK_BYTES = (TaskN / 64 + 1) * 32;
    constexpr int16_t GROUP_SHIFT = TaskN == 256 ? 3 : 2;
    constexpr int16_t PAIR_SHIFT = TaskN == 256 ? 2 : 1;
    MaskReg allRows = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> groupMaximum, oldReference, mergedReference, referenceDifference, zeroValues;
    RegTensor<float> multiplier;
    MaskReg oldValidRows, newValidRows, validHistory;
    // 每个 lane 对应一行 Q；参考和 alpha 在整个 task 组内共用。
    LoadAlign(groupMaximum, groupReference);
    Duplicate(zeroValues, 0.0F);
    if (firstGroup) {
        // 首组没有历史结果，直接使用当前参考。
        mergedReference = groupMaximum;
        Duplicate(multiplier, 0.0F);
    } else {
        LoadAlign(oldReference, previousReference);
        Max(mergedReference, groupMaximum, oldReference, allRows);
        Compares<float, CMPMODE::GT>(oldValidRows, oldReference, INVALID_GROUP_EXPONENT, allRows);
        Compares<float, CMPMODE::GT>(newValidRows, mergedReference, INVALID_GROUP_EXPONENT, allRows);
        MaskAnd(validHistory, oldValidRows, newValidRows, allRows);
        Sub(referenceDifference, oldReference, mergedReference, allRows);
        Select(referenceDifference, referenceDifference, zeroValues, validHistory);
        Muls(referenceDifference, referenceDifference, MXFP4_LN2, allRows);
        Exp(multiplier, referenceDifference, allRows);
        Select(multiplier, multiplier, zeroValues, validHistory);
    }
    StoreAlign(historyMultiplier, multiplier, allRows);
    StoreAlign(groupReference, mergedReference, allRows);
    StoreAlign(previousReference, mergedReference, allRows);

    MaskReg allHalfwords = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg allBytes = CreateMask<uint8_t, MaskPattern::ALL>();
    uint32_t validPairCount = 32;
    MaskReg validPairs = UpdateMask<uint16_t>(validPairCount);
    uint32_t validPaddingCount = RowCapacity;
    MaskReg validPadding = UpdateMask<uint16_t>(validPaddingCount);

    // 每行参考重复 GROUPS_PER_TASK 次，对应连续输入的同一行各个 S32 指数。
    RegTensor<uint32_t> referenceIndices;
    RegTensor<float> vreg_reference0, vreg_reference1, vreg_reference2, vreg_reference3;
    RegTensor<float> vreg_reference4, vreg_reference5, vreg_reference6, vreg_reference7;
    Arange(reinterpret_cast<RegTensor<int32_t>&>(referenceIndices), static_cast<int32_t>(0));
    ShiftRights(referenceIndices, referenceIndices, GROUP_SHIFT, allRows);
    Gather(vreg_reference0, mergedReference, referenceIndices);
    Adds(referenceIndices, referenceIndices, ROWS_PER_REGISTER, allRows);
    if constexpr (RowCapacity > 1 * ROWS_PER_REGISTER) {
        Gather(vreg_reference1, mergedReference, referenceIndices);
    }
    Adds(referenceIndices, referenceIndices, ROWS_PER_REGISTER, allRows);
    if constexpr (RowCapacity > 2 * ROWS_PER_REGISTER) {
        Gather(vreg_reference2, mergedReference, referenceIndices);
    }
    Adds(referenceIndices, referenceIndices, ROWS_PER_REGISTER, allRows);
    if constexpr (RowCapacity > 3 * ROWS_PER_REGISTER) {
        Gather(vreg_reference3, mergedReference, referenceIndices);
    }
    if constexpr (TaskN == 256) {
        Adds(referenceIndices, referenceIndices, ROWS_PER_REGISTER, allRows);
        if constexpr (RowCapacity > 4 * ROWS_PER_REGISTER) {
            Gather(vreg_reference4, mergedReference, referenceIndices);
        }
        Adds(referenceIndices, referenceIndices, ROWS_PER_REGISTER, allRows);
        if constexpr (RowCapacity > 5 * ROWS_PER_REGISTER) {
            Gather(vreg_reference5, mergedReference, referenceIndices);
        }
        Adds(referenceIndices, referenceIndices, ROWS_PER_REGISTER, allRows);
        if constexpr (RowCapacity > 6 * ROWS_PER_REGISTER) {
            Gather(vreg_reference6, mergedReference, referenceIndices);
        }
        Adds(referenceIndices, referenceIndices, ROWS_PER_REGISTER, allRows);
        if constexpr (RowCapacity > 7 * ROWS_PER_REGISTER) {
            Gather(vreg_reference7, mergedReference, referenceIndices);
        }
    }

    // Cast 后有效字节位于 0、4、8、...、252；压紧结果仅使用前64字节。
    RegTensor<uint8_t> compactByteIndices;
    Arange(reinterpret_cast<RegTensor<int8_t>&>(compactByteIndices), static_cast<int8_t>(0));
    ShiftLefts(compactByteIndices, compactByteIndices, static_cast<int16_t>(2), allBytes);

    // 压紧后相邻两个字节属于同一 Q 行的相邻 S32 组，按 uint16 散写。
    // 每路偏移为 S64编号*16 + 路内Q行号，单位为 uint16 元素。
    RegTensor<uint16_t> pairIndices, pairRows, pairColumns, pairOffsets;
    Arange(reinterpret_cast<RegTensor<int16_t>&>(pairIndices), static_cast<int16_t>(0));
    ShiftRights(pairRows, pairIndices, PAIR_SHIFT, allHalfwords);
    ShiftLefts(pairColumns, pairIndices, static_cast<int16_t>(16 - PAIR_SHIFT), allHalfwords);
    ShiftRights(pairColumns, pairColumns, static_cast<int16_t>(16 - PAIR_SHIFT), allHalfwords);
    ShiftLefts(pairColumns, pairColumns, static_cast<int16_t>(4), allHalfwords);
    Add(pairOffsets, pairColumns, pairRows, allHalfwords);

    // 每个 Q16 块最后一个32B区域是 padding。
    RegTensor<uint16_t> paddingIndices, paddingBlocks, paddingRows, paddingOffsets, paddingPairs;
    Arange(reinterpret_cast<RegTensor<int16_t>&>(paddingIndices), static_cast<int16_t>(0));
    ShiftRights(paddingBlocks, paddingIndices, static_cast<int16_t>(4), allHalfwords);
    Muls(paddingBlocks, paddingBlocks, static_cast<uint16_t>(ROW_BLOCK_BYTES / 2), allHalfwords);
    ShiftLefts(paddingRows, paddingIndices, static_cast<int16_t>(12), allHalfwords);
    ShiftRights(paddingRows, paddingRows, static_cast<int16_t>(12), allHalfwords);
    Add(paddingOffsets, paddingBlocks, paddingRows, allHalfwords);
    Duplicate(paddingPairs, static_cast<uint16_t>(0));

    // 输入统计环分首尾两段，输出在 UB 中按 task 顺序连续排列。
    for (uint16_t segmentIndex = 0; segmentIndex < 2; ++segmentIndex) {
        const uint16_t segmentTaskCount = segmentIndex == 0 ? firstTaskCount : secondTaskCount;
        __ubuf__ float* segmentExponent = segmentIndex == 0 ? firstTaskExponent : secondTaskExponent;
        __ubuf__ uint8_t* segmentScale = encodedScale +
            (segmentIndex == 0 ? 0 : firstTaskCount * TASK_SCALE_BYTES);
        for (uint16_t taskIndex = 0; taskIndex < segmentTaskCount; ++taskIndex) {
            __ubuf__ float* taskGroupExponent = segmentExponent + taskIndex * TASK_EXPONENT_ELEMENTS;
            __ubuf__ uint8_t* taskScale = segmentScale + taskIndex * TASK_SCALE_BYTES;
            // 数字后缀表示连续输入的64元素块编号；各路独立，量化结果原位复用指数寄存器
            RegTensor<float> vreg_scale0;
            RegTensor<float> vreg_scale1;
            RegTensor<float> vreg_scale2;
            RegTensor<float> vreg_scale3;
            RegTensor<float> vreg_scale4;
            RegTensor<float> vreg_scale5;
            RegTensor<float> vreg_scale6;
            RegTensor<float> vreg_scale7;
            MaskReg preg_valid0;
            MaskReg preg_valid1;
            MaskReg preg_valid2;
            MaskReg preg_valid3;
            MaskReg preg_valid4;
            MaskReg preg_valid5;
            MaskReg preg_valid6;
            MaskReg preg_valid7;
            RegTensor<int32_t> vreg_integer0;
            RegTensor<int32_t> vreg_integer1;
            RegTensor<int32_t> vreg_integer2;
            RegTensor<int32_t> vreg_integer3;
            RegTensor<int32_t> vreg_integer4;
            RegTensor<int32_t> vreg_integer5;
            RegTensor<int32_t> vreg_integer6;
            RegTensor<int32_t> vreg_integer7;
            RegTensor<uint8_t> vreg_bytes0;
            RegTensor<uint8_t> vreg_bytes1;
            RegTensor<uint8_t> vreg_bytes2;
            RegTensor<uint8_t> vreg_bytes3;
            RegTensor<uint8_t> vreg_bytes4;
            RegTensor<uint8_t> vreg_bytes5;
            RegTensor<uint8_t> vreg_bytes6;
            RegTensor<uint8_t> vreg_bytes7;

            RegTensor<uint8_t> vreg_packed0;
            RegTensor<uint8_t> vreg_packed1;
            RegTensor<uint8_t> vreg_packed2;
            RegTensor<uint8_t> vreg_packed3;
            RegTensor<uint8_t> vreg_packed4;
            RegTensor<uint8_t> vreg_packed5;
            RegTensor<uint8_t> vreg_packed6;
            RegTensor<uint8_t> vreg_packed7;

            // 先连续加载整个task，TaskN128 不访问后四路
            LoadAlign(vreg_scale0, taskGroupExponent + 0 * REGISTER_ELEMENTS);
            if constexpr (RowCapacity > 1 * ROWS_PER_REGISTER) {
                LoadAlign(vreg_scale1, taskGroupExponent + 1 * REGISTER_ELEMENTS);
            }
            if constexpr (RowCapacity > 2 * ROWS_PER_REGISTER) {
                LoadAlign(vreg_scale2, taskGroupExponent + 2 * REGISTER_ELEMENTS);
            }
            if constexpr (RowCapacity > 3 * ROWS_PER_REGISTER) {
                LoadAlign(vreg_scale3, taskGroupExponent + 3 * REGISTER_ELEMENTS);
            }
            if constexpr (TaskN == 256) {
                if constexpr (RowCapacity > 4 * ROWS_PER_REGISTER) {
                    LoadAlign(vreg_scale4, taskGroupExponent + 4 * REGISTER_ELEMENTS);
                }
                if constexpr (RowCapacity > 5 * ROWS_PER_REGISTER) {
                    LoadAlign(vreg_scale5, taskGroupExponent + 5 * REGISTER_ELEMENTS);
                }
                if constexpr (RowCapacity > 6 * ROWS_PER_REGISTER) {
                    LoadAlign(vreg_scale6, taskGroupExponent + 6 * REGISTER_ELEMENTS);
                }
                if constexpr (RowCapacity > 7 * ROWS_PER_REGISTER) {
                    LoadAlign(vreg_scale7, taskGroupExponent + 7 * REGISTER_ELEMENTS);
                }
            }

            // 在覆盖指数之前保存各小组的有效性，空组保持零编码。
            Compares<float, CMPMODE::GT>(preg_valid0, vreg_scale0, INVALID_GROUP_EXPONENT, allRows);
            if constexpr (RowCapacity > 1 * ROWS_PER_REGISTER) {
                Compares<float, CMPMODE::GT>(preg_valid1, vreg_scale1, INVALID_GROUP_EXPONENT, allRows);
            }
            if constexpr (RowCapacity > 2 * ROWS_PER_REGISTER) {
                Compares<float, CMPMODE::GT>(preg_valid2, vreg_scale2, INVALID_GROUP_EXPONENT, allRows);
            }
            if constexpr (RowCapacity > 3 * ROWS_PER_REGISTER) {
                Compares<float, CMPMODE::GT>(preg_valid3, vreg_scale3, INVALID_GROUP_EXPONENT, allRows);
            }
            if constexpr (TaskN == 256) {
                if constexpr (RowCapacity > 4 * ROWS_PER_REGISTER) {
                    Compares<float, CMPMODE::GT>(preg_valid4, vreg_scale4, INVALID_GROUP_EXPONENT, allRows);
                }
                if constexpr (RowCapacity > 5 * ROWS_PER_REGISTER) {
                    Compares<float, CMPMODE::GT>(preg_valid5, vreg_scale5, INVALID_GROUP_EXPONENT, allRows);
                }
                if constexpr (RowCapacity > 6 * ROWS_PER_REGISTER) {
                    Compares<float, CMPMODE::GT>(preg_valid6, vreg_scale6, INVALID_GROUP_EXPONENT, allRows);
                }
                if constexpr (RowCapacity > 7 * ROWS_PER_REGISTER) {
                    Compares<float, CMPMODE::GT>(preg_valid7, vreg_scale7, INVALID_GROUP_EXPONENT, allRows);
                }
            }

            // 每个元素减去所属 Q 行的参考指数。
            Sub(vreg_scale0, vreg_scale0, vreg_reference0, allRows);
            if constexpr (RowCapacity > 1 * ROWS_PER_REGISTER) {
                Sub(vreg_scale1, vreg_scale1, vreg_reference1, allRows);
            }
            if constexpr (RowCapacity > 2 * ROWS_PER_REGISTER) {
                Sub(vreg_scale2, vreg_scale2, vreg_reference2, allRows);
            }
            if constexpr (RowCapacity > 3 * ROWS_PER_REGISTER) {
                Sub(vreg_scale3, vreg_scale3, vreg_reference3, allRows);
            }
            if constexpr (TaskN == 256) {
                if constexpr (RowCapacity > 4 * ROWS_PER_REGISTER) {
                    Sub(vreg_scale4, vreg_scale4, vreg_reference4, allRows);
                }
                if constexpr (RowCapacity > 5 * ROWS_PER_REGISTER) {
                    Sub(vreg_scale5, vreg_scale5, vreg_reference5, allRows);
                }
                if constexpr (RowCapacity > 6 * ROWS_PER_REGISTER) {
                    Sub(vreg_scale6, vreg_scale6, vreg_reference6, allRows);
                }
                if constexpr (RowCapacity > 7 * ROWS_PER_REGISTER) {
                    Sub(vreg_scale7, vreg_scale7, vreg_reference7, allRows);
                }
            }

            // 普通量化编码为 e-R+125，CX 编码为 e-R+127-ceil(log2(7.25)) 或者 e-R+127-ceil(log2(7.25/1.5)) ，即 e-R+124。
            Adds(vreg_scale0, vreg_scale0, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
            if constexpr (RowCapacity > 1 * ROWS_PER_REGISTER) {
                Adds(vreg_scale1, vreg_scale1, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
            }
            if constexpr (RowCapacity > 2 * ROWS_PER_REGISTER) {
                Adds(vreg_scale2, vreg_scale2, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
            }
            if constexpr (RowCapacity > 3 * ROWS_PER_REGISTER) {
                Adds(vreg_scale3, vreg_scale3, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
            }
            if constexpr (TaskN == 256) {
                if constexpr (RowCapacity > 4 * ROWS_PER_REGISTER) {
                    Adds(vreg_scale4, vreg_scale4, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
                }
                if constexpr (RowCapacity > 5 * ROWS_PER_REGISTER) {
                    Adds(vreg_scale5, vreg_scale5, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
                }
                if constexpr (RowCapacity > 6 * ROWS_PER_REGISTER) {
                    Adds(vreg_scale6, vreg_scale6, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
                }
                if constexpr (RowCapacity > 7 * ROWS_PER_REGISTER) {
                    Adds(vreg_scale7, vreg_scale7, QUANT_SCHEMA != QuantSchema::OCP_QUANT ? MXFP4_CX_PSCALE_BIAS : 125.0F, allRows);
                }
            }

            // 编码下界钳制。
            Maxs(vreg_scale0, vreg_scale0, 0.0F, allRows);
            if constexpr (RowCapacity > 1 * ROWS_PER_REGISTER) {
                Maxs(vreg_scale1, vreg_scale1, 0.0F, allRows);
            }
            if constexpr (RowCapacity > 2 * ROWS_PER_REGISTER) {
                Maxs(vreg_scale2, vreg_scale2, 0.0F, allRows);
            }
            if constexpr (RowCapacity > 3 * ROWS_PER_REGISTER) {
                Maxs(vreg_scale3, vreg_scale3, 0.0F, allRows);
            }
            if constexpr (TaskN == 256) {
                if constexpr (RowCapacity > 4 * ROWS_PER_REGISTER) {
                    Maxs(vreg_scale4, vreg_scale4, 0.0F, allRows);
                }
                if constexpr (RowCapacity > 5 * ROWS_PER_REGISTER) {
                    Maxs(vreg_scale5, vreg_scale5, 0.0F, allRows);
                }
                if constexpr (RowCapacity > 6 * ROWS_PER_REGISTER) {
                    Maxs(vreg_scale6, vreg_scale6, 0.0F, allRows);
                }
                if constexpr (RowCapacity > 7 * ROWS_PER_REGISTER) {
                    Maxs(vreg_scale7, vreg_scale7, 0.0F, allRows);
                }
            }

            // 编码上界钳制，避免写入保留编码255。
            Mins(vreg_scale0, vreg_scale0, 254.0F, allRows);
            if constexpr (RowCapacity > 1 * ROWS_PER_REGISTER) {
                Mins(vreg_scale1, vreg_scale1, 254.0F, allRows);
            }
            if constexpr (RowCapacity > 2 * ROWS_PER_REGISTER) {
                Mins(vreg_scale2, vreg_scale2, 254.0F, allRows);
            }
            if constexpr (RowCapacity > 3 * ROWS_PER_REGISTER) {
                Mins(vreg_scale3, vreg_scale3, 254.0F, allRows);
            }
            if constexpr (TaskN == 256) {
                if constexpr (RowCapacity > 4 * ROWS_PER_REGISTER) {
                    Mins(vreg_scale4, vreg_scale4, 254.0F, allRows);
                }
                if constexpr (RowCapacity > 5 * ROWS_PER_REGISTER) {
                    Mins(vreg_scale5, vreg_scale5, 254.0F, allRows);
                }
                if constexpr (RowCapacity > 6 * ROWS_PER_REGISTER) {
                    Mins(vreg_scale6, vreg_scale6, 254.0F, allRows);
                }
                if constexpr (RowCapacity > 7 * ROWS_PER_REGISTER) {
                    Mins(vreg_scale7, vreg_scale7, 254.0F, allRows);
                }
            }

            // 空 S32 小组写零字节编码，对应 P 的清零由 Softmax 完成。
            Select(vreg_scale0, vreg_scale0, zeroValues, preg_valid0);
            if constexpr (RowCapacity > 1 * ROWS_PER_REGISTER) {
                Select(vreg_scale1, vreg_scale1, zeroValues, preg_valid1);
            }
            if constexpr (RowCapacity > 2 * ROWS_PER_REGISTER) {
                Select(vreg_scale2, vreg_scale2, zeroValues, preg_valid2);
            }
            if constexpr (RowCapacity > 3 * ROWS_PER_REGISTER) {
                Select(vreg_scale3, vreg_scale3, zeroValues, preg_valid3);
            }
            if constexpr (TaskN == 256) {
                if constexpr (RowCapacity > 4 * ROWS_PER_REGISTER) {
                    Select(vreg_scale4, vreg_scale4, zeroValues, preg_valid4);
                }
                if constexpr (RowCapacity > 5 * ROWS_PER_REGISTER) {
                    Select(vreg_scale5, vreg_scale5, zeroValues, preg_valid5);
                }
                if constexpr (RowCapacity > 6 * ROWS_PER_REGISTER) {
                    Select(vreg_scale6, vreg_scale6, zeroValues, preg_valid6);
                }
                if constexpr (RowCapacity > 7 * ROWS_PER_REGISTER) {
                    Select(vreg_scale7, vreg_scale7, zeroValues, preg_valid7);
                }
            }

            // 先交错完成各路 FP32 到 int32 的转换。
            Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer0, vreg_scale0, allRows);
            if constexpr (RowCapacity > 1 * ROWS_PER_REGISTER) {
                Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer1, vreg_scale1, allRows);
            }
            if constexpr (RowCapacity > 2 * ROWS_PER_REGISTER) {
                Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer2, vreg_scale2, allRows);
            }
            if constexpr (RowCapacity > 3 * ROWS_PER_REGISTER) {
                Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer3, vreg_scale3, allRows);
            }
            if constexpr (TaskN == 256) {
                if constexpr (RowCapacity > 4 * ROWS_PER_REGISTER) {
                    Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer4, vreg_scale4, allRows);
                }
                if constexpr (RowCapacity > 5 * ROWS_PER_REGISTER) {
                    Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer5, vreg_scale5, allRows);
                }
                if constexpr (RowCapacity > 6 * ROWS_PER_REGISTER) {
                    Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer6, vreg_scale6, allRows);
                }
                if constexpr (RowCapacity > 7 * ROWS_PER_REGISTER) {
                    Cast<int32_t, float, MXFP4_CAST_ZERO>(vreg_integer7, vreg_scale7, allRows);
                }
            }

            // 全部使用ZERO布局，编码位于每个32位槽的第0字节。
            Cast<uint8_t, int32_t, MXFP4_CAST_ZERO>(vreg_bytes0, vreg_integer0, allRows);
            if constexpr (RowCapacity > 1 * ROWS_PER_REGISTER) {
                Cast<uint8_t, int32_t, MXFP4_CAST_ZERO>(vreg_bytes1, vreg_integer1, allRows);
            }
            if constexpr (RowCapacity > 2 * ROWS_PER_REGISTER) {
                Cast<uint8_t, int32_t, MXFP4_CAST_ZERO>(vreg_bytes2, vreg_integer2, allRows);
            }
            if constexpr (RowCapacity > 3 * ROWS_PER_REGISTER) {
                Cast<uint8_t, int32_t, MXFP4_CAST_ZERO>(vreg_bytes3, vreg_integer3, allRows);
            }
            if constexpr (TaskN == 256) {
                if constexpr (RowCapacity > 4 * ROWS_PER_REGISTER) {
                    Cast<uint8_t, int32_t, MXFP4_CAST_ZERO>(vreg_bytes4, vreg_integer4, allRows);
                }
                if constexpr (RowCapacity > 5 * ROWS_PER_REGISTER) {
                    Cast<uint8_t, int32_t, MXFP4_CAST_ZERO>(vreg_bytes5, vreg_integer5, allRows);
                }
                if constexpr (RowCapacity > 6 * ROWS_PER_REGISTER) {
                    Cast<uint8_t, int32_t, MXFP4_CAST_ZERO>(vreg_bytes6, vreg_integer6, allRows);
                }
                if constexpr (RowCapacity > 7 * ROWS_PER_REGISTER) {
                    Cast<uint8_t, int32_t, MXFP4_CAST_ZERO>(vreg_bytes7, vreg_integer7, allRows);
                }
            }

            // 压紧后保持 Q 行优先，每路前64字节组成32个有效scale对。
            Gather(vreg_packed0, vreg_bytes0, compactByteIndices);
            if constexpr (RowCapacity > 1 * ROWS_PER_REGISTER) {
                Gather(vreg_packed1, vreg_bytes1, compactByteIndices);
            }
            if constexpr (RowCapacity > 2 * ROWS_PER_REGISTER) {
                Gather(vreg_packed2, vreg_bytes2, compactByteIndices);
            }
            if constexpr (RowCapacity > 3 * ROWS_PER_REGISTER) {
                Gather(vreg_packed3, vreg_bytes3, compactByteIndices);
            }
            if constexpr (TaskN == 256) {
                if constexpr (RowCapacity > 4 * ROWS_PER_REGISTER) {
                    Gather(vreg_packed4, vreg_bytes4, compactByteIndices);
                }
                if constexpr (RowCapacity > 5 * ROWS_PER_REGISTER) {
                    Gather(vreg_packed5, vreg_bytes5, compactByteIndices);
                }
                if constexpr (RowCapacity > 6 * ROWS_PER_REGISTER) {
                    Gather(vreg_packed6, vreg_bytes6, compactByteIndices);
                }
                if constexpr (RowCapacity > 7 * ROWS_PER_REGISTER) {
                    Gather(vreg_packed7, vreg_bytes7, compactByteIndices);
                }
            }

            // 每路首行决定基地址，pairOffsets 完成 Q 行优先到物理布局的重排。
            Scatter(reinterpret_cast<__ubuf__ uint16_t*>(taskScale +
                (0 * ROWS_PER_REGISTER / 16) * ROW_BLOCK_BYTES + (0 * ROWS_PER_REGISTER % 16) * 2),
                reinterpret_cast<RegTensor<uint16_t>&>(vreg_packed0), pairOffsets, validPairs);
            if constexpr (RowCapacity > 1 * ROWS_PER_REGISTER) {
                Scatter(reinterpret_cast<__ubuf__ uint16_t*>(taskScale +
                    (1 * ROWS_PER_REGISTER / 16) * ROW_BLOCK_BYTES + (1 * ROWS_PER_REGISTER % 16) * 2),
                    reinterpret_cast<RegTensor<uint16_t>&>(vreg_packed1), pairOffsets, validPairs);
            }
            if constexpr (RowCapacity > 2 * ROWS_PER_REGISTER) {
                Scatter(reinterpret_cast<__ubuf__ uint16_t*>(taskScale +
                    (2 * ROWS_PER_REGISTER / 16) * ROW_BLOCK_BYTES + (2 * ROWS_PER_REGISTER % 16) * 2),
                    reinterpret_cast<RegTensor<uint16_t>&>(vreg_packed2), pairOffsets, validPairs);
            }
            if constexpr (RowCapacity > 3 * ROWS_PER_REGISTER) {
                Scatter(reinterpret_cast<__ubuf__ uint16_t*>(taskScale +
                    (3 * ROWS_PER_REGISTER / 16) * ROW_BLOCK_BYTES + (3 * ROWS_PER_REGISTER % 16) * 2),
                    reinterpret_cast<RegTensor<uint16_t>&>(vreg_packed3), pairOffsets, validPairs);
            }
            if constexpr (TaskN == 256) {
                if constexpr (RowCapacity > 4 * ROWS_PER_REGISTER) {
                    Scatter(reinterpret_cast<__ubuf__ uint16_t*>(taskScale +
                        (4 * ROWS_PER_REGISTER / 16) * ROW_BLOCK_BYTES + (4 * ROWS_PER_REGISTER % 16) * 2),
                        reinterpret_cast<RegTensor<uint16_t>&>(vreg_packed4), pairOffsets, validPairs);
                }
                if constexpr (RowCapacity > 5 * ROWS_PER_REGISTER) {
                    Scatter(reinterpret_cast<__ubuf__ uint16_t*>(taskScale +
                        (5 * ROWS_PER_REGISTER / 16) * ROW_BLOCK_BYTES + (5 * ROWS_PER_REGISTER % 16) * 2),
                        reinterpret_cast<RegTensor<uint16_t>&>(vreg_packed5), pairOffsets, validPairs);
                }
                if constexpr (RowCapacity > 6 * ROWS_PER_REGISTER) {
                    Scatter(reinterpret_cast<__ubuf__ uint16_t*>(taskScale +
                        (6 * ROWS_PER_REGISTER / 16) * ROW_BLOCK_BYTES + (6 * ROWS_PER_REGISTER % 16) * 2),
                        reinterpret_cast<RegTensor<uint16_t>&>(vreg_packed6), pairOffsets, validPairs);
                }
                if constexpr (RowCapacity > 7 * ROWS_PER_REGISTER) {
                    Scatter(reinterpret_cast<__ubuf__ uint16_t*>(taskScale +
                        (7 * ROWS_PER_REGISTER / 16) * ROW_BLOCK_BYTES + (7 * ROWS_PER_REGISTER % 16) * 2),
                        reinterpret_cast<RegTensor<uint16_t>&>(vreg_packed7), pairOffsets, validPairs);
                }
            }
            // 当前 task 的四个 Q16 块 padding 一次性清零。
            Scatter(reinterpret_cast<__ubuf__ uint16_t*>(taskScale + TaskN / 64 * 32),
                paddingPairs, paddingOffsets, validPadding);
        }
    }
}

// DN 包装：接收 UB LocalTensor，跨流水同步仍由调用方负责。
template <uint32_t TaskN, QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT>
__aicore__ inline void UpdatePScaleDn(const LocalTensor<uint8_t>& encodedScale,
    const LocalTensor<float>& firstTaskExponent, const LocalTensor<float>& secondTaskExponent,
    const LocalTensor<float>& groupReference, const LocalTensor<float>& previousReference,
    const LocalTensor<float>& historyMultiplier, uint16_t firstTaskCount, uint16_t secondTaskCount, bool firstGroup)
{
    UpdatePScaleDnVf<TaskN, QUANT_SCHEMA>(
        (__ubuf__ uint8_t*)encodedScale.GetPhyAddr(), (__ubuf__ float*)firstTaskExponent.GetPhyAddr(),
        (__ubuf__ float*)secondTaskExponent.GetPhyAddr(), (__ubuf__ float*)groupReference.GetPhyAddr(),
        (__ubuf__ float*)previousReference.GetPhyAddr(), (__ubuf__ float*)historyMultiplier.GetPhyAddr(),
        firstTaskCount, secondTaskCount, firstGroup);
}

// ND 包装：接收 UB LocalTensor，跨流水同步仍由调用方负责。
template <uint32_t TaskN, QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT>
__aicore__ inline void UpdatePScaleNd(const LocalTensor<uint8_t>& encodedScale,
    const LocalTensor<float>& firstTaskExponent, const LocalTensor<float>& secondTaskExponent,
    const LocalTensor<float>& groupReference, const LocalTensor<float>& previousReference,
    const LocalTensor<float>& historyMultiplier, uint16_t firstTaskCount, uint16_t secondTaskCount, bool firstGroup, uint32_t rowCapacity = 64)
{
    if (rowCapacity == 16) {
        UpdatePScaleNdVf<TaskN, QUANT_SCHEMA, 16>(
        (__ubuf__ uint8_t*)encodedScale.GetPhyAddr(), (__ubuf__ float*)firstTaskExponent.GetPhyAddr(),
        (__ubuf__ float*)secondTaskExponent.GetPhyAddr(), (__ubuf__ float*)groupReference.GetPhyAddr(),
        (__ubuf__ float*)previousReference.GetPhyAddr(), (__ubuf__ float*)historyMultiplier.GetPhyAddr(),
        firstTaskCount, secondTaskCount, firstGroup);
    }
    else if (rowCapacity == 32) {
        UpdatePScaleNdVf<TaskN, QUANT_SCHEMA, 32>(
        (__ubuf__ uint8_t*)encodedScale.GetPhyAddr(), (__ubuf__ float*)firstTaskExponent.GetPhyAddr(),
        (__ubuf__ float*)secondTaskExponent.GetPhyAddr(), (__ubuf__ float*)groupReference.GetPhyAddr(),
        (__ubuf__ float*)previousReference.GetPhyAddr(), (__ubuf__ float*)historyMultiplier.GetPhyAddr(),
        firstTaskCount, secondTaskCount, firstGroup);
    }
    else if (rowCapacity == 48) {
        UpdatePScaleNdVf<TaskN, QUANT_SCHEMA, 48>(
        (__ubuf__ uint8_t*)encodedScale.GetPhyAddr(), (__ubuf__ float*)firstTaskExponent.GetPhyAddr(),
        (__ubuf__ float*)secondTaskExponent.GetPhyAddr(), (__ubuf__ float*)groupReference.GetPhyAddr(),
        (__ubuf__ float*)previousReference.GetPhyAddr(), (__ubuf__ float*)historyMultiplier.GetPhyAddr(),
        firstTaskCount, secondTaskCount, firstGroup);
    }
    else {
        UpdatePScaleNdVf<TaskN, QUANT_SCHEMA, 64>(
        (__ubuf__ uint8_t*)encodedScale.GetPhyAddr(), (__ubuf__ float*)firstTaskExponent.GetPhyAddr(),
        (__ubuf__ float*)secondTaskExponent.GetPhyAddr(), (__ubuf__ float*)groupReference.GetPhyAddr(),
        (__ubuf__ float*)previousReference.GetPhyAddr(), (__ubuf__ float*)historyMultiplier.GetPhyAddr(),
        firstTaskCount, secondTaskCount, firstGroup);
    }
}

} // namespace Mxfp4Api
#endif
