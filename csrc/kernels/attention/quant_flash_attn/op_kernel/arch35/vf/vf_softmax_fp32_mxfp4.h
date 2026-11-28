/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QFA_VF_SOFTMAX_FP32_MXFP4_H_
#define QFA_VF_SOFTMAX_FP32_MXFP4_H_
#include "kernel_operator.h"
#include "../quant_flash_attn_common_def.h"

namespace Mxfp4Api {
using BaseApi::QuantSchema;
using namespace AscendC;
using namespace AscendC::Reg;

constexpr uint32_t VEC1_ROWS = 64;
constexpr uint32_t QUANT_GROUP_SIZE = 32;
constexpr float INVALID_GROUP_EXPONENT = -3.402823466e+38F;
constexpr float MXFP4_LN2 = 0.6931471805599453F;
constexpr float MXFP4_INV_LN2 = 1.4426950408889634F;
constexpr float MXFP4_NEG_LOG2_CX_DELTA = -2.273018494406416F; // -log2(7.25/1.5)
constexpr float MXFP4_NEG_LOG2_CX = -2.8579809951275723F; // -log2(7.25)
constexpr float MXFP4_CX_SCALE_SHIFT = 3.0F; // ceil(log2(7.25))，ceil(log2(7.25/1.5))
constexpr float MXFP4_CX_PSCALE_BIAS = 127.0F - MXFP4_CX_SCALE_SHIFT;
constexpr float MXFP4_NEG_LN_DELTA = -0.4054651081081644F; // -ln(1.5)
constexpr float MXFP4_NEG_TWO_VALUE = -2.0F;
// 输出归一化使用默认硬件除法，精度按既有门限验收。
constexpr DivSpecificMode MXFP4_OUTPUT_DIV = {MaskMergeMode::ZEROING, false};

constexpr CastTrait MXFP4_CAST_ZERO = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                      RoundMode::CAST_ROUND};
constexpr CastTrait MXFP4_CAST_ONE = {RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING,
                                     RoundMode::CAST_ROUND};

constexpr CastTrait MXFP4_CAST_TWO = {RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING,
                                     RoundMode::CAST_ROUND};
constexpr CastTrait MXFP4_CAST_THREE = {RegLayout::THREE, SatMode::SAT, MaskMergeMode::ZEROING,
                                       RoundMode::CAST_ROUND};

struct SoftmaxFp32Mxfp4Params {
    uint32_t validRows;   // ND 使用有效 Q 行数；DN 固定计算全部 64 行。
    uint32_t validN;      // 当前 score 子块的有效 KV 数。
    uint32_t s2Offset;    // 当前子块在整个 task 中的 KV 偏移。
    uint64_t qGs1Start;   // 当前 AIV 第一行在当前 batch 的合轴位置。
    uint32_t qGroupSize;  // DN 为 1，ND 为实际 G。
    uint64_t kvStart;     // 当前子块在当前 batch 中的 KV 起点。
    int64_t causalOffset; // 右下 causal 偏移：KV 长度减 Q 长度，允许负值。
    float scoreScale;     // attention scale，只应用一次。
};

// BF16 索引将偶奇交错恢复为两列 Q64；字节索引将四路 FP4 合并结果恢复为八列 Q64。
// 字节索引前 64 项仍为 4*i，ND 路径用此前缀压紧单路 FP4。
__simd_vf__ inline void InitProbabilityPackingIndicesVf(__ubuf__ uint8_t* byteIndices,
                                                       __ubuf__ uint16_t* halfwordIndices)
{
    RegTensor<uint8_t> byteOffsets, bytePart;
    RegTensor<uint16_t> halfwordOffsets, halfwordRows, halfwordParity;
    MaskReg allBytes = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg allHalfwords = CreateMask<uint16_t, MaskPattern::ALL>();
    Arange(reinterpret_cast<RegTensor<int8_t>&>(byteOffsets), static_cast<int8_t>(0));
    ShiftRights(bytePart, byteOffsets, static_cast<int16_t>(6), allBytes);
    ShiftLefts(byteOffsets, byteOffsets, static_cast<int16_t>(2), allBytes);
    Or(byteOffsets, byteOffsets, bytePart, allBytes);
    StoreAlign(byteIndices, byteOffsets, allBytes);
    Arange(reinterpret_cast<RegTensor<int16_t>&>(halfwordOffsets), static_cast<int16_t>(0));
    ShiftLefts(halfwordRows, halfwordOffsets, static_cast<int16_t>(10), allHalfwords);
    ShiftRights(halfwordRows, halfwordRows, static_cast<int16_t>(10), allHalfwords);
    ShiftLefts(halfwordRows, halfwordRows, static_cast<int16_t>(1), allHalfwords);
    ShiftRights(halfwordParity, halfwordOffsets, static_cast<int16_t>(6), allHalfwords);
    Or(halfwordOffsets, halfwordRows, halfwordParity, allHalfwords);
    StoreAlign(halfwordIndices, halfwordOffsets, allHalfwords);
}

// Full DN chunks retain only the last eight columns during the S32 max pass.
// Tail/causal chunks use the general path; this path has no additional UB.
template <uint32_t ChunkN, QuantSchema QUANT_SCHEMA, bool ScorePreScaled>
__simd_vf__ inline void SoftmaxDnResidentEightVf(__ubuf__ uint8_t* packedPUb,
    __ubuf__ float* groupMaxUb, __ubuf__ float* globalMaxUb, __ubuf__ float* scoreUb,
    float scoreScale)
{
    MaskReg preg_135 = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_16bit = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_all_8bit = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg group_hasvalid = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> src0, src1, src2, src3, src4, src5, src6, src7;
    RegTensor<float> maximum0, maximum1, maximum2, maximum3, curr_group_max, global_group_max;
    RegTensor<bfloat16_t> src_bf16_0, src_bf16_1, src_bf16_2, src_bf16_3;
    RegTensor<float4_e2m1x2_t> quant_0, quant_1, quant_2, quant_3;
    LoadAlign(global_group_max, globalMaxUb);
    for (uint16_t group = 0; group < ChunkN / 32; ++group) {
        Duplicate(curr_group_max, INVALID_GROUP_EXPONENT);
        for (uint16_t block = 0; block < 4; ++block) {
            LoadAlign(src0, scoreUb + (group * 32 + block * 8 + 0) * 64);
            LoadAlign(src1, scoreUb + (group * 32 + block * 8 + 1) * 64);
            LoadAlign(src2, scoreUb + (group * 32 + block * 8 + 2) * 64);
            LoadAlign(src3, scoreUb + (group * 32 + block * 8 + 3) * 64);
            LoadAlign(src4, scoreUb + (group * 32 + block * 8 + 4) * 64);
            LoadAlign(src5, scoreUb + (group * 32 + block * 8 + 5) * 64);
            LoadAlign(src6, scoreUb + (group * 32 + block * 8 + 6) * 64);
            LoadAlign(src7, scoreUb + (group * 32 + block * 8 + 7) * 64);
            Max(maximum0, src0, src1, preg_135);
            Max(maximum1, src2, src3, preg_135);
            Max(maximum2, src4, src5, preg_135);
            Max(maximum3, src6, src7, preg_135);
            Max(maximum0, maximum0, maximum1, preg_135);
            Max(maximum2, maximum2, maximum3, preg_135);
            Max(maximum0, maximum0, maximum2, preg_135);
            Max(curr_group_max, curr_group_max, maximum0, preg_135);
        }
        if constexpr (!ScorePreScaled) {
            Muls(curr_group_max, curr_group_max, scoreScale, preg_135);
        }
        Muls(curr_group_max, curr_group_max, MXFP4_INV_LN2, preg_135);
        if constexpr (QUANT_SCHEMA != QuantSchema::OCP_QUANT) {
            Adds(curr_group_max, curr_group_max, QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT ? MXFP4_NEG_LOG2_CX_DELTA : MXFP4_NEG_LOG2_CX, preg_135);
            Truncate<float, RoundMode::CAST_CEIL>(curr_group_max, curr_group_max, preg_135);
        } else {
            Truncate<float, RoundMode::CAST_FLOOR>(curr_group_max, curr_group_max, preg_135);
        }
        Max(global_group_max, global_group_max, curr_group_max, preg_135);
        StoreAlign(groupMaxUb + group * 64, curr_group_max, preg_135);
        if constexpr (QUANT_SCHEMA != QuantSchema::OCP_QUANT) {
            Muls(curr_group_max, curr_group_max, MXFP4_LN2, preg_135);
            if constexpr (QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT) {
                Adds(curr_group_max, curr_group_max, MXFP4_NEG_LN_DELTA, preg_135);
            }
        } else {
            Adds(curr_group_max, curr_group_max, MXFP4_NEG_TWO_VALUE, preg_135);
            Muls(curr_group_max, curr_group_max, MXFP4_LN2, preg_135);
        }
            if constexpr (!ScorePreScaled) {
                Muls(src0, src0, scoreScale, preg_135);
                Muls(src1, src1, scoreScale, preg_135);
                Muls(src2, src2, scoreScale, preg_135);
                Muls(src3, src3, scoreScale, preg_135);
                Muls(src4, src4, scoreScale, preg_135);
                Muls(src5, src5, scoreScale, preg_135);
                Muls(src6, src6, scoreScale, preg_135);
                Muls(src7, src7, scoreScale, preg_135);
            }
            AscendC::Reg::ExpSub(src0, src0, curr_group_max, preg_135);
            AscendC::Reg::ExpSub(src1, src1, curr_group_max, preg_135);
            AscendC::Reg::ExpSub(src2, src2, curr_group_max, preg_135);
            AscendC::Reg::ExpSub(src3, src3, curr_group_max, preg_135);
            AscendC::Reg::ExpSub(src4, src4, curr_group_max, preg_135);
            AscendC::Reg::ExpSub(src5, src5, curr_group_max, preg_135);
            AscendC::Reg::ExpSub(src6, src6, curr_group_max, preg_135);
            AscendC::Reg::ExpSub(src7, src7, curr_group_max, preg_135);
            // 压缩4bit， 先转成 bf16
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_0, src0, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_1, src1, preg_135);
            DeInterleave(src_bf16_0, src_bf16_1, src_bf16_0, src_bf16_1);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_0, src_bf16_0, preg_all_16bit);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_2, src2, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_3, src3, preg_135);
            DeInterleave(src_bf16_2, src_bf16_3, src_bf16_2, src_bf16_3);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_1, src_bf16_2, preg_all_16bit);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_0, src4, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_1, src5, preg_135);
            DeInterleave(src_bf16_0, src_bf16_1, src_bf16_0, src_bf16_1);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_2, src_bf16_0, preg_all_16bit);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_2, src6, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_3, src7, preg_135);
            DeInterleave(src_bf16_2, src_bf16_3, src_bf16_2, src_bf16_3);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_3, src_bf16_2, preg_all_16bit);

            // Each FP4 cast writes one packed byte per 32-bit lane. Two fixed
            // deinterleave stages compact the four Q64 column pairs into 256B.
            DeInterleave(reinterpret_cast<RegTensor<uint16_t>&>(quant_0), reinterpret_cast<RegTensor<uint16_t>&>(quant_1),
                reinterpret_cast<RegTensor<uint16_t>&>(quant_0), reinterpret_cast<RegTensor<uint16_t>&>(quant_1));
            DeInterleave(reinterpret_cast<RegTensor<uint16_t>&>(quant_2), reinterpret_cast<RegTensor<uint16_t>&>(quant_3),
                reinterpret_cast<RegTensor<uint16_t>&>(quant_2), reinterpret_cast<RegTensor<uint16_t>&>(quant_3));
            DeInterleave(reinterpret_cast<RegTensor<uint8_t>&>(quant_0), reinterpret_cast<RegTensor<uint8_t>&>(quant_2),
                reinterpret_cast<RegTensor<uint8_t>&>(quant_0), reinterpret_cast<RegTensor<uint8_t>&>(quant_2));


            StoreAlign(packedPUb + (group * 32 + 24) * 32, reinterpret_cast<RegTensor<uint8_t>&>(quant_0), preg_all_8bit);
        for (uint16_t block = 0; block < 3; ++block) {
            LoadAlign(src0, scoreUb + (group * 32 + block * 8 + 0) * 64);
            LoadAlign(src1, scoreUb + (group * 32 + block * 8 + 1) * 64);
            LoadAlign(src2, scoreUb + (group * 32 + block * 8 + 2) * 64);
            LoadAlign(src3, scoreUb + (group * 32 + block * 8 + 3) * 64);
            LoadAlign(src4, scoreUb + (group * 32 + block * 8 + 4) * 64);
            LoadAlign(src5, scoreUb + (group * 32 + block * 8 + 5) * 64);
            LoadAlign(src6, scoreUb + (group * 32 + block * 8 + 6) * 64);
            LoadAlign(src7, scoreUb + (group * 32 + block * 8 + 7) * 64);
            if constexpr (!ScorePreScaled) {
                Muls(src0, src0, scoreScale, preg_135);
                Muls(src1, src1, scoreScale, preg_135);
                Muls(src2, src2, scoreScale, preg_135);
                Muls(src3, src3, scoreScale, preg_135);
                Muls(src4, src4, scoreScale, preg_135);
                Muls(src5, src5, scoreScale, preg_135);
                Muls(src6, src6, scoreScale, preg_135);
                Muls(src7, src7, scoreScale, preg_135);
            }
            AscendC::Reg::ExpSub(src0, src0, curr_group_max, preg_135);
            AscendC::Reg::ExpSub(src1, src1, curr_group_max, preg_135);
            AscendC::Reg::ExpSub(src2, src2, curr_group_max, preg_135);
            AscendC::Reg::ExpSub(src3, src3, curr_group_max, preg_135);
            AscendC::Reg::ExpSub(src4, src4, curr_group_max, preg_135);
            AscendC::Reg::ExpSub(src5, src5, curr_group_max, preg_135);
            AscendC::Reg::ExpSub(src6, src6, curr_group_max, preg_135);
            AscendC::Reg::ExpSub(src7, src7, curr_group_max, preg_135);
            // 压缩4bit， 先转成 bf16
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_0, src0, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_1, src1, preg_135);
            DeInterleave(src_bf16_0, src_bf16_1, src_bf16_0, src_bf16_1);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_0, src_bf16_0, preg_all_16bit);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_2, src2, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_3, src3, preg_135);
            DeInterleave(src_bf16_2, src_bf16_3, src_bf16_2, src_bf16_3);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_1, src_bf16_2, preg_all_16bit);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_0, src4, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_1, src5, preg_135);
            DeInterleave(src_bf16_0, src_bf16_1, src_bf16_0, src_bf16_1);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_2, src_bf16_0, preg_all_16bit);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_2, src6, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_3, src7, preg_135);
            DeInterleave(src_bf16_2, src_bf16_3, src_bf16_2, src_bf16_3);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_3, src_bf16_2, preg_all_16bit);

            // Each FP4 cast writes one packed byte per 32-bit lane. Two fixed
            // deinterleave stages compact the four Q64 column pairs into 256B.
            DeInterleave(reinterpret_cast<RegTensor<uint16_t>&>(quant_0), reinterpret_cast<RegTensor<uint16_t>&>(quant_1),
                reinterpret_cast<RegTensor<uint16_t>&>(quant_0), reinterpret_cast<RegTensor<uint16_t>&>(quant_1));
            DeInterleave(reinterpret_cast<RegTensor<uint16_t>&>(quant_2), reinterpret_cast<RegTensor<uint16_t>&>(quant_3),
                reinterpret_cast<RegTensor<uint16_t>&>(quant_2), reinterpret_cast<RegTensor<uint16_t>&>(quant_3));
            DeInterleave(reinterpret_cast<RegTensor<uint8_t>&>(quant_0), reinterpret_cast<RegTensor<uint8_t>&>(quant_2),
                reinterpret_cast<RegTensor<uint8_t>&>(quant_0), reinterpret_cast<RegTensor<uint8_t>&>(quant_2));


            StoreAlign(packedPUb + (group * 32 + block * 8) * 32, reinterpret_cast<RegTensor<uint8_t>&>(quant_0), preg_all_8bit);
        }
    }
    StoreAlign(globalMaxUb, global_group_max, preg_135);
}

// DN：固定 Q64；当前业务 (ChunkN,TaskN) 为 (128,256) 或 (128,128)。
// 模板要求 ChunkN>0、ChunkN/TaskN 均为64倍数且 ChunkN<=TaskN；TaskN 为完整task容量。
// 输入/原位暂存 scoreUb：[ChunkN,64] FP32，内部会写入尾部mask及causal mask后的score。
// ByteIdxUb/HalfIdxUb 保留包装接口兼容；固定 DeInterleave/PACK4 打包不读取索引表。
// 输出 packedPUb：完整task的物理布局为 [TaskN,32] 字节，每个KV位置包含64个Q行的打包FP4。
// 输出 groupMaxUb：完整task的 [TaskN/32,64] FP32小组指数（非原始score最大值）。
// packedPUb/groupMaxUb 均由调用方提前偏移到当前chunk起点；VF内从局部偏移0开始写。
// 输入/输出 globalMaxUb：[64] FP32累计指数；ResetGroupMax=true 时重新初始化，否则与历史值合并。
// scoreScale 为attention缩放；causal可见条件为 kvIndex<=queryRow+causalMaskLine（HasMask=true）。
// validQueryRows 当前仅保留接口，DN始终计算64行；要求0<validKvTokens<=ChunkN，空chunk由调用方跳过。
// alignedKvTokens32/64 分别为有效KV数量向32/64对齐；调用方预清零P并将小组指数初始化为无效哨兵。
template <uint32_t ChunkN = 128, uint32_t TaskN = 256,
          bool HasMask = true, QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT,
          bool ResetGroupMax = false, bool ScorePreScaled = false>
__simd_vf__ inline void SoftmaxQuantizeDnChunkFp32Vf(
    __ubuf__ uint8_t* packedPUb,
    __ubuf__ float* groupMaxUb,
    __ubuf__ float* globalMaxUb,
    __ubuf__ float* scoreUb,
    __ubuf__ uint8_t* ByteIdxUb,
    __ubuf__ uint16_t* HalfIdxUb,
    float scoreScale,
    int32_t causalMaskLine,
    uint32_t validQueryRows,
    uint32_t validKvTokens,
    uint16_t alignedKvTokens32,
    uint16_t alignedKvTokens64)
{
    static_assert(VEC1_ROWS == VEC1_ROWS, "DN VF requires Q64");
    static_assert(ChunkN > 0 && ChunkN % 64 == 0 && ChunkN <= TaskN && TaskN % 64 == 0,
                  "DN VF requires complete S64 storage blocks");
    // 有效 Q 行由输出阶段裁剪；本 VF 始终计算完整 Q64。
    (void)validQueryRows;
    // 调用方保证 0 < validKvTokens <= ChunkN，S32/S64 对齐长度由 validKvTokens 生成。

    // ====================== 分块常量 ======================
    const uint16_t ROWS_PER_GROUP = 32;
    const uint16_t GROUP_COUNT = alignedKvTokens32 / ROWS_PER_GROUP;
    const uint16_t GROUP_COUNT_ALIGN_64 = alignedKvTokens64 / ROWS_PER_GROUP;
    const uint16_t ROW_SUB_LOOP = 4;
    const uint16_t ITER_PER_GROUP = ROWS_PER_GROUP / ROW_SUB_LOOP;

    RegTensor<float> src0, src1, src2, src3, src4, src5, src6, src7;
    RegTensor<float> src_n0, src_n1, src_n2, src_n3, src_n4, src_n5, src_n6, src_n7;
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_chunk0;
    RegTensor<float> vreg_thresh;

    RegTensor<float> curr_group_max, next_group_max, global_group_max;
    RegTensor<bfloat16_t> src_bf16_0, src_bf16_1, src_bf16_2, src_bf16_3;
    RegTensor<float4_e2m1x2_t> quant_0, quant_1, quant_2, quant_3;

    MaskReg group_hasvalid;


    __ubuf__ float* src_ub0 = scoreUb;
    __ubuf__ float* src_ub1 = src_ub0 + VEC1_ROWS;
    __ubuf__ float* src_ub2 = src_ub0 + VEC1_ROWS * 2;
    __ubuf__ float* src_ub3 = src_ub0 + VEC1_ROWS * 3;

    MaskReg preg_135, preg_all_16bit, preg_all_8bit;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    preg_135 = CreateMask<float, MaskPattern::ALL>();
    preg_all_16bit = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_all_8bit = CreateMask<uint8_t, MaskPattern::ALL>();

    Duplicate(vreg_min, INVALID_GROUP_EXPONENT);

    // 填充不对齐的kv列为最小值
    if (validKvTokens < ChunkN) {
        for (uint16_t i = validKvTokens; i < ChunkN; ++i) {
            StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(scoreUb + i * VEC1_ROWS, vreg_min, preg_135);
        }
        LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }

    // 处理mask
    if constexpr (HasMask) {
        Arange(vreg_chunk0, (float)causalMaskLine);
        Duplicate(vreg_thresh, 0.0f);
        for (uint16_t iter_n = 0; iter_n < uint16_t(ChunkN / 4); ++iter_n) {
            LoadAlign(src0, src_ub0 + iter_n * VEC1_ROWS * 4);
            LoadAlign(src1, src_ub1 + iter_n * VEC1_ROWS * 4);
            LoadAlign(src2, src_ub2 + iter_n * VEC1_ROWS * 4);
            LoadAlign(src3, src_ub3 + iter_n * VEC1_ROWS * 4);
            Compare<float, CMPMODE::GE>(preg_compare0, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<float, CMPMODE::GE>(preg_compare1, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<float, CMPMODE::GE>(preg_compare2, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<float, CMPMODE::GE>(preg_compare3, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + iter_n * VEC1_ROWS * 4, src0, preg_135);
            StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + iter_n * VEC1_ROWS * 4, src1, preg_135);
            StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + iter_n * VEC1_ROWS * 4, src2, preg_135);
            StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + iter_n * VEC1_ROWS * 4, src3, preg_135);
        }
        LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }


    if constexpr (ResetGroupMax) {
        Duplicate(global_group_max, INVALID_GROUP_EXPONENT);
    } else {
        LoadAlign(global_group_max, globalMaxUb);
    }


    // 计算第一个32行形成的组的最大值
    Duplicate(curr_group_max, INVALID_GROUP_EXPONENT);
    for (uint16_t iter_n = 0; iter_n < ITER_PER_GROUP; ++iter_n) {
        LoadAlign(src0, src_ub0 + iter_n * VEC1_ROWS * 4);
        LoadAlign(src1, src_ub1 + iter_n * VEC1_ROWS * 4);
        LoadAlign(src2, src_ub2 + iter_n * VEC1_ROWS * 4);
        LoadAlign(src3, src_ub3 + iter_n * VEC1_ROWS * 4);

        Max(src0, src0, src1, preg_135);
        Max(src2, src2, src3, preg_135);
        Max(curr_group_max, curr_group_max, src0, preg_135);
        Max(curr_group_max, curr_group_max, src2, preg_135);
    }
    Compares<float, CMPMODE::GT>(
        group_hasvalid, curr_group_max, INVALID_GROUP_EXPONENT, preg_135); // 避免全组都是最小值的情况

    if constexpr (!ScorePreScaled) {
        Muls(curr_group_max, curr_group_max, scoreScale, preg_135); // 乘以scale
    }
    Muls(curr_group_max, curr_group_max, MXFP4_INV_LN2, preg_135); // 对齐到ln2

    // 基于cx的fp4量化优化
    // cx= 7.25，k_g = ceil(max_valid(s_g)/ln2 - log2(cx))
    // CX_QUANT: u_g=exp(s_g-k_g*ln2); only CX_DELTA_QUANT applies delta=1.5.
    if constexpr (QUANT_SCHEMA != QuantSchema::OCP_QUANT) {
        Adds(curr_group_max, curr_group_max, QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT ? MXFP4_NEG_LOG2_CX_DELTA : MXFP4_NEG_LOG2_CX, preg_135);
        Truncate<float, RoundMode::CAST_CEIL>(curr_group_max, curr_group_max, preg_135);
    } else {
        // 标准fp4量化
        // k_g = floor(max_valid(s_g) / ln2)
        // u_g = exp(s_g - (k_g-2)*ln2)
        Truncate<float, RoundMode::CAST_FLOOR>(curr_group_max, curr_group_max, preg_135);
    }
    // 使用MERGING模式，避免清零
    Max<float, MaskMergeMode::MERGING>(global_group_max, global_group_max, curr_group_max, group_hasvalid);
    StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(groupMaxUb, curr_group_max, group_hasvalid);

    if constexpr (QUANT_SCHEMA != QuantSchema::OCP_QUANT) {
        Muls(curr_group_max, curr_group_max, MXFP4_LN2, preg_135); // k_g*ln2
        if constexpr (QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT) {
            Adds(curr_group_max, curr_group_max, MXFP4_NEG_LN_DELTA, preg_135);  // k_g*ln2 - ln(delta)
        }
    } else {
        Adds(curr_group_max, curr_group_max, MXFP4_NEG_TWO_VALUE, preg_135); // -2
        Muls(curr_group_max, curr_group_max, MXFP4_LN2, preg_135);
    }

    for (uint16_t i = 0; i < GROUP_COUNT-1; ++i) { // 最后一组单独处理，不需要预取
        Duplicate(next_group_max, INVALID_GROUP_EXPONENT);
        // 每次处理 8 行，8*64 = 512个元素，512*0.5B = 256B，一个reg的大小
        for(uint16_t j = 0; j < ROWS_PER_GROUP / 8; ++j) {
            // 加载下一个行max
            uint16_t next_offset = (i+1) * ROWS_PER_GROUP + j * 8;
            LoadAlign(src_n0, src_ub0 + next_offset * VEC1_ROWS);
            LoadAlign(src_n1, src_ub0 + (next_offset + 1) * VEC1_ROWS);
            LoadAlign(src_n2, src_ub0 + (next_offset + 2) * VEC1_ROWS);
            LoadAlign(src_n3, src_ub0 + (next_offset + 3) * VEC1_ROWS);
            LoadAlign(src_n4, src_ub0 + (next_offset + 4) * VEC1_ROWS);
            LoadAlign(src_n5, src_ub0 + (next_offset + 5) * VEC1_ROWS);
            LoadAlign(src_n6, src_ub0 + (next_offset + 6) * VEC1_ROWS);
            LoadAlign(src_n7, src_ub0 + (next_offset + 7) * VEC1_ROWS);

            Max(src_n0, src_n0, src_n1, preg_135);
            Max(src_n2, src_n2, src_n3, preg_135);
            Max(src_n4, src_n4, src_n5, preg_135);
            Max(src_n6, src_n6, src_n7, preg_135);
            Max(src_n0, src_n0, src_n2, preg_135);
            Max(src_n4, src_n4, src_n6, preg_135);
            Max(next_group_max, next_group_max, src_n0, preg_135);
            Max(next_group_max, next_group_max, src_n4, preg_135);

            // 计算当前块：sub + exp
            uint16_t curr_offset = i * ROWS_PER_GROUP + j * 8;
            LoadAlign(src0, src_ub0 + curr_offset * VEC1_ROWS);
            LoadAlign(src1, src_ub0 + (curr_offset + 1) * VEC1_ROWS);
            LoadAlign(src2, src_ub0 + (curr_offset + 2) * VEC1_ROWS);
            LoadAlign(src3, src_ub0 + (curr_offset + 3) * VEC1_ROWS);
            LoadAlign(src4, src_ub0 + (curr_offset + 4) * VEC1_ROWS);
            LoadAlign(src5, src_ub0 + (curr_offset + 5) * VEC1_ROWS);
            LoadAlign(src6, src_ub0 + (curr_offset + 6) * VEC1_ROWS);
            LoadAlign(src7, src_ub0 + (curr_offset + 7) * VEC1_ROWS);

            // 乘以scale
            if constexpr (!ScorePreScaled) {
                Muls(src0, src0, scoreScale, group_hasvalid);
                Muls(src1, src1, scoreScale, group_hasvalid);
                Muls(src2, src2, scoreScale, group_hasvalid);
                Muls(src3, src3, scoreScale, group_hasvalid);
                Muls(src4, src4, scoreScale, group_hasvalid);
                Muls(src5, src5, scoreScale, group_hasvalid);
                Muls(src6, src6, scoreScale, group_hasvalid);
                Muls(src7, src7, scoreScale, group_hasvalid);
            }

            // exp(s-max)：使用减法与指数复合指令。
            AscendC::Reg::ExpSub(src0, src0, curr_group_max, group_hasvalid);
            AscendC::Reg::ExpSub(src1, src1, curr_group_max, group_hasvalid);
            AscendC::Reg::ExpSub(src2, src2, curr_group_max, group_hasvalid);
            AscendC::Reg::ExpSub(src3, src3, curr_group_max, group_hasvalid);
            AscendC::Reg::ExpSub(src4, src4, curr_group_max, group_hasvalid);
            AscendC::Reg::ExpSub(src5, src5, curr_group_max, group_hasvalid);
            AscendC::Reg::ExpSub(src6, src6, curr_group_max, group_hasvalid);
            AscendC::Reg::ExpSub(src7, src7, curr_group_max, group_hasvalid);

            // 压缩4bit， 先转成 bf16
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_0, src0, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_1, src1, preg_135);
            DeInterleave(src_bf16_0, src_bf16_1, src_bf16_0, src_bf16_1);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_0, src_bf16_0, preg_all_16bit);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_2, src2, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_3, src3, preg_135);
            DeInterleave(src_bf16_2, src_bf16_3, src_bf16_2, src_bf16_3);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_1, src_bf16_2, preg_all_16bit);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_0, src4, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_1, src5, preg_135);
            DeInterleave(src_bf16_0, src_bf16_1, src_bf16_0, src_bf16_1);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_2, src_bf16_0, preg_all_16bit);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_2, src6, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_3, src7, preg_135);
            DeInterleave(src_bf16_2, src_bf16_3, src_bf16_2, src_bf16_3);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_3, src_bf16_2, preg_all_16bit);

            // Each FP4 cast writes one packed byte per 32-bit lane. Two fixed
            // deinterleave stages compact the four Q64 column pairs into 256B.
            DeInterleave(reinterpret_cast<RegTensor<uint16_t>&>(quant_0), reinterpret_cast<RegTensor<uint16_t>&>(quant_1),
                reinterpret_cast<RegTensor<uint16_t>&>(quant_0), reinterpret_cast<RegTensor<uint16_t>&>(quant_1));
            DeInterleave(reinterpret_cast<RegTensor<uint16_t>&>(quant_2), reinterpret_cast<RegTensor<uint16_t>&>(quant_3),
                reinterpret_cast<RegTensor<uint16_t>&>(quant_2), reinterpret_cast<RegTensor<uint16_t>&>(quant_3));
            DeInterleave(reinterpret_cast<RegTensor<uint8_t>&>(quant_0), reinterpret_cast<RegTensor<uint8_t>&>(quant_2),
                reinterpret_cast<RegTensor<uint8_t>&>(quant_0), reinterpret_cast<RegTensor<uint8_t>&>(quant_2));

            StoreAlign(packedPUb + i * ROWS_PER_GROUP * 32 + j * 8 * 32, reinterpret_cast<RegTensor<uint8_t>&>(quant_0), preg_all_8bit);

        }

        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(globalMaxUb, global_group_max, preg_135);
        Compares<float, CMPMODE::GT>(
            group_hasvalid, next_group_max, INVALID_GROUP_EXPONENT, preg_135); // 避免全组都是最小值的情况

        if constexpr (!ScorePreScaled) {
            Muls(next_group_max, next_group_max, scoreScale, preg_135); // 乘以scale
        }
        Muls(next_group_max, next_group_max, MXFP4_INV_LN2, preg_135);
        if constexpr (QUANT_SCHEMA != QuantSchema::OCP_QUANT) {
            Adds(next_group_max, next_group_max, QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT ? MXFP4_NEG_LOG2_CX_DELTA : MXFP4_NEG_LOG2_CX, preg_135);
            Truncate<float, RoundMode::CAST_CEIL>(next_group_max, next_group_max, preg_135);
        } else {
            Truncate<float, RoundMode::CAST_FLOOR>(next_group_max, next_group_max, preg_135);
        }
        Max<float, MaskMergeMode::MERGING>(global_group_max, global_group_max, next_group_max, group_hasvalid);
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(groupMaxUb + ((i + 1) * VEC1_ROWS), next_group_max, group_hasvalid);

        if constexpr (QUANT_SCHEMA != QuantSchema::OCP_QUANT) {
            if constexpr (QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT) {
                Muls(next_group_max, next_group_max, MXFP4_LN2, preg_135);
                Adds(curr_group_max, next_group_max, MXFP4_NEG_LN_DELTA, preg_135);
            } else {
                Muls(curr_group_max, next_group_max, MXFP4_LN2, preg_135);
            }
        } else {
            Adds(next_group_max, next_group_max, MXFP4_NEG_TWO_VALUE, preg_135);
            Muls(curr_group_max, next_group_max, MXFP4_LN2, preg_135);
        }
    }

    // 处理最后一组
    {
        // 计算当前块：sub + exp
        uint16_t i = GROUP_COUNT-1;
        for(uint16_t j = 0; j < ROWS_PER_GROUP / 8; ++j) {
            uint16_t curr_offset = i * ROWS_PER_GROUP + j * 8;
            LoadAlign(src0, src_ub0 + curr_offset * VEC1_ROWS);
            LoadAlign(src1, src_ub0 + (curr_offset + 1) * VEC1_ROWS);
            LoadAlign(src2, src_ub0 + (curr_offset + 2) * VEC1_ROWS);
            LoadAlign(src3, src_ub0 + (curr_offset + 3) * VEC1_ROWS);
            LoadAlign(src4, src_ub0 + (curr_offset + 4) * VEC1_ROWS);
            LoadAlign(src5, src_ub0 + (curr_offset + 5) * VEC1_ROWS);
            LoadAlign(src6, src_ub0 + (curr_offset + 6) * VEC1_ROWS);
            LoadAlign(src7, src_ub0 + (curr_offset + 7) * VEC1_ROWS);

             // 乘以scale
            if constexpr (!ScorePreScaled) {
                Muls(src0, src0, scoreScale, group_hasvalid);
                Muls(src1, src1, scoreScale, group_hasvalid);
                Muls(src2, src2, scoreScale, group_hasvalid);
                Muls(src3, src3, scoreScale, group_hasvalid);
                Muls(src4, src4, scoreScale, group_hasvalid);
                Muls(src5, src5, scoreScale, group_hasvalid);
                Muls(src6, src6, scoreScale, group_hasvalid);
                Muls(src7, src7, scoreScale, group_hasvalid);
            }

            // exp(s-max)：使用减法与指数复合指令。
            AscendC::Reg::ExpSub(src0, src0, curr_group_max, group_hasvalid);
            AscendC::Reg::ExpSub(src1, src1, curr_group_max, group_hasvalid);
            AscendC::Reg::ExpSub(src2, src2, curr_group_max, group_hasvalid);
            AscendC::Reg::ExpSub(src3, src3, curr_group_max, group_hasvalid);
            AscendC::Reg::ExpSub(src4, src4, curr_group_max, group_hasvalid);
            AscendC::Reg::ExpSub(src5, src5, curr_group_max, group_hasvalid);
            AscendC::Reg::ExpSub(src6, src6, curr_group_max, group_hasvalid);
            AscendC::Reg::ExpSub(src7, src7, curr_group_max, group_hasvalid);


            // 压缩4bit， 先转成 bf16
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_0, src0, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_1, src1, preg_135);
            DeInterleave(src_bf16_0, src_bf16_1, src_bf16_0, src_bf16_1);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_0, src_bf16_0, preg_all_16bit);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_2, src2, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_3, src3, preg_135);
            DeInterleave(src_bf16_2, src_bf16_3, src_bf16_2, src_bf16_3);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_1, src_bf16_2, preg_all_16bit);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_0, src4, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_1, src5, preg_135);
            DeInterleave(src_bf16_0, src_bf16_1, src_bf16_0, src_bf16_1);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_2, src_bf16_0, preg_all_16bit);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_2, src6, preg_135);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(src_bf16_3, src7, preg_135);
            DeInterleave(src_bf16_2, src_bf16_3, src_bf16_2, src_bf16_3);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(quant_3, src_bf16_2, preg_all_16bit);

            // Each FP4 cast writes one packed byte per 32-bit lane. Two fixed
            // deinterleave stages compact the four Q64 column pairs into 256B.
            DeInterleave(reinterpret_cast<RegTensor<uint16_t>&>(quant_0), reinterpret_cast<RegTensor<uint16_t>&>(quant_1),
                reinterpret_cast<RegTensor<uint16_t>&>(quant_0), reinterpret_cast<RegTensor<uint16_t>&>(quant_1));
            DeInterleave(reinterpret_cast<RegTensor<uint16_t>&>(quant_2), reinterpret_cast<RegTensor<uint16_t>&>(quant_3),
                reinterpret_cast<RegTensor<uint16_t>&>(quant_2), reinterpret_cast<RegTensor<uint16_t>&>(quant_3));
            DeInterleave(reinterpret_cast<RegTensor<uint8_t>&>(quant_0), reinterpret_cast<RegTensor<uint8_t>&>(quant_2),
                reinterpret_cast<RegTensor<uint8_t>&>(quant_0), reinterpret_cast<RegTensor<uint8_t>&>(quant_2));

            StoreAlign(packedPUb + i * ROWS_PER_GROUP * 32 + j * 8 * 32, reinterpret_cast<RegTensor<uint8_t>&>(quant_0), preg_all_8bit);
        }
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(globalMaxUb, global_group_max, preg_135);
    }
    for (int i = GROUP_COUNT; i < GROUP_COUNT_ALIGN_64; ++i) {
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(groupMaxUb + i * VEC1_ROWS, vreg_min, preg_135);
    }


}



// ND：S64块在外、Q行四行交错展开，末尾0~3行单独处理；每个AIV至多64个Q行。
// 当前业务 (ChunkN,TaskN) 为 (128,256) 或 (64,128)；模板要求 ChunkN为64或128倍数且整除TaskN。
// 输入 scoreUb：[64,ChunkN] FP32；scoreScale 为attention缩放。
// ByteIdxUb/HalfIdxUb 保留包装接口兼容；固定 DeInterleave/PACK4 打包不读取索引表。
// 输出 packedPUb：完整task的物理布局为 [TaskN/64,64,32] 字节，每行32B对应连续64个KV的打包FP4。
// 输出 groupMaxUb：[64,TaskN/32] FP32小组指数（非原始score最大值）；两输出指针均指向完整task起点。
// chunkKvOffset 指定当前chunk在task内的KV偏移，须按64对齐且 chunkKvOffset+ChunkN<=TaskN。
// 输入/输出 globalMaxUb：[64] FP32累计指数；chunk末尾按Q行批量合并，组首由调用方初始化。
// validQueryRows为0~64，validKvTokens为0~ChunkN；调用方预清零整个P并将小组指数初始化为无效哨兵。
// causal可见条件为 queryRow>=causalFirstRow+kvIndex*causalRowStep（HasMask=true）；kvIndex为chunk内索引。
// causalRowStep为合轴G，causalFirstRow为第一个KV的最早可见本地Q行，可为负数，允许token的G跨M块。
template <uint32_t ChunkN, uint32_t TaskN,
          bool HasMask, QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT, bool ScorePreScaled = false>
__simd_vf__ inline void SoftmaxQuantizeNdChunkFp32Vf(
    __ubuf__ uint8_t* packedPUb, __ubuf__ float* groupMaxUb, __ubuf__ float* globalMaxUb,
    __ubuf__ float* scoreUb, __ubuf__ uint8_t* ByteIdxUb, __ubuf__ uint16_t* HalfIdxUb,
    float scoreScale, uint32_t validQueryRows, uint32_t validKvTokens, uint32_t chunkKvOffset,
    float causalFirstRow, float causalRowStep)
{
    static_assert((ChunkN == 64 || ChunkN % 128 == 0) && TaskN % ChunkN == 0,
                  "ND VF requires S64 or complete S128 storage blocks");
    constexpr uint32_t GROUPS_PER_TASK = TaskN / QUANT_GROUP_SIZE;
    const uint16_t alignedRows = static_cast<uint16_t>(validQueryRows / 4 * 4);
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    MaskReg allHalf = CreateMask<uint16_t, MaskPattern::ALL>();
    uint32_t n32 = 32, n2 = 2, n1 = 1, bytes32 = 32;
    MaskReg firstHalf = UpdateMask<float>(n32), secondHalf;
    MaskReg pairMask = UpdateMask<float>(n2), scalarMask = UpdateMask<float>(n1);
    MaskReg packedMask = UpdateMask<uint8_t>(bytes32);
    MaskNot(secondHalf, firstHalf, all);
    RegTensor<float> invalid, causalRows;
    RegTensor<int32_t> keyIndices, chunkIndices;
    RegTensor<uint32_t> pairIndices, storeOffsets;
    Duplicate(invalid, INVALID_GROUP_EXPONENT);
    Arange(keyIndices, static_cast<int32_t>(0));
    Arange(reinterpret_cast<RegTensor<int32_t>&>(storeOffsets), static_cast<int32_t>(0));
    Mins(pairIndices, storeOffsets, static_cast<uint32_t>(1), all);
    ShiftLefts(pairIndices, pairIndices, static_cast<int16_t>(5), all);
    MaskReg validKv;
    RegTensor<float> score0, first0, second0, exponent0, packedExponent0, maximum0;
    RegTensor<int32_t> queryIndex0;
    RegTensor<bfloat16_t> bf0, compactBf0;
    RegTensor<float4_e2m1x2_t> fp0;
    RegTensor<uint8_t> compactFp0;
    MaskReg visible0, groupValid0, expValid0;
    RegTensor<float> score1, first1, second1, exponent1, packedExponent1, maximum1;
    RegTensor<int32_t> queryIndex1;
    RegTensor<bfloat16_t> bf1, compactBf1;
    RegTensor<float4_e2m1x2_t> fp1;
    RegTensor<uint8_t> compactFp1;
    MaskReg visible1, groupValid1, expValid1;
    RegTensor<float> score2, first2, second2, exponent2, packedExponent2, maximum2;
    RegTensor<int32_t> queryIndex2;
    RegTensor<bfloat16_t> bf2, compactBf2;
    RegTensor<float4_e2m1x2_t> fp2;
    RegTensor<uint8_t> compactFp2;
    MaskReg visible2, groupValid2, expValid2;
    RegTensor<float> score3, first3, second3, exponent3, packedExponent3, maximum3;
    RegTensor<int32_t> queryIndex3;
    RegTensor<bfloat16_t> bf3, compactBf3;
    RegTensor<float4_e2m1x2_t> fp3;
    RegTensor<uint8_t> compactFp3;
    MaskReg visible3, groupValid3, expValid3;
    // One register holds both S32 exponents in their corresponding half.
    // Exponent conversion and reference construction therefore execute once per S64.
    for (uint16_t block = 0; block < ChunkN / 64; ++block) {
        const uint32_t firstToken = block * 64;
        const uint32_t firstGroup = chunkKvOffset / QUANT_GROUP_SIZE + block * 2;
        __ubuf__ uint8_t* packedBlock = packedPUb + (chunkKvOffset / 64 + block) * VEC1_ROWS * 32;
        Adds(chunkIndices, keyIndices, static_cast<int32_t>(firstToken), all);
        Compares<int32_t, CMPMODE::LT>(validKv, chunkIndices, static_cast<int32_t>(validKvTokens), all);
        if constexpr (HasMask) {
            Cast<float, int32_t, MXFP4_CAST_ZERO>(causalRows, chunkIndices, all);
            Muls(causalRows, causalRows, causalRowStep, all);
            Adds(causalRows, causalRows, causalFirstRow, all);
        }
        for (uint16_t row = 0; row < alignedRows; row += 4) {
            LoadAlign(score0, scoreUb + (row + 0) * ChunkN + firstToken);
            LoadAlign(score1, scoreUb + (row + 1) * ChunkN + firstToken);
            LoadAlign(score2, scoreUb + (row + 2) * ChunkN + firstToken);
            LoadAlign(score3, scoreUb + (row + 3) * ChunkN + firstToken);
            if constexpr (HasMask) {
                Duplicate(queryIndex0, static_cast<int32_t>(row + 0));
                Duplicate(queryIndex1, static_cast<int32_t>(row + 1));
                Duplicate(queryIndex2, static_cast<int32_t>(row + 2));
                Duplicate(queryIndex3, static_cast<int32_t>(row + 3));
                Cast<float, int32_t, MXFP4_CAST_ZERO>(first0, queryIndex0, all);
                Cast<float, int32_t, MXFP4_CAST_ZERO>(first1, queryIndex1, all);
                Cast<float, int32_t, MXFP4_CAST_ZERO>(first2, queryIndex2, all);
                Cast<float, int32_t, MXFP4_CAST_ZERO>(first3, queryIndex3, all);
                Compare<float, CMPMODE::GE>(visible0, first0, causalRows, all);
                Compare<float, CMPMODE::GE>(visible1, first1, causalRows, all);
                Compare<float, CMPMODE::GE>(visible2, first2, causalRows, all);
                Compare<float, CMPMODE::GE>(visible3, first3, causalRows, all);
                MaskAnd(visible0, visible0, validKv, all);
                MaskAnd(visible1, visible1, validKv, all);
                MaskAnd(visible2, visible2, validKv, all);
                MaskAnd(visible3, visible3, validKv, all);
            } else {
                visible0 = validKv;
                visible1 = validKv;
                visible2 = validKv;
                visible3 = validKv;
            }
            if constexpr (!ScorePreScaled) {
                Muls(score0, score0, scoreScale, all);
                Muls(score1, score1, scoreScale, all);
                Muls(score2, score2, scoreScale, all);
                Muls(score3, score3, scoreScale, all);
            }
            Select(score0, score0, invalid, visible0);
            Select(score1, score1, invalid, visible1);
            Select(score2, score2, invalid, visible2);
            Select(score3, score3, invalid, visible3);
            Reduce<ReduceType::MAX, float, float>(first0, score0, firstHalf);
            Reduce<ReduceType::MAX, float, float>(first1, score1, firstHalf);
            Reduce<ReduceType::MAX, float, float>(first2, score2, firstHalf);
            Reduce<ReduceType::MAX, float, float>(first3, score3, firstHalf);
            Reduce<ReduceType::MAX, float, float>(second0, score0, secondHalf);
            Reduce<ReduceType::MAX, float, float>(second1, score1, secondHalf);
            Reduce<ReduceType::MAX, float, float>(second2, score2, secondHalf);
            Reduce<ReduceType::MAX, float, float>(second3, score3, secondHalf);
            Duplicate(first0, first0, all);
            Duplicate(first1, first1, all);
            Duplicate(first2, first2, all);
            Duplicate(first3, first3, all);
            Duplicate(second0, second0, all);
            Duplicate(second1, second1, all);
            Duplicate(second2, second2, all);
            Duplicate(second3, second3, all);
            Select(exponent0, first0, second0, firstHalf);
            Select(exponent1, first1, second1, firstHalf);
            Select(exponent2, first2, second2, firstHalf);
            Select(exponent3, first3, second3, firstHalf);
            Compares<float, CMPMODE::GT>(groupValid0, exponent0, INVALID_GROUP_EXPONENT, all);
            Compares<float, CMPMODE::GT>(groupValid1, exponent1, INVALID_GROUP_EXPONENT, all);
            Compares<float, CMPMODE::GT>(groupValid2, exponent2, INVALID_GROUP_EXPONENT, all);
            Compares<float, CMPMODE::GT>(groupValid3, exponent3, INVALID_GROUP_EXPONENT, all);
            Muls(exponent0, exponent0, MXFP4_INV_LN2, groupValid0);
            Muls(exponent1, exponent1, MXFP4_INV_LN2, groupValid1);
            Muls(exponent2, exponent2, MXFP4_INV_LN2, groupValid2);
            Muls(exponent3, exponent3, MXFP4_INV_LN2, groupValid3);
            if constexpr (QUANT_SCHEMA != QuantSchema::OCP_QUANT) {
                Adds(exponent0, exponent0, QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT ? MXFP4_NEG_LOG2_CX_DELTA : MXFP4_NEG_LOG2_CX, groupValid0);
                Adds(exponent1, exponent1, QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT ? MXFP4_NEG_LOG2_CX_DELTA : MXFP4_NEG_LOG2_CX, groupValid1);
                Adds(exponent2, exponent2, QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT ? MXFP4_NEG_LOG2_CX_DELTA : MXFP4_NEG_LOG2_CX, groupValid2);
                Adds(exponent3, exponent3, QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT ? MXFP4_NEG_LOG2_CX_DELTA : MXFP4_NEG_LOG2_CX, groupValid3);
                Truncate<float, RoundMode::CAST_CEIL>(exponent0, exponent0, groupValid0);
                Truncate<float, RoundMode::CAST_CEIL>(exponent1, exponent1, groupValid1);
                Truncate<float, RoundMode::CAST_CEIL>(exponent2, exponent2, groupValid2);
                Truncate<float, RoundMode::CAST_CEIL>(exponent3, exponent3, groupValid3);
            } else {
                Truncate<float, RoundMode::CAST_FLOOR>(exponent0, exponent0, groupValid0);
                Truncate<float, RoundMode::CAST_FLOOR>(exponent1, exponent1, groupValid1);
                Truncate<float, RoundMode::CAST_FLOOR>(exponent2, exponent2, groupValid2);
                Truncate<float, RoundMode::CAST_FLOOR>(exponent3, exponent3, groupValid3);
            }
            Select(exponent0, exponent0, invalid, groupValid0);
            Select(exponent1, exponent1, invalid, groupValid1);
            Select(exponent2, exponent2, invalid, groupValid2);
            Select(exponent3, exponent3, invalid, groupValid3);
            Gather(packedExponent0, exponent0, pairIndices);
            Gather(packedExponent1, exponent1, pairIndices);
            Gather(packedExponent2, exponent2, pairIndices);
            Gather(packedExponent3, exponent3, pairIndices);
            Scatter(groupMaxUb + (row + 0) * GROUPS_PER_TASK + firstGroup, packedExponent0, storeOffsets, pairMask);
            Scatter(groupMaxUb + (row + 1) * GROUPS_PER_TASK + firstGroup, packedExponent1, storeOffsets, pairMask);
            Scatter(groupMaxUb + (row + 2) * GROUPS_PER_TASK + firstGroup, packedExponent2, storeOffsets, pairMask);
            Scatter(groupMaxUb + (row + 3) * GROUPS_PER_TASK + firstGroup, packedExponent3, storeOffsets, pairMask);




            if constexpr (QUANT_SCHEMA != QuantSchema::OCP_QUANT) {
                Muls(exponent0, exponent0, MXFP4_LN2, groupValid0);
                Muls(exponent1, exponent1, MXFP4_LN2, groupValid1);
                Muls(exponent2, exponent2, MXFP4_LN2, groupValid2);
                Muls(exponent3, exponent3, MXFP4_LN2, groupValid3);
                if constexpr (QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT) {
                    Adds(exponent0, exponent0, MXFP4_NEG_LN_DELTA, groupValid0);
                    Adds(exponent1, exponent1, MXFP4_NEG_LN_DELTA, groupValid1);
                    Adds(exponent2, exponent2, MXFP4_NEG_LN_DELTA, groupValid2);
                    Adds(exponent3, exponent3, MXFP4_NEG_LN_DELTA, groupValid3);
                }
            } else {
                Adds(exponent0, exponent0, MXFP4_NEG_TWO_VALUE, groupValid0);
                Adds(exponent1, exponent1, MXFP4_NEG_TWO_VALUE, groupValid1);
                Adds(exponent2, exponent2, MXFP4_NEG_TWO_VALUE, groupValid2);
                Adds(exponent3, exponent3, MXFP4_NEG_TWO_VALUE, groupValid3);
                Muls(exponent0, exponent0, MXFP4_LN2, groupValid0);
                Muls(exponent1, exponent1, MXFP4_LN2, groupValid1);
                Muls(exponent2, exponent2, MXFP4_LN2, groupValid2);
                Muls(exponent3, exponent3, MXFP4_LN2, groupValid3);
            }
            MaskAnd(expValid0, groupValid0, visible0, all);
            MaskAnd(expValid1, groupValid1, visible1, all);
            MaskAnd(expValid2, groupValid2, visible2, all);
            MaskAnd(expValid3, groupValid3, visible3, all);
            AscendC::Reg::ExpSub(score0, score0, exponent0, expValid0);
            AscendC::Reg::ExpSub(score1, score1, exponent1, expValid1);
            AscendC::Reg::ExpSub(score2, score2, exponent2, expValid2);
            AscendC::Reg::ExpSub(score3, score3, exponent3, expValid3);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(bf0, score0, all);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(bf1, score1, all);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(bf2, score2, all);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(bf3, score3, all);
            DeInterleave(compactBf0, bf0, bf0, bf0);
            DeInterleave(compactBf1, bf1, bf1, bf1);
            DeInterleave(compactBf2, bf2, bf2, bf2);
            DeInterleave(compactBf3, bf3, bf3, bf3);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(fp0, compactBf0, allHalf);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(fp1, compactBf1, allHalf);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(fp2, compactBf2, allHalf);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(fp3, compactBf3, allHalf);
            // PACK4 extracts one byte per FP4 32-bit lane; the low 32 lanes write exactly 32B.
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(
                reinterpret_cast<__ubuf__ int8_t*>(packedBlock + (row + 0) * 32),
                reinterpret_cast<RegTensor<int8_t>&>(fp0), firstHalf);
            // PACK4 extracts one byte per FP4 32-bit lane; the low 32 lanes write exactly 32B.
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(
                reinterpret_cast<__ubuf__ int8_t*>(packedBlock + (row + 1) * 32),
                reinterpret_cast<RegTensor<int8_t>&>(fp1), firstHalf);
            // PACK4 extracts one byte per FP4 32-bit lane; the low 32 lanes write exactly 32B.
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(
                reinterpret_cast<__ubuf__ int8_t*>(packedBlock + (row + 2) * 32),
                reinterpret_cast<RegTensor<int8_t>&>(fp2), firstHalf);
            // PACK4 extracts one byte per FP4 32-bit lane; the low 32 lanes write exactly 32B.
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(
                reinterpret_cast<__ubuf__ int8_t*>(packedBlock + (row + 3) * 32),
                reinterpret_cast<RegTensor<int8_t>&>(fp3), firstHalf);
        }
        for (uint16_t row = alignedRows; row < validQueryRows; ++row) {
            LoadAlign(score0, scoreUb + (row + 0) * ChunkN + firstToken);
            if constexpr (HasMask) {
                Duplicate(queryIndex0, static_cast<int32_t>(row + 0));
                Cast<float, int32_t, MXFP4_CAST_ZERO>(first0, queryIndex0, all);
                Compare<float, CMPMODE::GE>(visible0, first0, causalRows, all);
                MaskAnd(visible0, visible0, validKv, all);
            } else {
                visible0 = validKv;
            }
            if constexpr (!ScorePreScaled) {
                Muls(score0, score0, scoreScale, all);
            }
            Select(score0, score0, invalid, visible0);
            Reduce<ReduceType::MAX, float, float>(first0, score0, firstHalf);
            Reduce<ReduceType::MAX, float, float>(second0, score0, secondHalf);
            Duplicate(first0, first0, all);
            Duplicate(second0, second0, all);
            Select(exponent0, first0, second0, firstHalf);
            Compares<float, CMPMODE::GT>(groupValid0, exponent0, INVALID_GROUP_EXPONENT, all);
            Muls(exponent0, exponent0, MXFP4_INV_LN2, groupValid0);
            if constexpr (QUANT_SCHEMA != QuantSchema::OCP_QUANT) {
                Adds(exponent0, exponent0, QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT ? MXFP4_NEG_LOG2_CX_DELTA : MXFP4_NEG_LOG2_CX, groupValid0);
                Truncate<float, RoundMode::CAST_CEIL>(exponent0, exponent0, groupValid0);
            } else {
                Truncate<float, RoundMode::CAST_FLOOR>(exponent0, exponent0, groupValid0);
            }
            Select(exponent0, exponent0, invalid, groupValid0);
            Gather(packedExponent0, exponent0, pairIndices);
            Scatter(groupMaxUb + (row + 0) * GROUPS_PER_TASK + firstGroup, packedExponent0, storeOffsets, pairMask);

            if constexpr (QUANT_SCHEMA != QuantSchema::OCP_QUANT) {
                Muls(exponent0, exponent0, MXFP4_LN2, groupValid0);
                if constexpr (QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT) {
                    Adds(exponent0, exponent0, MXFP4_NEG_LN_DELTA, groupValid0);
                }
            } else {
                Adds(exponent0, exponent0, MXFP4_NEG_TWO_VALUE, groupValid0);
                Muls(exponent0, exponent0, MXFP4_LN2, groupValid0);
            }
            MaskAnd(expValid0, groupValid0, visible0, all);
            AscendC::Reg::ExpSub(score0, score0, exponent0, expValid0);
            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(bf0, score0, all);
            DeInterleave(compactBf0, bf0, bf0, bf0);
            Cast<float4_e2m1x2_t, bfloat16_t, MXFP4_CAST_ZERO>(fp0, compactBf0, allHalf);
            // PACK4 extracts one byte per FP4 32-bit lane; the low 32 lanes write exactly 32B.
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(
                reinterpret_cast<__ubuf__ int8_t*>(packedBlock + (row + 0) * 32),
                reinterpret_cast<RegTensor<int8_t>&>(fp0), firstHalf);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    // Each lane now represents one Q row. This replaces a reduction and two
    // scalar UB accesses for every row/S64 pair with one batched merge.
    RegTensor<uint32_t> rowOffsets;
    RegTensor<float> rowMaximum, chunkExponent;
    uint32_t rowCount = validQueryRows;
    MaskReg validRows = UpdateMask<float>(rowCount);
    Arange(reinterpret_cast<RegTensor<int32_t>&>(rowOffsets), static_cast<int32_t>(0));
    Muls(rowOffsets, rowOffsets, static_cast<uint32_t>(GROUPS_PER_TASK), all);
    LoadAlign(rowMaximum, globalMaxUb);
    for (uint16_t group = 0; group < ChunkN / QUANT_GROUP_SIZE; ++group) {
        Gather(chunkExponent, groupMaxUb + chunkKvOffset / QUANT_GROUP_SIZE + group, rowOffsets, validRows);
        Max<float, MaskMergeMode::MERGING>(rowMaximum, rowMaximum, chunkExponent, validRows);
    }
    StoreAlign(globalMaxUb, rowMaximum, all);

}

// 截断发生在相对位置计算之后，长序列的绝对 token 下标不会转换为 float。
template <uint32_t ChunkN>
__aicore__ inline float GetCausalFirstVisibleRow(const SoftmaxFp32Mxfp4Params& params)
{
    int64_t firstRow = (static_cast<int64_t>(params.kvStart) - params.causalOffset) * params.qGroupSize -
        static_cast<int64_t>(params.qGs1Start);
    const int64_t minimumRow = -static_cast<int64_t>(ChunkN) * params.qGroupSize;
    firstRow = firstRow < minimumRow ? minimumRow : firstRow;
    firstRow = firstRow > VEC1_ROWS ? VEC1_ROWS + 1 : firstRow;
    return static_cast<float>(firstRow);
}

template <uint32_t ChunkN, uint32_t TaskN, bool HasMask, QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT, bool ScorePreScaled = false>
__aicore__ inline void SoftmaxDnChunkFp32(const LocalTensor<uint8_t>& packedProbability,
    const LocalTensor<float>& taskGroupExponent, const LocalTensor<float>& runningGroupMaximum,
    const LocalTensor<float>& score, const LocalTensor<uint8_t>& byteIndices,
    const LocalTensor<uint16_t>& halfwordIndices, const SoftmaxFp32Mxfp4Params& params)
{
    // 空 chunk 不进入 VF，避免 GROUP_COUNT - 1 下溢。
    if (params.validN == 0) {
        return;
    }
    if constexpr (HasMask) {
        const int64_t lastKey = static_cast<int64_t>(params.kvStart) + params.validN - 1;
        const int64_t firstQuery = static_cast<int64_t>(params.qGs1Start) + params.causalOffset;
        if (firstQuery >= lastKey) {
            // Every real key is visible to every Q row; retain the KV-tail mask only.
            SoftmaxDnChunkFp32<ChunkN, TaskN, false, QUANT_SCHEMA, ScorePreScaled>(packedProbability,
                taskGroupExponent, runningGroupMaximum, score, byteIndices, halfwordIndices, params);
            return;
        }
    }
    // DN 中 qGroupSize 为 1；先用有符号 64 位计算相对位置，再截断至等价 mask 范围。
    int64_t causalMaskLine = static_cast<int64_t>(params.qGs1Start) + params.causalOffset -
        static_cast<int64_t>(params.kvStart);
    causalMaskLine = causalMaskLine < -static_cast<int64_t>(VEC1_ROWS) ?
        -static_cast<int64_t>(VEC1_ROWS) : causalMaskLine;
    causalMaskLine = causalMaskLine > static_cast<int64_t>(ChunkN) ?
        static_cast<int64_t>(ChunkN) : causalMaskLine;
    const uint16_t alignedKv32 = static_cast<uint16_t>((params.validN + 31U) / 32U * 32U);
    const uint16_t alignedKv64 = static_cast<uint16_t>((params.validN + 63U) / 64U * 64U);
    // P 按 KV 连续，局部指数按 [S32组,Q64] 连续；全局 max 跨 chunk 共用。
    __ubuf__ uint8_t* chunkProbability = (__ubuf__ uint8_t*)packedProbability.GetPhyAddr() +
        params.s2Offset * (VEC1_ROWS / 2);
    __ubuf__ float* chunkExponent = (__ubuf__ float*)taskGroupExponent.GetPhyAddr() +
        params.s2Offset / QUANT_GROUP_SIZE * VEC1_ROWS;
    if constexpr (!HasMask && TaskN == 256) {
        if (params.validN == ChunkN) {
            SoftmaxDnResidentEightVf<ChunkN, QUANT_SCHEMA, ScorePreScaled>(
                chunkProbability, chunkExponent, (__ubuf__ float*)runningGroupMaximum.GetPhyAddr(),
                (__ubuf__ float*)score.GetPhyAddr(), params.scoreScale);
            return;
        }
    }
    SoftmaxQuantizeDnChunkFp32Vf<ChunkN, TaskN, HasMask, QUANT_SCHEMA, false, ScorePreScaled>(
        chunkProbability, chunkExponent, (__ubuf__ float*)runningGroupMaximum.GetPhyAddr(),
        (__ubuf__ float*)score.GetPhyAddr(),
        (__ubuf__ uint8_t*)byteIndices.GetPhyAddr(), (__ubuf__ uint16_t*)halfwordIndices.GetPhyAddr(),
        params.scoreScale, static_cast<int32_t>(causalMaskLine), params.validRows, params.validN,
        alignedKv32, alignedKv64);
}

template <uint32_t ChunkN, uint32_t TaskN, bool HasMask, QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT, bool ScorePreScaled = false>
__aicore__ inline void SoftmaxNdChunkFp32(const LocalTensor<uint8_t>& packedProbability,
    const LocalTensor<float>& taskGroupExponent, const LocalTensor<float>& runningGroupMaximum,
    const LocalTensor<float>& score, const LocalTensor<uint8_t>& byteIndices,
    const LocalTensor<uint16_t>& halfwordIndices, const SoftmaxFp32Mxfp4Params& params)
{
    if constexpr (HasMask) {
        const int64_t lastKey = static_cast<int64_t>(params.kvStart) + params.validN - 1;
        const int64_t firstQuery = static_cast<int64_t>(params.qGs1Start / params.qGroupSize) +
            params.causalOffset;
        if (params.validN != 0 && firstQuery >= lastKey) { // 在第一个query能看到所有的kv时，在causal也不需要mask
            SoftmaxNdChunkFp32<ChunkN, TaskN, false, QUANT_SCHEMA, ScorePreScaled>(packedProbability,
                taskGroupExponent, runningGroupMaximum, score, byteIndices, halfwordIndices, params);
            return;
        }
    }
    SoftmaxQuantizeNdChunkFp32Vf<ChunkN, TaskN, HasMask, QUANT_SCHEMA, ScorePreScaled>(
        (__ubuf__ uint8_t*)packedProbability.GetPhyAddr(), (__ubuf__ float*)taskGroupExponent.GetPhyAddr(),
        (__ubuf__ float*)runningGroupMaximum.GetPhyAddr(), (__ubuf__ float*)score.GetPhyAddr(),
        (__ubuf__ uint8_t*)byteIndices.GetPhyAddr(), (__ubuf__ uint16_t*)halfwordIndices.GetPhyAddr(),
        params.scoreScale, params.validRows, params.validN, params.s2Offset,
        GetCausalFirstVisibleRow<ChunkN>(params), static_cast<float>(params.qGroupSize));
}
} // namespace Mxfp4Api
#endif
