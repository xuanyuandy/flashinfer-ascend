/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QFA_VF_OUTPUT_UPDATE_FP32_MXFP4_H_
#define QFA_VF_OUTPUT_UPDATE_FP32_MXFP4_H_
#include "vf_softmax_fp32_mxfp4.h"
#include "../quant_flash_attn_common_def.h"

namespace Mxfp4Api {
using BaseApi::QuantSchema;


// historyOutput/groupResult: FP32 [Dv,64]；groupSum/historySum/multiplier: FP32 [64]。
// 首组直接保存本组 PV/rowsum；后续组先乘历史缩放系数再累加，尚不归一化。
// validRows 为当前 AIV 的有效 Q 行数；空组的 PV 按零处理。
template <bool UpdateHistory, uint16_t ValueDimension = 128>
__simd_vf__ inline void processUpdateDnVF(__ubuf__ float *historyOutputUb, __ubuf__ float *groupResultUb, __ubuf__ float *historyMultiplierUb,
                                 __ubuf__ float *historySumUb, __ubuf__ float *groupSumUb, uint32_t validRows)
{
    // 更新 mm2
    MaskReg preg_all32 = CreateMask<uint32_t, MaskPattern::ALL>();
    static_assert(ValueDimension % 4 == 0, "DN update requires complete four-channel blocks");
    MaskReg preg_valid_rows = UpdateMask<float>(validRows);
    MaskReg preg_nonempty_rows;
    RegTensor<float> vreg_zero, vreg_current_sum;
    Duplicate(vreg_zero, 0.0F);
    LoadAlign(vreg_current_sum, groupSumUb);
    Select(vreg_current_sum, vreg_current_sum, vreg_zero, preg_valid_rows);
    Compares<float, CMPMODE::GT>(preg_nonempty_rows, vreg_current_sum, 0.0F, preg_valid_rows);
    if constexpr (UpdateHistory) {
        RegTensor<float> vreg_multiplier, vreg_history_sum;
        LoadAlign(vreg_multiplier, (__ubuf__ float *)historyMultiplierUb);
        LoadAlign(vreg_history_sum, (__ubuf__ float *)historySumUb);
        MulDstAdd(vreg_history_sum, vreg_multiplier, vreg_current_sum, preg_all32);

        StoreAlign((__ubuf__ float *)historySumUb, vreg_history_sum, preg_all32);

        for (uint16_t channel = 0; channel < ValueDimension; channel += 4) {
            RegTensor<float> vreg_pv1, vreg_pv2, vreg_pv3, vreg_pv4;
            RegTensor<float> vreg_out1, vreg_out2, vreg_out3, vreg_out4;
            LoadAlign(vreg_out1, (__ubuf__ float *)historyOutputUb + (channel + 0) * 64);
            LoadAlign(vreg_out2, (__ubuf__ float *)historyOutputUb + (channel + 1) * 64);
            LoadAlign(vreg_out3, (__ubuf__ float *)historyOutputUb + (channel + 2) * 64);
            LoadAlign(vreg_out4, (__ubuf__ float *)historyOutputUb + (channel + 3) * 64);


            LoadAlign(vreg_pv1, (__ubuf__ float *)groupResultUb + (channel + 0) * 64);
            Select(vreg_pv1, vreg_pv1, vreg_zero, preg_nonempty_rows);
            LoadAlign(vreg_pv2, (__ubuf__ float *)groupResultUb + (channel + 1) * 64);
            Select(vreg_pv2, vreg_pv2, vreg_zero, preg_nonempty_rows);
            LoadAlign(vreg_pv3, (__ubuf__ float *)groupResultUb + (channel + 2) * 64);
            Select(vreg_pv3, vreg_pv3, vreg_zero, preg_nonempty_rows);
            LoadAlign(vreg_pv4, (__ubuf__ float *)groupResultUb + (channel + 3) * 64);
            Select(vreg_pv4, vreg_pv4, vreg_zero, preg_nonempty_rows);

            MulDstAdd(vreg_out1, vreg_multiplier, vreg_pv1, preg_all32);
            MulDstAdd(vreg_out2, vreg_multiplier, vreg_pv2, preg_all32);
            MulDstAdd(vreg_out3, vreg_multiplier, vreg_pv3, preg_all32);
            MulDstAdd(vreg_out4, vreg_multiplier, vreg_pv4, preg_all32);

            StoreAlign((__ubuf__ float *)historyOutputUb + (channel + 0) * 64, vreg_out1, preg_all32);
            StoreAlign((__ubuf__ float *)historyOutputUb + (channel + 1) * 64, vreg_out2, preg_all32);
            StoreAlign((__ubuf__ float *)historyOutputUb + (channel + 2) * 64, vreg_out3, preg_all32);
            StoreAlign((__ubuf__ float *)historyOutputUb + (channel + 3) * 64, vreg_out4, preg_all32);
        }
    } else {
        StoreAlign((__ubuf__ float *)historySumUb, vreg_current_sum, preg_all32);

        for (uint16_t channel = 0; channel < ValueDimension; channel += 4) {
            RegTensor<float> vreg_out1, vreg_out2, vreg_out3, vreg_out4;
            LoadAlign(vreg_out1, (__ubuf__ float *)groupResultUb + (channel + 0) * 64);
            Select(vreg_out1, vreg_out1, vreg_zero, preg_nonempty_rows);
            LoadAlign(vreg_out2, (__ubuf__ float *)groupResultUb + (channel + 1) * 64);
            Select(vreg_out2, vreg_out2, vreg_zero, preg_nonempty_rows);
            LoadAlign(vreg_out3, (__ubuf__ float *)groupResultUb + (channel + 2) * 64);
            Select(vreg_out3, vreg_out3, vreg_zero, preg_nonempty_rows);
            LoadAlign(vreg_out4, (__ubuf__ float *)groupResultUb + (channel + 3) * 64);
            Select(vreg_out4, vreg_out4, vreg_zero, preg_nonempty_rows);

            StoreAlign((__ubuf__ float *)historyOutputUb + (channel + 0) * 64, vreg_out1, preg_all32);
            StoreAlign((__ubuf__ float *)historyOutputUb + (channel + 1) * 64, vreg_out2, preg_all32);
            StoreAlign((__ubuf__ float *)historyOutputUb + (channel + 2) * 64, vreg_out3, preg_all32);
            StoreAlign((__ubuf__ float *)historyOutputUb + (channel + 3) * 64, vreg_out4, preg_all32);
        }
    }
}

template <bool UpdateHistory, uint16_t ValueDimension = 128>
__aicore__ inline void processUpdateDn(const LocalTensor<float> &historyOutputUb, const LocalTensor<float> &groupResultUb,
                                     const LocalTensor<float> &historyMultiplierUb, const LocalTensor<float> &historySumUb,
                                     const LocalTensor<float> &groupSumUb, uint32_t validRows)
{
    __ubuf__ float *historyOutputPtr = (__ubuf__ float *)historyOutputUb.GetPhyAddr();
    __ubuf__ float *groupResultPtr = (__ubuf__ float *)groupResultUb.GetPhyAddr();
    __ubuf__ float *multiplierPtr = (__ubuf__ float *)historyMultiplierUb.GetPhyAddr();
    __ubuf__ float *historySumPtr = (__ubuf__ float *)historySumUb.GetPhyAddr();
    __ubuf__ float *groupSumPtr = (__ubuf__ float *)groupSumUb.GetPhyAddr();

    processUpdateDnVF<UpdateHistory, ValueDimension>(historyOutputPtr, groupResultPtr, multiplierPtr,
        historySumPtr, groupSumPtr, validRows);
}

// 输入 historyOutput: FP32 [Dv,64]，historySum: FP32 [64]，两者不被覆盖。
// BF16 输出为紧凑 [64,Dv]（没有实验版额外16列）；支持 Dv=128/256，空行输出零。
// FD 的 float 特化输出 [64,16]，channelOffset 指定输入通道起点，由调用方分块写入 workspace。
template <uint16_t ValueDimension = 128, typename OutputType = bfloat16_t, bool Normalize = true>
__simd_vf__ inline void processOutDnVF(__ubuf__ OutputType *outputUb, __ubuf__ float *historyOutputUb, __ubuf__ float *historySumUb, uint32_t channelOffset = 0)
{
    MaskReg preg_all32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    RegTensor<float> vreg_inverse_row_sum, vreg_ones;
    Duplicate(vreg_ones, (float)1.0);
    LoadAlign(vreg_inverse_row_sum, (__ubuf__ float *)historySumUb);
    static_assert(ValueDimension % 16 == 0, "DN output requires complete sixteen-channel blocks");
    MaskReg preg_nonempty_rows;
    RegTensor<float> vreg_zero;
    Duplicate(vreg_zero, 0.0F);
    Compares<float, CMPMODE::GT>(preg_nonempty_rows, vreg_inverse_row_sum, 0.0F, preg_all32);
    Select(vreg_inverse_row_sum, vreg_inverse_row_sum, vreg_ones, preg_nonempty_rows);
    static_assert(Normalize || IsSameType<OutputType, float>::value,
        "Only the raw FP32 FD transpose may skip normalization");
    if constexpr (Normalize) {
        // 每个 Q64 输出块只计算一次倒数，所有通道复用；FD 原始 PV 路径不做归一化。
        Div<float, &MXFP4_OUTPUT_DIV>(vreg_inverse_row_sum, vreg_ones, vreg_inverse_row_sum, preg_all32);
    }

    if constexpr (IsSameType<OutputType, float>::value) {
        // FD 保留 FP32：每次输出 [64,16]，复用现有 scratch，调用方按通道拼回 GM。
        RegTensor<uint32_t> vreg_row_indices, vreg_output_offsets;
        Arange(reinterpret_cast<RegTensor<int32_t>&>(vreg_row_indices), static_cast<int32_t>(0));
        Muls(vreg_output_offsets, vreg_row_indices, static_cast<uint32_t>(16), preg_all32);
        for (uint16_t channel = 0; channel < 16; ++channel) {
            RegTensor<float> vreg_output;
            LoadAlign(vreg_output, historyOutputUb + (channelOffset + channel) * 64);
            if constexpr (Normalize) {
                Mul(vreg_output, vreg_output, vreg_inverse_row_sum, preg_all32);
            }
            Select(vreg_output, vreg_output, vreg_zero, preg_nonempty_rows);
            Scatter(outputUb + channel, vreg_output, vreg_output_offsets, preg_all32);
        }
    } else {
        static_assert(IsSameType<OutputType, bfloat16_t>::value, "DN final output must be BF16");

        for (uint16_t channelBlock = 0; channelBlock < ValueDimension / 16; channelBlock++) {
            RegTensor<float> vreg_src1, vreg_src2, vreg_src3, vreg_src4, vreg_src5, vreg_src6, vreg_src7, vreg_src8;
            RegTensor<float> vreg_src9, vreg_src10, vreg_src11, vreg_src12, vreg_src13, vreg_src14, vreg_src15, vreg_src16;
            RegTensor<bfloat16_t> vreg_bf16_1, vreg_bf16_2;

            LoadAlign(vreg_src1, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 0);
            LoadAlign(vreg_src2, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 1);
            LoadAlign(vreg_src3, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 2);
            LoadAlign(vreg_src4, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 3);
            LoadAlign(vreg_src5, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 4);
            LoadAlign(vreg_src6, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 5);
            LoadAlign(vreg_src7, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 6);
            LoadAlign(vreg_src8, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 7);
            LoadAlign(vreg_src9, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 8);
            LoadAlign(vreg_src10, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 9);
            LoadAlign(vreg_src11, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 10);
            LoadAlign(vreg_src12, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 11);
            LoadAlign(vreg_src13, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 12);
            LoadAlign(vreg_src14, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 13);
            LoadAlign(vreg_src15, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 14);
            LoadAlign(vreg_src16, (__ubuf__ float *)historyOutputUb + channelBlock * 64 * 16 + 64 * 15);

            Mul(vreg_src1, vreg_src1, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src1, vreg_src1, vreg_zero, preg_nonempty_rows);
            Mul(vreg_src2, vreg_src2, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src2, vreg_src2, vreg_zero, preg_nonempty_rows);
            Mul(vreg_src3, vreg_src3, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src3, vreg_src3, vreg_zero, preg_nonempty_rows);
            Mul(vreg_src4, vreg_src4, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src4, vreg_src4, vreg_zero, preg_nonempty_rows);
            Mul(vreg_src5, vreg_src5, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src5, vreg_src5, vreg_zero, preg_nonempty_rows);
            Mul(vreg_src6, vreg_src6, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src6, vreg_src6, vreg_zero, preg_nonempty_rows);
            Mul(vreg_src7, vreg_src7, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src7, vreg_src7, vreg_zero, preg_nonempty_rows);
            Mul(vreg_src8, vreg_src8, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src8, vreg_src8, vreg_zero, preg_nonempty_rows);
            Mul(vreg_src9, vreg_src9, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src9, vreg_src9, vreg_zero, preg_nonempty_rows);
            Mul(vreg_src10, vreg_src10, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src10, vreg_src10, vreg_zero, preg_nonempty_rows);
            Mul(vreg_src11, vreg_src11, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src11, vreg_src11, vreg_zero, preg_nonempty_rows);
            Mul(vreg_src12, vreg_src12, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src12, vreg_src12, vreg_zero, preg_nonempty_rows);
            Mul(vreg_src13, vreg_src13, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src13, vreg_src13, vreg_zero, preg_nonempty_rows);
            Mul(vreg_src14, vreg_src14, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src14, vreg_src14, vreg_zero, preg_nonempty_rows);
            Mul(vreg_src15, vreg_src15, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src15, vreg_src15, vreg_zero, preg_nonempty_rows);
            Mul(vreg_src16, vreg_src16, vreg_inverse_row_sum, preg_all32);
            Select(vreg_src16, vreg_src16, vreg_zero, preg_nonempty_rows);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(vreg_bf16_1, vreg_src1, preg_all32);
            Cast<bfloat16_t, float, MXFP4_CAST_ONE>(vreg_bf16_2, vreg_src2, preg_all32);
            Or((RegTensor<uint16_t> &)vreg_src1, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
               preg_all16);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(vreg_bf16_1, vreg_src3, preg_all32);
            Cast<bfloat16_t, float, MXFP4_CAST_ONE>(vreg_bf16_2, vreg_src4, preg_all32);
            Or((RegTensor<uint16_t> &)vreg_src2, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
               preg_all16);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(vreg_bf16_1, vreg_src5, preg_all32);
            Cast<bfloat16_t, float, MXFP4_CAST_ONE>(vreg_bf16_2, vreg_src6, preg_all32);
            Or((RegTensor<uint16_t> &)vreg_src3, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
               preg_all16);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(vreg_bf16_1, vreg_src7, preg_all32);
            Cast<bfloat16_t, float, MXFP4_CAST_ONE>(vreg_bf16_2, vreg_src8, preg_all32);
            Or((RegTensor<uint16_t> &)vreg_src4, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
               preg_all16);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(vreg_bf16_1, vreg_src9, preg_all32);
            Cast<bfloat16_t, float, MXFP4_CAST_ONE>(vreg_bf16_2, vreg_src10, preg_all32);
            Or((RegTensor<uint16_t> &)vreg_src5, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
               preg_all16);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(vreg_bf16_1, vreg_src11, preg_all32);
            Cast<bfloat16_t, float, MXFP4_CAST_ONE>(vreg_bf16_2, vreg_src12, preg_all32);
            Or((RegTensor<uint16_t> &)vreg_src6, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
               preg_all16);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(vreg_bf16_1, vreg_src13, preg_all32);
            Cast<bfloat16_t, float, MXFP4_CAST_ONE>(vreg_bf16_2, vreg_src14, preg_all32);
            Or((RegTensor<uint16_t> &)vreg_src7, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
               preg_all16);

            Cast<bfloat16_t, float, MXFP4_CAST_ZERO>(vreg_bf16_1, vreg_src15, preg_all32);
            Cast<bfloat16_t, float, MXFP4_CAST_ONE>(vreg_bf16_2, vreg_src16, preg_all32);
            Or((RegTensor<uint16_t> &)vreg_src8, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
               preg_all16);

            Interleave(vreg_src1, vreg_src5, vreg_src1, vreg_src5);
            Interleave(vreg_src2, vreg_src6, vreg_src2, vreg_src6);
            Interleave(vreg_src3, vreg_src7, vreg_src3, vreg_src7);
            Interleave(vreg_src4, vreg_src8, vreg_src4, vreg_src8);

            Interleave(vreg_src1, vreg_src3, vreg_src1, vreg_src3);
            Interleave(vreg_src5, vreg_src7, vreg_src5, vreg_src7);
            Interleave(vreg_src2, vreg_src4, vreg_src2, vreg_src4);
            Interleave(vreg_src6, vreg_src8, vreg_src6, vreg_src8);

            Interleave(vreg_src1, vreg_src2, vreg_src1, vreg_src2);
            Interleave(vreg_src3, vreg_src4, vreg_src3, vreg_src4);
            Interleave(vreg_src5, vreg_src6, vreg_src5, vreg_src6);
            Interleave(vreg_src7, vreg_src8, vreg_src7, vreg_src8);

            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float *)outputUb + 8 * channelBlock + (ValueDimension / 16) * 8 * 8 * 0, (RegTensor<float> &)vreg_src1, ValueDimension / 16,
                preg_all32);
            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float *)outputUb + 8 * channelBlock + (ValueDimension / 16) * 8 * 8 * 1, (RegTensor<float> &)vreg_src2, ValueDimension / 16,
                preg_all32);
            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float *)outputUb + 8 * channelBlock + (ValueDimension / 16) * 8 * 8 * 2, (RegTensor<float> &)vreg_src3, ValueDimension / 16,
                preg_all32);
            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float *)outputUb + 8 * channelBlock + (ValueDimension / 16) * 8 * 8 * 3, (RegTensor<float> &)vreg_src4, ValueDimension / 16,
                preg_all32);
            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float *)outputUb + 8 * channelBlock + (ValueDimension / 16) * 8 * 8 * 4, (RegTensor<float> &)vreg_src5, ValueDimension / 16,
                preg_all32);
            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float *)outputUb + 8 * channelBlock + (ValueDimension / 16) * 8 * 8 * 5, (RegTensor<float> &)vreg_src6, ValueDimension / 16,
                preg_all32);
            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float *)outputUb + 8 * channelBlock + (ValueDimension / 16) * 8 * 8 * 6, (RegTensor<float> &)vreg_src7, ValueDimension / 16,
                preg_all32);
            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float *)outputUb + 8 * channelBlock + (ValueDimension / 16) * 8 * 8 * 7, (RegTensor<float> &)vreg_src8, ValueDimension / 16,
                preg_all32);
        }
    }
}

template <uint16_t ValueDimension = 128, typename OutputType = bfloat16_t, bool Normalize = true>
__aicore__ inline void processOutDn(const LocalTensor<OutputType>& output,
    const LocalTensor<float>& historyOutput, const LocalTensor<float>& historySum, uint32_t channelOffset = 0)
{
    processOutDnVF<ValueDimension, OutputType, Normalize>((__ubuf__ OutputType*)output.GetPhyAddr(),
        (__ubuf__ float*)historyOutput.GetPhyAddr(), (__ubuf__ float*)historySum.GetPhyAddr(), channelOffset);
}

// groupResultUb 为 FP32 [64,Dv+8]，第 Dv 列是本组 rowsum；historyOutputUb 为 FP32 [64,Dv]。
// historySumUb / historyMultiplierUb 为 FP32 [64]；UpdateHistory=false 不读取旧状态或 alpha。
// 有效 Q 行数在同一 Q 块的各组间保持不变；首组清零无效行，空组 PV 按零处理。
template <bool UpdateHistory, uint32_t ValueDimension>
__simd_vf__ inline void processUpdateNdVF(__ubuf__ float* historyOutputUb,
    __ubuf__ float* groupResultUb, __ubuf__ float* historyMultiplierUb,
    __ubuf__ float* historySumUb, uint32_t validQueryRows)
{
    static_assert(ValueDimension == 128 || ValueDimension == 256, "ND update supports Dv128/256");
    MaskReg preg_all32 = CreateMask<float, MaskPattern::ALL>();
    uint32_t remainingQueryRows = validQueryRows;
    MaskReg preg_valid_rows = UpdateMask<float>(remainingQueryRows);
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_multipliers;
    Duplicate(vreg_zero, 0.0F);
    {
        // 一个lane对应一个Q行：一次Gather取得所有有效行的sum，再一次连续写回[64]。
        RegTensor<uint32_t> vreg_row_indices, vreg_sum_offsets;
        RegTensor<float> vreg_group_sums, vreg_history_sums;
        Arange(reinterpret_cast<RegTensor<int32_t>&>(vreg_row_indices), static_cast<int32_t>(0));
        Muls(vreg_sum_offsets, vreg_row_indices, static_cast<uint32_t>(ValueDimension + 8), preg_all32);
        Gather(vreg_group_sums, groupResultUb + ValueDimension, vreg_sum_offsets, preg_valid_rows);
        Select(vreg_group_sums, vreg_group_sums, vreg_zero, preg_valid_rows);
        if constexpr (UpdateHistory) {
            LoadAlign(vreg_history_sums, historySumUb);
            LoadAlign(vreg_multipliers, historyMultiplierUb);
            MulAddDst(vreg_group_sums, vreg_history_sums, vreg_multipliers, preg_valid_rows);
            Select(vreg_group_sums, vreg_group_sums, vreg_zero, preg_valid_rows);
        }
        StoreAlign(historySumUb, vreg_group_sums, preg_all32);
    }
    for (uint16_t queryRow = 0; queryRow < validQueryRows; ++queryRow) {
        RegTensor<float> vreg_group_sum, vreg_multiplier;
        RegTensor<float> vreg_current_output0, vreg_current_output1;
        RegTensor<float> vreg_current_output2, vreg_current_output3;
        MaskReg preg_nonempty_group;
        __ubuf__ float* currentRowUb = groupResultUb + queryRow * (ValueDimension + 8);
        __ubuf__ float* historyRowUb = historyOutputUb + queryRow * ValueDimension;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(vreg_group_sum, currentRowUb + ValueDimension);
        Compares<float, CMPMODE::GT>(preg_nonempty_group, vreg_group_sum, 0.0F, preg_all32);
        if constexpr (UpdateHistory) {
            RegTensor<uint32_t> vreg_multiplier_indices;
            Duplicate(vreg_multiplier_indices, static_cast<uint32_t>(queryRow));
            Gather(vreg_multiplier, vreg_multipliers, vreg_multiplier_indices);
        }
        // Dv128两路、Dv256四路，按加载/选择/乘加/写回阶段交错。
        LoadAlign(vreg_current_output0, currentRowUb);
        LoadAlign(vreg_current_output1, currentRowUb + 64);
        if constexpr (ValueDimension == 256) {
            LoadAlign(vreg_current_output2, currentRowUb + 128);
            LoadAlign(vreg_current_output3, currentRowUb + 192);
        }
        Select(vreg_current_output0, vreg_current_output0, vreg_zero, preg_nonempty_group);
        Select(vreg_current_output1, vreg_current_output1, vreg_zero, preg_nonempty_group);
        if constexpr (ValueDimension == 256) {
            Select(vreg_current_output2, vreg_current_output2, vreg_zero, preg_nonempty_group);
            Select(vreg_current_output3, vreg_current_output3, vreg_zero, preg_nonempty_group);
        }
        if constexpr (UpdateHistory) {
            RegTensor<float> vreg_history_output0, vreg_history_output1;
            RegTensor<float> vreg_history_output2, vreg_history_output3;
            LoadAlign(vreg_history_output0, historyRowUb);
            LoadAlign(vreg_history_output1, historyRowUb + 64);
            if constexpr (ValueDimension == 256) {
                LoadAlign(vreg_history_output2, historyRowUb + 128);
                LoadAlign(vreg_history_output3, historyRowUb + 192);
            }
            MulAddDst(vreg_current_output0, vreg_history_output0, vreg_multiplier, preg_all32);
            MulAddDst(vreg_current_output1, vreg_history_output1, vreg_multiplier, preg_all32);
            if constexpr (ValueDimension == 256) {
                MulAddDst(vreg_current_output2, vreg_history_output2, vreg_multiplier, preg_all32);
                MulAddDst(vreg_current_output3, vreg_history_output3, vreg_multiplier, preg_all32);
            }
        }
        StoreAlign(historyRowUb, vreg_current_output0, preg_all32);
        StoreAlign(historyRowUb + 64, vreg_current_output1, preg_all32);
        if constexpr (ValueDimension == 256) {
            StoreAlign(historyRowUb + 128, vreg_current_output2, preg_all32);
            StoreAlign(historyRowUb + 192, vreg_current_output3, preg_all32);
        }
    }
    if constexpr (!UpdateHistory) {
        // rowsum尾行已在上面清零；首组还需初始化PV尾行，不读取无效Cube数据。
        for (uint16_t queryRow = validQueryRows; queryRow < 64; ++queryRow) {
            __ubuf__ float* historyRowUb = historyOutputUb + queryRow * ValueDimension;
            StoreAlign(historyRowUb, vreg_zero, preg_all32);
            StoreAlign(historyRowUb + 64, vreg_zero, preg_all32);
            if constexpr (ValueDimension == 256) {
                StoreAlign(historyRowUb + 128, vreg_zero, preg_all32);
                StoreAlign(historyRowUb + 192, vreg_zero, preg_all32);
            }
        }
    }
}

template <bool UpdateHistory, uint32_t ValueDimension>
__aicore__ inline void processUpdateNd(const LocalTensor<float>& historyOutput,
    const LocalTensor<float>& groupResult, const LocalTensor<float>& historyMultiplier,
    const LocalTensor<float>& historySum, uint32_t validQueryRows)
{
    processUpdateNdVF<UpdateHistory, ValueDimension>((__ubuf__ float*)historyOutput.GetPhyAddr(),
        (__ubuf__ float*)groupResult.GetPhyAddr(), (__ubuf__ float*)historyMultiplier.GetPhyAddr(),
        (__ubuf__ float*)historySum.GetPhyAddr(), validQueryRows);
}

// 输入 historyOutputUb 为 FP32 [64,Dv]，historySumUb 为 FP32 [64]，无效尾行由首组 update 清零。
// 输出为紧凑 [validRows,Dv]：普通输出归一化为 BF16；FD 由调用方直接写出原始 FP32 PV。
// BF16 分支采用双寄存器 Cast/Or/Gather 合并输出，空行输出零。
template <uint32_t ValueDimension, typename OutputType = bfloat16_t>
__simd_vf__ inline void processOutNdVF(__ubuf__ OutputType* outputUb,
    __ubuf__ float* historyOutputUb, __ubuf__ float* historySumUb, uint32_t validQueryRows = 64)
{
    static_assert(ValueDimension == 128 || ValueDimension == 256, "ND output supports Dv128/256");
    static_assert(IsSameType<OutputType, float>::value || IsSameType<OutputType, bfloat16_t>::value,
        "ND output must be FP32 or BF16");
    MaskReg preg_all32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all16 = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<float> vreg_zero, vreg_one;
    Duplicate(vreg_zero, 0.0F);
    Duplicate(vreg_one, 1.0F);
    RegTensor<float> vreg_inverse_row_sums;
    MaskReg preg_nonempty_rows;
    LoadAlign(vreg_inverse_row_sums, historySumUb);
    Compares<float, CMPMODE::GT>(preg_nonempty_rows, vreg_inverse_row_sums, 0.0F, preg_all32);
    Select(vreg_inverse_row_sums, vreg_inverse_row_sums, vreg_one, preg_nonempty_rows);
    // 一次向量除法计算 Q64 的全部行倒数，循环内通过寄存器 Gather 广播当前行。
    Div<float, &MXFP4_OUTPUT_DIV>(vreg_inverse_row_sums, vreg_one, vreg_inverse_row_sums, preg_all32);
    RegTensor<uint16_t> vreg_gather_indices, vreg_lane_indices, vreg_half_indices;
    if constexpr (!IsSameType<OutputType, float>::value) {
        Arange(reinterpret_cast<RegTensor<int16_t>&>(vreg_lane_indices), static_cast<int16_t>(0));
        ShiftLefts(vreg_gather_indices, vreg_lane_indices, static_cast<int16_t>(10), preg_all16);
        ShiftRights(vreg_gather_indices, vreg_gather_indices, static_cast<int16_t>(10), preg_all16);
        ShiftLefts(vreg_gather_indices, vreg_gather_indices, static_cast<int16_t>(1), preg_all16);
        ShiftRights(vreg_half_indices, vreg_lane_indices, static_cast<int16_t>(6), preg_all16);
        Or(vreg_gather_indices, vreg_gather_indices, vreg_half_indices, preg_all16);
    }
    for (uint16_t queryRow = 0; queryRow < validQueryRows; ++queryRow) {
        RegTensor<float> vreg_row_sum, vreg_inverse_row_sum;
        RegTensor<uint32_t> vreg_row_indices;
        MaskReg preg_nonempty_row;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(vreg_row_sum, historySumUb + queryRow);
        Compares<float, CMPMODE::GT>(preg_nonempty_row, vreg_row_sum, 0.0F, preg_all32);
        Duplicate(vreg_row_indices, static_cast<uint32_t>(queryRow));
        Gather(vreg_inverse_row_sum, vreg_inverse_row_sums, vreg_row_indices);
        if constexpr (IsSameType<OutputType, float>::value) {
            RegTensor<float> vreg_output0, vreg_output1;
            RegTensor<float> vreg_output2, vreg_output3;
            __ubuf__ float* historyRowUb = historyOutputUb + queryRow * ValueDimension;
            __ubuf__ OutputType* outputRowUb = outputUb + queryRow * ValueDimension;
            // Dv128两路、Dv256四路，按加载/归一化/选择/写回阶段交错。
            LoadAlign(vreg_output0, historyRowUb);
            LoadAlign(vreg_output1, historyRowUb + 64);
            if constexpr (ValueDimension == 256) {
                LoadAlign(vreg_output2, historyRowUb + 128);
                LoadAlign(vreg_output3, historyRowUb + 192);
            }
            Mul(vreg_output0, vreg_output0, vreg_inverse_row_sum, preg_all32);
            Mul(vreg_output1, vreg_output1, vreg_inverse_row_sum, preg_all32);
            if constexpr (ValueDimension == 256) {
                Mul(vreg_output2, vreg_output2, vreg_inverse_row_sum, preg_all32);
                Mul(vreg_output3, vreg_output3, vreg_inverse_row_sum, preg_all32);
            }
            Select(vreg_output0, vreg_output0, vreg_zero, preg_nonempty_row);
            Select(vreg_output1, vreg_output1, vreg_zero, preg_nonempty_row);
            if constexpr (ValueDimension == 256) {
                Select(vreg_output2, vreg_output2, vreg_zero, preg_nonempty_row);
                Select(vreg_output3, vreg_output3, vreg_zero, preg_nonempty_row);
            }
            StoreAlign(outputRowUb, vreg_output0, preg_all32);
            StoreAlign(outputRowUb + 64, vreg_output1, preg_all32);
            if constexpr (ValueDimension == 256) {
                StoreAlign(outputRowUb + 128, vreg_output2, preg_all32);
                StoreAlign(outputRowUb + 192, vreg_output3, preg_all32);
            }
        } else {
            RegTensor<float> vreg_output0, vreg_output1, vreg_output2, vreg_output3;
            RegTensor<OutputType> vreg_converted0, vreg_converted1, vreg_converted2, vreg_converted3;
            RegTensor<OutputType> vreg_interleaved0, vreg_interleaved1;
            RegTensor<OutputType> vreg_contiguous0, vreg_contiguous1;
            __ubuf__ float* historyRowUb = historyOutputUb + queryRow * ValueDimension;
            __ubuf__ OutputType* outputRowUb = outputUb + queryRow * ValueDimension;
            // 两路/四路FP32归一化，每两路合成一组连续128个BF16。
            LoadAlign(vreg_output0, historyRowUb);
            LoadAlign(vreg_output1, historyRowUb + 64);
            if constexpr (ValueDimension == 256) {
                LoadAlign(vreg_output2, historyRowUb + 128);
                LoadAlign(vreg_output3, historyRowUb + 192);
            }
            Mul(vreg_output0, vreg_output0, vreg_inverse_row_sum, preg_all32);
            Mul(vreg_output1, vreg_output1, vreg_inverse_row_sum, preg_all32);
            if constexpr (ValueDimension == 256) {
                Mul(vreg_output2, vreg_output2, vreg_inverse_row_sum, preg_all32);
                Mul(vreg_output3, vreg_output3, vreg_inverse_row_sum, preg_all32);
            }
            Select(vreg_output0, vreg_output0, vreg_zero, preg_nonempty_row);
            Select(vreg_output1, vreg_output1, vreg_zero, preg_nonempty_row);
            if constexpr (ValueDimension == 256) {
                Select(vreg_output2, vreg_output2, vreg_zero, preg_nonempty_row);
                Select(vreg_output3, vreg_output3, vreg_zero, preg_nonempty_row);
            }
            Cast<OutputType, float, MXFP4_CAST_ZERO>(vreg_converted0, vreg_output0, preg_all32);
            Cast<OutputType, float, MXFP4_CAST_ONE>(vreg_converted1, vreg_output1, preg_all32);
            if constexpr (ValueDimension == 256) {
                Cast<OutputType, float, MXFP4_CAST_ZERO>(vreg_converted2, vreg_output2, preg_all32);
                Cast<OutputType, float, MXFP4_CAST_ONE>(vreg_converted3, vreg_output3, preg_all32);
            }
            Or(reinterpret_cast<RegTensor<uint16_t>&>(vreg_interleaved0),
                reinterpret_cast<RegTensor<uint16_t>&>(vreg_converted0),
                reinterpret_cast<RegTensor<uint16_t>&>(vreg_converted1), preg_all16);
            if constexpr (ValueDimension == 256) {
                Or(reinterpret_cast<RegTensor<uint16_t>&>(vreg_interleaved1),
                    reinterpret_cast<RegTensor<uint16_t>&>(vreg_converted2),
                    reinterpret_cast<RegTensor<uint16_t>&>(vreg_converted3), preg_all16);
            }
            Gather(vreg_contiguous0, vreg_interleaved0, vreg_gather_indices);
            if constexpr (ValueDimension == 256) {
                Gather(vreg_contiguous1, vreg_interleaved1, vreg_gather_indices);
            }
            StoreAlign(outputRowUb, vreg_contiguous0, preg_all16);
            if constexpr (ValueDimension == 256) {
                StoreAlign(outputRowUb + 128, vreg_contiguous1, preg_all16);
            }
        }
    }
}

template <uint32_t ValueDimension, typename OutputType = bfloat16_t>
__aicore__ inline void processOutNd(const LocalTensor<OutputType>& output,
    const LocalTensor<float>& historyOutput, const LocalTensor<float>& historySum, uint32_t validQueryRows = 64)
{
    processOutNdVF<ValueDimension, OutputType>((__ubuf__ OutputType*)output.GetPhyAddr(),
        (__ubuf__ float*)historyOutput.GetPhyAddr(), (__ubuf__ float*)historySum.GetPhyAddr(), validQueryRows);
}

// LSE 使用自然对数参考；FD 输出 [64,8] 的原始 sum/整数 R，保持 workspace 形状不变。
template <QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT, bool BinaryFdReference = false>
__simd_vf__ inline void PrepareAttentionStatisticsVf(__ubuf__ float* lseOutput,
    __ubuf__ float* broadcastSum, __ubuf__ float* broadcastReference,
    __ubuf__ float* historySum, __ubuf__ float* groupReference)
{
    MaskReg allRows = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> rowSum, reference, safeSum, oneValues, invalidValues, lseValues;
    RegTensor<uint32_t> rowIndices, broadcastOffsets;
    MaskReg nonemptyRows;
    LoadAlign(rowSum, historySum);
    LoadAlign(reference, groupReference);
    Duplicate(oneValues, 1.0F);
    Duplicate(invalidValues, INVALID_GROUP_EXPONENT);
    Compares<float, CMPMODE::GT>(nonemptyRows, rowSum, 0.0F, allRows);
    Select(reference, reference, oneValues, nonemptyRows);
    if constexpr (QUANT_SCHEMA != QuantSchema::OCP_QUANT) {
        // Pscale includes 2^-ceil(log2(cx)); undo it when restoring score LSE.
        Adds(reference, reference, MXFP4_CX_SCALE_SHIFT, allRows);
    }
    Muls(reference, reference, MXFP4_LN2, allRows);
    if constexpr (QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT) {
        Adds(reference, reference, MXFP4_NEG_LN_DELTA, allRows);
    }
    Select(reference, reference, invalidValues, nonemptyRows);
    Select(safeSum, rowSum, oneValues, nonemptyRows);
    Ln(lseValues, safeSum, allRows);
    Add(lseValues, lseValues, reference, allRows);
    RegTensor<uint32_t> infinityBits;
    Duplicate(infinityBits, static_cast<uint32_t>(0x7f800000));
    Select(lseValues, lseValues, reinterpret_cast<RegTensor<float>&>(infinityBits), nonemptyRows);
    // LSE 每行预留 32B，允许逐行 DataCopyPad，不要求 UB 单个 float 地址对齐。
    Arange(reinterpret_cast<RegTensor<int32_t>&>(rowIndices), static_cast<int32_t>(0));
    Muls(broadcastOffsets, rowIndices, static_cast<uint32_t>(8), allRows);
    // 搬出仅使用每行的第一个 LSE；其余7个float是未使用的padding，不写入。
    Scatter(lseOutput, lseValues, broadcastOffsets, allRows);
    {
        // rowsum：先在寄存器中重复4次，再利用双搬出交错实现重复8次。
        RegTensor<float> repeated2Low, repeated2High;
        RegTensor<float> repeated4Rows0, repeated4Rows16, repeated4Rows32, repeated4Rows48;
        Interleave(repeated2Low, repeated2High, rowSum, rowSum);
        Interleave(repeated4Rows0, repeated4Rows16, repeated2Low, repeated2Low);
        Interleave(repeated4Rows32, repeated4Rows48, repeated2High, repeated2High);
        // 每次写16个Q行，共128个float；完整输出为[64,8]。
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            broadcastSum, repeated4Rows0, repeated4Rows0, allRows);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            broadcastSum + 128, repeated4Rows16, repeated4Rows16, allRows);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            broadcastSum + 256, repeated4Rows32, repeated4Rows32, allRows);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            broadcastSum + 384, repeated4Rows48, repeated4Rows48, allRows);
    }
    {
        if constexpr (BinaryFdReference) {
            // FD builds a_i from binary references, then merges raw PV with a_i / S.
            LoadAlign(reference, groupReference);
            Select(reference, reference, invalidValues, nonemptyRows);
        }
        // reference：先在寄存器中重复4次，再利用双搬出交错实现重复8次。
        RegTensor<float> repeated2Low, repeated2High;
        RegTensor<float> repeated4Rows0, repeated4Rows16, repeated4Rows32, repeated4Rows48;
        Interleave(repeated2Low, repeated2High, reference, reference);
        Interleave(repeated4Rows0, repeated4Rows16, repeated2Low, repeated2Low);
        Interleave(repeated4Rows32, repeated4Rows48, repeated2High, repeated2High);
        // 每次写16个Q行，共128个float；完整输出为[64,8]。
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            broadcastReference, repeated4Rows0, repeated4Rows0, allRows);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            broadcastReference + 128, repeated4Rows16, repeated4Rows16, allRows);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            broadcastReference + 256, repeated4Rows32, repeated4Rows32, allRows);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            broadcastReference + 384, repeated4Rows48, repeated4Rows48, allRows);
    }
}

// DN/ND 共用：输入[64] FP32；sum/reference输出[64,8]，LSE每行预留8个float但只写第0列。
// 只转换 LocalTensor 地址；流水同步由调用方负责。
template <QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT, bool BinaryFdReference = false>
__aicore__ inline void PrepareAttentionStatistics(const LocalTensor<float>& lseOutput,
    const LocalTensor<float>& broadcastSum, const LocalTensor<float>& broadcastReference,
    const LocalTensor<float>& historySum, const LocalTensor<float>& groupReference)
{
    PrepareAttentionStatisticsVf<QUANT_SCHEMA, BinaryFdReference>((__ubuf__ float*)lseOutput.GetPhyAddr(),
        (__ubuf__ float*)broadcastSum.GetPhyAddr(), (__ubuf__ float*)broadcastReference.GetPhyAddr(),
        (__ubuf__ float*)historySum.GetPhyAddr(), (__ubuf__ float*)groupReference.GetPhyAddr());
}
} // namespace Mxfp4Api
#endif
