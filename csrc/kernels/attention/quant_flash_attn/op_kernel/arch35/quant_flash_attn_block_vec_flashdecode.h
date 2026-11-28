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
 * \file quant_flash_attn_block_vec_flashdecode.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_FLASHDECODE_H
#define QUANT_FLASH_ATTN_BLOCK_VEC_FLASHDECODE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "memory_copy_arch35_quant_flash_attn.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_flash_attn_common_def.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "vf/vf_flash_decode_mxfp4.h"

namespace BaseApi {

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool useDn = false>
class QuantFlashAttnBlockVecFlashDecode {
    static_assert(IsSameType<INPUT_T, fp4x2_e2m1_t>::value, "FlashDecode requires FP4 input");
    static_assert(layout == LayOutTypeEnum::LAYOUT_TND && outLayout == LayOutTypeEnum::LAYOUT_TND,
                  "FlashDecode requires TND query and output");
    static_assert(KvLayoutType > 0, "FlashDecode requires paged KV");

public:
    // =================================类型定义区=================================
    struct TaskInfo {
        uint32_t bIdx;
        uint32_t n2Idx;
        uint32_t gS1Idx;
        uint32_t actualCombineLoopSize;
    };

    // The kernel checks that these buffers fit in the retired FA UB region.
    static __aicore__ inline constexpr uint32_t GetFdTotalUbSize()
    {
        return 5U * FD_LSE_BUF_BYTES + 4U * FD_MM2_BUF_BYTES + 3U * BUFFER_SIZE_BYTE_256B;
    }

private:
    // =================================常量区=================================
    static constexpr int64_t BYTE_BLOCK = 32UL;
    static constexpr int64_t REPEAT_BLOCK_BYTE = 256U;
    // 同步flag id参照flash_attn nd版FD: 避开TPipe AllocEventID分配的低段id,
    // 全部改用Mutex Lock/Unlock硬件信号量协议(无需初始化, flag初始态0)
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF1_FLAG = 8;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 2;
    static constexpr uint64_t SYNC_LSEOUTPUT_BUF_FLAG = 4;

    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T); // 32/4=8
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

    // dV按64对齐后的元素数(与FA侧dTemplateAlign64推导一致, 不依赖头文件)
    static constexpr uint32_t FD_DV_ALIGN64 = ((static_cast<uint32_t>(dVTemplateType) + 63U) / 64U) * 64U;
    // mm2系buffer实际最大行数(fdBalanceMBaseSize=8, 见FlashDecode主循环)
    static constexpr uint32_t FD_BALANCE_M_BASE = 8U;
    // lse系buffer(sum/max/exp)字节尺寸; mm2系buffer 8行*dAlign*fp32
    static constexpr uint32_t FD_LSE_BUF_BYTES = BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K; // 6144
    static constexpr uint32_t FD_MM2_BUF_BYTES = FD_BALANCE_M_BASE * FD_DV_ALIGN64 * sizeof(float);

    static constexpr bool HAS_MASK = hasAtten;
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;

    uint32_t preLoadNum_ = 2U;
    uint32_t dSizeV_Align_;
    using ConstInfoX = ConstInfo_t;

protected:
    GlobalTensor<float> lseSumFdGm_;
    GlobalTensor<float> lseMaxFdGm_;
    GlobalTensor<float> accumOutGm_;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<layout, true>();
    using QSeqParserType = ActualSeqLensParser<Q_MODE, int32_t, true>;
    using KvSeqParserType = ActualSeqLensParser<KV_MODE, int32_t>;

    QSeqParserType* qActSeqLensParser_ = nullptr;
    KvSeqParserType* kvActSeqLensParser_ = nullptr;

    // ================================类成员变量====================================
    // 结构体
    const ConstInfoX& constInfo_;
    TaskInfo taskInfo_{};

private:
    // ================================FD Local Buffer区(静态Tensor)====================================
    LocalTensor<T> fdSumBuf1_;          // FD_LSE_BUF_BYTES
    LocalTensor<T> fdSumBuf2_;          // FD_LSE_BUF_BYTES
    LocalTensor<T> fdMaxBuf1_;          // FD_LSE_BUF_BYTES
    LocalTensor<T> fdMaxBuf2_;          // FD_LSE_BUF_BYTES
    LocalTensor<T> fdLseExpBuf_;        // FD_LSE_BUF_BYTES
    LocalTensor<T> fdMm2ResBuf1_;       // FD_MM2_BUF_BYTES
    LocalTensor<T> fdMm2ResBuf2_;       // FD_MM2_BUF_BYTES
    LocalTensor<T> fdReduceBuf_;        // FD_MM2_BUF_BYTES
    LocalTensor<OUTPUT_T> fdOutputBuf_; // FD_MM2_BUF_BYTES

    LocalTensor<T> fdLseMaxUbBuf1_;    // 256B
    LocalTensor<T> fdLseMaxUbBuf2_;    // 256B
    LocalTensor<T> fdLseUbBuf_;        // 256B

public:
    __aicore__ inline QuantFlashAttnBlockVecFlashDecode(ConstInfoX& constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxFdGm, GlobalTensor<float> lseSumFdGm,
                                            GlobalTensor<float> accumOutGm, GlobalTensor<OUTPUT_T> attentionOutGm)
    {
        this->lseMaxFdGm_ = lseMaxFdGm;
        this->lseSumFdGm_ = lseSumFdGm;
        this->accumOutGm_ = accumOutGm;
        this->attentionOutGm_ = attentionOutGm;
    }

    __aicore__ inline void SetCuSeqLensParsers(QSeqParserType& qParser, KvSeqParserType& kvParser)
    {
        this->qActSeqLensParser_ = &qParser;
        this->kvActSeqLensParser_ = &kvParser;
    }

    __aicore__ inline void InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm)
    {
        this->softmaxLseGm_ = softmaxLseGm;
    }

    __aicore__ inline void InitParams()
    {
        this->dSizeV_Align_ = AttentionCommon::Align(constInfo_.dSizeV, FP32_REPEAT_ELEMENT_NUM);
    }

    template <uint32_t UB_FD_BASE_OFFSET>
    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIV {
            // The caller supplies a retired FA region and validates its capacity.
            // Preserve all bytes outside it, especially live score buffers and
            // cross-section state. Static aliases do not reset TPipe allocations.
            constexpr uint32_t OFF_SUM1 = UB_FD_BASE_OFFSET;
            constexpr uint32_t OFF_SUM2 = OFF_SUM1 + FD_LSE_BUF_BYTES;
            constexpr uint32_t OFF_MAX1 = OFF_SUM2 + FD_LSE_BUF_BYTES;
            constexpr uint32_t OFF_MAX2 = OFF_MAX1 + FD_LSE_BUF_BYTES;
            constexpr uint32_t OFF_LSEEXP = OFF_MAX2 + FD_LSE_BUF_BYTES;
            constexpr uint32_t OFF_MM2RES1 = OFF_LSEEXP + FD_LSE_BUF_BYTES;
            constexpr uint32_t OFF_MM2RES2 = OFF_MM2RES1 + FD_MM2_BUF_BYTES;
            constexpr uint32_t OFF_REDUCE = OFF_MM2RES2 + FD_MM2_BUF_BYTES;
            constexpr uint32_t OFF_OUTPUT = OFF_REDUCE + FD_MM2_BUF_BYTES;
            constexpr uint32_t OFF_LSEMAX1 = OFF_OUTPUT + FD_MM2_BUF_BYTES;
            constexpr uint32_t OFF_LSEMAX2 = OFF_LSEMAX1 + BUFFER_SIZE_BYTE_256B;
            constexpr uint32_t OFF_LSEUB = OFF_LSEMAX2 + BUFFER_SIZE_BYTE_256B;

            fdSumBuf1_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_SUM1, FD_LSE_BUF_BYTES).template ReinterpretCast<T>();
            fdSumBuf2_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_SUM2, FD_LSE_BUF_BYTES).template ReinterpretCast<T>();
            fdMaxBuf1_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_MAX1, FD_LSE_BUF_BYTES).template ReinterpretCast<T>();
            fdMaxBuf2_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_MAX2, FD_LSE_BUF_BYTES).template ReinterpretCast<T>();
            fdLseExpBuf_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSEEXP, FD_LSE_BUF_BYTES).template ReinterpretCast<T>();
            fdMm2ResBuf1_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_MM2RES1, FD_MM2_BUF_BYTES).template ReinterpretCast<T>();
            fdMm2ResBuf2_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_MM2RES2, FD_MM2_BUF_BYTES).template ReinterpretCast<T>();
            fdReduceBuf_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_REDUCE, FD_MM2_BUF_BYTES).template ReinterpretCast<T>();
            fdOutputBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_OUTPUT, FD_MM2_BUF_BYTES)
                               .template ReinterpretCast<OUTPUT_T>();
            fdLseMaxUbBuf1_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSEMAX1, BUFFER_SIZE_BYTE_256B)
                                  .template ReinterpretCast<T>();
            fdLseMaxUbBuf2_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSEMAX2, BUFFER_SIZE_BYTE_256B)
                                  .template ReinterpretCast<T>();
            fdLseUbBuf_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSEUB, BUFFER_SIZE_BYTE_256B).template ReinterpretCast<T>();
        }
    }

protected:
    __aicore__ inline void CopyAccumOutIn(LocalTensor<T>& accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount)
    {
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<T> copyInPadParams;
        copyInParams.blockCount = dealRowCount;
        copyInParams.blockLen = constInfo_.dSizeV * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (this->dSizeV_Align_ - constInfo_.dSizeV) / BLOCK_ELEMENT_NUM;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (this->dSizeV_Align_ - constInfo_.dSizeV) % BLOCK_ELEMENT_NUM;
        copyInPadParams.paddingValue = 0;
        uint64_t combineAccumOutOffset = startRow * constInfo_.dSizeV +                // taskoffset + g轴offset
                                         splitKVIndex * mBaseSize * constInfo_.dSizeV; // 份数offset

        DataCopyPad(accumOutLocal, accumOutGm_[combineAccumOutOffset], copyInParams, copyInPadParams);
    }
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1_ : fdSumBuf2_;
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1_ : fdMaxBuf2_;

        uint64_t combineLseOffset = (baseOffset + startRow) * BLOCK_ELEMENT_NUM;
        uint64_t combineLoopOffset = mBaseSize * BLOCK_ELEMENT_NUM;
        uint64_t dealRowCountAlign = dealRowCount * BLOCK_ELEMENT_NUM;

        for (uint32_t i = 0; i < taskInfo_.actualCombineLoopSize; ++i) {
            DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm_[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign); // 份数offset

            DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm_[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign);
        }
    }
    template <QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT>
    __aicore__ inline void ComputeScaleValue(LocalTensor<T>& lseExp, uint32_t dealRowCount,
                                             uint32_t actualCombineLoopSize, uint32_t cntM)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1_ : fdSumBuf2_;
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1_ : fdMaxBuf2_;
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1_ : fdLseMaxUbBuf2_;

        // C4 merges raw PV with normalized a_i / S multipliers.
        Mxfp4Api::PrepareFlashDecodeVf<QUANT_SCHEMA>(
            (__ubuf__ float*)lseMax.GetPhyAddr(), (__ubuf__ float*)lseSum.GetPhyAddr(),
            (__ubuf__ float*)lseExp.GetPhyAddr(), (__ubuf__ float*)fdLseUbBuf_.GetPhyAddr(),
            (__ubuf__ float*)lseMaxUb.GetPhyAddr(), dealRowCount, actualCombineLoopSize);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUTPUT_T>& attenOutUb, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{
            .tensor = attenOutUb,
            .rowCount = dealRowCount,
            .colCount = columnCount,
        };
        GmCoordGs1Merge gmCoord{.bIdx = taskInfo_.bIdx,
                                .n2Idx = taskInfo_.n2Idx,
                                .gS1Idx = taskInfo_.gS1Idx + startRow,
                                .dIdx = 0,
                                .gS1DealSize = dealRowCount,
                                .dDealSize = (uint32_t)constInfo_.dSizeV};

        constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
        FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm_;
        outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                          *qActSeqLensParser_);
        CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    }
    __aicore__ inline void ReduceFinalRes(LocalTensor<T>& reduceOut, LocalTensor<T>& mm2Res, LocalTensor<T>& lseLocal,
                                          uint32_t cntKV, uint32_t dealRowCount)
    {
        uint64_t dSizeV_Align = (uint64_t)this->dSizeV_Align_;
        FaVectorApi::ReduceFinalRes_VF<T>(reduceOut, lseLocal, mm2Res, dealRowCount, dSizeV_Align, cntKV);
    }
    __aicore__ inline void CopyFinalResOut(LocalTensor<T>& accumOutLocal, uint32_t startRow, uint32_t dealRowCount)
    {
        LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = fdOutputBuf_;
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (HAS_MASK) {
            DealInvalidRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align_);
        }
        Mutex::Lock<PIPE_V>(SYNC_FDOUTPUT_BUF_FLAG);
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo_.dSizeV};
        tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        // C4 must use the same ties-away conversion as its non-FD V2 output.
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND,
             dealRowCount * this->dSizeV_Align_);
        Mutex::Unlock<PIPE_V>(SYNC_FDOUTPUT_BUF_FLAG);
        Mutex::Lock<PIPE_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        Bmm2DataCopyOutTrans(tmpBmm2ResCastTensor, startRow, dealRowCount, this->dSizeV_Align_);
        Mutex::Unlock<PIPE_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    }
    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidRows(LocalTensor<UBOUT_T>& attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount)
    {
        // Only right-down causal sets HAS_MASK in the C4A4 tiling key.
        const uint64_t actSeqLensQ = qActSeqLensParser_->GetActualSeqLength(taskInfo_.bIdx);
        const int64_t actSeqLensKv = kvActSeqLensParser_->GetActualSeqLength(taskInfo_.bIdx);
        fa_base_vector::InvalidRowParams params{
            .actS1Size = actSeqLensQ,
            .gSize = static_cast<uint64_t>(constInfo_.realGSize),
            .gS1Idx = taskInfo_.gS1Idx + startRow,
            .dealRowCount = dealRowCount,
            .columnCount = columnCount,
            .preTokensPerBatch = 0,
            .nextTokensPerBatch = actSeqLensKv - static_cast<int64_t>(actSeqLensQ),
        };

        fa_base_vector::InvalidRows<UBOUT_T, fa_base_vector::UbInputFormat::S1G> invalidRows;
        invalidRows(attenOutUb, params);
    }

public:
    template <QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT>
    __aicore__ inline void FlashDecode(FDparamsX& fd)
    {
        if (!fd.fdCoreEnable) {
            return;
        }
        uint32_t fdBalanceMBaseSize = 8U;
        if constexpr (IsSameType<INPUT_T, float4_e2m1x2_t>::value) {
            // Each split carries eight FP32 copies per row. 32 splits require
            // 8192 bytes for eight rows, exceeding the 6144-byte statistics slot.
            ASSERT(fd.fdS2SplitNum != 0 && fd.fdS2SplitNum <= FD_LSE_BUF_BYTES / (8U * sizeof(T)));
            fdBalanceMBaseSize = AttentionCommon::Min(fdBalanceMBaseSize,
                static_cast<uint32_t>(FD_LSE_BUF_BYTES / (fd.fdS2SplitNum * 8U * sizeof(T))));
        }
        uint32_t fdBalanceMSplitNum = (fd.mLen + fdBalanceMBaseSize - 1) / fdBalanceMBaseSize;
        uint32_t fdBalanceMTailSize =
            (fd.mLen % fdBalanceMBaseSize == 0) ? fdBalanceMBaseSize : fd.mLen % fdBalanceMBaseSize;

        uint32_t reduceGlobaLoop = 0;
        uint32_t reduceMLoop = 0;

        uint32_t tmpFdS1gOuterMStart = 0;
        uint32_t tmpFdS1gOuterMEnd = fdBalanceMSplitNum - 1;
        taskInfo_.bIdx = fd.fdBN2Idx / constInfo_.realN2Size;
        taskInfo_.n2Idx = fd.fdBN2Idx % constInfo_.realN2Size;
        taskInfo_.gS1Idx = fd.fdMIdx * mBaseSize;
        taskInfo_.actualCombineLoopSize = fd.fdS2SplitNum; // 当前规约任务kv方向有几份
        uint64_t combineTaskPrefixSum = fd.fdWorkspaceIdx;
        uint64_t taskOffset = combineTaskPrefixSum * mBaseSize;

        for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
             ++fdS1gOuterMIdx) { // 左闭右闭
            uint32_t actualGSplitSize = fdBalanceMBaseSize;
            if (fdS1gOuterMIdx == fdBalanceMSplitNum - 1) {
                actualGSplitSize = fdBalanceMTailSize;
            }
            uint32_t startRow = fd.mStart + fdS1gOuterMIdx * fdBalanceMBaseSize;

            LocalTensor<T> lseExp = fdLseExpBuf_;
            LocalTensor<T> reduceOut = fdReduceBuf_;
            Mutex::Lock<PIPE_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);
            Mutex::Unlock<PIPE_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            for (uint32_t preLoadIdx = 0; preLoadIdx < preLoadNum_; ++preLoadIdx) {
                LocalTensor<T> mm2Res = ((reduceGlobaLoop + preLoadIdx) & 1) == 0 ? fdMm2ResBuf1_ : fdMm2ResBuf2_;
                Mutex::Lock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                Mutex::Unlock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
            }
            Mutex::Lock<PIPE_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            Mutex::Lock<PIPE_V>(SYNC_LSEOUTPUT_BUF_FLAG);
            ComputeScaleValue<QUANT_SCHEMA>(lseExp, actualGSplitSize, taskInfo_.actualCombineLoopSize, reduceMLoop);
            Mutex::Unlock<PIPE_V>(SYNC_LSEOUTPUT_BUF_FLAG);
            Mutex::Unlock<PIPE_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            if (constInfo_.isSoftmaxLseEnable) {
                // lse行无效在ComputeScaleValue的VF计算时已经进行了赋值inf处理
                LocalTensor<T> maxLseUb = fdLseUbBuf_;
                Mutex::Lock<PIPE_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                uint32_t mOffset = taskInfo_.gS1Idx + startRow;
                // LSE 输出改为 N-major 排布 [N2*G, T]: N 在外, T 在内
                uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(taskInfo_.bIdx);
                uint64_t bN2Offset = taskInfo_.n2Idx * constInfo_.realGSize * constInfo_.t1Size + prefixBS1;
                if constexpr (useDn) {
                    DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset,
                                                                             mOffset, actualGSplitSize, constInfo_);
                } else {
                    DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                   actualGSplitSize, constInfo_);
                }
                Mutex::Unlock<PIPE_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
            }

            for (uint32_t i = 0; i < taskInfo_.actualCombineLoopSize; ++i) {
                LocalTensor<T> mm2Res = (reduceGlobaLoop & 1) == 0 ? fdMm2ResBuf1_ : fdMm2ResBuf2_;
                if (i >= preLoadNum_) {
                    Mutex::Lock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    Mutex::Unlock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                }
                Mutex::Lock<PIPE_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                Mutex::Unlock<PIPE_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                reduceGlobaLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize);
            reduceMLoop += 1;
        }
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool useDn = false>
class QuantFlashAttnBlockVecFlashDecodeDummy {
public:
    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockVecFlashDecodeDummy(ConstInfoX& constInfo){};
};

} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_BLOCK_VEC_FLASHDECODE_H
