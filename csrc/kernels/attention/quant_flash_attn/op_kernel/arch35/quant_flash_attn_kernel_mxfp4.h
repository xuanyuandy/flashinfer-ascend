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
 * \file quant_flash_attn_kernel_mxfp4.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_KERNEL_MXFP4_H_
#define QUANT_FLASH_ATTN_KERNEL_MXFP4_H_

#include <cstdint>
#include <type_traits>
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "quant_flash_attn_common_def.h"
#include "quant_flash_attn_block_cube_mxfp4.h"
#include "quant_flash_attn_block_vec_mxfp4.h"
#include "quant_flash_attn_block_vec_flashdecode.h"
#include "memory_copy_arch35_quant_flash_attn.h"
#include "quant_flash_attn_tiling_data.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/memcopy/parser.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {

__aicore__ inline int64_t ClipMxfp4SInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class QuantFlashAttnKernelMxfp4 {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;
    static constexpr uint32_t s2SplitSize = 128U;

    static constexpr bool USE_DN = CubeBlockType::USE_DN; //DN 模式
    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;
    static_assert(CubeBlockType::ENABLE_QF322F32_PRE == VecFaBlockType::ENABLE_QF322F32_PRE,
                  "MXFP4 Cube and Vector must agree on MM1 Fixpipe scaling");

    static constexpr uint32_t PRELOAD_N = 20; // C2落后C1 20 轮
    static constexpr uint32_t GROUP_N = 16;  // 16个task的C2在L0C累积，然后执行一次V2
    static constexpr uint32_t DELAY_P_SCALE_N = 3; // P scale 在每个group的16个task的C1完成后，再过3轮，对这16个task进行一次pscale的update
    // KV lookahead 预取：全部布局（BBND/BNBD/NZ）、DN/ND、D128/D256 生效。创建指针领先
    // 执行 LEAD 个 tick：K 在创建时发射（提前 LEAD 拍），V 在 Bmm2 消费前 LEAD 拍发射；
    // 每流在途 (LEAD+1) 个槽（K 奇数槽、V 偶数槽，槽号 = 2*(taskLoop % (LEAD+1)) (+1 for K)）。
    // 任务环 live set 为 [t-20, t+LEAD]，大小 PRELOAD_N+1+LEAD。
    static constexpr uint32_t KV_PREFETCH_LEAD = 1U;
    static constexpr uint32_t KV_STREAM_SLOTS = KV_PREFETCH_LEAD + 1U; // 每流槽数
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1U + KV_PREFETCH_LEAD;

    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION; // 是否使用paged attn，true
    static constexpr bool FLASH_DECODE = VecFaBlockType::FLASH_DECODE; // 是否使用flash decoding
    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT; // TND 为主
    static constexpr LayOutTypeEnum LAYOUT_KV = CubeBlockType::LAYOUT; // PA_BBND 和 PA_NZ
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>(); // TND的情况下，是累积的形式，即 len[i] = act[i+1]-act[i]
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_KV, PAGE_ATTENTION>(); // paged attn的情况下，不是累积形式，即len[i] = act[i]

    using INPUT_T = typename CubeBlockType::Q_T; //Q 输入的数据类型
    using T = typename CubeBlockType::MM_T; // Cube block 定义的矩阵计算结果类型
    using OUT_T = typename VecFaBlockType::OUT_T; // 最终输出的数据类型

#ifdef __DAV_C310_CUBE__
    using CoreTask = Mxfp4CubeTask;
#else
    using CoreTask = Mxfp4VecTask;
#endif
    static constexpr bool TRACK_FD_SPLIT = FLASH_DECODE && std::is_same_v<CoreTask, Mxfp4VecTask>;

    // 输入tensor的GM的地址
    GlobalTensor<int32_t> cuSeqLensGmQ_;
    GlobalTensor<int32_t> cuSeqLensGmKv_;
    GlobalTensor<int32_t> seqUsedGmQ_;
    GlobalTensor<int32_t> seqUsedGmKv_;
    GlobalTensor<uint32_t> faMetaDataGm_;
    GlobalTensor<uint32_t> fdMetaDataGm_;
    GlobalTensor<float> softmaxLseGm_;
    __gm__ uint8_t* valuePtr_ = nullptr;

    // constinfo 和 3个主要成员
    ConstInfo_t constInfo_ {};
    TPipe* pipe_ = nullptr;
    CubeBlockType cubeBlock_;
    VecFaBlockType vecFaBlock_;
    VecFdBlockType vecFdBlock_;

    // tile调度的相关参数
    uint64_t actSeqLensKv_ = 0; // 当前 batch 实际参与计算的 KV token 数
    uint64_t actSeqLensQ_ = 0; // 当前 batch 实际参与计算的 Q token 数
    uint32_t curS2Start_ = 0; // 当前 M 块在本核需要处理的第一个 S2 块编号。结合 metadata 分片边界和 mask 范围计算
    uint32_t curS2End_ = 0; // 当前 M 块在本核需要处理的S2块的结束位置，不包括该位置，区间为 [curS2Start_, curS2End_)
    uint32_t prevBIdx_ = 0; // 上一个batch编号，用于判断是否到了一个新的batch，新的batch需要读取新的q和kv的长度
    uint32_t prevBN2Idx_ = 0; // 上一个BN2的合并的编号，用于判断是否到了一个新的head块，和prevGS1Idx_一起判断是否切到新的M
    uint32_t prevGS1Idx_ = 0; // 上一个M块的编号，需要切换到新的M块
    bool headS2Split_ = 0; // 判断当前M开始的S2块是否是第一个S2块
    bool tailS2Split_ = 0; // 判断当前M结束的S2块是否是最后一个S2块，用于确定fd的写入

    // metadata 的信息
    uint32_t sectionNum_; // 总的section数
    uint32_t metadataAicNum_; // metadata 为每个 section 存放的 AIC 记录数，用于定位 FA 记录
    uint32_t metadataAivNum_; // metadata 为每个 section 存放的 AIV 记录数，用于定位 FD 记录
    // fa metadata，基本块的范围，表示起点和终点
    uint32_t bN2Start_;
    uint32_t bN2End_;
    uint32_t gS1OStart_;
    uint32_t gS1OEnd_;
    uint32_t s2OStart_;
    uint32_t s2OEnd_;
    uint32_t coreFirstTmpOutWsPos_; // 当前核第一个需要保存的 FA 分片结果在 workspace 中的槽位索引，不是字节地址。后续根据布局换算实际地址
    // fd metadata
    FDparamsX fdParams_ {}; //fd合并的相关参数

    // q和kv的长度的解析
    typename std::conditional<(LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND), ActualSeqLensParser<Q_MODE, int32_t, true>,
                              ActualSeqLensParser<Q_MODE, int32_t>>::type qCuSeqLensParser_;
    typename std::conditional<(!PAGE_ATTENTION && LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND),
                              ActualSeqLensParser<KV_MODE, int32_t, true>, ActualSeqLensParser<KV_MODE, int32_t>>::type kvCuSeqLensParser_;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> qSeqUsedParser_;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> kvSeqUsedParser_;

    // ===========================================function========================================================================
    // KV lookahead 槽号公式：按任务号奇偶取槽，发射侧（创建时/滞后 1 拍）与消费侧
    // （Bmm1/Bmm2）使用同一公式，间隔为偶数拍，槽号天然一致。
    __aicore__ inline static uint32_t KeySlotOf(uint64_t taskLoop)
    {
        return 1U + 2U * static_cast<uint32_t>(taskLoop % KV_STREAM_SLOTS); // K 流占奇数槽
    }
    __aicore__ inline static uint32_t ValueSlotOf(uint64_t taskLoop)
    {
        return 2U * static_cast<uint32_t>(taskLoop % KV_STREAM_SLOTS); // V 流占偶数槽
    }

    __aicore__ inline QuantFlashAttnKernelMxfp4(): cubeBlock_(constInfo_), vecFaBlock_(constInfo_), vecFdBlock_(constInfo_){};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* sinks, __gm__ uint8_t* attnMask, __gm__ uint8_t* cuSeqLensQ,
                                __gm__ uint8_t* cuSeqLensKv, __gm__ uint8_t* blockTable,
                                __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* dequantScaleKey,
                                __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* pScale, __gm__ uint8_t* softmaxLse,
                                __gm__ uint8_t* attnOut, __gm__ uint8_t* workspace, __gm__ uint8_t* metadata,
                                __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedKv,
                                const QuantFlashAttnTilingData& tiling, TPipe* tPipe)
    {
        this->pipe_ = tPipe;
        InitConstInfo(tiling);
        constInfo_.needInitOutput = ((__gm__ uint32_t*)metadata)[QFA_HEAD_NEED_INIT_OUTPUT_INDEX] != 0;
        constInfo_.enableFlashDecode = static_cast<bool>(((__gm__ uint32_t*)metadata)[1]);

        valuePtr_ = value;

        // 当前 LAYOUT_Q 只为 LayOutTypeEnum::LAYOUT_TND
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmQ_.SetGlobalBuffer((__gm__ int32_t*)cuSeqLensQ, constInfo_.cuSeqLensQSize + 1);
            seqUsedGmQ_.SetGlobalBuffer((__gm__ int32_t*)sequsedQ, constInfo_.seqUsedQSize);
        } else {
            seqUsedGmQ_.SetGlobalBuffer((__gm__ int32_t*)sequsedQ, constInfo_.seqUsedQSize);
        }
        // 当前 LAYOUT_KV 只为 PA
        if constexpr (LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmKv_.SetGlobalBuffer((__gm__ int32_t*)cuSeqLensKv, constInfo_.cuSeqLensKVSize + 1);
            seqUsedGmKv_.SetGlobalBuffer((__gm__ int32_t*)sequsedKv, constInfo_.seqUsedKvSize);
        } else {
            seqUsedGmKv_.SetGlobalBuffer((__gm__ int32_t*)sequsedKv, constInfo_.seqUsedKvSize);
        }

        sectionNum_ = ((__gm__ uint32_t*)metadata)[METADATA_HEADER_SECTION_NUM_INDEX];
        metadataAicNum_ = ((__gm__ uint32_t*)metadata)[METADATA_HEADER_AIC_NUM_INDEX];
        metadataAivNum_ = ((__gm__ uint32_t*)metadata)[METADATA_HEADER_AIV_NUM_INDEX];

        faMetaDataGm_.SetGlobalBuffer((__gm__ uint32_t*)(metadata + METADATA_HEADER_OFFSET),
                                      sectionNum_ * metadataAicNum_ * METADATA_STRIDE);
        InitQCuSeqLensParser(cuSeqLensQ, sequsedQ);
        InitKvCuSeqLensParser(cuSeqLensKv, sequsedKv);

        if ASCEND_IS_AIV {
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                vecFaBlock_.InitVecBlock(tPipe, cuSeqLensQ, cuSeqLensKv, pScale, attnMask, softmaxLse, attnOut,
                                         workspace);
                vecFaBlock_.SetCuSeqLensParser(qCuSeqLensParser_);
            } else {
                vecFaBlock_.InitVecBlock(tPipe, sequsedQ, sequsedKv, pScale, attnMask, softmaxLse, attnOut, workspace);
                vecFaBlock_.SetCuSeqLensParser(qSeqUsedParser_);
            }
            vecFaBlock_.ClearOutput();
        }

        if ASCEND_IS_AIC {
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                cubeBlock_.InitCubeBlock(tPipe, query, key, value, blockTable, dequantScaleQuery,
                                         dequantScaleKey, dequantScaleValue, qCuSeqLensParser_, kvCuSeqLensParser_);
            } else {
                cubeBlock_.InitCubeBlock(tPipe, query, key, value, blockTable, dequantScaleQuery,
                                         dequantScaleKey, dequantScaleValue, qSeqUsedParser_, kvSeqUsedParser_);
            }
        }
        if constexpr (FLASH_DECODE) {
            if (constInfo_.enableFlashDecode) {
                if ASCEND_IS_AIV {
                    fdMetaDataGm_.SetGlobalBuffer(
                        (__gm__ uint32_t*)(metadata + METADATA_HEADER_OFFSET +
                                           METADATA_STRIDE * metadataAicNum_ * sectionNum_ * sizeof(uint32_t)),
                        sectionNum_ * metadataAivNum_ * METADATA_STRIDE);
                    vecFdBlock_.InitParams();
                    vecFdBlock_.InitGlobalTensor(this->vecFaBlock_.softmaxFDMaxGm_, this->vecFaBlock_.softmaxFDSumGm_,
                                                 this->vecFaBlock_.accumOutGm_, this->vecFaBlock_.attentionOutGm_);
                    if (constInfo_.isSoftmaxLseEnable) {
                        softmaxLseGm_.SetGlobalBuffer((__gm__ float*)softmaxLse);
                        vecFdBlock_.InitSoftmaxLseGm(softmaxLseGm_);
                    }
                    if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                        vecFdBlock_.SetCuSeqLensParsers(qCuSeqLensParser_, kvCuSeqLensParser_);
                    } else {
                        vecFdBlock_.SetCuSeqLensParsers(qSeqUsedParser_, kvSeqUsedParser_);
                    }
                }
            }
        }

    }
    __aicore__ inline void InitConstInfo(const QuantFlashAttnTilingData& tiling)
    {
        if ASCEND_IS_AIC {
            constInfo_.aicIdx = GetBlockIdx(); // 对于aic来说获取的自己的编号
        } else {
            constInfo_.aivIdx = GetBlockIdx(); // 对于aiv来说也是获取的自己的编号
            constInfo_.aicIdx = GetBlockIdx() / GetSubBlockNum(); // aiv直接的编号除以每个aic中aiv的数量就是其对应的aic的编号
            constInfo_.subBlockIdx = GetSubBlockIdx();
        }

        const QuantFlashAttnBaseParams& qfaBaseParams = tiling.baseTiling.quantFlashAttnBaseParams;
        const QuantFlashAttnAttenMaskParams& qfaAttenMaskParams = tiling.baseTiling.quantFlashAttnAttenMaskParams;
        const QuantFlashAttnPageAttentionParams& qfaPageAttentionParams = tiling.baseTiling.quantFlashAttnPageAttentionParams;
        const QuantFlashAttnWorkspaceParams& qfaWorkspaceParams = tiling.baseTiling.quantFlashAttnWorkspaceParams;

        constInfo_.bSize = qfaBaseParams.bSize;
        constInfo_.t1Size = qfaBaseParams.t1Size;
        constInfo_.n2Size = qfaBaseParams.n2Size;
        constInfo_.gSize = qfaBaseParams.gSize;
        constInfo_.s1Size = qfaBaseParams.s1Size;
        constInfo_.s2Size = qfaBaseParams.s2Size;
        constInfo_.dSizeV = qfaBaseParams.dSizeV;
        if constexpr (USE_DN) { // prefill不合轴
            constInfo_.realN2Size = constInfo_.n2Size * constInfo_.gSize;
            constInfo_.realGSize = 1;
        } else { // decode合轴
            constInfo_.realN2Size = constInfo_.n2Size;
            constInfo_.realGSize = constInfo_.gSize;
        }
        constInfo_.cuSeqLensQSize = qfaBaseParams.cuSeqLensQSize;
        constInfo_.cuSeqLensKVSize = qfaBaseParams.cuSeqLensKVSize;
        constInfo_.seqUsedQSize = qfaBaseParams.seqUsedQSize;
        constInfo_.seqUsedKvSize = qfaBaseParams.seqUsedKvSize;
        constInfo_.scaleValue = static_cast<float>(qfaBaseParams.scaleValue);
        constInfo_.coreNum = qfaBaseParams.coreNum;
        constInfo_.sparseMode =
            qfaAttenMaskParams.sparseMode;
        constInfo_.preTokens = qfaAttenMaskParams.winLefts;
        constInfo_.nextTokens = qfaAttenMaskParams.winRights;
        constInfo_.attenMaskS2Size = qfaAttenMaskParams.attenMaskS2Size;

        constInfo_.accumOutSize = qfaWorkspaceParams.accumOutSize;
        constInfo_.logSumExpSize = qfaWorkspaceParams.logSumExpSize;

        // strides
        constInfo_.keyStrides.bnStride = qfaBaseParams.keyStrides.bnStride; // 在PA_BBND [blocknum, blocksize, headnum, D]下，bnStride = blocksize*headnum*D
        constInfo_.keyStrides.n2Stride = qfaBaseParams.keyStrides.n2Stride; // 在PA_BBND [blocknum, blocksize, headnum, D]下，n2Stride = headnum*D，其实是两个token之间的距离，不是head
        constInfo_.valueStrides.bnStride = qfaBaseParams.valueStrides.bnStride;
        constInfo_.valueStrides.n2Stride = qfaBaseParams.valueStrides.n2Stride;
        constInfo_.kDescaleStrides.bnStride = qfaBaseParams.kDescaleStrides.bnStride;
        constInfo_.kDescaleStrides.n2Stride = qfaBaseParams.kDescaleStrides.n2Stride;
        constInfo_.vDescaleStrides.bnStride = qfaBaseParams.vDescaleStrides.bnStride;
        constInfo_.vDescaleStrides.n2Stride = qfaBaseParams.vDescaleStrides.n2Stride;

        // pageAttention
        if constexpr (PAGE_ATTENTION) {
            constInfo_.maxBlockNumPerBatch = qfaPageAttentionParams.maxBlockNumPerBatch;
            constInfo_.blockSize = qfaPageAttentionParams.blockSize;
        }

        // LSE
        constInfo_.isSoftmaxLseEnable = qfaBaseParams.isSoftMaxLseEnable;
    }


    __aicore__ inline void InitQCuSeqLensParser(__gm__ uint8_t* cuSeqLensQPtr, __gm__ uint8_t* sequsedQPtr)
    {
        // 当前 LAYOUT_Q 只能为 LayOutTypeEnum::LAYOUT_TND
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            qCuSeqLensParser_.Init(cuSeqLensQPtr, constInfo_.cuSeqLensQSize + 1, sequsedQPtr, constInfo_.seqUsedQSize);
        } else {
            qSeqUsedParser_.Init(seqUsedGmQ_, constInfo_.seqUsedQSize, constInfo_.s1Size);
        }
    }

    __aicore__ inline void InitKvCuSeqLensParser(__gm__ uint8_t* cuSeqLensKvPtr, __gm__ uint8_t* sequsedKvPtr)
    {
        // 当前PAGE_ATTENTION为true，并且LAYOUT_KV一定不是LayOutTypeEnum::LAYOUT_TND，走else分支
        if constexpr (!PAGE_ATTENTION && LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
            kvCuSeqLensParser_.Init(cuSeqLensKvPtr, constInfo_.cuSeqLensKVSize + 1, sequsedKvPtr,
                                    constInfo_.seqUsedKvSize);
        } else if constexpr (PAGE_ATTENTION && LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
            kvCuSeqLensParser_.Init(sequsedKvPtr, constInfo_.seqUsedKvSize, constInfo_.s2Size);
        } else {
            kvSeqUsedParser_.Init(seqUsedGmKv_, constInfo_.seqUsedKvSize, constInfo_.s2Size);
        }
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return METADATA_STRIDE * metadataAicNum_ * sectionIdx + METADATA_STRIDE * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return METADATA_STRIDE * metadataAivNum_ * sectionIdx + METADATA_STRIDE * coreIdx + metaIdx;
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_S2_END_INDEX, sectionIdx));
        if constexpr (TRACK_FD_SPLIT) {
            coreFirstTmpOutWsPos_ = faMetaDataGm_.GetValue(
                GetFAMetaDataIndex(constInfo_.aicIdx, QFA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));
        }
    }

    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        GetFASectionInfo(sectionIdx);
        prevBN2Idx_ = bN2Start_;
        prevGS1Idx_ = gS1OStart_;
        if constexpr (TRACK_FD_SPLIT) {
            headS2Split_ = false;
            tailS2Split_ = false;
        }
        DispatchRows();
    }

    // Both cores enumerate the same rows and task numbers. The task type selects the
    // descriptor fields and the pipeline at compile time.
    template <typename Task = CoreTask>
    __aicore__ inline void DispatchRows()
    {
        constexpr bool isCube = std::is_same_v<Task, Mxfp4CubeTask>;
        Task tasks[PRELOAD_TASK_CACHE_SIZE];
        uint32_t head = bN2Start_;
        uint32_t row = gS1OStart_;
        uint32_t s2 = s2OStart_;
        uint64_t created = 0;
        uint64_t tick = 0;
        while (ShouldDispatchTask(head, row, s2)) {
            const auto mode = GetTaskDealMode(head, row, s2);
            if (mode != TASK_DEAL_MODE::CREATE_TASK) {
                UpdateAxisInfo(mode, head, row, s2);
                continue;
            }

            const uint32_t realHead = head % constInfo_.realN2Size;
            const uint32_t queryStart = row * mBaseSize;
            const uint32_t rows = AttentionCommon::Min(static_cast<uint64_t>(mBaseSize),
                actSeqLensQ_ * constInfo_.realGSize - queryStart);
            const uint32_t alignedRows = (rows + 31U) & ~31U;
            uint32_t kvHead = 0;
            int64_t causalOffset = 0;
            bool splitRow = false;
            uint32_t workspaceSlot = 0;
            if constexpr (isCube) {
                kvHead = USE_DN ? realHead / constInfo_.gSize : realHead;
            } else {
                causalOffset = static_cast<int64_t>(actSeqLensKv_) - static_cast<int64_t>(actSeqLensQ_);
                if constexpr (FLASH_DECODE) {
                    const bool firstRow = head == bN2Start_ && row == gS1OStart_;
                    const bool lastRow = head == bN2End_ && row == gS1OEnd_;
                    splitRow = (firstRow && lastRow) || (firstRow && headS2Split_) || (lastRow && tailS2Split_);
                    workspaceSlot = coreFirstTmpOutWsPos_ + (!firstRow && lastRow && headS2Split_ && tailS2Split_);
                }
            }

            for (; s2 < curS2End_; ++s2) {
                Task& task = tasks[created % PRELOAD_TASK_CACHE_SIZE];
                task.loop = created;
                task.bIdx = prevBIdx_;
                task.realN2Idx = realHead;
                task.gS1Idx = queryStart;
                task.actMSize = rows;
                task.actMSizeAlign32 = alignedRows;
                task.s2Idx = s2 * s2BaseSize;
                task.actSingleLoopS2Size = AttentionCommon::Min(static_cast<uint64_t>(s2BaseSize),
                    actSeqLensKv_ - task.s2Idx);
                task.isLastS2Loop = s2 + 1U == curS2End_;
                task.isGroupLast = task.isLastS2Loop || (s2 - curS2Start_ + 1U) % GROUP_N == 0U;
                if constexpr (isCube) {
                    task.n2Idx = kvHead;
                    if (rows != 0 && task.actSingleLoopS2Size != 0) {
                        cubeBlock_.CacheKvPageIndices(task);
                        cubeBlock_.template PrefetchKvL1<true>(task, KeySlotOf(task.loop));
                    }
                } else {
                    task.isFirstS2Loop = created == 0 || s2 == curS2Start_;
                    task.causalOffset = causalOffset;
                    if constexpr (FLASH_DECODE) {
                        task.isS2SplitCore = splitRow;
                        task.faTmpOutWsPos = workspaceSlot;
                    }
                }
                ++created;
                // K for tick+LEAD has been issued. Advance one tick before reusing its parity slot.
                if (created > KV_PREFETCH_LEAD) {
                    ExecuteTick(tick++, created, tasks);
                }
            }
            UpdateAxisInfo(TASK_DEAL_MODE::SKIP_REMAINING_S2, head, row, s2);
        }
        if (created != 0) {
            while (tick < created + PRELOAD_N) {
                ExecuteTick(tick++, created, tasks);
            }
        }
    }

    __aicore__ inline void ExecuteTick(uint64_t tick, uint64_t created, Mxfp4CubeTask* tasks)
    {
        if (tick + KV_PREFETCH_LEAD >= PRELOAD_N) {
            const uint64_t v = tick + KV_PREFETCH_LEAD - PRELOAD_N;
            if (v < created) {
                auto& task = tasks[v % PRELOAD_TASK_CACHE_SIZE];
                if (task.actMSize != 0 && task.actSingleLoopS2Size != 0) {
                    cubeBlock_.template PrefetchKvL1<false>(task, ValueSlotOf(task.loop));
                }
            }
        }
        if (tick >= PRELOAD_N) {
            auto& pv = tasks[(tick - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE];
            cubeBlock_.ComputeBmm2(pv, ValueSlotOf(pv.loop));
        }
        if (tick < created) {
            auto& qk = tasks[tick % PRELOAD_TASK_CACHE_SIZE];
            cubeBlock_.ComputeBmm1(qk, KeySlotOf(qk.loop));
        }
    }

    __aicore__ inline void ExecuteTick(uint64_t tick, uint64_t created, Mxfp4VecTask* tasks)
    {
        if (tick >= PRELOAD_N) {
            auto& v2 = tasks[(tick - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE];
            if (v2.isGroupLast) {
                vecFaBlock_.ComputeVec2(v2);
            }
        }
        if (tick >= DELAY_P_SCALE_N && tick - DELAY_P_SCALE_N < created) {
            auto& scale = tasks[(tick - DELAY_P_SCALE_N) % PRELOAD_TASK_CACHE_SIZE];
            if (scale.isGroupLast) {
                vecFaBlock_.UpdatePScale(scale);
            }
        }
        if (tick < created) {
            vecFaBlock_.ComputeVec1(tasks[tick % PRELOAD_TASK_CACHE_SIZE]);
        }
    }

    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        if (bN2Cur != bN2End_) { // head 维度还没有到最后一个head，肯定有任务
            return bN2Cur < bN2End_;
        }
        if (gS1Cur != gS1OEnd_) { // 到最后一个head，这个head还没有到最后一个M块，肯定有任务
            return gS1Cur < gS1OEnd_;
        }
        return s2Cur < s2OEnd_; // 到最后一个head的最后一个M块，是否到了最后一个kv块

    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t& bN2Cur, uint32_t& gS1Cur,
                                          uint32_t& s2Cur)
    {
        uint64_t s2LoopTimes = (actSeqLensKv_ + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ_ * constInfo_.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        // 尚未到达有效区间, 快进s2Cur到curS2Start_, 不跳行
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start_;
            return;
        }
        // 如果是不跳过剩余S2的情况，在s2没处理完时，继续下一个s2块
        if (taskDealMode != TASK_DEAL_MODE::SKIP_REMAINING_S2) {
            // 当前S2未处理完
            if (s2Cur + 1 < s2LoopTimes) {
                s2Cur++;
                return;
            }
        }

        // 当前BN2未处理完，更新M块行号
        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }

        // 当前BN2已处理完，跳到下一个head
        gS1Cur = 0;
        bN2Cur++;
    }


    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask = (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_) && (s2Cur == s2OStart_); // 首个任务
        uint32_t bIdx = bN2Cur / constInfo_.realN2Size;
        if (isFirstTask || prevBIdx_ != bIdx) { // 换了一个batch，更新q和kv的长度
            prevBIdx_ = bIdx;
            if constexpr (LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
                actSeqLensKv_ = kvCuSeqLensParser_.GetActualSeqLength(bIdx);
            } else {
                actSeqLensKv_ = kvSeqUsedParser_.GetActualSeqLength(bIdx);
            }
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                actSeqLensQ_ = qCuSeqLensParser_.GetActualSeqLength(bIdx);
            } else {
                actSeqLensQ_ = qSeqUsedParser_.GetActualSeqLength(bIdx);
            }
        }
        uint64_t s2LoopTimes = (actSeqLensKv_ + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ_ * constInfo_.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) { // batch 空的任务
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO; // 第一个空任务，需要清零输出
            }
            return TASK_DEAL_MODE::SKIP_ZERO; // 跳过
        }

        // 计算每一行的起止点，只有当换行时（bN2Cur、gS1Cur更新）才需要重新计算 S2 的起点和终点
        // 有mask和没有mask的计算方式不同
        if (isFirstTask || bN2Cur != prevBN2Idx_ || gS1Cur != prevGS1Idx_) {
            if constexpr (!HAS_MASK) {
                CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
            }
            prevBN2Idx_ = bN2Cur;
            prevGS1Idx_ = gS1Cur;
        }

        // S2有效块区间为[curS2Start_, curS2End_), s2Cur尚未到达有效区间且该行有有效块,
        // 需快进到curS2Start_继续计算, 不能跳行 (BAND等sparse模式curS2Start_常>0)
        if (s2Cur < curS2Start_ && curS2Start_ < curS2End_) {
            return TASK_DEAL_MODE::NOT_START;
        }
        // 该行无有效块(curS2Start_>=curS2End_)或s2Cur已越过有效区间, 跳过当前行
        if (s2Cur < curS2Start_ || s2Cur >= curS2End_) {
            return TASK_DEAL_MODE::SKIP_REMAINING_S2;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start_ = 0U;
        curS2End_ = (static_cast<uint32_t>(actSeqLensKv_) + s2BaseSize - 1) / s2BaseSize;
        // 只有第一个开始的M行和最后一个结束的M行，s2的开始和结束才可能不是0和最后一个块
        // 中间的M行一定会处理一整行kv
        if ((bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
            if constexpr (TRACK_FD_SPLIT) {
                headS2Split_ = s2OStart_ != 0U;
            }
            curS2Start_ = s2OStart_;
        }

        if ((bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
            if constexpr (TRACK_FD_SPLIT) {
                tailS2Split_ = s2OEnd_ != 0U;
            }
            curS2End_ = s2OEnd_;
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        // 计算左右的边界，对一个q token，其可见范围为 [q_idx-preTokenLeftUp, q_idx+nextTokenLeftUp]，q_idx的意思是其在当前输入的query token 中的下标
        // 对于我们常用的右下角causcal，nextTokenLeftUp = actSeqLensKv_ - actSeqLensQ_
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(actSeqLensQ_, actSeqLensKv_, preTokenLeftUp, nextTokenLeftUp);

        // 计算当前M块的第一个token行和最后一个token行，包括合轴的行
        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        int64_t s1GLastToken =
            AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                 static_cast<int64_t>(actSeqLensQ_) * static_cast<int64_t>(constInfo_.realGSize)) -
            1;

        // 实际的 token 编号，去掉合轴的行
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<LAYOUT_Q>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo_.realGSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo_.realGSize);
        } else {
            if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ_) ==
                s1GLastToken / static_cast<int64_t>(actSeqLensQ_)) {
                // start and end locate in one G
                s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ_);
                s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ_);
            } else {
                // start and end locate in tow or more G, but working same as crossing one complete block
                s1FirstToken = 0;
                s1LastToken = static_cast<int64_t>(actSeqLensQ_);
            }
        }

        // 3. 变成块编号
        uint32_t s2StartWithSparse = 0U;
        uint32_t s2EndWithSparse = 0U;
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp; // 是左右闭区间
        // no valid token
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv_) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start_ = 0U;
            curS2End_ = 0U;
            return;
        }
        // get valid range
        s2FirstToken = ClipMxfp4SInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));
        s2LastToken = ClipMxfp4SInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));

        s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U; // +1 变成左右开区间

        // 4. 计算 curS2Start_, curS2End_
        curS2Start_ = s2StartWithSparse;
        curS2End_ = s2EndWithSparse;

        if (bN2Cur == bN2Start_ && gS1Cur == gS1OStart_) { // first line
            if constexpr (TRACK_FD_SPLIT) {
                headS2Split_ = s2OStart_ > s2StartWithSparse;
            }
            curS2Start_ = AttentionCommon::Max(s2StartWithSparse, s2OStart_);
        }
        if (bN2Cur == bN2End_ && gS1Cur == gS1OEnd_) { // last line
            if constexpr (TRACK_FD_SPLIT) {
                tailS2Split_ = s2OEnd_ > 0U;
            }
            curS2End_ = s2OEnd_ > 0U ? AttentionCommon::Min(s2EndWithSparse, s2OEnd_) : s2EndWithSparse;
        }
        return;
    }

    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t& preTokenLeftUp,
                                                 int64_t& nextTokenLeftUp)
    {
        preTokenLeftUp = constInfo_.preTokens;
        nextTokenLeftUp = constInfo_.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                        constInfo_.sparseMode); // 更新preTokenLeftUp和nextTokenLeftUp

        if (constInfo_.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
        }

        if (constInfo_.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL ||
            constInfo_.sparseMode == fa_base_vector::TREE) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
        } else if (constInfo_.sparseMode == fa_base_vector::BAND) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
        }
    }

    __aicore__ inline void FlashDecode(uint32_t sectionIdx)
    {
        // FA has retired every PV consumer. Reuse PV/history O, preserving score
        // buffers because AIC can start the next section before AIV finishes FD.
        static_assert(VecFdBlockType::GetFdTotalUbSize() <= VecFaBlockType::GetFdReuseUbSize(),
                      "FD exceeds reusable PV/history UB region");
        if (!constInfo_.enableFlashDecode) {
            return;
        }
        GetFDSectionInfo(sectionIdx);
        vecFdBlock_.template InitBuffers<VecFaBlockType::GetFdReuseUbOffset()>();
        AscendC::ICachePreLoad(2);
        SyncAll();
        vecFdBlock_.template FlashDecode<VecFaBlockType::QUANT_SCHEMA_VALUE>(fdParams_);
        SyncAll(); // 多section场景: 保证所有核FD读workspace完成后, 才能进入下一section FA覆写
    }

    __aicore__ inline void GetFDSectionInfo(uint32_t sectionIdx)
    {
        fdParams_.mLen = fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_M_NUM_INDEX, sectionIdx));
        fdParams_.fdCoreEnable = fdParams_.mLen > 0 ? 1U : 0U;
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        fdParams_.fdBN2Idx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_BN2_IDX_INDEX, sectionIdx));
        fdParams_.fdMIdx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_M_IDX_INDEX, sectionIdx));
        fdParams_.fdWorkspaceIdx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_WORKSPACE_IDX_INDEX, sectionIdx));
        fdParams_.fdS2SplitNum =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_WORKSPACE_NUM_INDEX, sectionIdx));
        fdParams_.mStart =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_M_START_INDEX, sectionIdx));
    }


    __aicore__ inline void Process()
    {
        if (constInfo_.aicIdx < constInfo_.coreNum) {
            if ASCEND_IS_AIV {
                vecFaBlock_.InitBuffers();
                vecFaBlock_.AllocEventID();
            } else {
                cubeBlock_.InitBuffers();
                cubeBlock_.AllocEventID();
            }
        }
        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfo_.aicIdx < constInfo_.coreNum) {
                FlashAttention(sectionIdx);
            }
            if constexpr (FLASH_DECODE) {
                if ASCEND_IS_AIV {
                    FlashDecode(sectionIdx);
                }
            }
        }

        if (constInfo_.aicIdx < constInfo_.coreNum) {
            if ASCEND_IS_AIV {
                vecFaBlock_.FreeEventID();
            } else {
                cubeBlock_.FreeEventID();
            }
        }



    }




};

} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_KERNEL_MXFP4_H_
