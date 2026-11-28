/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_MXFP4_H_
#define QUANT_FLASH_ATTN_BLOCK_VEC_MXFP4_H_
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "memory_copy_arch35_quant_flash_attn.h"
#include "quant_flash_attn_common_def.h"
#include "quant_flash_attn_mxfp4_cross_buffer.h"
#include "quant_flash_attn_block_vec_flashdecode.h"
#include "vf/vf_update_fp32_mxfp4.h"
#include "vf/vf_output_update_fp32_mxfp4.h"

namespace BaseApi {
using namespace AscendC;
using namespace regbaseutil;
using namespace AttentionCommon;

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool isFd = false, bool useDn = false, bool isDAligned = true,
          QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT, bool enableQf322F32Pre = false>
class QuantFlashAttnBlockVecMxfp4 {
public:
    static constexpr uint32_t mBaseSize = static_cast<uint32_t>(s1TemplateType);
    static constexpr uint32_t s2BaseSize = static_cast<uint32_t>(s2TemplateType);
    static constexpr uint32_t dBaseSize = static_cast<uint32_t>(dTemplateType);
    static constexpr uint32_t dVBaseSize = static_cast<uint32_t>(dVTemplateType);
    static constexpr uint32_t PRELOAD_N = 20;
    static constexpr uint32_t TASKS_PER_GROUP = 16;
    // Retire C2/V2 before issuing V1: at most one live group per in-flight task.
    static constexpr uint32_t GROUP_MAX_SLOTS = PRELOAD_N;
    static constexpr uint32_t VEC_ROWS = 64;
    static constexpr uint32_t s2SplitSize = 128U;
    static constexpr uint32_t P_SLOT_COUNT = 2;
    static constexpr uint32_t P_PACK_BYTES = VEC_ROWS * s2BaseSize / 2;
    static constexpr uint32_t TASK_GROUP_MAX_ELEMENTS = VEC_ROWS * s2BaseSize / 32;
    static constexpr uint32_t LOCAL_PSCALE_BYTES = VEC_ROWS / 16 * (s2BaseSize / 64 + 1) * 32;
    static constexpr bool REUSE_PV_OUTPUT = dBaseSize == 256;
    static constexpr uint32_t OUTPUT_SCRATCH_BYTES = 64 * 256 * sizeof(bfloat16_t);
    static constexpr uint32_t STATISTICS_BYTES = VEC_ROWS * 8 * sizeof(float);
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    static constexpr bool USE_DN = useDn;
    static constexpr QuantSchema QUANT_SCHEMA_VALUE = QUANT_SCHEMA;
    static_assert(QUANT_SCHEMA == QuantSchema::OCP_QUANT || QUANT_SCHEMA == QuantSchema::CX_QUANT ||
                  QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT, "Unsupported MXFP4 P quantization schema");
    static constexpr bool ENABLE_QF322F32_PRE = enableQf322F32Pre;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static_assert(mBaseSize == 128 && dBaseSize == dVBaseSize &&
        ((dBaseSize == 128 && s2BaseSize == 256) || (dBaseSize == 256 && s2BaseSize == 128)),
        "MXFP4 Vector 只支持 M128/D128/N256 和 M128/D256/N128");
    static_assert(layout == LayOutTypeEnum::LAYOUT_TND, "MXFP4 首版只支持 TND Q");
    static_assert(outLayout == LayOutTypeEnum::LAYOUT_TND || outLayout == LayOutTypeEnum::LAYOUT_NTD,
        "MXFP4 输出布局必须为 TND 或 NTD");
    static_assert(IsSameType<T, float>::value && IsSameType<OUTPUT_T, bfloat16_t>::value,
        "MXFP4 使用 FP32 中间计算和 BF16 输出");
    using OUT_T = OUTPUT_T;
    using MM1_OUT_T = T;
    using MM2_OUT_T = T;
    using ConstInfoX = ConstInfo_t;
    using QSeqParserType = ActualSeqLensParser<Q_MODE, int32_t, true>;
    using CrossBufferType = CVCrossBuffer<PRELOAD_N, USE_DN, s1TemplateType, s2TemplateType,
                                        dTemplateType, dVTemplateType>;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;
    GlobalTensor<float> accumOutGm_;
    GlobalTensor<float> softmaxFDSumGm_;
    GlobalTensor<float> softmaxFDMaxGm_;
    // FD carries integer binary references and raw PV; C4 never uses C8's external Pscale.
    float pScaleValue_ = 1.0F;

    __aicore__ inline QuantFlashAttnBlockVecMxfp4(ConstInfo_t& constInfo) : constInfo_(constInfo) {}

    __aicore__ inline void InitVecBlock(TPipe* pipe, __gm__ uint8_t* actualSeqQlenAddr,
        __gm__ uint8_t* actualSeqKvlenAddr, __gm__ uint8_t* pScale, __gm__ uint8_t* attenMask,
        __gm__ uint8_t* softmaxLse, __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace)
    {
        tPipe_ = pipe;
        attentionOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OUTPUT_T*>(attentionOut));
        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(softmaxLse));
        }
        if constexpr (FLASH_DECODE) {
            if (constInfo_.enableFlashDecode) {
                accumOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
                softmaxFDSumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + constInfo_.accumOutSize);
                softmaxFDMaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + constInfo_.accumOutSize +
                                              constInfo_.logSumExpSize);
            }
        }
    }

    __aicore__ inline void SetCuSeqLensParser(QSeqParserType& queryParser)
    {
        querySeqParser_ = &queryParser;
    }

    // 在分配常驻 UB 前调用，保持与主控初始化次序一致；所有 AIV 都必须到达初始化屏障。
    __aicore__ inline void ClearOutput()
    {
        if (!constInfo_.needInitOutput) {
            return;
        }
        const uint64_t totalQueryRows = querySeqParser_->GetTSize() * constInfo_.realN2Size * constInfo_.realGSize;
        const uint64_t rowsPerCore = (totalQueryRows + 2 * constInfo_.coreNum - 1) / (2 * constInfo_.coreNum);
        const uint64_t firstRow = static_cast<uint64_t>(constInfo_.aivIdx) * rowsPerCore;
        if (firstRow < totalQueryRows) {
            const uint64_t rowCount = AttentionCommon::Min(rowsPerCore, totalQueryRows - firstRow);
            matmul::InitOutput<OUTPUT_T>(attentionOutGm_[firstRow * constInfo_.dSizeV],
                rowCount * constInfo_.dSizeV, 0);
            // InitOutput 复用内部 UB；上一段 MTE3 必须完成后才能用同一区域初始化 LSE。
            PipeBarrier<PIPE_ALL>();
            if (constInfo_.isSoftmaxLseEnable) {
                matmul::InitOutput<float>(softmaxLseGm_[firstRow], rowCount, 3e+99);
            }
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
    }

    __aicore__ inline void InitBuffers()
    {
        l1BufferManager_.Init(tPipe_, 512 * 1024);
        ubBufferManager_.Init(tPipe_, GetCrossBufferBytes());
        crossBuffers_.InitBuffers(l1BufferManager_, ubBufferManager_);
        // Layout contract for FD reuse: score[2], PV reception, then contiguous history O.
        // FD must never touch score: AIC may issue the next section's C1 during FD.
        tPipe_->InitBuffer(historyOBuf_, VEC_ROWS * dVBaseSize * sizeof(float));
        tPipe_->InitBuffer(globalRowSum_, VEC_ROWS * sizeof(float));
        tPipe_->InitBuffer(taskGroupMax_, TASK_GROUP_MAX_ELEMENTS * PRELOAD_N * sizeof(float));
        tPipe_->InitBuffer(globalCurrentMax_, GROUP_MAX_SLOTS * VEC_ROWS * sizeof(float));
        tPipe_->InitBuffer(globalOldMax_, VEC_ROWS * sizeof(float));
        tPipe_->InitBuffer(alphaUpdate_, GROUP_MAX_SLOTS * VEC_ROWS * sizeof(float));
        tPipe_->InitBuffer(pNd2NzIdx_, 256);
        tPipe_->InitBuffer(pNd2NzHalfwordIdx_, 256);
        tPipe_->InitBuffer(pBuffers_[0], P_PACK_BYTES);
        tPipe_->InitBuffer(pBuffers_[1], P_PACK_BYTES);
        tPipe_->InitBuffer(tempCommBuf_, GetTransientUbSize());
        // Fixed-lane packing no longer reads the legacy index tables. Keep
        // their reserved regions so this change does not move any UB view.
    }

    static __aicore__ inline constexpr uint32_t GetCrossBufferBytes()
    {
        return CrossBufferType::SCORE_SLOT_COUNT * CrossBufferType::ubBmm1BufferBytes_ + CrossBufferType::ubBmm2BufferBytes_;
    }

    static __aicore__ inline constexpr uint32_t GetPersistUbSize()
    {
        return VEC_ROWS * dVBaseSize * sizeof(float) + 2 * VEC_ROWS * sizeof(float) +
            TASK_GROUP_MAX_ELEMENTS * PRELOAD_N * sizeof(float) +
            2 * GROUP_MAX_SLOTS * VEC_ROWS * sizeof(float) + 512 + P_SLOT_COUNT * P_PACK_BYTES;
    }

    static __aicore__ inline constexpr uint32_t GetFdReuseUbOffset()
    {
        return CrossBufferType::SCORE_SLOT_COUNT * CrossBufferType::ubBmm1BufferBytes_;
    }

    static __aicore__ inline constexpr uint32_t GetFdReuseUbSize()
    {
        return CrossBufferType::ubBmm2BufferBytes_ + VEC_ROWS * dVBaseSize * sizeof(float);
    }

    static __aicore__ inline constexpr uint32_t GetTransientUbSize()
    {
        // P 使用独立双槽；D256 输出复用已消费的 PV，独立临时区只保留 Pscale。
        // FD 另行复用已退休的 PV/history O，不占用此临时区。
        return (REUSE_PV_OUTPUT ? 0U : OUTPUT_SCRATCH_BYTES) + 3 * STATISTICS_BYTES;
    }
    static_assert(GetFdReuseUbOffset() % 32 == 0 && GetFdReuseUbSize() % 32 == 0,
        "MXFP4 FD reusable UB region must be 32-byte aligned");
    static_assert(TASKS_PER_GROUP * LOCAL_PSCALE_BYTES <= GetTransientUbSize(),
        "MXFP4 整组 Pscale 超出临时 UB 容量");
    static_assert(P_SLOT_COUNT == 2, "MXFP4 P uses two separately allocated UB buffers");
    static_assert(!REUSE_PV_OUTPUT || CrossBufferType::ubBmm2BufferBytes_ >=
        OUTPUT_SCRATCH_BYTES + 3 * STATISTICS_BYTES, "PV region cannot hold output scratch");
    static_assert(GetCrossBufferBytes() + GetPersistUbSize() + GetTransientUbSize() <= 248 * 1024,
        "MXFP4 Vector UB 超出 248KiB 业务预算");

    __aicore__ inline void AllocEventID()
    {
        // 持久事件必须 Alloc 占用 ID；Fetch 只查询空闲 ID，会令多个槽取到同一 ID。
        vectorToCopyEvent_ = tPipe_->AllocEventID<HardEvent::V_MTE3>();
        copyToVectorEvent_ = tPipe_->AllocEventID<HardEvent::MTE3_V>();
        for (uint32_t slot = 0; slot < P_SLOT_COUNT; ++slot) {
            pVectorToCopyEvents_[slot] = tPipe_->AllocEventID<HardEvent::V_MTE3>();
            pCopyToVectorEvents_[slot] = tPipe_->AllocEventID<HardEvent::MTE3_V>();
            SetFlag<HardEvent::MTE3_V>(pCopyToVectorEvents_[slot]);
        }
        crossBuffers_.InitSync();
    }

    __aicore__ inline void FreeEventID()
    {
        // 回收最后一次 P 搬运（或从未使用槽的初始信用），再释放事件。
        for (uint32_t slot = 0; slot < P_SLOT_COUNT; ++slot) {
            WaitFlag<HardEvent::MTE3_V>(pCopyToVectorEvents_[slot]);
            tPipe_->ReleaseEventID<HardEvent::V_MTE3>(pVectorToCopyEvents_[slot]);
            tPipe_->ReleaseEventID<HardEvent::MTE3_V>(pCopyToVectorEvents_[slot]);
        }
        tPipe_->ReleaseEventID<HardEvent::V_MTE3>(vectorToCopyEvent_);
        tPipe_->ReleaseEventID<HardEvent::MTE3_V>(copyToVectorEvent_);
    }

    __aicore__ inline void ComputeVec1(Mxfp4VecTask& runInfo)
    {
        if (runInfo.actMSize == 0 || runInfo.actSingleLoopS2Size == 0) {
            return;
        }
        const uint32_t taskSlot = runInfo.loop % PRELOAD_N;
        const uint32_t probabilitySlot = runInfo.loop % P_SLOT_COUNT;
        const uint32_t firstRow = constInfo_.subBlockIdx * GetAivRowCapacity(runInfo);
        const uint32_t validRows = GetValidQueryRows(runInfo);
        // 只在本槽再次写入前等待 MTE3，另一槽的 P 搬运不阻塞本次 V1。
        WaitFlag<HardEvent::MTE3_V>(pCopyToVectorEvents_[probabilitySlot]);
        LocalTensor<uint8_t> packedQuantP = pBuffers_[probabilitySlot].Get<uint8_t>();
        LocalTensor<float> taskExponent = taskGroupMax_.Get<float>()[taskSlot * TASK_GROUP_MAX_ELEMENTS]; // 取对应的max存储位置
        LocalTensor<float> groupMaximum = BeginProbabilityGroup(runInfo); // 判断当前是否是一个新的group， 取当前group的max
        LocalTensor<uint8_t> byteIndices = pNd2NzIdx_.Get<uint8_t>();
        LocalTensor<uint16_t> halfwordIndices = pNd2NzHalfwordIdx_.Get<uint16_t>();
        Duplicate<uint32_t>(packedQuantP.ReinterpretCast<uint32_t>(), 0, P_PACK_BYTES / sizeof(uint32_t));
        Duplicate<float>(taskExponent, Mxfp4Api::INVALID_GROUP_EXPONENT, TASK_GROUP_MAX_ELEMENTS);
        PipeBarrier<PIPE_V>();
        for (uint32_t chunkOffset = 0; chunkOffset < runInfo.actSingleLoopS2Size; chunkOffset += s2SplitSize) {
            const uint32_t scoreSlot = scoreSequence_++ % CrossBufferType::SCORE_SLOT_COUNT;
            crossBuffers_.waitBmm1Buffer(scoreSlot);
            if (validRows != 0) {
                Mxfp4Api::SoftmaxFp32Mxfp4Params params {};
                params.validRows = validRows;
                params.validN = AttentionCommon::Min(s2SplitSize, runInfo.actSingleLoopS2Size - chunkOffset);
                params.s2Offset = chunkOffset;
                params.qGs1Start = static_cast<uint64_t>(runInfo.gS1Idx) + firstRow;
                params.qGroupSize = USE_DN ? 1U : constInfo_.realGSize;
                params.kvStart = static_cast<uint64_t>(runInfo.s2Idx) + chunkOffset;
                params.causalOffset = runInfo.causalOffset;
                params.scoreScale = constInfo_.scaleValue;
                LocalTensor<float> score = crossBuffers_.GetBmm1Slot(scoreSlot);
                if constexpr (USE_DN) {
                    if (ENABLE_QF322F32_PRE && CrossBufferType::CanScaleInFixpipe(params.scoreScale)) {
                        Mxfp4Api::SoftmaxDnChunkFp32<s2SplitSize, s2BaseSize, HAS_MASK, QUANT_SCHEMA, true>(
                            packedQuantP, taskExponent, groupMaximum, score, byteIndices, halfwordIndices, params);
                    } else {
                        Mxfp4Api::SoftmaxDnChunkFp32<s2SplitSize, s2BaseSize, HAS_MASK, QUANT_SCHEMA, false>(
                            packedQuantP, taskExponent, groupMaximum, score, byteIndices, halfwordIndices, params);
                    }
                } else {
                    if (ENABLE_QF322F32_PRE && CrossBufferType::CanScaleInFixpipe(params.scoreScale)) {
                        Mxfp4Api::SoftmaxNdChunkFp32<s2SplitSize, s2BaseSize, HAS_MASK, QUANT_SCHEMA, true>(
                            packedQuantP, taskExponent, groupMaximum, score, byteIndices, halfwordIndices, params);
                    } else {
                        Mxfp4Api::SoftmaxNdChunkFp32<s2SplitSize, s2BaseSize, HAS_MASK, QUANT_SCHEMA, false>(
                            packedQuantP, taskExponent, groupMaximum, score, byteIndices, halfwordIndices, params);
                    }
                }
                PipeBarrier<PIPE_V>();
            }
            // 无有效 Q 的 AIV 仍须消费并归还每个 score 槽。
            crossBuffers_.setBmm1Buffer(scoreSlot);
        }
        SetFlag<HardEvent::V_MTE3>(pVectorToCopyEvents_[probabilitySlot]);
        WaitFlag<HardEvent::V_MTE3>(pVectorToCopyEvents_[probabilitySlot]);
        CopyProbabilityToL1(crossBuffers_.GetPSlot(taskSlot), packedQuantP, runInfo);
        SetFlag<HardEvent::MTE3_V>(pCopyToVectorEvents_[probabilitySlot]);
        SealProbabilityGroup(runInfo);
    }

    __aicore__ inline void UpdatePScale(const Mxfp4VecTask& runInfo)
    {
        GroupOwner& owner = GetGroupOwner(runInfo, GroupState::SEALED);
        ASSERT(runInfo.isGroupLast || runInfo.isLastS2Loop);
        const uint32_t groupSlot = runInfo.v1GroupId % GROUP_MAX_SLOTS;
        LocalTensor<float> reference = globalCurrentMax_.Get<float>()[groupSlot * VEC_ROWS];
        LocalTensor<float> previousReference = globalOldMax_.Get<float>();
        LocalTensor<float> multiplier = alphaUpdate_.Get<float>()[groupSlot * VEC_ROWS];
        const uint32_t firstTaskSlot = owner.firstTaskLoop % PRELOAD_N;
        const uint16_t firstTaskCount = static_cast<uint16_t>(
            owner.taskCount < PRELOAD_N - firstTaskSlot ? owner.taskCount : PRELOAD_N - firstTaskSlot);
        const uint16_t secondTaskCount = static_cast<uint16_t>(owner.taskCount - firstTaskCount);
        LocalTensor<float> firstTaskExponent = taskGroupMax_.Get<float>()[
            firstTaskSlot * TASK_GROUP_MAX_ELEMENTS];
        LocalTensor<float> secondTaskExponent = taskGroupMax_.Get<float>();
        LocalTensor<uint8_t> encodedScale = tempCommBuf_.Get<uint8_t>();
        if constexpr (USE_DN) {
            Mxfp4Api::UpdatePScaleDn<s2BaseSize, QUANT_SCHEMA>(encodedScale,
                firstTaskExponent, secondTaskExponent, reference, previousReference, multiplier,
                firstTaskCount, secondTaskCount, owner.firstGroup);
        } else {
            Mxfp4Api::UpdatePScaleNd<s2BaseSize, QUANT_SCHEMA>(encodedScale,
                firstTaskExponent, secondTaskExponent, reference, previousReference, multiplier,
                firstTaskCount, secondTaskCount, owner.firstGroup, GetAivRowCapacity(runInfo));
        }
        WaitVectorBeforeCopy();
        const uint32_t copiedScaleBytes = GetAivRowCapacity(runInfo) / 16 * (s2BaseSize / 64 + 1) * 32;
        const uint32_t scaleOffset = constInfo_.subBlockIdx * copiedScaleBytes;
        const uint16_t sourceStride = static_cast<uint16_t>((LOCAL_PSCALE_BYTES - copiedScaleBytes) / 32);
        const uint16_t destinationStride = static_cast<uint16_t>(
            (CrossBufferType::l1PScaleBufferBytes_ - copiedScaleBytes) / 32);
        LocalTensor<uint8_t> firstScaleSlot = crossBuffers_.GetPScaleSlot(firstTaskSlot);
        DataCopy(firstScaleSlot[scaleOffset], encodedScale,
            {firstTaskCount, static_cast<uint16_t>(copiedScaleBytes / 32), sourceStride, destinationStride});
        if (secondTaskCount != 0) {
            LocalTensor<uint8_t> secondScaleSlot = crossBuffers_.GetPScaleSlot(0);
            DataCopy(secondScaleSlot[scaleOffset], encodedScale[firstTaskCount * LOCAL_PSCALE_BYTES],
                {secondTaskCount, static_cast<uint16_t>(copiedScaleBytes / 32), sourceStride, destinationStride});
        }
        WaitCopyBeforeVector();
        owner.state = GroupState::READY;
        // 一组只发布一次；Cube 在本组第一个 C2 等待两侧 AIV，后续 C2 复用该组的 scale。
        crossBuffers_.setPscaleUpdate(runInfo.v1GroupId);
    }

    __aicore__ inline void ComputeVec2(const Mxfp4VecTask& runInfo)
    {
        GroupOwner& owner = GetGroupOwner(runInfo, GroupState::READY);
        ASSERT(runInfo.isGroupLast || runInfo.isLastS2Loop);
        const uint32_t groupSlot = runInfo.v1GroupId % GROUP_MAX_SLOTS;
        LocalTensor<float> groupResult = crossBuffers_.GetBmm2Slot(0);
        LocalTensor<float> historyOutput = historyOBuf_.Get<float>();
        LocalTensor<float> historySum = globalRowSum_.Get<float>();
        LocalTensor<float> multiplier = alphaUpdate_.Get<float>()[groupSlot * VEC_ROWS];
        LocalTensor<float> reference = globalCurrentMax_.Get<float>()[groupSlot * VEC_ROWS];
        crossBuffers_.watBmm2Buffer();
        if constexpr (USE_DN) {
            LocalTensor<float> groupSum = groupResult[dVBaseSize * VEC_ROWS];
            if (owner.firstGroup) {
                Mxfp4Api::processUpdateDn<false, dVBaseSize>(historyOutput, groupResult, multiplier,
                    historySum, groupSum, GetValidQueryRows(runInfo));
            } else {
                Mxfp4Api::processUpdateDn<true, dVBaseSize>(historyOutput, groupResult, multiplier,
                    historySum, groupSum, GetValidQueryRows(runInfo));
            }
        } else {
            if (owner.firstGroup) {
                Mxfp4Api::processUpdateNd<false, dVBaseSize>(historyOutput, groupResult, multiplier,
                    historySum, GetValidQueryRows(runInfo));
            } else {
                Mxfp4Api::processUpdateNd<true, dVBaseSize>(historyOutput, groupResult, multiplier,
                    historySum, GetValidQueryRows(runInfo));
            }
        }
        PipeBarrier<PIPE_V>();
        // Reused PV remains owned by V2 until every output MTE3 read completes.
        if constexpr (REUSE_PV_OUTPUT) {
            if (runInfo.isLastS2Loop) {
                WriteAttentionResult(runInfo, historyOutput, historySum, reference);
            }
        }
        crossBuffers_.setBmm2Buffer();
        if constexpr (!REUSE_PV_OUTPUT) {
            if (runInfo.isLastS2Loop) {
                WriteAttentionResult(runInfo, historyOutput, historySum, reference);
            }
        }
        // 输出搬运及本组参考、alpha 的所有读取均已结束，才允许未来 V1 复用本组槽。
        PipeBarrier<PIPE_V>();
        owner.state = GroupState::FREE;
    }

private:
    enum class GroupState : uint8_t { FREE, ACCUMULATING, SEALED, READY };
    struct GroupOwner {
        uint64_t id = ~uint64_t{0}; // 当前槽所属的 group 编号
        GroupState state = GroupState::FREE; // 前状态
        uint32_t firstTaskLoop = 0; // 组内第一个 task 的 loop
        uint32_t taskCount = 0; // 当前组实际包含多少个 task
        bool firstGroup = false; // 是否为当前 Q 块/S2 分段的首组
    };
    ConstInfo_t& constInfo_;
    TPipe* tPipe_ = nullptr;
    QSeqParserType* querySeqParser_ = nullptr;
    BufferManager<BufferType::UB> ubBufferManager_; // 用于初始化vector和cube共享的buffer
    BufferManager<BufferType::L1> l1BufferManager_; // 用于初始化vector和cube共享的buffer
    CrossBufferType crossBuffers_; // vector和cube共享的buffer，包括ub上的mm1和mm2的结果，以及l1上的p和pscale
    TBuf<TPosition::VECCALC> historyOBuf_; // DN: [Dv,64] 未归一化；ND: [64,Dv]
    TBuf<TPosition::VECCALC> globalRowSum_; // 每行的和，softmax的分母
    TBuf<TPosition::VECCALC> taskGroupMax_; // 每32个元素一组的max
    TBuf<TPosition::VECCALC> globalCurrentMax_; // 每行的最大值，当前块的结果
    TBuf<TPosition::VECCALC> globalOldMax_; // 每行的最大值，旧的块的结果
    TBuf<TPosition::VECCALC> alphaUpdate_; // 用于更新 O
    TBuf<TPosition::VECCALC> pNd2NzIdx_; // Reserved legacy index view; fixed packing does not read it.
    TBuf<TPosition::VECCALC> pNd2NzHalfwordIdx_; // Reserved to keep the verified UB layout stable.
    TBuf<TPosition::VECCALC> pBuffers_[P_SLOT_COUNT]; // 每槽保存一个 task 的本 AIV FP4 P
    TBuf<TPosition::VECCALC> tempCommBuf_; // Pscale 编码、输出转换和统计，不与 P 双槽混用
    GroupOwner groupOwners_[GROUP_MAX_SLOTS];
    uint64_t scoreSequence_ = 0; // 连续编号每个 QK score 子块，通过 % 2 选择两个 score UB 槽
    uint64_t nextGroupId_ = 0; // 为 V1 的每个新 group 分配连续编号，通过 % GROUP_MAX_SLOTS 选择组参考槽
    TEventID vectorToCopyEvent_, copyToVectorEvent_;
    TEventID pVectorToCopyEvents_[P_SLOT_COUNT], pCopyToVectorEvents_[P_SLOT_COUNT];

    __aicore__ inline uint32_t GetAivRowCapacity(const Mxfp4VecTask& runInfo) const
    {
        // ND short M blocks use both AIVs. Each share is a whole Q16 scale block.
        return USE_DN ? VEC_ROWS : runInfo.actMSizeAlign32 / CV_RATIO;
    }

    __aicore__ inline uint32_t GetValidQueryRows(const Mxfp4VecTask& runInfo) const
    {
        const uint32_t firstRow = constInfo_.subBlockIdx * GetAivRowCapacity(runInfo);
        return runInfo.actMSize > firstRow ? AttentionCommon::Min(GetAivRowCapacity(runInfo), runInfo.actMSize - firstRow) : 0;
    }

    __aicore__ inline LocalTensor<float> BeginProbabilityGroup(Mxfp4VecTask& runInfo)
    {
        const uint32_t groupSlot = nextGroupId_ % GROUP_MAX_SLOTS;
        GroupOwner& owner = groupOwners_[groupSlot];
        LocalTensor<float> maximum = globalCurrentMax_.Get<float>()[groupSlot * VEC_ROWS];
        if (owner.state == GroupState::FREE) {
            owner.id = nextGroupId_;
            owner.state = GroupState::ACCUMULATING;
            owner.firstTaskLoop = runInfo.loop;
            owner.taskCount = 0;
            owner.firstGroup = runInfo.isFirstS2Loop;
            Duplicate<float>(maximum, Mxfp4Api::INVALID_GROUP_EXPONENT, VEC_ROWS);
        } else {
            ASSERT(owner.state == GroupState::ACCUMULATING && owner.id == nextGroupId_);
            ASSERT(!runInfo.isFirstS2Loop);
            ASSERT(runInfo.loop == owner.firstTaskLoop + owner.taskCount);
        }
        ASSERT(owner.taskCount < TASKS_PER_GROUP);
        runInfo.v1GroupId = nextGroupId_;
        return maximum;
    }

    __aicore__ inline void SealProbabilityGroup(const Mxfp4VecTask& runInfo)
    {
        GroupOwner& owner = GetGroupOwner(runInfo, GroupState::ACCUMULATING);
        ++owner.taskCount;
        if (runInfo.isGroupLast || runInfo.isLastS2Loop) {
            PipeBarrier<PIPE_V>();
            owner.state = GroupState::SEALED;
            ++nextGroupId_;
        } else {
            ASSERT(owner.taskCount < TASKS_PER_GROUP);
        }
    }

    __aicore__ inline GroupOwner& GetGroupOwner(const Mxfp4VecTask& runInfo, GroupState expectedState)
    {
        GroupOwner& owner = groupOwners_[runInfo.v1GroupId % GROUP_MAX_SLOTS];
        ASSERT(owner.id == runInfo.v1GroupId && owner.state == expectedState);
        return owner;
    }

    __aicore__ inline void WaitVectorBeforeCopy()
    {
        SetFlag<HardEvent::V_MTE3>(vectorToCopyEvent_);
        WaitFlag<HardEvent::V_MTE3>(vectorToCopyEvent_);
    }

    __aicore__ inline void WaitCopyBeforeVector()
    {
        SetFlag<HardEvent::MTE3_V>(copyToVectorEvent_);
        WaitFlag<HardEvent::MTE3_V>(copyToVectorEvent_);
    }

    __aicore__ inline void CopyProbabilityToL1(const LocalTensor<uint8_t>& destination,
                                              const LocalTensor<uint8_t>& source, const Mxfp4VecTask& runInfo)
    {
        const uint32_t firstRow = constInfo_.subBlockIdx * GetAivRowCapacity(runInfo);
        if constexpr (USE_DN) {
            DataCopy(destination[constInfo_.subBlockIdx * P_PACK_BYTES], source,
                {1, static_cast<uint16_t>(P_PACK_BYTES / 32), 0, 0});
        } else {
            DataCopy(destination[firstRow * 32], source,
                {static_cast<uint16_t>(s2BaseSize / 64), static_cast<uint16_t>(GetAivRowCapacity(runInfo)),
                 static_cast<uint16_t>(VEC_ROWS - GetAivRowCapacity(runInfo)),
                 static_cast<uint16_t>(mBaseSize - GetAivRowCapacity(runInfo))});
        }
    }

    __aicore__ inline void WriteAttentionResult(const Mxfp4VecTask& runInfo,
        const LocalTensor<float>& historyOutput, const LocalTensor<float>& rowSum,
        const LocalTensor<float>& reference)
    {
        const uint32_t validRows = GetValidQueryRows(runInfo);
        if (validRows == 0) {
            return;
        }
        LocalTensor<uint8_t> scratch;
        if constexpr (REUSE_PV_OUTPUT) {
            scratch = crossBuffers_.GetBmm2Slot(0).template ReinterpretCast<uint8_t>();
        } else {
            scratch = tempCommBuf_.Get<uint8_t>();
        }
        constexpr uint32_t statisticsOffset = OUTPUT_SCRATCH_BYTES;
        LocalTensor<float> lse = scratch[statisticsOffset].ReinterpretCast<float>();
        LocalTensor<float> broadcastSum = scratch[statisticsOffset + STATISTICS_BYTES].ReinterpretCast<float>();
        LocalTensor<float> broadcastReference = scratch[statisticsOffset + 2 * STATISTICS_BYTES].ReinterpretCast<float>();
        if (constInfo_.isSoftmaxLseEnable ||
            (FLASH_DECODE && constInfo_.enableFlashDecode && runInfo.isS2SplitCore)) {
            Mxfp4Api::PrepareAttentionStatistics<QUANT_SCHEMA, true>(
                lse, broadcastSum, broadcastReference, rowSum, reference);
        }
        if constexpr (FLASH_DECODE) {
            if (constInfo_.enableFlashDecode && runInfo.isS2SplitCore) {
                const uint64_t workspaceRow = static_cast<uint64_t>(runInfo.faTmpOutWsPos) * mBaseSize +
                                              constInfo_.subBlockIdx * GetAivRowCapacity(runInfo);
                if constexpr (USE_DN) {
                    LocalTensor<float> outputTile = scratch.ReinterpretCast<float>();
                    for (uint32_t channelOffset = 0; channelOffset < constInfo_.dSizeV; channelOffset += 16) {
                        Mxfp4Api::processOutDn<dVBaseSize, float, false>(outputTile, historyOutput, rowSum, channelOffset);
                        WaitVectorBeforeCopy();
                        const uint32_t validChannels = AttentionCommon::Min(16U, constInfo_.dSizeV - channelOffset);
                        DataCopyExtParams outputCopy {static_cast<uint16_t>(validRows),
                            static_cast<uint32_t>(validChannels * sizeof(float)), (16 - validChannels) / 8,
                            static_cast<uint32_t>((constInfo_.dSizeV - validChannels) * sizeof(float)), 0};
                        DataCopyPad(accumOutGm_[workspaceRow * constInfo_.dSizeV + channelOffset], outputTile, outputCopy);
                        WaitCopyBeforeVector();
                    }
                    WaitVectorBeforeCopy();
                } else {
                    // FD consumes raw FP32 PV with normalized a_i / S merge multipliers.
                    WaitVectorBeforeCopy();
                    DataCopyExtParams outputCopy {static_cast<uint16_t>(validRows),
                        static_cast<uint32_t>(constInfo_.dSizeV * sizeof(float)), (dVBaseSize - constInfo_.dSizeV) / 8, 0, 0};
                    DataCopyPad(accumOutGm_[workspaceRow * constInfo_.dSizeV], historyOutput, outputCopy);
                }
                DataCopy(softmaxFDSumGm_[workspaceRow * 8], broadcastSum, validRows * 8);
                DataCopy(softmaxFDMaxGm_[workspaceRow * 8], broadcastReference, validRows * 8);
                WaitCopyBeforeVector();
                return;
            }
        }
        LocalTensor<OUTPUT_T> convertedOutput = scratch.ReinterpretCast<OUTPUT_T>();
        if constexpr (USE_DN) {
            Mxfp4Api::processOutDn<dVBaseSize>(convertedOutput, historyOutput, rowSum);
        } else {
            Mxfp4Api::processOutNd<dVBaseSize>(convertedOutput, historyOutput, rowSum, validRows);
        }
        WaitVectorBeforeCopy();
        const uint64_t batchTokenBase = querySeqParser_->GetTBase(runInfo.bIdx);
        const uint64_t totalHeads = static_cast<uint64_t>(constInfo_.realN2Size) * constInfo_.realGSize;
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            const uint32_t groupSize = constInfo_.realGSize;
            const uint64_t firstMergedRow = static_cast<uint64_t>(runInfo.gS1Idx) +
                constInfo_.subBlockIdx * GetAivRowCapacity(runInfo);
            uint32_t copiedRows = 0;
            while (copiedRows < validRows) {
                const uint64_t mergedRow = firstMergedRow + copiedRows;
                const uint32_t groupOffset = mergedRow % groupSize;
                const uint32_t remainingRows = validRows - copiedRows;
                const uint32_t rowsPerCopy = AttentionCommon::Min(groupSize - groupOffset, remainingRows);
                const uint16_t copyCount = static_cast<uint16_t>(
                    groupOffset == 0 && remainingRows >= groupSize ? remainingRows / groupSize : 1);
                const uint64_t outputRow = (batchTokenBase + mergedRow / groupSize) * totalHeads +
                    static_cast<uint64_t>(runInfo.realN2Idx) * groupSize + groupOffset;
                DataCopyExtParams outputCopy {copyCount,
                    static_cast<uint32_t>(rowsPerCopy * dVBaseSize * sizeof(OUTPUT_T)), 0,
                    static_cast<uint32_t>((totalHeads - rowsPerCopy) * dVBaseSize * sizeof(OUTPUT_T)), 0};
                DataCopyPad(attentionOutGm_[outputRow * dVBaseSize],
                    convertedOutput[copiedRows * dVBaseSize], outputCopy);
                copiedRows += copyCount * rowsPerCopy;
            }
            if (constInfo_.isSoftmaxLseEnable) {
                const uint32_t firstGroupOffset = firstMergedRow % groupSize;
                for (uint32_t head = 0; head < groupSize; ++head) {
                    const uint32_t firstLocalRow = (head + groupSize - firstGroupOffset) % groupSize;
                    if (firstLocalRow >= validRows) {
                        continue;
                    }
                    const uint16_t rowCount = static_cast<uint16_t>(
                        (validRows - firstLocalRow - 1) / groupSize + 1);
                    const uint64_t queryHead = static_cast<uint64_t>(runInfo.realN2Idx) * groupSize + head;
                    const uint64_t queryToken = batchTokenBase + (firstMergedRow + firstLocalRow) / groupSize;
                    // Each UB statistic occupies 32 bytes; GM LSE is contiguous in T.
                    DataCopyExtParams lseCopy {rowCount, sizeof(float), groupSize - 1, 0, 0};
                    DataCopyPad(softmaxLseGm_[queryHead * constInfo_.t1Size + queryToken],
                        lse[firstLocalRow * 8], lseCopy);
                }
            }
        } else {
            for (uint32_t localRow = 0; localRow < validRows; ++localRow) {
                const uint64_t mergedRow = static_cast<uint64_t>(runInfo.gS1Idx) +
                    constInfo_.subBlockIdx * GetAivRowCapacity(runInfo) + localRow;
                const uint64_t queryToken = batchTokenBase + mergedRow / constInfo_.realGSize;
                const uint64_t queryHead = static_cast<uint64_t>(runInfo.realN2Idx) * constInfo_.realGSize +
                    mergedRow % constInfo_.realGSize;
                const uint64_t outputRow = outLayout == LayOutTypeEnum::LAYOUT_TND ?
                    queryToken * totalHeads + queryHead : queryHead * constInfo_.t1Size + queryToken;
                DataCopyExtParams outputCopy {1, static_cast<uint32_t>(constInfo_.dSizeV * sizeof(OUTPUT_T)), 0, 0, 0};
                DataCopyPad(attentionOutGm_[outputRow * constInfo_.dSizeV], convertedOutput[localRow * dVBaseSize], outputCopy);
                if (constInfo_.isSoftmaxLseEnable) {
                    DataCopyExtParams lseCopy {1, sizeof(float), 0, 0, 0};
                    DataCopyPad(softmaxLseGm_[queryHead * constInfo_.t1Size + queryToken], lse[localRow * 8], lseCopy);
                }
            }
        }
        WaitCopyBeforeVector();
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool isFd = false, bool useDn = false, bool isDAligned = true,
          QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT, bool enableQf322F32Pre = false>
class QuantFlashAttnBlockVecMxfp4Dummy {
public:
    static constexpr QuantSchema QUANT_SCHEMA_VALUE = QUANT_SCHEMA;
    static constexpr bool ENABLE_QF322F32_PRE = enableQf322F32Pre;
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    using OUT_T = OUTPUT_T;
    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockVecMxfp4Dummy(ConstInfoX& constInfo) {}
};
} // namespace BaseApi
#endif
