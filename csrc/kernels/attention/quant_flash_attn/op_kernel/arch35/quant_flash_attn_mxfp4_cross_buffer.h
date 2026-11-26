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
 * \file quant_flash_attn_mxfp4_cross_buffer.h
 * \brief Cross-core buffers and synchronization for the MXFP4 Cube and Vector blocks.
 */
#ifndef QUANT_FLASH_ATTN_MXFP4_CROSS_BUFFER_H_
#define QUANT_FLASH_ATTN_MXFP4_CROSS_BUFFER_H_

#include <cstdint>
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/buffers_policy.h"
#include "quant_flash_attn_common_def.h"
#include "../../../common/op_kernel/vector_common.h"

namespace BaseApi {
// common class cv buffers
template <uint32_t PRELOAD_N = 20, bool useDn = false,
        S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
        S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
        DTemplateType dTemplateType = DTemplateType::Aligned128,
        DTemplateType dVTemplateType = DTemplateType::Aligned128>
class CVCrossBuffer {
public:

    // 核间同步事件
    static constexpr uint8_t SYNC_MODE_4 = 4;
    static constexpr uint16_t CROSS_CORE_SYNC_V1_C1[2] = {0, 1};
    static constexpr uint16_t CROSS_CORE_SYNC_C1_V1[2] = {2, 3};
    // Up to twenty single-task groups can be in flight across Q blocks. A
    // single counting flag can accumulate seventeen ready groups before C2;
    // alternate two channels so each counter stays below the hardware limit 15.
    static constexpr uint16_t CROSS_CORE_SYNC_PSCALE_C2[2] = {4, 7};
    static constexpr uint16_t CROSS_CORE_SYNC_V2_C2 = 5;
    static constexpr uint16_t CROSS_CORE_SYNC_C2_V2 = 6;

    static constexpr bool USE_DN = useDn;
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr uint32_t s2SplitSize = 128U;

    static constexpr uint32_t SCORE_SLOT_COUNT = 2;

    static __aicore__ inline bool CanScaleInFixpipe(float scale)
    {
        const uint32_t bits = *reinterpret_cast<uint32_t*>(&scale);
        // QF322F32_PRE discards scale mantissa bits 12:0. Only use it when
        // those bits are already zero; arbitrary API scales remain in FP32 VF.
        return scale > 0.0F && bits < 0x7f800000U && (bits & 0x1fffU) == 0;
    }


    // L1 buffer
    Buffer<BufferType::L1, SyncType::NO_SYNC> l1PBuffers_; // 20 个 task的p
    Buffer<BufferType::L1, SyncType::NO_SYNC> l1PScaleBuffers_; // 20 个 task的p scale
    static constexpr uint32_t l1PBufferBytes_ = mBaseSize * s2BaseSize / 2;
    // 比 mBaseSize * s2BaseSize /32 多了 mBaseSize * 2，按 16 * 64 的小块，每个小块有 16 * 2 个scale，对应32字节。
    // 把16个query对应的所有一行s2BaseSize有s2BaseSize/64个块，作为一个组，在这个组后面补一个 16 * 2 个scale作为padding，那么16个query行就有 (s2BaseSize/64 + 1) 个小块。
    // 同时，一共有  (mBaseSize / 16) 个小组
    static constexpr uint32_t l1PScaleBufferBytes_ = (mBaseSize / 16) * (s2BaseSize/64 + 1) * 32;


    //UB buffer
    Buffer<BufferType::UB, SyncType::NO_SYNC> ubBmm1Buffers_; // qk
    Buffer<BufferType::UB, SyncType::NO_SYNC> ubBmm2Buffers_; // pv
    static constexpr uint32_t ubBmm1BufferBytes_ = mBaseSize / CV_RATIO * s2SplitSize * sizeof(float);
    // 对于pv结果，因为我们要使用cube计算rowsum，所以需要进行对齐
    // 对于 DN，我们计算的是(PV)^T = [D, M]，所以补齐一行为 [D+1, M]，而ND的结果是[M, D]，本来是[M, D+1]，但是为了让每行都32B对齐，所以增加7个float padding，变成 [M, D+8]
    static constexpr uint32_t ubBmm2BufferBytes_ = USE_DN? mBaseSize / CV_RATIO * (dVBaseSize + 1) * sizeof(float) : mBaseSize / CV_RATIO * (dVBaseSize + 8) * sizeof(float);


    /*============================function============================================*/
    __aicore__ inline CVCrossBuffer(){}

    __aicore__ inline void InitBuffers(BufferManager<BufferType::L1> &l1BufferManagerPtr_,
        BufferManager<BufferType::UB> &ubBufferManagerPtr_)
    {
        l1PBuffers_ = l1BufferManagerPtr_.AllocBuffer<SyncType::NO_SYNC>(l1PBufferBytes_ * PRELOAD_N);
        l1PScaleBuffers_ = l1BufferManagerPtr_.AllocBuffer<SyncType::NO_SYNC>(l1PScaleBufferBytes_ * PRELOAD_N);
        ubBmm1Buffers_ = ubBufferManagerPtr_.AllocBuffer<SyncType::NO_SYNC>(ubBmm1BufferBytes_ * SCORE_SLOT_COUNT);
        ubBmm2Buffers_ = ubBufferManagerPtr_.AllocBuffer<SyncType::NO_SYNC>(ubBmm2BufferBytes_);
    }

    __aicore__ inline void InitSync()
    {
        if ASCEND_IS_AIV {
            // 两个 QK 接收槽初始为空，允许 Cube 写入。
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(
                CROSS_CORE_SYNC_V1_C1[0]);
            if constexpr (SCORE_SLOT_COUNT == 2) {
                CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[1]);
            }

            // PV 接收槽初始为空，允许 Cube 写入。
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(
                CROSS_CORE_SYNC_V2_C2);
        }
    }

    __aicore__ inline LocalTensor<uint8_t> GetPSlot(uint32_t slot) {
        LocalTensor<uint8_t> probabilityL1Tensor = l1PBuffers_.GetTensor<uint8_t>();
        LocalTensor<uint8_t> probabilitySlotTensor = probabilityL1Tensor[slot * l1PBufferBytes_];
        return probabilitySlotTensor;
    }

    __aicore__ inline LocalTensor<uint8_t> GetPScaleSlot(uint32_t slot) {
        LocalTensor<uint8_t> probabilityScaleL1Tensor = l1PScaleBuffers_.GetTensor<uint8_t>();
        LocalTensor<uint8_t> probabilityScaleSlotTensor = probabilityScaleL1Tensor[slot * l1PScaleBufferBytes_];
        return probabilityScaleSlotTensor;
    }

    __aicore__ inline LocalTensor<float> GetBmm1Slot(uint32_t slot) {
        LocalTensor<float> scoreUbTensor = ubBmm1Buffers_.GetTensor<float>();
        LocalTensor<float> scoreSlotTensor = scoreUbTensor[slot * (ubBmm1BufferBytes_ / sizeof(float))];
        return scoreSlotTensor;
    }

    __aicore__ inline LocalTensor<float> GetBmm2Slot(uint32_t slot) {
        LocalTensor<float> pvResultUbTensor = ubBmm2Buffers_.GetTensor<float>();
        LocalTensor<float> pvResultSlotTensor = pvResultUbTensor[slot * (ubBmm2BufferBytes_ / sizeof(float))];
        return pvResultSlotTensor;
    }

    __aicore__ inline void setBmm1Buffer(uint32_t slot) {
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C1_V1[slot]);
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C1_V1[slot] + 16);
        } else {
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[slot]);
        }
    }

    __aicore__ inline void waitBmm1Buffer(uint32_t slot) {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[slot]);
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[slot] + 16);
        } else {
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C1_V1[slot]);
        }
    }

    __aicore__ inline void setBmm2Buffer() {
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2);
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2 + 16);
        } else {
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
        }
    }

    __aicore__ inline void watBmm2Buffer() {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2 + 16);
        } else {
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C2_V2);
        }
    }

    __aicore__ inline void setPscaleUpdate(uint64_t groupId) {
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_PSCALE_C2[groupId % 2]);
        }
    }

    __aicore__ inline void waitPscaleUpdate(uint64_t groupId) {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2[groupId % 2]);
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2[groupId % 2] + 16);
        }
    }
};

} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_MXFP4_CROSS_BUFFER_H_
