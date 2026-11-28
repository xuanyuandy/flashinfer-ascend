/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 220
#error "IndexerPrologueQW requires arch22 (Ascend910B / Ascend910C)"
#else

/* !
 * \file quant_group_matmul_custom.asc
 * \brief
 */

#include "lib/matmul_intf.h"
#include "indexer_prologue_qw_tiling_device.h"
#include <cstdlib>
#include <cmath>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "qw_post.h"
using QuantGroupMatmulCustomTilingData = AscendC::tiling::IndexerPrologueQWFullFusionTilingData;

using AscendC::LocalTensor;
using AscendC::GlobalTensor;
using AscendC::MatmulType;

using aT = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t>;
using bT_Dense = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t, true>;
using bT_Prequant = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, int8_t, false>;
using BiasT = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;
using cT = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;

using MT_Dense = matmul::MatmulImpl<aT, bT_Dense, cT, BiasT, CFG_MDL>;
using MT_Prequant = matmul::MatmulImpl<aT, bT_Prequant, cT, BiasT, CFG_MDL>;
using wA = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bfloat16_t>;
using wB = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using wC = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bfloat16_t>;
using wBias = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
using WMT = matmul::MatmulImpl<wA, wB, wC, wBias, CFG_MDL>;

constexpr uint32_t BROADCAST_DIM = 2;
constexpr uint64_t SYNC_AIV_TO_AIC = 3;
constexpr uint64_t SYNC_AIC_TO_AIV = 5;
constexpr uint64_t SYNC_ACTIVATION_READY = 9;
constexpr uint64_t SYNC_W_READY = 10;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t DQ_ROWS_PER_TILE = 4;

struct MNConfig {
    uint32_t m = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t mIdx = 0;
    uint32_t nIdx = 0;
    uint32_t blockDimM = 0;
    uint32_t blockDimN = 0;
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t offsetM = 0;
    uint64_t workSpaceOffset = 0;
};

template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t srcDim0)
{
    AscendC::DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T);
    // 32: int32 -> float16, 为防止跨行数据进入同一32B block，提前每行按偶数block对齐
    params.dstStride = AscendC::Ceil(dim0 * sizeof(T), 32) % 2;

    AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> dst, const LocalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t srcDim0, uint32_t dstDim0)
{
    AscendC::DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    // 32: ub访问粒度为32B
    params.srcStride = (srcDim0 - dim0) * sizeof(T) / 32;
    params.dstStride = (dstDim0 - dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

template <typename MT>
class QuantGroupMatmul {
public:
    __aicore__ inline QuantGroupMatmul(MT& matmul)
        : mm(matmul)
    {}
    __aicore__ inline void Init(GM_ADDR qr, GM_ADDR x, GM_ADDR weight, GM_ADDR wqb, GM_ADDR scale,
                                GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR xW, GM_ADDR weightW, GM_ADDR wOut,
                                GM_ADDR ropeSin, GM_ADDR ropeCos, GM_ADDR q8, GM_ADDR qs, GM_ADDR w16,
                                GM_ADDR workspace, QuantGroupMatmulCustomTilingData* tilingData,
                                AscendC::TPipe* tPipeIn, bool denseIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void DynamicQuantRows(GlobalTensor<bfloat16_t>& src, GlobalTensor<int8_t>& dstQ,
                                            GlobalTensor<float>& dstScale, uint32_t rows);
    __aicore__ inline void ActivationDQ();
    __aicore__ inline void WeightDQ();
    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig);
    __aicore__ inline void WCompute();
    __aicore__ inline void VectorCompute(uint32_t groupIdx, MNConfig& mnConfig);
    __aicore__ inline void PostCompute();
    __aicore__ inline void ComputeDequantAndActivate(MNConfig& mnConfig, uint32_t curVecBaseM, uint32_t alignBaseN,
                                                     uint32_t curVecBaseN, uint32_t offsetM);
    __aicore__ inline void DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);
    __aicore__ inline void DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig, uint32_t curBaseM, uint32_t alignBaseN,
                                                        uint32_t offsetM);

private:
    MT& mm;
    GlobalTensor<int8_t> xGm;
    GlobalTensor<bfloat16_t> qrGm;
    GlobalTensor<int8_t> weightGm;
    GlobalTensor<bfloat16_t> wqbGm;
    GlobalTensor<int32_t> biasGm;
    GlobalTensor<int32_t> mmOutGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> perTokenScaleGm;
    GlobalTensor<int64_t> groupListGm;
    GlobalTensor<bfloat16_t> yGm;
    GlobalTensor<bfloat16_t> xWGm;
    GlobalTensor<bfloat16_t> weightWGm;
    GlobalTensor<bfloat16_t> wOutGm;
    GM_ADDR ropeSinAddr = nullptr;
    GM_ADDR ropeCosAddr = nullptr;
    GM_ADDR q8Addr = nullptr;
    GM_ADDR qsAddr = nullptr;
    GM_ADDR w16Addr = nullptr;
    GM_ADDR yAddr = nullptr;
    GM_ADDR wOutAddr = nullptr;
    // define the que
    AscendC::TQue<AscendC::TPosition::VECIN, 1> vecInQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> vecOutQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> scaleInQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> perTokenScaleInQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> postQsOutQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> postWOutQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> postWbInQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuff;
    LocalTensor<float> scaleInUb;
    LocalTensor<float> perTokenScaleInUb;
    LocalTensor<float> dequantMiddleResult;
    LocalTensor<uint8_t> sharedTmpLocal;
    LocalTensor<float> mulsResultLocal;
    LocalTensor<float> pertokenBrcbLocal;
    LocalTensor<float> actResultLocal;
    uint32_t subBlockIdx;
    uint32_t coreIdx;
    uint32_t cubeCount = 0;
    AscendC::TPipe* pipe;
    QuantGroupMatmulCustomTilingData* tiling;
    bool denseLayout = false;
};

template <typename MT>
__aicore__ inline void QuantGroupMatmul<MT>::Init(GM_ADDR qr, GM_ADDR x, GM_ADDR weight, GM_ADDR wqb, GM_ADDR scale,
                                                  GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR xW, GM_ADDR weightW,
                                                  GM_ADDR wOut, GM_ADDR ropeSin, GM_ADDR ropeCos, GM_ADDR q8,
                                                  GM_ADDR qs, GM_ADDR w16, GM_ADDR workspace,
                                                  QuantGroupMatmulCustomTilingData* tilingData, AscendC::TPipe* tPipeIn,
                                                  bool denseIn)
{
    qrGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t*>(qr),
        static_cast<uint64_t>(tilingData->get_totalInGroup()) *
            tilingData->get_k()); // scalar-review: layout-view; BF16 activation rows
    xGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int8_t*>(x)); // scalar-review: layout-view; typed GM view, vector/DMA access only
    weightGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int8_t*>(weight)); // scalar-review: layout-view; INT8 ND trans-B weight
    wqbGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wqb),
                          4096ULL * 1280ULL); // scalar-review: layout-view; dense BF16 weight ND
    mmOutGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(GetUserWorkspace(workspace))); // scalar-review: layout-view; INT32 user CV workspace
    scaleGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(scale)); // scalar-review: layout-view; per-channel scale DMA source
    perTokenScaleGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(perTokenScale)); // scalar-review: layout-view; per-token scale DMA source
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y)); // scalar-review: layout-view; BF16 ND output view
    xWGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(xW),
                         static_cast<uint64_t>(tilingData->get_totalInGroup()) * 5120); // scalar-review: layout-view; W activation ND
    weightWGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(weightW),
                              5120 * 32); // scalar-review: layout-view; W weight ND
    wOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wOut),
                           static_cast<uint64_t>(tilingData->get_totalInGroup()) * 32); // scalar-review: layout-view; W BF16 ND output
    ropeSinAddr = ropeSin;
    ropeCosAddr = ropeCos;
    q8Addr = q8;
    qsAddr = qs;
    w16Addr = w16;
    yAddr = y;
    wOutAddr = wOut;
    tiling = tilingData;
    denseLayout = denseIn;
    subBlockIdx = AscendC::GetSubBlockIdx();
    coreIdx = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        coreIdx /= AscendC::GetTaskRation();
    }
    pipe = tPipeIn;
    InitUbBuffer();
}

template <typename MT>
__aicore__ inline void QuantGroupMatmul<MT>::InitUbBuffer()
{
    if ASCEND_IS_AIC {
        return;
    }
    pipe->InitBuffer(scaleInQueue, BUFFER_NUM, tiling->mmTilingData.baseN * sizeof(float));
    pipe->InitBuffer(perTokenScaleInQueue, BUFFER_NUM, tiling->mmTilingData.baseM * sizeof(float));
    pipe->InitBuffer(postQsOutQueue, 1, 64);
    pipe->InitBuffer(postWOutQueue, 2, 64);
    pipe->InitBuffer(postWbInQueue, 2, 64);
    pipe->InitBuffer(vecInQueue, BUFFER_NUM, tiling->get_ubCalSize() * sizeof(cT::T));
    pipe->InitBuffer(vecOutQueue, BUFFER_NUM, tiling->get_ubCalSize() * sizeof(bfloat16_t));
    pipe->InitBuffer(tmpBuff, tiling->get_ubRestBytes());
    uint32_t ubCalSizeFloat = tiling->get_ubCalSize() * sizeof(float);
    // ub分配，依次划分中间结果
    dequantMiddleResult = tmpBuff.GetWithOffset<float>(tiling->get_ubCalSize(), 0);
    pertokenBrcbLocal = tmpBuff.GetWithOffset<float>(tiling->get_ubCalSize(), ubCalSizeFloat);
    // 2: 偏移两份，前面为反量化输出和pertoken scale
    mulsResultLocal = tmpBuff.GetWithOffset<float>(tiling->get_ubCalSize(), 2 * ubCalSizeFloat);
    // 3: 再上面的基础上再偏移一份
    actResultLocal = tmpBuff.GetWithOffset<float>(tiling->get_ubCalSize(), 3 * ubCalSizeFloat);
    // api需要的临时空间，复用中间结果的空间
    // 2: ub临时空间总共4份，高层api分配两份
    sharedTmpLocal = tmpBuff.GetWithOffset<uint8_t>(2 * ubCalSizeFloat, 2 * ubCalSizeFloat);
}

template <typename MT>
__aicore__ inline void QuantGroupMatmul<MT>::DynamicQuantRows(GlobalTensor<bfloat16_t>& src, GlobalTensor<int8_t>& dstQ,
                                                              GlobalTensor<float>& dstScale, uint32_t rows)
{
    const uint32_t k = tiling->get_k();
    const uint32_t taskRation = AscendC::GetTaskRation();
    const uint32_t totalVec = tiling->get_coreNum() * taskRation;
    const uint32_t firstRow = coreIdx * taskRation + subBlockIdx;
    LocalTensor<float> xF = dequantMiddleResult;
    LocalTensor<float> qF = mulsResultLocal;
    LocalTensor<float> scalar = pertokenBrcbLocal;
    LocalTensor<float> mx = scalar;
    LocalTensor<float> c127 = scalar[16];
    LocalTensor<float> scaleOut = scalar[32];
    LocalTensor<uint8_t> workRM = actResultLocal.ReinterpretCast<uint8_t>();
    constexpr uint32_t SCALE_SLOT_FLOATS = 8; // 32B: each DataCopyPad source block starts aligned
    Duplicate(c127, 127.0f, 8);
    const uint32_t rowCount = firstRow < rows ? (rows - firstRow + totalVec - 1) / totalVec : 0;
    // scalar-review: batch-control; four whole rows per tile, preserving the original row-to-AIV mapping
    for (uint32_t tileRow = 0; tileRow < rowCount; tileRow += DQ_ROWS_PER_TILE) {
        const uint32_t curRows = (rowCount - tileRow < DQ_ROWS_PER_TILE) ? rowCount - tileRow : DQ_ROWS_PER_TILE;
        const uint32_t row = firstRow + tileRow * totalVec;
        const uint32_t elems = curRows * k;
        LocalTensor<bfloat16_t> input = vecInQueue.AllocTensor<bfloat16_t>();
        const uint32_t srcRowGap = static_cast<uint32_t>((totalVec - 1) * k * sizeof(bfloat16_t));
        AscendC::DataCopyExtParams copyRow{static_cast<uint16_t>(curRows),
                                           static_cast<uint32_t>(k * sizeof(bfloat16_t)), srcRowGap, 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> padBf{false, 0, 0, 0};
        DataCopyPad(input, src[static_cast<uint64_t>(row) * k], copyRow, padBf);
        vecInQueue.EnQue(input);
        LocalTensor<bfloat16_t> inLocal = vecInQueue.DeQue<bfloat16_t>();
        Cast(xF, inLocal, AscendC::RoundMode::CAST_NONE, elems);
        vecInQueue.FreeTensor(inLocal);
        AscendC::PipeBarrier<PIPE_V>();

        Abs(qF, xF, elems);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t rowShape[2] = {curRows, k};
        ReduceMax<float, AscendC::Pattern::Reduce::AR, false>(mx, qF, workRM, rowShape, true);
        AscendC::PipeBarrier<PIPE_V>();
        // Match standalone DynamicQuant: scale=127/amax, quant uses Muls,
        // and the published scale is the scalar reciprocal 1/scale.
        Div(mx, c127, mx, curRows);
        const event_t scaleVs = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(scaleVs);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(scaleVs);
        // scalar-review: batch-control; retain the original per-row scalar reciprocal and Muls rounding order
        for (uint32_t lane = 0; lane < curRows; ++lane) {
            const float scale = mx.GetValue(lane);
            const float publishedScale = 1.0f / scale;
            Duplicate(scaleOut[lane * SCALE_SLOT_FLOATS], publishedScale, 1);
            Muls(qF[lane * k], xF[lane * k], scale, k);
        }
        AscendC::PipeBarrier<PIPE_V>();

        LocalTensor<int32_t> qI32 = qF.ReinterpretCast<int32_t>();
        Cast(qI32, qF, AscendC::RoundMode::CAST_RINT, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetDeqScale((half)1.000000e+00f);
        LocalTensor<half> qHalf = qI32.ReinterpretCast<half>();
        Cast(qHalf, qI32, AscendC::RoundMode::CAST_ROUND, elems);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<int8_t> qOut = vecOutQueue.AllocTensor<int8_t>();
        Cast(qOut, qHalf, AscendC::RoundMode::CAST_TRUNC, elems);
        AscendC::PipeBarrier<PIPE_V>();
        vecOutQueue.EnQue(qOut);

        LocalTensor<int8_t> qResult = vecOutQueue.DeQue<int8_t>();
        const uint32_t dstQRowGap = static_cast<uint32_t>((totalVec - 1) * k * sizeof(int8_t));
        const uint32_t dstScaleRowGap = static_cast<uint32_t>((totalVec - 1) * sizeof(float));
        AscendC::DataCopyExtParams copyQ{static_cast<uint16_t>(curRows), k, 0, dstQRowGap, 0};
        AscendC::DataCopyExtParams copyScale{static_cast<uint16_t>(curRows),
                                             static_cast<uint32_t>(sizeof(float)), 0, dstScaleRowGap, 0};
        DataCopyPad(dstQ[static_cast<uint64_t>(row) * k], qResult, copyQ);
        const int32_t scaleEvent = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(scaleEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(scaleEvent);
        DataCopyPad(dstScale[row], scaleOut, copyScale);
        // qOut/scaleOut may be reused by the next tile. V_MTE3 only orders
        // production before the copy; MTE3_V is the consumed-completion edge
        // required before the Vector pipe overwrites or frees the UB slots.
        const int32_t copyBackEvent = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(copyBackEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(copyBackEvent);
        vecOutQueue.FreeTensor(qResult);
    }
}

template <typename MT>
__aicore__ inline void QuantGroupMatmul<MT>::ActivationDQ()
{
    DynamicQuantRows(qrGm, xGm, perTokenScaleGm, tiling->get_totalInGroup());
}

template <typename MT>
__aicore__ inline void QuantGroupMatmul<MT>::WeightDQ()
{
    DynamicQuantRows(wqbGm, weightGm, scaleGm, 4096);
}

template <typename MT>
__aicore__ inline void QuantGroupMatmul<MT>::Process()
{
    if ASCEND_IS_AIV {
        ActivationDQ();
        if (denseLayout) {
            WeightDQ();
        }
        // Every AIV reaches the same barrier, including those with no rows.
        AscendC::SyncAll<true>();
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_ACTIVATION_READY);
    } else {
        // Q tiles may read any token, so all AIV producers must be complete.
        AscendC::CrossCoreWaitFlag(SYNC_ACTIVATION_READY);
    }
    MNConfig mnConfig;
    mnConfig.baseM = tiling->mmTilingData.baseM;
    mnConfig.baseN = tiling->mmTilingData.baseN;
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = mnConfig.baseN;
    mnConfig.blockDimN = AscendC::Ceil(tiling->get_n(), mnConfig.singleN);
    // scalar-review: batch-control; single-group tile scheduler, arithmetic is official Matmul/vector API
    for (uint32_t groupIdx = 0, preCount = 0; groupIdx < tiling->get_groupNum(); ++groupIdx) {
        uint32_t m = tiling->get_totalInGroup();
        if (m <= 0) {
            continue;
        }
        mnConfig.m = static_cast<uint32_t>(m);
        mnConfig.blockDimM = AscendC::Ceil(mnConfig.m, mnConfig.singleM);
        uint32_t curCount = preCount + mnConfig.blockDimN * mnConfig.blockDimM;
        uint32_t curBlock = coreIdx >= preCount ? coreIdx : coreIdx + tiling->get_coreNum();

        // scalar-review: batch-control; core-stride tile assignment, no scalar element computation
        while (curBlock < curCount) {
            mnConfig.mIdx = (curBlock - preCount) / mnConfig.blockDimN;
            mnConfig.nIdx = (curBlock - preCount) % mnConfig.blockDimN;
            MMCompute(groupIdx, mnConfig);
            VectorCompute(groupIdx, mnConfig);
            curBlock += tiling->get_coreNum();
        }
        preCount = curCount % tiling->get_coreNum();
        mnConfig.offsetM += mnConfig.m;
    }
    WCompute();
    PostCompute();
    // One mixed-core barrier publishes both projection outputs to every AIV.
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
        QWPost post;
        const uint32_t taskRation = AscendC::GetTaskRation();
        post.Init(yAddr, wOutAddr, ropeSinAddr, ropeCosAddr, q8Addr, qsAddr, w16Addr, &vecInQueue, &vecOutQueue,
                  &scaleInQueue, &postWbInQueue, &postQsOutQueue, &postWOutQueue, &tmpBuff, tiling->get_totalInGroup(),
                  coreIdx * taskRation + subBlockIdx, tiling->get_coreNum() * taskRation, coreIdx,
                  tiling->get_wScale());
        post.Process();
    }
}

template <typename MT>
__aicore__ inline void QuantGroupMatmul<MT>::WCompute()
{
    if ASCEND_IS_AIV {
        return;
    }
    // Close Q before constructing any per-core W object. Empty W-tile cores
    // still return through PostCompute and the final mixed-core barrier.
    mm.End();
    const uint32_t singleCoreM = tiling->wTilingData.singleCoreM;
    const uint32_t singleCoreN = tiling->wTilingData.singleCoreN;
    if (coreIdx >= static_cast<uint32_t>(tiling->wTilingData.usedCoreNum) || singleCoreM == 0 || singleCoreN == 0) {
        return;
    }
    const uint32_t mSingleBlocks = AscendC::Ceil(static_cast<uint32_t>(tiling->wTilingData.M), singleCoreM);
    const uint32_t mCoreIdx = coreIdx % mSingleBlocks;
    const uint32_t nCoreIdx = coreIdx / mSingleBlocks;
    const uint32_t tailM = static_cast<uint32_t>(tiling->wTilingData.M) > mCoreIdx * singleCoreM ?
                               (static_cast<uint32_t>(tiling->wTilingData.M) - mCoreIdx * singleCoreM < singleCoreM ?
                                    static_cast<uint32_t>(tiling->wTilingData.M) - mCoreIdx * singleCoreM :
                                    singleCoreM) :
                               0;
    const uint32_t tailN = static_cast<uint32_t>(tiling->wTilingData.N) > nCoreIdx * singleCoreN ?
                               (static_cast<uint32_t>(tiling->wTilingData.N) - nCoreIdx * singleCoreN < singleCoreN ?
                                    static_cast<uint32_t>(tiling->wTilingData.N) - nCoreIdx * singleCoreN :
                                    singleCoreN) :
                               0;
    if (tailM == 0 || tailN == 0) {
        return;
    }
    const uint64_t offsetA = static_cast<uint64_t>(mCoreIdx) * tiling->wTilingData.Ka * singleCoreM;
    const uint64_t offsetB = static_cast<uint64_t>(nCoreIdx) * tiling->wTilingData.Kb * singleCoreN;
    const uint64_t offsetC = static_cast<uint64_t>(mCoreIdx) * tiling->wTilingData.N * singleCoreM +
                             static_cast<uint64_t>(nCoreIdx) * singleCoreN;
    WMT wmm;
    wmm.SetSubBlockIdx(0);
    wmm.Init(&tiling->wTilingData, pipe);
    wmm.SetOrgShape(tiling->get_totalInGroup(), 32, 5120, 5120);
    wmm.SetTensorA(xWGm[offsetA], false);
    wmm.SetTensorB(weightWGm[offsetB], true);
    wmm.SetTail(tailM, tailN);
    // Each output tile keeps the official full-K IterateAll reduction order;
    // no K split, atomics, or cross-core accumulation is introduced.
    wmm.IterateAll(wOutGm[offsetC]);
    wmm.End();
}

template <typename MT>
__aicore__ inline void QuantGroupMatmul<MT>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.blockDimN - 1) {
        curSingleN = tiling->get_n() - tailN;
    }
    uint32_t curSingleM = mnConfig.singleM;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t xOffset = (mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM) * tiling->get_k();
    uint64_t weightOffset =
        groupIdx * tiling->get_n() * tiling->get_k() + tailN * tiling->get_k(); // for no transpose nz weight
    mnConfig.workSpaceOffset =
        mnConfig.singleN * mnConfig.singleM * (coreIdx + (cubeCount % tiling->get_parallNum()) * tiling->get_coreNum());
    if ASCEND_IS_AIC {
        if (cubeCount >= tiling->get_parallNum()) {
            AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
        }
        mm.SetOrgShape(mnConfig.m, tiling->get_n(), tiling->get_k());
        mm.SetSingleShape(curSingleM, curSingleN, tiling->get_k());
        mm.SetTensorA(xGm[xOffset]);
        auto weightSlice = weightGm[weightOffset];
        if (mnConfig.blockDimM == 1) {
            weightSlice.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        mm.SetTensorB(weightSlice, denseLayout);
        uint64_t worskspaceOffset = mnConfig.workSpaceOffset;
        // scalar-review: batch-control; official Matmul base-block iterator
        while (mm.Iterate()) {
            mm.GetTensorC(mmOutGm[worskspaceOffset], 0, true);
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV); // 2: mode为2, group内同步
            worskspaceOffset += (mnConfig.baseM * mnConfig.baseN);
        }
    }
    cubeCount++;
}

template <typename MT>
__aicore__ inline void QuantGroupMatmul<MT>::VectorCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    if ASCEND_IS_AIC {
        return;
    }
    uint32_t curCubeSingleN = mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.blockDimN - 1) {
        curCubeSingleN = tiling->get_n() - mnConfig.nIdx * mnConfig.singleN;
    }
    uint32_t curCubeSingleM = mnConfig.singleM;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curCubeSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    }
    uint32_t vecBaseM =
        tiling->get_ubCalSize() / (AscendC::Ceil(mnConfig.baseN, uint32_t(8)) * 8); //  8: num int32_t in 32B ub block
    vecBaseM = vecBaseM < curCubeSingleM ? vecBaseM : curCubeSingleM;
    uint32_t curVecBaseN = mnConfig.baseN;
    uint64_t scaleOffset = groupIdx * tiling->get_n() + mnConfig.nIdx * mnConfig.singleN;
    uint64_t outOffset =
        (mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM) * tiling->get_n() + mnConfig.nIdx * mnConfig.singleN;
    uint32_t taskRation = AscendC::GetTaskRation();
    // scalar-review: batch-control; vector N-tile loop with block DataCopy/AscendDequant
    for (uint32_t offsetN = 0, vecCount = 0; offsetN < curCubeSingleN; offsetN += mnConfig.baseN) {
        if (unlikely(offsetN + mnConfig.baseN >= curCubeSingleN)) {
            curVecBaseN = curCubeSingleN - offsetN;
        }
        uint32_t alignBaseN = AscendC::Ceil(curVecBaseN, uint32_t(8)) * 8; //  8: num int32_t in 32B ub block
        DataCopyScale(curVecBaseN, alignBaseN, scaleOffset + offsetN);
        uint32_t curVecBaseM = vecBaseM;
        uint64_t mmOutOffset = mnConfig.workSpaceOffset + offsetN * mnConfig.baseM;
        AscendC::CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
        // scalar-review: batch-control; vector M-tile loop, split across two AIV subcores
        for (uint32_t offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM) {
            vecCount++;
            if (taskRation != 0 && vecCount % taskRation != subBlockIdx) {
                continue;
            }
            if (unlikely(offsetM + vecBaseM >= curCubeSingleM)) {
                curVecBaseM = curCubeSingleM - offsetM;
            }
            // 使用AscendDequant接口做perchannel反量化
            LocalTensor<cT::T> mmOutLocal = vecInQueue.AllocTensor<cT::T>();
            DataCopyPad2D(mmOutLocal, mmOutGm[mmOutOffset + offsetM * curVecBaseN], curVecBaseM, curVecBaseN,
                          curVecBaseN);
            vecInQueue.EnQue(mmOutLocal);
            ComputeDequantAndActivate(mnConfig, curVecBaseM, alignBaseN, curVecBaseN, offsetM);
            LocalTensor<bfloat16_t> yLocal = vecOutQueue.DeQue<bfloat16_t>();
            DataCopyPad2D(yGm[outOffset + offsetM * tiling->get_n() + offsetN], yLocal, curVecBaseM, curVecBaseN,
                          alignBaseN, tiling->get_n());
            vecOutQueue.FreeTensor(yLocal);
        }
        scaleInQueue.FreeTensor(scaleInUb);
    }
    AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC); // 2: mode为2, group内同步
}

template <typename MT>
__aicore__ inline void QuantGroupMatmul<MT>::PostCompute()
{
    if ASCEND_IS_AIC {
        if (likely(cubeCount > 0)) { // exits last cycle wait
            uint32_t loop = cubeCount < tiling->get_parallNum() ? cubeCount : tiling->get_parallNum();
            // scalar-review: batch-control; drain exactly the outstanding AIV completion signals
            for (int32_t idx = 0; idx < loop; ++idx) {
                AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
            }
        }
    }
}

template <typename MT>
__aicore__ inline void QuantGroupMatmul<MT>::ComputeDequantAndActivate(MNConfig& mnConfig, uint32_t curVecBaseM,
                                                                       uint32_t alignBaseN, uint32_t curVecBaseN,
                                                                       uint32_t offsetM)
{
    DataCopyPerTokenScaleAndBrcb(mnConfig, curVecBaseM, alignBaseN, offsetM);
    LocalTensor<int32_t> mmOutInUb = vecInQueue.DeQue<cT::T>();
    const uint32_t computeSize = curVecBaseM * alignBaseN;

    // Production QuantBMM first forms weightScale*perTokenScale, then dequantizes once.
    // scalar-review: batch-control; one vector-width scale-product per row, no scalar element math
    for (uint32_t row = 0; row < curVecBaseM; ++row) {
        Mul(pertokenBrcbLocal[row * alignBaseN], pertokenBrcbLocal[row * alignBaseN], scaleInUb, alignBaseN);
    }
    AscendC::PipeBarrier<PIPE_V>();
    Cast(dequantMiddleResult, mmOutInUb, AscendC::RoundMode::CAST_RINT, computeSize);
    AscendC::PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(mmOutInUb);
    Mul(mulsResultLocal, dequantMiddleResult, pertokenBrcbLocal, computeSize);
    AscendC::PipeBarrier<PIPE_V>();

    // C2: preserve the official dequant and publish the production BF16 projection.
    LocalTensor<bfloat16_t> yLocalInUb = vecOutQueue.AllocTensor<bfloat16_t>();
    Cast(yLocalInUb, mulsResultLocal, AscendC::RoundMode::CAST_RINT, computeSize);
    AscendC::PipeBarrier<PIPE_V>();
    vecOutQueue.EnQue(yLocalInUb);
}

template <typename MT>
__aicore__ inline void QuantGroupMatmul<MT>::DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset)
{
    // GM拷贝scale
    AscendC::DataCopyPadExtParams<float> padParams;
    AscendC::DataCopyExtParams scaleParams{1, static_cast<uint32_t>(curBaseN * sizeof(float)), 1, 1, 0};
    LocalTensor<float> scaleLocal = scaleInQueue.AllocTensor<float>();
    DataCopyPad(scaleLocal, scaleGm[scaleOffset], scaleParams, padParams);
    scaleInQueue.EnQue(scaleLocal);

    scaleInUb = scaleInQueue.DeQue<float>();
    scaleInUb.SetSize(alignBaseN);
}

template <typename MT>
__aicore__ inline void QuantGroupMatmul<MT>::DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig, uint32_t curBaseM,
                                                                          uint32_t alignBaseN, uint32_t offsetM)
{
    uint64_t perTokenScaleOffset = mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM + offsetM;
    // GM拷贝per token scale
    AscendC::DataCopyPadExtParams<float> padParams;
    AscendC::DataCopyExtParams perTokenScaleParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};
    LocalTensor<float> perTokenScaleLocal = perTokenScaleInQueue.AllocTensor<float>();
    DataCopyPad(perTokenScaleLocal, perTokenScaleGm[perTokenScaleOffset], perTokenScaleParams, padParams);
    perTokenScaleInQueue.EnQue(perTokenScaleLocal);

    perTokenScaleInUb = perTokenScaleInQueue.DeQue<float>();
    const uint32_t broadCastDst[BROADCAST_DIM] = {curBaseM, alignBaseN};
    const uint32_t broadCastSrc[BROADCAST_DIM] = {curBaseM, 1};
    AscendC::BroadCast<float, BROADCAST_DIM, 1>(pertokenBrcbLocal, perTokenScaleInUb, broadCastDst, broadCastSrc,
                                                sharedTmpLocal);
    perTokenScaleInQueue.FreeTensor(perTokenScaleInUb);
}

__aicore__ inline void CopyTiling(QuantGroupMatmulCustomTilingData* tiling, GM_ADDR tilingGM)
{
    uint32_t* ptr =
        reinterpret_cast<uint32_t*>(tiling); // scalar-review: layout-view; stack tiling words, not element compute
    auto tiling32 = reinterpret_cast<__gm__ uint32_t*>(tilingGM); // scalar-review: layout-view; GM tiling words

    // scalar-review: index-init; fixed-size tiling header copy to stack before Matmul Init
    for (uint32_t i = 0; i < sizeof(QuantGroupMatmulCustomTilingData) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

template <typename MT>
__aicore__ inline void RunQW(GM_ADDR qr, GM_ADDR x, GM_ADDR weight, GM_ADDR wqb, GM_ADDR scale, GM_ADDR perTokenScale,
                             GM_ADDR y, GM_ADDR xW, GM_ADDR weightW, GM_ADDR wOut, GM_ADDR ropeSin, GM_ADDR ropeCos,
                             GM_ADDR q8, GM_ADDR qs, GM_ADDR w16, GM_ADDR workspace,
                             QuantGroupMatmulCustomTilingData* tilingData)
{
    AscendC::TPipe tPipe;
    MT mm;
    if ASCEND_IS_AIC {
        mm.SetSubBlockIdx(0);
        mm.Init(&tilingData->mmTilingData, &tPipe);
    }
    QuantGroupMatmul<MT> op(mm);
    op.Init(qr, x, weight, wqb, scale, perTokenScale, y, xW, weightW, wOut, ropeSin, ropeCos, q8, qs, w16, workspace,
            tilingData, &tPipe, tilingData->get_denseLayout() != 0);
    op.Process();
}

static __aicore__ inline void RunQwEntry(
    GM_ADDR qr, GM_ADDR x, GM_ADDR weight, GM_ADDR wqb, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR y,
    GM_ADDR xW, GM_ADDR weightW, GM_ADDR wOut, GM_ADDR ropeSin, GM_ADDR ropeCos, GM_ADDR q8, GM_ADDR qs,
    GM_ADDR w16, GM_ADDR workspace, GM_ADDR tilingGm)
{
    QuantGroupMatmulCustomTilingData tilingData = {};
    CopyTiling(&tilingData, tilingGm);
    if (tilingData.get_denseLayout() != 0) {
        RunQW<MT_Dense>(qr, x, weight, wqb, scale, perTokenScale, y, xW, weightW, wOut, ropeSin, ropeCos, q8, qs, w16,
                        workspace, &tilingData);
    } else {
        RunQW<MT_Prequant>(qr, x, weight, wqb, scale, perTokenScale, y, xW, weightW, wOut, ropeSin, ropeCos, q8, qs,
                           w16, workspace, &tilingData);
    }
}

// The package-level macro selects one literal CANN entry; both branches share
// RunQwEntry, so the device algorithm has a single source of truth.
#ifdef INDEXER_PROLOGUE_QW_PREQUANTIZED
extern "C" __global__ __aicore__ void fi_indexer_prologue_qw_prequantized(
#else
extern "C" __global__ __aicore__ void fi_indexer_prologue_qw(
#endif
    GM_ADDR qr, GM_ADDR x, GM_ADDR weight, GM_ADDR wqb, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR xW,
    GM_ADDR weightW, GM_ADDR wOut, GM_ADDR ropeSin, GM_ADDR ropeCos, GM_ADDR q8, GM_ADDR qs, GM_ADDR w16,
    GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunQwEntry(qr, x, weight, wqb, scale, perTokenScale, y, xW, weightW, wOut, ropeSin, ropeCos, q8, qs, w16,
               workspace, tilingGm);
}
#endif  // __CCE_AICORE__ == 220
