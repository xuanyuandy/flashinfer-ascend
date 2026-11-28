/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 220
#error "IndexerPrologueK requires arch22 (Ascend910B / Ascend910C)"
#else

/*
 * Single mixed-launch IndexerPrologueK full fusion.
 *   AIC:  [<=16,512] x [512,128] projection, full K=512, BF16 VECIN publication
 *   AIV:  MergeN RMSNorm -> adjacent-pair tail RoPE -> per-row dynamic INT8
 *         quantization -> independent-stride key/scale cache publication.
 * The normalized BF16 boundary and R2 quant boundaries are preserved: the
 * amax tail uses the BF16 RoPE round trip, while division uses raw FP32 RoPE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace optiling {
// Device-side ABI mirror of the host TILING_DATA fields, in identical order.
struct IndexerPrologueKFullFusionTilingData {
    int32_t totalRows;
    int32_t numBatch;
    int32_t numSlotPerBatch;
    int32_t usedCores;
    int32_t baseM;
    uint64_t kCacheBlockStride;
    uint64_t kCacheSlotStride;
    uint64_t kScaleBlockStride;
    uint64_t kScaleSlotStride;
    float eps;
    int32_t normIsFp32;
    // Host TILING_DATA_FIELD_DEF_STRUCT always aligns nested structs to 8.
    // TCubeTiling itself only requires 4-byte alignment in a device mirror.
    alignas(8) TCubeTiling mmTiling;
};
} // namespace optiling

namespace {
constexpr int32_t INPUT_DIM = 512;
constexpr int32_t HEAD_DIM = 128;
constexpr int32_t ROPE_DIM = 64;
constexpr int32_t ROPE_HALF = 32;
static_assert(2 * ROPE_HALF + ROPE_DIM == 128, "BuildTables must fit the 128-entry int32 buffer");
constexpr int32_t BASE_M = 16;
constexpr float INV_HEAD_DIM = 1.0f / 128.0f;
constexpr float SCALE_FLOOR = 1e-12f;
constexpr uint32_t REDUCE_SHAPE_SEG[2] = {1, ROPE_DIM};

__aicore__ inline void ReduceSumMultiN128(LocalTensor<float>& dst, LocalTensor<float>& src, LocalTensor<float>& work,
                                          int32_t rowCount)
{
    Duplicate(work, 0.0f, rowCount * 64);
    PipeBarrier<PIPE_V>();
    for (int32_t elemIndex = 0; elemIndex < 128;
         elemIndex += 64) { // scalar-review: small-tile; exactly two 64-lane chunks
        Add(work, src[elemIndex], work, 64, rowCount, {1, 1, 1, 8, 16, 8});
        PipeBarrier<PIPE_V>();
    }
    AscendCUtils::SetMask<float>(64);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceSum<float, false>(dst, work, 64, rowCount, 1, 1, 8);
    }
#else
    WholeReduceSum<float, false>(dst, work, 64, rowCount, 1, 1, 8);
#endif
    PipeBarrier<PIPE_V>();
}

__aicore__ inline float ReduceSum128(LocalTensor<float>& src)
{
    Add(src, src, src[64], 64);
    PipeBarrier<PIPE_V>();
    AscendCUtils::SetMask<float>(64);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceSum<float, false>(src, src, 64, 1, 0, 1, 0);
    }
#else
    WholeReduceSum<float, false>(src, src, 64, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    const event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVS);
    WaitFlag<HardEvent::V_S>(eventVS);
    return src.GetValue(0);
}

class KernelIndexerPrologueKFullFusion {
public:
#if defined(DTYPE_NORM_WEIGHT)
    using NormWeightT = DTYPE_NORM_WEIGHT;
#else
    using NormWeightT = bfloat16_t;
#endif
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    __aicore__ inline void Init(GM_ADDR latent, GM_ADDR wk, GM_ADDR normWeight, GM_ADDR ropeCos, GM_ADDR ropeSin,
                                GM_ADDR indices, GM_ADDR kCache, GM_ADDR kScaleCache, GM_ADDR workspace,
                                const optiling::IndexerPrologueKFullFusionTilingData& tilingData)
    {
        totalRows = tilingData.totalRows;
        usedCores = tilingData.usedCores;
        baseM = tilingData.baseM;
        normIsFp32 = tilingData.normIsFp32;
        eps = tilingData.eps;
        keyBlockStride = tilingData.kCacheBlockStride;
        keySlotStride = tilingData.kCacheSlotStride;
        scaleBlockStride = tilingData.kScaleBlockStride;
        scaleSlotStride = tilingData.kScaleSlotStride;
        mmTiling = tilingData.mmTiling;
        latentArgs = latent;
        indicesArgs = indices;
        workspaceArgs = workspace;
        normWeightArgs = normWeight;

        wkGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wk),
                             HEAD_DIM * INPUT_DIM); // scalar-review: layout-view; typed GM view
        normGm.SetGlobalBuffer(reinterpret_cast<__gm__ NormWeightT*>(normWeight),
                               HEAD_DIM); // scalar-review: layout-view; typed GM view
        normGmF32.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(normWeight),
                                  HEAD_DIM); // scalar-review: layout-view; typed GM view
        cosGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ropeCos),
                              totalRows * ROPE_DIM); // scalar-review: layout-view; typed GM view
        sinGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ropeSin),
                              totalRows * ROPE_DIM); // scalar-review: layout-view; typed GM view
        const int32_t nb = tilingData.numBatch;
        const int32_t ns = tilingData.numSlotPerBatch;
        const uint64_t keySpan = nb > 0 && ns > 0 ? static_cast<uint64_t>(nb - 1) * keyBlockStride +
                                                        static_cast<uint64_t>(ns - 1) * keySlotStride + HEAD_DIM :
                                                    0;
        const uint64_t scaleSpan = nb > 0 && ns > 0 ? static_cast<uint64_t>(nb - 1) * scaleBlockStride +
                                                          static_cast<uint64_t>(ns - 1) * scaleSlotStride + 1 :
                                                      0;
        kCacheGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(kCache),
                                 keySpan); // scalar-review: layout-view; strided cache view
        kScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(kScaleCache),
                                 scaleSpan); // scalar-review: layout-view; strided cache view

        pipe.InitBuffer(inQueX, 1, BASE_M * HEAD_DIM * sizeof(bfloat16_t));
        pipe.InitBuffer(inQueGamma, 1, HEAD_DIM * sizeof(float));
        pipe.InitBuffer(inQueSin, 2, ROPE_DIM * sizeof(float));
        pipe.InitBuffer(inQueCos, 2, ROPE_DIM * sizeof(float));
        pipe.InitBuffer(inQueIdx, 1, BASE_M * 2 * sizeof(int32_t));
        pipe.InitBuffer(outQueQ, 2, HEAD_DIM * sizeof(int8_t));
        pipe.InitBuffer(outQueS, 2, 16 * sizeof(half));
        pipe.InitBuffer(bufY, BASE_M * HEAD_DIM * sizeof(bfloat16_t));
        pipe.InitBuffer(bufRstd, BASE_M * sizeof(float));
        pipe.InitBuffer(bufX32, BASE_M * HEAD_DIM * sizeof(float));
        pipe.InitBuffer(bufSq, BASE_M * HEAD_DIM * sizeof(float));
        pipe.InitBuffer(bufRstdWork, BASE_M * 64 * sizeof(float));
        pipe.InitBuffer(bufGamma32, HEAD_DIM * sizeof(float));
        pipe.InitBuffer(bufTables, 128 * sizeof(uint32_t));
        pipe.InitBuffer(bufBf16, (HEAD_DIM + ROPE_DIM) * sizeof(bfloat16_t));
        pipe.InitBuffer(bufHalf, HEAD_DIM * sizeof(half));
        pipe.InitBuffer(bufWork, 512);
        BuildTables();
    }

    __aicore__ inline void InitTile(int32_t tile)
    {
        rowBegin = tile * baseM;
        rowCount = totalRows - rowBegin;
        if (rowCount > baseM) {
            rowCount = baseM;
        }
        latentGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(latentArgs) +
                                     rowBegin * INPUT_DIM, // scalar-review: layout-view; tile base view
                                 rowCount * INPUT_DIM);
        idxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indicesArgs) + rowBegin * 2,
                              rowCount * 2); // scalar-review: layout-view; tile base view
        mmWorkspaceGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(GetUserWorkspace(workspaceArgs)) +
                GetBlockIdx() * BASE_M * HEAD_DIM, // scalar-review: layout-view; per-core Matmul workspace only
            BASE_M * HEAD_DIM);
    }

    __aicore__ inline void Process()
    {
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &mmTiling);
        const int32_t totalTiles = (totalRows + baseM - 1) / baseM;
        for (int32_t tile = GetBlockIdx(); tile < totalTiles;
             tile += usedCores) { // scalar-review: batch-control; core-strided tile loop
            InitTile(tile);
            ProcessTile();
        }
        mm.End();
    }

    __aicore__ inline void ProcessTile()
    {
        LocalTensor<float> gamma32 = bufGamma32.Get<float>();
        if (normIsFp32 != 0) {
            LocalTensor<float> gamma = inQueGamma.AllocTensor<float>();
            DataCopy(gamma, normGmF32, HEAD_DIM);
            inQueGamma.EnQue(gamma);
            gamma = inQueGamma.DeQue<float>();
            LoadGamma32(gamma32, gamma);
            inQueGamma.FreeTensor(gamma);
        } else {
            LocalTensor<NormWeightT> gamma = inQueGamma.AllocTensor<NormWeightT>();
            DataCopy(gamma, normGm, HEAD_DIM);
            inQueGamma.EnQue(gamma);
            gamma = inQueGamma.DeQue<NormWeightT>();
            LoadGamma32(gamma32, gamma);
            inQueGamma.FreeTensor(gamma);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<bfloat16_t> x = inQueX.AllocTensor<bfloat16_t>();
        mm.SetWorkspace(mmWorkspaceGm);
        mm.SetTensorA(latentGm, false);
        mm.SetTensorB(wkGm, true);
        if (rowCount < baseM) {
            mm.SetTail(rowCount, HEAD_DIM, INPUT_DIM);
        }
        mm.IterateAll<false>(mmWorkspaceGm, 0, false, true);
        mm.WaitIterateAll();
        DataCopyExtParams cpX{1, static_cast<uint32_t>(rowCount * HEAD_DIM * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padX{false, 0, 0, 0};
        DataCopyPad(x, mmWorkspaceGm, cpX, padX);
        inQueX.EnQue(x);
        x = inQueX.DeQue<bfloat16_t>();

        LocalTensor<bfloat16_t> y = bufY.Get<bfloat16_t>();
        LocalTensor<float> x32 = bufX32.Get<float>();
        LocalTensor<float> sq = bufSq.Get<float>();
        LocalTensor<float> work = bufRstdWork.Get<float>();
        LocalTensor<float> rstd = bufRstd.Get<float>();
        LocalTensor<int32_t> idx = inQueIdx.AllocTensor<int32_t>();
        DataCopyExtParams cpIdx{1, static_cast<uint32_t>(rowCount * 2 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padIdx{false, 0, 0, 0};
        DataCopyPad(idx, idxGm, cpIdx, padIdx);
        inQueIdx.EnQue(idx);
        idx = inQueIdx.DeQue<int32_t>();

        // Ordinary MergeN/R2 post: RMSNorm, RoPE, full-row amax, q8/scale publication.
        if (rowCount > 1) {
            const int32_t tileElems = rowCount * HEAD_DIM;
            Cast(x32, x, RoundMode::CAST_NONE, tileElems);
            PipeBarrier<PIPE_V>();
            Mul(sq, x32, x32, tileElems);
            PipeBarrier<PIPE_V>();
            ReduceSumMultiN128(rstd, sq, work, rowCount);
            Muls(rstd, rstd, INV_HEAD_DIM, rowCount);
            PipeBarrier<PIPE_V>();
            Adds(rstd, rstd, eps, rowCount);
            PipeBarrier<PIPE_V>();
            Sqrt(rstd, rstd, rowCount);
            PipeBarrier<PIPE_V>();
            Duplicate(work, 1.0f, rowCount);
            PipeBarrier<PIPE_V>();
            Div(rstd, work, rstd, rowCount);
            PipeBarrier<PIPE_V>();
            const event_t rstdReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(rstdReady);
            WaitFlag<HardEvent::V_S>(rstdReady);
        }
        for (int32_t row = 0; row < rowCount;
             ++row) { // scalar-review: small-tile; at most 16 rows, 128-lane vector math per row
            float rstdValue = 0.0f;
            if (rowCount == 1) {
                const event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                const event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
                Cast(x32, x[row * HEAD_DIM], RoundMode::CAST_NONE, HEAD_DIM);
                PipeBarrier<PIPE_V>();
                Mul(sq, x32, x32, HEAD_DIM);
                PipeBarrier<PIPE_V>();
                const float sum = ReduceSum128(sq);
                SetFlag<HardEvent::V_S>(eventVS);
                WaitFlag<HardEvent::V_S>(eventVS);
                rstdValue = 1.0f / sqrt(sum * INV_HEAD_DIM + eps);
                rstd.SetValue(row, rstdValue);
                SetFlag<HardEvent::S_V>(eventSV);
                WaitFlag<HardEvent::S_V>(eventSV);
            } else {
                rstdValue = rstd.GetValue(row);
            }
            Muls(sq, x32[row * HEAD_DIM], rstdValue, HEAD_DIM);
            PipeBarrier<PIPE_V>();
            Mul(sq, sq, gamma32, HEAD_DIM);
            PipeBarrier<PIPE_V>();
            Cast(y[row * HEAD_DIM], sq, RoundMode::CAST_RINT, HEAD_DIM);
            PipeBarrier<PIPE_V>();
        }
        for (int32_t row = 0; row < rowCount; ++row) { // scalar-review: batch-control; independent full-row post
            ComputePostRow(row, y[row * HEAD_DIM], x32, idx);
        }
        inQueIdx.FreeTensor(idx);
        inQueX.FreeTensor(x);
    }

private:
    __aicore__ inline void LoadGamma32(LocalTensor<float>& dst, LocalTensor<float>& src)
    {
        Adds(dst, src, 0.0f, HEAD_DIM);
    }

    __aicore__ inline void LoadGamma32(LocalTensor<float>& dst, LocalTensor<bfloat16_t>& src)
    {
        Cast(dst, src, RoundMode::CAST_NONE, HEAD_DIM);
    }

    __aicore__ inline void CopyAddRows(LocalTensor<int32_t> dst, LocalTensor<int32_t> src, int32_t rowOffset,
                                       int32_t count)
    {
        DataCopy(dst, src, count);
        PipeBarrier<PIPE_ALL>(); // one-shot local-copy/table initialization, not a Cube/Vector dependency
        Adds(dst, dst, rowOffset, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildTables()
    {
        LocalTensor<int32_t> tab = bufTables.Get<int32_t>();
        LocalTensor<int32_t> evOff = tab;
        LocalTensor<int32_t> odOff = tab[ROPE_HALF];
        LocalTensor<int32_t> itOff = tab[2 * ROPE_HALF];
        // scalar-review: index-init; one-shot gather-table seed, followed by vector doubling
        for (int32_t i = 0; i < 8; ++i) {
            evOff.SetValue(i, 8 * i);
        }
        PipeBarrier<PIPE_ALL>();
        CopyAddRows(evOff[8], evOff, 64, 8);
        CopyAddRows(evOff[16], evOff, 128, 16);
        CopyAddRows(odOff, evOff, 4, ROPE_HALF);
        const int32_t seed[8] = {0, 128, 4, 132, 8, 136, 12, 140};
        // scalar-review: index-init; one-shot interleave-table seed
        for (int32_t i = 0; i < 8; ++i) {
            itOff.SetValue(i, seed[i]);
        }
        PipeBarrier<PIPE_ALL>();
        CopyAddRows(itOff[8], itOff, 16, 8);
        CopyAddRows(itOff[16], itOff, 32, 16);
        // 64-entry interleave table: 8 -> 16 -> 32 -> 64; the final step
        // extends the first 32 entries and must not write past tab[127].
        CopyAddRows(itOff[32], itOff, 64, ROPE_HALF);
    }

    __aicore__ inline void ComputePostRow(int32_t localRow, LocalTensor<bfloat16_t> nbf, LocalTensor<float> f,
                                          LocalTensor<int32_t>& idx)
    {
        const int32_t globalRow = rowBegin + localRow;
        LocalTensor<float> sinLocal = inQueSin.AllocTensor<float>();
        LocalTensor<float> cosLocal = inQueCos.AllocTensor<float>();
        DataCopyExtParams cpRope{1, ROPE_DIM * sizeof(float), 0, 0, 0};
        DataCopyPadExtParams<float> padF{false, 0, 0, 0};
        DataCopyPad(sinLocal, sinGm[globalRow * ROPE_DIM], cpRope, padF);
        DataCopyPad(cosLocal, cosGm[globalRow * ROPE_DIM], cpRope, padF);
        inQueSin.EnQue(sinLocal);
        inQueCos.EnQue(cosLocal);
        sinLocal = inQueSin.DeQue<float>();
        cosLocal = inQueCos.DeQue<float>();

        // R2 FP32 scratch layout; f aliases the dead projection x32 buffer.
        LocalTensor<float> n32 = f[416];
        LocalTensor<float> t32full = f[544];
        LocalTensor<float> oe32 = f[608];
        LocalTensor<float> oo32 = f[640];
        LocalTensor<float> e32 = f[672];
        LocalTensor<float> o32 = f[704];
        LocalTensor<float> sinE = f[736];
        LocalTensor<float> sinO = f[768];
        LocalTensor<float> cosE = f[800];
        LocalTensor<float> cosO = f[832];
        LocalTensor<float> t1 = f[992];
        LocalTensor<float> t2 = f[1024];
        LocalTensor<float> rowf = f[1120];
        LocalTensor<float> absf = f[1248];
        LocalTensor<float> amax = f[1376];
        LocalTensor<float> scaleV = f[1408];
        LocalTensor<float> qf = f[1536];
        LocalTensor<float> sv = f[1664];
        LocalTensor<uint8_t> work = bufWork.Get<uint8_t>();
        LocalTensor<bfloat16_t> bf16v = bufBf16.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> ropeBf = bf16v[HEAD_DIM];
        LocalTensor<uint32_t> tab = bufTables.Get<int32_t>().ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> evOff = tab;
        LocalTensor<uint32_t> odOff = tab[ROPE_HALF];
        LocalTensor<uint32_t> itOff = tab[2 * ROPE_HALF];
        LocalTensor<half> q16 = bufHalf.Get<half>();

        Cast(n32, nbf, RoundMode::CAST_NONE, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        LocalTensor<bfloat16_t> tailBf = nbf[ROPE_DIM];
        Cast(t32full, tailBf, RoundMode::CAST_NONE, ROPE_DIM);
        PipeBarrier<PIPE_V>();
        Gather(e32, t32full, evOff, 0, ROPE_HALF);
        Gather(o32, t32full, odOff, 0, ROPE_HALF);
        Gather(sinE, sinLocal, evOff, 0, ROPE_HALF);
        Gather(sinO, sinLocal, odOff, 0, ROPE_HALF);
        Gather(cosE, cosLocal, evOff, 0, ROPE_HALF);
        Gather(cosO, cosLocal, odOff, 0, ROPE_HALF);
        PipeBarrier<PIPE_V>();
        Mul(t1, e32, cosE, ROPE_HALF);
        Mul(t2, o32, sinE, ROPE_HALF);
        PipeBarrier<PIPE_V>();
        Sub(oe32, t1, t2, ROPE_HALF);
        PipeBarrier<PIPE_V>();
        Mul(t1, o32, cosO, ROPE_HALF);
        Mul(t2, e32, sinO, ROPE_HALF);
        PipeBarrier<PIPE_V>();
        Add(oo32, t1, t2, ROPE_HALF);
        PipeBarrier<PIPE_V>();
        Gather(rowf[ROPE_DIM], oe32, itOff, 0, ROPE_DIM);
        PipeBarrier<PIPE_V>();

        // BF16 RoPE publication is used only for amax, matching R2/golden.
        Cast(ropeBf, rowf[ROPE_DIM], RoundMode::CAST_RINT, ROPE_DIM);
        PipeBarrier<PIPE_V>();
        Cast(t32full, ropeBf, RoundMode::CAST_NONE, ROPE_DIM);
        PipeBarrier<PIPE_V>();
        Cast(rowf, nbf, RoundMode::CAST_NONE, ROPE_DIM);
        PipeBarrier<PIPE_V>();
        Abs(absf, rowf, ROPE_DIM);
        PipeBarrier<PIPE_V>();
        Abs(absf[ROPE_DIM], t32full, ROPE_DIM);
        PipeBarrier<PIPE_V>();
        ReduceMax<float, Pattern::Reduce::AR, false>(amax, absf, work, REDUCE_SHAPE_SEG, true);
        PipeBarrier<PIPE_V>();
        ReduceMax<float, Pattern::Reduce::AR, false>(t2, absf[ROPE_DIM], work, REDUCE_SHAPE_SEG, true);
        PipeBarrier<PIPE_V>();
        const event_t amaxEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(amaxEvent);
        WaitFlag<HardEvent::V_S>(amaxEvent);
        float amaxF = amax.GetValue(0);
        const float amaxTail = t2.GetValue(0);
        if (amaxTail > amaxF) {
            amaxF = amaxTail;
        }
        float scaleSafe = amaxF / 127.0f;
        if (scaleSafe < SCALE_FLOOR) {
            scaleSafe = SCALE_FLOOR;
        }
        const event_t scaleToVectorEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(scaleToVectorEvent);
        WaitFlag<HardEvent::S_V>(scaleToVectorEvent);
        Duplicate(scaleV, scaleSafe, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        // Head division consumes normalized BF16->FP32. Tail division consumes
        // raw FP32 RoPE; replacing it with the BF16 amax copy changes q8.
        Div(qf, n32, scaleV, ROPE_DIM);
        Div(qf[ROPE_DIM], rowf[ROPE_DIM], scaleV, ROPE_DIM);
        PipeBarrier<PIPE_V>();
        Mins(qf, qf, 127.0f, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Maxs(qf, qf, -127.0f, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Cast(q16, qf, RoundMode::CAST_TRUNC, HEAD_DIM);
        PipeBarrier<PIPE_V>();

        LocalTensor<int8_t> q8 = outQueQ.AllocTensor<int8_t>();
        Cast(q8, q16, RoundMode::CAST_NONE, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        outQueQ.EnQue(q8);
        LocalTensor<half> s16 = outQueS.AllocTensor<half>();
        Duplicate(sv, scaleSafe, 8);
        PipeBarrier<PIPE_V>();
        Cast(s16, sv, RoundMode::CAST_NONE, 8);
        PipeBarrier<PIPE_V>();
        outQueS.EnQue(s16);
        q8 = outQueQ.DeQue<int8_t>();
        s16 = outQueS.DeQue<half>();

        const int32_t b = idx.GetValue(localRow * 2);
        const int32_t s = idx.GetValue(localRow * 2 + 1);
        const bool skipWrite = (b == -1 && s == -1);
        if (!skipWrite) {
            const uint64_t keyOffset =
                static_cast<uint64_t>(b) * keyBlockStride + static_cast<uint64_t>(s) * keySlotStride;
            const uint64_t scaleOffset =
                static_cast<uint64_t>(b) * scaleBlockStride + static_cast<uint64_t>(s) * scaleSlotStride;
            DataCopyExtParams cpQ{1, HEAD_DIM * sizeof(int8_t), 0, 0, 0};
            DataCopyExtParams cpS{1, sizeof(half), 0, 0, 0};
            DataCopyPad(kCacheGm[keyOffset], q8, cpQ);
            DataCopyPad(kScaleGm[scaleOffset], s16, cpS);
            // Source slots remain allocated until both cache writes complete.
            const event_t copyOutDone = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(copyOutDone);
            WaitFlag<HardEvent::MTE3_V>(copyOutDone);
        }
        outQueQ.FreeTensor(q8);
        outQueS.FreeTensor(s16);
        inQueSin.FreeTensor(sinLocal);
        inQueCos.FreeTensor(cosLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueX;
    TQue<TPosition::VECIN, 1> inQueGamma;
    TQue<TPosition::VECIN, 1> inQueSin;
    TQue<TPosition::VECIN, 1> inQueCos;
    TQue<TPosition::VECIN, 1> inQueIdx;
    TQue<TPosition::VECOUT, 1> outQueQ;
    TQue<TPosition::VECOUT, 1> outQueS;
    TBuf<TPosition::VECCALC> bufY;
    TBuf<TPosition::VECCALC> bufRstd;
    TBuf<TPosition::VECCALC> bufX32;
    TBuf<TPosition::VECCALC> bufSq;
    TBuf<TPosition::VECCALC> bufRstdWork;
    TBuf<TPosition::VECCALC> bufGamma32;
    TBuf<TPosition::VECCALC> bufTables;
    TBuf<TPosition::VECCALC> bufBf16;
    TBuf<TPosition::VECCALC> bufHalf;
    TBuf<TPosition::VECCALC> bufWork;
    Matmul<AType, BType, CType, BiasType> mm;
    TCubeTiling mmTiling;

    GlobalTensor<bfloat16_t> latentGm;
    GlobalTensor<bfloat16_t> wkGm;
    GlobalTensor<NormWeightT> normGm;
    GlobalTensor<float> normGmF32;
    GlobalTensor<float> cosGm;
    GlobalTensor<float> sinGm;
    GlobalTensor<int32_t> idxGm;
    GlobalTensor<int8_t> kCacheGm;
    GlobalTensor<half> kScaleGm;
    GlobalTensor<bfloat16_t> mmWorkspaceGm;
    GM_ADDR latentArgs = nullptr;
    GM_ADDR indicesArgs = nullptr;
    GM_ADDR workspaceArgs = nullptr;
    GM_ADDR normWeightArgs = nullptr;
    int32_t normIsFp32 = 0;
    int32_t totalRows = 0;
    int32_t usedCores = 0;
    int32_t baseM = BASE_M;
    int32_t rowBegin = 0;
    int32_t rowCount = 0;
    uint64_t keyBlockStride = 0;
    uint64_t keySlotStride = 0;
    uint64_t scaleBlockStride = 0;
    uint64_t scaleSlotStride = 0;
    float eps = 0.0f;
};
} // namespace

extern "C" __global__ __aicore__ void fi_indexer_prologue_k(GM_ADDR latent, GM_ADDR wk, GM_ADDR norm_weight,
                                                                     GM_ADDR rope_cos, GM_ADDR rope_sin,
                                                                     GM_ADDR indices, GM_ADDR k_cache,
                                                                     GM_ADDR k_scale_cache, GM_ADDR k_cache_out,
                                                                     GM_ADDR k_scale_cache_out, GM_ADDR workspace,
                                                                     GM_ADDR tiling)
{
    (void)k_cache;
    (void)k_scale_cache;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    GET_TILING_DATA_WITH_STRUCT(optiling::IndexerPrologueKFullFusionTilingData, tilingData, tiling);
    KernelIndexerPrologueKFullFusion op;
    op.Init(latent, wk, norm_weight, rope_cos, rope_sin, indices, k_cache_out, k_scale_cache_out, workspace,
            tilingData);
    op.Process();
}
#endif  // __CCE_AICORE__ == 220
