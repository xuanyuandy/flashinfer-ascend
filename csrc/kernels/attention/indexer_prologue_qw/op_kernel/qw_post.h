/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * IndexerPrologueQw fused epilogue kernel (Vector / AIV).
 *
 * Inputs (per token t; fixed geometry N=32 heads, D=128, Dr=64):
 *   qb[T,4096] BF16   W8A8 projection output from activation DynamicQuant and
 *                     QuantMatmulWeightNz (not a dense BF16 MatMul)
 *   wb[T,32]   BF16   weights_proj output (aclnnMatmul)
 *   sin/cos[T,64] FP32 interleave rope tables (model-side sliced)
 * Outputs:
 *   q8[T,32,128] INT8  per-(token,head) dynamically quantized q
 *   qs[T,32]     FP16  per-(token,head) scale = absmax/127
 *   w16[T,32]    FP16  wb * (softmax_scale * n_heads**-0.5)
 *
 * Token-tile design (see design/tile_level): all heavy math batched across
 * the whole 32x128 token with 2D byte-offset Gather tables (A2 has no
 * vector Shift/Bitwise; knowledge card a2_bit_reorder_gather_byte_offset_
 * no_shift; K-side 10/10-passed precedent). Split-domain quant:
 *   x32[4096] -> headC[32,64] (gather) | e/o split (gather) -> rotate
 *   -> oe|oo (in-place into e|o slots, 4 rotate temps live in dead x32)
 *   -> rotBf (bf16 publish) -> tailRow[32,64] (bf16-domain gather + cast,
 *      the bf16 round-trip is the golden's re-read boundary)
 *   amax: ReduceMax AR {32,64} per half + elementwise Max -> scale[32]
 *   quant: Broadcast scale [32,1]->[32,64], Div + Round (nearest-even) per half,
 *      fp32->fp16 CAST_NONE, interleave-out via half-domain gather, fp16->int8 cast.
 * Every table has a UNIFORM per-row increment (512/128/64 bytes) so the
 * seed+doubling construction is exact; odd-lane tables reuse the even-lane
 * table through Gather's srcBaseOffset (+4 fp32 / bf16 pairs use dedicated
 * 2-aligned offsets). All gather offsets stay inside their source buffer.
 *
 * Precision contract (vs model.py golden, K-side precedent):
 *   - head quant input: exact bf16 GEMM output (cast fp32) -> ZERO boundary
 *     difference vs golden.
 *   - tail quant input: bf16 round-trip of the fp32 rope result (golden
 *     publishes rope to bf16 in place and re-reads it for quant).
 *   - rotate mul/sub order bit-aligned with golden: oe = e*cosE - o*sinE,
 *     oo = o*cosO + e*sinO.
 *   - scale = Div(mx, 127), clamp_min 2**-24, rounded to FP16 and converted
 *     back to FP32 as the divisor (production quantize_indexer_query);
 *     FP16 scale is published with the same rounded value.
 *   - quant chain: Div + block Round (nearest-even, matching nearbyint) +
 *     Mins/Maxs clamp + fp32->fp16 CAST_NONE + fp16->int8 CAST_NONE.
 *     No direct fp32->int8 cast (unsupported).
 *   - W: exact bf16 -> fp32 cast, fp32 scalar mul, fp32->fp16 CAST_NONE.
 *   - no RMSNorm in this operator -> no vector-Rsqrt hazard.
 *
 * scalar-review exemptions (scalar-to-vector.md):
 *   - batch-control: token-level task loop, ComputeToken math is batched
 *     vector (1024/2048/4096 wide)
 *   - index-init: 8-value table seeds in Init (one-shot, then vector
 *     Copy+Adds doubling; zero per-launch GM table traffic)
 *   - no GM scalar access anywhere (all indices derived from token id)
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t N_HEADS = 32;
constexpr int32_t HEAD_DIM = 128;
constexpr int32_t ROPE_DIM = 64;
constexpr int32_t ROPE_HALF = 32;                      // pairs per head
constexpr int32_t TOKEN_ELEMS = N_HEADS * HEAD_DIM;    // 4096
constexpr int32_t EO_ELEMS = 2 * N_HEADS * ROPE_HALF;  // 2048 (e|o -> oe|oo)
constexpr int32_t HALF_ELEMS = N_HEADS * ROPE_HALF;    // 1024
constexpr float SCALE_FLOOR = 5.9604644775390625e-08f; // 2**-24, production query-quant floor

__aicore__ inline int32_t MinI32(int32_t a, int32_t b)
{
    return a < b ? a : b;
}

template <typename Q>
struct QueueRef {
    Q* ptr;
    template <typename T>
    __aicore__ inline auto AllocTensor()
    {
        return ptr->template AllocTensor<T>();
    }
    template <typename T>
    __aicore__ inline auto DeQue()
    {
        return ptr->template DeQue<T>();
    }
    template <typename T>
    __aicore__ inline void EnQue(LocalTensor<T>& tensor)
    {
        ptr->EnQue(tensor);
    }
    template <typename T>
    __aicore__ inline void FreeTensor(LocalTensor<T>& tensor)
    {
        ptr->FreeTensor(tensor);
    }
};

struct PostBuffer {
    LocalTensor<uint8_t> data;
    template <typename T>
    __aicore__ inline LocalTensor<T> Get()
    {
        return data.ReinterpretCast<T>();
    }
};

class QWPost {
public:
    __aicore__ inline void Init(GM_ADDR qb, GM_ADDR wb, GM_ADDR ropeSin, GM_ADDR ropeCos, GM_ADDR q8, GM_ADDR qs,
                                GM_ADDR w16, TQue<TPosition::VECIN, 1>* vecIn, TQue<TPosition::VECOUT, 1>* vecOut,
                                TQue<TPosition::VECIN, 1>* scaleIn, TQue<TPosition::VECIN, 2>* perTokenScaleIn,
                                TQue<TPosition::VECOUT, 1>* qsOut, TQue<TPosition::VECOUT, 2>* wOut,
                                TBuf<TPosition::VECCALC>* tmp, int32_t totalTokensIn, int32_t firstRowIn,
                                int32_t totalVecIn, int32_t coreIdxIn, float wScaleIn)
    {
        totalTokens = totalTokensIn;
        firstRow = firstRowIn;
        totalVec = totalVecIn;
        coreIdx = coreIdxIn;
        wScale = wScaleIn;
        inQueQb.ptr = vecIn;
        inQueSin.ptr = scaleIn;
        inQueCos.ptr = scaleIn;
        inQueWb.ptr = perTokenScaleIn;
        outQueQ8.ptr = vecOut;
        outQueQs.ptr = qsOut;
        outQueW.ptr = wOut;

        const uint64_t n = static_cast<uint64_t>(totalTokens);
        qbGm.SetGlobalBuffer((__gm__ bfloat16_t*)qb, n * TOKEN_ELEMS);
        wbGm.SetGlobalBuffer((__gm__ bfloat16_t*)wb, n * N_HEADS);
        sinGm.SetGlobalBuffer((__gm__ float*)ropeSin, n * ROPE_DIM);
        cosGm.SetGlobalBuffer((__gm__ float*)ropeCos, n * ROPE_DIM);
        q8Gm.SetGlobalBuffer((__gm__ int8_t*)q8, n * TOKEN_ELEMS);
        qsGm.SetGlobalBuffer((__gm__ half*)qs, n * N_HEADS);
        wGm.SetGlobalBuffer((__gm__ half*)w16, n * N_HEADS);

        // Alias the now-dead projection scratch: about 113KB, with later
        // q16/q16g/scale/broadcast/work views sharing expired lifetimes.
        bufS.data = tmp->GetWithOffset<uint8_t>(16384, 0);
        bufEO.data = tmp->GetWithOffset<uint8_t>(8192, 16384);
        bufRotBf.data = tmp->GetWithOffset<uint8_t>(4096, 24576);
        bufTailBf.data = tmp->GetWithOffset<uint8_t>(4096, 28672);
        bufQ16.data = tmp->GetWithOffset<uint8_t>(8192, 24576);
        bufTailF.data = tmp->GetWithOffset<uint8_t>(8192, 32768);
        bufHeadC.data = tmp->GetWithOffset<uint8_t>(8192, 40960);
        bufAbs.data = tmp->GetWithOffset<uint8_t>(8192, 49152);
        bufQf.data = tmp->GetWithOffset<uint8_t>(8192, 49152);
        bufScBC.data = tmp->GetWithOffset<uint8_t>(8192, 16384);
        bufQ16g.data = tmp->GetWithOffset<uint8_t>(8192, 0);
        bufTrig.data = tmp->GetWithOffset<uint8_t>(16384, 57344);
        bufMisc.data = tmp->GetWithOffset<uint8_t>(1024, 73728);
        bufTables.data = tmp->GetWithOffset<uint8_t>(40960, 74752);

        BuildTables();
        Duplicate(c127(), 127.0f, 32);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        // scalar-review: batch-control; token task loop, ComputeToken math is batched vector
        for (int32_t tok = firstRow; tok < totalTokens; tok += totalVec) {
            CopyInToken(tok);
            ComputeToken();
            CopyOutToken(tok);
        }
    }

private:
    // ---- misc slot views (bufMisc, 256B) -----------------------------
    __aicore__ inline LocalTensor<float> mx()
    {
        return bufMisc.Get<float>(); // [32] merged amax
    }
    __aicore__ inline LocalTensor<float> mxH()
    {
        return bufMisc.Get<float>()[32]; // [32] head amax
    }
    __aicore__ inline LocalTensor<float> mxT()
    {
        return bufMisc.Get<float>()[64]; // [32] tail amax
    }
    __aicore__ inline LocalTensor<float> scale()
    {
        return bufMisc.Get<float>()[96]; // [32]
    }
    __aicore__ inline LocalTensor<half> scale16()
    {
        // byte 768: after w32 [640,768), disjoint from c127 [512,640).
        return bufMisc.Get<half>()[384]; // [32] FP16-rounded scale scratch
    }
    __aicore__ inline LocalTensor<float> c127()
    {
        return bufMisc.Get<float>()[128]; // [32] = 127
    }
    __aicore__ inline LocalTensor<float> w32()
    {
        return bufMisc.Get<float>()[160]; // [32]
    }

    __aicore__ inline void CopyAddRows(LocalTensor<int32_t> dst, LocalTensor<int32_t> src, int32_t rowOffset,
                                       int32_t count)
    {
        DataCopy(dst, src, count);
        PipeBarrier<PIPE_ALL>();
        Adds(dst, dst, rowOffset, count);
        PipeBarrier<PIPE_ALL>();
    }

    // seed + doubling helper: replicate a row0 block of `blk` int32 values
    // to `total` entries with per-row increment `rowStep`
    __aicore__ inline void ReplicateRows(LocalTensor<int32_t> t, int32_t blk, int32_t total, int32_t rowStep)
    {
        int32_t filled = blk;
        // scalar-review: index-init; one-shot table doubling loop (<=5 iters, vector Copy/Adds body)
        while (filled < total) {
            int32_t grow = MinI32(filled, total - filled);
            CopyAddRows(t[filled], t, rowStep * (filled / blk), grow);
            filled += grow;
        }
    }

    __aicore__ inline void BuildTables()
    {
        // byte-offset tables for Gather (srcOffset in BYTES; built in the
        // int32 domain -- Adds has no uint32 overload -- consumed as uint32
        // via ReinterpretCast at the call sites).
        // layout: evOff[1024] | repE[1024] | hdOff[2048] | tlOff[2048] | itH[4096]
        LocalTensor<int32_t> tab = bufTables.Get<int32_t>();
        LocalTensor<int32_t> evOff = tab;       // [1024] fp32 rope split
        LocalTensor<int32_t> repE = tab[1024];  // [1024] fp32 sin/cos replicate
        LocalTensor<int32_t> hdOff = tab[2048]; // [2048] fp32 head compact
        LocalTensor<int32_t> tlOff = tab[4096]; // [2048] bf16 tail interleave-in
        LocalTensor<int32_t> itH = tab[6144];   // [4096] half interleave-out

        // evOff row0 = 4*(64 + 2j) = 256 + 8j; rows: +512B (128 f32 per row)
        // scalar-review: index-init; one-shot evOff row-0 seed, then vector doubling
        for (int32_t i = 0; i < 8; ++i) {
            evOff.SetValue(i, 256 + 8 * i);
        }
        PipeBarrier<PIPE_ALL>();
        CopyAddRows(evOff[8], evOff, 64, 8);
        CopyAddRows(evOff[16], evOff, 128, 16);
        ReplicateRows(evOff, 32, 1024, 512);

        // repE: pure repetition of row0 = 8j (sin/cos even lanes), +0/row
        // scalar-review: index-init; one-shot repE row-0 seed, then vector doubling
        for (int32_t i = 0; i < 8; ++i) {
            repE.SetValue(i, 8 * i);
        }
        PipeBarrier<PIPE_ALL>();
        CopyAddRows(repE[8], repE, 64, 8);
        CopyAddRows(repE[16], repE, 128, 16);
        ReplicateRows(repE, 32, 1024, 0);

        // hdOff row0 = 4j (head lanes j=0..63); rows: +512B
        // scalar-review: index-init; one-shot hdOff head-ramp seed, then vector doubling
        for (int32_t i = 0; i < 8; ++i) {
            hdOff.SetValue(i, 4 * i);
        }
        PipeBarrier<PIPE_ALL>();
        CopyAddRows(hdOff[8], hdOff, 32, 8);
        CopyAddRows(hdOff[16], hdOff, 64, 16);
        CopyAddRows(hdOff[32], hdOff, 128, 32);
        ReplicateRows(hdOff, 64, 2048, 512);

        // tlOff (bf16 domain, 2-aligned): rotBf = [oe(1024) | oo(1024)] elems
        //   j' even -> elem 32h + j'/2  -> byte 64h + j'
        //   j' odd  -> elem 1024+32h+(j'-1)/2 -> byte 2048 + 64h + j' - 1
        // row0 pattern: {0, 2048, 2, 2050, 4, 2052, 6, 2054, ...}; rows: +64B
        // scalar-review: index-init; one-shot tlOff interleave seed, then vector doubling
        constexpr int32_t seed[8] = {0, 2048, 2, 2050, 4, 2052, 6, 2054};
        static_assert((seed[0] | seed[1] | seed[2] | seed[3] | seed[4] | seed[5] | seed[6] | seed[7]) % 2 == 0,
                      "Gather BF16 source offsets must be two-byte aligned");
        // scalar-review: index-init; one-shot tlOff seed (8 values), then vector doubling
        for (int32_t i = 0; i < 8; ++i) {
            tlOff.SetValue(i, seed[i]);
        }
        PipeBarrier<PIPE_ALL>();
        CopyAddRows(tlOff[8], tlOff, 8, 8);
        CopyAddRows(tlOff[16], tlOff, 16, 16);
        CopyAddRows(tlOff[32], tlOff, 32, 32);
        ReplicateRows(tlOff, 64, 2048, 64);

        // itH (half domain): src = q16 [qh(2048) | qt(2048)]
        //   out[h*128+j] = qh[h*64+j] (j<64)  -> byte 2*(64h + j)
        //                = qt[h*64+j-64]      -> byte 4096 + 2*(64h + j - 64)
        // row0 block [128]: head ramp 2j (j<64), tail = head + 4096; rows: +128B
        // scalar-review: index-init; one-shot itH head-ramp seed, then vector doubling
        for (int32_t i = 0; i < 8; ++i) {
            itH.SetValue(i, 2 * i);
        }
        PipeBarrier<PIPE_ALL>();
        CopyAddRows(itH[8], itH, 16, 8);
        CopyAddRows(itH[16], itH, 32, 16);
        CopyAddRows(itH[32], itH, 64, 32);
        CopyAddRows(itH[64], itH, 4096, 64);
        ReplicateRows(itH, 128, 4096, 128);
    }

    __aicore__ inline void CopyInToken(int32_t tok)
    {
        const int64_t t = static_cast<int64_t>(tok);
        LocalTensor<bfloat16_t> qbLocal = inQueQb.AllocTensor<bfloat16_t>();
        LocalTensor<float> sinLocal = bufTailF.Get<float>();
        LocalTensor<float> cosLocal = bufQf.Get<float>();
        LocalTensor<bfloat16_t> wbLocal = inQueWb.AllocTensor<bfloat16_t>();
        DataCopyExtParams cpQb{1, static_cast<uint32_t>(TOKEN_ELEMS * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padBf{false, 0, 0, 0};
        DataCopyPad(qbLocal, qbGm[t * TOKEN_ELEMS], cpQb, padBf);
        DataCopyExtParams cpRope{1, static_cast<uint32_t>(ROPE_DIM * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padF{false, 0, 0, 0};
        DataCopyPad(sinLocal, sinGm[t * ROPE_DIM], cpRope, padF);
        DataCopyPad(cosLocal, cosGm[t * ROPE_DIM], cpRope, padF);
        DataCopyExtParams cpW{1, static_cast<uint32_t>(N_HEADS * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(wbLocal, wbGm[t * N_HEADS], cpW, padBf);
        inQueWb.EnQue(wbLocal);
        inQueQb.EnQue(qbLocal);
        const int32_t ropeEvent = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(ropeEvent);
        WaitFlag<HardEvent::MTE2_V>(ropeEvent);
    }

    __aicore__ inline void ComputeToken()
    {
        PipeBarrier<PIPE_V>();
        LocalTensor<bfloat16_t> qbLocal = inQueQb.DeQue<bfloat16_t>();
        LocalTensor<float> sinLocal = bufTailF.Get<float>();
        LocalTensor<float> cosLocal = bufQf.Get<float>();
        LocalTensor<bfloat16_t> wbLocal = inQueWb.DeQue<bfloat16_t>();

        LocalTensor<float> s = bufS.Get<float>();   // [4096] x32, then temps
        LocalTensor<float> eo = bufEO.Get<float>(); // [2048]
        LocalTensor<bfloat16_t> rotBf = bufRotBf.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> tailBf = bufTailBf.Get<bfloat16_t>();
        LocalTensor<float> tailF = bufTailF.Get<float>(); // [2048]
        LocalTensor<float> headC = bufHeadC.Get<float>(); // [2048]
        LocalTensor<float> absf = bufAbs.Get<float>();    // [2048]
        LocalTensor<float> qf = bufQf.Get<float>();       // [2048]
        LocalTensor<float> scBC = bufScBC.Get<float>();   // [2048] as [32,64]
        LocalTensor<half> q16 = bufQ16.Get<half>();       // [4096]
        LocalTensor<half> q16g = bufQ16g.Get<half>();     // [4096]
        LocalTensor<float> trig = bufTrig.Get<float>();   // [4096]
        LocalTensor<float> sinE = trig;
        LocalTensor<float> sinO = trig[1024];
        LocalTensor<float> cosE = trig[2048];
        LocalTensor<float> cosO = trig[3072];
        LocalTensor<uint32_t> tab = bufTables.Get<int32_t>().ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> evOffU = tab;
        LocalTensor<uint32_t> repEU = tab[1024];
        LocalTensor<uint32_t> hdOffU = tab[2048];
        LocalTensor<uint32_t> tlOffU = tab[4096];
        LocalTensor<uint32_t> itHU = tab[6144];
        LocalTensor<uint8_t> workRM = bufEO.Get<uint8_t>();
        LocalTensor<uint8_t> workBC = bufS.Get<uint8_t>();

        // ---- token fp32 (exact bf16->fp32); head compaction first -------
        Cast(s, qbLocal, RoundMode::CAST_NONE, TOKEN_ELEMS);
        Gather(headC, s, hdOffU, 0, 2048); // [32,64] head lanes

        // ---- rope even/odd split (odd lanes via srcBaseOffset +4) ------
        Gather(eo, s, evOffU, 0, 1024);       // e32
        Gather(eo[1024], s, evOffU, 4, 1024); // o32
        Gather(sinE, sinLocal, repEU, 0, 1024);
        Gather(sinO, sinLocal, repEU, 4, 1024);
        Gather(cosE, cosLocal, repEU, 0, 1024);
        Gather(cosO, cosLocal, repEU, 4, 1024);

        // ---- rotate: 4 products into the dead x32 region, then
        //      oe/oo overwrite the e/o slots in place --------------------
        Mul(s, eo, cosE, 1024);                // e*cosE
        Mul(s[1024], eo[1024], sinE, 1024);    // o*sinE
        Mul(s[2048], eo[1024], cosO, 1024);    // o*cosO
        Mul(s[3072], eo, sinO, 1024);          // e*sinO
        Sub(eo, s, s[1024], 1024);             // oe -> e slot
        Add(eo[1024], s[2048], s[3072], 1024); // oo -> o slot

        // ---- bf16 publish + tail interleave-in (bf16-domain gather,
        //      the round-trip == golden's re-read of the bf16 rope row) --
        Cast(rotBf, eo, RoundMode::CAST_RINT, EO_ELEMS);
        Gather(tailBf, rotBf, tlOffU, 0, 2048); // [32,64] tail rows
        Cast(tailF, tailBf, RoundMode::CAST_NONE, EO_ELEMS);

        // ---- per-(token,head) amax over the full 128 lanes -------------
        const uint32_t halfShape[2] = {static_cast<uint32_t>(N_HEADS), static_cast<uint32_t>(ROPE_DIM)};
        Abs(absf, headC, EO_ELEMS);
        ReduceMax<float, Pattern::Reduce::AR, false>(mxH(), absf, workRM, halfShape, true);
        Abs(absf, tailF, EO_ELEMS);
        ReduceMax<float, Pattern::Reduce::AR, false>(mxT(), absf, workRM, halfShape, true);
        PipeBarrier<PIPE_ALL>();
        Max(mx(), mxH(), mxT(), N_HEADS);
        PipeBarrier<PIPE_ALL>();
        Div(scale(), mx(), c127(), N_HEADS); // IEEE div == golden /127
        PipeBarrier<PIPE_ALL>();
        Maxs(scale(), scale(), SCALE_FLOOR, N_HEADS);
        // Production quantize_indexer_query uses the FP16-rounded scale as
        // the FP32 divisor. Round-trip before Broadcast and FP16 publication.
        Cast(scale16(), scale(), RoundMode::CAST_NONE, N_HEADS);
        PipeBarrier<PIPE_ALL>();
        Cast(scale(), scale16(), RoundMode::CAST_NONE, N_HEADS);
        // Keep the completed scale visible to Broadcast and the fp16 publish.
        PipeBarrier<PIPE_ALL>();

        // ---- quant: broadcast scale, per-half div/clamp, snap, cast ----
        const uint32_t bcastDst[2] = {static_cast<uint32_t>(N_HEADS), static_cast<uint32_t>(ROPE_DIM)};
        const uint32_t bcastSrc[2] = {static_cast<uint32_t>(N_HEADS), 1};
        Broadcast<float, 2, 1>(scBC, scale(), bcastDst, bcastSrc, workBC);
        Div(qf, headC, scBC, EO_ELEMS);
        Round(qf, qf, EO_ELEMS);
        Mins(qf, qf, 127.0f, EO_ELEMS);
        Maxs(qf, qf, -127.0f, EO_ELEMS);
        Cast(q16, qf, RoundMode::CAST_NONE, EO_ELEMS); // head half fp16
        Div(qf, tailF, scBC, EO_ELEMS);
        Round(qf, qf, EO_ELEMS);
        Mins(qf, qf, 127.0f, EO_ELEMS);
        Maxs(qf, qf, -127.0f, EO_ELEMS);
        Cast(q16[2048], qf, RoundMode::CAST_NONE, EO_ELEMS); // tail half fp16

        // ---- interleave-out in the half domain, then int8 --------------
        // q16 is [head 32x64 | tail 32x64]. Use independent 1D copies per
        // head to avoid strided multi-row DataCopy boundary behavior.
        for (uint32_t head = 0; head < N_HEADS; ++head) {
            DataCopy(q16g[head * 128], q16[head * 64], 64);
            DataCopy(q16g[head * 128 + 64], q16[2048 + head * 64], 64);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<int8_t> q8Local = outQueQ8.AllocTensor<int8_t>();
        Cast(q8Local, q16g, RoundMode::CAST_NONE, TOKEN_ELEMS); // fp16 -> int8
        outQueQ8.EnQue(q8Local);

        // ---- scale publish fp16, with optional W packed in the next 32 lanes
        LocalTensor<half> qsLocal = outQueQs.AllocTensor<half>();
        Cast(qsLocal, scale(), RoundMode::CAST_NONE, N_HEADS);
        outQueQs.EnQue(qsLocal);

        LocalTensor<half> wLocal = outQueW.AllocTensor<half>();
        Cast(w32(), wbLocal, RoundMode::CAST_NONE, N_HEADS);
        PipeBarrier<PIPE_V>();
        Muls(w32(), w32(), wScale, N_HEADS);
        PipeBarrier<PIPE_V>();
        Cast(wLocal, w32(), RoundMode::CAST_NONE, N_HEADS);
        outQueW.EnQue(wLocal);

        inQueQb.FreeTensor(qbLocal);
        inQueWb.FreeTensor(wbLocal);
    }

    __aicore__ inline void CopyOutToken(int32_t tok)
    {
        const int64_t t = static_cast<int64_t>(tok);
        LocalTensor<int8_t> q8Local = outQueQ8.DeQue<int8_t>();
        LocalTensor<half> qsLocal = outQueQs.DeQue<half>();
        LocalTensor<half> wLocal = outQueW.DeQue<half>();
        DataCopyExtParams cpQ8{1, static_cast<uint32_t>(TOKEN_ELEMS * sizeof(int8_t)), 0, 0, 0};
        DataCopyPad(q8Gm[t * TOKEN_ELEMS], q8Local, cpQ8);
        const int32_t auxEvent = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(auxEvent);
        WaitFlag<HardEvent::V_MTE3>(auxEvent);
        DataCopyExtParams cpS{1, static_cast<uint32_t>(N_HEADS * sizeof(half)), 0, 0, 0};
        DataCopyPad(qsGm[t * N_HEADS], qsLocal, cpS);
        DataCopyPad(wGm[t * N_HEADS], wLocal, cpS);
        PipeBarrier<PIPE_MTE3>();
        const int32_t wDoneEvent = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(wDoneEvent);
        WaitFlag<HardEvent::MTE3_V>(wDoneEvent);
        outQueQ8.FreeTensor(q8Local);
        outQueQs.FreeTensor(qsLocal);
        outQueW.FreeTensor(wLocal);
        PipeBarrier<PIPE_V>();
    }

private:
    QueueRef<TQue<TPosition::VECIN, 1>> inQueQb;
    QueueRef<TQue<TPosition::VECIN, 1>> inQueSin;
    QueueRef<TQue<TPosition::VECIN, 1>> inQueCos;
    QueueRef<TQue<TPosition::VECIN, 2>> inQueWb;
    QueueRef<TQue<TPosition::VECOUT, 1>> outQueQ8;
    QueueRef<TQue<TPosition::VECOUT, 1>> outQueQs;
    QueueRef<TQue<TPosition::VECOUT, 2>> outQueW;
    PostBuffer bufS;
    PostBuffer bufEO;
    PostBuffer bufRotBf;
    PostBuffer bufTailBf;
    PostBuffer bufTailF;
    PostBuffer bufHeadC;
    PostBuffer bufAbs;
    PostBuffer bufQf;
    PostBuffer bufScBC;
    PostBuffer bufQ16;
    PostBuffer bufQ16g;
    PostBuffer bufTrig;
    PostBuffer bufMisc;
    PostBuffer bufTables;

    GlobalTensor<bfloat16_t> qbGm;
    GlobalTensor<bfloat16_t> wbGm;
    GlobalTensor<float> sinGm;
    GlobalTensor<float> cosGm;
    GlobalTensor<int8_t> q8Gm;
    GlobalTensor<half> qsGm;
    GlobalTensor<half> wGm;

    int32_t totalTokens = 0;
    int32_t firstRow = 0;
    int32_t totalVec = 1;
    int32_t coreIdx = 0;
    float wScale = 0.0f;
};
