#pragma once

using namespace AscendC;

namespace {
template <HardEvent Event>
__aicore__ inline void Fence() {
    SetFlag<Event>(0);
    WaitFlag<Event>(0);
}

__aicore__ inline void Cube(GM_ADDR a, GM_ADDR b, GM_ADDR scratch,
                            const FiMatmulBiasSiluTilingData &t) {
    TPipe pipe;
    TBuf<TPosition::A1> a1buf, b1buf;
    TBuf<TPosition::A2> a2buf;
    TBuf<TPosition::B2> b2buf;
    TBuf<TPosition::CO1> cbuf;
    pipe.InitBuffer(a1buf, 512); pipe.InitBuffer(b1buf, 512);
    pipe.InitBuffer(a2buf, 512); pipe.InitBuffer(b2buf, 512);
    pipe.InitBuffer(cbuf, 1024);
    auto a1 = a1buf.Get<bfloat16_t>(); auto b1 = b1buf.Get<bfloat16_t>();
    auto a2 = a2buf.Get<bfloat16_t>(); auto b2 = b2buf.Get<bfloat16_t>();
    auto c = cbuf.Get<float>();
    GlobalTensor<bfloat16_t> ga, gb;
    GlobalTensor<float> gc;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(a));
    gb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(b));
    gc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch));
    Nd2NzParams copy{};
    copy.ndNum = 1; copy.nValue = 16; copy.dValue = 16;
    copy.dstNzC0Stride = 16; copy.dstNzNStride = 1;
    LoadData2DParamsV2 load{};
    load.mStep = 1; load.kStep = 1; load.srcStride = 1; load.dstStride = 1;
    for (uint32_t m = 0; m < t.m; m += 16) {
        for (uint32_t n = 0; n < t.n; n += 16) {
            for (uint32_t k = 0; k < t.k; k += 16) {
                copy.srcDValue = t.k;
                DataCopy(a1, ga[m * t.k + k], copy);
                copy.srcDValue = t.n;
                DataCopy(b1, gb[k * t.n + n], copy);
                Fence<HardEvent::MTE2_MTE1>();
                load.ifTranspose = false; LoadData(a2, a1, load);
                load.ifTranspose = true; LoadData(b2, b1, load);
                Fence<HardEvent::MTE1_M>();
                Fence<HardEvent::MTE1_MTE2>();
                MmadParams mm{};
                mm.m = 16; mm.n = 16; mm.k = 16;
                mm.cmatrixInitVal = (k == 0); mm.disableGemv = true;
                Mmad(c, a2, b2, mm);
                Fence<HardEvent::M_MTE1>();
            }
            Fence<HardEvent::M_FIX>();
            FixpipeParamsC310 fix{};
            fix.mSize = 16; fix.nSize = 16;
            fix.srcStride = 16; fix.dstStride = t.n;
            Fixpipe<float, float, CFG_ROW_MAJOR>(gc[m * t.n + n], c, fix);
            Fence<HardEvent::FIX_M>();
            // Finish the GM store before reusing the single L0C tile. This
            // conservative barrier is required by the validated 3510 path.
            PipeBarrier<PIPE_ALL>();
        }
    }
    // Mode 2 broadcasts the completed FIX writes to both vector subcores.
    CrossCoreSetFlag<2, PIPE_FIX>(0);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Vector(GM_ADDR bias, GM_ADDR out, GM_ADDR scratch,
                              const FiMatmulBiasSiluTilingData &t) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> xb, yb, bb, bf, ob;
    pipe.InitBuffer(xb, 256 * 4); pipe.InitBuffer(yb, 256 * 4);
    pipe.InitBuffer(bb, 256 * 4); pipe.InitBuffer(bf, 256 * 2);
    pipe.InitBuffer(ob, 256 * 2);
    auto x = xb.Get<float>(); auto y = yb.Get<float>(); auto b = bb.Get<float>();
    auto raw = bf.Get<bfloat16_t>(); auto result = ob.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> gbias, gout;
    GlobalTensor<float> gin;
    gbias.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bias));
    gout.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(out));
    gin.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch));
    DataCopy(raw, gbias, t.n);
    Fence<HardEvent::MTE2_V>();
    Cast(b, raw, RoundMode::CAST_NONE, t.n);
    PipeBarrier<PIPE_V>();
    CrossCoreWaitFlag(0);
    for (uint32_t m = GetSubBlockIdx(); m < t.m; m += GetSubBlockNum()) {
        DataCopy(x, gin[m * t.n], t.n);
        Fence<HardEvent::MTE2_V>();
        Add(x, x, b, t.n); PipeBarrier<PIPE_V>();
        Muls(y, x, -1.0f, t.n); PipeBarrier<PIPE_V>();
        Exp(y, y, t.n); PipeBarrier<PIPE_V>();
        Adds(y, y, 1.0f, t.n); PipeBarrier<PIPE_V>();
        Div(x, x, y, t.n); PipeBarrier<PIPE_V>();
        Cast(result, x, RoundMode::CAST_RINT, t.n);
        Fence<HardEvent::V_MTE3>();
        DataCopy(gout[m * t.n], result, t.n);
        Fence<HardEvent::MTE3_MTE2>();
    }
    PipeBarrier<PIPE_ALL>();
}
}
