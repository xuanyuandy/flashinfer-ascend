#pragma once

using namespace AscendC;

namespace {
template <HardEvent Event>
__aicore__ inline void Fence() { SetFlag<Event>(0); WaitFlag<Event>(0); }

__aicore__ inline void Matmul(GM_ADDR a, GM_ADDR b, GM_ADDR out, const FiMatmulTilingData &t) {
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
    GlobalTensor<bfloat16_t> ga, gb, gout;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(a));
    gb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(b));
    gout.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(out));
    Nd2NzParams copy{};
    copy.ndNum = 1; copy.nValue = 16; copy.dValue = 16;
    copy.dstNzC0Stride = 16; copy.dstNzNStride = 1;
    LoadData2DParamsV2 load{};
    load.mStep = 1; load.kStep = 1; load.srcStride = 1; load.dstStride = 1;
    const uint32_t columns = t.n / 16;
    for (uint32_t tile = GetBlockIdx(); tile < (t.m / 16) * columns; tile += GetBlockNum()) {
        uint32_t m = (tile / columns) * 16, n = (tile % columns) * 16;
        for (uint32_t k = 0; k < t.k; k += 16) {
            copy.srcDValue = t.k; DataCopy(a1, ga[m * t.k + k], copy);
            copy.srcDValue = t.n; DataCopy(b1, gb[k * t.n + n], copy);
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
        fix.quantPre = QuantMode_t::F322BF16;
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(gout[m * t.n + n], c, fix);
        Fence<HardEvent::FIX_M>();
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}
}
