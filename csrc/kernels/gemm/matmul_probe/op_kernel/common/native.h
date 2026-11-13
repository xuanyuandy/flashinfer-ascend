#pragma once

// Architecture-independent FP32 correctness probe; shared by the declared targets.
__aicore__ inline void ProbeMatmul(GM_ADDR a, GM_ADDR b, GM_ADDR c) {
    auto pa = reinterpret_cast<__gm__ float *>(a);
    auto pb = reinterpret_cast<__gm__ float *>(b);
    auto pc = reinterpret_cast<__gm__ float *>(c);
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 16; ++n) {
            float value = 0;
            for (int k = 0; k < 16; ++k) value += pa[m * 16 + k] * pb[k * 16 + n];
            pc[m * 16 + n] = value;
        }
    }
}
