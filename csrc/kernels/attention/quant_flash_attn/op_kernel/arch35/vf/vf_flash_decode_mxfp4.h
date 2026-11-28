/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QFA_VF_FLASH_DECODE_MXFP4_H_
#define QFA_VF_FLASH_DECODE_MXFP4_H_

#include "vf_softmax_fp32_mxfp4.h"
#include "../quant_flash_attn_common_def.h"

namespace Mxfp4Api {
using BaseApi::QuantSchema;

// Every row is repeated eight times, matching the existing FD workspace ABI.
// C4 stores integer references R and raw PV. Normalize the merge multipliers
// by the total row sum so ReduceFinalRes produces the final output directly.
template <QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT>
__simd_vf__ inline void PrepareFlashDecodeVf(__ubuf__ float* splitReference,
    __ubuf__ float* splitSum, __ubuf__ float* splitMultiplier, __ubuf__ float* finalLse,
    __ubuf__ float* finalReference, uint32_t rowCount, uint32_t splitCount)
{
    uint32_t validElements = rowCount * 8;
    MaskReg rows = UpdateMask<float>(validElements);
    RegTensor<float> maximum, reference, sum, total, zero, one, invalid;
    Duplicate(zero, 0.0F);
    Duplicate(one, 1.0F);
    Duplicate(invalid, INVALID_GROUP_EXPONENT);
    Duplicate(maximum, INVALID_GROUP_EXPONENT);
    for (uint16_t split = 0; split < splitCount; ++split) {
        LoadAlign(reference, splitReference + split * rowCount * 8);
        Max(maximum, maximum, reference, rows);
    }
    Duplicate(total, 0.0F);
    for (uint16_t split = 0; split < splitCount; ++split) {
        RegTensor<float> difference, factor, fallback, biasedExponent;
        RegTensor<int32_t> exponentBits;
        MaskReg nonempty, normalPower;
        LoadAlign(reference, splitReference + split * rowCount * 8);
        LoadAlign(sum, splitSum + split * rowCount * 8);
        Compares<float, CMPMODE::GT>(nonempty, sum, 0.0F, rows);
        Sub(difference, reference, maximum, rows);
        Select(difference, difference, zero, nonempty);
        // Normal powers of two are represented exactly. Retain the existing
        // exponential behavior for the subnormal range instead of clamping it.
        Compares<float, CMPMODE::GE>(normalPower, difference, -126.0F, rows);
        Adds(biasedExponent, difference, 127.0F, rows);
        Cast<int32_t, float, MXFP4_CAST_ZERO>(exponentBits, biasedExponent, rows);
        ShiftLefts(exponentBits, exponentBits, static_cast<int16_t>(23), rows);
        Muls(fallback, difference, MXFP4_LN2, rows);
        Exp(fallback, fallback, rows);
        Select(factor, reinterpret_cast<RegTensor<float>&>(exponentBits), fallback, normalPower);
        Select(factor, factor, zero, nonempty);
        StoreAlign(splitMultiplier + split * rowCount * 8, factor, rows);
        MulAddDst(total, sum, factor, rows);
    }
    MaskReg nonemptyRows;
    RegTensor<float> safeSum, lse, naturalReference;
    RegTensor<uint32_t> infinityBits;
    Compares<float, CMPMODE::GT>(nonemptyRows, total, 0.0F, rows);
    Select(safeSum, total, one, nonemptyRows);
    Select(naturalReference, maximum, zero, nonemptyRows);
    if constexpr (QUANT_SCHEMA != QuantSchema::OCP_QUANT) {
        // Split references remain binary R; only the final LSE restores this shift.
        Adds(naturalReference, naturalReference, MXFP4_CX_SCALE_SHIFT, rows);
    }
    Muls(naturalReference, naturalReference, MXFP4_LN2, rows);
    if constexpr (QUANT_SCHEMA == QuantSchema::CX_DELTA_QUANT) {
        Adds(naturalReference, naturalReference, MXFP4_NEG_LN_DELTA, rows);
    }
    Ln(lse, safeSum, rows);
    Add(lse, lse, naturalReference, rows);
    Duplicate(infinityBits, static_cast<uint32_t>(0x7f800000));
    Select(lse, lse, reinterpret_cast<RegTensor<float>&>(infinityBits), nonemptyRows);
    Select(naturalReference, naturalReference, invalid, nonemptyRows);
    StoreAlign(finalLse, lse, rows);
    StoreAlign(finalReference, naturalReference, rows);

    // Reuse the stored a_i only after all stores complete. LSE above uses the
    // original S; raw PV below is merged with a_i / S (without a split sum).
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    for (uint16_t split = 0; split < splitCount; ++split) {
        RegTensor<float> weight;
        LoadAlign(weight, splitMultiplier + split * rowCount * 8);
        Div<float, &MXFP4_OUTPUT_DIV>(weight, weight, safeSum, rows);
        Select(weight, weight, zero, nonemptyRows);
        StoreAlign(splitMultiplier + split * rowCount * 8, weight, rows);
    }
}

} // namespace Mxfp4Api
#endif
