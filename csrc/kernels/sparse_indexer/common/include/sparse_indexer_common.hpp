/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_COMMON_HPP
#define FLASHINFER_SPARSE_INDEXER_COMMON_HPP

#include <cstdint>

#include "catlass/detail/macros.hpp"

namespace SparseIndexer::Common {

inline constexpr uint32_t kMaxBatch = 16;

template <class T, class U>
CATLASS_HOST_DEVICE constexpr uint32_t AlignUp(const T& value, const U& alignment)
{
    return static_cast<uint32_t>((value + alignment - 1) / alignment * alignment);
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr uint32_t CeilDiv(const T& value, const U& divisor)
{
    return divisor == 0 ? 0 : static_cast<uint32_t>((value + divisor - 1) / divisor);
}

} // namespace SparseIndexer::Common

#endif // FLASHINFER_SPARSE_INDEXER_COMMON_HPP
