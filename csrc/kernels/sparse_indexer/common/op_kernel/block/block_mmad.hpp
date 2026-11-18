/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_BLOCK_MMAD_HPP
#define FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_BLOCK_MMAD_HPP

#include <type_traits>

#include "catlass/arch/resource.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "kernel_operator.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "key_loader.hpp"
#include "sparse_indexer_common.hpp"

namespace SparseIndexer::Common::Block {

template <class DtypeA_, class DtypeB_, class KeyLoader_ = ContiguousKeyLoader<Catlass::Arch::Ascend950,
                                                   Catlass::Gemm::GemmType<DtypeB_, Catlass::layout::ColumnMajor>>>
class BlockMmad {
public:
    using KeyLoaderParams = typename KeyLoader_::Params;

    struct Params {
        KeyLoaderParams keyLoader{};
    };

    using DtypeA = DtypeA_;
    using DtypeB = DtypeB_;
    using KeyLoader = KeyLoader_;
    using DtypeAccum = typename Catlass::Gemm::helper::ElementAccumulatorSelector<DtypeA, DtypeB>::ElementAccumulator;
    using DtypeC = DtypeAccum;

    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutA = Catlass::layout::RowMajor;
    using LayoutB = Catlass::layout::ColumnMajor;
    using LayoutC = Catlass::layout::RowMajor;

    using AType = Catlass::Gemm::GemmType<DtypeA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<DtypeB, LayoutB>;
    using L1AType = Catlass::Gemm::GemmType<DtypeA, Catlass::layout::zN, AscendC::TPosition::A1>;
    using L1BType = Catlass::Gemm::GemmType<DtypeB, Catlass::layout::nZ, AscendC::TPosition::A1>;

    using CopyGmToL1A = Catlass::Gemm::Tile::CopyGmToL1DynamicOptimized<ArchTag, AType>;
    using CopyL1ToL0A = Catlass::Gemm::Tile::CopyL1ToL0A<ArchTag, L1AType>;
    using CopyL1ToL0B = Catlass::Gemm::Tile::CopyL1ToL0B<ArchTag, L1BType>;
    using TileMmad = Catlass::Gemm::Tile::TileMmad<ArchTag, AType, BType, void>;

    using TensorL0C = tla::Tensor<
        AscendC::LocalTensor<DtypeAccum>, decltype(tla::MakeLayoutL0C(uint32_t{}, uint32_t{})),
        tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::CO1>;
    using TensorUbC = tla::Tensor<
        AscendC::LocalTensor<DtypeC>, decltype(tla::MakeLayout<DtypeC, LayoutC>(uint32_t{}, uint32_t{})),
        tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;
    using CopyL0CToUb = Catlass::Gemm::Tile::CopyL0CToUBTla<
        ArchTag, TensorL0C, TensorUbC, Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_N,
        Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, false>;

    CATLASS_DEVICE
    BlockMmad(
        AscendC::LocalTensor<DtypeC> const& ubC, AscendC::GlobalTensor<DtypeA> const& gmQ,
        AscendC::GlobalTensor<DtypeB> const& gmK, Catlass::GemmCoord const& problemShape,
        Catlass::GemmCoord const& tileShape, Catlass::Arch::Resource<ArchTag>& resource,
        Params const& params,
        AscendC::GlobalTensor<int32_t> const& gmTable,
        AscendC::GlobalTensor<DtypeB> const& gmZero)
        : problemShape_(problemShape), tileShape_(tileShape), resource_(resource), gmA_(gmQ), gmB_(gmK), ubC_(ubC)
    {
        layoutA_ = LayoutA::template MakeLayout<DtypeA>(problemShape_.m(), problemShape_.k());
        layoutB_ = LayoutB::template MakeLayout<DtypeB>(problemShape_.k(), problemShape_.n());
        keyLoader_.Setup(gmB_, layoutB_, params.keyLoader, gmTable, gmZero);

        uint32_t kByA = ArchTag::L0A_SIZE / sizeof(DtypeA) / tileShape_.m();
        uint32_t kByB = ArchTag::L0B_SIZE / sizeof(DtypeB) / tileShape_.n();
        kPart_ = AscendC::Std::min(kByA, kByB) / L1AAlign::K_ALIGNED * L1AAlign::K_ALIGNED;
        kPart_ = AscendC::Std::min(kPart_, tileShape_.k());

        uint32_t l1ABytes = tileShape_.m() * tileShape_.k() * sizeof(DtypeA);
        l1A_ = resource_.l1Buf.template GetBufferByByte<DtypeA>(0);
        l1B_ = resource_.l1Buf.template GetBufferByByte<DtypeB>(l1ABytes);
        l0A_ = resource_.l0ABuf.template GetBufferByByte<DtypeA>(0);
        l0B_ = resource_.l0BBuf.template GetBufferByByte<DtypeB>(0);
        l0C_ = resource_.l0CBuf.template GetBufferByByte<DtypeAccum>(0);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    CATLASS_DEVICE
    ~BlockMmad()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void operator()(uint32_t mStart, uint32_t actualM, KeyCoord const& keyCoord, uint32_t actualN)
    {
        uint32_t kTileCount = SparseIndexer::Common::CeilDiv(problemShape_.k(), kPart_);
        uint32_t alignedM = SparseIndexer::Common::AlignUp(actualM, L1AAlign::M_ALIGNED);
        uint32_t alignedN = SparseIndexer::Common::AlignUp(actualN, L1BAlign::N_ALIGNED);
        auto layoutAInL1 = Catlass::layout::zN::template MakeLayout<DtypeA>(alignedM, kPart_);
        auto layoutBInL1 = Catlass::layout::nZ::template MakeLayout<DtypeB>(kPart_, alignedN);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        for (uint32_t kIdx = 0; kIdx < kTileCount; ++kIdx) {
            uint32_t kStart = kIdx * kPart_;
            uint32_t actualK = AscendC::Std::min(kPart_, problemShape_.k() - kStart);
            int64_t offsetA = layoutA_.GetOffset(Catlass::MatrixCoord(mStart, kStart));

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            copyGmToL1A_(
                l1A_, gmA_[offsetA], layoutAInL1, layoutA_.GetTileLayout(Catlass::MatrixCoord(actualM, actualK)));
            keyLoader_.Load(l1B_, layoutBInL1, kStart, actualK, keyCoord, actualN);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            copyL1ToL0A_(l0A_, l1A_, layoutAInL1, layoutAInL1);
            copyL1ToL0B_(l0B_, l1B_, layoutBInL1, layoutBInL1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
            tileMmad_(l0C_, l0A_, l0B_, actualM, actualN, actualK, kIdx == 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        }

        constexpr uint32_t aivPerAic = 2;
        constexpr uint32_t l0cFractalN = 16;
        uint32_t splitNAlign = l0cFractalN * aivPerAic;
        uint32_t splitN = SparseIndexer::Common::AlignUp(actualN, splitNAlign);
        auto tensorL0C = tla::MakeTensor(l0C_, tla::MakeLayoutL0C(actualM, splitN), Catlass::Arch::PositionL0C{});
        auto tensorUbC = tla::MakeTensor(
            ubC_,
            tla::MakeLayout(
                tla::MakeShape(actualM, splitN),
                tla::MakeStride(static_cast<int64_t>(splitN / aivPerAic), tla::Int<1>{}),
                tla::MakeShape(actualM, splitN)),
            Catlass::Arch::PositionUB{});

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        copyL0CToUb_(tensorUbC, tensorL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

private:
    using L1AAlign = Catlass::Gemm::helper::L1AlignHelper<DtypeA, LayoutA>;
    using L1BAlign = Catlass::Gemm::helper::L1AlignHelper<DtypeB, LayoutB>;

    Catlass::GemmCoord problemShape_{};
    Catlass::GemmCoord tileShape_{};
    uint32_t kPart_{0};
    LayoutA layoutA_{};
    LayoutB layoutB_{};
    Catlass::Arch::Resource<ArchTag>& resource_;
    AscendC::GlobalTensor<DtypeA> const& gmA_;
    AscendC::GlobalTensor<DtypeB> const& gmB_;
    AscendC::LocalTensor<DtypeC> const& ubC_;
    AscendC::LocalTensor<DtypeA> l1A_;
    AscendC::LocalTensor<DtypeB> l1B_;
    AscendC::LocalTensor<DtypeA> l0A_;
    AscendC::LocalTensor<DtypeB> l0B_;
    AscendC::LocalTensor<DtypeAccum> l0C_;
    CopyGmToL1A copyGmToL1A_;
    KeyLoader keyLoader_;
    CopyL1ToL0A copyL1ToL0A_;
    CopyL1ToL0B copyL1ToL0B_;
    CopyL0CToUb copyL0CToUb_;
    TileMmad tileMmad_;
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_BLOCK_MMAD_HPP
