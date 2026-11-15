#pragma once

#include "catlass/gemm/kernel/basic_matmul_tla.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "flashinfer_ascend/probe_tiling.h"

using namespace Catlass;
using namespace tla;
using Policy = Gemm::MmadPingpong<Arch::AtlasA2, true, false>;
using L1Shape = Shape<Int<128>, Int<128>, Int<128>>;
using L0Shape = Shape<Int<128>, Int<128>, Int<32>>;
using Tag = layout::RowMajor;
using Copy = Gemm::Tile::PackedTileCopyTla<Arch::AtlasA2, float, Tag, float, Tag, float, Tag, void>;
using Mmad = Gemm::Block::BlockMmadTla<Policy, L1Shape, L0Shape, float, float, float, void, Copy>;
using Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
using Kernel = Gemm::Kernel::BasicMatmulTla<Mmad, void, Scheduler>;
