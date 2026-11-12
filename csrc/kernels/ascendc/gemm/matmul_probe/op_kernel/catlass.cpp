// Instantiate the public CATLASS BasicMatmulTla API with the same probe ABI.
// See CATLASS example 43_ascend950_basic_matmul for the template contract.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
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
using Policy = Gemm::MmadPingpong<Arch::Ascend950, true, false, 1, false, 2, 2, 2, 2>;
using L1Shape = Shape<Int<256>, Int<256>, Int<128>>;
using L0Shape = Shape<Int<256>, Int<256>, Int<32>>;
using Tag = layout::RowMajor;
using Copy = Gemm::Tile::PackedTileCopyTla<Arch::Ascend950, float, Tag, float, Tag, float, Tag, void>;
using Mmad = Gemm::Block::BlockMmadTla<Policy, L1Shape, L0Shape, float, float, float, void, Copy>;
using Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
using Kernel = Gemm::Kernel::BasicMatmulTla<Mmad, void, Scheduler>;

extern "C" __global__ __aicore__ void FI_DEVICE_ENTRY(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    FI_PROBE_CHECK_TILING();
    auto la = tla::MakeLayout<float, Tag>(16u, 16u);
    auto lb = tla::MakeLayout<float, Tag>(16u, 16u);
    auto lc = tla::MakeLayout<float, Tag>(16u, 16u);
    Kernel::Params params{GemmCoord{16, 16, 16}, a, la, b, lb, c, lc};
    Kernel kernel;
    kernel(params);
}
