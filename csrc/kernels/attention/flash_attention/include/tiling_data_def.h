#pragma once
#if CATLASS_ARCH == 2201
#include "arch22/tiling_data_def.h"
#elif CATLASS_ARCH == 3510
#include "arch35/tiling_data_def.h"
#else
#error "FlashAttention requires an explicit supported CATLASS architecture"
#endif
