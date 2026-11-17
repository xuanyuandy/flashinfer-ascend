#ifndef FI_OP_PACKAGE
#include "flashinfer_ascend/launch_abi.h"

extern "C" void FI_HOST_ENTRY(void *stream, unsigned char *a, unsigned char *b, unsigned char *c) {
    FI_DEVICE_ENTRY<<<FI_BLOCK_DIM, nullptr, stream>>>(a, b, c, nullptr, nullptr);
}
#endif
