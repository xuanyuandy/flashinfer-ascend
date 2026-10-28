// Fixed FP32 16x16x16 probe ABI v1. Pointers are device addresses; stream is
// caller-owned. No workspace or tiling buffer is consumed by this probe.
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void FI_HOST_ENTRY(void *stream, unsigned char *a, unsigned char *b, unsigned char *c);
#ifdef __cplusplus
}
#endif
