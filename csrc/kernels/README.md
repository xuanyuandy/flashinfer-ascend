# Ascend C kernel sources

`flashinfer/data/kernels/manifest.schema.json` stores the existing kernel manifest JSON schema.
The packaged registry is `flashinfer/data/kernels/registry.json` and is
populated by the production Ascend C declarations. A kernel must not be
registered before its artifact and capability constraints are real and tested.

All Ascend C operator sources live under `csrc/kernels/ascendc/`, grouped by operator:

- `matmul_probe/`: bounded v2 native and CATLASS ACLNN probes. Frozen v1/raw
  launch fixtures live in `tests/fixtures/legacy_ascendc/`, outside build inputs.
- `matmul_bias_silu/`: production native implementation.
- `flash_attention/`: production CATLASS implementation.

Each operator owns `op_host/` (OpDef, inference and tiling), `op_kernel/`
(device code) and `include/` (shared tiling layouts). Build recipes remain in
`csrc/recipes/`; shared CANN build templates are in `csrc/cmake/op_package/`.
An op-package is the compiled vendor tree (host API/tiling libraries, NPU
binaries and metadata), produced by both AOT and JIT. Wheel build inputs
retain this layout under `flashinfer/data/build/csrc/`; sdist retains the
repository's `csrc/` tree.
Triton Python implementations live in `flashinfer/triton/`.
