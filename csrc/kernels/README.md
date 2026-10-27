# Ascend C kernel sources

`flashinfer/data/kernels/manifest.schema.json` stores the existing kernel manifest JSON schema.
The packaged registry is `flashinfer/data/kernels/registry.json` and is
populated by the production Ascend C declarations. A kernel must not be
registered before its artifact and capability constraints are real and tested.

All Ascend C device sources live under `csrc/kernels/ascendc/`, grouped by operator:

- `matmul_probe/`: bounded v1 native and CATLASS build/ABI probes.
- `matmul_bias_silu/`: production native implementation.
- `flash_attention/`: production CATLASS implementation.

Build recipes live in `csrc/recipes/`, host bridges in `csrc/op_package/`, and
shared launch contracts in `csrc/include/flashinfer_ascend/`. Wheel build inputs
retain this layout under `flashinfer/data/build/csrc/`; sdist retains the
repository's `csrc/` tree.
Triton Python implementations live in `flashinfer/triton/`.
