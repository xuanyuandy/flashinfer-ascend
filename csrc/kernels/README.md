# Ascend C kernel sources

`tools/metadata/kernels/manifest.schema.json` stores the existing kernel manifest JSON schema.
The packaged registry is `flashinfer/data/kernels/registry.json` and is
populated by the production Ascend C declarations. Capability claims require real evidence. Registration is independent of optional
binaries; execution requires a matching frozen contract and verified artifact.

All Ascend C operator sources live under `csrc/kernels/<domain>/<op>/`:

- `gemm/matmul_probe/`: bounded v2 native and CATLASS ACLNN probes. Frozen v1/raw
  launch fixtures live in `tests/fixtures/legacy_ascendc/`, outside build inputs.
- `gemm/matmul/` and `gemm/matmul_bias_silu/`: production native implementations.
- `gemm/grouped_matmul_swiglu_quant/`: grouped integer GEMM, SwiGLU and quantization.
- `attention/flash_attention/`: production CATLASS implementation.

Each operator owns `op_host/` (OpDef, inference and tiling), `op_kernel/`
(device code) and `include/` (shared tiling layouts). Production recipes are generated under
`build/generated/csrc/recipes/`; `csrc/recipes/` retains explicit probes. Shared CANN build templates are in `csrc/cmake/op_package/`.
An op-package is the compiled vendor tree (host API/tiling libraries, NPU
binaries and metadata), produced by both AOT and JIT. Wheel build inputs
retain this layout under `flashinfer/data/build/csrc/`; sdist retains the
repository's `csrc/` tree.
Triton Python implementations live in `flashinfer/triton/`.

Logical definitions and backend descriptor/runner declarations follow
[the operator layout](../../docs/operator-layout.md); generated recipes and
packaged indexes are derived from those declarations.
