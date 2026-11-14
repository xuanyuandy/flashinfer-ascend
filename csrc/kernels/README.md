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
- `attention/sparse_flash_attention/`: arch22 MLA sparse attention, with an Ascend910B AOT-only declaration.

Each operator owns `op_host/` (OpDef, inference and tiling), `op_kernel/`
(device code) and `include/` (shared tiling layouts). Production recipes are generated under
`build/generated/csrc/recipes/`; `csrc/recipes/` retains explicit probes. Shared CANN build templates are in
`csrc/cmake/operator/` for authored operators and `csrc/cmake/op_package/` for explicit probes.
An op-package is the compiled vendor tree (host API/tiling libraries, NPU
binaries and metadata), produced by both AOT and JIT. Wheel build inputs
retain this layout under `flashinfer/data/build/csrc/`; sdist retains the
repository's `csrc/` tree.
Triton Python implementations live in `flashinfer/triton/`.

Device entry `.cpp` files stay directly under `op_kernel/`; implementation headers
live in architecture directories, following vLLM-Ascend's per-operator layout:

```text
<domain>/<op>/
├── op_kernel/
│   ├── <entry>.cpp          # stable native entry; selects __CCE_AICORE__
│   ├── arch22/              # A2/A3 implementation, where provided
│   ├── arch35/              # A5 implementation, where provided
│   └── common/              # architecture-independent code, where applicable
├── op_host/
│   ├── <op>.cpp             # shared OpDef, inference and tiling entry
│   └── arch22/ or arch35/   # platform-specific tiling implementation
└── include/                 # shared host/device wire layouts
```

GMM and SparseFlashAttention implementations and tilers currently live in `arch22/`. Matmul,
MatmulBiasSilu and CATLASS FlashAttention implementations live in `arch35/`;
their entries reject other device architectures at compile time. The native
FP32 Matmul probe uses `common/`; its CATLASS variant uses `arch35/`.
Directories describe actual implementations, and do not expand their declared
SoC support. Host ABI and Python runners remain shared. Source moves require
recipe regeneration and spec refreezing before compiling or loading artifacts.

Logical definitions and backend descriptor/runner declarations follow
[the operator layout](../../docs/operator-layout.md); generated recipes and
packaged indexes are derived from those declarations.
