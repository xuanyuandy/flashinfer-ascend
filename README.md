# FlashInfer Ascend

FlashInfer Ascend is an inference operator library and kernel integration
framework for Ascend NPU. It follows FlashInfer's API domains, direct functions
and stateful plan/run interfaces, with CANN, Ascend C and Triton implementations.
The project is under active development; supported parameters and execution
modes are defined per operator.

[Features](#core-features) · [Operators](#supported-operators) ·
[Getting started](#getting-started) · [Testing](#testing) ·
[Documentation](#documentation)

For building an AOT wheel and testing it in a fresh conda environment, see
[`README-wheel-conda.md`](README-wheel-conda.md).
Ordinary wheel builds compile and bundle the registered production Ascend C operators.

For the current operator/interface inventory, upstream call patterns and Ascend
implementation gaps, see [operator coverage](docs/operator-coverage.md) and
[public API lifecycle tests](docs/operator-call-patterns.md). Not every operator
requires a wrapper or a public `plan()` step.

## Core features

- **FlashInfer-style Tensor APIs**: direct functions for single-request attention,
  GEMM, normalization, RoPE and sampling; plan/run wrappers for paged/ragged batch
  attention; a run-only `SegmentGEMMWrapper`.
- **Backend selection**: input and hardware signatures, a kernel registry,
  capability filtering, explicit `backend`/`kernel_id` selection and explainable
  dispatch. Valid measured tuning records can guide selection; otherwise the
  dispatcher uses a deterministic policy, without benchmarking during a call.
- **Ascend C AOT and controlled JIT**: ordinary wheel builds compile and bundle
  the registered production operators. Verified AOT hits require no compiler;
  missing artifacts may trigger JIT when policy permits. Corrupt artifacts fail
  explicitly.
- **TorchNPU / ACLNN integration**: opt-in Torch functional/out schemas,
  FakeTensor metadata and a C++ `OpCommand` bridge for production Ascend C
  operators. The existing Python/ACLNN execution path remains the default.
- **Extensible dtype contracts**: v2 recipes can declare exact ordered
  `dtype_signatures`. FA exercises FP16/BF16 compilation, package verification,
  dispatch and installed-wheel execution; undeclared mixed combinations fail.
- **Developer tooling**: artifact integrity checks, installed-wheel selftest,
  regression profiles with logs/JUnit/exit codes, and operator/interface inventories.

## Supported operators

The current registry contains **45 logical ops and 81 implementation descriptors**.
Of these, **43 ops bind at least one NPU runner**; two internal Attention ops are
reference-only. These are registration counts: dtype, shape, layout, SoC and
graph support remain bounded by each implementation's contract and evidence.

| Family | Representative public APIs | Current scope |
| --- | --- | --- |
| Attention | `single_prefill_with_kv_cache`, `single_decode_with_kv_cache`, paged/ragged batch wrappers, `single_prefill_with_kv_cache_catlass` | CANN single/batch routes and bounded CATLASS FA; ordinary batch routes compose gather/FIA/concatenation |
| GEMM and linear operations | `mm_bf16`, `bmm_bf16`, `bmm_fp8`, `gemm_fp8_nt_groupwise`, `SegmentGEMMWrapper`, `matmul_bias_silu` | CANN BF16/FP8 routes, run-time segment metadata, Ascend C fused MatmulBiasSiLU |
| Mixture of Experts | `fused_moe`, `fp8_per_tensor_scale_moe`, `fp8_block_scale_moe` | CANN-based routing/expert computation with interface-specific constraints; distributed EP numerics unverified |
| Normalization | `rmsnorm`, `fused_add_rmsnorm`, `gemma_rmsnorm`, `gemma_fused_add_rmsnorm`, `layernorm` | CANN routes and a Triton implementation of Gemma fused Add RMSNorm |
| Activations | `silu_and_mul`, `gelu_and_mul`, `gelu_tanh_and_mul` | Gated activation routes |
| RoPE | `apply_rope`, `apply_rope_pos_ids`, `apply_rope_with_cos_sin_cache` and in-place forms | Position/frequency/cache and mutation contracts are checked separately |
| Sampling and decoding | sampling from logits/probabilities, top-k/top-p/min-p, renormalization, `chain_speculative_sampling` | Device filtering/sampling compositions; no cross-platform RNG equivalence claim |
| KV cache | `append_paged_kv_cache` | Ordinary paged KV append |
| Attention state merge | `merge_state`, `merge_states` | Output/LSE merging; multi-level cascade wrappers are a separate gap |

See the [complete local op list](docs/generated/operator-coverage/local-operators.md)
and [upstream interface comparison](docs/operator-coverage.md) for per-interface
details. Compatibility names such as `cudnn_batch_decode_with_kv_cache` do not
require a CUDA backend on Ascend.

### Precompiled Ascend C operators

These two production operators are compiled into a normal AOT wheel; CANN
operators use the installed TorchNPU/CANN stack, and Triton has its own JIT route.

| Operator | Dtype and supported inputs |
| --- | --- |
| CATLASS FlashAttention | FP16 or BF16, Q/K/V/out sharing one dtype; contiguous NHD/ND, batch=1, Q `[S,H,128]`, K/V `[T,H,128]`; S/T are multiples of 128 in 128..512, H=1..8; noncausal, unmasked |
| MatmulBiasSiLU | BF16, contiguous ND; A `[M,K]`, B `[K,N]`, bias `[N]`; M/N/K are multiples of 16 in 16..256; computes `silu(A @ B + bias)` |

Both currently expose synchronous eager execution and are `NOT_CAPTURABLE`.

## Getting started

### Environment

The latest targeted hardware and wheel checks used **Ascend950**, Python 3.12,
Torch `2.10.0+cpu`, TorchNPU `2.10.0.post7` and CANN 9.2. Other version combinations
need their own verification. AOT probes have been cross-compiled for other SoCs;
that does not establish numerical support on those devices.

For a source build, prepare matching Torch/TorchNPU libraries and headers, CANN
with `bisheng` and its packaging tools, CMake, a C++17 compiler, and libffi
development headers/libraries. Activate that environment and source the local
CANN `set_env.sh` before building. CATLASS builds use the pinned checkout declared
in [the FA recipe](csrc/recipes/flash_attention.json).

### Build the wheel

Run from the repository root in the prepared build environment:

```bash
export FLASHINFER_CATLASS_ROOT=../catlass-ref
python3 -B -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
python3 -B setup.py bdist_wheel
```

The default build includes both production AOT packages, their source inputs and
the native TorchNPU adapter. CATLASS headers are needed at build time and are not
bundled as a dependency checkout. `FLASHINFER_AOT_ALL=0` produces a source-only
wheel; `FLASHINFER_AOT_PREBUILT` imports verified existing packages.
The [wheel and conda guide](README-wheel-conda.md) covers environment preparation,
alternative build modes and toolchain/spec refreshes when changing build inputs.

### Install and verify

Activate a **clean runtime environment** with the matching Torch/TorchNPU and CANN
runtime, then install the wheel produced above. The following example uses the
current Linux x86-64 wheel name and starts from the repository root:

```bash
python3 -m pip install --no-deps --force-reinstall dist/flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl
cd tests
python3 -B flash_attention_acceptance.py
python3 -B matmul_bias_silu_acceptance.py
python3 -B -m flashinfer selftest
```

The FA script tests FP16 and BF16 through the native bridge. The standalone
selftest uses the installed package's bundled AOT operators with JIT disabled.
The scripts check that `flashinfer` comes from the runtime environment;
[editable-install isolation](README-wheel-conda.md) matters when validating a wheel.

### Basic usage

After installing a complete AOT wheel:

```python
import torch
import torch_npu
import flashinfer
from flashinfer.aclnn import enable_aclnn

enable_aclnn()  # Register Torch schemas and select the native TorchNPU bridge.

for dtype in (torch.float16, torch.bfloat16):
    q = (torch.rand(128, 2, 128) * 2 - 1).to(dtype).npu()
    k = (torch.rand(256, 2, 128) * 2 - 1).to(dtype).npu()
    v = (torch.rand(256, 2, 128) * 2 - 1).to(dtype).npu()
    output = torch.ops.flashinfer.single_prefill_with_kv_cache_catlass(q, k, v)
    supplied_out = torch.empty_like(output)
    flashinfer.single_prefill_with_kv_cache_catlass(q, k, v, out=supplied_out)
    print(output.shape, output.dtype)
```

The Python public function also accepts Torch tensors without `enable_aclnn()`;
it then uses the default Python/ACLNN adapter. Details and functional/out schemas
are in the [Torch/ACLNN guide](docs/aclnn-torch-adapter.md).

### Calling styles

| Style | Use it for |
| --- | --- |
| Direct function | Single-request attention, normalization, RoPE, activations, sampling and ordinary GEMM |
| Construct → `plan()` → `run()` | `BatchDecodeWithPagedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper`, `BatchPrefillWithRaggedKVCacheWrapper` |
| Construct → `run()` | `SegmentGEMMWrapper`; metadata is supplied to `run()`, without a public plan step |

A batch wrapper's plan retains the metadata and configuration needed for later
runs. New tensor contents may reuse a compatible plan; request structure or
configuration changes must satisfy that wrapper's replan/update contract.
Examples and public UT requirements are in [operator calling patterns](docs/operator-call-patterns.md).

## Current boundaries

- The holistic `BatchAttention` facade has Host/reference lifecycle coverage;
  its default production NPU provider is not connected. It is separate from the
  ordinary executable CANN batch wrappers.
- Graph capture/replay evidence covers one bounded single-decode configuration.
  Production Ascend C FA/MatmulBiasSiLU and ordinary batch capture are unverified;
  class names containing `CUDAGraph` do not extend that evidence.
- The upstream MLA, sparse/POD attention, complete shared-prefix/cascade wrappers,
  FP4/MX quantization, communication and distributed operator families are not
  fully implemented here. The [gap inventory](docs/operator-coverage.md) distinguishes
  missing interfaces, Host contracts and available backend components.
- Kernel availability remains specific to dtype, shape, layout, hardware and
  installed dependencies. Reference execution requires explicit selection;
  unsupported input and missing required dependencies fail explicitly.

## Testing

Choose a scope for the change being made:

```bash
python3 -B tools/run_regression.py --profile quick
python3 -B tools/run_regression.py --profile compile tests/test_flash_attention_real.py
python3 -B tools/run_regression.py --profile full
```

`quick` excludes `real_compile` tests but may use an NPU. For compilation changes,
run the affected compile tests. `full` already includes them, so a release run
does not need the same compile matrix repeated afterwards. The runner records
full logs, JUnit and the original pytest exit code, and treats skips or an empty
run as failure. Published verification scopes are in
[the architecture report](docs/reports/architecture-refactor-2026-10-28.md) and
[the FA dtype extension report](docs/reports/fa-dual-dtype-2026-10-29.md);
their results apply to their recorded baselines.

## Documentation

| Topic | Entry point |
| --- | --- |
| Build, install and call an operator | [Wheel/conda quick start](README-wheel-conda.md) |
| Runnable CANN/Triton examples | [Ascend950 guide](docs/a5.md), [smoke script](examples/a5_smoke.py) |
| Framework architecture and implementation boundaries | [Framework design](docs/framework_design.md) |
| FA call chain and multi-backend integration | [Facade, dispatch, AOT/JIT and ACLNN](docs/flash-attention-call-chain.md) |
| Operator inventory and upstream comparison | [Coverage overview](docs/operator-coverage.md), [generated op list](docs/generated/operator-coverage/local-operators.md) |
| Add a kernel or dtype | [Operator onboarding](docs/adding-a-custom-operator.md), [source layout](csrc/kernels/README.md) |
| TorchNPU queue and ACLNN integration | [Torch/ACLNN adapter](docs/aclnn-torch-adapter.md) |
| Public wrapper lifecycles | [Calling patterns and UT](docs/operator-call-patterns.md) |
| AOT loading and JIT policy | [Artifact discovery](docs/aot-runtime-discovery.md) |
| Detailed Attention contracts | [Documentation index](docs/README.md), [Host contracts below](#attention-framework-contracts) |

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [operator onboarding guide](docs/adding-a-custom-operator.md).
The README organization follows flashinfer-ori: `README.md`; all capability
descriptions above refer to the Ascend implementation. The project uses
[Apache-2.0](LICENSE). The migrated CATLASS FA implementation retains its
[source license](csrc/kernels/ascendc/flash_attention/LICENSE) and notices.

## Attention framework contracts

<details>
<summary>Reference environment, Host framework contracts and detailed developer documentation</summary>

### Reference development environment

For runnable A5 (Ascend950) CANN/Triton operators, environment setup and a map
of the current code, see [`docs/a5.md`](docs/a5.md). From an initialized CANN
shell with CATLASS configured, the reference conda recipe (Python 3.11, editable
development install) can be used as follows. The latest hardware checks above
used a separate Python 3.12 environment:

```bash
conda env create -f environment-a5.yml
conda activate flashinfer-a5
python examples/a5_smoke.py
```

The FlashInfer-compatible Attention framework track covers single/batch
prefill and decode, paged/ragged KV metadata, mixed `BatchAttention`, plan/run
semantics, and a zero-dependency
scalar reference oracle for small conformance cases. Its provider and artifact
contracts are separate from the runnable operator backends described above.

The stateful Attention lifecycle matches FlashInfer: users call `plan()` once and
then call `run()` without passing a plan or provider handle. Provider selection,
page-table materialization, operation-version binding and callable authorization
remain internal to the wrapper:

```python
from flashinfer.attention import BatchAttention

attention = BatchAttention(kv_layout="HND", device="npu:0")
attention.plan(
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_len_arr,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    page_size,
    causal=True,
)
output, lse = attention.run(q, (k_cache, v_cache))
```

After `plan()`, `attention.plan_selection` provides optional read-only
diagnostics about the chosen mode, provider operation, backend and registry
generation. It never contains an executable plan, module, callable or provider
handle, and callers never pass it to `run()`.

This example documents the target interface. Internally, a declarative
bootstrap composes exact package versions, capability evidence, kernel/ABI
provenance, provider adapters and tensor materializers into the automatic NPU
resolver. The packaged declaration set is intentionally empty today: no real
CANN or flash-attention-npu package is imported or called until a separately
verified integration is installed.

Provider integrations are installed atomically at process bootstrap. Each NPU
`BatchAttention` instance captures one immutable resolver generation when it is
constructed, so a later integration update cannot change an already-created
wrapper's plan/run authority. This bootstrap control remains separate from the
model-facing API.

Quantized provider candidates have an additional admission boundary: every
capability `QuantSpec` must map key/value scale and, when applicable,
independent zero-point sources to exact quantization arguments in the selected
catalog operation. Merely exposing quantization-named package parameters is
not treated as runtime support.

The unchanged paged/mixed `kv_cache` run slot can carry an
`AttentionOperatorQuantizedKVInput` for a future provider integration. Single
prefill/decode and ragged prefill keep their upstream-style separate `k` and
`v` positions; each carries an `AttentionOperatorQuantizedTensorInput` with an
explicit logical shape, and the facade or wrapper combines them privately.
The framework unwraps K/V storage and injects only the scale, zero-point and
optional runtime multiplier arguments authorized by the selected exact
binding; neither public input exposes a provider plan or callable handle.

The documentation map and content policy are in
[`docs/README.md`](docs/README.md). The authoritative design is in
[`docs/framework_design.md`](docs/framework_design.md); the Host Attention contract
architecture is in [`docs/architecture.md`](docs/architecture.md). The Attention architecture
includes host-side framework contracts. Production Ascend C build and installed-wheel
execution are described in the wheel guide above.

The Attention-specific framework contract is documented in
[`docs/attention/framework.md`](docs/attention/framework.md).
The FlashInfer-aligned wrapper-owned plan/run lifecycle and the boundary for
future CANN/flash-attention-npu provider selection are documented in
[`docs/attention/plan_run_dispatch_design.md`](docs/attention/plan_run_dispatch_design.md).
The executable Host contract for INT8/UINT8/packed-INT4 KV Cache is documented
in [`docs/attention/quantization.md`](docs/attention/quantization.md).
Versioned correctness trace and replay semantics are documented in
[`docs/attention/trace.md`](docs/attention/trace.md).
The separate lifecycle/stream/resource-ownership trace schema for injected JIT
and provider routes is documented in
[`docs/attention/protocol_trace.md`](docs/attention/protocol_trace.md).
Workspace capacity/query/lifecycle semantics are documented in
[`docs/attention/workspace.md`](docs/attention/workspace.md).
The joint plan/workspace/tensor/capability identity used to guard execution
reuse is documented in
[`docs/attention/execution_identity.md`](docs/attention/execution_identity.md).
The framework-independent tensor/stride/alias/stream adapter target is in
[`docs/attention/tensor_contract.md`](docs/attention/tensor_contract.md).
The optional metadata-only Torch adapter and its explicit runtime verification
boundary are documented in
[`docs/attention/torch_adapter.md`](docs/attention/torch_adapter.md).
The versioned exceptional-value and metadata admission policies are in
[`docs/attention/numerics.md`](docs/attention/numerics.md) and
[`docs/attention/resource_limits.md`](docs/attention/resource_limits.md).
The evidence-bearing backend support declaration and dispatch gate are defined
in [`docs/attention/capability_profile.md`](docs/attention/capability_profile.md).
The revalidatable plan-to-profile-to-kernel selection record is specified in
[`docs/attention/dispatch_receipt.md`](docs/attention/dispatch_receipt.md).
Structured kernel provenance and launcher ABI requirements are documented in
[`docs/attention/kernel_artifact.md`](docs/attention/kernel_artifact.md).
The frozen Attention C/POD/error ABI and address-generation/event lease state
machine are specified in
[`docs/attention/launcher_abi.md`](docs/attention/launcher_abi.md).
The canonical six-mode CSR/page-table plan metadata wire format is documented
in [`docs/attention/plan_metadata_wire.md`](docs/attention/plan_metadata_wire.md).
The validated auxiliary/run-options contract and TensorView/KV/aux POD
materialization path are documented in
[`docs/attention/launch_binding.md`](docs/attention/launch_binding.md).
The ownership-complete Host/device lease model and canonical 13-argument
submission record are documented in
[`docs/attention/launch_packet.md`](docs/attention/launch_packet.md).
The Host-only artifact verification, symbol resolution, provider submission,
completion-event, and unload lifecycle are documented in
[`docs/attention/launcher_provider.md`](docs/attention/launcher_provider.md).
The two single-request injected-JIT compatibility entries model upstream scratch,
output/LSE allocation and argument ordering only; they do not compile or load an
Ascend artifact and remain `framework` parity.
The independent FlashInfer-style JIT framework package—immutable specs,
registry, explicit environment, cache identity and Attention specialization—is
documented in
[`docs/attention/jit_framework.md`](docs/attention/jit_framework.md). It is
Host-only and deliberately exposes no compiler or default dynamic loader. An internally
selected `ascendc_jit` route must now produce an exact cache-hit binding during
wrapper `plan()`, then reverify the selected artifact bytes through an injected
reader and resolve its exact internal module symbols through an injected loader
before package callable import. An injected planner binder binds the loaded
module's `plan` entry point to the provider plan factory; an executor binder then
binds the same module, exact package callable and executor. All identities are
frozen into the active plan and revalidated by `run()` before any provider
executor may be called.
Untrusted trace/corpus decoding limits are specified in
[`docs/attention/json_envelope.md`](docs/attention/json_envelope.md).
Independent quantization-drift and future backend-error budgets are specified
in [`docs/attention/accuracy.md`](docs/attention/accuracy.md), with a replayable
paired accuracy corpus.
Explicit quantized storage/scale/zero-point blocking, padding, layout catalogs,
and non-executing converter plans are specified in
[`docs/attention/quant_physical_layout.md`](docs/attention/quant_physical_layout.md).
The two single-request APIs, the three paged/ragged batch wrappers and mixed
`BatchAttention` share the same internal provider-selection policy while
keeping Host reference execution explicit. Their public contract and remaining
capability boundaries are documented in
[`docs/attention/frontend_contract.md`](docs/attention/frontend_contract.md).

### Framework diagnostics

The regression command below selects the current quick scope. The CLI commands
inspect or validate the separate Host framework contracts; case/protocol paths
are placeholders for caller-supplied files:

```bash
python3 -B tools/run_regression.py --profile quick
python3 -m flashinfer show-config
python3 -m flashinfer parity-report --scope attention
python3 -m flashinfer list-kernels
python3 -m flashinfer attention-capabilities
python3 -m flashinfer attention-replay path/to/case.json
python3 -m flashinfer attention-coverage --replay --require-complete
python3 -m flashinfer attention-corpus --pretty
python3 -m flashinfer attention-accuracy
python3 -m flashinfer attention-accuracy-corpus --pretty
python3 -m flashinfer attention-protocol-validate path/to/protocol.json
```

The framework diagnostics validate contracts and injected substitutes. Their
success does not establish NPU operator correctness or graph support. Quick
regression is a separate scope and can execute NPU tests.

</details>
