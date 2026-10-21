# FlashInfer API parity

Both packaged ledgers use upstream commit
`2895577720741ddfedff118202c89584996b8875`:

- `flashinfer/data/api_parity.json` inventories the tracked public APIs.
- `flashinfer/data/attention_api_parity.json` gives the scoped Attention view.

Check their metadata, baseline agreement and shared Attention statuses with:

```bash
python -m flashinfer parity-check
```

The command needs neither Torch nor an accelerator. CI runs it against the
installed package, which includes the JSON files. Exit code 0 means ledger
integrity passed; it does **not** mean all APIs are implemented. Invalid metadata
returns 2. `--require-complete` additionally returns 1 for incomplete coverage.

The original 147-row CANN coverage matrix and vLLM/SGLang consumer evidence have
not been provided in the current workspace. Both ledgers therefore retain
`inventory_status=bootstrap` and `coverage_matrix.status=missing_source`.
Additional rows are declarations extracted from the pinned upstream source;
their count is not a count of implemented operators or a substitute for that
matrix. Unverified CANN mappings and consumer usage stay `unverified`.

Each entry carries the upstream symbol, CANN coverage and operator names, local
implementation status, and independent `vllm` / `sglang` usage fields. Source
provenance includes the full commit, file, line and SHA-256 of the source file.
`ParityEntry.to_dict()` preserves parameter-level semantics and other additive
metadata when loading and writing entries.

Implementation levels have these meanings:

- `reference`: an executable Host oracle;
- `framework`: a facade, planning, adapter, JIT or query contract;
- `functional`: a callable implementation with capability and numerical evidence;
- `optimized`: a functional implementation with performance evidence;
- `missing`: the tracked contract has not been represented.

Semantic compatibility is independent of implementation status. Every
`intentionally_different` entry requires a reason. NVIDIA-specific interfaces
recorded this way are excluded from the pending implementation gaps. Local NPU
extensions and retained historical compatibility shims have an explicit
`api_kind` and are excluded from the upstream coverage denominator.

The six `attention_surfaces` distinguish Host reference execution, external
provider routing and built-in NPU execution. Five surfaces have tested eager
CANN routes. Mixed paged `BatchAttention` still needs production NPU integration.
Native maximum workspace queries and graph capture remain T9/T10 work and do not
count as completed NPU functionality.

Render one view or enforce the future release gate with:

```bash
python -m flashinfer parity-report --scope all
python -m flashinfer parity-report --scope attention
python -m flashinfer parity-check --require-complete
```

Reproduce the declaration inventory without importing the CUDA project:

```bash
python work_items/scripts/update_parity_inventory.py --upstream /path/to/flashinfer-ori
```

The script reads the pinned Git object regardless of the upstream working tree's
HEAD. It preserves reviewed local statuses, source evidence and parameter notes.
Unknown mappings remain pending until the original matrix is available. The
coverage gate stays incomplete until both that inventory and the scoped stable
APIs are resolved by implementations or documented intentional differences.
