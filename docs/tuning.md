# Local and release performance evidence (T20A / T20B)

Tuning is an explicit offline operation. Import and dispatch never collect
timings or compile candidates. Without a usable measured champion, dispatch
keeps the existing T6 filtering, T7 deterministic ordering, and T19A stack
structure. A measurement with no significant difference is useful evidence;
it has `winner: null` and does not override the heuristic.

## FA：CATLASS AOT 与 Triton JIT 隔离接入

`tools/tune_flash_attention.py` 使用独立 registry，不修改默认 81 条实现或生产精确集合断言。
先检查匹配的 CATLASS AOT 及两种候选的数值，再采集精确 dtype/shape 记录。默认测量公共
Torch out 接口；`--timing-boundary runner` 测量已选定 runner，两者均包含同步 eager 下发，
须按记录中的边界解释，不能作为纯设备执行时间。轮数、噪声门槛及显著性规则保持不变。

```bash
python3 -B tools/tune_flash_attention.py --output-dir build/fa-tuning \
  --dtype float16 bfloat16 --shape 128,128,1 --shape 512,512,8
export FLASHINFER_NPU_TUNING_DB=build/fa-tuning/tuning.sqlite3
```

运行端在 `kernel_registry_scope(flash_attention_registry())` 中调用原 FA 接口即可使用记录。
FP16/BF16、shape、descriptor/capability、spec 或执行身份不匹配时拒绝复用；登记作用域退出后
恢复默认候选。软件身份是进程内快照，替换软件包或执行代码须重启进程。未测 shape、无显著
差异及过期记录沿用原确定性选择。`--require-both-backends` 仅验收实际选择是否包含两者，
不会人为指定冠军，未达到条件以非零状态退出并保留全部记录。

真机路由 UT 中的合成计时仅用于验证两条 shape 记录能驱动两个真实 runner，不构成性能证据，
不能发布或作为生产调优记录。实际测量与未验证事项见 [隔离验证报告](reports/triton-fa-isolated-2026-10-29.md)。

## Collect the two existing Gemma configurations

Use a free NPU, the same CPU affinity in both runs, and paths outside the
checkout. The collector loads T19A's real BLOCK=128/256 binaries and checks both
against the same float32 `(4, 128)` inputs, including exact residual mutation.
It prepares independent inputs before every timed sample. Compilation,
allocation, reset and correctness checks are excluded from timing.

```bash
export PYTHONPATH=.:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=0
PY=/root/miniforge3/envs/flashinfer-a5/bin/python
$PY tools/t20_collect_gemma.py --cache-dir /tmp/t20-jit \
  --db /tmp/t20-local.sqlite3 --output /tmp/t20-quiet.json --affinity 120,121
$PY tools/t20_collect_gemma.py --cache-dir /tmp/t20-jit \
  --db /tmp/t20-local.sqlite3 --output /tmp/t20-busy.json --affinity 120,121 \
  --busy-workers 8
```

The optional CPU workers are confined to the collector's affinity, capped at
16 workers, have a 120-second deadline, and are terminated on exit. They create
actual CPU scheduling contention on the launch CPUs without saturating an
entire shared server. System load alone cannot describe contention on a CPU
subset. Inspect both system and affinity CPU utilization, background versus
measurement activity, CPU PSI, and the raw load counters. NPU process/utilization
snapshots and the software/artifact identities are also saved by this example.

## Measurement and conservative decision

Defaults: 50 warmup batches per candidate, 31 paired rounds, 20 independent
invocations per batch, 10 ms between rounds. AB/BA order alternates for the two
configurations; more candidates use a cyclic order. A sample is elapsed wall
time divided by invocations. It **includes Python launch and synchronization**;
it is caller-observed latency, not isolated device execution time.

The nominal fastest median must beat **every** compared candidate under all
three conditions:

1. Median gap exceeds `max(3% * slower_median, 3 * (s_fast + s_slow))`, where
   `s = max(1.4826 * MAD, IQR / 1.349)`. MAD and IQR are robust estimates of spread;
   taking the larger also protects against a narrow center with scheduling tails.
2. The paired count of improvements exceeding 3% passes an exact one-sided sign
   test at `0.01 / (candidate_count * (candidate_count - 1))`. Correcting over
   all directed pairs accounts for choosing the proposed winner from these same
   data. Ties count against the proposed winner.
3. Median advantage exceeds the practical threshold in both execution orders.

These deliberately conservative gates may refuse small real improvements.
The sign test assumes independent pairs; the reported alpha is **not** an
unconditional confidence guarantee for serially correlated shared-host data.
Spacing, reversed orders and a spread margin provide additional protection.
All samples, including outliers and drift, remain in the record. The collector
does not retry until a desired winner appears.

`BenchmarkCandidate(descriptor, prepare)` lets other callers explicitly provide
already validated implementations. `prepare(iterations)` returns a callable
performing that many invocations on independently prepared inputs. Call
`collect_tuning_record(..., synchronize=...)`, inspect its result, then explicitly
call `save_tuning_record(record)`. There is no automatic search space or tuning
trigger in dispatch.

## Storage and lookup

`FLASHINFER_NPU_TUNING_DB` names the local SQLite file. Its default is
`$XDG_CACHE_HOME/flashinfer/tuning.sqlite3` (or `~/.cache/flashinfer/tuning.sqlite3`).
An empty value disables **both** local and release lookup, retaining T20A's
all-tuning-off contract. To use only release data, point this variable at a
nonexistent local path. Reading never creates a database. Unreadable, locked or
malformed local data is skipped; valid release evidence can still be used.

Each appended measurement carries:

```text
op, bucket_key, capability_fingerprint, collected_at, record_id
bucket: semantic signature + shape template + T6.3 ShapeRange declarations
hardware: soc_model, aicore_count, ub_size_bytes
candidates: kernel_id -> descriptor fingerprint + capability fingerprint
measurement: config, timestamps, raw paired samples/orders, machine observations,
             timer/unit, caller metadata including software and artifact identities
result: winner or null, median/MAD/IQR/p10/p90/min/max, pairwise gaps/gates/p-values
```

`capability_fingerprint` in the database key is the canonical SHA256 of the
entire `HardwareSignature`, including core count and UB capacity. Each candidate's
`CapabilitySpec` fingerprint is stored separately. Both tiers check the identities
of **all compared candidates**, not only the champion, before using a record.

The bucket uses the **intersection of the actual T6.3 `shape_ranges`**, with
negative axes normalized against input rank. It does not invent small/medium/large
or power-of-two boundaries. Non-ranged axes and dtype, logical/physical layout and
specializations stay exact. For the two Gemma configurations, hidden ranges
`[1,128]` and `[1,256]` intersect to `[1,128]`; rows remain exactly 4. No declared
ranges means exact shape matching. Records retain their actual measured shape;
bucket reuse does not mean every shape in the interval was measured.

`read_tuning_record(signature, hardware, *, registry=None)` returns a copy of
full **valid** evidence, including inconclusive results.
`query_tuning_record(signature, hardware, *, registry=None)` keeps its
`Optional[str]` return contract and returns only a valid significant champion.
Two-argument calls validate against the global registry. Callers collecting
private kernels must pass the same current private registry; dispatch does this
automatically for its registry. This avoids treating an unregistered kernel as
current simply because its old ID still exists in a measurement.

## Two tiers and conflict policy

Resolution is **valid local evidence > valid release evidence > deterministic
heuristic**. A local measurement observes the user's launch overhead, software
environment and contention on that machine. An otherwise identical hardware
signature does not make a different release machine's measurements more
representative. Release provenance alone is not a reason to override the user's
valid measurement. Dates are compared within a tier, never to promote a newer
release measurement above local evidence.

The newest valid matching local row (append sequence) wins. A newer valid
`winner: null` suppresses both an older local winner and a release winner: it is
positive evidence that the conservative gates found no significant difference,
so selection stays heuristic. An **invalid** row is skipped rather than acting
as an inconclusive result; lookup can continue to an older valid local row and
then the release tier. Release export keeps the newest measurement per exact
key by `finished_at`, breaking equal-time ties by `record_id`, and inserts in
that order so overlapping buckets also resolve deterministically.

Records never bypass the existing capability, target SoC, policy, JIT, custom or
reference filters. If a valid record's champion is filtered out, the existing
heuristic selects among survivors. The T19A stack selection and implementation
selection, and the T7 fallback order, are unchanged.

Explain retains T20A's distributions, sample counts, noise threshold, paired
p-value, time and load. It adds `tuning source=local|release`. Skipped records
carry `source`, `record_id`, and the rejection reason, including saved/current
hardware fields or the candidate and mismatching fingerprint. At most eight
rejections are displayed per resolution; this limit never stops validation.

## Invalidation rules

Both tiers run the same checks. No new hardware or kernel identity scheme is
introduced; the schema is still record version 1.

| Condition | Check and consequence |
|---|---|
| Kernel source changed | Every candidate's saved `descriptor_fingerprint` must equal its current `KernelDescriptor.fingerprint`. This includes `SourceRef.digest` when source is declared. |
| Kernel configuration changed | The same descriptor comparison covers configuration expressed by the descriptor: ID, compile mode, constraints, workspace, target, ABI, and op/tiling schema versions. Capability changes are checked independently below. |
| Declared compiled artifact changed | `ArtifactRef` digest, size, build ID, format, entry point and target enter the descriptor fingerprint; a changed reference invalidates the measurement. |
| Capability changed | Every saved `capability_fingerprint` must equal the current `CapabilitySpec.fingerprint`, including dtype, layout, SoC support, core requirement, specialization and shape ranges. Some legacy descriptors omit capability fields from their own fingerprint; the separate comparison is required. |
| A compared candidate disappeared | Missing IDs in the current registry invalidate the entire comparison. A changed loser could now beat the old winner; an inconclusive comparison can become stale too. |
| Hardware differs | Compare the complete stored `HardwareSignature` with the current one: `soc_model`, `aicore_count`, **and** `ub_size_bytes` must all be equal. Zero or unknown fields cannot establish a match and are refused even against another zero. Also verify the SQL hardware digest agrees with the payload. |
| Schema differs | Record and release manifest versions must be integer 1. SQLite `user_version=1` is current; `0` is accepted only for legacy T20A databases whose individual payloads still pass all version-1 checks. Other database versions are refused. There is no implicit migration. |
| Payload is corrupt or conclusion was edited | Reuse T20A validation, including recomputing its unchanged noise/significance analysis from raw samples. Reject the row and continue resolution. |
| Workload does not match | Keep T20A's bucket matching: exact semantics and non-ranged axes plus the actual T6.3 range intersection. This is a miss, not authority to widen capability. |

These rules depend on accurate descriptor declarations. Source owners must
refresh `SourceRef.digest` when code changes and represent configuration changes
in the descriptor/ID; artifact owners must update `ArtifactRef` when replacing a
binary. Lookup does not hash source/binary files or probe the compiler on the hot
path. A binary modified without updating its declared digest remains the loader's
artifact-verification responsibility.

T20A's free-form software/environment and JIT artifact metadata is retained for
review. It is not a current runtime identity: `HardwareSignature` intentionally
excludes CANN/driver versions, and the dispatch registry does not expose the
current `JitSpec` or loaded JIT binary. This change does **not** claim automatic
detection of an undeclared compiler/options/provider change. Until that identity
is bound to descriptors, discard/recollect affected records after such changes.
No expiry time is invented: age alone neither validates nor invalidates evidence.

On the current A5, an explicit probe returned `Ascend950 / 56 / UB=0`. UB capacity
is therefore unknown to this environment's existing probe. Such measurements
can still be collected and saved for audit, but cannot steer dispatch or be
exported as trusted release evidence. The collector prints that distinction and
the runtime rejection explanation. Fixing the hardware probe is separate work.

Writes refresh this process's lookup and dispatch caches. Other-process writes
or an in-place package replacement require `reload_tuning_records()` or a process
restart. Fresh resolutions automatically apply the rules above. Cache keys
include both data roots/reload revision and current candidate identities; dispatch
also retains its existing registry identity/revision and hardware keys. Cache hits
perform no stat, SQL, collection, device import or compiler probe.

## Release layout, packaging and size

Upstream `flashinfer-ori/flashinfer/tuning_configs/` stores executable Python
`best_configs` dictionaries in files such as
`v0_1_trtllm_fused_moe_NVIDIA_B200.py` and `..._NVIDIA_GB200.py`, selected by device
name and CUTLASS version. Here a SoC directory is only a file partition; it cannot
assert compatibility. We retain the T20A key
`(op, workload bucket digest, full HardwareSignature digest)` instead of copying
upstream SKU filenames or importing executable tuning data.

```text
flashinfer/tuning/data/
  manifest.json                 # schema_version: 1, socs: [...] (currently empty)
  Ascend950/measurements.sqlite3 # only after reviewed measurements exist
  Ascend910B/measurements.sqlite3
```

`pyproject.toml` explicitly includes `tuning/data/manifest.json` and
`tuning/data/*/measurements.sqlite3` as setuptools package data in both wheel and
sdist. Runtime reads the manifest lazily and opens only the current SoC's shard,
using SQLite `mode=ro`. It also compares each record's hardware payload, even if
the shard name and SQL index claim otherwise. It never writes package data or
merges it into a local cache. Tests build wheel/sdist with temporary synthetic
shards, exclude a stray local DB, and run an installed-wheel release hit with
provider imports, timing and subprocess/compiler probes forbidden.

Export retains only the newest full measurement per key, including a latest
inconclusive result; it does not carry the append-only local history or any kernel
binaries. Raw samples, orders, observations and provenance remain in the selected
payload so the result can be audited. The uncompressed **total of all shards and
the manifest is capped at 8 MiB**; export fails rather than silently truncating
evidence. `--max-bytes` may impose a smaller budget. The repository budget test
also checks packaged data against this cap. Unsupported SoCs simply miss.

## Producing release data

Use explicitly reviewed output JSONs from the existing collector. Before
promotion, review correctness checks, machine/software/artifact identities,
quiet/busy conditions, raw samples and applicability. A synthetic unit fixture
is never release performance evidence. The exporter checks schema, hardware
completeness, current descriptor/capability identities and the unchanged T20A
conclusion, but cannot establish that a caller actually measured real hardware.

```bash
# Output must be a new staging directory; no implicit discovery of user caches.
PYTHONPATH=. /root/miniforge3/envs/flashinfer-a5/bin/python tools/t20_export_release.py \
  --output-dir /tmp/reviewed-release-data \
  --registry-factory examples.two_level_gemma:configuration_registry \
  /path/to/reviewed-quiet.json /path/to/reviewed-busy.json
```

After review, copy only that staged manifest and its SoC shards into
`flashinfer/tuning/data/`. Keep the input JSONs and collection commands in the
release evidence archive; selected payloads also remain intact in each shard.
Given the same input JSONs and descriptors, export is byte reproducible,
independent of input-file order. Repeating physical measurements is a different
operation and is subject to noise; it is not promised to reproduce timestamps,
latencies or winners bit for bit. No search or automatic collection is added.

**This release intentionally has zero performance records.** We have only one
A5 and T20A found no significant difference. That does not establish a portable
release champion. The empty manifest is reproducible with no inputs:

```bash
PYTHONPATH=. /root/miniforge3/envs/flashinfer-a5/bin/python tools/t20_export_release.py \
  --output-dir /tmp/empty-release-data
```

User runtime JSON/SQLite measurements stay outside the checkout. `.gitignore`
excludes local SQLite databases and sidecars, with a narrow exception for the
reviewed release shard path; package-data patterns also exclude stray local DBs.
