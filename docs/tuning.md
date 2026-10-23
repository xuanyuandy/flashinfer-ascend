# Local performance evidence (T20A)

Tuning is an explicit offline operation. Import and dispatch never collect
timings or compile candidates. Without a usable measured champion, dispatch
keeps the existing T6 filtering, T7 deterministic ordering, and T19A stack
structure. A measurement with no significant difference is useful evidence;
it has `winner: null` and does not override the heuristic.

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
An empty value disables lookup. Reading a missing database never creates it.
Unreadable, locked or malformed data produces a deterministic heuristic miss.

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
`CapabilitySpec` fingerprint is stored separately. Descriptor/capability identity
of a returned champion is checked before it can affect selection.

The bucket uses the **intersection of the actual T6.3 `shape_ranges`**, with
negative axes normalized against input rank. It does not invent small/medium/large
or power-of-two boundaries. Non-ranged axes and dtype, logical/physical layout and
specializations stay exact. For the two Gemma configurations, hidden ranges
`[1,128]` and `[1,256]` intersect to `[1,128]`; rows remain exactly 4. No declared
ranges means exact shape matching. Records retain their actual measured shape;
bucket reuse does not mean every shape in the interval was measured.

`read_tuning_record(signature, hardware)` returns full matching evidence,
including inconclusive results. `query_tuning_record(signature, hardware)` keeps
its `Optional[str]` contract and returns only a significant champion. The latest
matching record takes precedence within this local measurement history, so new
inconclusive evidence suppresses an older winner. Explain reports the record,
sample counts, dispersion, noise threshold, paired p-value, time and load; an
inconclusive hit explicitly explains why selection remains heuristic.

Writes refresh this process's lookup and decision caches. When another process
writes the database, use `reload_tuning_records()` or restart. T20B will address
release database packaging, release/local priority and automatic invalidation
rules, including software/environment changes. T20A records that metadata but
does not claim to implement those policies. Runtime JSON/SQLite measurements
are local products and must not be checked into this repository.
