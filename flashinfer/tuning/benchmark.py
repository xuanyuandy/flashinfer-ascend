"""Noise-aware, opt-in benchmarking; no torch/provider imports or background work."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import statistics
import time
from typing import Callable
import uuid

from ..runtime.schema import KernelDescriptor


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup: int = 50
    rounds: int = 31
    iterations: int = 20
    round_interval_s: float = 0.01
    minimum_relative_gap: float = 0.03
    noise_multiplier: float = 3.0
    family_alpha: float = 0.01

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.warmup, self.rounds, self.iterations))
                or self.warmup < 1 or self.rounds < 25 or self.iterations < 1):
            raise ValueError("require warmup >= 1, rounds >= 25, iterations >= 1")
        if not (all(math.isfinite(v) for v in (self.round_interval_s, self.minimum_relative_gap,
                                               self.noise_multiplier, self.family_alpha))
                and self.round_interval_s >= 0
                and 0 < self.minimum_relative_gap < 1 and self.noise_multiplier >= 3
                and 0 < self.family_alpha <= 0.01):
            raise ValueError("invalid noise/significance thresholds")


@dataclass(frozen=True)
class BenchmarkCandidate:
    descriptor: KernelDescriptor
    # Prepare independent, correct inputs OUTSIDE each timed sample. The returned
    # callable executes exactly `iterations` invocations, even for in-place ops.
    prepare: Callable[[int], Callable[[], None]]


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _quantile(values, q):
    values = sorted(values)
    p = (len(values) - 1) * q
    lo = int(p)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (p - lo)


def _distribution(samples):
    median = statistics.median(samples)
    mad = statistics.median(abs(v - median) for v in samples)
    iqr = _quantile(samples, .75) - _quantile(samples, .25)
    return dict(median_us=median, mad_us=mad, robust_sigma_us=max(1.4826 * mad, iqr / 1.349),
                p10_us=_quantile(samples, .1), p90_us=_quantile(samples, .9),
                iqr_us=iqr,
                min_us=min(samples), max_us=max(samples))


def _analyze(samples, config, orders):
    """Require a practical gap, a three-robust-sigma noise margin, a Bonferroni
    corrected paired sign test, and an advantage in BOTH execution orders.
    Ties count against the contender; all outliers remain in the saved samples.
    """
    stats = {key: _distribution(values) for key, values in samples.items()}
    best = min(stats, key=lambda key: (stats[key]["median_us"], key))
    comparisons = []
    for other in sorted(set(samples) - {best}):
        a, b = stats[best], stats[other]
        gap = b["median_us"] - a["median_us"]
        required = max(config.minimum_relative_gap * b["median_us"],
                       config.noise_multiplier * (a["robust_sigma_us"] + b["robust_sigma_us"]))
        wins = sum(x < y * (1 - config.minimum_relative_gap)
                   for x, y in zip(samples[best], samples[other]))
        n = len(samples[best])
        p = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
        order_gaps = []
        for before in (True, False):
            indices = [i for i, order in enumerate(orders)
                       if (order.index(best) < order.index(other)) == before]
            order_gaps.append((statistics.median(samples[other][i] for i in indices) -
                               statistics.median(samples[best][i] for i in indices)) if indices else None)
        # The proposed winner was selected from these data. Correct over ALL
        # ordered pairs, including that selection, not only its K-1 opponents.
        alpha = config.family_alpha / (len(samples) * (len(samples) - 1))
        accepted = (gap > required and p <= alpha
                    and all(g is not None and g > config.minimum_relative_gap * b["median_us"]
                            for g in order_gaps))
        comparisons.append(dict(candidate=best, other=other, median_gap_us=gap,
                                required_gap_us=required, paired_practical_wins=wins,
                                paired_rounds=n, sign_test_p=p, order_gaps_us=order_gaps,
                                significant=accepted))
    winner = best if comparisons and all(c["significant"] for c in comparisons) else None
    return dict(status="winner" if winner else "no_significant_difference",
                winner=winner, distributions=stats, comparisons=comparisons,
                rule=(f"paired-sign+{config.noise_multiplier:g}-max(MAD*1.4826,IQR/1.349)+"
                      f"{100 * config.minimum_relative_gap:g}%-gap+both-orders/v1"),
                confidence_note="sign-test assumes independent pairs; robust spread and order checks also required")


def _machine_snapshot():
    """Linux pressure/CPU counters are observations, not inferred from load alone."""
    snapshot = dict(at=_utc(), monotonic_ns=time.monotonic_ns(),
                    load_average=list(os.getloadavg()), cpu_count=os.cpu_count())
    try:
        snapshot["affinity"] = sorted(os.sched_getaffinity(0))
    except AttributeError:
        snapshot["affinity"] = []
    for name, path in (("cpu_pressure", "/proc/pressure/cpu"), ("loadavg_raw", "/proc/loadavg")):
        try:
            snapshot[name] = Path(path).read_text().strip()
        except OSError:
            snapshot[name] = None
    try:
        snapshot["cpu_ticks"] = {line.split()[0]: [int(v) for v in line.split()[1:9]]
                                 for line in Path("/proc/stat").read_text().splitlines()
                                 if line.startswith("cpu")}
    except OSError:
        snapshot["cpu_ticks"] = {}
    return snapshot


def _machine_delta(before, after):
    busy = {}
    for cpu, end in after["cpu_ticks"].items():
        if cpu not in before["cpu_ticks"]:
            continue
        delta = [b - a for a, b in zip(before["cpu_ticks"][cpu], end)]
        total = sum(delta)
        if total > 0:
            busy[cpu] = 100 * (total - delta[3] - delta[4]) / total
    affinity = [busy[f"cpu{i}"] for i in before["affinity"] if f"cpu{i}" in busy]
    return dict(system_cpu_busy_percent=busy.get("cpu"),
                affinity_cpu_busy_percent=statistics.mean(affinity) if affinity else None,
                per_cpu_busy_percent=busy)


def _bucket(signature, descriptors):
    # Reuse actual T6.3 ShapeRange declarations and their intersection.
    # Unranged axes remain exact; no guessed size/power-of-two grid.
    ranges = {}
    shapes = [list(shape) for shape in signature.shapes]
    for descriptor in descriptors:
        for bound in descriptor.capability.shape_ranges:
            axis = bound.axis % len(shapes[bound.tensor_index])
            normalized = dict(bound.to_dict(), axis=axis)
            ranges[(bound.tensor_index, axis, bound.name, bound.minimum, bound.maximum)] = normalized
            shapes[bound.tensor_index][axis] = None
    semantics = signature.to_dict()
    semantics.pop("op")
    semantics["shapes"] = shapes
    return dict(signature=semantics, shape_ranges=[ranges[k] for k in sorted(ranges)])


def collect_tuning_record(signature, hardware, candidates, *, synchronize,
                          config=BenchmarkConfig(), metadata=None):
    """Measure explicitly supplied implementations, synchronizing timed wall work.

    Callers must validate semantic correctness before collection. Compilation,
    fixture setup, and correctness checks stay outside timing. Timings include
    Python launch and final synchronization: this is caller-observed latency.
    """
    candidates = sorted(candidates, key=lambda c: c.descriptor.kernel_id)
    if len(candidates) < 2 or len({c.descriptor.kernel_id for c in candidates}) != len(candidates):
        raise ValueError("need at least two distinct comparable candidates")
    for candidate in candidates:
        reasons = (candidate.descriptor.capability.unsupported_reasons(signature, hardware)
                   + candidate.descriptor.unsupported_target_reasons(hardware.soc_model))
        if reasons:
            raise ValueError(f"ineligible candidate {candidate.descriptor.kernel_id}: {reasons}")
    started = _utc()
    background_before = _machine_snapshot()
    time.sleep(.25)
    background_after = _machine_snapshot()
    for _ in range(config.warmup):
        for candidate in candidates:
            run = candidate.prepare(config.iterations)
            synchronize()
            run()
            synchronize()
    before = _machine_snapshot()
    samples = {c.descriptor.kernel_id: [] for c in candidates}
    orders, sample_times = [], []
    for round_index in range(config.rounds):
        offset = round_index % len(candidates)
        order = candidates[offset:] + candidates[:offset]
        orders.append([c.descriptor.kernel_id for c in order])
        sample_times.append(_utc())
        for candidate in order:
            run = candidate.prepare(config.iterations)
            synchronize()
            begin = time.perf_counter_ns()
            run()
            synchronize()
            elapsed = (time.perf_counter_ns() - begin) / (1000 * config.iterations)
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("timer returned a nonpositive/nonfinite duration")
            samples[candidate.descriptor.kernel_id].append(elapsed)
        if round_index + 1 < config.rounds:
            time.sleep(config.round_interval_s)
    after = _machine_snapshot()
    return dict(schema_version=1, record_id=uuid.uuid4().hex, op=signature.op,
                signature=signature.to_dict(), hardware=asdict(hardware),
                bucket=_bucket(signature, [c.descriptor for c in candidates]),
                candidates={c.descriptor.kernel_id: dict(
                    descriptor_fingerprint=c.descriptor.fingerprint,
                    capability_fingerprint=c.descriptor.capability.fingerprint,
                ) for c in candidates},
                measurement=dict(config=asdict(config), started_at=started, finished_at=_utc(),
                                 timer="synchronized_wall/perf_counter_ns", unit="us/invocation",
                                 samples_us=samples, sample_times=sample_times, orders=orders,
                                 background_probe_seconds=.25,
                                 machine_background=_machine_delta(background_before, background_after),
                                 machine_before=before, machine_after=after,
                                 machine_activity=_machine_delta(before, after), metadata=metadata or {}),
                result=_analyze(samples, config, orders))
