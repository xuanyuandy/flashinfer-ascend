"""Capture/replay every builtin's auto request without importing any provider.

Run with the same Python and environment before and after a dispatch change:
    python tools/t19_selection_audit.py --write /tmp/t19-before.json
    python tools/t19_selection_audit.py --compare /tmp/t19-before.json
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path

from flashinfer.runtime import (
    DispatchError, GraphCapability, HardwareSignature, InputSignature, dispatch, get_global_registry,
)


def origin(descriptor):
    backend = descriptor.backend.value
    if backend == "external":
        return "external:" + descriptor.kernel_id.partition("/")[0]
    return {"aclnn": "cann", "ascendc_aot": "ascendc",
            "ascendc_jit": "ascendc"}.get(backend, backend)


def snapshot():
    registry = get_global_registry()
    rows = []
    for descriptor in registry.descriptors():
        cap = descriptor.capability
        signature = InputSignature(
            descriptor.op, dtypes=(cap.dtypes[0] if cap.dtypes else "float16",),
            shapes=((4, 128),), layouts=(cap.layouts[0] if cap.layouts else "",),
            specializations=tuple((k, v[0]) for k, v in cap.specializations),
        )
        hardware = HardwareSignature(
            soc_model=cap.soc_models[0] if cap.soc_models else "Ascend910B",
            aicore_count=32,
        )
        # Include reference-only ops while retaining backend='auto'. Also
        # record the ordinary hot-path result with reference opt-in disabled.
        decision = dispatch(signature, hardware, allow_reference=True, registry=registry)
        try:
            hot_path = dispatch(signature, hardware, registry=registry).kernel_id
        except DispatchError:
            hot_path = None
        rows.append(dict(
            record=descriptor.kernel_id, stack=origin(descriptor),
            fingerprint=descriptor.fingerprint, signature=asdict(signature),
            hardware=asdict(hardware), selected=decision.kernel_id,
            fallback=list(decision.fallback_chain), hot_path_selected=hot_path,
            filters=[asdict(report) for report in decision.reports],
        ))
    # JSON normalization also turns enum/string subclasses and tuples into
    # the same representation used by a persisted pre-change snapshot.
    return json.loads(json.dumps(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", type=Path)
    mode.add_argument("--compare", type=Path)
    parser.add_argument("--allow-graph-metadata-change", action="store_true",
                        help="explicitly allow only proven graph metadata fingerprint changes; retain baseline")
    args = parser.parse_args()
    rows = snapshot()
    groups = defaultdict(list)
    for row in rows:
        groups[row["stack"]].append(row)
    print("records:", len(rows))
    for stack, members in sorted(groups.items()):
        print("stack:", stack, "records:", len(members))
    duplicates = [(stack, op, count) for stack, members in groups.items()
                  for op, count in Counter(r["signature"]["op"] for r in members).items()
                  if count > 1]
    print("multi_config_op_source_groups:", len(duplicates), duplicates)
    if args.write:
        args.write.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        print("baseline:", args.write)
        return
    before = json.loads(args.compare.read_text())
    assert [r["record"] for r in rows] == [r["record"] for r in before]
    print("| record | before auto | after auto |")
    print("|---|---|---|")
    changed = []
    for old, new in zip(before, rows):
        print("| %s | %s | %s |" % (new["record"], old["selected"], new["selected"]))
        if old["fingerprint"] != new["fingerprint"] and args.allow_graph_metadata_change:
            descriptor = get_global_registry().get(new["record"])
            legacy = replace(descriptor, graph_capability=GraphCapability.NOT_CAPTURABLE,
                             graph_param_updates=(), graph_replay=None)
            assert legacy.fingerprint == old["fingerprint"], (
                new["record"], "change is not limited to graph metadata")
            changed.append(new["record"])
            print("graph metadata fingerprint change:", new["record"],
                  old["fingerprint"], "->", new["fingerprint"])
        else:
            assert old["fingerprint"] == new["fingerprint"], (
                new["record"], "fingerprint", old["fingerprint"], new["fingerprint"])
        for key in ("stack", "signature", "hardware", "selected",
                    "fallback", "hot_path_selected", "filters"):
            assert old[key] == new[key], (new["record"], key, old[key], new[key])
    print(f"PASS {len(rows)}/{len(rows)} auto selections unchanged; "
          f"{len(rows) - len(changed)}/{len(rows)} descriptor fingerprints unchanged")
    if changed:
        print("PASS %d graph-only fingerprint changes verified against the original baseline" % len(changed))
    print("PASS hot-path selections, fallback order and per-candidate filters unchanged")


if __name__ == "__main__":
    main()
