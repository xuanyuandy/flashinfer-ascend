"""Shared command-line checks for the current and legacy Ascend C JIT probes."""

import argparse
import json
import os
from pathlib import Path
import sys


def run_probe(prepare, execute, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--catlass-root", type=Path)
    parser.add_argument("--cann-root", type=Path, default=os.environ.get("ASCEND_HOME_PATH"))
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--rebuild-check", action="store_true")
    args = parser.parse_args()
    from flashinfer.jit import DiskJitBuildStore, JitCacheIndex, JitCompilationPolicy, build_and_load

    spec, backend = prepare(args)
    store = DiskJitBuildStore(args.cache_dir)
    print("declared:", spec.kernel_id, "spec:", spec.fingerprint, flush=True)
    print("disk_cache_hit:", store.lookup(spec) is not None, flush=True)
    if args.cache_only:
        import subprocess

        def forbidden(*args, **kwargs):
            raise AssertionError("cache-only load must not probe/invoke a compiler")

        subprocess.Popen = forbidden
        backend.compile = forbidden
    cache = JitCacheIndex()
    loaded = build_and_load(spec, backend=backend, store=store, cache=cache,
        policy=JitCompilationPolicy.CACHE_ONLY if args.cache_only else JitCompilationPolicy.ENABLED)
    if args.rebuild_check:
        if args.cache_only:
            raise ValueError("--rebuild-check requires compilation enabled")
        before = store.lookup(spec)
        (store.directory(before) / "unexpected-file").write_text("corruption test\n")
        assert store.lookup(spec) is None
        loaded = build_and_load(spec, backend=backend, store=store, cache=cache)
        assert store.lookup(spec).fingerprint == before.fingerprint
        print("PASS corruption rebuild retains the same immutable cache record", flush=True)
    record = store.lookup(spec)
    print("LOAD_OK:", record.artifact.format.value, "entry:", record.artifact.entry_point, flush=True)
    print("artifact:", store.artifact_path(record), flush=True)
    result = execute(loaded.opaque_module, spec.entry_points[0], args.cann_root)
    print("PASS", json.dumps(dict(kernel=spec.kernel_id, **result), sort_keys=True), flush=True)
    assert "torch_npu" not in sys.modules and "triton" not in sys.modules
