"""Prepare per-kernel acceptance releases through the framework AOT builder."""

from pathlib import Path


def build_release(kernel_id, root):
    from flashinfer.aot.build import build_production_packages

    root = Path(root)
    return build_production_packages(
        output_dir=root / "release",
        cache_dir=root / "build-cache",
        kernel_ids=(kernel_id,),
    )[0]
