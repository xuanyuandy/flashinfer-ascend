"""Freeze the production declaration and/or build its portable AOT release.

Run from the source checkout using the supported Python/CANN environment:
  python -B tools/build_matmul_bias_silu.py --freeze --output /mnt/fi/t2-release

Freezing is a release-author action, never part of import, dispatch or an AOT
hit. The resulting static JSON is shipped with the production registry entry.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def freeze_contract():
    """Compatibility selector; capability/workspace/ABI come from the catalog."""
    from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID
    from flashinfer.aot.declarations import freeze_production_kernels

    return freeze_production_kernels(source_root=ROOT, kernel_ids=(MATMUL_BIAS_SILU_KERNEL_ID,))[0]


def build_release(root):
    """Build/export the exact frozen production identity; return its directory."""
    from flashinfer.aot.build import build_production_packages
    from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID

    root = Path(root)
    return build_production_packages(output_dir=root / "release", cache_dir=root / "build-cache",
                                     kernel_ids=(MATMUL_BIAS_SILU_KERNEL_ID,))[0]


def main():
    from tools._production_release import release_main

    release_main(description=__doc__, freeze_contract=freeze_contract, build_release=build_release,
                 freeze_help='refresh packaged registry/spec JSON')


if __name__ == "__main__":
    main()
