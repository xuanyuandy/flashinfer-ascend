"""Freeze and build the bounded CATLASS FlashAttention production entry.

  FLASHINFER_CATLASS_ROOT=/mnt/fi/catlass-ref \
    /root/miniforge3/envs/flashinfer-a5/bin/python -B \
    tools/build_flash_attention.py --freeze --output /mnt/fi/t4-release

This release-author operation updates only the T4 entry. Import, dispatch and
verified AOT hits never discover a compiler or read the CATLASS checkout.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def freeze_contract():
    """Compatibility selector; new operators use the catalog-driven CLI."""
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID
    from flashinfer.aot.declarations import freeze_production_kernels

    return freeze_production_kernels(source_root=ROOT, kernel_ids=(FLASH_ATTENTION_KERNEL_ID,))[0]


def build_release(root):
    """Build/export the existing frozen T4 identity using the common builder."""
    from flashinfer.aot.build import build_production_packages
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID

    root = Path(root)
    return build_production_packages(output_dir=root / "release", cache_dir=root / "build-cache",
                                     kernel_ids=(FLASH_ATTENTION_KERNEL_ID,))[0]


def main():
    from tools._production_release import release_main

    release_main(description=__doc__, freeze_contract=freeze_contract, build_release=build_release,
                 freeze_help='refresh only T4 registry/spec JSON', catlass_override=True)


if __name__ == "__main__":
    main()
