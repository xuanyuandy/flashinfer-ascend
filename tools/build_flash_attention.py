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


def _descriptor(recipe):
    from tools._production_release import production_descriptor

    # The kernel keeps intermediate tiles in L1/UB. Dynamic ACLNN workspace
    # includes CANN's 16 MiB system prefix; eager uses the exact callback size.
    return production_descriptor(recipe, op="ascendc.single_prefill_catlass", dtype=("float16", "bfloat16"),
                                 workspace_bytes=16 << 20)


def _upsert(records, record):
    """Replace T4 in place while preserving all other frozen records."""
    from tools._production_release import upsert

    upsert(records, record)


def freeze_contract():
    """Resolve the pinned dependency and freeze only T4's descriptor/spec."""
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID, _dependency_roots
    from tools._production_release import freeze_declaration

    return freeze_declaration(
        ROOT, recipe_name="flash_attention", kernel_id=FLASH_ATTENTION_KERNEL_ID,
        descriptor_factory=_descriptor, dependency_roots=_dependency_roots,
        specialization=(b"flash_attention;catlass-example49;contract-v2;fp16,bf16;NHD;B1;"
                        b"D128;H1..8;S,T128..512:128;unmasked;fixed-length"),
    )


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
