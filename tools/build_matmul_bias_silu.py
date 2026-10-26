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


def _descriptor(recipe):
    from tools._production_release import production_descriptor

    bounds = {bound.name: bound for bound in recipe.shape_constraints}
    # The validated release reserves 16 MiB for system workspace. Planning
    # declares a finite-contract upper bound; eager uses the dynamic ACLNN size.
    workspace_bytes = (16 << 20) + 4 * bounds["M"].maximum * bounds["N"].maximum
    return production_descriptor(recipe, op="gemm.matmul_bias_silu", dtype="bfloat16",
                                 workspace_bytes=workspace_bytes)


def freeze_contract():
    """Write registry and frozen recipe/spec using the current verified SDK."""
    from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID
    from tools._production_release import freeze_declaration

    return freeze_declaration(
        ROOT, recipe_name="matmul_bias_silu", kernel_id=MATMUL_BIAS_SILU_KERNEL_ID,
        descriptor_factory=_descriptor,
        specialization=b"matmul_bias_silu;contract-v2;bf16;row-major;M,N,K=16..256:16",
        append=True, generated_at="2026-10-25", allow_new_frozen=True,
    )


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
