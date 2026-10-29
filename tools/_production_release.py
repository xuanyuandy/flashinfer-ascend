"""Compatibility CLI for historical per-operator release selectors."""

import argparse
import json
import os
from pathlib import Path


def release_main(*, description, freeze_contract, build_release, freeze_help, catlass_override=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--freeze", action="store_true", help=freeze_help)
    parser.add_argument("--output", type=Path, help="build and export into this directory")
    if catlass_override:
        parser.add_argument("--catlass-root", type=Path, help="override FLASHINFER_CATLASS_ROOT for this build")
    args = parser.parse_args()
    if not args.freeze and args.output is None:
        parser.error("supply --freeze and/or --output")
    if catlass_override and args.catlass_root is not None:
        os.environ["FLASHINFER_CATLASS_ROOT"] = str(args.catlass_root)
    result = {}
    if args.freeze:
        descriptor, recipe, spec = freeze_contract()
        result.update(kernel_id=descriptor.kernel_id, recipe=recipe.fingerprint,
                      spec=spec.fingerprint, toolchain=spec.environment_fingerprint)
    if args.output is not None:
        result["release"] = str(build_release(args.output))
    print(json.dumps(result, sort_keys=True))
