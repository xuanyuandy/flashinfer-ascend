"""Generate/check operator declarations or build through the existing AOT pipeline."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "check", "build"))
    parser.add_argument("--kernel-id", action="append", default=[])
    parser.add_argument("--source-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    from .authoring import generate
    selected = generate(source_root=args.source_root, kernel_ids=args.kernel_id, check=args.action == "check")
    if args.kernel_id and set(selected) != set(args.kernel_id):
        parser.error("requested kernel is absent from author directories")
    if args.action == "build":
        if not selected:
            parser.error("build requires at least one discovered operator")
        if not args.output_dir:
            parser.error("build requires --output-dir")
        from flashinfer.aot.declarations import freeze_production_kernels
        from flashinfer.aot.build import build_production_packages
        from .implementation import implementation_entries
        native = {r["kernel_id"] for r in implementation_entries() if r["backend"] == "ascendc"}
        selected = tuple(k for k in selected if k in native)
        if not selected:
            parser.error("build requires a source-backed Ascend C implementation")
        freeze_production_kernels(source_root=args.source_root, kernel_ids=selected)
        build_production_packages(source_root=args.source_root, kernel_ids=selected,
                                  output_dir=args.output_dir, cache_dir=args.cache_dir)
    print(json.dumps(dict(action=args.action, kernels=list(selected), result="PASS")))


if __name__ == "__main__":
    main()
