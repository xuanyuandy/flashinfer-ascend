"""Export bounded per-SoC tuning shards from reviewed measurement JSON files.

  python -B -m tools.export_tuning_data measurements.json --output-dir build/tuning-release

No collection, compilation or search is performed. With no input records,
export an empty manifest. Output must use a new directory; publication rules
are documented in docs/tuning.md."""

import argparse
import importlib
import json
from pathlib import Path

from flashinfer.tuning._release import MAX_RELEASE_BYTES, export_release_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, nargs="*", help="reviewed raw measurement JSON files")
    parser.add_argument("--output-dir", type=Path, required=True, help="new staging directory")
    parser.add_argument("--registry-factory", help="MODULE:CALLABLE accepting a SoC; default: global registry")
    parser.add_argument("--max-bytes", type=int, default=MAX_RELEASE_BYTES)
    args = parser.parse_args()
    factory = None
    if args.registry_factory:
        module, name = args.registry_factory.split(":", 1)
        factory = getattr(importlib.import_module(module), name)
    result = export_release_data(
        [json.loads(path.read_text()) for path in args.records], args.output_dir,
        registry_factory=factory, max_bytes=args.max_bytes,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
