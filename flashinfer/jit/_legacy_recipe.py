"""Internal, immutable v1 decoding for existing artifacts and legacy tests.

No public construction, configuration or release entry point imports these
aliases. They do not upgrade v1 metadata or grant v2 capabilities.
"""

import json
from pathlib import Path

from .recipe import _Recipe as BuildRecipe, _BuildConfig as AscendCBuildConfig
from .recipe import _resolve_build_recipe as resolve_build_recipe
from .recipe import _LegacyBuildDependency, _LEGACY_CATLASS_REVISION


def load_build_recipes(path):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    recipes = tuple(BuildRecipe.from_dict(record) for record in records)
    if any(recipe.schema_version != 1 for recipe in recipes):
        from flashinfer.runtime import SchemaError
        raise SchemaError("internal legacy fixture reader requires exactly schema_version=1")
    return recipes


def export_op_package(directory, output_dir):
    from flashinfer.aot.build import _export_op_package
    return _export_op_package(directory, output_dir, allow_legacy=True)


__all__ = []
