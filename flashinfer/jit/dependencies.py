"""Build dependency locations shared by release-time AOT and runtime JIT.

Locations select inputs, not their identity. resolve_build_recipe still checks
the declared Git revision and actual header contents before compilation.
"""

import os
from pathlib import Path

from flashinfer.runtime import SchemaError


def dependency_roots(recipe, source_root, *, overrides=None, require_explicit=False):
    """Resolve every declared dependency without operator-name switches."""
    roots = {}
    overrides = overrides or {}
    for dependency in recipe.dependencies:
        variable = "FLASHINFER_%s_ROOT" % dependency.name.upper()
        explicit = overrides.get(dependency.name) or os.environ.get(variable)
        if require_explicit and not explicit:
            raise SchemaError(
                "%s not built due to missing dependency %s: %s is not set; "
                "set %s or pass --dependency-root %s=PATH"
                % (recipe.kernel_id, dependency.name, variable, variable, dependency.name)
            )
        location = Path(explicit or Path(source_root) / "3rdparty" / dependency.name)
        if not (location / dependency.include_dir).is_dir():
            raise SchemaError(
                "%s not built due to missing dependency %s: %s dependency headers are missing "
                "at %s; set %s or pass --dependency-root %s=PATH "
                "(dependency headers are not bundled in the wheel)"
                % (recipe.kernel_id, dependency.name, dependency.name, location,
                   variable, dependency.name)
            )
        roots[dependency.name] = location
    return roots


__all__ = ["dependency_roots"]
