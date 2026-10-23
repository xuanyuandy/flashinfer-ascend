"""Immutable Ascend C build contracts; importing/constructing them performs no I/O.

Resolution is an explicit build preparation step. Only the selected recipe's
dependency closure is read, and the result is bound into the ordinary JitSpec.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Optional, Tuple

from flashinfer.runtime import KernelDescriptor, SchemaError, SourceRef

from .core import _canonical_hash, _sha256


CATLASS_REVISION = "41bf90da655bba3c66d0acd7e00abe33960ecfd6"
_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def _path(value):
    if (not isinstance(value, str) or not value or value == "."
            or str(PurePosixPath(value)) != value or PurePosixPath(value).is_absolute()
            or ".." in PurePosixPath(value).parts or "\\" in value
            or any(c in value for c in "\0\n\r;")):
        raise SchemaError("recipe paths must be normalized relative POSIX paths")
    return value


def _strings(values):
    if (not isinstance(values, (tuple, list)) or any(
            not isinstance(v, str) or not v or any(c in v for c in "\0\n\r;")
            for v in values)):
        raise SchemaError("recipe flags must be an array of safe non-empty strings")
    return tuple(values)


@dataclass(frozen=True)
class BuildDependency:
    name: str
    revision: str
    include_dir: str = "include"

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", self.name):
            raise SchemaError("invalid recipe dependency name")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise SchemaError("dependency revision must be a full Git commit")
        _path(self.include_dir)
        if self.name == "catlass" and self.revision != CATLASS_REVISION:
            raise SchemaError("this adapter requires the pinned CATLASS revision")


@dataclass(frozen=True)
class BuildRecipe:
    recipe_id: str
    version: int
    kernel_id: str
    implementation: str
    target_soc: str
    npu_arch: str
    source: SourceRef
    sources: Tuple[SourceRef, ...]
    headers: Tuple[SourceRef, ...]
    include_dirs: Tuple[str, ...]
    dependencies: Tuple[BuildDependency, ...]
    definitions: Tuple[str, ...]
    compile_options: Tuple[str, ...]
    link_options: Tuple[str, ...]
    device_entry: str
    host_entry: str
    launch_abi: str
    tiling_mode: str
    tiling_schema_version: int
    block_dim: int
    workspace_bytes: int
    task_type: str
    sync_policy: str
    toolchain: str = "ascendc"
    output_kind: str = "host_shared"
    output_adapter: str = "ascendc.host_shared.v1"
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or type(self.version) is not int or self.version < 1:
            raise SchemaError("unsupported BuildRecipe version")
        for name in ("recipe_id", "kernel_id", "target_soc", "npu_arch", "launch_abi"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or any(c.isspace() or c in "\0;" for c in value):
                raise SchemaError("invalid recipe " + name)
        if self.implementation not in {"native", "catlass"} or self.toolchain != "ascendc":
            raise SchemaError("recipe requires native/catlass Ascend C implementation")
        # 13-II must define its own package manifest and loader, not reinterpret a .so.
        if self.output_kind != "host_shared" or self.output_adapter != "ascendc.host_shared.v1":
            raise SchemaError("13-I supports only host_shared / ascendc.host_shared.v1")
        for name in ("sources", "headers", "dependencies"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        refs = self.sources + self.headers
        if (not isinstance(self.source, SourceRef) or self.source not in self.sources
                or any(not isinstance(r, SourceRef) or r.language != "ascendc" for r in refs)
                or len({r.locator for r in refs}) != len(refs)):
            raise SchemaError("recipe requires unique Ascend C sources/headers including source")
        for ref in refs:
            _path(ref.locator)
        if any(not isinstance(d, BuildDependency) for d in self.dependencies):
            raise SchemaError("recipe dependencies must be BuildDependency")
        names = tuple(d.name for d in self.dependencies)
        if len(set(names)) != len(names) or ("catlass" in names) != (self.implementation == "catlass"):
            raise SchemaError("CATLASS dependency belongs only to a catlass recipe")
        object.__setattr__(self, "include_dirs", tuple(_path(p) for p in self.include_dirs))
        for name in ("definitions", "compile_options", "link_options"):
            object.__setattr__(self, name, _strings(getattr(self, name)))
        if any(not _SYMBOL.fullmatch(d.partition("=")[0]) or d.startswith("FI_")
               for d in self.definitions):
            raise SchemaError("invalid/reserved recipe definition")
        if any(not _SYMBOL.fullmatch(getattr(self, n)) for n in ("host_entry", "device_entry")):
            raise SchemaError("recipe entries must be C symbols")
        if (self.tiling_mode != "none" or self.tiling_schema_version != 1
                or type(self.block_dim) is not int or self.block_dim < 1
                or self.workspace_bytes != 0 or self.task_type != "aic_only"
                or self.sync_policy != "compiler_default"):
            raise SchemaError("host_shared.v1 supports explicit fixed tiling, AIC-only, no workspace")

    def to_dict(self):
        value = asdict(self)
        value["source"] = self.source.to_dict()
        for name in ("sources", "headers"):
            value[name] = [r.to_dict() for r in getattr(self, name)]
        return value

    @classmethod
    def from_dict(cls, value):
        try:
            data = dict(value)
            data["source"] = SourceRef.from_dict(data["source"])
            for name in ("sources", "headers"):
                data[name] = tuple(SourceRef.from_dict(r) for r in data[name])
            data["dependencies"] = tuple(BuildDependency(**d) for d in data["dependencies"])
            return cls(**data)
        except (KeyError, TypeError, ValueError) as error:
            raise SchemaError("invalid BuildRecipe manifest: %s" % error) from error

    @property
    def fingerprint(self):
        return _canonical_hash(self.to_dict())

    def validate_descriptor(self, descriptor: KernelDescriptor):
        target = descriptor.artifact.target_soc if descriptor.artifact else descriptor.target_soc
        if (descriptor.kernel_id != self.kernel_id or descriptor.source != self.source
                or target != self.target_soc
                or descriptor.tiling_schema_version != self.tiling_schema_version):
            raise SchemaError("BuildRecipe does not bind kernel/source/target_soc/tiling")
        if descriptor.launch_abi is None or (
                descriptor.launch_abi.abi_name != self.launch_abi
                or descriptor.launch_abi.entry_point != self.host_entry):
            raise SchemaError("BuildRecipe does not bind host launch ABI")


@dataclass(frozen=True)
class AscendCBuildConfig:
    """Select an existing variant before resolving dependencies or declaring a unit."""

    implementation: str = "native"
    output_kind: str = "host_shared"
    target_soc: str = "Ascend950"
    overrides: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self):
        try:
            pairs = tuple(tuple(p) for p in self.overrides)
        except TypeError as error:
            raise SchemaError("invalid Ascend C build overrides") from error
        if (self.implementation not in {"native", "catlass"}
                or self.output_kind != "host_shared"
                or not isinstance(self.target_soc, str) or not self.target_soc
                or any(len(p) != 2 or not isinstance(p[0], str) or not p[0]
                       or p[1] not in {"native", "catlass"} for p in pairs)
                or len({p[0] for p in pairs}) != len(pairs)):
            raise SchemaError("invalid Ascend C build configuration")
        object.__setattr__(self, "overrides", tuple(sorted(pairs)))

    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["overrides"] = tuple(data.get("overrides", {}).items())
        return cls(**data)

    def select(self, recipes, recipe_id):
        implementation = dict(self.overrides).get(recipe_id, self.implementation)
        matches = [r for r in recipes if (r.recipe_id, r.implementation, r.target_soc, r.output_kind)
                   == (recipe_id, implementation, self.target_soc, self.output_kind)]
        if len(matches) != 1:
            raise SchemaError("expected one recipe for %s implementation=%s target=%s output=%s; found %d"
                              % (recipe_id, implementation, self.target_soc, self.output_kind, len(matches)))
        return matches[0]


def load_build_recipes(path):
    """Explicit manifest read; never done by import or the auto selector."""
    return tuple(BuildRecipe.from_dict(r) for r in json.loads(Path(path).read_text(encoding="utf-8")))


def _source_bytes(root, ref):
    path = (root / ref.locator).resolve(strict=True)
    if not path.is_relative_to(root):
        raise SchemaError("recipe source escapes source_root")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.digest:
        raise SchemaError("SourceRef digest mismatch: %s" % ref.locator)
    return payload


def _header_tree(root):
    if not root.is_dir():
        raise SchemaError("dependency include directory is missing: %s; initialize the pinned submodule explicitly" % root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SchemaError("dependency headers must not contain symlinks: %s" % path)
        if path.is_file():
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:
        raise SchemaError("dependency header tree is empty: %s" % root)
    return _canonical_hash(files)


@dataclass(frozen=True)
class ResolvedDependency:
    declaration: BuildDependency
    root: Path
    headers_digest: str

    def __post_init__(self):
        if not isinstance(self.declaration, BuildDependency):
            raise SchemaError("resolved dependency requires BuildDependency")
        object.__setattr__(self, "root", Path(self.root))
        _sha256("dependency headers_digest", self.headers_digest)

    def to_dict(self):
        return dict(**asdict(self.declaration), headers_digest=self.headers_digest)


@dataclass(frozen=True)
class ResolvedBuildRecipe:
    recipe: BuildRecipe
    source_root: Path
    dependencies: Tuple[ResolvedDependency, ...]
    toolchain_fingerprint: str
    build_files: Tuple[Tuple[str, str], ...]

    def __post_init__(self):
        if not isinstance(self.recipe, BuildRecipe):
            raise SchemaError("resolved recipe requires BuildRecipe")
        object.__setattr__(self, "source_root", Path(self.source_root))
        dependencies = tuple(self.dependencies)
        if tuple(d.declaration for d in dependencies) != self.recipe.dependencies:
            raise SchemaError("resolved dependencies must bind the recipe dependency closure")
        object.__setattr__(self, "dependencies", dependencies)
        _sha256("toolchain_fingerprint", self.toolchain_fingerprint)
        files = tuple((_path(p), _sha256("build input digest", h)) for p, h in self.build_files)
        if {p for p, _ in files} != {"csrc/CMakeLists.txt", "csrc/cmake/HostShared.cmake"} or len(files) != 2:
            raise SchemaError("resolved recipe must bind both host_shared CMake inputs")
        object.__setattr__(self, "build_files", files)

    def to_dict(self):
        return dict(recipe=self.recipe.to_dict(), dependencies=[d.to_dict() for d in self.dependencies],
                    toolchain_fingerprint=self.toolchain_fingerprint, build_files=list(self.build_files))

    @property
    def fingerprint(self):
        return _canonical_hash(self.to_dict())


def resolve_build_recipe(recipe: BuildRecipe, *, source_root: Path,
                         toolchain_fingerprint: str, dependency_roots: Optional[Mapping[str, Path]] = None):
    """Read selected inputs, verify pinned commits and hash actual header content.

    Native resolution never looks up CATLASS, runs Git, or reads its directory.
    Toolchain discovery is a separate explicit operation, not a resolver side effect.
    """
    import subprocess

    root = Path(source_root).resolve(strict=True)
    for ref in recipe.sources + recipe.headers:
        _source_bytes(root, ref)
    dependencies = []
    for declaration in recipe.dependencies:
        location = Path((dependency_roots or {}).get(declaration.name, root / "3rdparty" / declaration.name)).resolve()
        include = location / declaration.include_dir
        # Check headers first: an uninitialized submodule must not accidentally
        # inherit the parent repository's HEAD from `git -C ...`.
        digest = _header_tree(include)
        revision = subprocess.check_output(["git", "-C", str(location), "rev-parse", "HEAD"],
                                           text=True, stderr=subprocess.STDOUT).strip()
        if revision != declaration.revision:
            raise SchemaError("dependency %s revision mismatch: expected %s, got %s"
                              % (declaration.name, declaration.revision, revision))
        dependencies.append(ResolvedDependency(declaration, location, digest))
    build_files = tuple((p, hashlib.sha256((root / p).read_bytes()).hexdigest()) for p in (
        "csrc/CMakeLists.txt", "csrc/cmake/HostShared.cmake"))
    return ResolvedBuildRecipe(recipe, root, tuple(dependencies), toolchain_fingerprint, build_files)


__all__ = ["BuildRecipe", "BuildDependency", "AscendCBuildConfig", "ResolvedBuildRecipe",
           "load_build_recipes", "resolve_build_recipe"]
