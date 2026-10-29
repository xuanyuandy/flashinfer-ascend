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
from typing import Mapping, Optional, Tuple, Union

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
class TensorParameter:
    """An ordered ACLNN tensor argument and its device-visible shape contract."""

    name: str
    role: str
    dtype: str
    shape: Tuple[Union[str, int], ...]

    def __post_init__(self):
        if (not isinstance(self.name, str) or not _SYMBOL.fullmatch(self.name)
                or self.name in {"workspace", "tiling", "executor", "stream", "workspace_bytes"}):
            raise SchemaError("invalid/reserved tensor parameter name")
        if not isinstance(self.role, str) or self.role not in {"input", "output"}:
            raise SchemaError("tensor parameter role must be input or output")
        if (not isinstance(self.dtype, str) or self.dtype not in {
                "bfloat16", "float16", "float32", "float64", "int8", "uint8", "int16", "int32", "int64", "bool"}):
            raise SchemaError("unsupported tensor parameter dtype")
        if (not isinstance(self.shape, (tuple, list))
                or any(not ((type(d) is int and d > 0)
                            or (isinstance(d, str) and _SYMBOL.fullmatch(d))) for d in self.shape)):
            raise SchemaError("tensor parameter shape requires positive dimensions or declared symbols")
        object.__setattr__(self, "shape", tuple(self.shape))


@dataclass(frozen=True)
class ShapeConstraint:
    """Finite host/dispatch bounds for one shared tensor dimension."""

    name: str
    minimum: int
    maximum: int
    multiple_of: int = 1

    def __post_init__(self):
        if (not isinstance(self.name, str) or not _SYMBOL.fullmatch(self.name)
                or any(type(v) is not int for v in (self.minimum, self.maximum, self.multiple_of))
                or self.minimum < 1 or self.maximum < self.minimum or self.multiple_of < 1
                or ((self.minimum + self.multiple_of - 1) // self.multiple_of) * self.multiple_of > self.maximum):
            raise SchemaError("shape constraint requires a named, nonempty finite positive range")


@dataclass(frozen=True)
class _Recipe:
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
    workspace_bytes: Union[int, str]
    task_type: str
    sync_policy: str
    toolchain: str = "ascendc"
    output_kind: str = "host_shared"
    output_adapter: str = "ascendc.host_shared.v1"
    schema_version: int = 1
    host_sources: Tuple[SourceRef, ...] = ()
    package_vendor: Optional[str] = None
    package_op_type: Optional[str] = None
    tensor_parameters: Tuple[TensorParameter, ...] = ()
    shape_constraints: Tuple[ShapeConstraint, ...] = ()
    tiling_host_entry: Optional[str] = None
    workspace_host_entry: Optional[str] = None

    def __post_init__(self):
        if (type(self.schema_version) is not int or self.schema_version not in {1, 2}
                or type(self.version) is not int or self.version < 1):
            raise SchemaError("unsupported BuildRecipe version")
        for name in ("recipe_id", "kernel_id", "target_soc", "npu_arch", "launch_abi"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or any(c.isspace() or c in "\0;" for c in value):
                raise SchemaError("invalid recipe " + name)
        if self.implementation not in {"native", "catlass"} or self.toolchain != "ascendc":
            raise SchemaError("recipe requires native/catlass Ascend C implementation")
        if (self.output_kind, self.output_adapter) not in {
                ("host_shared", "ascendc.host_shared.v1"),
                ("op_package", "ascendc.op_package.v1")}:
            raise SchemaError("unsupported Ascend C output kind/adapter")
        if self.output_kind == "host_shared":
            if self.host_sources or self.package_vendor is not None or self.package_op_type is not None:
                raise SchemaError("host_shared cannot declare an op-package contract")
        elif (not self.host_sources or not isinstance(self.package_vendor, str)
              or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.package_vendor)
              or not isinstance(self.package_op_type, str) or not _SYMBOL.fullmatch(self.package_op_type)
              or self.host_entry != "aclnn" + self.package_op_type):
            raise SchemaError("op_package requires host sources, vendor and matching ACLNN op type/entry")
        for name in ("sources", "headers", "host_sources", "dependencies"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        refs = self.sources + self.headers + self.host_sources
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
        object.__setattr__(self, "tensor_parameters", tuple(self.tensor_parameters))
        object.__setattr__(self, "shape_constraints", tuple(self.shape_constraints))
        if self.schema_version == 1:
            # Do not relax the frozen v1 protocol when adding a new contract.
            expected_tiling = "none" if self.output_kind == "host_shared" else "fixed"
            if (self.tiling_mode != expected_tiling or self.tiling_schema_version != 1
                    or type(self.block_dim) is not int or self.block_dim < 1
                    or self.workspace_bytes != 0 or self.task_type != "aic_only"
                    or self.sync_policy != "compiler_default"):
                raise SchemaError("Ascend C v1 supports explicit fixed tiling, AIC-only, no workspace")
            if (self.tensor_parameters or self.shape_constraints or self.tiling_host_entry is not None
                    or self.workspace_host_entry is not None):
                raise SchemaError("Ascend C v1 cannot declare v2 tensor/host contracts")
        else:
            self._validate_v2()

    def _validate_v2(self):
        if (self.output_kind != "op_package" or self.tiling_mode != "host_dynamic"
                or type(self.tiling_schema_version) is not int or self.tiling_schema_version < 2
                or type(self.block_dim) is not int or self.block_dim < 1):
            raise SchemaError("Ascend C v2 requires op_package, host_dynamic tiling schema >= 2 and positive block_dim")
        if not isinstance(self.tiling_host_entry, str) or not _SYMBOL.fullmatch(self.tiling_host_entry):
            raise SchemaError("host_dynamic tiling requires a tiling_host_entry C symbol")
        if self.workspace_bytes == "dynamic":
            if not isinstance(self.workspace_host_entry, str) or not _SYMBOL.fullmatch(self.workspace_host_entry):
                raise SchemaError("dynamic workspace requires a workspace_host_entry C symbol")
        elif type(self.workspace_bytes) is not int or self.workspace_bytes < 0:
            raise SchemaError("v2 workspace_bytes must be nonnegative integer or dynamic")
        elif self.workspace_host_entry is not None:
            raise SchemaError("fixed workspace must not declare workspace_host_entry")
        if self.task_type == "aic_aiv":
            if self.sync_policy != "cross_core_flags":
                raise SchemaError("aic_aiv requires explicit cross_core_flags synchronization")
        elif self.task_type != "aic_only" or self.sync_policy != "compiler_default":
            raise SchemaError("unsupported v2 task_type/sync_policy")
        parameters = self.tensor_parameters
        if (not parameters or any(not isinstance(p, TensorParameter) for p in parameters)
                or len({p.name for p in parameters}) != len(parameters)):
            raise SchemaError("v2 requires unique ordered tensor_parameters")
        # CANN-generated ACLNN APIs place input tensors before output tensors.
        roles = [p.role for p in parameters]
        if roles != sorted(roles):
            raise SchemaError("ACLNN tensor_parameters must declare inputs before outputs")
        constraints = self.shape_constraints
        symbols = {d for p in parameters for d in p.shape if isinstance(d, str)}
        if (any(not isinstance(c, ShapeConstraint) for c in constraints)
                or len({c.name for c in constraints}) != len(constraints)
                or {c.name for c in constraints} != symbols):
            raise SchemaError("shape_constraints must bind every tensor shape symbol exactly once")

    @property
    def tensor_parameter_names(self):
        """ACLNN stage-one order; the frozen probe ABI supplies the v1 default."""
        return tuple(p.name for p in self.tensor_parameters) if self.schema_version == 2 else ("a", "b", "c")

    @property
    def device_parameter_order(self):
        """Device GM arguments; CANN supplies workspace and tiling via executor."""
        return self.tensor_parameter_names + ("workspace", "tiling")

    def validate_tensor_metadata(self, *, shapes, dtypes):
        """Reject unsupported v2 metadata before dispatch can resolve/build a unit.

        No tensor library import, device access, source read or compiler discovery
        is required. Returned dimensions may also be used by host preparation.
        """
        if self.schema_version != 2:
            raise SchemaError("tensor metadata validation requires recipe v2")
        names = set(self.tensor_parameter_names)
        if set(shapes) != names or set(dtypes) != names:
            raise SchemaError("dispatch tensor arguments must match recipe parameters: " + ", ".join(self.tensor_parameter_names))
        dimensions = {}
        for parameter in self.tensor_parameters:
            if dtypes[parameter.name] != parameter.dtype:
                raise SchemaError("dispatch dtype for %s must be %s; got %s"
                                  % (parameter.name, parameter.dtype, dtypes[parameter.name]))
            shape = tuple(shapes[parameter.name])
            if len(shape) != len(parameter.shape):
                raise SchemaError("dispatch shape for %s requires rank %d; got %s"
                                  % (parameter.name, len(parameter.shape), shape))
            for actual, declared in zip(shape, parameter.shape):
                if type(actual) is not int or actual < 1:
                    raise SchemaError("dispatch shape for %s requires positive integer dimensions" % parameter.name)
                if isinstance(declared, str):
                    if declared in dimensions and dimensions[declared] != actual:
                        raise SchemaError("dispatch shape symbol %s is inconsistent across tensor arguments" % declared)
                    dimensions[declared] = actual
                elif actual != declared:
                    raise SchemaError("dispatch shape for %s requires dimension %d; got %d"
                                      % (parameter.name, declared, actual))
        for constraint in self.shape_constraints:
            actual = dimensions[constraint.name]
            if not constraint.minimum <= actual <= constraint.maximum or actual % constraint.multiple_of:
                raise SchemaError("dispatch shape %s=%d is outside declared range [%d, %d], multiple_of=%d"
                                  % (constraint.name, actual, constraint.minimum,
                                     constraint.maximum, constraint.multiple_of))
        return dimensions

    def to_dict(self):
        value = asdict(self)
        value["source"] = self.source.to_dict()
        for name in ("sources", "headers", "host_sources"):
            value[name] = [r.to_dict() for r in getattr(self, name)]
        if self.schema_version == 1:
            # Preserve all existing v1 identities, including op-package recipes.
            for name in ("tensor_parameters", "shape_constraints", "tiling_host_entry", "workspace_host_entry"):
                value.pop(name)
        # Preserve the 13-I serialization/fingerprints for host_shared recipes.
        if self.output_kind == "host_shared":
            for name in ("host_sources", "package_vendor", "package_op_type"):
                value.pop(name)
        return value

    @classmethod
    def from_dict(cls, value):
        try:
            data = dict(value)
            data["source"] = SourceRef.from_dict(data["source"])
            for name in ("sources", "headers"):
                data[name] = tuple(SourceRef.from_dict(r) for r in data[name])
            data["host_sources"] = tuple(SourceRef.from_dict(r) for r in data.get("host_sources", ()))
            data["dependencies"] = tuple(BuildDependency(**d) for d in data["dependencies"])
            data["tensor_parameters"] = tuple(TensorParameter(**p) for p in data.get("tensor_parameters", ()))
            data["shape_constraints"] = tuple(ShapeConstraint(**c) for c in data.get("shape_constraints", ()))
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
class BuildRecipe(_Recipe):
    """Public Ascend C contract: explicit Tensor ABI and host tiling only.

    Legacy records are decoded only by the internal artifact compatibility
    reader. New declarations must never silently produce the old probe ABI.
    """

    output_kind: str = "op_package"
    output_adapter: str = "ascendc.op_package.v1"
    schema_version: int = 2

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise SchemaError("public BuildRecipe requires schema_version=2; legacy v1 is read-only compatibility")
        super().__post_init__()


def _recipe_from_artifact(value):
    """Decode an existing identity without upgrading its serialized contract."""
    if value.get("schema_version") == 1:
        return _Recipe.from_dict(value)
    return BuildRecipe.from_dict(value)


@dataclass(frozen=True)
class _BuildConfig:
    """Select an existing variant before resolving dependencies or declaring a unit."""

    implementation: str = "native"
    output_kind: str = "host_shared"
    target_soc: str = "Ascend950"
    overrides: Tuple[Tuple[str, str], ...] = ()
    # A missing selector preserves the original probe CLI/config behavior.
    # ``all`` selects the frozen production registry, including each entry's
    # own implementation rather than the global implementation default.
    recipe_id: str = "matmul_probe"

    def __post_init__(self):
        try:
            pairs = tuple(tuple(p) for p in self.overrides)
        except TypeError as error:
            raise SchemaError("invalid Ascend C build overrides") from error
        if (self.implementation not in {"native", "catlass"}
                or self.output_kind not in {"host_shared", "op_package"}
                or not isinstance(self.target_soc, str) or not self.target_soc
                or not isinstance(self.recipe_id, str) or not self.recipe_id
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


@dataclass(frozen=True)
class AscendCBuildConfig(_BuildConfig):
    """Select a v2 ACLNN recipe; raw host launch is an internal test facility."""

    output_kind: str = "op_package"

    def __post_init__(self):
        super().__post_init__()
        if self.output_kind != "op_package":
            raise SchemaError("public Ascend C configuration requires v2 op_package; host_shared is internal legacy")

    def select(self, recipes, recipe_id):
        recipe = super().select(recipes, recipe_id)
        if not isinstance(recipe, BuildRecipe) or recipe.schema_version != 2:
            raise SchemaError("public Ascend C configuration cannot select a legacy v1 recipe")
        return recipe


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
    recipe: _Recipe
    source_root: Path
    dependencies: Tuple[ResolvedDependency, ...]
    toolchain_fingerprint: str
    build_files: Tuple[Tuple[str, str], ...]

    def __post_init__(self):
        if not isinstance(self.recipe, _Recipe):
            raise SchemaError("resolved recipe requires BuildRecipe")
        object.__setattr__(self, "source_root", Path(self.source_root))
        dependencies = tuple(self.dependencies)
        if tuple(d.declaration for d in dependencies) != self.recipe.dependencies:
            raise SchemaError("resolved dependencies must bind the recipe dependency closure")
        object.__setattr__(self, "dependencies", dependencies)
        _sha256("toolchain_fingerprint", self.toolchain_fingerprint)
        files = tuple((_path(p), _sha256("build input digest", h)) for p, h in self.build_files)
        expected = _build_inputs(self.recipe)
        if {p for p, _ in files} != set(expected) or len(files) != len(expected):
            raise SchemaError("resolved recipe must bind all output adapter CMake inputs")
        object.__setattr__(self, "build_files", files)

    def to_dict(self):
        return dict(recipe=self.recipe.to_dict(), dependencies=[d.to_dict() for d in self.dependencies],
                    toolchain_fingerprint=self.toolchain_fingerprint, build_files=list(self.build_files))

    @property
    def fingerprint(self):
        return _canonical_hash(self.to_dict())


def _build_inputs(recipe):
    if recipe.output_kind == "host_shared":
        return ("csrc/CMakeLists.txt", "csrc/cmake/HostShared.cmake")
    if recipe.schema_version == 1:
        # Internal frozen fixtures retain their original paths and identity.
        return ("csrc/CMakeLists.txt", "csrc/cmake/OpPackage.cmake",
                "csrc/op_package/host/CMakeLists.txt", "csrc/op_package/kernel/CMakeLists.txt")
    return ("csrc/CMakeLists.txt", "csrc/cmake/OpPackage.cmake",
            "csrc/cmake/op_package/host/CMakeLists.txt", "csrc/cmake/op_package/kernel/CMakeLists.txt")


def _resolve_build_recipe(recipe: _Recipe, *, source_root: Path,
                         toolchain_fingerprint: str, dependency_roots: Optional[Mapping[str, Path]] = None):
    """Read selected inputs, verify pinned commits and hash actual header content.

    Native resolution never looks up CATLASS, runs Git, or reads its directory.
    Toolchain discovery is a separate explicit operation, not a resolver side effect.
    """
    import subprocess

    root = Path(source_root).resolve(strict=True)
    for ref in recipe.sources + recipe.headers + recipe.host_sources:
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
    build_files = tuple((p, hashlib.sha256((root / p).read_bytes()).hexdigest()) for p in _build_inputs(recipe))
    return ResolvedBuildRecipe(recipe, root, tuple(dependencies), toolchain_fingerprint, build_files)


def resolve_build_recipe(recipe: BuildRecipe, *, source_root: Path,
                         toolchain_fingerprint: str, dependency_roots: Optional[Mapping[str, Path]] = None):
    """Resolve only a public v2 contract; old artifacts are read-only."""
    if not isinstance(recipe, BuildRecipe) or recipe.schema_version != 2:
        raise SchemaError("resolve_build_recipe requires a public v2 recipe; legacy artifacts are read-only")
    return _resolve_build_recipe(recipe, source_root=source_root,
        toolchain_fingerprint=toolchain_fingerprint, dependency_roots=dependency_roots)


__all__ = ["BuildRecipe", "BuildDependency", "AscendCBuildConfig", "ResolvedBuildRecipe",
           "load_build_recipes", "resolve_build_recipe"]
