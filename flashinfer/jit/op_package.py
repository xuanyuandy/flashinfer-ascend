"""Standard CANN op-package verification and explicit compiler-free ACLNN load.

No SDK discovery, optional imports, installers or filesystem work at import.
The release directory is a copy of a verified JitBuildStore generation, not a
second cache. Activate before the first custom-op execution; one package
identity is supported per process, including lazy activation after NPU init.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
from threading import RLock
from typing import Optional

from flashinfer.runtime import ArtifactFormat, SchemaError

from ._elf import elf_symbols
from .core import JitSpec, _canonical_hash
from .recipe import _Recipe, _recipe_from_artifact, _path


OP_PACKAGE_ADAPTER_HASH = _canonical_hash("ascendc.op_package.v1")


class OpPackageSymbolError(SchemaError):
    """An otherwise verified package cannot supply its required ACLNN entry."""


def package_target(soc):
    """Pure requested-product -> CANN compute-unit/architecture mapping.

    A family mapping does not assert successful compilation or device support.
    The full requested product is still bound into the recipe/spec/release.
    """
    key = soc.lower()
    known = {"ascend910b1": ("ascend910b", "dav-2201"),
             "ascend910_9391": ("ascend910_93", "dav-2201"),
             "ascend310p1": ("ascend310p", "dav-2002")}
    if re.fullmatch(r"ascend950[a-z0-9_]*", key):
        return "ascend950", "dav-3510"
    if key not in known:
        raise SchemaError("unsupported op-package SOC: " + soc)
    return known[key]


def _file(root, relative):
    _path(relative)
    root = Path(root).resolve(strict=True)
    path = root / relative
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise SchemaError("op-package path escapes its immutable root: " + relative)
    return path


def collect_package(spec, recipe, build, directory):
    """Collect only CPack's staged standard vendor tree and installer, never run it."""
    vendors = list(build.glob("_CPack_Packages/*/External/*.run/packages/vendors/" + recipe.package_vendor))
    installers = list(build.glob("*.run"))
    if len(vendors) != 1 or len(installers) != 1:
        raise SchemaError("CANN must produce one staged standard vendor tree and one .run")
    shutil.copytree(vendors[0].parent.parent, directory / "packages")
    shutil.copy2(installers[0], directory / "op-package.run")
    vendor = "packages/vendors/" + recipe.package_vendor
    data = dict(schema_version=1, output_adapter=recipe.output_adapter,
        spec_fingerprint=spec.fingerprint, recipe=recipe.to_dict(),
        vendor_path=vendor, api_library=vendor + "/op_api/lib/libcust_opapi.so",
        compute_unit=package_target(recipe.target_soc)[0], run_file="op-package.run")
    (directory / spec.output_path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    verify_op_package(directory, spec)


def verify_op_package(directory, spec):
    """Verify manifest binding, ACLNN host ELF, config references and NPU ET_EXEC.

    A direct device -c ET_REL and a host ET_DYN are both rejected as package
    kernels. Extensions and CANN's filename hash never select the loader.
    The surrounding store/release manifest verifies every auxiliary byte.
    """
    if (spec.generator_id != "ascendc" or spec.output_path != "op-package.json"
            or dict(spec.input_fingerprints).get("output_adapter") != OP_PACKAGE_ADAPTER_HASH):
        raise SchemaError("spec does not declare the op-package adapter")
    data = json.loads(_file(directory, spec.output_path).read_text())
    if set(data) != {"schema_version", "output_adapter", "spec_fingerprint", "recipe",
                     "vendor_path", "api_library", "compute_unit", "run_file"}:
        raise SchemaError("invalid op-package manifest fields")
    recipe = _recipe_from_artifact(data["recipe"])
    vendor = "packages/vendors/" + str(recipe.package_vendor)
    compute = package_target(recipe.target_soc)[0]
    if (data["schema_version"] != 1 or data["output_adapter"] != "ascendc.op_package.v1"
            or recipe.output_kind != "op_package" or data["spec_fingerprint"] != spec.fingerprint
            or recipe.target_soc != spec.target_soc or recipe.source != spec.source
            or recipe.kernel_id != spec.kernel_id or spec.entry_points != (recipe.host_entry,)
            or dict(spec.input_fingerprints).get("build_recipe") != recipe.fingerprint
            or data["compute_unit"] != compute or data["vendor_path"] != vendor
            or data["api_library"] != vendor + "/op_api/lib/libcust_opapi.so"
            or data["run_file"] != "op-package.run"):
        raise SchemaError("op-package manifest does not bind recipe/spec/layout")
    machines = {"x86_64": 62, "aarch64": 183}
    symbols = elf_symbols(_file(directory, data["api_library"]).read_bytes(),
        machine=machines[platform.machine()], elf_type=3, dynamic_only=True)
    for entry in (recipe.host_entry, recipe.host_entry + "GetWorkspaceSize"):
        if entry not in symbols:
            raise OpPackageSymbolError("op-package host ELF has no ACLNN symbol: " + entry)
    kernel = vendor + "/op_impl/ai_core/tbe/kernel/"
    config = json.loads(_file(directory, kernel + "config/" + compute + "/binary_info_config.json").read_text())
    if set(config) != {recipe.package_op_type}:
        raise SchemaError("binary_info_config must identify exactly the declared op type")
    binaries = config[recipe.package_op_type]["binaryList"]
    if not binaries:
        raise SchemaError("op-package contains no precompiled kernel")
    for item in binaries:
        for key in ("binPath", "jsonPath"):
            if not item[key].startswith(compute + "/"):
                raise SchemaError("binary_info_config references a different compute unit")
        binary = _file(directory, kernel + item["binPath"]).read_bytes()
        entries = elf_symbols(binary, machine=0x1029, elf_type=2)
        metadata = json.loads(_file(directory, kernel + item["jsonPath"]).read_text())
        # CANN renames the source entry to <OpType>_<digest>_<tiling-key>.
        # Check the actual names recorded in its binary metadata against ELF,
        # instead of treating the source C symbol as the packaged entry.
        generated = [k["kernelName"] for k in metadata.get("kernelList", ())]
        mixed = recipe.schema_version == 2 and recipe.task_type == "aic_aiv"
        required_entries = ([name + suffix for name in generated for suffix in ("_mix_aic", "_mix_aiv")]
                            if mixed else generated)
        if (not generated or any(not name.startswith(recipe.package_op_type + "_") for name in generated)
                or any(name not in entries for name in required_entries)):
            raise SchemaError("op-package NPU ET_EXEC does not define its CANN metadata entries")
        if mixed:
            if (metadata.get("binFileSuffix") != ".o" or metadata.get("coreType") != "MIX"
                    or metadata.get("magic") != "RT_DEV_BINARY_MAGIC_ELF"
                    or metadata.get("intercoreSync") != 1
                    or any(k.get("kernelType") not in {"MIX_AIC", "MIX_AIV"}
                           or k.get("crossCoreSync") != 1
                           for k in metadata["kernelList"])):
                raise SchemaError("op-package kernel metadata does not declare the v2 synchronized AIC+AIV executable contract")
        else:
            # v1 retains its AIC-only metadata rules unchanged. 310P has unified
            # AiCore metadata without the A2/A3/A5 core_type field.
            magic = "RT_DEV_BINARY_MAGIC_ELF" if compute == "ascend310p" else "RT_DEV_BINARY_MAGIC_ELF_AICUBE"
            if (metadata.get("binFileSuffix") != ".o" or metadata.get("coreType") != "AiCore"
                    or metadata.get("magic") != magic
                    or (compute != "ascend310p" and metadata.get("core_type") != "AIC")):
                raise SchemaError("op-package kernel metadata does not declare the target's AIC executable contract")
        if recipe.schema_version == 2:
            support = metadata.get("supportInfo", {})
            if support.get("opMode") != "dynamic":
                raise SchemaError("v2 package metadata must declare dynamic host tiling")
            for role, key in (("input", "inputs"), ("output", "outputs")):
                declared = [(p.name, p.dtype) for p in recipe.tensor_parameters if p.role == role]
                packaged = [(p.get("name"), p.get("dtype")) for p in support.get(key, ())]
                if packaged != declared:
                    raise SchemaError("v2 package tensor parameter order/types differ from recipe: " + role)
            if (recipe.workspace_bytes == "dynamic"
                    and metadata.get("workspace", {}).get("size") != [-1]):
                raise SchemaError("v2 dynamic workspace must have a host-computed metadata size")
        if metadata.get("sha256") != hashlib.sha256(binary).hexdigest():
            raise SchemaError("CANN kernel metadata digest does not match its NPU executable")
    _file(directory, data["run_file"])
    return data, recipe


_load_lock = RLock()
_active_package = None


@dataclass(frozen=True)
class OpPackageModule:
    """Owns ACLNN function pointers; caller owns ACL tensors, stream and workspace.

    get_workspace_size(*tensors) accepts recipe-ordered aclTensor* and returns
    (bytes, aclOpExecutor*). Named arguments are reordered from the same recipe.
    launch(workspace,bytes,executor,stream) is stage two. The host tiling callback
    computes workspace and writes the tiling data: CANN's executor transports
    that data as the device kernel's last GM buffer argument, after workspace.
    It is deliberately not an extra aclTensor argument to GetWorkspaceSize.
    Both stages raise on any nonzero CANN status. Successful dlopen alone does
    not claim execution or numerical correctness.
    """

    recipe: _Recipe
    vendor_path: Path
    api: object
    # Retain the directory descriptor backing CANN's dot-free /proc alias for
    # the active module's process lifetime. No second copy/cache is created.
    registration_path: Optional[Path] = None
    _registration_fd: Optional[int] = field(default=None, repr=False, compare=False)

    def get_workspace_size(self, *tensors, **named_tensors):
        import ctypes as C

        names = self.recipe.tensor_parameter_names
        if named_tensors:
            if tensors or set(named_tensors) != set(names):
                raise SchemaError("ACLNN tensor arguments must match recipe order: " + ", ".join(names))
            tensors = tuple(named_tensors[name] for name in names)
        if len(tensors) != len(names):
            raise SchemaError("ACLNN requires %d tensor arguments (%s); got %d"
                              % (len(names), ", ".join(names), len(tensors)))
        size, executor = C.c_uint64(), C.c_void_p()
        status = getattr(self.api, self.recipe.host_entry + "GetWorkspaceSize")(
            *tensors, C.byref(size), C.byref(executor))
        if status or not executor.value:
            raise RuntimeError("%sGetWorkspaceSize returned %s; executor=%s"
                               % (self.recipe.host_entry, status, executor.value))
        return size.value, executor

    def launch(self, workspace, workspace_bytes, executor, stream):
        status = getattr(self.api, self.recipe.host_entry)(workspace, workspace_bytes, executor, stream)
        if status:
            raise RuntimeError("%s returned %s" % (self.recipe.host_entry, status))


def load_op_package(directory, *, target_soc):
    """Load a verified release/cache generation without sources or a compiler.

    Set ASCEND_CUSTOM_OPP_PATH before custom-op execution; no .run installation.
    CANN caches vendor registrations, so changing packages in-process is rejected.
    CANN runtime/driver shared libraries must be on the library search path.
    Known gap: this loader does not compare the current CANN version with the
    recorded build environment; target/spec checks are not that comparison.
    """
    import ctypes as C
    from .cache import verify_build_directory

    directory = Path(directory).resolve(strict=True)
    record = verify_build_directory(directory)
    if record.artifact.format != ArtifactFormat.ASCENDC_OP_PACKAGE or record.artifact.target_soc != target_soc:
        raise SchemaError("release is not an op-package for the requested SOC")
    spec = JitSpec.from_dict(json.loads((directory / "manifest.json").read_text())["spec"])
    data, recipe = verify_op_package(directory, spec)
    global _active_package
    with _load_lock:
        identity = record.fingerprint
        if _active_package is not None:
            if _active_package[0] != identity:
                raise SchemaError("CANN op-package registration is process-scoped; load another package in a fresh process")
            return _active_package[1]
        vendor = _file(directory, data["vendor_path"])
        previous = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
        registration_fd = None
        registration_path = vendor
        try:
            if "." in str(vendor):
                # CANN 9.2's NnopbaseReadJsonConfig derives the companion JSON
                # from the first '.' in a full binary path. Ordinary JIT store
                # paths (.objects, ~/.cache) consequently become e.g. jit/.json.
                # Expose the SAME verified directory through a stable Linux fd;
                # preserve the vendor basename, which CANN uses as its identity.
                # Keep this fd alive with the process-scoped active package.
                registration_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                registration_path = (Path("/proc") / str(os.getpid()) / "fd"
                                     / str(registration_fd) / data["vendor_path"])
                if not registration_path.is_dir():
                    raise SchemaError("CANN op-package dotted paths require Linux /proc directory fd access")
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(registration_path) + (os.pathsep + previous if previous else "")
            api = C.CDLL(str(_file(directory, data["api_library"])), mode=C.RTLD_LOCAL)
            workspace = getattr(api, recipe.host_entry + "GetWorkspaceSize")
            workspace.argtypes = ([C.c_void_p] * len(recipe.tensor_parameter_names)
                                  + [C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)])
            workspace.restype = C.c_int
            launch = getattr(api, recipe.host_entry)
            launch.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
            launch.restype = C.c_int
        except Exception:
            if registration_fd is not None:
                os.close(registration_fd)
            if previous is None:
                os.environ.pop("ASCEND_CUSTOM_OPP_PATH", None)
            else:
                os.environ["ASCEND_CUSTOM_OPP_PATH"] = previous
            raise
        module = OpPackageModule(recipe, vendor, api, registration_path, registration_fd)
        _active_package = identity, module
        return module


__all__ = ["OpPackageModule", "OpPackageSymbolError", "load_op_package"]
