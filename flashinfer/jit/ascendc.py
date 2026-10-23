"""Explicit CANN discovery and the shared host_shared compiler driver.

There is no SDK probing at module scope, backend construction or dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from flashinfer.runtime import SchemaError

from .core import _canonical_hash
from .errors import JitCompilationError
from .recipe import ResolvedBuildRecipe, _header_tree, _source_bytes


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cann_identity(root):
    files = list((root / "share/info").glob("*/version.info"))
    files += list((root / "compiler/tikcpp/ascendc_kernel_cmake").rglob("*.cmake"))
    files += [root / "lib64/cmake/asc-config.cmake"]
    return _canonical_hash({p.relative_to(root).as_posix(): _digest(p) for p in sorted(files)})


@dataclass(frozen=True)
class AscendCToolchain:
    cann_root: Path
    compiler: Path
    cmake: Path
    host_compiler: Path
    identity: tuple

    @classmethod
    def discover(cls, cann_root=None):
        """Opt-in discovery for a real build request, never called by selection."""
        location = cann_root or os.environ.get("ASCEND_HOME_PATH")
        if not location:
            raise SchemaError("Ascend C build requires CANN: supply cann_root or ASCEND_HOME_PATH")
        root = Path(location).resolve(strict=True)
        compiler = root / "bin/bisheng"
        cmake = shutil.which("cmake")
        host = shutil.which("c++")
        if not compiler.is_file() or cmake is None or host is None:
            raise SchemaError("Ascend C build requires CANN bisheng, cmake and a host C++ compiler")
        if not (root / "lib64/cmake/asc-config.cmake").is_file():
            raise SchemaError("this host_shared adapter requires CANN's ASC CMake package")
        tools = {"bisheng": compiler, "cmake": Path(cmake), "host_cxx": Path(host)}
        identity = {}
        for name, path in tools.items():
            identity[name] = _canonical_hash(dict(binary=_digest(path), version=subprocess.check_output(
                [str(path), "--version"], text=True, stderr=subprocess.STDOUT, timeout=30)))
            identity[name + "_binary"] = _digest(path)
        # Record installed CANN versions and the compiler/adapter configuration,
        # not just an environment-provided version string.
        identity["cann"] = _cann_identity(root)
        identity["host_abi"] = _canonical_hash(dict(machine=platform.machine(),
            python_abi=sys.implementation.cache_tag, libc=platform.libc_ver()))
        identity["driver"] = _digest(__file__)
        return cls(root, compiler, Path(cmake), Path(host), tuple(sorted(identity.items())))

    @property
    def fingerprint(self):
        return _canonical_hash(dict(self.identity))

    def validate(self):
        identity = dict(self.identity)
        for name, path in (("bisheng", self.compiler), ("cmake", self.cmake), ("host_cxx", self.host_compiler)):
            if _digest(path) != identity[name + "_binary"]:
                raise SchemaError("Ascend C toolchain changed after resolution: " + name)
        if _cann_identity(self.cann_root) != identity["cann"] or _digest(__file__) != identity["driver"]:
            raise SchemaError("CANN configuration or compiler driver changed after resolution")


def _cmake_value(value):
    # CMake bracket arguments preserve spaces, backslashes, quotes and '$'.
    marker = "="
    while "]" + marker + "]" in str(value):
        marker += "="
    return "[" + marker + "[" + str(value) + "]" + marker + "]"


def compile_host_shared(spec, resolved: ResolvedBuildRecipe, toolchain: AscendCToolchain, directory: Path):
    """Compile either implementation from a frozen recipe; owns no cache or loader."""
    recipe = resolved.recipe
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "build.log"
    command = ()
    scratch = directory / "ascendc-build"
    try:
        if (spec.generator_id != "ascendc" or not spec.is_compilation_unit
                or spec.source != recipe.source or spec.kernel_id != recipe.kernel_id
                or spec.target_soc != recipe.target_soc or spec.entry_points != (recipe.host_entry,)
                or dict(spec.input_fingerprints).get("resolved_recipe") != resolved.fingerprint
                or dict(spec.input_fingerprints).get("build_recipe") != recipe.fingerprint
                or spec.environment_fingerprint != toolchain.fingerprint
                or resolved.toolchain_fingerprint != toolchain.fingerprint):
            raise SchemaError("Ascend C spec does not bind the frozen recipe/toolchain")
        if (recipe.target_soc, recipe.npu_arch) != ("Ascend950", "dav-3510"):
            raise SchemaError("host_shared.v1 currently validates only Ascend950 / dav-3510")
        toolchain.validate()
        scratch.mkdir()
        inputs = scratch / "inputs"
        inputs.mkdir()
        for ref in recipe.sources + recipe.headers:
            path = inputs / ref.locator
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_source_bytes(resolved.source_root, ref))
        for locator, expected in resolved.build_files:
            payload = (resolved.source_root / locator).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise SchemaError("CMake build input changed after resolution: " + locator)
            path = inputs / locator
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        includes = [inputs / p for p in recipe.include_dirs]
        for dependency in resolved.dependencies:
            source = dependency.root / dependency.declaration.include_dir
            if _header_tree(source) != dependency.headers_digest:
                raise SchemaError("dependency headers changed after resolution: " + dependency.declaration.name)
            target = scratch / "dependencies" / dependency.declaration.name / "include"
            shutil.copytree(source, target)
            if _header_tree(target) != dependency.headers_digest:
                raise SchemaError("dependency changed while snapshotting: " + dependency.declaration.name)
            includes.append(target)
        definitions = list(recipe.definitions) + [
            "FI_DEVICE_ENTRY=" + recipe.device_entry, "FI_HOST_ENTRY=" + recipe.host_entry,
            "FI_BLOCK_DIM=" + str(recipe.block_dim)]
        variables = dict(FI_SOURCES=[inputs / s.locator for s in recipe.sources], FI_INCLUDES=includes,
                         FI_DEFINITIONS=definitions,
                         FI_COMPILE_OPTIONS=["--npu-arch=" + recipe.npu_arch, *recipe.compile_options, *spec.compile_options],
                         FI_LINK_OPTIONS=recipe.link_options, FI_OUTPUT=[directory / spec.output_path])
        request = scratch / "recipe.cmake"
        request.write_text("\n".join("set(%s %s)" % (k, " ".join(_cmake_value(v) for v in values))
                                     for k, values in variables.items()) + "\n", encoding="utf-8")
        output = (directory / spec.output_path).resolve()
        if not output.is_relative_to(directory) or output.is_relative_to(scratch):
            raise SchemaError("Ascend C output must stay outside private compiler scratch")
        output.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        # No implicit third-party include roots can leak into a native build.
        for key in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "LIBRARY_PATH",
                    "CXXFLAGS", "CFLAGS", "LDFLAGS", "ASCEND_CUSTOM_OPP_PATH"):
            env.pop(key, None)
        env["ASCEND_HOME_PATH"] = str(toolchain.cann_root)
        env["CMAKE_PREFIX_PATH"] = str(toolchain.cann_root / "lib64/cmake")
        commands = [
            [str(toolchain.cmake), "-S", str(inputs / "csrc"), "-B", str(scratch / "cmake"),
             "-DFI_RECIPE_FILE=" + str(request), "-DCMAKE_ASC_COMPILER=" + str(toolchain.compiler),
             "-DCMAKE_CXX_COMPILER=" + str(toolchain.host_compiler), "-DCMAKE_BUILD_TYPE=Release"],
            [str(toolchain.cmake), "--build", str(scratch / "cmake"), "--verbose", "-j2"],
        ]
        with log_path.open("w", encoding="utf-8") as log:
            for command in commands:
                log.write("command: " + json.dumps(command) + "\n")
                log.flush()
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        check=False, timeout=600)
                if result.returncode:
                    raise JitCompilationError(log_path.read_text(errors="replace")[-6000:],
                        kernel_id=spec.kernel_id, spec_fingerprint=spec.fingerprint,
                        command=command, returncode=result.returncode, log_path=log_path)
        # Keep a small auditable receipt. CMake trees can contain symlinks and
        # third-party headers; they are private scratch, never store payloads.
        stable_commands = []
        replacements = ((str(directory), "${BUILD_DIR}"), (str(toolchain.compiler), "${ASCENDC_COMPILER}"),
                        (str(toolchain.cmake), "${CMAKE}"), (str(toolchain.host_compiler), "${HOST_CXX}"))
        for command in commands:
            normalized = []
            for arg in command:
                for path, marker in replacements:
                    arg = arg.replace(path, marker)
                normalized.append(arg)
            stable_commands.append(normalized)
        (directory / "ascendc-recipe.json").write_text(json.dumps(dict(
            resolved=resolved.to_dict(), toolchain=dict(toolchain.identity), commands=stable_commands,
            compile_options=spec.compile_options), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except JitCompilationError:
        raise
    except Exception as error:
        with log_path.open("a", encoding="utf-8") as log:
            log.write("%s: %s\n" % (type(error).__name__, error))
        raise JitCompilationError(str(error), kernel_id=spec.kernel_id,
            spec_fingerprint=spec.fingerprint, command=command, log_path=log_path) from error
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


__all__ = ["AscendCToolchain"]
