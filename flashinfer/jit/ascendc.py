"""Explicit CANN discovery and the shared BuildRecipe compiler driver.

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
    files += list((root / "compiler/tikcpp/ascendc_kernel_cmake").rglob("*.py"))
    files += list((root / "compiler/tikcpp/ascendc_kernel_cmake").rglob("*.sh"))
    files += [root / "lib64/cmake/asc-config.cmake"]
    if (root / "tools/opbuild/op_build").is_file():
        files += [root / "tools/opbuild/op_build"]
    return _canonical_hash({p.relative_to(root).as_posix(): _digest(p) for p in sorted(files)})


def _driver_identity():
    root = Path(__file__).parent
    return _canonical_hash({p: _digest(root / p) for p in ("ascendc.py", "op_package.py")})


def _host_standard_includes(compiler):
    """Use the selected host compiler's standard library, never ambient CPATH.

    A partially installed newer GCC can make bisheng select a header directory
    that has no C++ headers. These compiler-owned roots are explicit toolchain
    inputs, separate from recipe-owned sources and third-party dependencies.
    """
    environment = dict(os.environ)
    for name in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "CXXFLAGS"):
        environment.pop(name, None)
    output = subprocess.check_output([str(compiler), "-E", "-x", "c++", "-v", "-"],
        input="", text=True, stderr=subprocess.STDOUT, env=environment, timeout=30)
    search = output.split('#include <...> search starts here:', 1)[-1].split('End of search list.', 1)[0]
    directories = tuple(Path(line.strip()) for line in search.splitlines() if line.strip().startswith('/'))
    standard = tuple(p for p in directories if (p / "cstdint").is_file() or (p / "bits/c++config.h").is_file())
    return tuple(dict.fromkeys(standard + tuple(p for p in directories if p.name == "backward" and p.parent in standard)))


def _standard_headers_identity(directories):
    files = {p.resolve() for directory in directories for p in Path(directory).rglob('*') if p.is_file()}
    return _canonical_hash(dict(roots=[str(p) for p in directories],
                                files={str(p): _digest(p) for p in sorted(files)}))


@dataclass(frozen=True)
class AscendCToolchain:
    cann_root: Path
    compiler: Path
    cmake: Path
    host_compiler: Path
    identity: tuple
    host_standard_includes: tuple = ()

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
        identity["driver"] = _driver_identity()
        standard_includes = _host_standard_includes(host)
        identity["host_cxx_headers"] = _standard_headers_identity(standard_includes)
        return cls(root, compiler, Path(cmake), Path(host), tuple(sorted(identity.items())), standard_includes)

    @property
    def fingerprint(self):
        return _canonical_hash(dict(self.identity))

    def validate(self):
        identity = dict(self.identity)
        for name, path in (("bisheng", self.compiler), ("cmake", self.cmake), ("host_cxx", self.host_compiler)):
            if _digest(path) != identity[name + "_binary"]:
                raise SchemaError("Ascend C toolchain changed after resolution: " + name)
        if _cann_identity(self.cann_root) != identity["cann"] or _driver_identity() != identity["driver"]:
            raise SchemaError("CANN configuration or compiler driver changed after resolution")
        if (self.host_standard_includes and
                _standard_headers_identity(self.host_standard_includes) != identity.get("host_cxx_headers")):
            raise SchemaError("host C++ standard headers changed after resolution")


def _cmake_value(value):
    # CMake bracket arguments preserve spaces, backslashes, quotes and '$'.
    marker = "="
    while "]" + marker + "]" in str(value):
        marker += "="
    return "[" + marker + "[" + str(value) + "]" + marker + "]"


def compile_recipe(spec, resolved: ResolvedBuildRecipe, toolchain: AscendCToolchain, directory: Path):
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
        if recipe.output_kind == "host_shared":
            if (recipe.target_soc, recipe.npu_arch) != ("Ascend950", "dav-3510"):
                raise SchemaError("host_shared.v1 currently validates only Ascend950 / dav-3510")
        else:
            from .op_package import package_target

            compute_unit, architecture = package_target(recipe.target_soc)
            if architecture != recipe.npu_arch:
                raise SchemaError("op-package target_soc/npu_arch mismatch")
            if len(recipe.sources) != 1:
                raise SchemaError("op_package.v1 requires one device translation unit")
        toolchain.validate()
        scratch.mkdir()
        inputs = scratch / "inputs"
        inputs.mkdir()
        for ref in recipe.sources + recipe.headers + recipe.host_sources:
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
        options = list(recipe.compile_options) + list(spec.compile_options)
        options.extend("-I" + str(path) for path in toolchain.host_standard_includes)
        if recipe.output_kind == "host_shared":
            options.insert(0, "--npu-arch=" + recipe.npu_arch)
        else:
            definitions += ["FI_OP_PACKAGE=1", "FI_PACKAGE_OP_TYPE=" + recipe.package_op_type,
                            "FI_COMPUTE_UNIT=" + compute_unit]
        variables = dict(FI_SOURCES=[inputs / s.locator for s in recipe.sources], FI_INCLUDES=includes,
                         FI_DEFINITIONS=definitions,
                         FI_COMPILE_OPTIONS=options, FI_OUTPUT_KIND=[recipe.output_kind],
                         FI_LINK_OPTIONS=recipe.link_options, FI_OUTPUT=[directory / spec.output_path])
        if recipe.output_kind == "op_package":
            variables.update(FI_HOST_SOURCES=[inputs / s.locator for s in recipe.host_sources],
                FI_PACKAGE_VENDOR=[recipe.package_vendor], FI_PACKAGE_OP_TYPE=[recipe.package_op_type],
                FI_INPUT_ROOT=[inputs])
            if recipe.cmake_entry is not None:
                variables["FI_OPERATOR_CMAKE"] = [inputs / recipe.cmake_entry]
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
        cmake_source = inputs / ("csrc/cmake/operator" if recipe.extra_build_files else "csrc")
        commands = [
            [str(toolchain.cmake), "-S", str(cmake_source), "-B", str(scratch / "cmake"),
             "-DFI_RECIPE_FILE=" + str(request), "-DCMAKE_ASC_COMPILER=" + str(toolchain.compiler),
             "-DCMAKE_CXX_COMPILER=" + str(toolchain.host_compiler), "-DCMAKE_BUILD_TYPE=Release"],
            [str(toolchain.cmake), "--build", str(scratch / "cmake"), "--verbose", "-j2"],
        ]
        if recipe.output_kind == "op_package":
            commands[0] += ["-DASCEND_CANN_PACKAGE_PATH=" + str(toolchain.cann_root),
                "-DASCEND_COMPUTE_UNIT=" + compute_unit, "-DASCEND_PYTHON_EXECUTABLE=" + sys.executable,
                "-DENABLE_SOURCE_PACKAGE=OFF", "-DENABLE_BINARY_PACKAGE=ON",
                "-DASCEND_SKIP_FAILED_COMPUTE_UNIT=OFF", "-Dvendor_name=" + recipe.package_vendor,
                "-DCMAKE_INSTALL_PREFIX=" + str(scratch / "cmake")]
            commands.append([str(toolchain.cmake), "--build", str(scratch / "cmake"),
                             "--target", "binary", "package", "--verbose", "-j2"])
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
        if recipe.output_kind == "op_package":
            from .op_package import collect_package

            collect_package(spec, recipe, scratch / "cmake", directory)
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
