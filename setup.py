"""Production AOT wheels by default; explicit source-only opt-out.

FLASHINFER_AOT_CONFIG=<json> FLASHINFER_AOT_SOCS=ascend950,... builds packages
through the same recipe driver/store. FLASHINFER_AOT_PREBUILT=<release-dir>
packages already verified exports without requiring a compiler.
An ordinary wheel build includes every registered production op-package.
FLASHINFER_AOT_ALL=0 builds a source-only wheel. Explicit ALL=1, CONFIG and
PREBUILT modes remain mutually exclusive. No installer runs.
"""

import os
from pathlib import Path
import shutil

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


def aot_mode():
    all_value = os.environ.get("FLASHINFER_AOT_ALL")
    if all_value not in {None, "", "0", "1"}:
        raise ValueError("FLASHINFER_AOT_ALL must be 0 or 1")
    modes = [name for name, enabled in (
        ("FLASHINFER_AOT_ALL", all_value == "1"),
        ("FLASHINFER_AOT_CONFIG", os.environ.get("FLASHINFER_AOT_CONFIG")),
        ("FLASHINFER_AOT_PREBUILT", os.environ.get("FLASHINFER_AOT_PREBUILT")),
    ) if enabled]
    if len(modes) > 1:
        raise ValueError("FLASHINFER_AOT_ALL, FLASHINFER_AOT_CONFIG and "
                         "FLASHINFER_AOT_PREBUILT are mutually exclusive")
    if modes:
        return modes[0]
    return None if all_value in {"", "0"} else "FLASHINFER_AOT_ALL"


def aot_requested():
    return aot_mode() is not None


def build_catlass_root():
    explicit = os.environ.get("FLASHINFER_CATLASS_ROOT")
    if explicit:
        return explicit
    root = Path(__file__).resolve().parent
    for location in (root / "3rdparty/catlass", root.parent / "catlass-ref"):
        if (location / "include").is_dir():
            return str(location)
    return None


class AOTDistribution(Distribution):
    def has_ext_modules(self):
        # ctypes host ELFs and NPU binaries are platform-specific, but have no
        # CPython extension ABI. AOTWheel supplies the corresponding wheel tag.
        return aot_requested() or super().has_ext_modules()


class AOTWheel(bdist_wheel):
    def get_tag(self):
        python, abi, platform = super().get_tag()
        return ("py3", "none", platform) if aot_requested() else (python, abi, platform)


def build_inputs():
    for path in sorted(Path("csrc").rglob("*")):
        if path.is_file() and (path.suffix in {".cpp", ".h", ".inc", ".cmake", ".json"}
                               or path.name == "CMakeLists.txt"):
            yield path


class BuildPy(build_py):
    def run(self):
        mode = aot_mode()
        super().run()
        root = Path(self.build_lib) / "flashinfer/data/build"
        if root.exists():
            shutil.rmtree(root)
        for path in build_inputs():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        config = os.environ.get("FLASHINFER_AOT_CONFIG")
        prebuilt = os.environ.get("FLASHINFER_AOT_PREBUILT")
        destination = Path(self.build_lib) / "flashinfer/data/op_packages"
        native_destination = Path(self.build_lib) / "flashinfer/data/aclnn"
        if native_destination.exists():
            shutil.rmtree(native_destination)
        # Reusing a build directory must never contaminate a later pure wheel.
        if destination.exists():
            shutil.rmtree(destination)
        if mode in {"FLASHINFER_AOT_ALL", "FLASHINFER_AOT_CONFIG"}:
            from flashinfer.aot.build import build_production_packages, compile_and_package
            from flashinfer.aclnn.build import build_adapter

            build_adapter(native_destination)
            native_log = native_destination / "build.log"
            if native_log.is_file():
                shutil.move(native_log, Path(self.build_lib).parent / "aclnn-native-build.log")

            targets = tuple(s.strip() for s in os.environ.get("FLASHINFER_AOT_SOCS", "").split(",") if s.strip())
            options = dict(output_dir=destination, targets=targets,
                cache_dir=os.environ.get("FLASHINFER_AOT_CACHE_DIR"),
                cann_root=os.environ.get("ASCEND_HOME_PATH"),
                catlass_root=build_catlass_root())
            if mode == "FLASHINFER_AOT_ALL":
                build_production_packages(**options)
            else:
                compile_and_package(config, **options)
        elif prebuilt:
            from flashinfer.aot.build import export_op_package
            from flashinfer.aclnn.build import bundle_adapter

            # PREBUILT remains compiler-free, but the single execution path
            # requires a compatible prebuilt Torch bridge as well as packages.
            bundle_adapter(native_destination)
            release = Path(prebuilt).resolve(strict=True)
            packages = (release,) if (release / "manifest.json").is_file() else tuple(
                p for p in sorted(release.iterdir()) if p.is_dir() and (p / "manifest.json").is_file())
            if not packages:
                raise ValueError("FLASHINFER_AOT_PREBUILT contains no verified op-package releases")
            for package in packages:
                export_op_package(package, destination)
        self._aot_outputs = [str(p) for p in destination.rglob("*") if p.is_file()] if destination.exists() else []
        self._native_outputs = ([str(p) for p in native_destination.iterdir()
                                 if p.name in {"libflashinfer_aclnn.so", "manifest.json"}]
                                if native_destination.exists() else [])

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + [
            str(Path(self.build_lib) / "flashinfer/data/build" / p) for p in build_inputs()] + getattr(self, "_aot_outputs", []) + getattr(self, "_native_outputs", [])


setup(cmdclass={"build_py": BuildPy, "bdist_wheel": AOTWheel}, distclass=AOTDistribution)
