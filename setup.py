"""Pure source distribution by default; explicit per-SOC AOT release wheels.

FLASHINFER_AOT_CONFIG=<json> FLASHINFER_AOT_SOCS=ascend950,... builds packages
through the same recipe driver/store. FLASHINFER_AOT_PREBUILT=<release-dir>
packages already verified exports without requiring a compiler. No installer runs.
"""

import os
from pathlib import Path
import shutil

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


def aot_requested():
    return bool(os.environ.get("FLASHINFER_AOT_CONFIG") or os.environ.get("FLASHINFER_AOT_PREBUILT"))


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
    for directory in ("csrc", "kernels/ascendc"):
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file() and (path.suffix in {".cpp", ".h", ".inc", ".cmake", ".json"}
                                   or path.name == "CMakeLists.txt"):
                yield path


class BuildPy(build_py):
    def run(self):
        super().run()
        root = Path(self.build_lib) / "flashinfer/data/build"
        for path in build_inputs():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        config = os.environ.get("FLASHINFER_AOT_CONFIG")
        prebuilt = os.environ.get("FLASHINFER_AOT_PREBUILT")
        if config and prebuilt:
            raise ValueError("select FLASHINFER_AOT_CONFIG or FLASHINFER_AOT_PREBUILT")
        destination = Path(self.build_lib) / "flashinfer/data/op_packages"
        # Reusing a build directory must never contaminate a later pure wheel.
        if destination.exists():
            shutil.rmtree(destination)
        if config:
            from flashinfer.aot.build import compile_and_package

            targets = tuple(s.strip() for s in os.environ.get("FLASHINFER_AOT_SOCS", "").split(",") if s.strip())
            compile_and_package(config, output_dir=destination, targets=targets,
                cache_dir=os.environ.get("FLASHINFER_AOT_CACHE_DIR"),
                cann_root=os.environ.get("ASCEND_HOME_PATH"),
                catlass_root=os.environ.get("FLASHINFER_CATLASS_ROOT"))
        elif prebuilt:
            from flashinfer.aot.build import export_op_package

            release = Path(prebuilt).resolve(strict=True)
            packages = (release,) if (release / "manifest.json").is_file() else tuple(
                p for p in sorted(release.iterdir()) if p.is_dir() and (p / "manifest.json").is_file())
            if not packages:
                raise ValueError("FLASHINFER_AOT_PREBUILT contains no verified op-package releases")
            for package in packages:
                export_op_package(package, destination)
        self._aot_outputs = [str(p) for p in destination.rglob("*") if p.is_file()] if destination.exists() else []

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + [
            str(Path(self.build_lib) / "flashinfer/data/build" / p) for p in build_inputs()] + getattr(self, "_aot_outputs", [])


setup(cmdclass={"build_py": BuildPy, "bdist_wheel": AOTWheel}, distclass=AOTDistribution)
