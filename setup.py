"""Production AOT wheels by default; explicit source-only opt-out.

FLASHINFER_AOT_CONFIG=<json> FLASHINFER_AOT_SOCS=Ascend950 builds one family's packages
through the same recipe driver/store. FLASHINFER_AOT_PREBUILT=<release-dir>
packages already verified exports without requiring a compiler.
An ordinary wheel includes the local family: A2/A3 share the 910B distribution.
FLASHINFER_AOT_ALL=0 builds a source-only wheel. Explicit ALL=1, CONFIG and
PREBUILT modes remain mutually exclusive. No installer runs.
"""

import os
import json
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
    def parse_config_files(self, filenames=None, ignore_option_errors=False):
        super().parse_config_files(filenames=filenames, ignore_option_errors=ignore_option_errors)
        if aot_requested():
            from flashinfer._wheel import distribution_name
            self.metadata.name = distribution_name(self.wheel_target)

    @property
    def wheel_target(self):
        if not hasattr(self, "_wheel_target"):
            from flashinfer._wheel import wheel_target
            self._wheel_target = wheel_target()
        return self._wheel_target

    def has_ext_modules(self):
        # ctypes host ELFs and NPU binaries are platform-specific, but have no
        # CPython extension ABI. AOTWheel supplies the corresponding wheel tag.
        return aot_requested() or super().has_ext_modules()


class AOTWheel(bdist_wheel):
    def finalize_options(self):
        if aot_requested():
            from flashinfer._wheel import distribution_name
            self.distribution.metadata.name = distribution_name(self.distribution.wheel_target)
        super().finalize_options()

    def get_tag(self):
        python, abi, platform = super().get_tag()
        return ("py3", "none", platform) if aot_requested() else (python, abi, platform)


def build_inputs():
    for path in sorted(Path("csrc").rglob("*")):
        if (any(part in {"__pycache__", ".pytest_cache", ".git"} for part in path.parts)
                or path.suffix in {".pyc", ".pyo"}):
            continue
        if path.is_file() and (path.suffix in {".cpp", ".h", ".inc", ".cmake", ".json", ".py"}
                or path.name in {"CMakeLists.txt", "LICENSE"}
                or path.is_relative_to(Path("csrc/kernels"))):
            yield path


    import json
    seen = set()
    from flashinfer._resources import recipe_root
    for recipe_file in sorted(recipe_root(Path.cwd()).glob("*.json")):
        for recipe in json.loads(recipe_file.read_text()):
            for reference in recipe.get("extra_build_files", ()):
                path = Path(reference["locator"])
                if path.parts[0] == "flashinfer" and path not in seen:
                    if path.is_symlink() or not path.resolve().is_relative_to(Path.cwd().resolve()):
                        raise ValueError("build hook must remain inside the source checkout")
                    seen.add(path)
                    yield path


class BuildPy(build_py):
    def run(self):
        mode = aot_mode()
        target = self.distribution.wheel_target if mode else None
        from flashinfer.operators.authoring import generate
        generate(source_root=Path.cwd())
        package_output = Path(self.build_lib) / "flashinfer"
        package_source = Path(__file__).resolve().parent / "flashinfer"
        if package_output.exists():
            output = package_output.resolve()
            if output.is_relative_to(package_source) or package_source.is_relative_to(output):
                raise ValueError("build output must not overlap the package source directory")
            # build_py only overwrites existing modules; removed modules would
            # otherwise leak from an earlier build into a new wheel.
            shutil.rmtree(package_output)
        super().run()
        from flashinfer._resources import data_root, recipe_root
        resources = Path(self.build_lib) / "flashinfer/data"
        shutil.copytree("tools/metadata", resources, dirs_exist_ok=True)
        shutil.copytree(data_root(), resources, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("op_packages", "aclnn", "build"))
        root = Path(self.build_lib) / "flashinfer/data/build"
        if root.exists():
            shutil.rmtree(root)
        for path in build_inputs():
            build_target = root / path
            build_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, build_target)
        shutil.copytree(recipe_root(), root / "csrc/recipes", dirs_exist_ok=True)
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
            from flashinfer.aot.build import build_production_packages
            from flashinfer.aclnn.build import build_adapter
            from flashinfer.aot.declarations import freeze_production_kernels

            catlass = build_catlass_root()
            freeze_production_kernels(dependency_roots={"catlass": catlass} if catlass else None)
            shutil.copytree(data_root() / "kernels", resources / "kernels", dirs_exist_ok=True)

            build_adapter(native_destination)
            native_log = native_destination / "build.log"
            if native_log.is_file():
                shutil.move(native_log, Path(self.build_lib).parent / "aclnn-native-build.log")

            options = dict(output_dir=destination,
                cache_dir=os.environ.get("FLASHINFER_AOT_CACHE_DIR"),
                cann_root=os.environ.get("ASCEND_HOME_PATH"),
                catlass_root=build_catlass_root())
            if mode == "FLASHINFER_AOT_ALL":
                from flashinfer._wheel import production_kernel_ids
                kernel_ids = production_kernel_ids(target)
                build_production_packages(kernel_ids=kernel_ids, **options)
            else:
                from flashinfer.jit import AscendCBuildConfig
                from flashinfer._wheel import family_socs, production_kernel_ids
                from flashinfer.operators.implementation import implementations

                # CONFIG selects authored variants across the delivery family.
                # Every unit retains its own strict physical target binding.
                selected_config = AscendCBuildConfig.from_file(config)
                if selected_config.recipe_id == "all":
                    build_production_packages(kernel_ids=production_kernel_ids(target), **options)
                else:
                    implementation = dict(selected_config.overrides).get(
                        selected_config.recipe_id, selected_config.implementation)
                    selected = tuple(d.kernel_id for d in implementations(backend="ascendc")
                        if d.native.target_soc in family_socs(target)
                        and d.native.name == selected_config.recipe_id
                        and d.native.build.implementation == implementation)
                    if not selected:
                        raise ValueError("CONFIG has no production implementation in wheel family " + target)
                    build_production_packages(kernel_ids=selected, **options)
                from flashinfer.jit.cache import verify_build_directory
                releases = tuple(p for p in destination.iterdir()
                                 if p.is_dir() and (p / "manifest.json").is_file())
                for release in releases:
                    verify_build_directory(release)
                kernel_ids = tuple(json.loads((p / "manifest.json").read_text())["spec"]["kernel_id"]
                                   for p in releases)
        elif prebuilt:
            from flashinfer.aot.build import export_op_package
            from flashinfer.aclnn.build import bundle_adapter

            # PREBUILT remains compiler-free, but the single execution path
            # requires a compatible prebuilt Torch bridge as well as packages.
            bundle_adapter(native_destination)
            from flashinfer._wheel import prebuilt_packages, canonical_target, restore_prebuilt_specs
            packages = tuple(p for p in prebuilt_packages(prebuilt)
                if canonical_target(json.loads((p / "manifest.json").read_text())["spec"]["target_soc"]) == target)
            if not packages:
                raise ValueError("PREBUILT contains no op-package for wheel target " + target)
            for package in packages:
                export_op_package(package, destination)
            restore_prebuilt_specs(package_output, packages)
            kernel_ids = tuple(json.loads((p / "manifest.json").read_text())["spec"]["kernel_id"] for p in packages)
        if mode:
            from flashinfer._wheel import finalize_resources
            finalize_resources(package_output, target=target, kernel_ids=kernel_ids)
        self._aot_outputs = [str(p) for p in destination.rglob("*") if p.is_file()] if destination.exists() else []
        self._native_outputs = ([str(p) for p in native_destination.iterdir()
                                 if p.name in {"libflashinfer_aclnn.so", "manifest.json"}]
                                if native_destination.exists() else [])

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + [
            str(Path(self.build_lib) / "flashinfer/data/build" / p) for p in build_inputs()] + getattr(self, "_aot_outputs", []) + getattr(self, "_native_outputs", [])


setup(cmdclass={"build_py": BuildPy, "bdist_wheel": AOTWheel}, distclass=AOTDistribution)
