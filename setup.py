"""Package JIT source inputs without compiling or including optional CATLASS."""

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


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

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + [
            str(Path(self.build_lib) / "flashinfer/data/build" / p) for p in build_inputs()]


setup(cmdclass={"build_py": BuildPy})
