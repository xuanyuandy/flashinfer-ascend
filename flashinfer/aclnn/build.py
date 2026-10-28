"""Explicit build of the Torch/ACLNN host adapter; never called by import/run."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def build_adapter(output_dir, *, source_root=None):
    import torch
    import torch_npu
    from torch.utils.cpp_extension import include_paths, library_paths

    root = Path(source_root) if source_root else Path(__file__).resolve().parents[2]
    source = root / "csrc/aclnn_torch_adapter/adapter.cpp"
    if not source.is_file():
        root = Path(__file__).resolve().parents[1] / "data/build"
        source = root / "csrc/aclnn_torch_adapter/adapter.cpp"
    cann = Path(os.environ["ASCEND_HOME_PATH"]).resolve(strict=True)
    npu = Path(torch_npu.__file__).resolve().parent
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("Building the ACLNN Torch adapter requires c++")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".aclnn-build-", dir=destination) as temporary:
        library = Path(temporary) / "libflashinfer_aclnn.so"
        command = [compiler, "-std=c++17", "-O2", "-g0", "-fPIC", "-shared",
                   "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch.compiled_with_cxx11_abi())), str(source),
                   "-o", str(library)]
        for include in include_paths() + [str(npu / "include"), str(cann / "include")]:
            command += ["-I", include]
        for directory in library_paths() + [str(npu / "lib"), str(cann / "lib64")]:
            command += ["-L", directory]
        command += ["-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_npu", "-lascendcl", "-lffi", "-ldl",
                    "-Wl,-rpath,$ORIGIN/../../../torch/lib:$ORIGIN/../../../torch_npu/lib"]
        log = destination / "build.log"
        with log.open("w") as stream:
            stream.write(json.dumps(command) + "\n")
            stream.flush()
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("ACLNN Torch adapter build failed; see " + str(log))
        metadata = dict(schema_version=1, library=library.name,
                        sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                        torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        manifest = Path(temporary) / "manifest.json"
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        # Rebuilding must not overwrite an inode mapped by an existing process.
        os.replace(library, destination / library.name)
        os.replace(manifest, destination / manifest.name)
    return tuple(destination / name for name in ("libflashinfer_aclnn.so", "manifest.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    options = parser.parse_args()
    for path in build_adapter(options.output_dir):
        print(path)
