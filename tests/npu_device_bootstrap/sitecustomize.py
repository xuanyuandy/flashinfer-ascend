"""Optional regression-only default device, including generated Python probes.

Do not import optional providers here: import-safety tests must still observe
their normal imports. Restore TorchNPU's original initializer before any graph
tracing, and leave explicit device choices to the caller.
"""

import functools
import importlib.abc
import importlib.machinery
import os
import sys


_selected = os.environ.get("FLASHINFER_TEST_NPU_DEVICE")


class _Loader:
    def __init__(self, loader):
        self.loader = loader

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        original = module._lazy_init

        @functools.wraps(original)
        def initialize(*args, **kwargs):
            result = original(*args, **kwargs)
            if module._initialized:
                module._lazy_init = original
                module.set_device(int(_selected))
            return result

        module._lazy_init = initialize


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch_npu.npu":
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader)
            return spec


if _selected is not None:
    sys.meta_path.insert(0, _Finder())
