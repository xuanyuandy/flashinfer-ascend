"""Compatibility imports; Attention compilation protocols live in its domain."""

import sys
from flashinfer.attention import jit as _module

# Alias the existing modules so historical imports retain identical type objects.
for _name, _value in tuple(sys.modules.items()):
    if _name == _module.__name__ or _name.startswith(_module.__name__ + "."):
        sys.modules[__name__ + _name.removeprefix(_module.__name__)] = _value
