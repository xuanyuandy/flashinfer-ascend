"""Compatibility exports for the fused_moe domain."""
from . import fused_moe as _domain
globals().update({name: value for name, value in vars(_domain).items() if not name.startswith("__")})
