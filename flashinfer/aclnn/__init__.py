"""Default Torch/ACLNN execution used automatically by public operator calls.

Import performs no device work, native loading or compilation. Public facades
register their Torch schemas internally; there is no user activation switch.
"""

__all__ = []
