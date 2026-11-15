"""Canonical device models and shared source architectures, without probing."""

import re


ARCH22_SOCS = ("Ascend910B", "Ascend910C")
GENERAL_ASCEND_SOCS = ARCH22_SOCS + ("Ascend950",)
SOC_TARGETS = {
    "Ascend910B": ("ascend910b", "dav-2201"),
    "Ascend910C": ("ascend910_93", "dav-2201"),
    "Ascend950": ("ascend950", "dav-3510"),
}


def canonical_soc_model(value):
    """Normalize actual A2/A3/A5 model aliases; keep B and C distinct."""
    key = value.replace(" ", "").lower()
    if re.fullmatch(r"ascend910b[0-9]*", key):
        return "Ascend910B"
    if re.fullmatch(r"ascend910c[0-9]*|ascend910_93[0-9]*", key):
        return "Ascend910C"
    if re.fullmatch(r"ascend950[a-z0-9_]*", key):
        return "Ascend950"
    raise ValueError("unsupported Ascend device model: " + value)
