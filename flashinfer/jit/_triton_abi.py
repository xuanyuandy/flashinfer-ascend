"""Explicit device symbol contracts for single-entry and mixed Triton binaries."""

from ._elf import elf_symbols
from ..runtime.schema import SchemaError


def triton_binary_mode(config):
    declaration = config.get("binary", {"mode": "single"})
    if set(declaration) != {"mode"} or declaration["mode"] not in {"single", "mix"}:
        raise ValueError("Triton binary must declare exactly mode=single or mode=mix")
    return declaration["mode"]


def verify_triton_symbols(payload, entry, mode="single"):
    """Mixed ABI requires BOTH derived device entries, never an arbitrary suffix."""
    if mode not in {"single", "mix"}:
        raise SchemaError("unknown Triton binary mode")
    required = ({entry} if mode == "single" else
                {entry + "_mix_aic", entry + "_mix_aiv"})
    symbols = elf_symbols(payload, machine=0x1029, elf_type=2)
    if not required <= symbols:
        raise SchemaError("Triton %s ELF is missing defined entry symbols %r" % (
            mode, sorted(required - symbols)))
    return symbols
