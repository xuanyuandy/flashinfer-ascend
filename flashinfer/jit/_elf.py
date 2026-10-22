"""Read defined ELF symbols without probing a compiler or requiring binutils."""

import struct

from flashinfer.runtime import SchemaError


def elf_symbols(payload: bytes, *, machine: int, elf_type: int, dynamic_only: bool = False):
    """Validate ELF64 little endian and return defined global/weak functions."""
    if len(payload) < 64 or payload[:6] != b"\x7fELF\x02\x01":
        raise SchemaError("expected a little-endian ELF64 artifact")
    header = struct.unpack_from("<HHIQQQIHHHHHH", payload, 16)
    if header[0] != elf_type or header[1] != machine:
        raise SchemaError("ELF type/machine mismatch: got %s/%#x, expected %s/%#x"
                          % (header[0], header[1], elf_type, machine))
    offset, size, count = header[5], header[10], header[11]
    if size != 64 or not count or offset + size * count > len(payload):
        raise SchemaError("invalid ELF section table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", payload, offset + i * size)
                for i in range(count)]
    names = set()
    for section in sections:
        if section[1] not in ((11,) if dynamic_only else (2, 11)):  # SHT_SYMTAB, SHT_DYNSYM
            continue
        if section[6] >= count or section[9] != 24 or section[5] % 24:
            raise SchemaError("invalid ELF symbol table")
        strings = sections[section[6]]
        if (strings[4] + strings[5] > len(payload)
                or section[4] + section[5] > len(payload)):
            raise SchemaError("ELF symbol/string table is truncated")
        data = payload[strings[4]:strings[4] + strings[5]]
        for pos in range(section[4], section[4] + section[5], 24):
            name, info, _, index, _, _ = struct.unpack_from("<IBBHQQ", payload, pos)
            if info >> 4 in (1, 2) and info & 15 in (2, 10) and index != 0:
                end = data.find(b"\0", name)
                if end < 0:
                    raise SchemaError("invalid ELF symbol name")
                names.add(data[name:end].decode("utf-8"))
    return frozenset(names)
