"""Read retained ELF metadata without executing or loading the inspected file.

This deliberately supports ELF64, little-endian x86-64 ET_EXEC/ET_DYN files
with ordinary program/section header numbering and a dynamic symbol section.
Malformed or unsupported inputs raise ValueError; filesystem errors propagate.
No claim about the runtime loader's eventual choice follows from this metadata.
"""

from hashlib import sha256
from pathlib import Path
import struct


def _metadata(data):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def span(offset, size):
        require(0 <= offset <= len(data) and 0 <= size <= len(data) - offset,
                "ELF table or payload exceeds file bounds")
        return data[offset:offset + size]

    def unpack(fmt, offset):
        return struct.unpack(fmt, span(offset, struct.calcsize(fmt)))

    header = unpack("<16sHHIQQQIHHHHHH", 0)
    ident, kind, machine, version = header[:4]
    require(ident[:7] == b"\x7fELF\x02\x01\x01", "expected ELF64 little-endian version 1")
    require(machine == 62 and kind in (2, 3) and version == 1,
            "expected x86-64 executable or shared ELF")
    _, phoff, shoff, _, ehsize, phsize, phnum, shsize, shnum, shstr = header[4:]
    require(ehsize == 64 and phsize == 56 and shsize == 64, "unsupported ELF header sizes")
    require(0 < phnum < 65535 and shnum > 0 and shstr != 65535,
            "missing tables or unsupported extended ELF numbering")
    span(phoff, phsize * phnum)
    span(shoff, shsize * shnum)
    programs = [unpack("<IIQQQQQQ", phoff + phsize * i) for i in range(phnum)]
    sections = [unpack("<IIQQQQIIQQ", shoff + shsize * i) for i in range(shnum)]

    def mapped(address, size):
        offsets = set()
        for p in programs:
            if p[0] == 1 and p[3] <= address and size <= p[5] - (address - p[3]):
                offsets.add(p[2] + address - p[3])
        require(len(offsets) == 1, "dynamic address has no unique file-backed PT_LOAD mapping")
        offset = offsets.pop()
        span(offset, size)
        return offset

    dynamic = [p for p in programs if p[0] == 2]
    require(len(dynamic) == 1, "expected exactly one PT_DYNAMIC segment")
    p = dynamic[0]
    require(p[5] % 16 == 0, "misaligned dynamic table")
    span(p[2], p[5])
    tags = {}
    terminated = False
    for offset in range(p[2], p[2] + p[5], 16):
        tag, value = unpack("<qQ", offset)
        if tag == 0:
            terminated = True
            break
        tags.setdefault(tag, []).append(value)
    require(terminated, "unterminated dynamic table")

    def one(tag, required=False):
        values = tags.get(tag, [])
        require(len(values) <= 1 and (values or not required),
                "missing or repeated singleton dynamic tag " + str(tag))
        return values[0] if values else None

    string_size = one(10, True)  # DT_STRSZ
    strings = span(mapped(one(5, True), string_size), string_size)  # DT_STRTAB

    def string(table, offset):
        require(0 <= offset < len(table), "string index outside string table")
        end = table.find(b"\0", offset)
        require(end >= 0, "unterminated ELF string")
        try:
            return table[offset:end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("non-UTF-8 ELF string") from exc

    def dynamic_string(tag):
        offset = one(tag)
        return None if offset is None else string(strings, offset)

    symbols = [s for s in sections if s[1] == 11]  # SHT_DYNSYM
    require(len(symbols) == 1, "expected exactly one dynamic symbol section")
    sym = symbols[0]
    require(sym[9] == 24 and sym[5] % 24 == 0 and one(11, True) == 24,
            "unsupported dynamic symbol entry size")
    require(sym[3] == one(6, True) and mapped(sym[3], sym[5]) == sym[4],
            "dynamic symbol section differs from DT_SYMTAB")
    require(sym[6] < len(sections), "invalid dynamic symbol string-table link")
    symstr = sections[sym[6]]
    require(symstr[1] == 3 and symstr[3] == one(5, True) and symstr[5] == string_size,
            "dynamic symbol strings differ from DT_STRTAB")
    require(symstr[4] == mapped(symstr[3], symstr[5]), "invalid symbol string-table mapping")
    exports = set()
    for offset in range(sym[4], sym[4] + sym[5], 24):
        name, info, other, shndx, _, _ = unpack("<IBBHQQ", offset)
        text = string(strings, name)
        # Global, weak or GNU-unique definitions with default/protected visibility.
        if shndx != 0 and info >> 4 in (1, 2, 10) and other & 3 in (0, 3):
            if text.startswith("ml_tirust_"):
                exports.add(text)

    requirements = {}
    need_address, need_count = one(0x6ffffffe), one(0x6fffffff)  # VERNEED / VERNEEDNUM
    require((need_address is None) == (need_count is None), "incomplete version requirement tags")
    if need_address is not None:
        require(0 < need_count <= len(data) // 16, "invalid version requirement count")
        seen = set()
        for index in range(need_count):
            require(need_address not in seen, "cyclic version requirement chain")
            seen.add(need_address)
            ver, count, filename, auxiliary, following = unpack("<HHIII", mapped(need_address, 16))
            require(ver == 1 and 0 < count <= len(data) // 16 and auxiliary >= 16,
                    "invalid version requirement record")
            library = string(strings, filename)
            versions = requirements.setdefault(library, set())
            aux_address, aux_seen = need_address + auxiliary, set()
            for item in range(count):
                require(aux_address not in aux_seen, "cyclic version auxiliary chain")
                aux_seen.add(aux_address)
                _, _, _, name, next_aux = unpack("<IHHII", mapped(aux_address, 16))
                versions.add(string(strings, name))
                require((next_aux == 0) == (item == count - 1), "version auxiliary count/chain mismatch")
                if next_aux:
                    require(next_aux >= 16, "overlapping version auxiliary records")
                aux_address += next_aux
            require((following == 0) == (index == need_count - 1), "version requirement count/chain mismatch")
            if following:
                require(following >= 16, "overlapping version requirement records")
            need_address += following

    return {
        "elf_class": 64,
        "endianness": "little",
        "machine": "x86_64",
        "elf_type": {2: "ET_EXEC", 3: "ET_DYN"}[kind],
        "soname": dynamic_string(14),
        "needed": [string(strings, offset) for offset in tags.get(1, [])],
        "runpath": dynamic_string(29),
        "rpath": dynamic_string(15),
        "exported_ml_tirust_symbols": sorted(exports),
        "version_requirements": {library: sorted(versions)
                                 for library, versions in sorted(requirements.items())},
    }


def inspect(path):
    """Return JSON-serializable hash, byte count and literal ELF metadata."""
    data = Path(path).read_bytes()
    return {"sha256": sha256(data).hexdigest(), "bytes": len(data), "metadata": _metadata(data)}
