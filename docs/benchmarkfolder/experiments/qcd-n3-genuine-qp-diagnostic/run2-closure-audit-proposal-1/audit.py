"""Independent static ELF audit; never invokes or loads inspected binaries.

Read-only original inputs. The only outputs are additive audit JSON and Markdown
in this script's directory. ELF metadata is parsed with Python's stdlib.
"""
from pathlib import Path
import hashlib
import json
import re
import struct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = HERE.parent / "source-proposal2"
WORK = ROOT / ".tirust-build/qcd-n3-genuine-qp-diagnostic"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def elf(path):
    literal = str(path)
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()

    def require(ok, message):
        if not ok:
            raise ValueError(f"{path}: {message}")

    def span(offset, size):
        require(0 <= offset <= len(data) and 0 <= size <= len(data) - offset,
                "out-of-bounds ELF field")
        return data[offset:offset + size]

    def unpack(fmt, offset):
        return struct.unpack(fmt, span(offset, struct.calcsize(fmt)))

    def cstr(table, offset):
        require(0 <= offset < len(table), "out-of-bounds string")
        end = table.find(b"\0", offset)
        require(end >= offset, "unterminated string")
        return table[offset:end].decode("utf-8")

    header = unpack("<16sHHIQQQIHHHHHH", 0)
    require(header[0][:7] == b"\x7fELF\x02\x01\x01" and header[2] == 62,
            "expected ELF64 little-endian x86-64")
    require(header[9:11] == (56, header[10]) and header[11] == 64,
            "unexpected table entry size")
    programs = [unpack("<IIQQQQQQ", header[5] + 56 * i)
                for i in range(header[10])]
    sections = [unpack("<IIQQQQIIQQ", header[6] + 64 * i)
                for i in range(header[12])]
    names = sections[header[13]]
    names = span(names[4], names[5])
    by_name = {cstr(names, s[0]): s for s in sections}

    def section(name):
        s = by_name[name]
        return span(s[4], s[5])

    dynstr = section(".dynstr")
    dynamic = section(".dynamic")
    tags = {}
    for offset in range(0, len(dynamic), 16):
        tag, value = struct.unpack_from("<qQ", dynamic, offset)
        if tag == 0:
            break
        tags.setdefault(tag, []).append(value)

    def value(tag):
        values = tags.get(tag, [])
        require(len(values) <= 1, "duplicate singleton dynamic tag")
        return cstr(dynstr, values[0]) if values else None

    definitions = {}
    if ".gnu.version_d" in by_name:
        table = section(".gnu.version_d")
        offset = 0
        while True:
            ver, flags, ndx, count, _, aux, following = struct.unpack_from("<HHHHIII", table, offset)
            require(ver == 1 and count > 0, "invalid version definition")
            name, _ = struct.unpack_from("<II", table, offset + aux)
            definitions[ndx] = cstr(dynstr, name)
            if not following:
                break
            require(following >= 20, "overlapping version definition")
            offset += following

    requirements, requirement_indices = {}, {}
    if ".gnu.version_r" in by_name:
        table = section(".gnu.version_r")
        offset = 0
        while True:
            ver, count, filename, aux, following = struct.unpack_from("<HHIII", table, offset)
            require(ver == 1 and count > 0, "invalid version requirement")
            library = cstr(dynstr, filename)
            local = offset + aux
            for _ in range(count):
                _, flags, ndx, name, next_aux = struct.unpack_from("<IHHII", table, local)
                name = cstr(dynstr, name)
                requirements.setdefault(library, []).append(name)
                requirement_indices[ndx & 0x7fff] = (library, name)
                local += next_aux
            if not following:
                break
            require(following >= 16, "overlapping version requirement")
            offset += following

    build_ids = []
    for s in sections:
        if s[1] != 7:
            continue
        table = span(s[4], s[5])
        offset = 0
        while offset + 12 <= len(table):
            nsize, dsize, kind = struct.unpack_from("<III", table, offset)
            offset += 12
            name = table[offset:offset + nsize]
            offset += (nsize + 3) & ~3
            payload = table[offset:offset + dsize]
            offset += (dsize + 3) & ~3
            if name.rstrip(b"\0") == b"GNU" and kind == 3:
                build_ids.append(payload.hex())

    symtab = section(".dynsym")
    versions = section(".gnu.version") if ".gnu.version" in by_name else None
    exported, imported = set(), []
    for i, offset in enumerate(range(0, len(symtab), 24)):
        name, info, other, shndx, _, _ = struct.unpack_from("<IBBHQQ", symtab, offset)
        name = cstr(dynstr, name)
        ndx = struct.unpack_from("<H", versions, i * 2)[0] & 0x7fff if versions else 1
        if shndx and info >> 4 in (1, 2, 10) and other & 3 in (0, 3):
            exported.add((name, definitions.get(ndx)))
        if not shndx and ndx in requirement_indices:
            library, version = requirement_indices[ndx]
            imported.append({"name": name, "library": library, "version": version,
                             "weak": info >> 4 == 2})
    interp = [span(p[2], p[5]).rstrip(b"\0").decode("utf-8")
              for p in programs if p[0] == 3]
    record = {
        "literal_path": literal, "resolved_path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data), "gnu_build_ids": build_ids,
        "comment": section(".comment").rstrip(b"\0").decode("utf-8").split("\0") if ".comment" in by_name else [],
        "pt_interp": interp[0] if interp else None,
        "soname": value(14), "needed": [cstr(dynstr, n) for n in tags.get(1, [])],
        "rpath": value(15), "runpath": value(29),
        "version_definitions": sorted(set(definitions.values())),
        "version_requirements": {k: sorted(set(v)) for k, v in sorted(requirements.items())},
        "versioned_import_count": len(imported),
    }
    return record, exported, imported


def main():
    inputs = [SOURCE / "bindings.json", SOURCE / "source-index.json",
              WORK / "run2/loader.log", WORK / "run2/loader.json", WORK / "run2/result.json",
              WORK / "build2/plan.json", WORK / "build2/link.map", WORK / "build2/build.json"]
    before = {str(p): digest(p) for p in inputs}
    binding = json.loads(inputs[0].read_text())
    loader_text = (WORK / "run2/loader.log").read_text()
    actual_literals = re.findall(r"(?:=>\s+|^\s*)(/\S+)\s+\(0x[0-9a-f]+\)", loader_text, re.M)
    require_count = len(actual_literals) == 15 and "not found" not in loader_text
    if not require_count:
        raise ValueError("Expected exactly fifteen resolved loader entries and no missing objects")
    actual = [elf(p) for p in actual_literals]
    expected = [elf(p) for p in binding["runtime"]]
    if any(r[0]["sha256"] != binding["runtime"][r[0]["literal_path"]] for r in expected):
        raise ValueError("Historical expected library bytes changed")
    actual_by_soname = {r[0]["soname"]: r for r in actual}
    if len(actual_by_soname) != 15:
        raise ValueError("Non-unique actual SONAMEs")
    comparisons = []
    for old, _, _ in expected:
        new = actual_by_soname[old["soname"]][0]
        if old["resolved_path"] != new["resolved_path"]:
            comparisons.append({"soname": old["soname"], "expected": old, "actual": new,
                                "byte_identical": old["sha256"] == new["sha256"],
                                "build_id_identical": old["gnu_build_ids"] == new["gnu_build_ids"],
                                "version_definitions_identical": old["version_definitions"] == new["version_definitions"],
                                "version_requirements_identical": old["version_requirements"] == new["version_requirements"]})
    executable = elf(WORK / "build2/diagnostic")
    donor = elf(Path(binding["donor"]) / "SubProcesses/P0_uux_zgg/check")
    missing_needed, missing_versions, missing_symbols = [], [], []
    edges, symbol_count = [], 0
    for record, _, imports in actual + [executable]:
        for soname in record["needed"]:
            if soname not in actual_by_soname:
                missing_needed.append([record["resolved_path"], soname])
        for soname, versions in record["version_requirements"].items():
            provider = actual_by_soname.get(soname)
            absent = sorted(set(versions) - set(provider[0]["version_definitions"])) if provider else versions
            edges.append({"consumer": record["resolved_path"], "provider_soname": soname,
                          "required_versions": versions, "missing_versions": absent})
            if absent:
                missing_versions.append(edges[-1])
        for symbol in imports:
            provider = actual_by_soname.get(symbol["library"])
            symbol_count += 1
            if not provider or (symbol["name"], symbol["version"]) not in provider[1]:
                missing_symbols.append({"consumer": record["resolved_path"], **symbol})
    link_map = (WORK / "build2/link.map").read_text()
    selected = [{"line": n, "text": line} for n, line in enumerate(link_map.splitlines(), 1)
                if line.startswith("LOAD ") and any(name in line for name in
                   ("libgcc_s.so", "libquadmath.so", "libstdc++.so", "libgfortran.so"))]
    runtime = {r[0]["resolved_path"]: r[0]["sha256"] for r in actual}
    if len(runtime) != 15 or len(comparisons) != 3:
        raise ValueError("Unexpected closure change count")
    after = {str(p): digest(p) for p in inputs}
    if before != after:
        raise ValueError("Original input changed during audit")
    result = {"schema": "qcd-n3-run2-static-closure-audit-1", "native_calls": 0,
              "compiler_calls": 0, "loader_calls": 0, "inputs_sha256": before,
              "input_bytes_unchanged": True, "actual_runtime_count": len(actual),
              "actual_runtime": [r[0] for r in actual], "expected_runtime": [r[0] for r in expected],
              "substitutions": comparisons, "actual_runtime_resolved_sha256": runtime,
              "diagnostic": executable[0], "donor": donor[0], "link_map_compiler_loads": selected,
              "dependency_version_edges": edges, "missing_needed": missing_needed,
              "missing_versions": missing_versions, "versioned_imports_checked": symbol_count,
              "missing_versioned_symbols": missing_symbols,
              "static_interface_checks_passed": not (missing_needed or missing_versions or missing_symbols),
              "limitations": ["Versioned symbol checks do not establish numerical correctness or unversioned symbol semantics.",
                              "The loader log is a trace, not an amplitude execution or a worker runtime map.",
                              "The proposed successor must preserve strict exact-path/hash runtime checks and use a new source/run namespace."]}
    output = HERE / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": digest(output),
                      "actual_runtime_count": len(actual), "substitutions": [
                          {k: c[k] for k in ("soname", "byte_identical", "build_id_identical", "version_definitions_identical", "version_requirements_identical")}
                          for c in comparisons], "static_interface_checks_passed": result["static_interface_checks_passed"],
                      "missing_needed": len(missing_needed), "missing_versions": len(missing_versions),
                      "versioned_imports_checked": symbol_count, "missing_versioned_symbols": len(missing_symbols)}, indent=2))


if __name__ == "__main__":
    main()
