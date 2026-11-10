"""Static ELF observations only; never invoke the target loader or driver."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
V2 = HERE.parent / "diagnostics-bridge-proposal-2"
V3 = HERE.parent / "diagnostics-private-build-complete-3"
WORK = ROOT / ".tirust-build/tirust-private-diagnostics-3"
V2_SHA = "9c962ee3f7117eba4399d5db3253533ecf92dbcf138964b18b446c363f4ca1d4"
V3_SHA = "47a9283df6e28d19a048ae7886b42882b77f77e90f5c695eb11bb03fd688d73d"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packet(directory, name, expected):
    require(digest(directory / name) == expected, "Referenced index changed")
    index = json.loads((directory / name).read_text())
    for name, item in index["files"].items():
        require(Path(name).name == name, "Unsafe referenced payload")
        path = directory / name
        require(not path.is_symlink() and path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], "Referenced payload changed")


def parse(text):
    needed = re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", text)
    sonames = re.findall(r"\(SONAME\).*?\[([^\]]+)\]", text)
    runpath = re.findall(r"\(RUNPATH\).*?\[([^\]]+)\]", text)
    rpath = re.findall(r"\(RPATH\).*?\[([^\]]+)\]", text)
    interp = re.findall(r"Requesting program interpreter: ([^\]]+)\]", text)
    require(len(sonames) <= 1 and len(interp) <= 1, "Ambiguous ELF identity")
    require(not rpath, "Unreviewed transitive DT_RPATH")
    exports, imports, weak_imports = [], [], []
    in_symbols = False
    for line in text.splitlines():
        if line.startswith("Symbol table '.dynsym'"):
            in_symbols = True
            continue
        if in_symbols and line.startswith("Version "):
            in_symbols = False
        if not in_symbols:
            continue
        fields = line.split()
        if len(fields) >= 8 and fields[0].endswith(":") and fields[0][:-1].isdigit():
            name, bind, visibility, section = fields[7], fields[4], fields[5], fields[6]
            if bind not in ("GLOBAL", "WEAK", "UNIQUE") or visibility not in ("DEFAULT", "PROTECTED"):
                continue
            if section != "UND":
                exports.append(name)
            elif bind == "WEAK":
                weak_imports.append(name)
            else:
                imports.append(name)
    definitions, requirements = [], {}
    section, provider = None, None
    for line in text.splitlines():
        if line.startswith("Version definition section"):
            section = "definitions"
        elif line.startswith("Version needs section"):
            section = "requirements"
        if section == "definitions":
            match = re.search(r"Name: (\S+)", line)
            if match:
                definitions.append(match.group(1))
        elif section == "requirements":
            match = re.search(r"File: (\S+)", line)
            if match:
                provider = match.group(1)
                requirements[provider] = []
            match = re.search(r"Name: (\S+)", line)
            if match:
                require(provider is not None, "Version need has no provider")
                requirements[provider].append(match.group(1))
    return {"needed": needed, "soname": sonames[0] if sonames else None,
            "runpath": runpath, "rpath": rpath, "interpreter": interp[0] if interp else None,
            "exports": sorted(exports), "strong_imports": sorted(imports), "weak_imports": sorted(weak_imports),
            "version_definitions": sorted(set(definitions)), "version_requirements": requirements}


def qualify(observation):
    for item in observation["objects"].values():
        require(re.search(r"Class:\s+ELF64", item["stdout"]) and
                re.search(r"Data:\s+2's complement, little endian", item["stdout"]) and
                re.search(r"Machine:\s+Advanced Micro Devices X86-64", item["stdout"]),
                "Expected x86-64 ELF64 little-endian object")
    objects = {name: parse(item["stdout"]) for name, item in observation["objects"].items()}
    original = observation["original_runtime_providers"]
    private, driver, old_bridge = observation["private_dso"], observation["driver"], observation["old_bridge"]
    require(len(original) == 14 and old_bridge in original, "Original runtime set changed")
    providers = {name: checksum for name, checksum in original.items() if name != old_bridge}
    providers[private] = observation["objects"][private]["sha256"]
    aliases = {}
    for name in providers:
        elf = objects[name]
        alias = elf["soname"] if elf["soname"] else Path(name).name
        require(alias not in aliases and "/" not in alias, "Ambiguous or unsafe provider alias")
        aliases[alias] = name
    require(len(aliases) == 14, "Expected fourteen exact provider aliases")
    require(objects[driver]["interpreter"] == observation["loader"] and aliases["ld-linux-x86-64.so.2"] == observation["loader"], "Original interpreter changed")
    require(objects[private]["needed"] == objects[old_bridge]["needed"] and
            objects[private]["version_requirements"] == objects[old_bridge]["version_requirements"], "Private bridge dependency/version delta")
    expected_exports = set(objects[old_bridge]["exports"]) | {"ml_tirust_private_dp_diagnostics_v1"}
    require(set(objects[private]["exports"]) == expected_exports and len(expected_exports) == 13 and
            all(name.startswith("ml_tirust_") for name in expected_exports), "Private export delta changed")
    reached, pending, edges = set(), [driver], []
    while pending:
        name = pending.pop(0)
        if name in reached:
            continue
        reached.add(name)
        for needed in objects[name]["needed"]:
            require(needed in aliases, "Dependency outside exact alias set: " + needed)
            target = aliases[needed]
            edges.append({"from": name, "needed": needed, "provider": target})
            pending.append(target)
        for needed, versions in objects[name]["version_requirements"].items():
            require(needed in aliases and set(versions) <= set(objects[aliases[needed]]["version_definitions"]), "Unsatisfied GNU version requirement")
    require(reached == set(providers) | {driver} and old_bridge not in reached, "Static reachable closure is not exact replacement")
    raw_exported = {symbol for name in reached for symbol in objects[name]["exports"]}
    exported = {symbol.replace("@@", "@") for symbol in raw_exported}
    for name in reached:
        for symbol in objects[name]["strong_imports"]:
            require(symbol.replace("@@", "@") in exported or
                    ("@" not in symbol and any(x.startswith(symbol + "@@") for x in raw_exported)),
                    "Unresolved strong dynamic symbol: " + symbol)
    for symbol in expected_exports:
        owners = [name for name in reached if symbol in objects[name]["exports"]]
        require(owners == [private], "Private ABI interposition within reviewed closure")
    return {"providers": providers, "aliases": aliases, "dependency_edges": edges,
            "static_reachable_objects": sorted(reached), "private_exports": sorted(expected_exports),
            "preserved_original_providers": 13, "replaced_bridge_only": True,
            "static_version_and_strong_symbol_requirements_satisfied": True,
            "dynamic_runtime_maps_observed": False, "private_dso_loaded": False, "tensor_evaluations": 0,
            "required_launch_policy": "Exact alias directory in target-only LD_LIBRARY_PATH; original glibc2.40 loader; no ambient providers or EB3 coexistence"}


def collect():
    packet(V2, "source-index.json", V2_SHA)
    packet(V3, "index.json", V3_SHA)
    require(not (HERE / "static-observation.json").exists(), "Static observation already exists")
    bindings = json.loads((V2 / "bindings.json").read_text())
    terminal = json.loads((V3 / "build-terminal.json").read_text())
    providers = bindings["original_runtime_providers"]
    old = next(name for name in providers if Path(name).name == "libmadloop_tirust.so")
    private = str(WORK / "target/release/libmadloop_tirust_private_diag.so")
    driver = str(WORK / "capture/stability-check")
    files = dict(providers)
    for name, item in terminal["artifacts"].items():
        require(digest(WORK / name) == item["sha256"], "V3 artifact changed")
    files[private] = terminal["artifacts"]["target/release/libmadloop_tirust_private_diag.so"]["sha256"]
    files[driver] = terminal["artifacts"]["capture/stability-check"]["sha256"]
    tool = str(Path(shutil.which("readelf")).resolve())
    observation = {"tool": {"path": tool, "sha256": digest(tool)}, "original_runtime_providers": providers,
                   "private_dso": private, "driver": driver, "old_bridge": old,
                   "loader": next(name for name in providers if Path(name).name == "ld-linux-x86-64.so.2"), "objects": {}}
    for name, expected in files.items():
        require(digest(name) == expected, "Bound artifact/provider changed")
        argv = [tool, "-W", "-h", "-l", "-d", "-V", "--dyn-syms", name]
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30, check=False)
        require(result.returncode == 0 and not result.stderr, "Static readelf inspection failed")
        require(digest(name) == expected, "Artifact/provider changed during static inspection")
        observation["objects"][name] = {"sha256": expected, "argv": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    result = qualify(observation)
    for name, value in (("static-observation.json", observation), ("static-qualification.json", result)):
        with (HERE / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")
    print(json.dumps({"static_objects": len(files), "qualified_provider_count": len(result["providers"]), "private_dso_loaded": False, "tensor_evaluations": 0}, sort_keys=True))


if __name__ == "__main__":
    require(sys.argv[1:] == ["collect"], "Use static_audit.py collect; readelf only")
    collect()
