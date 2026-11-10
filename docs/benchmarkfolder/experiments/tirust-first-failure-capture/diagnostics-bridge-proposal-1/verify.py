"""Pure source-packet verification. No build, loader or numerical execution."""
import hashlib
import json
from pathlib import Path
import sys

import bridge_patch
import driver_patch
import build
import protocol

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def main():
    build.checked_source_packet()
    bindings = build.checked_bindings()
    require = protocol.require
    rs = bridge_patch.prepare((ROOT / bindings["base_bridge_source"]).read_bytes())
    c = driver_patch.prepare((ROOT / bindings["base_driver_source"]).read_bytes())
    require(rs == (HERE / "private-lib.rs").read_bytes(), "Rust derivation changed")
    require(c == (HERE / "private-replay.c").read_bytes(), "C derivation changed")
    base_manifest = (ROOT / ".tirust-build/qp-transport-install/mg/HEPTools/tirust/Cargo.toml").read_bytes()
    private_manifest = base_manifest.replace(b'name = "madloop_tirust"', b'name = "madloop_tirust_private_diag"')
    require(private_manifest == (HERE / "private-Cargo.toml").read_bytes(), "Private manifest changes beyond lib name")
    for name, info in bindings["cache_source_roots"].items():
        files = build.inventory(ROOT / info["path"])
        require(len(files) == info["files"] and build.tree_hash(files) == info["sha256"], "Source cache pin changed: " + name)
    keys = []
    for expected in bindings["inputs"]:
        raw = (ROOT / expected["path"]).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected["sha256"] and len(raw) == expected["bytes"], "Captured input changed")
        parsed = protocol.input_record(raw)
        require(all(parsed[k] == expected[k] for k in parsed), "Captured request words changed")
        keys.append(tuple(expected["input_key"][k] for k in ("case_id", "phase", "point", "method", "group")))
    require(len(keys) == len(set(keys)) == 6 and set(keys) == {(1, 1, 25, m, g) for m in (1, 2, 3) for g in (59, 61)}, "Six exact keys")
    artifacts = bindings["candidate_artifacts"]
    require(artifacts["dso_sha256"] is None and artifacts["executable_sha256"] is None and artifacts["native_evaluations"] == 0 and artifacts["private_dso_loaded"] is False, "Unobserved artifacts misrepresented")
    print(json.dumps({"source_packet_verified": True, "inputs": len(keys), "cache_roots": len(bindings["cache_source_roots"]),
                      "original_providers": len(bindings["original_runtime_providers"]), "native_evaluations": 0,
                      "build_or_loader_invoked": False}, sort_keys=True))


if __name__ == "__main__":main()
