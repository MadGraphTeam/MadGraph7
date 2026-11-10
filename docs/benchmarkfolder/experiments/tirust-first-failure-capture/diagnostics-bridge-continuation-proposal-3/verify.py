"""Pure preflight of the pinned continuation source and existing closed inputs."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continue_build as c


def main():
    c.packet_check(c.HERE, "source-index.json")
    config = json.loads((c.HERE / "bindings.json").read_text())
    base = c.v2_helpers();bindings = base.checked_bindings();c.source_cache_check(base, bindings)
    c.require(c.canonical_hash(c.tree_state(c.OLD)) == config["original_v2_tree_state_sha256"], "Original V2 guard changed")
    path = c.ROOT / config["imported_inventory"]["path"]
    c.require(c.digest(path) == config["imported_inventory"]["sha256"], "Import manifest changed")
    imported = json.loads(path.read_text());c.require(len(imported) == 1161, "Import count changed")
    for name, expected in imported.items():c.require(c.digest(c.OLD / name) == expected, "Closed partial artifact changed")
    for name, expected in config["additional_file_pins"].items():c.require(c.digest(Path(name)) == expected, "Tool/native pin changed")
    print(json.dumps({"source_proposal_verified": True, "imported_outputs": 1161,
                      "original_v2_unchanged": True, "conditional_reuse_only": True,
                      "build_started": False, "private_dso_loaded": False, "tensor_evaluations": 0}, sort_keys=True))


if __name__ == "__main__":main()
