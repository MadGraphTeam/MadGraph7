"""Explicit current lifecycle hook; an absent selection stays pending.

The successor package must close and acquire its exact index hash first. This
adapter never falls back to the old full-source proof or weakens its validator.
"""
import importlib.util
import json
from pathlib import Path, PurePosixPath

INDEX = "experiments/subprocess-cache-lifecycle-current-qualification-2/evidence-index.json"


def legacy():
    source = Path(__file__).with_name("campaign_report_lifecycle.py")
    spec = importlib.util.spec_from_file_location("strict_lifecycle_identity", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(evidence, manifest, pin=None):
    strict = legacy()
    if pin is None:
        result = strict.pending()
        result["reason"] = "No explicit closed current-source portable proof selected; historical full-source proof does not qualify changed source."
        return result
    strict.require(isinstance(pin, dict) and set(pin) == {"index", "sha256"}
            and pin["index"] == INDEX, "Current lifecycle requires an exact successor2 index pin")
    strict.require(isinstance(pin["sha256"], str) and len(pin["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in pin["sha256"]),
            "Current lifecycle requires a lowercase SHA-256 index pin")
    raw_index = evidence.add(INDEX, pin["sha256"])
    index = json.loads(raw_index)
    strict.require(index.get("schema_version") == 1 and isinstance(index.get("files"), dict)
            and index["files"], "Unknown or empty current lifecycle index")
    prefix = str(PurePosixPath(INDEX).parent)

    def read_bytes(name):
        strict.require(isinstance(name, str) and name and "\\" not in name,
                       "Invalid current lifecycle path")
        path = PurePosixPath(name)
        strict.require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
                       "Current lifecycle verifier escaped its index")
        if name == "evidence-index.json":
            return raw_index
        identity = index["files"].get(name)
        strict.require(isinstance(identity, dict) and identity.get("kind") == "file",
                       "Unbound current lifecycle input: " + name)
        raw = evidence.add(prefix + "/" + name, identity["sha256"])
        strict.require(type(identity["size"]) is int and len(raw) == identity["size"],
                       "Current lifecycle input size differs")
        return raw

    def read_json(name):
        def invalid(value):
            raise ValueError("Nonfinite current lifecycle JSON: " + value)
        return json.loads(read_bytes(name), parse_constant=invalid)

    verifier = strict.retained_verifier(read_bytes("verify_evidence.py"))
    proof = verifier(read_bytes=read_bytes)
    strict.require(proof.get("passed") is True and proof.get("index_sha256") == strict.digest(raw_index),
                   "Current lifecycle portable verification did not pass this index")
    summary = read_json("qualification-summary.json")
    strict.require(summary.get("schema_version") == 1
            and summary.get("kind") == "tirust-generated-subprocess-cache-lifecycle"
            and summary.get("passed") is True and proof.get("summary") == summary,
            "Current lifecycle summary differs from verified replay")
    # The strict source/native check is reused unchanged from the old collector.
    identity = strict.validate_candidate(read_json("materialized-source.json"), read_json("source-freeze.json"),
                                         evidence.json(strict.INSTALLER), manifest)
    strict.require(summary["implementation_revision"] == identity["implementation_revision"],
                   "Current lifecycle implementation revision differs")
    strict.require(proof.get("physical_workers") == 4 and proof.get("measured_physical_calls") == 76
            and proof.get("tensor_key_calls") == 96 and proof.get("card_smoke_passed") is True
            and proof.get("historical_symlinks_are_metadata_only") is True,
            "Current lifecycle proof scope differs")
    return {"status": "passed", "evidence": INDEX, "index_sha256": strict.digest(raw_index),
            "summary_evidence": prefix + "/qualification-summary.json", "identity": identity,
            "qualification": summary, "verification": {key: value for key, value in proof.items() if key != "summary"},
            "scope": "Four fresh workers, 76 checked native-DP physical calls and 96 separate DP/QP tensor-key requests; exact cache/observer parity. Physical-QP rescues, rank-six, EW, poles-only and runtime remain outside this lifecycle scope."}
