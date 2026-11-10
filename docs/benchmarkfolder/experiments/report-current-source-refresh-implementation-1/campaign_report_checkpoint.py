"""Explicit portable numerical checkpoint ingestion; no live-run discovery.

Only an exact source-qualified retainer is imported. Its offline replay runs
inside a temporary package reconstructed solely from the selected index.
The retainer replays pure historical parser functions, never worker commands.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
from types import ModuleType

SCHEMA = "madloop-tirust-portable-numerical-checkpoint-v1"
SOURCE_INDEX_SHA256 = "da149a5cdca4872061b081db9378f9185949f76db102c49f25b621cfa3b77fb1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(name):
    require(isinstance(name, str) and name and "\\" not in name, "Invalid checkpoint path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
            "Unsafe checkpoint path")
    return name


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parsed(raw):
    def invalid(value):
        raise ValueError("Nonfinite checkpoint JSON: " + value)
    return json.loads(raw, parse_constant=invalid)


def collect(evidence, pin, manifest, config, require_full=True):
    """Reparse only the selected closed, content-addressed portable package."""
    require(isinstance(pin, dict) and set(pin) == {"index", "sha256"},
            "Checkpoint requires an exact index pin")
    require(isinstance(pin["sha256"], str) and len(pin["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in pin["sha256"]),
            "Checkpoint requires a lowercase SHA-256 index pin")
    name = safe(pin["index"])
    index_raw = evidence.add(name, pin["sha256"])
    index = parsed(index_raw)
    require(index.get("schema") == SCHEMA and index.get("closed") is True
            and index.get("performance_acceptance_eligible") is False
            and index.get("overall_goal_qualified") is False,
            "Checkpoint is not closed untimed numerical evidence")
    prefix = str(PurePosixPath(name).parent)
    binding = {"id": manifest["id"], "manifest_sha256": sha(evidence.add("campaigns/" + manifest["id"] + "/manifest.json"))}
    require(index.get("campaign_binding") == binding, "Checkpoint belongs to another campaign")
    files = index.get("files")
    require(isinstance(files, dict) and files and "source-index.json" in files
            and index.get("source_index_sha256") == SOURCE_INDEX_SHA256,
            "Checkpoint retainer source is not the qualified source-only proposal")
    members = {}
    for relative, identity in files.items():
        safe(relative)
        require(isinstance(identity, dict) and set(identity) == {"sha256", "bytes"}
                and type(identity["bytes"]) is int, "Invalid checkpoint member identity")
        raw = evidence.add(prefix + "/" + relative, identity["sha256"])
        require(len(raw) == identity["bytes"], "Checkpoint member byte count differs")
        members[relative] = raw
    source_index = members["source-index.json"]
    require(sha(source_index) == SOURCE_INDEX_SHA256, "Checkpoint source-index pin differs")
    sources = parsed(source_index)
    require(sources.get("schema") == "madloop-tirust-source-only-numerical-retention-proposal-v1",
            "Unexpected checkpoint source proposal")
    # Validate the entire trusted source closure before importing retain.py.
    for relative, identity in {**sources["files"], **sources.get("validation_files", {})}.items():
        raw = members.get(safe(relative))
        require(raw is not None and sha(raw) == identity["sha256"] and len(raw) == identity["bytes"],
                "Checkpoint source closure differs: " + relative)
    with tempfile.TemporaryDirectory(prefix="report-closed-checkpoint-") as temporary:
        package = Path(temporary)
        (package / "index.json").write_bytes(index_raw)
        for relative, raw in members.items():
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        # SourceFileLoader may create __pycache__ before the retainer's exact
        # package inventory check. Execute only the already-verified source
        # bytes, with no bytecode cache and no process-global Python flag.
        def source_module(relative, name):
            module = ModuleType(name)
            module.__file__ = str(package / relative)
            exec(compile(members[relative], module.__file__, "exec"), module.__dict__)
            return module
        retainer = source_module("retain.py", "source_qualified_closed_numerical_retainer")
        def load_collector(package_root=package):
            require(Path(package_root) == package, "Retained collector escaped the reconstructed package")
            return source_module("campaign_report_numerical.py", "frozen_numerical_checkpoint_collector")
        # Keep every replay/validation function unchanged; only replace its
        # trusted source import mechanism with the equivalent cache-free loader.
        retainer.load_collector = load_collector
        retainer.replay_completed(package, expected_index_sha256=pin["sha256"])
        summary = parsed(members["summary.json"])
    expected = [row["id"] for row in config["cases"]]
    ids = summary["case_ids"]
    require(len(ids) == len(set(ids)) and set(ids) <= set(expected)
            and summary["campaign_binding"] == binding and ids == index["case_ids"]
            and summary["selected_case_count"] == len(ids)
            and summary["selected_lane_slots"] == 6 * len(ids)
            and summary["performance_acceptance_eligible"] is False
            and summary["overall_goal_qualified"] is False,
            "Checkpoint summary selection or scope differs")
    require(summary["numerical"]["campaign_binding"] == binding,
            "Checkpoint numerical identity differs")
    if require_full:
        terminal = summary["supervisor_terminal"]
        require(ids == expected and len(ids) == 11 and summary["full_matrix_closed"] is True
                and isinstance(terminal, dict) and terminal.get("observed_terminal") is True
                and type(terminal.get("exit_code")) is int,
                "Current report requires an observed terminal eleven-case checkpoint")
    return {"pin": pin, "summary": summary, "numerical": numerical_outcome(summary)}


def numerical_outcome(summary):
    """A failed terminal run cannot become a passing ordinary-matrix gate."""
    numerical = summary["numerical"]
    if summary["selected_status"] != "passed" or (summary["full_matrix_closed"]
            and summary["supervisor_terminal"]["exit_code"] != 0):
        numerical = {**numerical, "status": "failed"}
    return numerical
