#!/usr/bin/env python3
"""Bounded metadata admission only; never load a DSO or execute native code."""
from pathlib import Path
import argparse
import hashlib
import json
import re

SOURCE = "77eb57b82d4858ff302fa9870778e01cc86d7326"
NATIVE_SOURCE = "8b74cdbb3e106f2859514bb4ca95d9fe5d70b370"
DSO = "d427667a0f2f55d0bd96778c1259ffcf52e3925341890b81c38f5f94e1eb7052"
MAX_BYTES = 500_000


def require(value, message):
    if not value:
        raise ValueError(message)


def derive(root):
    root = Path(root).resolve()
    here = Path(__file__).resolve().parent
    selected = json.loads((here / "tooling-admission-inputs.json").read_text())
    require(selected["source_commit"] == SOURCE, "source selector mismatch")
    require(selected["native_source_commit"] == NATIVE_SOURCE, "native selector mismatch")
    records = selected["files"]
    docs = {}
    total = 0
    for role, record in records.items():
        relative = Path(record["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                "unsafe input path")
        path = root / relative
        require(not path.is_symlink(), "symlink input")
        require(path.stat().st_size == record["bytes"] <= MAX_BYTES,
                "unbounded or changed metadata input")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == record["sha256"],
                "changed metadata input: " + role)
        docs[role] = raw.decode("utf-8")
        total += len(raw)
    review = json.loads(docs["source_review"])
    require(review["blockers"] == [], "source blockers remain")
    require(review["capability_and_observer_placement"]
            ["timed_Fortran_body_byte_exact_original"] is True,
            "source placement proof absent")
    native = json.loads(docs["toy_native"])
    for key in ("coefficient_and_warmup_bytes_equal",
                "getter_preserves_IEEE_flags_and_diagnostic",
                "mutex_released", "test50_zero_skips"):
        require(native[key] is True, "native toy check missing: " + key)
    require(native["status"] == "passed", "native toy incomplete")
    require(native["native_DSO_sha256"] == DSO, "toy DSO mismatch")
    require(native["performance_qualified"] is False, "toy performance overclaim")
    require(native["real_primary_check_tensor_calls"] == 52,
            "toy tensor call count mismatch")
    require(native["full_SM_amplitude_calls"] == 0, "toy scope mismatch")
    routes = []
    for row in native["observations"]["current"]:
        loops = row["loops"]
        require(loops in (2, 3), "unexpected toy traversal")
        observation = row["observation"]
        route = observation["route_statistics"]
        count = loops * 2
        expected = [count] + [0] * 9 + [count] + [0] * 9
        require(route["available"] is True and route["delta"] == expected,
                "actual route delta mismatch")
        usage = observation["precision_usage"]
        require(usage["bridge_native_dp_only"] is True,
                "bridge-only scope changed")
        require(usage["native_dp_only"] is None,
                "unknown Fortran-cache provenance was overclassified")
        routes.append({"loops": loops, "primary_direct": count,
                       "check_direct": count, "all_other_deltas": 0,
                       "whole_amplitude_native_dp_only": None})
    require(len(routes) == 2, "incomplete toy observations")
    log = docs["root_tooling_log"]
    require(re.search(r"Ran 50 tests in [0-9.]+s", log) is not None,
            "50-test summary missing")
    require(re.search(r"^OK$", log, re.M) is not None and
            re.search(r"skip|FAILED|ERROR", log, re.I) is None,
            "tooling tests failed or skipped")
    manifest = json.loads(docs["manifest"])
    require(manifest["id"] == "tirust-native-tiers-1", "wrong campaign")
    exporter = manifest["candidate"]["exporter_identity"]
    require(exporter["revision"] == SOURCE and exporter["dirty"] is False,
            "wrong or dirty registered source")
    require(exporter["files_sha256"] ==
            "46fd7762300360621518310a92a4d0ee51c7e80bec5da5e64d0888dd131c9bfc",
            "exporter inventory mismatch")
    require(manifest["candidate"]["numerical_identity_sha256"] ==
            "bc16a37d599bcb0f02c17228bdc7ca6f70ecd00880a6bb8276744fae4d8223ca",
            "numerical identity mismatch")
    require(manifest["baseline"]["revision"] ==
            "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f", "baseline changed")
    artifact = json.loads(docs["artifact_admission"])
    require(artifact["final_artifact_qualified"] is True and
            artifact["artifact"]["shared_library_sha256"] == DSO,
            "final artifact not admitted")
    require(artifact["canonical_publication_admitted"] is False,
            "prototype admission unexpectedly authorizes canonical output")
    require(artifact["native_release_tests"] ==
            {"failed": 0, "ignored": 0, "passed": 16},
            "native release test count mismatch")
    require(artifact["root_integration_tests"]["passed"] == 67 and
            artifact["root_integration_tests"]["skips"] == 0,
            "root integration scope mismatch")
    return {
        "schema": "madloop-route-aware-tooling-admission-v1",
        "source_commit": SOURCE,
        "native_source_commit": NATIVE_SOURCE,
        "shared_library_sha256": DSO,
        "campaign": "tirust-native-tiers-1",
        "manifest": records["manifest"],
        "bounded_metadata_files": len(records),
        "bounded_metadata_bytes": total,
        "input_receipts": records,
        "source_review_clear": True,
        "native_toy_qualification_passed": True,
        "native_toy_scope": native["scope"],
        "native_toy_tensor_calls": 52,
        "full_SM_amplitude_calls_in_toy": 0,
        "toy_route_observations": routes,
        "toy_performance_qualified": False,
        "tooling_tests": {"passed": 50, "skips": 0,
                          "duplicate_log_counted_again": False},
        "native_release_tests": artifact["native_release_tests"],
        "root_integration_tests": artifact["root_integration_tests"],
        "fresh_local_numerical_lanes_pending": 66,
        "fresh_local_runtime_gates_pending": 8,
        "fresh_performance_records_admitted": False,
        "large_binary_hashes_recomputed": False,
        "new_native_physics_evaluations": 0,
        "render_or_large_replay_performed": False,
        "historical_performance_relabelled": False,
        "final_packet_admitted": False,
        "canonical_publication_admitted": False,
        "goal_complete": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = derive(args.root)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print("admitted small source/native/tooling metadata; no performance admission")


if __name__ == "__main__":
    main()
