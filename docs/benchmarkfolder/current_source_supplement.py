"""Explicit closed progress for a separate source; never alter main report gates."""
import importlib.util
import math
from pathlib import Path, PurePosixPath
import sys
import tempfile


SCHEMA = "madloop-tirust-source-progress-selection-v1"
CURRENT = "entry-latch-cache-guard-1"
HISTORICAL = "invariant-symmetry-1"
DOCS = "docs/benchmarkfolder/"
HISTORICAL_DOSSIER = "fa88e07bd94c08dbaafd8e8ad642d82dbbacf1220dea8462145281040543c558"
MANIFEST = "4a27dda7a0977cd1b0ed0d779b30c403da50968bf494aa96d31cf18fc9fe2076"
SOURCE = "afb6e0bea64ea0932751ff4869c206bcea9477d40a39a54bb99d0f06d85b75a6"
NUMERICAL = "3c13ad88b825d9d19b1030b62b317c0f31759e002d8dfdb9050142301a7cae24"
DSO = "eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9"
CACHE = "a8a1d3c9d50f4a7bd4feeecd7fadf40ad81e32be1d254ef29fe15e658317483f"
LANES = (("baseline", 1), *(("candidate", b) for b in (1, 6, 7, 8, 9)))
ADMITTED_NUMERICAL = ({
    "closure_index": {"path": "campaigns/entry-latch-cache-guard-1/n1-qualification-1/closure-index.json",
        "sha256": "3888b1ca29d3e368c748c141462ea4286a8c88fd14ecc00a80928c24e8555dfa"},
    "replay_index": {"path": "campaigns/entry-latch-cache-guard-1/n1-closed-replay-1/index.json",
        "sha256": "3c0d75d00da25fb258ce1618111d0cb8108a27d1b8d605bfaed09fa953853f2c"}},)
ADMITTED_NUMERICAL += ({'closure_index': {'path': 'campaigns/entry-latch-cache-guard-1/remaining-qualification-1/li_gg_h/checkpoint.json',
                    'sha256': 'aafac6b45e83e6c8669179c5d52e4380249020e744c294ec2e5e4a0ed5761419'},
  'replay_api': 'one-case-v1',
  'replay_index': {'path': 'experiments/closed-continuation-verifier-1/source-index.json',
                   'sha256': 'b08adff102fdaabe6dc290e3724a211238664c22e27546ad3f45ab3fc50660e0'}},
 {'closure_index': {'path': 'campaigns/entry-latch-cache-guard-1/remaining-qualification-1/qcd_n2_gg_ttx/checkpoint.json',
                    'sha256': '589cd77e8056e454e0d1405054fc91e1eb5aeccc7db82c21f9914f69c7fa2461'},
  'replay_api': 'one-case-v1',
  'replay_index': {'path': 'experiments/closed-continuation-verifier-1/source-index.json',
                   'sha256': 'b08adff102fdaabe6dc290e3724a211238664c22e27546ad3f45ab3fc50660e0'}},
 {'closure_index': {'path': 'campaigns/entry-latch-cache-guard-1/remaining-qualification-1/ew_n2_epem_mupmum/checkpoint.json',
                    'sha256': '31a132c08b0874eba7663f5557d7f3ce4f9dba0c0ef2a01f6c32da86afb5fa69'},
  'replay_api': 'one-case-v1',
  'replay_index': {'path': 'experiments/closed-continuation-verifier-1/source-index.json',
                   'sha256': 'b08adff102fdaabe6dc290e3724a211238664c22e27546ad3f45ab3fc50660e0'}},
 {'closure_index': {'path': 'campaigns/entry-latch-cache-guard-1/remaining-qualification-1/mixed_uux_ga/checkpoint.json',
                    'sha256': '461bc164f716da9ed0226cab9443357f765659461479959361d5ea96181bcea4'},
  'replay_api': 'one-case-v1',
  'replay_index': {'path': 'experiments/closed-continuation-verifier-1/source-index.json',
                   'sha256': 'b08adff102fdaabe6dc290e3724a211238664c22e27546ad3f45ab3fc50660e0'}},
 {'closure_index': {'path': 'campaigns/entry-latch-cache-guard-1/remaining-qualification-1/li_gg_hh/checkpoint.json',
                    'sha256': 'de9ccc727f2e4782d02357e5026f7fed2803329d71c1b4d6ae5532c7f59704f2'},
  'replay_api': 'one-case-v1',
  'replay_index': {'path': 'experiments/closed-continuation-verifier-1/source-index.json',
                   'sha256': 'b08adff102fdaabe6dc290e3724a211238664c22e27546ad3f45ab3fc50660e0'}},
 {'closure_index': {'path': 'campaigns/entry-latch-cache-guard-1/remaining-qualification-1/qcd_n3_uux_zgg/checkpoint.json',
                    'sha256': '5da76edf538e19fb894bf1f4ba212bf0c0fd984f6e9e4db330035de0d5efdc4d'},
  'replay_api': 'one-case-v1',
  'replay_index': {'path': 'experiments/closed-continuation-verifier-1/source-index.json',
                   'sha256': 'b08adff102fdaabe6dc290e3724a211238664c22e27546ad3f45ab3fc50660e0'}},
 {'closure_index': {'path': 'campaigns/entry-latch-cache-guard-1/remaining-qualification-1/ew_n3_epem_ttxa/checkpoint.json',
                    'sha256': 'cadaa89a53a18ba4da930573f362eb571862154d92cd6abfa78861dfc81bbf95'},
  'replay_index': {'path': 'experiments/closed-continuation-verifier-1/source-index.json',
                   'sha256': 'b08adff102fdaabe6dc290e3724a211238664c22e27546ad3f45ab3fc50660e0'},
  'replay_api': 'one-case-v1'})

PROMOTION = {"path": "experiments/fortran-lifecycle-promotion-1/index.json",
    "sha256": "8f222c796d3c135440e87d4678eb1a35b37a8ffecda6a5afe12a8124d0550e49"}
HANDOFF = {"path": "campaigns/entry-latch-cache-guard-1/timing-handoff-1/index.json",
    "sha256": "a2651ac1c0439e66b51566b4f3959d25d845f26e2239ed35426e962fb7b32aaf"}



def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(name):
    require(isinstance(name, str) and name and "\\" not in name,
            "Invalid supplement path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and not any(p in (".", "..") for p in path.parts)
            and path.as_posix() == name, "Unsafe supplement path")
    return name


def pin_json(inputs, pin):
    require(isinstance(pin, dict) and set(pin) == {"path", "sha256"}, "Expected an exact supplement pin")
    checksum = pin["sha256"]
    require(isinstance(checksum, str) and len(checksum) == 64
            and all(c in "0123456789abcdef" for c in checksum), "Invalid supplement checksum")
    return inputs.json(safe(pin["path"]), checksum)


def bind_index(inputs, pin, *, repository_paths=False):
    """Bind only declared payloads, with one explicit path convention per role."""
    index = pin_json(inputs, pin)
    require(isinstance(index.get("files"), dict) and index["files"], "Empty supplement inventory")
    prefix = str(PurePosixPath(pin["path"]).parent)
    names = {pin["path"]}
    for relative, record in index["files"].items():
        safe(relative)
        if repository_paths:
            require(relative.startswith(DOCS), "Numerical closure payload is outside the docs root")
            name = relative[len(DOCS):]
        else:
            name = prefix + "/" + relative
        raw = inputs.add(safe(name), record["sha256"])
        require(type(record.get("bytes")) is int and len(raw) == record["bytes"],
                "Supplement payload byte count differs")
        names.add(name)
    return index, names


def replay(inputs, names, module_name, argument_root, *arguments):
    """Run the admitted pure verifier in an indexed text-only relocation."""
    require(sys.flags.optimize == 0, "Closed supplement replay requires unoptimized Python")
    require(module_name in names, "Supplement verifier is not indexed")
    with tempfile.TemporaryDirectory(prefix="madloop-report-supplement-") as directory:
        root = Path(directory)
        for name in sorted(names):
            target = root / DOCS / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(inputs.add(name, inputs.files[name]["sha256"]))
        module_path = root / DOCS / module_name
        previous, previous_path = sys.dont_write_bytecode, sys.path[:]
        try:
            sys.dont_write_bytecode = True
            spec = importlib.util.spec_from_file_location("closed_report_supplement", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.verify(root / argument_root, *arguments)
        finally:
            sys.dont_write_bytecode = previous
            sys.path[:] = previous_path


def admit_numerical(selections):
    require(isinstance(selections, list) and selections, "No explicitly closed numerical checkpoints")
    require(all(selected in ADMITTED_NUMERICAL for selected in selections),
            "Numerical supplement needs an explicitly admitted closed checkpoint/replay pair")
    keys = [(selected["closure_index"]["path"], selected["closure_index"]["sha256"]) for selected in selections]
    require(len(keys) == len(set(keys)), "Duplicate numerical checkpoint")


def numerical(inputs, selections, manifest, config):
    admit_numerical(selections)
    observed = {}
    lane_records = {}
    for selected in selections:
        closure, closure_names = bind_index(inputs, selected["closure_index"], repository_paths=True)
        portable, portable_names = bind_index(inputs, selected["replay_index"])
        require(closure.get("manifest_sha256") == MANIFEST
                and closure.get("performance_acceptance_eligible") is False,
                "Numerical supplement belongs to another source or timing scope")
        module = str(PurePosixPath(selected["replay_index"]["path"]).parent) + "/verify.py"
        if selected.get("replay_api") == "one-case-v1":
            require(closure.get("schema") == "madloop-tirust-one-case-closure-v1",
                    "Expected the admitted one-case checkpoint schema")
            result = replay(inputs, closure_names | portable_names, module, "",
                            DOCS + selected["closure_index"]["path"], selected["closure_index"]["sha256"])
            case_ids = [row.get("case") for row in closure.get("cases", [])]
            require(case_ids == [closure.get("case", {}).get("id")], "One-case checkpoint identity differs")
        else:
            require(portable.get("closure_index_sha256") == selected["closure_index"]["sha256"],
                    "Portable replay and numerical closure differ")
            result = replay(inputs, closure_names | portable_names, module, "")
            case_ids = closure.get("case_ids")
        require(isinstance(case_ids, list) and case_ids and len(case_ids) == len(set(case_ids))
                and [row.get("case") for row in closure.get("cases", [])] == case_ids,
                "Numerical supplement case inventory differs")
        require(result.get("index_sha256") == selected["closure_index"]["sha256"]
                and result.get("cases") == case_ids and result.get("lanes_replayed") == 6 * len(case_ids)
                and type(result.get("regression_and_target_passed")) is bool
                and result.get("native_executions") == 0
                and result.get("payloads_verified") == len(closure["files"])
                and result.get("performance_acceptance") is False,
                "Portable numerical replay did not qualify its declared closed cases")
        for case in closure["cases"]:
            identifier = case["case"]
            require(identifier not in observed, "Duplicate numerical case across selected checkpoints")
            lanes = case.get("lanes", [])
            require(case.get("status") in ("passed", "failed") and type(case.get("returncode")) is int
                    and [(row.get("source"), row.get("backend")) for row in lanes] == list(LANES),
                    "Incomplete or unordered six-lane numerical case")
            dp = qp = 0
            for lane in lanes:
                require(lane.get("status") == "complete" and lane.get("matched_saved_points") == 100
                        and type(lane.get("numerical_agreement_passed")) is bool
                        and type(lane.get("ordinary_target_passed")) is bool
                        and lane.get("performance_acceptance_eligible") is False,
                        "Unqualified numerical lane in progress supplement")
                counts = lane.get("accepted_backend_precision_counts", {})
                require(all(type(value) is int and value >= 0 for value in counts.values())
                        and set(counts) <= {str(lane["backend"]) + ":DP", str(lane["backend"]) + ":QP"}
                        and sum(counts.values()) == 100, "Incomplete accepted precision counts")
                dp += counts.get(str(lane["backend"]) + ":DP", 0)
                qp += counts.get(str(lane["backend"]) + ":QP", 0)
            lane_records[identifier] = {"lanes": lanes}
            if selected.get("replay_api") == "one-case-v1":
                diagnostics = result.get("lanes", [])
                require([(row.get("source"), row.get("backend")) for row in diagnostics] == list(LANES)
                        and all(raw.get("regression_passed") == closed["numerical_agreement_passed"]
                            and raw.get("ordinary_target_passed") == closed["ordinary_target_passed"]
                            and raw.get("accepted_backend_precision_counts") == closed["accepted_backend_precision_counts"]
                            for raw, closed in zip(diagnostics, lanes)),
                        "Portable per-lane diagnostics differ from the closed checkpoint")
                lane_records[identifier]["portable_lane_diagnostics"] = diagnostics
            observed[identifier] = (dp, qp,
                all(lane["numerical_agreement_passed"] for lane in lanes),
                all(lane["ordinary_target_passed"] for lane in lanes),
                case["status"] == "passed" and case["returncode"] == 0)
        require(result["regression_and_target_passed"] == all(
            observed[identifier][2] and observed[identifier][3] for identifier in case_ids),
            "Portable numerical pass flags differ from the closed lane outcomes")
    require(set(observed) <= {row["id"] for row in config["cases"]}, "Unknown numerical case")
    rows = [{"id": row["id"], "process": row["process"], "role": row["role"],
             "status": ("qualified" if observed[row["id"]][2] and observed[row["id"]][4]
                        else "failed") if row["id"] in observed else "pending",
             "closed_lanes": 6 if row["id"] in observed else 0,
             "regression_pass": observed[row["id"]][2] if row["id"] in observed else None,
             "ordinary_target_pass": observed[row["id"]][3] if row["id"] in observed else None,
             "accepted_dp": observed.get(row["id"], (0, 0))[0],
             "accepted_qp": observed.get(row["id"], (0, 0))[1],
             "evidence": lane_records.get(row["id"], {})} for row in config["cases"]]
    misses = [{"case": identifier, "source": lane["source"], "backend": lane["backend"],
               "scalar_misses": lane["ordinary_target_miss_count"],
               "distinct_points": len(lane["target_miss_points"]),
               "columns": lane["target_miss_columns"],
               "maximum_finite_symmetric_residual": lane["maximum_symmetric_finite_error"]}
              for identifier, record in lane_records.items()
              for lane in record.get("portable_lane_diagnostics", []) if lane["ordinary_target_miss_count"] > 0]
    build_notes = []
    if "ew_n3_epem_ttxa" in observed:
        build_notes.append("The EW n3 snapshot binds 375 unchanged inherited baseline objects, 426 generated source pins and the donor compiler flags. The original baseline full-build receipt was not located in this audit; this does not demonstrate a fresh baseline full rebuild. Candidate exports and numerical objects are fresh. Numerical agreement and this baseline provenance limitation remain separate.")
    return {"closed_cases": len(observed), "qualified_cases": sum(row["status"] == "qualified" for row in rows), "total_cases": 11,
            "target_misses": misses, "build_scope_notes": build_notes, "closed_lanes": 6 * len(observed),
            "total_lanes": 66, "points_per_lane": 100, "cases": rows}


def validate(inputs, path, main_campaign, main_dossier):
    selected = inputs.json(safe(path))
    require(selected.get("schema") == SCHEMA and set(selected) == {
        "schema", "historical_dossier", "current_manifest", "numerical_checkpoints",
        "cache_measurement", "promotion", "local_handoff", "context_path"}, "Unknown source progress selection")
    admit_numerical(selected["numerical_checkpoints"])
    require(main_campaign == HISTORICAL and selected["historical_dossier"]["path"] == main_dossier
            and selected["historical_dossier"]["sha256"] == HISTORICAL_DOSSIER,
            "Source progress must retain the published historical campaign/dossier")
    require(selected["current_manifest"]["path"] == "campaigns/" + CURRENT + "/manifest.json"
            and selected["current_manifest"]["sha256"] == MANIFEST, "Current manifest needs its exact admitted pin")
    require(selected["cache_measurement"] == {"path": "experiments/fortran-lazy-tirust-clear-measurement-1/index.json",
            "sha256": CACHE} and selected["promotion"] == PROMOTION and selected["local_handoff"] == HANDOFF,
            "Cache, promotion and local handoff need their exact admitted pins")
    context = safe(selected["context_path"])
    require(context.startswith("campaigns/" + CURRENT + "/report-supplements/")
            and context.endswith("/report-context.json"), "Supplement needs a separate explicit context path")
    historical = pin_json(inputs, selected["historical_dossier"])
    require(historical.get("campaign_binding", {}).get("id") == HISTORICAL,
            "Historical dossier has a different campaign")
    manifest = pin_json(inputs, selected["current_manifest"])
    require(selected["current_manifest"]["sha256"] == MANIFEST and manifest.get("id") == CURRENT
            and manifest["candidate"]["exporter_identity"]["files_sha256"] == SOURCE
            and manifest["candidate"]["numerical_identity_sha256"] == NUMERICAL
            and manifest["candidate"]["native"]["files"]["lib/libmadloop_tirust.so"] == DSO,
            "Current source progress has another production/native identity")
    config_path = "campaigns/" + CURRENT + "/cases.json"
    config = inputs.json(config_path, manifest["shared_inputs"]["files"]["cases.json"])
    require(len(config.get("cases", [])) == 11 and len({row["id"] for row in config["cases"]}) == 11,
            "Current progress must retain all eleven configured cases")
    progress = numerical(inputs, selected["numerical_checkpoints"], manifest, config)
    cache, cache_names = bind_index(inputs, selected["cache_measurement"])
    cache_root = str(PurePosixPath(selected["cache_measurement"]["path"]).parent)
    measured = replay(inputs, cache_names, cache_root + "/verify.py", DOCS + cache_root)
    require(measured.get("verified") is True and measured.get("byte_parity") is True
            and measured.get("performance_acceptance_eligible") is False,
            "Cache experiment replay or performance scope differs")
    rows = [{"label": label, "ratio": measured[kind][key]["geometric_mean_ratio"],
             "lower_95": measured[kind][key]["lower_95"], "upper_95": measured[kind][key]["upper_95"]}
            for label, kind, key in (("Empty wrapper; no context", "micro", "0"),
                ("Active wrapper; context registered", "micro", "1"),
                ("Full QCD n1; CutTools", "full_amplitude", "1"),
                ("Full QCD n1; TIRust", "full_amplitude", "9"))]
    require(all(all(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] > 0
                    for key in ("ratio", "lower_95", "upper_95"))
                and row["lower_95"] <= row["upper_95"] for row in rows),
            "Invalid cache timing ratio or interval")
    promotion, _ = bind_index(inputs, selected["promotion"])
    handoff, handoff_names = bind_index(inputs, selected["local_handoff"], repository_paths=True)
    handoff_path = "campaigns/" + CURRENT + "/timing-handoff-1/LOCAL_TIMING_TODO.md"
    require(handoff_path in handoff_names, "Displayed local handoff is not authenticated")
    inputs.add("PRECISION_STREAM_TODO.md")
    require(promotion.get("schema") == "fortran-lifecycle-promotion-v1"
            and handoff.get("actual_local_measurements") is False, "Unexpected promotion/local scope")
    return {"historical": {"campaign": HISTORICAL,
                "source_revision": historical["registered_exporter_revision"],
                "pdf_commit": "913a746201eb6ef806ccb219c890a7f3486d6322",
                "pdf_sha256": "91ceeae99968df742e7e141467e3bab321394a0512328be7df9be0f5b86d559a"},
            "current": {"campaign": CURRENT, "revision": manifest["candidate"]["exporter_identity"]["revision"],
                "source_sha256": SOURCE, "numerical_identity_sha256": NUMERICAL, "native_dso_sha256": DSO},
            "numerical": progress, "cache": {"pairs": 7, "scope": "guard alone", "rows": rows,
                "numerical_byte_parity": True}, "local": {"total_lanes": 66, "status": "pending",
                "handoff_path": "campaigns/" + CURRENT + "/timing-handoff-1/LOCAL_TIMING_TODO.md",
                "precision_todo_path": "PRECISION_STREAM_TODO.md"}, "delivery_complete": False}, context
