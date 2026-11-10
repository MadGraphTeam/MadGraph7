"""Explicit closed-evidence refresh; no active-run discovery or native execution."""
from pathlib import PurePosixPath


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bind_index(evidence, name, checksum, root=None):
    index = evidence.json(name, checksum)
    prefix = str(PurePosixPath(name).parent) if root is None else root
    require(isinstance(index.get("files"), dict) and index["files"], "Empty closed evidence index")
    for relative, identity in index["files"].items():
        path = PurePosixPath(relative)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe closed evidence path")
        expected = identity if isinstance(identity, str) else identity["sha256"]
        raw = evidence.add(prefix + "/" + relative, expected)
        if isinstance(identity, dict) and "bytes" in identity:
            require(len(raw) == identity["bytes"], "Closed evidence byte count differs")
    return index


def selection(evidence, path, manifest, config):
    value = evidence.json(path)
    if value.get("schema") == "madloop-tirust-closed-report-selection-v2":
        return current_selection(evidence, value, manifest, config)
    require(value.get("schema") == "madloop-tirust-closed-report-selection-v1"
            and value.get("campaign") == manifest["id"], "Unknown report selection")
    root = "campaigns/" + manifest["id"]
    cases = []
    for name, checksum in value["checkpoints"].items():
        index = bind_index(evidence, name, checksum, root)
        require(index.get("schema") == "madloop-tirust-closed-numerical-checkpoint-v1"
                and index.get("performance_acceptance_eligible") is False,
                "Checkpoint is not closed untimed evidence")
        summary_name = str(PurePosixPath(name).parent) + "/summary.json"
        require(summary_name.removeprefix(root + "/") in index["files"], "Unbound checkpoint summary")
        summary = evidence.json(summary_name)
        require(summary["saved_points_per_lane"] == 100
                and summary["lane_count"] == 6 * summary["case_count"]
                and summary["performance_acceptance_eligible"] is False,
                "Incomplete checkpoint lane count")
        cases.extend(summary["case_ids"])
    require(len(cases) == len(set(cases)) and sorted(cases) == sorted(value["closed_cases"])
            and set(cases) <= {case["id"] for case in config["cases"]},
            "Closed case allowlist differs from its checkpoints")
    require(value["performance_status"] == "deferred to Hua-Sheng"
            and value["final_performance_accepted"] is False, "Shared-host results cannot establish local acceptance")
    for name, checksum in value["documents"].items():
        evidence.add(name, checksum)
    for name, checksum in value["packages"].items():
        bind_index(evidence, name, checksum)
    return value


def next_campaign(evidence, selected, report):
    """Bind a fresh source registration without importing any of its results."""
    require(set(selected) == {"manifest", "summary"}, "Unknown next-campaign selection")
    for pin in selected.values():
        require(set(pin) == {"path", "sha256"} and isinstance(pin["sha256"], str)
                and len(pin["sha256"]) == 64 and all(c in "0123456789abcdef" for c in pin["sha256"]),
                "Next campaign requires exact manifest and summary pins")
    manifest = evidence.json(selected["manifest"]["path"], selected["manifest"]["sha256"])
    summary = evidence.json(selected["summary"]["path"], selected["summary"]["sha256"])
    require(manifest.get("schema") == "madloop-tirust-campaign-v1"
            and manifest["id"] != report["campaign_binding"]["id"]
            and summary["campaign_binding"] == {"id": manifest["id"], "manifest_sha256": selected["manifest"]["sha256"]},
            "Next registration is not a separately bound campaign")
    candidate = manifest["candidate"]
    require(candidate["exporter_identity"]["dirty"] is False
            and candidate["exporter_identity"]["files_sha256"] != report["registered_source_sha256"]
            and manifest["baseline"]["revision"] == "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f",
            "Next candidate source or baseline differs from its declared scope")
    cases = summary["cases"]
    require(len(cases) == 11 and {row["id"] for row in cases} == {row["id"] for row in report["numerical"]["cases"]}
            and summary["gate"]["passed"] is False and summary["gate"]["complete"] is False
            and summary["gate"]["failed_cases"] == [] and summary["gate"]["validation_errors"] == [],
            "Next registration is not an unmeasured eleven-case campaign")
    for row in cases:
        require(row["status"] == "pending" and row["complete"] is False and row["retained_attempts"] == []
                and row["performance"]["pairs"] == 0 and row["performance"]["geometric_mean_ratio"] is None
                and len(row["correctness"]) == len(row["timings"]) == 5
                and {check["backend"] for check in row["correctness"]} == {1, 6, 7, 8, 9}
                and {timing["backend"] for timing in row["timings"]} == {1, 6, 7, 8, 9}
                and all(check["status"] == "pending" and check["samples"] == 0 and check["passed"] is None
                        for check in row["correctness"])
                and all(timing["status"] == "pending" and timing["samples"] == 0
                        and timing["median_seconds_per_point"] is None for timing in row["timings"]),
                "Next registration contains observations; select a distinct measured dossier")
    return {"id": manifest["id"], "case_count": len(cases), "manifest": selected["manifest"],
            "summary": selected["summary"], "registered_exporter_revision": candidate["exporter_identity"]["revision"],
            "source_sha256": candidate["exporter_identity"]["files_sha256"],
            "numerical_identity_sha256": candidate["numerical_identity_sha256"],
            "fresh_numerical_qualification": "pending", "precise_performance_acceptance": "deferred to Hua-Sheng",
            "displayed_measurements_from_this_campaign": False}


def supplement(evidence, selected, report):
    """Expose bounded claim anchors while leaving scientific gates unchanged."""
    observed = report["numerical"]["cases"]
    closed = [row for row in observed if row["id"] in selected["closed_cases"]]
    require(len(closed) == len(selected["closed_cases"])
            and all(row["all_six_complete"] and row["regression_pass"] for row in closed),
            "Selected closed numerical evidence failed reparse")
    adaptive = evidence.json("experiments/adaptive-qp-rescue-private-tests/attempt1/execution.json")
    require(adaptive["complete"] is True and adaptive["test_result"] == {"failed": 0, "ignored": 0, "passed": 20}
            and adaptive["physical_amplitude_qualified"] is False and adaptive["production_promoted"] is False,
            "Private adaptive test scope differs")
    born = evidence.json("experiments/born-effective-point-native/ordinary/summary.json")
    qp = evidence.json("experiments/born-effective-point-native/fixed-qp/exact-summary.json")
    require(born["state"] == "complete" and len(born["runs"]) == 12
            and all(row["passed"] and all(c["relative"] == 0 for c in row["coefficients"]) for row in born["comparisons"])
            and qp["full_rows"] == qp["full_requested_view_exact_matches"] == qp["full_C8_coordinate_exact_matches"] == 24
            and all(row["ret"] == 120 for row in qp["records"]), "Born diagnostic scope differs")
    runtime = evidence.json("experiments/qcd-topology-rescue-higher-precision-native2/audit-result.json")
    require(runtime["verification"] == "pass" and runtime["group_calls"] == 232
            and runtime["runtime_capture_complete"] is True and runtime["ordinary_parity_exact"] is True
            and runtime["physics_campaign_accepted"] is False and runtime["production_promoted"] is False,
            "All-512 diagnostic scope differs")
    stability = evidence.json("experiments/default-stability-current-1/evidence/execution-1/adjudication.json")
    require(stability["cases"] == 8 and stability["default_dp_failures_retained"] == 4
            and stability["tighter_policy_qp_rescues"] == 4, "Current stability limitation was lost")
    for row in report["stress"]:
        if row["title"] == "Default stability estimator":
            row.update(text="Current eb3d replay now confirms the historical synthetic limitation: all four 1e-3 projections accept inaccurate DP results (maximum finite symmetric error 7.03e-4); all four 1e-6 projections rescue to QP (maximum symmetric error 5.97e-16). This is a synthetic host-helper fixture, not a physical amplitude. The production default remains unchanged; the registered qualification policy is 1e-6.",
                       evidence="experiments/default-stability-current-1/README.md",
                       native_library_sha256=report["implementation"]["library_sha256"],
                       detail={"current_native_replay": "completed", "physical_amplitude_qualified": False})
        if row["title"] == "QCD higher precision":
            row.update(status="diagnostic", text="A separate private all-512-bit replay completed 232 tensor calls with actual runtime mapping and exact ordinary-point parity. It changes arithmetic precision and its admission gate jointly. Forty-six contracted tensor slots retain relative discrepancies, despite close final projections. It does not qualify the adaptive 256-to-512 library or general physical accuracy.",
                       evidence="experiments/qcd-topology-rescue-higher-precision-native2/README.md")
    triage = evidence.json("campaigns/split-dp-terminal-1/experiments/qcd-n4-target-triage-1/analysis.json")
    require(triage["new_native_evaluations"] == 0 and triage["higher_precision_adjudication"] is None
            and triage["performance_eligible"] is False, "QCD n4 triage promoted to truth/timing")
    refreshed = {
        "closed_cases": len(closed), "expected_cases": 11, "closed_lanes": 6 * len(closed),
        "excluded_active_cases": [row["id"] for row in observed if row["id"] not in selected["closed_cases"]],
        "performance_status": selected["performance_status"], "final_performance_accepted": False,
        "amendment_path": "PLAN-AMENDMENT-2026-11-10.md",
        "local_handoff_path": selected.get("local_handoff_path"),
        "qcd_n4_target_missed_points": len(triage["target_points"]),
        "branch": "madloop_tirust", "future_pr_target": "madloop_ninjars",
        "private_diagnostics": [
            {"title": "Experimental adaptive-precision bridge", "text": "An isolated development build of the TIRust bridge tries 256-bit MPFR first and retries once at 512 bits only for a narrowly identified conditioning refusal. Fortran inputs remain genuine 113-bit QP. Twenty Rust tests pass, including real full/poles tensor kernels and injected failure/cache controls. The separate shared library is built; these tests did not load it through Fortran. Mixed-tier physical amplitudes, dynamic transport, IEEE/trap behavior and performance remain unqualified. It is not installed as the production backend used in this campaign.", "evidence": "experiments/adaptive-qp-rescue-private-tests/README.md"},
            {"title": "Born/effective-point consistency", "text": "Sixteen private exports compile; 51 controlled Fortran checks pass. Twelve ordinary real/CMS legacy runs preserve every compared coefficient exactly. Sixteen fixed-QP runs establish the requested Born/coordinate view, including 24 exact full-arm observations; status 120 is unchecked. No backend 8/9, retained stress amplitude or performance qualification follows from this milestone.", "evidence": "experiments/born-effective-point-native/README.md"},
            {"title": "Installer metadata hardening remains open", "text": "The production installer still accepts boolean values as integer ABI/schema versions; two regression tests remain deliberately failing while the campaign source is frozen. An isolated two-guard correction passes 27 installer/card tests and two extra float-version rejection checks. It has not been applied to production. The passed installed-ABI row covers its identified earlier test groups, not all integration audits.", "evidence": "experiments/strict-metadata-prototype/README.md"},
            {"title": "QCD n4 target misses", "text": "All six 100-point lanes pass 3e-6. TIRust accepts 14 points in DP and 86 in QP. The 1e-8 target is missed at 45 distinct points by at least one candidate reducer; original discrepancies and all pairwise comparisons remain retained. No independent finite/single-pole truth or higher-precision adjudication is supplied by this read-only triage.", "evidence": "campaigns/split-dp-terminal-1/experiments/qcd-n4-target-triage-1/README.md"},
        ],
    }
    if selected.get("implementation_packages") is not None:
        import importlib.util
        from pathlib import Path
        source = Path(__file__).with_name("campaign_report_implementation.py")
        spec = importlib.util.spec_from_file_location("campaign_report_implementation", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.apply(refreshed, module.collect(evidence, selected["implementation_packages"]))
    if selected.get("closed_provisional") is not None:
        import importlib.util
        from pathlib import Path
        source = Path(__file__).with_name("campaign_report_provisional.py")
        spec = importlib.util.spec_from_file_location("campaign_report_provisional", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        refreshed["provisional"] = module.collect(evidence, selected["closed_provisional"], report)
    if selected.get("next_campaign") is not None:
        refreshed["next_campaign"] = next_campaign(evidence, selected["next_campaign"], report)
    return refreshed


CURRENT_SCHEMA = "madloop-tirust-closed-report-selection-v2"
ARCHIVE = "history/report-before-invariant-symmetry-qualification/archive-manifest.json"
ARCHIVE_SHA = "3ac7e02340a1148f9c22d9d7cec2e0b81493c429d9b3ac264e616ff2bb5d9dec"
CURRENT_SUPPORT_PACKAGES = {
    "experiments/default-stability-current-1/evidence-index.json": "e5d3f09114c96f491e27c8f7ddd65469072a55512da9dc23b947492d81ba2bb3",
    "experiments/qcd-topology-rescue-higher-precision-native2/evidence-index.json": "d665b517532be1e87f3e7002edbe8ec6131c814fa26df3d7b002adf0eb409dcf",
}


def sibling(name):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive(evidence, pin):
    require(pin == {"index": ARCHIVE, "sha256": ARCHIVE_SHA}, "Historical report archive pin differs")
    value = evidence.json(ARCHIVE, ARCHIVE_SHA)
    require(value["selected_campaign"] == "split-dp-terminal-1"
            and value["every_ledger_input_verified_against_git"] is True
            and value["complete_ledger_input_count"] == 4294,
            "Historical report closure differs")
    prefix = str(PurePosixPath(ARCHIVE).parent)
    for name in ("benchmark.pdf", "report-build.json", "style.typ"):
        identity = value["files"][name]
        raw = evidence.add(prefix + "/" + name, identity["sha256"])
        require(len(raw) == identity["bytes"], "Historical archive bytes differ")
    return {"campaign": value["selected_campaign"], "pdf_path": prefix + "/benchmark.pdf",
            "pdf_sha256": value["pdf_sha256"], "report_commit": value["report_commit"],
            "pages": 19, "current_performance_eligible": False}


def current_selection(evidence, value, manifest, config):
    required = {"schema", "campaign", "numerical_checkpoint", "closed_cases", "documents", "packages",
                "implementation_packages", "historical_archive", "local_handoff_path", "snapshot_date",
                "performance_status", "final_performance_accepted", "current_lifecycle"}
    require(set(value) == required and value["campaign"] == manifest["id"],
            "Current selection requires explicit closed inputs and no old timing or next-campaign join")
    require(value["performance_status"] == "deferred to Hua-Sheng"
            and value["final_performance_accepted"] is False,
            "Shared-host diagnostics cannot establish local performance acceptance")
    checkpoint = sibling("campaign_report_checkpoint").collect(evidence, value["numerical_checkpoint"], manifest, config)
    require(value["closed_cases"] == checkpoint["summary"]["case_ids"],
            "Selection differs from portable checkpoint cases")
    if value["current_lifecycle"] is not None:
        sibling("campaign_report_current_lifecycle").collect(evidence, manifest, value["current_lifecycle"])
    archive(evidence, value["historical_archive"])
    for name, checksum in value["documents"].items():
        evidence.add(name, checksum)
    require(value["local_handoff_path"] == "campaigns/invariant-symmetry-1/timing-handoff-3/LOCAL_TIMING_TODO.md"
            and value["local_handoff_path"] in value["documents"]
            and "PRECISION_STREAM_TODO.md" in value["documents"], "Current handoff3/precision TODO is unbound")
    require(value["packages"] == CURRENT_SUPPORT_PACKAGES,
            "Current report must retain the closed stability and higher-precision disclosures")
    for name, checksum in value["packages"].items():
        bind_index(evidence, name, checksum)
    return value


def preserve_historical_disclosures(evidence, report):
    """Keep the completed additive disclosures from the archived report."""
    stability = evidence.json("experiments/default-stability-current-1/evidence/execution-1/adjudication.json")
    require(stability["cases"] == 8 and stability["default_dp_failures_retained"] == 4
            and stability["tighter_policy_qp_rescues"] == 4, "Retained stability limitation was lost")
    runtime = evidence.json("experiments/qcd-topology-rescue-higher-precision-native2/audit-result.json")
    require(runtime["verification"] == "pass" and runtime["group_calls"] == 232
            and runtime["runtime_capture_complete"] is True and runtime["ordinary_parity_exact"] is True
            and runtime["physics_campaign_accepted"] is False and runtime["production_promoted"] is False,
            "Retained all-512 diagnostic scope differs")
    for row in report["stress"]:
        if row["title"] == "Default stability estimator":
            row.update(text="Historical split-dp-terminal-1 disclosure, same installed eb3d bridge: its completed replay confirms four inaccurate DP acceptances at the production 1e-3 default (maximum finite symmetric error 7.03e-4) and four QP rescues at 1e-6 (maximum symmetric error 5.97e-16). This synthetic host-helper fixture does not qualify physical amplitudes; the registered policy remains 1e-6.",
                       evidence="experiments/default-stability-current-1/README.md",
                       native_library_sha256=report["implementation"]["library_sha256"],
                       detail={"current_native_replay": "completed", "physical_amplitude_qualified": False})
        elif row["title"] == "QCD higher precision":
            row.update(status="diagnostic", text="Historical split-dp-terminal-1 disclosure: a separate private all-512-bit replay completed 232 tensor calls with actual runtime mapping and exact ordinary-point parity. Forty-six contracted tensor slots retain discrepancies despite close final projections. The changed precision and admission gate do not qualify the adaptive 256-to-512 library, general physical accuracy or runtime acceptance.",
                       evidence="experiments/qcd-topology-rescue-higher-precision-native2/README.md")


def current_supplement(evidence, selected, report):
    require(selected.get("schema") == CURRENT_SCHEMA, "Not a current-source selection")
    rows = report["numerical"]["cases"]
    # All rows, failed and incomplete lanes included, come from the closed payload.
    complete = [row for row in rows if row["all_six_complete"]]
    passed = [row for row in complete if row["regression_pass"] is True]
    target_misses = [row["id"] for row in complete if row["ordinary_target_pass"] is False]
    refreshed = {"closed_cases": len(selected["closed_cases"]), "expected_cases": 11,
        "closed_lanes": 6 * len(selected["closed_cases"]), "complete_cases": len(complete),
        "regression_passed_cases": len(passed), "current_target_miss_cases": target_misses,
        "numerical_status": report["numerical"]["status"], "excluded_active_cases": [],
        "performance_status": selected["performance_status"], "final_performance_accepted": False,
        "amendment_path": "PLAN-AMENDMENT-2026-11-10.md", "local_handoff_path": selected["local_handoff_path"],
        "precision_stream_todo_path": "PRECISION_STREAM_TODO.md",
        "historical_report": archive(evidence, selected["historical_archive"]),
        "branch": "madloop_tirust", "future_pr_target": "madloop_ninjars",
        "private_diagnostics": [
          {"title": "Historical split-dp-terminal-1 QCD n4 target misses",
           "text": "The archived split-dp-terminal-1 report retains 45 distinct target-miss points and its old DP/QP counts. Those observations belong to that historical source. Current 1e-8 target outcomes are derived only from this report's closed current checkpoint.",
           "evidence": ARCHIVE},
          {"title": "Guarded adaptive prototype build remains unqualified",
           "text": "The guarded prototype's build attempts failed under a 4 GiB memory limit and a 600-second timeout. No successful new DSO or physical fix is established. The main installed eb3d bridge remains distinct from experimental private adaptive builds; none is promoted by these diagnostics.",
           "evidence": "experiments/cayley-guarded-adaptive-build-proposal-3/README.md"},
        ]}
    preserve_historical_disclosures(evidence, report)
    module = sibling("campaign_report_implementation")
    module.apply(refreshed, module.collect(evidence, selected["implementation_packages"]))
    return refreshed
