"""Read-only ingestion of pinned statistics qualification and accuracy disclosures.

These collectors bind completed portable inventories and inspect their recorded
scope. They do not rerun scientific adjudication, inspect live native artifacts,
or execute retained scripts. Ordinary numerical and timing gates are separate.
"""
from pathlib import PurePosixPath
import re


HISTORICAL = "49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12"
CURRENT = "eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9"
EW = "experiments/ew-qp-mass-prototype/observer-native-design/attempt2/execution-1/"
QCD = "experiments/qcd-topology-rescue-native/"
DEFAULT = "experiments/default-stability-policy-audit/"
TRIAGE = "campaigns/split-dp-terminal-1/experiments/qcd-n3-target-triage-1/"
STATISTICS = "experiments/statistics-roundtrip-current-1/"
PINS = {
    STATISTICS + "manifest.json": "a45647db8786a739ec581d5a5f99733615649d9eeab27679323f74756172a56e",
    EW + "evidence-index.json": "4012228d621d39956944027d5682033781c88791c6dc11f90abc4a6fe30e1f7b",
    EW + "geometry-audit/evidence-index.json": "27fdfb0548c982b6e7b15e72806e95b3f1934efc97d990fe298124f7195ad326",
    QCD + "evidence-index.json": "c7d185c4991fdfacf5c76710d1790b7ccf31c5c2ec0f9270cc4f72ec3dfdfc32",
    DEFAULT + "index.json": "85e7ff21e76cd8547ef1278cd0a6150ade30793368f9db226a13507acf511cfa",
    TRIAGE + "index.json": "10b5a3339dfba063c4c5a264d769f9e5615387965c503fb7a3e2e2d44c421915",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(name):
    require(isinstance(name, str) and name and "\\" not in name, "Invalid qualification evidence path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
            "Qualification evidence path escapes its retained directory")
    return name


def bind_file(evidence, name, identity):
    checksum = identity if isinstance(identity, str) else identity["sha256"]
    require(isinstance(checksum, str) and re.fullmatch(r"[0-9a-f]{64}", checksum),
            "Invalid qualification evidence checksum")
    raw = evidence.add(relative(name), checksum)
    if isinstance(identity, dict):
        require(identity.get("kind", "file") == "file", "Qualification inventory contains a nonregular file")
        size = identity.get("bytes", identity.get("size"))
        require(size is None or len(raw) == size, "Qualification evidence size differs: " + name)
    return raw


def bind_package(evidence, name, field="files"):
    """Absent frozen evidence stays pending; present malformed evidence fails."""
    if not evidence.exists(name):
        return None
    index = evidence.json(name, PINS[name])
    require(index.get("schema", index.get("schema_version")) == 1,
            "Unknown qualification inventory schema")
    files = index.get(field)
    require(isinstance(files, dict) and files, "Empty qualification inventory")
    parent = PurePosixPath(name).parent
    prefix = "" if str(parent) == "." else str(parent) + "/"
    for item, identity in files.items():
        bind_file(evidence, prefix + relative(item), identity)
    return index


def bind_references(evidence, files):
    """Only explicit report-relative references; machine paths stay metadata."""
    prefix = "docs/benchmarkfolder/"
    for name, identity in files.items():
        require(name.startswith(prefix), "Qualification reference is not portable report evidence")
        bind_file(evidence, name[len(prefix):], identity)


def registered_sources(manifest):
    sources = dict(manifest["candidate"]["exporter_identity"]["files"])
    for name, record in manifest["candidate"].get("numerical_sources", {}).items():
        require(name not in sources or sources[name] == record["sha256"], "Registered source inventories disagree")
        sources[name] = record["sha256"]
    return sources


def statistics_scope(index, before, after, output, manifest):
    require(index.get("passed") is True and index.get("test_count") == 10
            and index.get("compiled_fortran_subtests") == 4
            and index.get("native_amplitude_or_rust_reducer_executed") is False
            and index.get("production_or_test_source_edited") is False
            and index.get("historical92_pass_invocation_recovered") is False,
            "Statistics qualification scope differs")
    require(after.get("passed") is True and after.get("returncode") == 0
            and after.get("timed_out") is False and after.get("cleanup") == []
            and after.get("historical92_pass_invocation_recovered") is False,
            "Statistics invocation did not complete within its declared scope")
    require(after["source_sha256_after"] == before["source_sha256"]
            and after["python_module_sha256_after"] == {key: item["sha256"] for key, item in before["python_modules"].items()}
            and after["tool_sha256_after"] == {key: item["sha256"] for key, item in before["tools"].items()},
            "Statistics before/after source or environment identities differ")
    require(before["argv"][1:3] == ["-c", "12"]
            and before["argv"][4:] == ["-m", "pytest", "-v", "-p", "no:cacheprovider",
                                      "tests/unit_tests/various/test_ninjars_statistics.py"]
            and before["cpu"] == 12 and before["timeout_seconds"] == 120,
            "Statistics invocation differs from the bounded recorded test")
    matches = re.findall(r"^(tests/unit_tests/various/test_ninjars_statistics\.py::\S+) PASSED \[\s*\d+%\]$",
                         output, re.MULTILINE)
    require(len(matches) == len(set(matches)) == 10
            and re.search(r"=+ 10 passed, 4 subtests passed in [0-9.]+s =+", output),
            "Statistics raw output lacks ten unique passes and four compiled subtests")
    production = {name: checksum for name, checksum in before["source_sha256"].items()
                  if name.startswith(("Template/", "madgraph/"))}
    require(len(production) == 10, "Incomplete statistics production-source inventory")
    registered = registered_sources(manifest)
    require(all(registered.get(name) == checksum for name, checksum in production.items()),
            "Statistics qualification differs from selected campaign production sources")
    return production


def collect_statistics(evidence, manifest):
    index_name = STATISTICS + "manifest.json"
    index = bind_package(evidence, index_name, "artifacts")
    if index is None:
        return {"status": "pending", "evidence": None,
                "scope": "A frozen current-source statistics invocation is required; historical unbound test logs do not establish it."}
    before = evidence.json(STATISTICS + "before.json")
    after = evidence.json(STATISTICS + "results.json")
    require(after["before_sha256"] == index["artifacts"]["before.json"]
            and after["raw_output_sha256"] == index["artifacts"]["pytest-output.txt"]
            and before["runner_sha256"] == index["artifacts"]["run.py"],
            "Statistics invocation, output or runner binding differs")
    production = statistics_scope(index, before, after,
                                 evidence.add(STATISTICS + "pytest-output.txt").decode(), manifest)
    return {"status": "passed", "evidence": index_name, "tests": 10, "compiled_fortran_subtests": 4,
            "registered_production_sources": production, "scope":
            "One retained invocation passed 10 tests and four compiled synthetic Fortran tally/writer subtests, using current production Python readers. Legacy aliases, TIRust/NinjaRS precision counters and aggregation are covered. This qualifies the identified statistics sources; it neither recovers the historical unbound 92-pass invocation nor establishes amplitude, native-reducer or performance results."}


def ew_scope(core, geometry, summary, analysis, runs):
    require(core.get("transport_input_comparison_eligible") is True
            and core.get("observed_points") == 8 and core.get("observed_native_calls") == 1600,
            "EW transport capture scope differs")
    for record in (core, geometry):
        require(all(record.get(key) is False for key in ("checked_qp_qualified", "physical_accuracy_qualified",
                    "performance_eligible", "threshold_tirust_aggregate_complete")),
                "Unchecked or incomplete EW diagnostic promoted to acceptance")
    require(geometry.get("native_execution_performed") is False and geometry.get("geometry_records") == 800
            and len(analysis["geometry_records"]) == 800, "EW offline geometry scope differs")
    require(summary.get("envelope_verified") is True and summary.get("transport_interpretation_eligible") is True,
            "EW recorded transport envelope is unqualified")
    require(len(runs["cases"]) == 4 and all(row["loaded_libraries"]["libmadloop_tirust.so"]["sha256"] == HISTORICAL
                                          for row in runs["cases"]), "EW diagnostic uses a different native artifact")
    for arm in ("original", "candidate"):
        require(summary["cases"][arm + "-9"]["refusals"] == [0, 37]
                and summary["cases"][arm + "-8"]["refusals"] == [0, 0], "EW refusal inventory differs")
    for record in analysis["native_summary"].values():
        for point in record["points"]:
            require(point["groups"] == 200 and point["public_status"] == [["INTEGER", "120", "FFFFF"]]
                    and all(row[2] == "BFF0000000000000" for row in point["accuracy_bits"]),
                    "EW fixed-QP status or unevaluated accuracy changed")


def qcd_scope(adjudication, guard):
    require(adjudication.get("physics_accepted") is False and adjudication.get("native_executed") is False
            and guard.get("physics_accepted") is False and guard.get("native_executed") is False,
            "QCD prototype adjudication promoted to physical acceptance")
    require(guard["historical_library_sha256"] == HISTORICAL, "QCD diagnostic uses a different native artifact")
    remaining = adjudication["remaining_tirust_consistency"]
    require(remaining["calls"] == 51 and remaining["all_reported_ill_conditioned"] is True
            and remaining["supplied_exact_singular"] == remaining["cartesian_exact_singular"] == 0,
            "QCD remaining-refusal classification differs")
    norm = adjudication["normalization"]
    require(norm["topology-b:8"]["host_code"] == norm["topology-c:8"]["host_code"] == 322
            and norm["topology-b:8"]["passes_3e_6_double_pole_only"] is False
            and norm["topology-c:8"]["passes_3e_6_double_pole_only"] is True
            and all(norm[arm + ":9"]["host_code"] == 420 for arm in ("original-b", "original-c", "topology-b", "topology-c"))
            and all(row["independent_finite_single_truth_available"] is False for row in norm.values()),
            "QCD host-status, normalization or finite/single-pole scope differs")


def default_scope(observations, manifest):
    require(observations["historical_bridge_dso_sha256"] == HISTORICAL
            and observations["current_campaign_bridge_dso_sha256"] == CURRENT
            and manifest["candidate"]["native"]["files"]["lib/libmadloop_tirust.so"] == CURRENT,
            "Default-stability historical/current native identity differs")
    require(all(observations.get(key) is False for key in ("current_campaign_bridge_replayed_for_this_fixture",
                "full_physical_amplitude_qualification", "performance_qualification", "production_fix_justified_by_this_case_alone"))
            and observations["same_fixture_other_backend_default_host_measurements"] == []
            and len(observations["rows"]) == 8, "Historical default-stability scope was broadened")


def triage_scope(analysis, manifest):
    require(manifest["id"] == analysis["campaign"] == "split-dp-terminal-1"
            and analysis["case"] == "qcd_n3_uux_zgg"
            and manifest["candidate"]["native"]["files"]["lib/libmadloop_tirust.so"] == CURRENT,
            "QCD n3 triage belongs to another campaign or native artifact")
    require(analysis["new_native_evaluations"] == 0 and analysis["performance_eligible"] is False
            and analysis["fpe_flags_observed"] is False and analysis["internal_retries_observed"] is False
            and analysis["higher_precision"]["stability_certified"] is False,
            "QCD n3 unchecked diagnostics or unobserved attempts promoted to qualification")
    points = analysis["target_points"]
    require(len(points) == len(set(points)) == 28, "QCD n3 target-point inventory differs")
    for backend in (6, 7, 8, 9):
        row = analysis["original_comparisons"]["candidate-" + str(backend)]
        require(row["passed"] is True and row["ordinary_target_passed"] is False
                and row["target_missed_points"] == points and row["target_missed_components"] == [1]
                and row["target_missed_coefficient_comparisons"] == 56,
                "Original QCD n3 finite target misses were changed")


def collect_stress(evidence, manifest):
    rows = []
    def row(title, text, name, *, native=HISTORICAL, status="open", detail=None, required=True):
        rows.append({"title": title, "status": status, "scope": "Pinned retained diagnostic; scientific adjudication is not rerun by the report",
                     "required_for_stress_acceptance": required,
                     "native_library_sha256": native, "text": text, "evidence": name, "detail": detail})
    core = bind_package(evidence, EW + "evidence-index.json")
    geometry = bind_package(evidence, EW + "geometry-audit/evidence-index.json")
    if core is not None and geometry is not None:
        bind_references(evidence, geometry["parent_indices"])
        bind_references(evidence, geometry["retained_source_and_input_bindings"])
        summary = evidence.json(EW + "summary.json")
        analysis = evidence.json(EW + "geometry-audit/analysis.json")
        ew_scope(core, geometry, summary, analysis, evidence.json(EW + "run.json"))
        row("EW threshold mass transport", "Historical 49af, private mass-transport prototype. QP squaring of existing MP masses reduces the recorded group-5 Gram discrepancy from 1.018% to 1.60e-20, while both arms still refuse 37 threshold tensor parents. Status 120 and accuracy −1 are unchecked; the TIRust sum is incomplete. Matching successful scalar/rank-one subsets does not qualify the failed tensors, full amplitude or performance.",
            EW + "geometry-audit/README.md", detail={"core_index": EW + "evidence-index.json", "geometry_index": EW + "geometry-audit/evidence-index.json"})
    else:
        row("EW threshold mass transport", "Frozen transport and geometry packages are both required; the original threshold remains unqualified.", None, status="pending")

    if bind_package(evidence, QCD + "evidence-index.json") is not None:
        adjudication = evidence.json(QCD + "analysis1/adjudication.json")
        guard = evidence.json(QCD + "analysis1/guard-source-audit.json")
        bind_references(evidence, adjudication["bindings"])
        qcd_scope(adjudication, guard)
        row("QCD soft/collinear prototype", "Historical 49af, six private replays. Topology correction plus Born evaluation at repaired coordinates gives NinjaRS status 322 and a 1.33e-15 analytic double-pole error; topology-only status 322 still misses 3e-6. Finite/single-pole truth remains unavailable. TIRust retains 51 nonsingular conditioning refusals and failed status 420; its incomplete sums are not amplitudes accepted for physics.",
            QCD + "RESULTS.md", detail={"normalization": adjudication["normalization"], "guard_summary": guard["summary"]})
    else:
        row("QCD soft/collinear prototype", "The frozen six-run prototype package is unavailable; full-amplitude qualification remains open.", None, status="pending")

    if bind_package(evidence, DEFAULT + "index.json") is not None:
        observations = evidence.json(DEFAULT + "observations.json")
        bindings = evidence.json(DEFAULT + "source-bindings.json")
        bind_references(evidence, bindings["historical_repository_files"])
        default_scope(observations, manifest)
        row("Default stability estimator", "Historical 49af synthetic host-helper case: default 1e-3 accepts correlated DP error beyond 3e-6. The imaginary symmetric error is 7.03e-4, but ordinary relative error is 1.40e-3. Shared 1e-6 rescues this helper to QP; the heuristic is not a truth-error bound. This fixture was not replayed on current eb3d or the other reducers’ default host paths. Default robustness remains open.",
            DEFAULT + "README.md", detail={"observations": observations, "current_native_replay": "not performed"})
    else:
        row("Default stability estimator", "The historical correlated-error limitation is unresolved; its frozen scope audit is unavailable.", None, status="pending")

    if bind_package(evidence, TRIAGE + "index.json") is not None:
        analysis = evidence.json(TRIAGE + "analysis.json")
        inputs = evidence.json(TRIAGE + "inputs.json")
        bind_references(evidence, inputs["files"])
        triage_scope(analysis, manifest)
        row("QCD n3 ordinary target", "Current eb3d campaign: all 100 points pass 3e-6, while the same 28 finite-term 1e-8 misses against CutTools remain. Retained independent QP comparisons support a CutTools DP reference bias and a smaller Ninja/NinjaRS discrepancy at point 39; a reducer cluster alone is not truth. Those historical fixed-QP results are unchecked status 120, not an automatic-rescue certificate. Original comparisons and acceptance gates are unchanged.",
            TRIAGE + "README.md", native=CURRENT, required=False, detail={"target_points": analysis["target_points"],
                "higher_precision_stability_certified": False, "internal_retries_observed": False,
                "fpe_flags_observed": False, "new_native_evaluations": 0})
    else:
        row("QCD n3 ordinary target", "Current ordinary-target misses require a frozen diagnosis; no acceptance result is altered.", None, native=CURRENT, status="pending", required=False)

    # Preserve the existing CMS disclosure; this task adds no new CMS result.
    cms = "experiments/complex-mass/README.md"
    if evidence.exists(cms):
        evidence.tree("experiments/complex-mass")
        row("Complex-mass extremes", "Nineteen moderate points agree; original extreme automatic-policy points remain unqualified. Private corrections are unchecked diagnostics.", cms, native=None)
    else:
        row("Complex-mass extremes", "Complex-mass stress qualification remains open.", None, status="pending")
    # No discovery of ongoing 512-bit work: a later result requires a reviewed
    # new report-source change and frozen dossier, never implicit promotion.
    row("QCD higher precision", "A runtime-qualified 512-bit capture is pending in this report. Provisional or incomplete runs do not establish higher-precision acceptance.", None, native=None, status="pending", required=False)
    return rows
