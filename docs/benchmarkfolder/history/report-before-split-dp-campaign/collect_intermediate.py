"""Freeze the explicitly selected, completed report evidence. No native calls.

Only the report supplement is written. Historical evidence is never modified;
active experiments are deliberately absent from this allowlist.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text())


def collect():
    files = {}

    def freeze(name):
        path = ROOT / name
        if path.is_symlink():
            raise ValueError("Evidence must be a regular retained file: " + name)
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        for candidate in candidates:
            if "__pycache__" in candidate.parts or not candidate.is_file():
                continue
            if candidate.is_symlink():
                raise ValueError("Evidence symlink: " + str(candidate))
            raw = candidate.read_bytes()
            files[candidate.relative_to(ROOT).as_posix()] = {
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    sections = []

    def section(title, note):
        result = {"title": title, "note": note, "claims": []}
        sections.append(result)
        return result["claims"]

    def claim(group, identifier, title, status, scope, text, evidence, assertions=()):
        for path in evidence:
            freeze(path)
        anchors = []
        for path, pointer in assertions:
            value = load(path)
            for key in pointer:
                value = value[key]
            anchors.append({"path": path, "pointer": pointer, "expected": value})
        group.append({"id": identifier, "title": title, "status": status,
                      "scope": scope, "text": text, "evidence": evidence,
                      "assertions": anchors, "performance_acceptance_eligible": False})

    implementation = section("Implemented changes and actual installation",
        "These are implementation and interface checks. The optimized candidate has no completed final five-backend physics/runtime campaign. Historical c91 measurements remain separate.")
    production = "experiments/n1-profile/production/"
    install = "qualification/tirust-install/current-source/"
    terminal = "experiments/dp-terminal-production/"
    for path in (production, install, terminal, "experiments/dp-terminal/"):
        freeze(path)
    claim(implementation, "workspace", "Separate full-DP workspaces", "pass", "Production source; native lifecycle parity",
        "One immutable plan retains separate primary/check workspaces; failure removes both workspaces and clears tensor results. Eleven native tests pass. The initial reconstructed release (92da56e2…) matched the original 49af release across 90 calls and 110 event records. QP and poles-only paths are unchanged by this optimization; QP is covered by separate unit/smoke checks.",
        [production+"README.md", production+"release-qualification.json", production+"release-validation.json"],
        [(production+"release-qualification.json", ["validation_passed"]), (production+"release-qualification.json", ["compared_event_records"])])
    claim(implementation, "terminal", "Guarded terminal Fortran update", "pass", "Frozen 8ae62e25d8; rank-1 × rank-1 Born interference",
        "The exporter proves the terminal consumer and floating-point prerequisites before selecting the specialized four-component update; otherwise it retains the original operation. Five CLI exports select 78 updates. The final n1 CutTools replay preserves all 100 points bit for bit. Tests and failed initial proof/configuration attempts are retained. Later higher-rank work is outside this snapshot; no complete five-backend runtime qualification is claimed.",
        [terminal+"README.md", terminal+"validation-summary.json", terminal+"compiled-final-tests.json"],
        [(terminal+"validation-summary.json", ["selected_updates"]), (terminal+"validation-summary.json", ["n1_replay"])])
    claim(implementation, "installer", "Actual current-source CLI publication", "pass", "Pinned downloads; cached build; isolated destination",
        "The actual installer used committed source 606603e720 and published DSO 095e957990c2…. Its real Fortran smoke covers DP, binary128 input, poles-only evaluation and caching. A separate raw-ABI replay against 49af gives 80 successes and 10 expected failures per library, with all 110 events, cache snapshots and output bytes identical. This is a cached installation, not a fresh-cache or full-amplitude claim. Main configuration drift during concurrent CLI activity is explicitly retained.",
        [install+"README.md", install+"check-installation.json", install+"release-validation.json", install+"config-drift-attribution.json"],
        [(install+"check-installation.json", ["native_artifacts"]), (install+"check-installation.json", ["final_native_replay"]), (install+"check-installation.json", ["main_configuration_unchanged"])])

    pilots = section("Later historical ordinary-point observations",
        "These three runs supplement the immutable eight-case pilot index. They use historical generated sources and library bytes, not the pending optimized candidate. Reported DP/QP counts are final accepted outcomes, not internal attempt counts. No timings here qualify performance acceptance.")
    for identifier, title, status, text in (
        ("ew-n3", "EW 2→3", "pass", "All five reducers complete the 100-point stream and pass both 3e-6 regression and the 1e-8 target. TIRust accepts 82 points in DP and 18 in QP. This establishes this historical sample only."),
        ("qcd-n4", "QCD 2→4", "partial", "Baseline/candidate CutTools, Ninja and COLLIER each complete 100 points and pass 3e-6. Ninja and COLLIER miss the 1e-8 target; their worst raw relative residuals against CutTools are 8.26e-7 and 8.30e-7, and they also disagree with each other. CutTools accepts 88 DP/12 QP; Ninja 97 DP/3 QP. Backend 8 has a resource/controller failure despite a saved raw stream; backend 9 times out before READY. Neither failed lane is qualified."),
        ("ew-n4", "EW 2→4", "partial", "CutTools, Ninja, COLLIER and NinjaRS complete their 100-point observations and pass 3e-6. Ninja, COLLIER and NinjaRS miss the 1e-8 target. CutTools accepts 93 DP/7 QP. TIRust reaches the unchanged 1,800-second startup deadline before completing warmup; partial warmup data are not a complete observation or performance result.")):
        path = "experiments/numerical/strict-policy-"+identifier+"-warmup-v2/"
        freeze(path)
        assertions = [(path+"results.json", ["performance_eligible"])]
        for index, row in enumerate(load(path+"results.json")["results"]):
            assertions += [(path+"results.json", ["results", index, key]) for key in ("status", "source", "backend")]
            if row["status"] == "complete":
                assertions += [(path+"results.json", ["results", index, "observation", "status_counts"]),
                    (path+"results.json", ["results", index, "comparison", "passed"]),
                    (path+"results.json", ["results", index, "comparison", "ordinary_target_passed"])]
            else:
                assertions.append((path+"results.json", ["results", index, "error"]))
        claim(pilots, "late-"+identifier.replace("-n", ""), title, status, "100 saved ordinary physical points; historical policy", text,
              [path+"results.json"], assertions)
    for path in ("experiments/numerical/qcd-n4-warmup-v2-index.json", "experiments/numerical/ew-n4-warmup-v2-index.json",
                 "experiments/numerical/qcd-n4-backend8-resource-failure/"):
        freeze(path)

    limitations = section("Retained accuracy and completeness limitations",
        "Independent pole identities, higher precision and native failure flags qualify different questions. An unchecked positive-mode code 120 does not establish that every tensor contributed, and a false failure flag alone does not prove accuracy.")
    ew = "experiments/ew-threshold-adjudication/"
    cms = "experiments/complex-mass/"
    n3 = "experiments/qcd-n3-adjudication/fpe-completeness/"
    cohorts = "experiments/cohorts/"
    born = "experiments/qcd-born-fallback-prototype/"
    for path in (ew, cms, n3, born, cohorts+"pilot-qcd-n1-tirust-fpe/", cohorts+"schema2-n1-qualification.md"):
        freeze(path)
    claim(limitations, "ew-threshold", "EW threshold remains unqualified", "fail", "Frozen original-policy stress and private tensor diagnostics",
        "TIRust serves all 200 ordinary-control groups but refuses 37 of 200 threshold groups; automatic code 420 correctly rejects the point. Forced-QP code 120 contains an incomplete sum. An independent pole-ratio identity agrees with NinjaRS to 1.68e-11, while accepted Ninja QP differs by 111 in relative ratio despite its host estimate. Private higher-precision tensor experiments do not establish a repaired full amplitude; the finite coefficient remains unresolved.",
        [ew+"CURRENT-STATUS.md", ew+"summary.json", ew+"tensor-replay/analytic-poles/summary.json"])
    claim(limitations, "cms", "Complex-mass extremes remain open", "partial", "Original automatic policy versus private unchecked corrections",
        "All five reducers pass the 19 moderate points. The two extreme points remain unqualified under the original automatic policy. Private CutTools scalar-cutoff and QP denominator-transport corrections resolve one fixed-QP discrepancy to 2.11e-11 and agree with an independent double-pole formula. Those results remain unchecked diagnostics, with no promoted CutTools change or automatic stability qualification.",
        [cms+"README.md", cms+"summary.json", cms+"direct-qp/cuttools-adjudication/qp-transport/summary.json"])
    claim(limitations, "n3-completeness", "QCD 2→3 QP completeness supplement", "pass", "Four retained fixed-QP diagnostic points",
        "At original points 39, 60, 91 and 93, the added TIRust trace observes both host failure flags false on all nine public calls, and the final values exactly match the original QP adjudication. An independently pinned reparse covers the earlier parser-provenance omission. The result supports reported completeness for these calls; it does not turn unchecked QP into automatic acceptance.",
        [n3+"README.md", n3+"verification-1/verification.json"], [(n3+"verification-1/verification.json", ["status"]), (n3+"verification-1/verification.json", ["flags"])])
    claim(limitations, "cohorts", "Native-DP and rescued cohort protocol", "partial", "Private tracing driver; acceptance executable unchanged",
        "The complete 100-point stream is warmed before cohort membership is frozen. Per-call status, absolute NinjaRS retry counters and DP/QP failure flags expose compensated DP, rescue and migrations. The schema-2 n1 run has no flagged calls or migrations, but its seven timed batches last only 0.800–0.809 seconds: the diagnostic median is null. Final-candidate DP/QP cohort measurements are pending; internal algorithm attempts are not reconstructed.",
        [cohorts+"schema2-n1-qualification.md", cohorts+"pilot-qcd-n1-tirust-fpe/results.json"], [(cohorts+"pilot-qcd-n1-tirust-fpe/results.json", ["cohorts", "dp", "eligible_diagnostic"]), (cohorts+"pilot-qcd-n1-tirust-fpe/results.json", ["cohorts", "dp", "median_seconds_per_point"])])
    claim(limitations, "born-prototype", "Private Born fallback does not repair stress", "fail", "Unpromoted prototype; same original input",
        "The Born-only fallback and host edge fixtures pass their local arithmetic, exception and policy checks. The full soft/collinear physical request still fails checked loop evaluation with code 420. A finite repaired Born term alone does not qualify the virtual amplitude; this prototype remains private.",
        [born+"STATUS.md", born+"amplitude-native-1/adjudication.json", born+"tree-native-2/analysis.json"])

    rejected = section("Private optimization experiments not promoted",
        "These experiments preserve their failures and limitations. Shared-host diagnostic intervals cannot replace the per-process acceptance gate. The earlier prefix optimization remains separately documented; no earlier NinjaRS gain is attributed to these changes.")
    claim(rejected, "terminal-bulk", "Bulk terminal-guard alternative", "unpromoted", "Private alternative to the selected elemental guard",
        "All coefficients remain bit-identical. Relative to historical NinjaRS, the diagnostic ratio is 1.04235 [1.03396, 1.05081]; its upper bound exceeds 1.05. The faster elemental variant shown earlier does not erase this retained result. Neither interval is a formal campaign measurement.",
        ["experiments/dp-terminal/README.md", "experiments/dp-terminal/analysis-summary.json"],
        [("experiments/dp-terminal/analysis-summary.json", ["variants", "packed", "whole_amplitude_diagnostic", "paired_statistics", "terminal_over_baseline"])])
    for identifier, title, dirname, text in (
        ("adapter-o3", "Adapter-only −O3", "whole-amplitude/split-adapter-o3-1", "Exact amplitude parity; variant/original ratio 0.99442 with interval [0.98317, 1.00580]. No established benefit."),
        ("adapter-scratch", "Reusable scratch and pointer mapping", "fortran-scratch", "Native layout checks and exact amplitude parity pass. Ratio 0.99748 [0.98826, 1.00679] establishes no whole-amplitude gain."),
        ("adapter-sequence", "Sequence-associated tensor layout", "fortran-sequence", "Native layout checks and exact amplitude parity pass. Ratio 1.00073 [0.99014, 1.01143] establishes no gain."),
        ("common-allocation", "Common Fortran allocation changes", "fortran-common-allocation", "Five reducers × 100 points preserve rows, status and accuracy exactly. Ratio 1.01120 [0.99373, 1.02898] establishes no gain."),
        ("finite-scan", "Remove adapter finite-value scan", "fortran-dp-finite", "The defensive-contract audit rejects removing the public wrapper's independent finite-value check. No accepted native throughput result is claimed.")):
        path="experiments/n1-profile/"+dirname+"/"
        freeze(path)
        evidence = path+("summary.json" if identifier=="adapter-o3" else "README.md")
        if not (ROOT/evidence).is_file(): evidence=path+"evidence-index.json"
        claim(rejected, identifier, title, "rejected" if identifier=="finite-scan" else "unpromoted",
              "Private diagnostic; unchanged production selection", text, [evidence])

    diagnostic_timings = []
    for title,path,pointer in (
        ("DP workspace split / original TIRust", "experiments/n1-profile/whole-amplitude/split-workspace-1/summary.json", ["paired_statistics", "split_over_original"]),
        ("DP workspace split / historical NinjaRS baseline", "experiments/n1-profile/whole-amplitude/split-workspace-1/summary.json", ["paired_statistics", "split_over_baseline"]),
        ("Terminal update / original TIRust", "experiments/dp-terminal/analysis-summary.json", ["variants", "elemental", "whole_amplitude_diagnostic", "paired_statistics", "terminal_over_original"]),
        ("Terminal update / historical NinjaRS baseline", "experiments/dp-terminal/analysis-summary.json", ["variants", "elemental", "whole_amplitude_diagnostic", "paired_statistics", "terminal_over_baseline"])):
        freeze(str(Path(path).parent))
        value=load(path)
        for key in pointer:value=value[key]
        diagnostic_timings.append({"title":title,"path":path,"pointer":pointer,"statistics":value,
                                   "performance_acceptance_eligible":False})
    failure="runs/qcd_n1_uux_z/tirust-first-attempt-3/summary.json"
    freeze(str(Path(failure).parent))
    freeze("history/report-before-status-refresh/index.json")
    # Bind old displayed qualification prose and its complete frozen packages.
    for path in ("qualification/comparison-backends", "qualification/poles-only",
                 "qualification/tirust-install", "qualification/tirust-bridge",
                 "qualification/madloop-adapter", "qualification.json"):
        freeze(path)
    freeze("experiments/physical-stress")
    freeze("experiments/fortran-prefix")
    freeze("experiments/poles-diagnostics")
    freeze("experiments/numerical/completed-pilots-index.json")
    for name in load("experiments/numerical/completed-pilots-index.json")["files"]:
        freeze(name)
    installer=load(install+"check-installation.json")
    return {"schema":"madloop-tirust-intermediate-v1","kind":"intermediate",
        "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "final_campaign_status":"pending","final_campaign_accepted":False,"new_measurements_performed":False,
        "sections":sections,"diagnostic_timings":diagnostic_timings,
        "authoritative_failure":{"case":"qcd_n1_uux_z","path":failure,"statistics":load(failure)["statistics"]},
        "installer_identity":{"path":install+"check-installation.json","library_sha256":installer["native_artifacts"]["libmadloop_tirust.so"]["sha256"],"source_revision":installer["source_revision"]},
        "excluded_active_work":["experiments/dp-higher-rank","experiments/ew-threshold-geometry-supplement"],
        "qualification":load("qualification.json"),"files":dict(sorted(files.items()))}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh",action="store_true",help="Replace only the current supplement after reviewing newly frozen evidence")
    args=parser.parse_args()
    result=collect()
    with (ROOT/"intermediate-status.json").open("w" if args.refresh else "x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    print("Frozen",len(result["files"]),"report dependencies; no numerical execution")
