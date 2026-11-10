#!/usr/bin/env python3
"""Offline, allowlisted replay of a closed QCD n3 result; no native imports."""

import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from portable_inputs import read_input


HERE = Path(__file__).resolve().parent
CURRENT = "docs/benchmarkfolder/campaigns/invariant-symmetry-1/experiments/numerical/full100-warmup-qcd_n3_uux_zgg-1/results.json"
PREVIOUS = "docs/benchmarkfolder/campaigns/split-dp-terminal-1/experiments/numerical/full100-warmup-qcd_n3_uux_zgg-1/results.json"
TRIAGE = "docs/benchmarkfolder/campaigns/split-dp-terminal-1/experiments/qcd-n3-target-triage-1/"
PINS = {
    CURRENT: "59bf41a90d581fb757b2240b57162fc85c5e1f29ec14a1d54713a2255188b373",
    PREVIOUS: "8eb88c74c3c9496440c38fef052f335e4ffca851d36baa3a17a129177ffeecde",
    TRIAGE + "index.json": "10b5a3339dfba063c4c5a264d769f9e5615387965c503fb7a3e2e2d44c421915",
    TRIAGE + "inputs.json": "29dab3a329bce5577c3f353dff51bc90ca53a2dc5cdc26a2df97ec7ea5a736ac",
    TRIAGE + "analysis.json": "e7e10b434e3bf2eddd37e75ca2998f938800e674d19e6c5853a6ac991f436fea",
    TRIAGE + "README.md": "753037a6a592bcb9bb8d07663ca73d06dac1dcd426221ecbae2ce113f63e7fff",
    TRIAGE + "analyze.py": "c94b23ad7d76e2e11dd50ff3cd9ecc59a1e04916a2dd818f6aba7fd8516a3d02",
}
NAMES = {1: "CutTools", 6: "Ninja", 7: "COLLIER", 8: "NinjaRS", 9: "TIRust"}
COLUMNS = ["Born", "finite", "single pole", "double pole"]
STATUS = {211: (1, "DP"), 216: (6, "DP"), 217: (7, "DP"),
          218: (8, "DP"), 219: (9, "DP"), 323: (9, "QP")}


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def same_float(a, b):
    return a.hex() == b.hex()


def same_row(a, b):
    return (a.keys() == b.keys()
            and all(same_float(a[k], b[k]) if isinstance(a[k], float)
                    else all(same_float(x, y) for x, y in zip(a[k], b[k]))
                    if k == "coefficients" else a[k] == b[k] for k in a))


def raw_check(lane):
    obs = lane["observation"]
    assert set(obs["raw_outputs"]) == {"dimensions.dat", "warmup.dat"}
    assert obs["raw_outputs"]["dimensions.dat"].split() == ["1", "1", "1"]
    assert obs["warmup_first_call"] == 1 and obs["warmup_last_call"] == 100
    assert len(obs["rows"]) == 200
    parsed = []
    for line in obs["raw_outputs"]["warmup.dat"].splitlines():
        if line.startswith("#"):
            continue
        fields = line.split()
        assert len(fields) == 9
        call, point, order, status = map(int, fields[:4])
        values = list(map(float, fields[4:]))
        assert all(math.isfinite(x) for x in values) and values[4] >= 0
        parsed.append((call, point, order, status, values))
    assert [(c, p, o) for c, p, o, _, _ in parsed] == [
        (c, max(1, c), o) for c in range(101) for o in (0, 1)]
    for (_, p, o, status, values), row in zip(parsed[2:], obs["rows"]):
        assert (row["point"], row["order"], row["status"]) == (p, o, status)
        assert (row["actual_backend"], row["precision"]) == STATUS[status]
        assert row["actual_backend"] == lane["backend"]
        assert all(same_float(a, b) for a, b in zip(row["coefficients"], values[:4]))
        assert same_float(row["accuracy"], values[4])
    for zero, one in zip(obs["rows"][::2], obs["rows"][1::2]):
        assert same_row(zero | {"order": 1}, one)
    counts = Counter(str(r["status"]) for r in obs["rows"][::2])
    assert dict(counts) == obs["status_counts"]
    precision = Counter(f'{r["actual_backend"]}:{r["precision"]}' for r in obs["rows"][::2])
    assert dict(precision) == obs["backend_precision_counts"]
    return {"raw_rows": len(parsed), "accepted_rows": 200,
            "unique_physical_points": 100, "duplicated_orders": [0, 1],
            "status_counts_per_point": dict(counts),
            "accepted_backend_precision_per_point": dict(precision),
            "qp_points": [r["point"] for r in obs["rows"][::2] if r["precision"] == "QP"]}


def residual(reference, candidate, column):
    a, b = reference["coefficients"][column], candidate["coefficients"][column]
    delta, scale = abs(a - b), max(abs(a), abs(b))
    symmetric = delta / abs(a + b) if a + b else (0.0 if not delta else None)
    finite = max(abs(reference["coefficients"][1]), abs(candidate["coefficients"][1]))
    pole = scale / finite if column > 1 and finite else None
    return {"point": reference["point"], "order": reference["order"], "column": column,
            "reference": a, "candidate": b, "relative_error": delta / scale if scale else 0.0,
            "symmetric_relative_error": symmetric, "pole_to_finite": pole,
            "passed": ((symmetric is not None and symmetric < 3e-6)
                       or (pole is not None and pole < 3e-6))}


def passes(row, threshold):
    return ((row["symmetric_relative_error"] is not None and row["symmetric_relative_error"] < threshold)
            or (row["pole_to_finite"] is not None and row["pole_to_finite"] < threshold))


def comparisons(a, b):
    assert [(r["point"], r["order"]) for r in a] == [(r["point"], r["order"]) for r in b]
    return [residual(x, y, c) for x, y in zip(a, b) for c in range(4)]


def condensed(rows):
    missed = [r for r in rows if not passes(r, 1e-8)]
    worst = max(rows, key=lambda r: r["symmetric_relative_error"] or 0.0)
    return {"passed_3e6": all(passes(r, 3e-6) for r in rows),
            "passed_target_1e8": not missed, "coefficient_comparisons": len(rows),
            "target_missed_comparisons": len(missed),
            "target_missed_points": sorted({r["point"] for r in missed}),
            "target_missed_columns": sorted({r["column"] for r in missed}),
            "maximum_relative_error": max(r["relative_error"] for r in rows),
            "worst_symmetric_residual": worst}


def analyze():
    blobs, inputs = {}, {}
    for name, expected in PINS.items():
        blob = read_input(HERE, name, PINS)
        actual = hashlib.sha256(blob).hexdigest()
        assert actual == expected, (name, actual)
        blobs[name] = blob
        inputs[name] = {"sha256": actual, "bytes": len(blob)}
    # The previous triage is a secondary retained observation, not a fresh replay
    # of its 219 dependency files or an endorsement of unchecked QP truth.
    old_index = json.loads(blobs[TRIAGE + "index.json"])
    for name in ("inputs.json", "analysis.json", "README.md", "analyze.py"):
        assert old_index["files"][name] == inputs[TRIAGE + name]
    old_inputs = json.loads(blobs[TRIAGE + "inputs.json"])
    assert old_inputs["files"][PREVIOUS] == inputs[PREVIOUS]
    current, previous = (json.loads(blobs[name]) for name in (CURRENT, PREVIOUS))
    assert current["case"] == previous["case"]
    assert current["case"]["id"] == "qcd_n3_uux_zgg"
    keys = [("baseline", 1)] + [("candidate", b) for b in NAMES]
    assert [(x["source"], x["backend"]) for x in current["results"]] == keys
    assert [(x["source"], x["backend"]) for x in previous["results"]] == keys
    baseline = current["results"][0]["observation"]["rows"]
    lanes, original, prior_equality = {}, {}, {}
    for lane, old in zip(current["results"], previous["results"]):
        key = f'{lane["source"]}-{lane["backend"]}'
        assert lane["status"] == old["status"] == "complete"
        assert lane["performance_eligible"] is False
        lanes[key] = raw_check(lane)
        raw_check(old)
        rows, old_rows = lane["observation"]["rows"], old["observation"]["rows"]
        equality = [same_row(a, b) for a, b in zip(rows, old_rows)]
        assert all(equality)
        prior_equality[key] = {"all_accepted_rows_bit_identical": True,
                               "matched_rows": len(rows), "matched_coefficient_values": 4 * len(rows),
                               "embedded_raw_outputs_identical": lane["observation"]["raw_outputs"] == old["observation"]["raw_outputs"]}
        recomputed = comparisons(baseline, rows)
        retained = lane["comparison"]
        assert retained["residuals"] == recomputed
        assert retained["tolerance"] == 3e-6 and retained["ordinary_target_tolerance"] == 1e-8
        summary = condensed(recomputed)
        assert summary["passed_3e6"] == retained["passed"]
        assert summary["passed_target_1e8"] == retained["ordinary_target_passed"]
        assert summary["maximum_relative_error"] == retained["maximum_relative_error"]
        assert retained["failures"] == [r for r in recomputed if not r["passed"]]
        original[key] = summary | {"by_column": {
            COLUMNS[c]: condensed([r for r in recomputed if r["column"] == c]) for c in range(4)}}
    backend_rows = {x["backend"]: x["observation"]["rows"] for x in current["results"][1:]}
    for rows in backend_rows.values():
        assert all(same_float(x["coefficients"][0], y["coefficients"][0]) for x, y in zip(rows, baseline))
    assert all(same_row(x, y) for x, y in zip(baseline, backend_rows[1]))
    target_points = original["candidate-9"]["target_missed_points"]
    assert len(target_points) == 28
    assert all(original[f"candidate-{b}"]["target_missed_points"] == target_points for b in (6, 7, 8, 9))
    pairwise = {f"{a}-{b}": condensed(comparisons(backend_rows[a], backend_rows[b]))
                for a, b in itertools.combinations(NAMES, 2)}
    selected = {b: {r["point"]: r for r in rows if r["order"] == 0} for b, rows in backend_rows.items()}
    details, ranks = [], []
    for p in range(1, 101):
        finite = {str(b): selected[b][p]["coefficients"][1] for b in NAMES}
        absolute = {str(b): finite[str(b)] - finite["1"] for b in (6, 7, 8, 9)}
        spread = [residual(selected[a][p], selected[b][p], 1)
                  for a, b in itertools.combinations((6, 7, 8, 9), 2)]
        ranks.append({"point": p, "tirust_vs_ct": residual(selected[1][p], selected[9][p], 1),
                      "maximum_non_ct_pair": max(spread, key=lambda r: r["symmetric_relative_error"])})
        if p in target_points:
            details.append({"point": p, "accepted_results": {str(b): selected[b][p] for b in NAMES},
                            "signed_finite_delta_from_ct": absolute,
                            "same_side_of_ct": len({math.copysign(1.0, v) for v in absolute.values()}) == 1,
                            "pairwise_finite": {f"{a}-{b}": residual(selected[a][p], selected[b][p], 1)
                                                for a, b in itertools.combinations(NAMES, 2)}})
    old_analysis = json.loads(blobs[TRIAGE + "analysis.json"])
    assert old_analysis["target_points"] == target_points
    # Retain historical diagnostic provenance without treating status120 as checked.
    qp = old_analysis["higher_precision"]
    assert qp["stability_certified"] is False
    assert all(r["status"] == 120 and r["accuracy"] == -1.0 and not r["stability_checked"]
               for r in qp["qp28_rows"])
    return {"schema_version": 1, "case": current["case"], "campaign": "invariant-symmetry-1",
            "scope": "Closed result only; no campaign-wide qualification or new scientific evaluation",
            "new_native_evaluations": 0, "new_builds": 0, "performance_eligible": False,
            "inputs": inputs,
            "metrics": {"registered": "abs(a-b)/abs(a+b)<3e-6 or pole magnitude/finite maximum<3e-6",
                        "target": "same criterion at1e-8", "raw_relative": "abs(a-b)/max(abs(a),abs(b))",
                        "columns": COLUMNS},
            "observations": lanes, "original_comparisons": original, "pairwise_all100": pairwise,
            "target_points": target_points, "target_point_results": details,
            "all_target_points_same_side_of_ct": all(r["same_side_of_ct"] for r in details),
            "largest_tirust_ct_finite_differences": sorted(ranks, key=lambda r: r["tirust_vs_ct"]["symmetric_relative_error"], reverse=True)[:10],
            "non_ct_pairs_excluding_point39": {
                f"{a}-{b}": condensed([r for r in comparisons(backend_rows[a], backend_rows[b]) if r["point"] != 39])
                for a, b in itertools.combinations((6, 7, 8, 9), 2)},
            "previous_closed_result_equality": prior_equality,
            "historical_qp_context": {
                "source": TRIAGE + "analysis.json", "status": 120, "accuracy": -1.0,
                "stability_certified": False, "new_comparison_or_requalification": False,
                "retained_ct_qp_points": sorted({r["point"] for r in qp["qp28_rows"]}),
                "prior_diagnostic_claim": "Prior packet reports CT/Ninja/TIRust fixed-QP agreement on39/60/91/93 and CT-QP support for the28-point finite discrepancy diagnosis.",
                "qualification_limit": "This replay verifies the prior packet hash and identical ordinary observations, not native effective inputs, warning-free completeness, or the full prior219-file provenance chain."},
            "limitations": [
                "Accepted backend/precision statuses are final outcomes, not counts of internal attempts or tensor working precision.",
                "IEEE flags, internal retry records and native effective kinematics/model inputs are absent from the opened current result.",
                "A reducer cluster and stability accuracy estimates are not independent finite-term truth or a numerical error bound.",
                "Historical status120 is unchecked; it does not prove a complete sum or supply fresh same-input higher-precision qualification.",
                "No active qcd_n4, mutable campaign summary/status, loader, reducer, or timing was inspected or run."],
            "remaining_higher_precision_todos": [
                "Prioritize points93 (largest baseline discrepancy) and39 (largest non-CT disagreement); retain60/91 as secondary controls.",
                "Before a new finite/pole adjudication, bind and observe actual physical, repaired DP/MP, scale, mass, coupling and Born inputs across independently forced higher-precision reducers.",
                "Retain all statuses, IEEE state and output warnings; reject ##W03 omitted-contribution output before comparisons and keep status120 unchecked even when warning-free.",
                "Use independent higher-precision/analytic evidence for the finite term; a TIRust512 success or majority agreement alone does not establish truth.",
                "Do not change the registered3e-6 criterion or treat the1e-8 target miss as a newly failed acceptance gate."]}


def main():
    index = json.loads((HERE / "index.json").read_text())
    for name, expected in index["files"].items():
        assert Path(name).name == name and not (HERE / name).is_symlink()
        blob = (HERE / name).read_bytes()
        assert {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)} == expected
    result = analyze()
    assert encoded(result) == (HERE / "analysis.json").read_text()
    print("PASS: seven pinned closed inputs; 1,200 accepted rows; 28 finite-only target points; no native execution")


if __name__ == "__main__":
    main()
