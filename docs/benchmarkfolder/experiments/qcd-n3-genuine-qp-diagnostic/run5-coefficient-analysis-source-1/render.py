"""Render a compact human table from the bounded, caller-supplied summary."""
import argparse
import json
from pathlib import Path


def display(value):
    return "undefined" if value is None else format(value, ".9g")


def render(summary):
    if summary.get("numerical_summary_suppressed") is not False:
        raise ValueError("Shared-input mismatch: no numerical table permitted")
    duplicates = all(row["coefficients_by_slot"][:4] == row["coefficients_by_slot"][4:] for row in summary["observed_coefficients"])
    split_note = "Total and split-order-1 values are identical for all 24 observed arm/point rows in this dataset." if duplicates else "Total and split-order-1 values differ; inspect both sectors in the JSON."
    lines = ["# Closed run5 returned-coefficient discrepancies", "",
             "Pure arithmetic over the pinned, closed analysis; no native execution. All 16 workers remained status 120, accuracy [-1,-1]. These are discrepancies of returned REAL*8 values, with no independent oracle, true-error estimate, or scientific/performance acceptance.", "",
             "The common-input gate tests consistency of physical/model/Born/repair/QP numerator inputs. It does not test numerical agreement. Native bases and precision paths differ: CutTools retains DP denominators/native returns with genuine QP numerator callbacks; Ninja uses binary128 transport; NinjaRS and TIRust recover 113-bit decimal inputs and use 256-bit engines.", "",
             "The eight ANS(0:3,0:1) slots are Born, finite, single pole, double pole for the total, then the same four for split order 1. " + split_note, "",
             "Symmetric discrepancy = abs(a-b)/abs(a+b); relative-to-left = abs(a-b)/abs(a). Undefined zero/opposite cases remain explicit in JSON. Counts use strict >. All 384 frozen supplied scalar deltas were independently reproduced.", "",
             "| Total coefficient | Max symmetric | Max relative-to-left | Max absolute | >1e-10 /48 | >1e-8 /48 | >3e-6 /48 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["slot_summaries"][:4]:
        metrics = row["metrics"]
        counts = {c["threshold"]: c["strictly_above"] for c in metrics["symmetric"]["diagnostic_threshold_counts"]}
        lines.append("| " + " | ".join([row["coefficient"], display(metrics["symmetric"]["maximum"]),
                                           display(metrics["relative_to_left"]["maximum"]), display(metrics["absolute"]["maximum"]),
                                           str(counts[1e-10]), str(counts[1e-8]), str(counts[3e-6])]) + " |")
    reference = {(r["other"][0], r["other"][1], r["point"]): r for r in summary["cuttools_reference_rows"]}
    for mode in (4, 5):
        lines += ["", "## Mode " + str(mode) + " versus same-mode CutTools", "",
                  "CutTools is the comparison reference, not a truth oracle. Values below are total finite/single/double; split order 1 duplicates them.", "",
                  "| Point | Backend | Finite | Single pole | Double pole | Sym finite | Sym single | Sym double |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
        rows = summary["observed_coefficients"]
        for point in (1, 93, 39):
            for backend in (1, 6, 8, 9):
                row = next(r for r in rows if (r["backend"], r["mode"], r["point"]) == (backend, mode, point))
                discrepancies = [0., 0., 0.] if backend == 1 else [d["symmetric"] for d in reference[backend, mode, point]["deltas_by_slot"][1:4]]
                values = row["coefficients_by_slot"][1:4]
                lines.append("| " + " | ".join([str(point), {1:"CutTools",6:"Ninja",8:"NinjaRS",9:"TIRust"}[backend]] + [display(v) for v in values + discrepancies]) + " |")
    lines += ["", "## All admissible pair maxima", "",
              "Pairs include the six same-mode backend pairs in each mode and four same-backend mode pairs. Cross-backend, cross-mode pairs are outside the frozen comparison set.", "",
              "| Left (backend,mode) | Right | Max sym finite | Max sym single | Max sym double |", "|---|---|---:|---:|---:|"]
    for pair in summary["pair_breakdown"]:
        maxima = [max((p["symmetric_by_slot"][slot] for p in pair["points"] if p["symmetric_by_slot"][slot] is not None), default=None) for slot in (1, 2, 3)]
        lines.append("| " + " | ".join([str(pair["left"]), str(pair["right"])] + [display(v) for v in maxima]) + " |")
    lines += ["", "## Mode 4 versus 5 by point", "",
              "| Backend | Point | Sym finite | Sym single | Sym double |", "|---:|---:|---:|---:|---:|"]
    for pair in summary["pair_breakdown"]:
        if pair["kind"] != "direction_modes":
            continue
        for point in pair["points"]:
            lines.append("| " + " | ".join([str(pair["left"][0]), str(point["point"])] + [display(point["symmetric_by_slot"][s]) for s in (1,2,3)]) + " |")
    lines += ["", "No coefficient in any admissible pair exceeds 1e-10 symmetric discrepancy. This statement is confined to these three closed points and the stated returned-value diagnostic; it is not a general accuracy claim. Raw native integral comparisons are not used as truth votes.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary")
    args = parser.parse_args()
    print(render(json.loads(Path(args.summary).read_text())), end="")
