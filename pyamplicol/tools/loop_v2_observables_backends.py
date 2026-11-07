#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""The both-backends check of the loop-induced observables (m-lit LIT-OBS
Done; PLAN §4.0 rule L6; row LI-O1).

On the LI-O1 rows (g g > h g g and u u~ > h g g, every dipole T_i.T_j and
every gluon's spin class declared) every observable call kind runs once on
each reduction backend, ``reduction="ninjars"`` and ``reduction="tirust"``,
with the stability checks (one rotation and the reduction-direction check),
at the rows' reference points:

* colour-correlated: ``evaluate_correlated_many`` over "born" and every
  declared dipole;
* spin-correlated: ``evaluate_correlated`` with Recola's transverse vectors
  (``tests/fixtures/loop/required/li/correlations``), the colour metric and
  one dipole per gluon;
* resolved: ``evaluate_resolved`` contracted and on the amplitude-basis
  anchors;
* single-helicity selection: ``evaluate(helicities=[h])`` for every row.

Each output passes when |N - T| <= 10 max(E_N, E_T) (L6, binary64: the
backends' absolute estimates E of the same entry). The check exits 0 when
every output passes, 1 when one fails and 2 when TIRust does not serve the
loop-induced observable calls yet (``reduction.ReductionUnavailable``,
before LIT-TIR wires the RESOLVED route). It writes one JSON record
(``--output``) with every comparison.

Run (licence in the environment, the staged native of the slice)::

    source workspace/env.sh
    export SYMBOLICA_LICENSE=...   # grepped, never written down
    PYTHONPATH=<community build>:<worktree>/src python \\
        tools/loop_v2_observables_backends.py --work-dir <scratch> \\
        --output <scratch>/backends.json

``--work-dir`` keeps the generated artifacts (re-used when present).
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "loop" / "required" / "li"
ROWS = {"li_gg_hgg": "g g > h g g", "li_uux_hgg": "u u~ > h g g"}
BACKENDS = ("ninjars", "tirust")
# PLAN §4.0 L6 (binary64): within ten times the larger estimate.
FACTOR = 10


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _artifact(work: Path, name: str, expression: str) -> Path:
    sys.path.insert(0, str(ROOT))
    from tests.unit.test_loop_v2_induced_observables import _generate

    output = work / name / "artifact"
    if not (output / "artifact.json").is_file():
        _generate(work / name, name, expression)
    return output


def _entries(name: str, runtime, backend: str, checks) -> dict:
    """Every observable output of one row on one backend: label -> (value,
    absolute estimate) of the first sector's finite part per point."""

    from pyamplicol.correlators import CorrelatedRequest

    fixture = _fixture(name)
    points = fixture["points"]
    scales = [str(Fraction(mu) ** 2) for mu in fixture["mu_r"]]
    options = {"mu_r2": scales, "stability_checks": checks, "reduction": backend}
    out: dict = {}

    def keep(label, result, index):
        for p, (values, estimate) in enumerate(
            zip(result.values, result.stability, strict=True)
        ):
            value = values[0] if index is None else index(values)
            error = estimate.absolute[0] if index is None else index(estimate)
            out[f"{label}/p{p}"] = (complex(value.finite), float(error.finite))

    ids = [item.id for item in runtime.available_color_correlations()]
    many = runtime.evaluate_correlated_many(
        points, {i: CorrelatedRequest(i) for i in ids}, **options
    )
    for key, result in many.items():
        keep(f"correlated/{key}", result, None)
    recola = json.loads((FIXTURES / "correlations" / f"{name}-recola.json").read_text())
    for row in recola["rows"]:
        if row["kind"] not in ("sc", "scc"):
            continue
        leg = row["legs"][0]
        history = "born" if row["kind"] == "sc" else f"T{leg}T{row['legs'][1]}"
        p = row["point"]
        vector = tuple((Decimal(re), Decimal(im)) for re, im in row["vector"])
        result = runtime.evaluate_correlated_many(
            [points[p]],
            {"x": CorrelatedRequest(history, {leg: vector})},
            mu_r2=[scales[p]],
            stability_checks=checks,
            reduction=backend,
        )["x"]
        value = result.values[0][0]
        error = result.stability[0].absolute[0]
        out[f"spin/{leg}/{history}/p{p}"] = (complex(value.finite), float(error.finite))
    for resolution in ("contracted", "amplitude-basis"):
        resolved = runtime.evaluate_resolved(
            points, color_resolution=resolution, **options
        )
        for p, (values, estimates) in enumerate(
            zip(resolved.values, resolved.stability, strict=True)
        ):
            for h, helicity in enumerate(resolved.helicity_ids):
                for c, colour in enumerate(resolved.virtual_color_ids):
                    value = values[0][h][c]
                    error = estimates[h][c].absolute[0]
                    out[f"resolved/{resolution}/{helicity}/{colour}/p{p}"] = (
                        complex(value.finite),
                        float(error.finite),
                    )
        if resolution == "contracted":
            helicities = resolved.helicity_ids
    for helicity in helicities:
        selected = runtime.evaluate(points, helicities=[helicity], **options)
        keep(f"selection/{helicity}", selected, None)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", default=sorted(ROWS))
    args = parser.parse_args()

    from pyamplicol import Runtime
    from pyamplicol.api import VirtualStabilityChecks
    from pyamplicol.loops.v2.reduction import ReductionUnavailable

    checks = VirtualStabilityChecks(1, reduction_direction=True)
    record: dict = {"rule": "L6: |N - T| <= 10 max(E_N, E_T) (binary64)", "rows": {}}
    failed = 0
    for name in args.rows:
        runtime = Runtime.load(_artifact(args.work_dir, name, ROWS[name]), process="p")
        try:
            values = {b: _entries(name, runtime, b, checks) for b in BACKENDS}
        except ReductionUnavailable as error:
            record["unavailable"] = str(error)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=1) + "\n")
            print(f"TIRust does not serve the loop-induced observables yet: {error}")
            return 2
        rows = []
        for label, (n, e_n) in values["ninjars"].items():
            t, e_t = values["tirust"][label]
            bound = FACTOR * max(e_n, e_t)
            ok = abs(n - t) <= bound
            failed += not ok
            rows.append(
                {
                    "output": label,
                    "ninjars": [n.real, n.imag],
                    "tirust": [t.real, t.imag],
                    "difference": abs(n - t),
                    "bound": bound,
                    "passed": ok,
                }
            )
        record["rows"][name] = rows
        worst = max(row["difference"] / row["bound"] for row in rows if row["bound"])
        print(f"{name}: {len(rows)} outputs, worst |N - T| / bound = {worst:.3g}")
    record["passed"] = not failed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    print("PASS" if not failed else f"FAIL ({failed} outputs)")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
