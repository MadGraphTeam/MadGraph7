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
backends' absolute estimates E of the same entry). Every result's lane
diagnostics must name the arm's backend as the effective one
(``reduction.effective``), and the two arms must differ somewhere: a TIRust
arm equal to the NinjaRS one bitwise on every output was not reduced by
TIRust. The check exits 0 when every output passes, 1 when one fails (or
the NinjaRS arm itself does not run), and 2 when the TIRust arm was not
exercised: TIRust does not serve the loop-induced observable calls yet
(``reduction.ReductionUnavailable``: the RESOLVED route or the executor's
``runs(reduction=)`` not wired, m-lit LIT-TIR), a result names another
effective backend, or the arms agree bitwise. It writes one JSON record
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
BACKENDS = ("ninjars", "tirust")
# PLAN §4.0 L6 (binary64): within ten times the larger estimate.
FACTOR = 10


class NotExercised(Exception):
    """A result of the TIRust arm was not reduced by TIRust."""


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _artifact(work: Path, name: str, expression: str) -> Path:
    from tests.unit.loop_v2_induced_observables_rows import generate

    output = work / name / "artifact"
    if not (output / "artifact.json").is_file():
        generate(work / name, name, expression)
    return output


def _effective(result, backend: str, label: str) -> None:
    """Every lane diagnostic of ``result`` names ``backend`` as effective."""

    found = {
        None if lane.reduction is None else lane.reduction.effective
        for lane in result.diagnostics
    }
    if found != {backend}:
        raise NotExercised(
            f"{label}: the {backend!r} arm's results name the effective backends "
            f"{sorted(map(str, found))}"
        )


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
        _effective(result, backend, label)
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
        _effective(result, backend, f"spin/{leg}/{history}")
        value = result.values[0][0]
        error = result.stability[0].absolute[0]
        out[f"spin/{leg}/{history}/p{p}"] = (complex(value.finite), float(error.finite))
    for resolution in ("contracted", "amplitude-basis"):
        resolved = runtime.evaluate_resolved(
            points, color_resolution=resolution, **options
        )
        _effective(resolved, backend, f"resolved/{resolution}")
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
    parser.add_argument("--rows", nargs="+", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from pyamplicol import Runtime
    from pyamplicol.api import VirtualStabilityChecks
    from pyamplicol.loops.v2.reduction import ReductionUnavailable
    from tests.unit.loop_v2_induced_observables_rows import ROWS

    checks = VirtualStabilityChecks(1, reduction_direction=True)
    record: dict = {"rule": "L6: |N - T| <= 10 max(E_N, E_T) (binary64)", "rows": {}}

    def write() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=1) + "\n")

    def not_exercised(reason: str) -> int:
        record["not_exercised"] = reason
        write()
        print(f"TIRust arm not exercised: {reason}")
        return 2

    failed = 0
    for name in args.rows or sorted(ROWS):
        runtime = Runtime.load(_artifact(args.work_dir, name, ROWS[name]), process="p")
        try:
            ninjars = _entries(name, runtime, "ninjars", checks)
        except (ReductionUnavailable, NotExercised) as error:
            # The reference arm must run: an error, not a missing TIRust.
            record["error"] = f"NinjaRS arm: {error}"
            write()
            print(f"FAIL: the NinjaRS arm does not run: {error}")
            return 1
        try:
            tirust = _entries(name, runtime, "tirust", checks)
        except (ReductionUnavailable, NotExercised) as error:
            return not_exercised(str(error))
        if all(tirust[label][0] == value for label, (value, _e) in ninjars.items()):
            return not_exercised(
                f"{name}: every output equals the NinjaRS arm's bitwise "
                f"({len(ninjars)} outputs)"
            )
        rows = []
        for label, (n, e_n) in ninjars.items():
            t, e_t = tirust[label]
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
        worst = max(
            (
                row["difference"] / row["bound"]
                if row["bound"]
                else (0.0 if not row["difference"] else float("inf"))
                for row in rows
            ),
            default=0.0,
        )
        print(f"{name}: {len(rows)} outputs, worst |N - T| / bound = {worst:.3g}")
    record["passed"] = not failed
    write()
    print("PASS" if not failed else f"FAIL ({failed} outputs)")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
