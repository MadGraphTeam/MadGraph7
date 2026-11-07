#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""The both-backends check of the loop-induced observables (m-lit LIT-OBS
Done; PLAN §4.0 rule L6; row LI-O1).

On the LI-O1 rows (g g > h g g and u u~ > h g g, every dipole T_i.T_j and
every gluon's spin class declared) every observable call kind runs once on
each reduction backend, ``reduction="ninjars"`` and ``reduction="tirust"``,
with every pole (``poles="all"``), at the rows' reference points, in one
carrier (``--arithmetic``: binary64 by default, ``double-double`` or
``arbitrary`` with ``--precision`` digits):

* colour-correlated: ``evaluate_correlated_many`` over "born" and every
  declared dipole;
* spin-correlated: ``evaluate_correlated`` with Recola's transverse vectors
  (``tests/fixtures/loop/required/li/correlations``), the colour metric and
  one dipole per gluon;
* resolved: ``evaluate_resolved`` contracted and on the amplitude-basis
  anchors;
* single-helicity selection: ``evaluate(helicities=[h])`` for every row.

Every sector and Laurent coefficient (finite part and both poles) of every
output is compared. In binary64 an entry passes when |N - T| <= 10 max(E_N,
E_T) (L6: the backends' absolute estimates E of the same entry; ``checks()``:
the rescaled reduction re-test and three generic rotations, the recommended
configuration ``stability.RECOMMENDED_CHECKS`` with two more rotations).
Without the rescaled re-test no master integral is evaluated twice and the
estimate misses the masters' own rounding (DESIGN §10.4): at u u~ > h g g's
point 1 the binary64 error of T1.T4 is then 25 times its estimate (7 times
with the re-test). One rotation is one sample of the dressing's rounding,
which no reduction re-test repeats: against the double-double values (the
stand-in for an accurate second arm) the recommended checks leave 3 of the
2322 binary64 entries of the two rows beyond ten times their estimate (14.7
at u u~ > h g g's numerical-zero anchor amplitude-0 of h11, point 0, which
the first rotation moves by 3 % of its error; 10.2 at a spin output
and a g g > h g g anchor), two or three rotations none (at most 7.9 and 7.7
times); in double-double and arbitrary precision when |N - T| <=
1e-24 m (L6 after m-lit LIT-TIR-MP), m the largest |coefficient| of the
output's sector in either arm (``stability.relative``'s metric: a vanishing
pole is measured against its finite part), without checks (the rule reads
no estimate; the checks would cost about four evaluations per point).
Every result's lane diagnostics must name the arm's backend as the effective
one (``reduction.effective``), and the two arms must differ somewhere: a TIRust
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
    # the double-double arm of L6 (once TIRust serves double-double):
    ... tools/loop_v2_observables_backends.py --work-dir <scratch> \
        --output <scratch>/backends-dd.json --arithmetic double-double

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
# PLAN §4.0 L6: binary64 within ten times the larger estimate; double-double
# and arbitrary precision within 1e-24 of the sector's largest coefficient.
FACTOR = 10
RELATIVE = Decimal("1e-24")
# The binary64 rule's generic rotations (``default_rotation(0..2)``; module
# docstring), with the rescaled reduction re-test.
ROTATIONS = 3
COEFFICIENTS = ("finite", "single_pole", "double_pole")


class NotExercised(Exception):
    """A result of the TIRust arm was not reduced by TIRust."""


def checks():
    """The stability checks of the binary64 rule: the recommended
    configuration (``stability.RECOMMENDED_CHECKS``: the rescaled reduction
    re-test and ``default_rotation(0)``) with two more generic rotations."""

    from pyamplicol.api import VirtualStabilityChecks

    return VirtualStabilityChecks(ROTATIONS, reduction_direction=True, rescaled=True)


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


def _parts(value) -> tuple[Decimal, Decimal]:
    """A public scalar (float, complex, Decimal or ``CorrelatedValue``) as
    exact (real, imaginary) decimals."""

    real = getattr(value, "real", value)
    imag = getattr(value, "imag", 0)
    return Decimal(real), Decimal(imag)


def _laurents(values, estimate) -> list:
    """Per sector the (coefficient, (re, im), absolute estimate) of every
    computed Laurent coefficient; the estimate is 0 without checks
    (``estimate`` None: double-double and arbitrary precision)."""

    errors = [None] * len(values) if estimate is None else estimate.absolute
    out = []
    for laurent, error in zip(values, errors, strict=True):
        out.append(
            [
                (
                    name,
                    _parts(getattr(laurent, name)),
                    0.0 if error is None else float(getattr(error, name)),
                )
                for name in COEFFICIENTS
                if getattr(laurent, name) is not None
            ]
        )
    return out


def _estimates(stability, count: int) -> list:
    """A result's per-point estimates, None each without checks."""

    return [None] * count if stability is None else list(stability)


def _entries(name: str, runtime, backend: str, stability, carrier: dict) -> dict:
    """Every observable output of one row on one backend: label -> per
    sector the computed Laurent coefficients with their absolute
    estimates (``_laurents``) per point."""

    from pyamplicol.correlators import CorrelatedRequest

    fixture = _fixture(name)
    points = fixture["points"]
    scales = [str(Fraction(mu) ** 2) for mu in fixture["mu_r"]]
    settings = {
        "stability_checks": stability,
        "reduction": backend,
        "poles": "all",
        **carrier,
    }
    options = {"mu_r2": scales, **settings}
    out: dict = {}

    def keep(label, result):
        _effective(result, backend, label)
        estimates = _estimates(result.stability, len(result.values))
        for p, (values, estimate) in enumerate(
            zip(result.values, estimates, strict=True)
        ):
            out[f"{label}/p{p}"] = _laurents(values, estimate)

    ids = [item.id for item in runtime.available_color_correlations()]
    many = runtime.evaluate_correlated_many(
        points, {i: CorrelatedRequest(i) for i in ids}, **options
    )
    for key, result in many.items():
        keep(f"correlated/{key}", result)
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
            **settings,
        )["x"]
        label = f"spin/{leg}/{history}"
        _effective(result, backend, label)
        (estimate,) = _estimates(result.stability, 1)
        out[f"{label}/p{p}"] = _laurents(result.values[0], estimate)
    for resolution in ("contracted", "amplitude-basis"):
        resolved = runtime.evaluate_resolved(
            points, color_resolution=resolution, **options
        )
        _effective(resolved, backend, f"resolved/{resolution}")
        per_point = _estimates(resolved.stability, len(resolved.values))
        for p, (values, estimates) in enumerate(
            zip(resolved.values, per_point, strict=True)
        ):
            for h, helicity in enumerate(resolved.helicity_ids):
                for c, colour in enumerate(resolved.virtual_color_ids):
                    out[f"resolved/{resolution}/{helicity}/{colour}/p{p}"] = _laurents(
                        [sector[h][c] for sector in values],
                        None if estimates is None else estimates[h][c],
                    )
        if resolution == "contracted":
            helicities = resolved.helicity_ids
    for helicity in helicities:
        selected = runtime.evaluate(points, helicities=[helicity], **options)
        keep(f"selection/{helicity}", selected)
    return out


def _compare(ninjars: list, tirust: list, binary64: bool) -> list:
    """Per sector and coefficient of one output: the L6 comparison."""

    rows = []
    for sector, (n_sector, t_sector) in enumerate(zip(ninjars, tirust, strict=True)):
        largest = max(
            (
                abs(complex(*parts)) if binary64 else _modulus(parts)
                for _name, parts, _error in (*n_sector, *t_sector)
            ),
            default=0,
        )
        for (name, n, e_n), (other, t, e_t) in zip(n_sector, t_sector, strict=True):
            if name != other:
                raise ValueError(f"the arms computed different Laurent orders: {name}")
            difference = _modulus((n[0] - t[0], n[1] - t[1]))
            if binary64:
                bound = Decimal(FACTOR * max(e_n, e_t))
            else:
                bound = RELATIVE * Decimal(largest)
            rows.append(
                {
                    "sector": sector,
                    "coefficient": name,
                    "ninjars": [str(n[0]), str(n[1])],
                    "tirust": [str(t[0]), str(t[1])],
                    "difference": float(difference),
                    "bound": float(bound),
                    "passed": difference <= bound,
                }
            )
    return rows


def _modulus(parts) -> Decimal:
    from decimal import localcontext

    with localcontext() as context:
        context.prec = 80
        return (parts[0] * parts[0] + parts[1] * parts[1]).sqrt()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", default=None)
    parser.add_argument(
        "--arithmetic",
        choices=("double", "double-double", "arbitrary"),
        default="double",
    )
    parser.add_argument("--precision", type=int, default=None)
    args = parser.parse_args()
    binary64 = args.arithmetic == "double"
    carrier = {"arithmetic": args.arithmetic}
    if args.precision is not None:
        carrier["precision"] = args.precision

    sys.path.insert(0, str(ROOT))
    from pyamplicol import Runtime
    from pyamplicol.loops.v2.reduction import ReductionUnavailable
    from tests.unit.loop_v2_induced_observables_rows import ROWS

    # The estimates of the binary64 rule; the high-precision rule reads none.
    stability = checks() if binary64 else None
    rule = (
        "L6: |N - T| <= 10 max(E_N, E_T) (binary64)"
        if binary64
        else f"L6: |N - T| <= 1e-24 max |coefficient| ({args.arithmetic})"
    )
    record: dict = {"rule": rule, "carrier": carrier, "rows": {}}

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
            ninjars = _entries(name, runtime, "ninjars", stability, carrier)
        except (ReductionUnavailable, NotExercised) as error:
            # The reference arm must run: an error, not a missing TIRust.
            record["error"] = f"NinjaRS arm: {error}"
            write()
            print(f"FAIL: the NinjaRS arm does not run: {error}")
            return 1
        try:
            tirust = _entries(name, runtime, "tirust", stability, carrier)
        except (ReductionUnavailable, NotExercised) as error:
            return not_exercised(str(error))
        if all(
            [[parts for _n, parts, _e in sector] for sector in tirust[label]]
            == [[parts for _n, parts, _e in sector] for sector in value]
            for label, value in ninjars.items()
        ):
            return not_exercised(
                f"{name}: every output equals the NinjaRS arm's bitwise "
                f"({len(ninjars)} outputs)"
            )
        rows = []
        for label, value in ninjars.items():
            for row in _compare(value, tirust[label], binary64):
                failed += not row["passed"]
                rows.append({"output": label, **row})
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
