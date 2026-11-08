#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""The rank > N+2 execution check of TIR-only artifacts (m-lit LIT-EFT, row
LI-X1 as amended by the USER 2026-11-06 ~14:55: run-only, no benchmark, no
provider reference, NO correctness or gauge-invariance claim).

Numerators beyond NinjaRS's rank <= N+2 have no correct rational term in any
model we have (dimension-eight operators would provide it): their results run
through TIRust and are not validated physics. What this check shows is that
such an artifact generates, loads and evaluates through TIRust, that NinjaRS
refuses it typed, and that the two backends agree wherever NinjaRS can serve.

Rows (HC_NLO_X0 v1.3, the heft restriction with the card's parameters and
cosa = 1, top vetoed, QCD corrections, model R2/UV, full colour, MU_R = 125):

* ``x0x0``: g g > x0 x0, two x0 G G insertions (QNP = 2, the order of one
  dimension-six operator pair). By power counting in Feynman gauge a vertex
  of a dimension-five or -six operator with both loop legs raises the rank by
  at most one, so two insertions reach rank N+2 (the box at rank 6) and no
  further: NinjaRS serves every family. L6 on the public API (the summed
  virtual, every sector and Laurent coefficient; ``tools/loop_v2_observables_
  backends.py``'s comparison: |N - T| <= 10 max(E_N, E_T) in binary64, E from
  three generic rotations and the rescaled re-test; in double-double within
  1e-24 of the sector's largest coefficient).
* ``x0x0x0``: g g > x0 x0 x0, three insertions (QNP = 3): its pentagons reach
  rank N+3, so the artifact is TIR-only (declared ``reductions =
  ["tirust"]``):
  - ``auto`` resolves to TIRust with its reason in binary64, double-double
    and arbitrary precision, and on every level of a rescue ladder (no
    NinjaRS fallback); an explicit NinjaRS, a NinjaRS rescue level and the
    cross-backend check are refused typed (``ReductionUnavailable``). In
    binary64 TIRust's conditioning gates (and its method choice's check)
    refuse the rank-8 pentagons' sub-boxes at most of these points, which
    then fail closed with TIRust's reason; double-double and arbitrary
    precision serve every point, and the default rescue ladder moves the
    failed points to double-double (TIRust there too);
  - the native runtime the SDKs use (``evaluate_v2_artifact``) serves it on
    TIRust by default, bitwise the Python API's binary64 values;
  - the families NinjaRS reduces (rank <= N+2) agree lane by lane (the
    executor's family mask: NinjaRS dresses those families only, TIRust
    every family): |T - N| <= 10 max(E_N, E_T, 4 u |N|) + 1e-12 |N| per
    family, loop order and sector, every Laurent coefficient, E the lane's
    rescaled re-test deviation (``tests/unit/test_loop_v2_tir.py``'s T2
    rule; a single family is not a gauge-invariant quantity, so rotations
    estimate nothing per lane).

Run (licence in the environment, the staged native of the package)::

    source workspace/env.sh
    export SYMBOLICA_LICENSE=...   # grepped, never written down
    PYTHONPATH=<community build>:<worktree>/src python \\
        tools/loop_v2_eft_execution.py --work-dir <scratch> \\
        --output <scratch>/eft.json

``--model`` is the HC_NLO_X0 UFO directory (default
``PYAMPLICOL_HC_UFO_REFERENCE``, else the workspace copy); ``--work-dir``
keeps the generated artifacts (re-used when present). Exit 0 when every
check passes, 1 otherwise; one JSON record (``--output``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "tools" / "loop_reference" / "hc_nlo_x0.json"
WORKSPACE_MODEL = Path(
    "/shared/vhirschi/t3code/pyamplicol/workspace/reference/sources/hc-nlo-x0/"
    "extract/HC_NLO_X0_UFO"
)
# The rows: expression, the largest family rank excess over N expected.
ROWS = {"x0x0": ("g g > x0 x0", 2), "x0x0x0": ("g g > x0 x0 x0", 3)}
# The points' sqrt(s): twelve values from just above the three-x0 threshold
# to 13 TeV (binary64 TIRust serves some of them, module docstring).
SQRT_S = tuple(
    Fraction(x)
    for x in (500, 600, 700, 900, 1000, 1500, 2500, 4000, 6000, 8000, 10000, 13000)
)
MU_R = "125"
# The reason of ``auto`` on a TIR-only artifact (``reduction.resolve``).
ONLY = "auto: TIRust, the only backend the artifact admits"
# test_loop_v2_tir.py's T2 rule on raw lanes; L6's double-double rule.
MARGIN = 10.0
FLOOR = 4 * 2.0**-53
ROUNDING = 1e-12
DOUBLE_DOUBLE = 1e-24
FLOOR_DD = 4 * 2.0**-104
# Points of the arbitrary-precision run (40 digits).
ARBITRARY_POINTS = 3


def model_path(given: Path | None) -> Path:
    if given is not None:
        return given
    configured = os.environ.get("PYAMPLICOL_HC_UFO_REFERENCE")
    return Path(configured) if configured else WORKSPACE_MODEL


def _card() -> dict:
    return json.loads(CARD.read_text())


def loop_card(model: Path, expression: str, output: Path) -> str:
    """The generation card of a row (``test_loop_v2_hc._loop_card``'s, heft
    restriction, top vetoed)."""

    card = _card()
    loop = card["loop_model"]
    scheme = "\n".join(f"{k} = {json.dumps(v)}" for k, v in loop["scheme"].items())
    fixed = ", ".join(
        f"{name} = {json.dumps(value)}"
        for name, value in card["fixed_parameters"].items()
    )
    return f"""
action = "generate"
[model]
source = {json.dumps(str(model))}
restriction = "heft"
[process]
entries = [{{expression = {json.dumps(expression)}, name = "p"}}]
loop_coupling_orders = {json.dumps(loop["loop_coupling_orders"])}
coupling_order_policy = "minimal"
[color]
accuracy = "full"
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{{fixed}}}
[loop.model]
loop_coupling_orders = {json.dumps(loop["loop_coupling_orders"])}
uv_complete = {json.dumps(loop["uv_complete"])}
model_r2_complete = {json.dumps(loop["model_r2_complete"])}
[loop.model.scheme]
{scheme}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
particle_veto = ["t"]
"""


def generate(work: Path, key: str, model: Path) -> Path:
    """The artifact of row ``key`` under ``work`` (generated once, with the
    post-build validation on)."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    output = work / key / "artifact"
    if not (output / "artifact.json").is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        path = work / key / "card.toml"
        path.write_text(loop_card(model, ROWS[key][0], output))
        Generator(resolve_config(path)).generate(
            ProcessSet((ProcessRequest.parse(ROWS[key][0], name="p"),)), output
        )
    return output


def parameters() -> dict[str, str]:
    """The heft restriction's external parameters (cosa = 1) without the
    fixed ones (the card's)."""

    card = _card()
    fixed = card["fixed_parameters"]
    return {k: v for k, v in card["restrictions"]["heft"].items() if k not in fixed}


def points(key: str) -> tuple[list[list[list[str]]], list[str]]:
    """Three on-shell points of row ``key`` (``supplied_points``, 30 digits;
    the x0 mass 125) and mu_R² per point."""

    reference = ROOT / "tools" / "loop_reference"
    if str(reference) not in sys.path:
        sys.path.insert(0, str(reference))
    import supplied_points

    finals = ROWS[key][0].split(">")[1].split()
    masses = [Fraction(0), Fraction(0)] + [Fraction(125)] * len(finals)
    found, _scales, _labels = supplied_points.generate(
        masses, count=len(SQRT_S), sqrt_s_values=list(SQRT_S)
    )
    text = [[[str(x) for x in leg] for leg in point] for point in found]
    return text, [str(Fraction(MU_R) ** 2)] * len(text)


def runtime(artifact: Path):
    from pyamplicol import Runtime

    loaded = Runtime.load(artifact, process="p")
    loaded.set_model_parameters(parameters())
    return loaded


def _effective(result) -> list:
    return sorted(
        {
            (lane.reduction.requested, lane.reduction.effective, lane.reduction.reason)
            for lane in result.diagnostics
            if lane.reduction is not None
        }
    )


def _values(result) -> list:
    return [
        None if point is None else [[str(x) for x in laurent] for laurent in point]
        for point in result.values
    ]


def public_agreement(loaded, pts, scales) -> dict:
    """L6 on the public API's summed virtual (``x0x0``): both backends with
    three generic rotations and the rescaled re-test, on the points TIRust
    serves in binary64 (the others fail closed with its reason, recorded)."""

    sys.path.insert(0, str(ROOT / "tools"))
    from loop_v2_observables_backends import _compare, _laurents, checks

    arms, failed = {}, {}
    for backend in ("ninjars", "tirust"):
        result = loaded.evaluate(
            pts,
            mu_r2=scales,
            stability_checks=checks(),
            reduction=backend,
            on_failure="return" if backend == "tirust" else "raise",
        )
        assert all(lane.reduction.effective == backend for lane in result.diagnostics)
        failed[backend] = {
            p: (lane.detail or "")[:300]
            for p, lane in enumerate(result.diagnostics)
            if lane.status != "success"
        }
        arms[backend] = [
            None if p in failed[backend] else _laurents(values, estimate)
            for p, (values, estimate) in enumerate(
                zip(result.values, result.stability, strict=True)
            )
        ]
    rows = []
    for p, (n, t) in enumerate(zip(arms["ninjars"], arms["tirust"], strict=True)):
        if t is not None:
            rows += [{"point": p, **row} for row in _compare(n, t, True)]
    differ = any(row["difference"] > 0 for row in rows)
    double_double = double_double_agreement(loaded, pts, scales)
    return {
        "rule": "L6: |N - T| <= 10 max(E_N, E_T), E from three generic rotations "
        "and the rescaled re-test (binary64)",
        "tirust_failed": failed["tirust"],
        "rows": rows,
        "passed": bool(rows)
        and all(row["passed"] for row in rows)
        and differ
        and not failed["ninjars"]
        and double_double["passed"],
        "worst": max(
            (r["difference"] / r["bound"] for r in rows if r["bound"]), default=0.0
        ),
        "backends_differ": differ,
        "double_double": double_double,
    }


def double_double_agreement(loaded, pts, scales) -> dict:
    """L6 in double-double on the public API's summed virtual (``x0x0``):
    |N - T| <= 1e-24 of the sector's largest coefficient, no checks
    (``tools/loop_v2_observables_backends.py``'s extended-carrier rule)."""

    from loop_v2_observables_backends import _compare, _laurents

    arms = {}
    for backend in ("ninjars", "tirust"):
        result = loaded.evaluate(
            pts, mu_r2=scales, arithmetic="double-double", reduction=backend
        )
        assert all(lane.reduction.effective == backend for lane in result.diagnostics)
        arms[backend] = [_laurents(values, None) for values in result.values]
    rows = []
    for p, (n, t) in enumerate(zip(arms["ninjars"], arms["tirust"], strict=True)):
        rows += [{"point": p, **row} for row in _compare(n, t, False)]
    return {
        "rule": "L6: |N - T| <= 1e-24 max |coefficient| of the sector (double-double)",
        "rows": len(rows),
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "worst": max(
            (r["difference"] / r["bound"] for r in rows if r["bound"]), default=0.0
        ),
    }


def _refused(call) -> str | None:
    """The typed refusal of ``call`` (``ReductionUnavailable``), else None."""

    from pyamplicol.loops.v2.reduction import ReductionUnavailable

    try:
        call()
    except ReductionUnavailable as error:
        return str(error)
    return None


def _size(z) -> float:
    return abs(complex(float(z.real), float(z.imag)))


def _statuses(result) -> list:
    return [
        {
            "status": lane.status,
            "effective": None if lane.reduction is None else lane.reduction.effective,
            "detail": None if lane.status == "success" else (lane.detail or "")[:300],
        }
        for lane in result.diagnostics
    ]


def _relative(a_values, b_values) -> float:
    """The largest deviation of two points' sectors, per Laurent
    coefficient relative to the sector's largest one."""

    worst = 0.0
    for a, b in zip(a_values, b_values, strict=True):
        scale = max(abs(float(x)) for x in (*a, *b)) or 1.0
        for x, y in zip(a, b, strict=True):
            worst = max(worst, abs(float(Decimal(str(x)) - Decimal(str(y)))) / scale)
    return worst


def tir_only_checks(artifact: Path, loaded, pts, scales) -> dict:
    """The TIR-only artifact's execution checks (``x0x0x0``; module
    docstring)."""

    from pyamplicol.api import (
        VirtualRescueLevel,
        VirtualRescuePolicy,
        VirtualStabilityChecks,
    )
    from pyamplicol.loops.native import load_loop_native

    described = loaded.inspect()
    out: dict = {
        "declared_reductions": described.get("reductions"),
        "inspect_reduction": described.get("reduction"),
    }
    runs: dict = {}
    # Binary64 with one rotation and the rescaled re-test: TIRust serves the
    # points its conditioning gates (and method choice) accept, the others
    # fail closed with its reason (on_failure="return").
    binary64 = loaded.evaluate(
        pts,
        mu_r2=scales,
        stability_checks=VirtualStabilityChecks(1, True, rescaled=True),
        on_failure="return",
    )
    runs["double"] = {"reduction": _effective(binary64), "points": _statuses(binary64)}
    results = {"double": binary64}
    for arithmetic, precision, count in (
        ("double-double", None, len(pts)),
        ("arbitrary", 40, ARBITRARY_POINTS),
    ):
        result = loaded.evaluate(
            pts[:count],
            mu_r2=scales[:count],
            arithmetic=arithmetic,
            precision=precision,
            on_failure="return",
        )
        runs[arithmetic] = {
            "reduction": _effective(result),
            "points": _statuses(result),
        }
        results[arithmetic] = result
    served = {
        name: [p for p, lane in enumerate(r.diagnostics) if lane.status == "success"]
        for name, r in results.items()
    }
    runs["served"] = {
        name: f"{len(v)}/{len(results[name].diagnostics)}" for name, v in served.items()
    }
    # TIRust's own consistency across carriers (no correctness claim).
    runs["binary64_vs_double_double"] = max(
        (
            _relative(results["double"].values[p], results["double-double"].values[p])
            for p in served["double"]
        ),
        default=None,
    )
    runs["double_double_vs_arbitrary"] = max(
        (
            _relative(
                results["double-double"].values[p], results["arbitrary"].values[p]
            )
            for p in served["arbitrary"]
            if p in served["double-double"]
        ),
        default=None,
    )
    runs["values_double_double"] = _values(results["double-double"])
    out["runs"] = runs
    # The default rescue ladder (binary64, double-double, arbitrary): a point
    # TIRust fails closed in binary64 moves on, every level on TIRust.
    rescued = loaded.evaluate(pts, mu_r2=scales, stability=VirtualRescuePolicy())
    # A ladder whose binary64 and double-double targets no point meets: one
    # point reaches the last level.
    forced = loaded.evaluate(
        pts[:1],
        mu_r2=scales[:1],
        stability=VirtualRescuePolicy(
            (
                VirtualRescueLevel("double", target=1e-300),
                VirtualRescueLevel("double-double", target=1e-300),
                VirtualRescueLevel("arbitrary", 40),
            )
        ),
    )

    def attempts(result) -> list:
        return [
            [
                {
                    "arithmetic": attempt.arithmetic,
                    "effective": attempt.reduction.effective,
                    "reason": attempt.reduction.reason,
                    "failure": None
                    if attempt.failure is None
                    else attempt.failure[:200],
                }
                for attempt in record.attempts
            ]
            for record in result.rescue
        ]

    out["rescue"] = {
        "default_levels": [record.level for record in rescued.rescue],
        "default_attempts": attempts(rescued),
        "forced_attempts": attempts(forced),
    }
    out["refusals"] = {
        "explicit ninjars": _refused(
            lambda: loaded.evaluate(pts[:1], mu_r2=scales[:1], reduction="ninjars")
        ),
        "ninjars rescue level": _refused(
            lambda: loaded.evaluate(
                pts[:1],
                mu_r2=scales[:1],
                stability=VirtualRescuePolicy(
                    (VirtualRescueLevel("double", reduction="ninjars"),)
                ),
            )
        ),
        "cross-backend check": _refused(
            lambda: loaded.evaluate(
                pts[:1],
                mu_r2=scales[:1],
                stability_checks=VirtualStabilityChecks(1, cross_backend=True),
            )
        ),
    }
    # The SDKs' native runtime: TIRust by default, the Python API's binary64
    # bits on the points TIRust serves, the same points failed closed.
    native = load_loop_native()
    momenta = [[x for leg in point for x in leg] for point in pts]
    plain = loaded.evaluate(pts, mu_r2=scales, on_failure="return")
    sdk = native.evaluate_v2_artifact(
        str(artifact), momenta, scales, process="p", parameters=parameters()
    )
    out["sdk"] = []
    for (values, failures), python, lane in zip(
        sdk, plain.values, plain.diagnostics, strict=True
    ):
        if lane.status != "success":
            out["sdk"].append(
                "both failed" if values is None else "SDK served, API failed"
            )
            continue
        same = values is not None and [list(map(float, x)) for x in values[1]] == [
            list(map(float, x)) for x in python
        ]
        out["sdk"].append("bitwise" if same else f"differs: {failures[:1]}")
    out["lanes"] = {
        "double": lane_agreement(loaded, pts, scales, "double"),
        "double-double": lane_agreement(loaded, pts, scales, "double-double"),
    }
    out["checks"] = {
        "declared": out["declared_reductions"] == ["tirust"],
        "auto_tirust_everywhere": all(
            list(map(tuple, run["reduction"])) == [("auto", "tirust", ONLY)]
            for run in (runs["double"], runs["double-double"], runs["arbitrary"])
        ),
        "every_carrier_serves_or_fails_closed_typed": all(
            entry["status"] == "success" or "TIRust" in (entry["detail"] or "")
            for name in ("double", "double-double", "arbitrary")
            for entry in runs[name]["points"]
        )
        and all(served[name] for name in results),
        "rescue_on_tirust": all(
            attempt["effective"] == "tirust"
            for result in ("default_attempts", "forced_attempts")
            for record in out["rescue"][result]
            for attempt in record
        )
        and [len(record) for record in out["rescue"]["forced_attempts"]] == [3]
        and all(level is not None for level in out["rescue"]["default_levels"]),
        "ninjars_refused_typed": all(r is not None for r in out["refusals"].values()),
        "sdk_on_tirust": all(
            entry in ("bitwise", "both failed") for entry in out["sdk"]
        ),
        "lanes_agree": all(entry["passed"] for entry in out["lanes"].values()),
    }
    out["passed"] = all(out["checks"].values())
    return out


def lane_agreement(loaded, pts, scales, arithmetic: str) -> dict:
    """TIRust against NinjaRS lane by lane on the families NinjaRS reduces
    (module docstring), on the points TIRust serves in the carrier: E the
    lane's rescaled re-test deviation, the floor and rounding terms
    binary64's (4 u, 1e-12) or double-double's (4 u, 1e-24)."""

    from pyamplicol.loops.v2.families import reducer_rank_refusal
    from pyamplicol.loops.v2.reduction import running
    from pyamplicol.runtime.virtual import _inputs

    backend = loaded._backend
    executor = backend._executor
    points, mu = _inputs(pts, scales, backend.external_count)
    _carrier, setting = backend._carrier(arithmetic, None)
    binary64 = arithmetic == "double"
    # The rule's floor and rounding term in the carrier: binary64's, and
    # double-double's at its precision (the extended carriers' rule, L6).
    floor, rounding = (FLOOR, ROUNDING) if binary64 else (FLOOR_DD, DOUBLE_DOUBLE)
    families = json.loads(backend._native_process.plan_json)["families"]
    served = tuple(
        reducer_rank_refusal(len(f["sequence"]), f["rank"], "ninjars") is None
        for f in families
    )
    arms, failed = {}, {}
    for name, mask in (("ninjars", served), ("tirust", None)):
        executor.family_mask = mask
        try:
            with running(name):
                _precision, evaluations, failures = executor._evaluated(
                    points, mu, dict(backend._parameters), setting, "all", "rescaled"
                )
        finally:
            executor.family_mask = None
        failed[name] = sorted(failures)
        arms[name] = [
            None
            if index in failures
            else (
                {k: v.coefficients for k, v in e.families.items()},
                None
                if e.retest_families is None
                else {k: v.coefficients for k, v in e.retest_families.items()},
            )
            for index, e in enumerate(evaluations)
        ]
    count, worst, beyond, worst_lane = 0, 0.0, 0, None
    for n_point, t_point in zip(arms["ninjars"], arms["tirust"], strict=True):
        if n_point is None or t_point is None:
            continue
        (n_direct, n_check), (t_direct, t_check) = n_point, t_point
        beyond += len(set(t_direct) - set(n_direct))
        for lane, n in n_direct.items():
            t = t_direct[lane]
            size = max(_size(z) for z in (*n, *t))
            deviation = max(_size(a - b) for a, b in zip(t, n, strict=True))
            e_n = max(_size(a - b) for a, b in zip(n, n_check[lane], strict=True))
            e_t = max(_size(a - b) for a, b in zip(t, t_check[lane], strict=True))
            bound = MARGIN * max(e_n, e_t, floor * size) + rounding * size
            ratio = deviation / bound if bound else float(deviation > 0)
            if ratio > worst:
                worst, worst_lane = (
                    ratio,
                    {
                        "lane": str(lane),
                        "deviation": deviation,
                        "scale": size,
                        "e_ninjars": e_n,
                        "e_tirust": e_t,
                    },
                )
            count += 1
    return {
        "rule": (
            "|T - N| <= 10 max(E_N, E_T, 4u|N|) + r |N| per lane, E the rescaled "
            f"re-test deviation, u the carrier's unit roundoff, r = {rounding:g}"
        ),
        "worst_lane": worst_lane,
        "families": len(families),
        "ninjars_families": sum(served),
        "failed_points": failed,
        "lanes_compared": count,
        "tirust_only_lanes": beyond,
        "worst": worst,
        "passed": count > 0 and beyond > 0 and not failed["ninjars"] and worst <= 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--rows", nargs="+", default=sorted(ROWS))
    args = parser.parse_args()
    model = model_path(args.model)
    record: dict = {
        "claim": "execution only: rank > N+2 results carry no correctness claim "
        "(no correct R2 for them; USER 2026-11-06 ~14:55)",
        "model": str(model.name),
        "rows": {},
    }
    passed = True
    for key in args.rows:
        artifact = generate(args.work_dir, key, model)
        loaded = runtime(artifact)
        pts, scales = points(key)
        buckets = loaded.inspect()["buckets"] if hasattr(loaded, "inspect") else None
        entry: dict = {"process": ROWS[key][0], "buckets": buckets}
        if key == "x0x0":
            entry["agreement"] = public_agreement(loaded, pts, scales)
            ok = entry["agreement"]["passed"]
        else:
            entry.update(tir_only_checks(artifact, loaded, pts, scales))
            ok = entry["passed"]
        entry["passed"] = ok
        passed = passed and ok
        record["rows"][key] = entry
        print(f"{key}: {'PASS' if ok else 'FAIL'}")
    record["passed"] = passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1, default=str) + "\n")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
