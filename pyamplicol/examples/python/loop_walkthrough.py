# SPDX-License-Identifier: 0BSD
"""One-loop walkthrough of a packaged loop example, from its generate card.

The `pyamplicol check` pole and Ward checks of the card's request, generation
(typed API, with the example's correlator declaration), evaluation with the
per-type default reduction backend and the other one, the coloured per-stage
profile (checked and unchecked), colour- and spin-correlated evaluation,
helicity selection, partial weights per squared coupling order with their sum
check, and the native SDK client. Run from an examples workspace, e.g.

    python python/loop_walkthrough.py loop_gg_zgg_generate.toml --points 2
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import resource
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from pyamplicol import Runtime
from pyamplicol.api import (
    BenchmarkRunner,
    Generator,
    ProcessRequest,
    ProcessSet,
    ReductionUnavailable,
    VirtualStabilityChecks,
)
from pyamplicol.config import resolve_config
from pyamplicol.correlators import CorrelatedRequest, CorrelatorConfig
from pyamplicol.reporting import render_summary

# The recommended stability checks: one rotation and the rescaled
# reduction-direction re-test.
RECOMMENDED = VirtualStabilityChecks(1, True, rescaled=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("card", type=Path, help="the example's generate card")
    parser.add_argument(
        "--artifact", type=Path, help="artifact directory (default: the card's)"
    )
    parser.add_argument("--points", type=int, default=2, help="evaluation points")
    parser.add_argument("--batch", type=int, default=32, help="profile batch size")
    parser.add_argument("--samples", type=int, default=3, help="timed profile calls")
    parser.add_argument(
        "--regenerate", action="store_true", help="generate even if it exists"
    )
    parser.add_argument(
        "--lc-card",
        type=Path,
        help="a leading-colour variant card (Born-interfered helicity selection)",
    )
    parser.add_argument(
        "--native-client", type=Path, help="the compiled native/loop_client"
    )
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    return parser


def _step(title: str) -> None:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"\n=== {title} === (peak RSS so far {peak:.2f} GiB)", flush=True)


def _example_name(card: Path) -> str:
    return card.stem.removesuffix("_generate")


def _generate(card: Path, artifact: Path, regenerate: bool) -> float:
    """The example's artifact through the typed API, with its correlator
    declaration (data/<example>_correlators.json) when it has one."""

    resolution = resolve_config(card)
    config = resolution.effective
    probes = config.generation.loop_induced_probes
    print(
        f"generation.loop_induced_probes = {str(probes).lower()} (off by default; "
        "enable with [generation] loop_induced_probes = true or "
        "--loop-induced-probes; `pyamplicol check` runs the same checks on demand)",
        flush=True,
    )
    if artifact.is_dir() and not regenerate:
        print(f"reusing {artifact} (--regenerate generates it again)", flush=True)
        return 0.0
    declaration = card.parent / "data" / f"{_example_name(card)}_correlators.json"
    correlators = (
        CorrelatorConfig.from_json_dict(json.loads(declaration.read_text()))
        if declaration.is_file()
        else None
    )
    requests = ProcessSet(
        tuple(
            ProcessRequest.parse(entry.expression, name=entry.name)
            for entry in config.process.entries
        )
    )
    begin = time.perf_counter()
    Generator(resolution).generate(
        requests, artifact, mode="replace", correlators=correlators
    )
    seconds = time.perf_counter() - begin
    print(f"generated {artifact} in {seconds:.1f} s", flush=True)
    return seconds


def _generate_apart(card: Path, artifact: Path, regenerate: bool) -> float:
    """``_generate`` in a child process: the generation's memory (several GiB
    for an electroweak request) returns to the system before the evaluation
    steps load the artifact."""

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        return pool.submit(_generate, card, artifact, regenerate).result()


def _check(target: Path, points: int, color: str) -> int:
    """`pyamplicol check` on a card or an artifact, in its own process (it
    regenerates the request in memory)."""

    command = [sys.executable, "-m", "pyamplicol", "check", str(target)]
    command += ["--points", str(points), "--progress", "off", "--color", color]
    print("$", " ".join(command[2:]), flush=True)
    return subprocess.run(command, check=False).returncode


def _points(card: Path, count: int) -> list[list[list[str]]]:
    path = card.parent / "data" / f"{_example_name(card)}_momenta.json"
    points = json.loads(path.read_text())
    return [points[i % len(points)] for i in range(count)]


def _scale(point: list[list[Any]], initial: int) -> str:
    """mu_R^2 = s of the initial state (the convention of `check`)."""

    total = [math.fsum(float(p[mu]) for p in point[:initial]) for mu in range(4)]
    return repr(total[0] ** 2 - total[1] ** 2 - total[2] ** 2 - total[3] ** 2)


def _backend(evaluation: Any) -> str:
    lane = evaluation.diagnostics[0]
    reduction = getattr(lane, "reduction", None)
    if reduction is None:
        return "unrecorded"
    return f"{reduction.effective} (requested {reduction.requested})"


def _finite(evaluation: Any) -> list[float]:
    return [float(sector[0]) for point in evaluation.values for sector in point]


def _median_seconds(call: Any, rounds: int = 3) -> float:
    found = []
    for _ in range(rounds):
        begin = time.perf_counter()
        call()
        found.append(time.perf_counter() - begin)
    return statistics.median(found)


def _transverse(momentum: list[Any]) -> list[str]:
    """A real unit vector orthogonal to the leg's momentum (v.p = 0)."""

    p = [float(x) for x in momentum[1:]]
    axis = [0.0, 0.0, 1.0] if abs(p[2]) < 0.9 * math.hypot(*p) else [1.0, 0.0, 0.0]
    n = [
        p[1] * axis[2] - p[2] * axis[1],
        p[2] * axis[0] - p[0] * axis[2],
        p[0] * axis[1] - p[1] * axis[0],
    ]
    norm = math.hypot(*n)
    return ["0", *(repr(x / norm) for x in n)]


def _correlated(runtime: Any, card: Path, points: Any, scales: Any, color: bool):
    declaration = card.parent / "data" / f"{_example_name(card)}_correlators.json"
    if not declaration.is_file():
        print("no correlator declaration for this example")
        return
    declared = json.loads(declaration.read_text())
    requests: dict[str, CorrelatedRequest] = {"total": CorrelatedRequest("born")}
    for history in declared["color_correlations"]:
        requests[history["id"]] = CorrelatedRequest(history["id"])
    for legs in declared["spin_correlations"]:
        vectors = {
            leg: [_transverse(point[leg - 1]) for point in points] for leg in legs
        }
        requests["spin " + ",".join(map(str, legs))] = CorrelatedRequest(
            "born", vectors
        )
    many = runtime.evaluate_correlated_many(points, requests, mu_r2=scales)
    print(render_summary(many, color=color))


def _selection(runtime: Any, points: Any, scales: Any) -> str:
    """One helicity against the summed call: its rows only are formed."""

    # The first helicity whose value at the first point is not zero (a
    # one-helicity call forms that helicity's rows only).
    helicity = next(
        h
        for h in runtime.physics.helicity_ids
        if float(
            runtime.evaluate(
                points[:1], mu_r2=scales[:1], poles="none", helicities=[h]
            ).values[0][0][0]
        )
    )
    one = runtime.evaluate(points, mu_r2=scales, poles="none", helicities=[helicity])
    summed = runtime.evaluate(points, mu_r2=scales, poles="none")
    t_one = _median_seconds(
        lambda: runtime.evaluate(
            points, mu_r2=scales, poles="none", helicities=[helicity]
        )
    )
    t_all = _median_seconds(
        lambda: runtime.evaluate(points, mu_r2=scales, poles="none")
    )
    print(
        f"helicity {helicity}: {_finite(one)[0]:.6e} of the summed "
        f"{_finite(summed)[0]:.6e} (point 0, first order); time one / summed = "
        f"{t_one / t_all:.2f} (median of 3 calls each, {len(points)} points)"
    )
    return helicity


def _deviation(label: str, evaluation: Any, reference: Any) -> None:
    """The largest relative deviation of ``evaluation``'s finite parts from
    ``reference``'s (the 'auto' call) over the points it served."""

    deviations = [
        abs(float(a[0]) - float(b[0])) / max(abs(float(b[0])), 1e-300)
        for mine, theirs in zip(
            () if evaluation is None else evaluation.values,
            reference.values,
            strict=False,
        )
        if mine is not None
        for a, b in zip(mine, theirs, strict=True)
    ]
    if deviations:
        print(
            f"{label}: largest relative deviation of the finite part from "
            f"'auto' {max(deviations):.2e} over "
            f"{len(deviations)} served point(s) and order(s)"
        )


def _orders(runtime: Any, points: Any, scales: Any, color: bool) -> None:
    """The partial weights per squared coupling order of one evaluation, and
    their explicit sum against another public path at the same settings that
    does not read the partials: the correlated route's "born" history summed
    over the orders (a separate call; where it reads the same contraction as
    the plain evaluation it agrees bitwise, otherwise to rounding). The
    deviation is printed in units of the binary64 unit roundoff u of the
    pre-cancellation scale S (the estimate's ``magnitude``) and must stay
    within the evaluation's own estimate E (ten times, the project's
    agreement rule). The resolved evaluation summed over helicities and
    colour components, a different contraction altogether, is the test
    suite's check (it costs far more memory at full colour)."""

    options = {"mu_r2": scales, "stability_checks": RECOMMENDED}
    view = runtime.evaluate_orders(points, **options)
    print(render_summary(view, color=color))
    if len(view.orders) == 1:
        print("one squared-order sector: its partial is the value")
    total = runtime.evaluate_correlated_many(
        points, {"total": CorrelatedRequest("born")}, **options
    )["total"]
    worst = ulps = 0.0
    for point, summed in enumerate(view.sum_orders()):
        estimate = view.evaluation.stability[point]
        for k in range(3):
            independent = math.fsum(complex(s[k]).real for s in total.values[point])
            deviation = abs(float(summed[k]) - independent)
            bound = 10 * math.fsum(float(e[k]) for e in estimate.absolute)
            worst = max(worst, deviation / bound if bound else 0.0)
            scale = math.fsum(float(m[k]) for m in estimate.magnitude)
            ulps = max(ulps, deviation / (scale * 2.0**-53) if scale else 0.0)
    verdict = "PASS" if worst <= 1 else "FAIL"
    agreement = (
        "bitwise equal"
        if ulps == 0
        else f"largest deviation {ulps:.3g} u S ({worst:.3g} x 10 E)"
    )
    print(f"sum over orders vs the correlated born total: {verdict}, {agreement}")
    if worst > 1:
        raise SystemExit("the partial weights do not sum to the total")


def _native(client: Path, artifact: Path, process: str, points, scales, helicity):
    path = artifact.parent / f"{artifact.name}_momenta.txt"
    lines = [f"{len(points)} {len(points[0])}"]
    for point, scale in zip(points, scales, strict=True):
        lines.append(" ".join([scale, *(str(x) for leg in point for x in leg)]))
    path.write_text("\n".join(lines) + "\n")
    environment = {k: v for k, v in os.environ.items() if k != "SYMBOLICA_LICENSE"}
    command = [str(client), str(artifact), process, str(path)]
    if helicity is not None:
        command.append(helicity)
    print("$ SYMBOLICA_LICENSE unset;", " ".join(command), flush=True)
    found = subprocess.run(command, env=environment, check=False)
    if found.returncode:
        raise SystemExit(f"the native client failed ({found.returncode})")


def main() -> int:
    args = _parser().parse_args()
    color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty())
    card = args.card.resolve()
    config = resolve_config(card).effective
    artifact = (args.artifact or config.generation.output).resolve()
    if args.native_client is not None and not args.native_client.is_file():
        raise SystemExit(
            f"--native-client {args.native_client}: no such file; build it first "
            "with `make -C native loop` (see examples/README.md)"
        )
    times: dict[str, float] = {}
    begin = time.perf_counter()

    # The check first: it regenerates the card's request in memory in its own
    # process, so its memory never adds to this process's generation.
    _step("1. pyamplicol check of the card (poles and Ward identities)")
    clock = time.perf_counter()
    code = _check(card, args.points, args.color)
    times["check"] = time.perf_counter() - clock
    print(
        f"check exit code {code} (the artifact's: pyamplicol check {artifact.name} "
        f"or pyamplicol {_example_name(card)}_check.toml)"
    )

    _step("2. Generation")
    times["generation"] = _generate_apart(card, artifact, args.regenerate)

    _step("3. Evaluation on both reduction backends")
    runtime = Runtime.load(artifact)
    kind = runtime.contribution_kind
    initial = sum(
        1 for leg in runtime.physics.external_particles if leg.state == "incoming"
    )
    points = _points(card, args.points)
    scales = [_scale(point, initial) for point in points]
    checked = runtime.evaluate(points, mu_r2=scales, stability_checks=RECOMMENDED)
    print(render_summary(checked, color=color))
    print(f"backend: {_backend(checked)} (the per-type default of 'auto')")
    finite = runtime.evaluate(points, mu_r2=scales, poles="none")
    other = "ninjars" if kind == "loop-induced" else "tirust"
    # An explicit backend has no fallback: a point it refuses is a failed
    # point (returned, with its reason), never a value of the other backend.
    try:
        explicit = runtime.evaluate(
            points,
            mu_r2=scales,
            reduction=other,
            stability_checks=RECOMMENDED,
            on_failure="return",
        )
    except ReductionUnavailable as error:
        # A native build without the backend refuses it, typed: never a
        # silent switch to the other one.
        print(f"explicit {other}: refused ({error})")
        explicit = None
    print(f"finite only: {_finite(finite)[:4]}")
    _deviation(f"explicit {other}", explicit, finite)
    refused = False
    for index, lane in enumerate(() if explicit is None else explicit.diagnostics):
        if not lane.valid:
            refused = True
            reason = lane.detail or "; ".join(map(str, lane.failures))
            print(f"explicit {other}: point {index} failed ({str(reason)[:200]})")
    if refused:
        # The points its own checks refuse, reduced without them: a timing
        # mode with no accuracy certificate, shown against 'auto' only.
        unchecked = runtime.evaluate(
            points,
            mu_r2=scales,
            reduction=other,
            reduction_unchecked=True,
            on_failure="return",
        )
        _deviation(
            f"explicit {other} unchecked (no accuracy certificate)", unchecked, finite
        )

    _step("4. Profile: time per stage, checked default and unchecked setting")
    for unchecked in (False, True):
        profile = resolve_config(
            {
                "action": "benchmark",
                "schema_version": 1,
                "benchmark": {
                    "batch_size": args.batch,
                    "minimum_samples": args.samples,
                    "warmup_runs": 1,
                },
                "evaluation": {"reduction_unchecked": unchecked},
            }
        ).effective
        result = BenchmarkRunner(profile).run(artifact, points=points)
        print(render_summary(result, color=color))
        times["unchecked" if unchecked else "checked"] = result.seconds_per_point

    _step("5. Colour- and spin-correlated evaluation")
    _correlated(runtime, card, points, scales, color)

    _step("6. Helicity selection")
    helicity = None
    if kind == "loop-induced":
        helicity = _selection(runtime, points, scales)
    elif args.lc_card is not None:
        lc_config = resolve_config(args.lc_card.resolve()).effective
        lc_artifact = lc_config.generation.output.resolve()
        _generate_apart(args.lc_card.resolve(), lc_artifact, args.regenerate)
        _selection(Runtime.load(lc_artifact), points, scales)
    else:
        print(
            "a Born-interfered helicity selection exists at leading colour only; "
            "this full-colour artifact has none (pass --lc-card for an LC variant)"
        )

    _step("7. Partial weights per squared coupling order")
    _orders(runtime, points, scales, color)

    if args.native_client is not None:
        _step("8. The same calls through the native SDK")
        _native(
            args.native_client,
            artifact,
            runtime.physics.process_id,
            points,
            scales,
            helicity,
        )

    times["total"] = time.perf_counter() - begin
    _step("Times")
    for name, value in times.items():
        unit = "s/point" if name in ("checked", "unchecked") else "s"
        print(f"{name}: {value:.4g} {unit}")
    return 0 if code == 0 else code


if __name__ == "__main__":
    raise SystemExit(main())
