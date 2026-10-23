#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Generation time, memory and stage digests of the v2 generator (WP3.1).

``measure`` generates one process with ``LoopSlice.generate`` (loop_sm with
MB = 0 by default, QCD corrections at Born + 2 in QCD; LC and NLC with the
committed QCD colour continuation) and records the wall time and peak
resident memory of every generator stage (``/proc/self/clear_refs`` resets
the high-water mark between stages), the structure counts and, with
``--digests``, the stage digests of ``loops.v2.snapshot`` (S1 families, S4
schedules, S5 seeds and, with ``--export``, S8 the plan). ``--engine
python`` runs the families and schedules in Python (their oracles), so two
records of one process compare the native stages with Python's.
``--export`` also exports plan v4 (``forward=False``) and records its size.

``diff`` compares the digests of two records and exits non-zero on a
difference.

``validate`` generates a process of ``tests/fixtures/loop/required``, exports
its plan and evaluates the fixture's points with the native executor, then
compares the Born and the three Laurent coefficients of the virtual with
every provider by the fixture's metric and tolerance. Catani's pole check is
skipped: its exact dipole matrices cost hours at seven legs (WP3.1 I5c); the
poles are compared with the providers instead. With ``--inputs python``
(default) the per-point trees, Born and counterterms are Python inputs; with
``--inputs native`` the executor forms the trees, the Born, the counterterm
and external-field components itself (``native_trees``; the counterterms
from the slice's counterterm closure) and every point is native.

``poles`` validates a process without provider references by other means
(MASTER_PLAN §0.1): native inputs at ``--points`` exact points
(``tools/loop_reference/exact_points``), the double pole of the bare virtual
over the Born against the universal -(alpha_s/2pi) sum_i C_i of its massless
coloured legs (at ``--accuracy lc`` the Casimirs' leading-colour parts; NLC
only with adjoint legs), and with ``--rotation`` the stability estimate of
one rotation check.

Usage:
    python tools/loop_v2_generate.py measure "g g > g g g g" --born QCD=4
        [--accuracy fc|lc|nlc] [--ignored 23] [--engine native|python]
        [--digests] [--export] [--json OUT.json] [--model COMPILED.json]
    python tools/loop_v2_generate.py diff A.json B.json
    python tools/loop_v2_generate.py validate gg_gggg [--points 0 1 2]
        [--precision double|dd] [--stability] [--inputs python|native]
        [--json OUT.json]
    python tools/loop_v2_generate.py poles "g g > g g g g g" --born QCD=5
        [--accuracy fc|lc|nlc] [--points 3] [--rotation] [--tolerance 1e-8]
        [--json OUT.json]
"""

from __future__ import annotations

import argparse
import functools
import json
import resource
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    ("assemble", "build_tree_blocks"),
    ("assemble", "_field_level_families"),
    ("generator_input", "NativeGeneration.families"),
    ("generator_input", "NativeGeneration.schedules"),
    ("assemble", "generate_families"),
    ("assemble", "build_schedules"),
    ("assemble", "SeedColour"),
    ("assemble", "generate_counterterm_sites"),
    ("assemble", "external_field_insertions"),
    ("assemble", "build_loop_kernel_catalogue"),
    ("assemble", "SiteClosure"),
    ("assemble", "LeafWeights"),
    ("assemble", "_born_terms"),
    ("assemble", "_born_tensors"),
    ("assemble", "build_seed_table"),
    ("assemble", "extend_seed_table"),
    ("assemble", "seed_fourier"),
    ("assemble", "LoopSlice._prune_colour"),
    ("reference_executor", "reading_skeletons"),
)


def _status(key: str) -> int:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith(key):
                return int(line.split()[1]) * 1024
    return 0


def _progress(record: dict[str, Any], *keys: str) -> None:
    """Print some record entries to stderr as soon as they are known (a long
    run that fails later keeps them)."""

    print(json.dumps({key: record[key] for key in keys}), file=sys.stderr, flush=True)


def _reset_peak() -> None:
    try:
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")
    except OSError:  # not Linux, or not permitted: peaks become cumulative
        pass


class _Timers:
    """Wall time and peak memory per stage (nested stages counted inside)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._peaks: list[int] = []

    def wrap(self, name: str, fn):
        @functools.wraps(fn)
        def timed(*args, **kwargs):
            if self._peaks:
                self._peaks[-1] = max(self._peaks[-1], _status("VmHWM"))
            _reset_peak()
            self._peaks.append(_status("VmRSS"))
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                wall = time.perf_counter() - start
                peak = max(self._peaks.pop(), _status("VmHWM"))
                if self._peaks:
                    self._peaks[-1] = max(self._peaks[-1], peak)
                _reset_peak()
                self.events.append(
                    {
                        "stage": name,
                        "depth": len(self._peaks),
                        "wall_s": round(wall, 3),
                        "peak_gb": round(peak / 2**30, 3),
                    }
                )
                # Each stage as it ends: a long run that fails keeps its record.
                print(json.dumps(self.events[-1]), file=sys.stderr, flush=True)

        return timed

    def install(self) -> None:
        from pyamplicol.loops.v2 import assemble, generator_input
        from pyamplicol.loops.v2 import reference_executor as rx

        modules = {
            "assemble": assemble,
            "generator_input": generator_input,
            "reference_executor": rx,
        }
        for module_name, attribute in STAGES:
            owner: Any = modules[module_name]
            *path, name = attribute.split(".")
            for part in path:
                owner = getattr(owner, part)
            setattr(owner, name, self.wrap(attribute, getattr(owner, name)))


def _request(args) -> Any:
    from pyamplicol.loops.v2.assemble import SliceRequest

    born = {k: int(v) for k, v in (p.split("=") for p in args.born.split(","))}
    loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
    options: dict[str, Any] = {}
    if args.accuracy != "fc":
        from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

        declaration = ROOT / "examples/loop/source_colour/loop_sm_qcd_subset.json"
        options["colour_declaration"] = LoopColorDeclaration.from_dict(
            json.loads(declaration.read_text())
        )
        options["accuracy"] = args.accuracy
    return SliceRequest(
        args.process,
        born,
        loop,
        fixed_parameters={"MB": "0"},
        ignored_particle_ids=tuple(int(x) for x in args.ignored.split(",") if x),
        **options,
    )


def _compiled_model(path: str | None) -> Any:
    from pyamplicol.models.loading import load_compiled_model

    if path:
        return load_compiled_model(path)
    import os
    import tempfile
    from dataclasses import replace

    from pyamplicol.loops.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )
    from pyamplicol.models.loading import compile_model_source

    ufo = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not ufo:
        raise SystemExit("set PYAMPLICOL_NLO_UFO_REFERENCE or pass --model")
    root = Path(tempfile.mkdtemp(prefix="loop-v2-generate-"))
    model = compile_model_source(Path(ufo), cache_dir=root / "model-cache")
    model = replace(
        model,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            model.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    return load_compiled_model(model.write(root / "loop-sm.json"))


def measure(args) -> dict[str, Any]:
    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2 import generator_input, snapshot
    from pyamplicol.loops.v2.assemble import LoopSlice

    if args.engine == "python":
        generator_input.NATIVE_STAGES = frozenset()
    timers = _Timers()
    timers.install()
    compiled = _compiled_model(args.model)
    native = load_loop_native()
    request = _request(args)
    start = time.perf_counter()
    slice_ = timers.wrap("LoopSlice.generate", LoopSlice.generate)(
        compiled, request, native
    )
    record: dict[str, Any] = {
        "process": args.process,
        "born": args.born,
        "accuracy": args.accuracy,
        "engine": args.engine,
        "generate_s": round(time.perf_counter() - start, 3),
    }
    schedules = slice_.plan.schedules.values()
    record["counts"] = {
        "families": len(slice_.families.families),
        "routings": sum(len(f.closures) for f in slice_.families.families.values()),
        "prefix_states": len(slice_.families.prefixes),
        "scheduled_families": len(slice_.plan.schedules),
        "skeletons": sum(len(s.trie.leaves) for s in schedules),
        "trie_nodes": sum(len(s.trie.nodes) for s in schedules),
        "contents": len(slice_.plan.view._contents),
        "blocks": len(slice_.blocks.blocks),
        "seed_classes": len(slice_.table.cells),
        "born_basis": len(slice_.basis.orders),
        "ct_sites": len(slice_.sites.sites),
    }
    if args.export:
        from pyamplicol.loops.v2.plan_export import export_plan

        exported = timers.wrap("export_plan", export_plan)(slice_, forward=False)
        text = timers.wrap("plan_json", exported.json)()
        record["plan_json_mb"] = round(len(text.encode()) / 2**20, 2)
    if args.digests:
        start = time.perf_counter()
        record["digests"] = snapshot.stage_digests(slice_, plan=args.export)
        record["digests_s"] = round(time.perf_counter() - start, 3)
    record["events"] = timers.events
    record["max_rss_gb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3
    )
    return record


# The tolerance of the provider fixtures of tests/fixtures/loop/required.
REQUIRED_TOLERANCE = "2e-8"


def _relative(a: float, b: float, floor: float = 0.0) -> float:
    scale = max(abs(a), abs(b), floor)
    return 0.0 if scale == 0 else abs(a - b) / scale


def validate(args) -> dict[str, Any]:
    from pyamplicol.loops.v2 import assemble

    checked = assemble._PointEvaluation._stable_coloured_legs
    assemble._PointEvaluation._stable_coloured_legs = lambda self: False
    try:
        return _validate(args)
    finally:
        assemble._PointEvaluation._stable_coloured_legs = checked


def _validate(args) -> dict[str, Any]:
    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
    from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan
    from pyamplicol.loops.v2.stability import Rotation, StabilityChecks

    reference = json.loads(
        (ROOT / "tests/fixtures/loop/required" / f"{args.name}.json").read_text()
    )
    if (
        set(reference["born_orders"]) - {"QCD", "QED"}
        or reference["correction"] != "QCD"
    ):
        raise SystemExit("validate covers QCD corrections")
    process = reference["process"].split(" [")[0]
    process = " ".join(t for t in process.split() if "=" not in t)
    born = dict(reference["born_orders"])  # bounds: QED=0 excludes EW Borns
    loop = dict(reference["loop_orders"])
    # NJet's massless five-flavour QCD is loop_sm without the top
    # (``test_loop_v2_topfree``); its fixtures name the scheme.
    top_free = reference.get("scheme") == "massless-qcd-nf5"
    request = SliceRequest(
        process,
        born,
        loop,
        fixed_parameters=dict(reference.get("fixed_parameters", {"MB": "0"})),
        ignored_particle_ids=(6,) if top_free else (),
    )
    parameters = dict(reference["parameters"])
    record: dict[str, Any] = {
        "name": args.name,
        "process": process,
        "counterterms": request.counterterms,
    }
    start = time.perf_counter()
    slice_ = LoopSlice.generate(
        _compiled_model(args.model), request, load_loop_native()
    )
    record["generate_s"] = round(time.perf_counter() - start, 1)
    record["generate_peak_gb"] = round(_status("VmHWM") / 2**30, 3)
    _progress(record, "generate_s", "generate_peak_gb")
    _reset_peak()
    start = time.perf_counter()
    if args.inputs == "native":
        executor = NativeExecutor(slice_, forward=False, inputs="native")
        trees = executor.exported.trees
        record["tree_program_mb"] = round(
            (len(trees.json()) + sum(len(saved) for saved in trees.kernels)) / 1e6, 2
        )
        record["counterterm_terms"] = len(trees.program["counterterms"]["terms"])
        program = trees.program.get("counterterm_trees")
        if program is not None:
            # The tree recursion's counterterm program (WP2.2), natively.
            record["counterterm_interactions"] = sum(
                len(step["interactions"]) for step in program["interactions"]["steps"]
            )
    else:
        executor = NativeExecutor(
            slice_, forward=False, exported=export_plan(slice_, forward=False)
        )
    record["export_s"] = round(time.perf_counter() - start, 1)
    record["export_peak_gb"] = round(_status("VmHWM") / 2**30, 3)
    record["plan_json_mb"] = round(len(executor._plan_json) / 1e6, 2)
    _progress(record, "export_s", "export_peak_gb", "plan_json_mb")
    record["inputs"] = args.inputs
    compared = ("born", "finite", "single_pole", "double_pole")
    precision = Precision.double_double() if args.precision == "dd" else None
    checks = None
    if args.stability:
        checks = StabilityChecks(
            rotations=(Rotation.about("z", "3/5", "4/5"),), reduction_direction=True
        )
    indices = args.points or range(len(reference["points"]))
    # The required fixtures' tolerance; NJet fixtures state none.
    tolerance = float(reference.get("tolerance", REQUIRED_TOLERANCE))
    record["tolerance"] = tolerance
    record["points"] = []
    failed = False
    for index in indices:
        point = reference["points"][index]
        scale = str(Fraction(reference["mu_r"][index]) ** 2)
        start = time.perf_counter()
        (result,) = executor.evaluate(
            [point],
            [scale],
            parameters=parameters,
            precision=precision,
            stability=checks,
        )
        # Totals over the squared-order sectors: the Born, then the virtual's
        # finite part, single and double pole (``PointResult.totals``).
        mine = [float(x) for x in result.totals[:4]]
        floor = abs(mine[0]) * float(parameters["aS"]) / (2 * 3.141592653589793)
        errors = {}
        for provider, values in reference["rows"][index]["providers"].items():
            errors[provider] = {
                key: _relative(
                    value, float(values[key]), 0.0 if key == "born" else floor
                )
                for key, value in zip(
                    ("born", "finite", "single_pole", "double_pole"), mine, strict=True
                )
                if key in compared and key in values
            }
        worst = max(e for row in errors.values() for e in row.values())
        failed |= worst > tolerance
        entry = {
            "index": index,
            "seconds": round(time.perf_counter() - start, 1),
            "values": mine,
            "errors": errors,
            "max_error": worst,
        }
        if result.stability is not None:
            entry["stability_relative"] = result.stability.relative
        entry["rss_gb"] = round(_status("VmRSS") / 2**30, 3)
        record["points"].append(entry)
        print(json.dumps(entry), flush=True)
    record["passed"] = not failed
    record["max_rss_gb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3
    )
    return record


# Quadratic Casimirs of SU(3) by UFO colour representation, and their
# leading-colour parts (C_F = N_c/2 - 1/(2 N_c), C_6 = N_c + 1 - 2/N_c, C_A =
# N_c): the double pole of a leading-colour virtual is -(alpha_s/2pi) times
# the leading parts' sum times the leading-colour Born, as every Casimir
# leads with N_c^1. At next-to-leading colour that holds only with adjoint
# legs (C_A is one power of N_c); a quark's -1/(2 N_c) meets Born powers the
# virtual's projection drops.
_CASIMIRS = {1: Fraction(0), 3: Fraction(4, 3), 6: Fraction(10, 3), 8: Fraction(3)}
_LEADING_CASIMIRS = {1: Fraction(0), 3: Fraction(3, 2), 6: Fraction(3), 8: Fraction(3)}


def poles(args) -> dict[str, Any]:
    """Universal double pole and rotation stability with native inputs."""

    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2.assemble import LoopSlice, _kernel_parameters
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.loops.v2.stability import Rotation, StabilityChecks
    from tools.loop_reference.exact_points import generate

    timers = _Timers()
    timers.install()
    compiled = _compiled_model(args.model)
    request = _request(args)
    record: dict[str, Any] = {
        "process": args.process,
        "born": args.born,
        "accuracy": args.accuracy,
    }
    start = time.perf_counter()
    slice_ = LoopSlice.generate(compiled, request, load_loop_native())
    record["generate_s"] = round(time.perf_counter() - start, 1)
    # The stage timers reset the high-water mark: the largest stage peak.
    record["generate_peak_gb"] = max(
        [round(_status("VmHWM") / 2**30, 3)]
        + [event["peak_gb"] for event in timers.events]
    )
    _progress(record, "generate_s", "generate_peak_gb")
    start = time.perf_counter()
    # The bare virtual's double pole is checked: no counterterm closure.
    executor = NativeExecutor(
        slice_, forward=False, inputs="native", counterterms=False
    )
    record["export_with_trees_s"] = round(time.perf_counter() - start, 1)
    record["plan_json_mb"] = round(len(executor._plan_json) / 1e6, 2)
    record["tree_program_mb"] = round(
        (
            len(executor.exported.trees.json())
            + sum(len(saved) for saved in executor.exported.trees.kernels)
        )
        / 1e6,
        2,
    )
    record["rss_after_export_gb"] = round(_status("VmRSS") / 2**30, 3)
    _progress(record, "export_with_trees_s", "plan_json_mb", "tree_program_mb")
    _progress(record, "rss_after_export_gb")
    by_pdg = {int(p["pdg_code"]): p for p in slice_.data["particles"]}
    # Coloured particles are stable, as the public path requires
    # (``artifact.require_zero_coloured_widths``): a width in the tree
    # propagators but not in the loop's breaks the pole structure (gg > tt~ g
    # with WT = 1.508: double pole off by 6e-7 of the Born; 4e-12 with 0).
    parameters = {"aS": args.alpha_s, "MT": "173"}
    parameters.update(
        {
            str(p["width"]): "0"
            for p in by_pdg.values()
            if abs(int(p["color"])) != 1 and p.get("width") not in (None, "ZERO", "0")
        }
    )
    record["parameters"] = parameters
    values = _kernel_parameters(compiled, {**parameters, **request.fixed_parameters})
    masses, casimirs = [], Fraction(0)
    table = _LEADING_CASIMIRS if args.accuracy == "lc" else _CASIMIRS
    for leg in sorted(slice_.blocks.process.legs, key=lambda leg: leg.label):
        particle = by_pdg[int(leg.pdg)]
        mass = particle.get("mass")
        mass = Fraction(0) if mass in (None, "ZERO", "0") else values[mass]
        masses.append(mass)
        colour = abs(int(particle["color"]))
        if colour not in table:
            raise SystemExit(f"no Casimir for colour representation {colour}")
        if args.accuracy == "nlc" and mass == 0 and colour not in (1, 8):
            raise SystemExit(
                "the universal double pole holds at NLC only with massless adjoint legs"
            )
        if mass == 0:
            casimirs += table[colour]
    expected = -float(Fraction(args.alpha_s) * casimirs) / (2 * 3.141592653589793)
    record["expected_double_pole_over_born"] = expected
    checks = None
    if args.rotation:
        checks = StabilityChecks(rotations=(Rotation.about("z", "3/5", "4/5"),))
    record["points"] = []
    worst = 0.0
    for index, point in enumerate(generate(masses, count=args.points)):
        start = time.perf_counter()
        (result,) = executor.evaluate(
            [[[str(x) for x in p] for p in point]],
            ["1000000"],
            parameters=parameters,
            stability=checks,
        )
        born, *bare = (float(x) for x in result.totals[:4])
        deviation = abs(bare[2] / born - expected) / abs(expected)
        worst = max(worst, deviation)
        entry = {
            "index": index,
            "seconds": round(time.perf_counter() - start, 2),
            "born": born,
            "bare": bare,
            "double_pole_over_born": bare[2] / born,
            "deviation": deviation,
        }
        if result.stability is not None:
            entry["stability_relative"] = result.stability.relative
        entry["rss_gb"] = round(_status("VmRSS") / 2**30, 3)
        record["points"].append(entry)
        print(json.dumps(entry), flush=True)
    record["max_deviation"] = worst
    record["passed"] = worst <= args.tolerance
    record["events"] = timers.events
    record["max_rss_gb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3
    )
    return record


def diff(a: dict[str, Any], b: dict[str, Any]) -> int:
    left, right = a.get("digests", {}), b.get("digests", {})
    status = 0
    for stage in sorted(set(left) | set(right)):
        same = left.get(stage) == right.get(stage)
        status |= not same
        print(f"{stage}: {'equal' if same else 'DIFFERENT'}")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("measure")
    run.add_argument("process")
    run.add_argument("--born", required=True, help="e.g. QCD=3 or QCD=2,QED=2")
    run.add_argument("--accuracy", choices=("fc", "lc", "nlc"), default="fc")
    run.add_argument("--ignored", default="", help="ignored particle ids, e.g. 23")
    run.add_argument("--engine", choices=("native", "python"), default="native")
    run.add_argument("--digests", action="store_true")
    run.add_argument("--export", action="store_true")
    run.add_argument("--model", help="compiled loop model JSON")
    run.add_argument("--json", help="write the record here")
    compare = commands.add_parser("diff")
    compare.add_argument("a")
    compare.add_argument("b")
    check = commands.add_parser("validate")
    check.add_argument("name", help="a fixture of tests/fixtures/loop/required")
    check.add_argument("--points", type=int, nargs="*")
    check.add_argument("--precision", choices=("double", "dd"), default="double")
    check.add_argument("--stability", action="store_true")
    check.add_argument("--inputs", choices=("python", "native"), default="python")
    check.add_argument("--model", help="compiled loop model JSON")
    check.add_argument("--json", help="write the record here")
    pole = commands.add_parser("poles")
    pole.add_argument("process")
    pole.add_argument("--born", required=True, help="e.g. QCD=5")
    pole.add_argument("--accuracy", choices=("fc", "lc", "nlc"), default="fc")
    pole.add_argument("--ignored", default="", help="ignored particle ids, e.g. 23")
    pole.add_argument("--points", type=int, default=3)
    pole.add_argument("--rotation", action="store_true")
    pole.add_argument("--alpha-s", default="0.118")
    pole.add_argument("--tolerance", type=float, default=1e-8)
    pole.add_argument("--model", help="compiled loop model JSON")
    pole.add_argument("--json", help="write the record here")
    args = parser.parse_args(argv)
    if args.command == "diff":
        return diff(
            json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
        )
    if args.command in ("validate", "poles"):
        record = validate(args) if args.command == "validate" else poles(args)
        if args.json:
            Path(args.json).write_text(json.dumps(record, indent=1))
        print(
            json.dumps(
                {k: v for k, v in record.items() if k not in ("points", "events")}
            )
        )
        return 0 if record["passed"] else 1
    record = measure(args)
    text = json.dumps(record, indent=1)
    if args.json:
        Path(args.json).write_text(text)
    print(json.dumps({k: v for k, v in record.items() if k != "events"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
