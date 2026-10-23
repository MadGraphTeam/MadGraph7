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

Usage:
    python tools/loop_v2_generate.py measure "g g > g g g g" --born QCD=4
        [--accuracy fc|lc|nlc] [--ignored 23] [--engine native|python]
        [--digests] [--export] [--json OUT.json] [--model COMPILED.json]
    python tools/loop_v2_generate.py diff A.json B.json
"""

from __future__ import annotations

import argparse
import functools
import json
import resource
import sys
import time
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
    args = parser.parse_args(argv)
    if args.command == "diff":
        return diff(
            json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
        )
    record = measure(args)
    text = json.dumps(record, indent=1)
    if args.json:
        Path(args.json).write_text(text)
    print(json.dumps({k: v for k, v in record.items() if k != "events"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
