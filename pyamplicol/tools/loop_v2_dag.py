#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Stage-isolated measurements of the loop-request DAG builders (WP3.1 A).

Each run builds one stage of one request with one lane and writes a JSON
record: counts, the stage digest, the wall time of every sub-stage, the
process peak memory and the number of native decodes (rule 9). Python
oracle records are stored once per (process, stage) and compared against the
native lane (``RUST-MIGRATION.md`` §2 rule 6).

Stages (digests: ``loops.v2.snapshot``):

* ``s1p``: S1', the loop-request block DAG (``dag_input.loop_request_blocks``,
  the stage of ``build_tree_blocks``: default reachability, the request's
  loop-order bounds) and its blocks.
* ``s1pp``: S1'', the counterterm program's DAG, its blocks and clone lines
  (``ct_closure.counterterm_program_blocks``, the production chain: CT
  lowering, the recursion with its inert rules, ``CountertermBlocks.build``,
  the U(1) leakage check), chained after the request model, the Born and the
  loop blocks (``build_tree_blocks`` and ``with_ward_kernels``).
  ``--blocks native`` builds those loop blocks natively (equal to Python's
  by their S1' digest; rule 6: no oracle run repeats an upstream Python
  stage whose native twin is proven equal).

* ``pct``: P-ct, the counterterm program's interaction program and saved
  kernels in the native lane's export (``native_trees._CountertermTrees.of``),
  chained after the native S1'' chain (``--blocks`` as for ``s1pp``; the CT
  DAG is always the native builder's, equal to Python's by S1''): ``native``
  is A5's program (``dag_input.NATIVE_PROGRAM``), ``python`` the oracle on
  the decoded DAG (``TreeRows`` and ``_interaction_program``).

Lanes: ``python`` (the oracle: ``dag_input.NATIVE_DAG`` off) or ``native``
(the production stage with the flag on). A record whose stage fell back to
the tree engine names the reason (``fallback``).
The request follows ``tools/loop_v2_generate.py``: ``--born`` bounds, loop
orders = Born + 2 in QCD unless ``--loop`` is given.

Example::

    python tools/loop_v2_dag.py s1p "g g > g g g" --born QCD=3 --lane native \\
        --model loop-sm.json --out s1p.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # the ``tools`` package, run as a script
    sys.path.insert(0, str(ROOT))


def _orders(text: str) -> dict[str, int]:
    return {
        name.strip(): int(value)
        for name, value in (item.split("=") for item in text.split(",") if item)
    }


def _peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


class _Clock:
    def __init__(self, log: Path | None) -> None:
        self.seconds: dict[str, float] = {}
        self._log = log
        self._start = time.perf_counter()

    def __call__(self, name: str, work: Callable[[], Any]) -> Any:
        self._note(f"start {name}")
        start = time.perf_counter()
        value = work()
        self.seconds[name] = round(time.perf_counter() - start, 3)
        self._note(f"done {name} {self.seconds[name]} s peak {_peak_gb():.2f} GB")
        return value

    def _note(self, text: str) -> None:
        line = f"[{time.perf_counter() - self._start:9.1f} s] {text}"
        print(line, file=sys.stderr, flush=True)
        if self._log is not None:
            with self._log.open("a") as stream:
                stream.write(line + "\n")


def _load_model(args: argparse.Namespace) -> Any:
    from pyamplicol.models.loading import load_compiled_model

    if args.model is not None:
        return load_compiled_model(args.model)
    # The loop_sm recipe of tests/conftest.py (scalar-arity view).
    from dataclasses import replace

    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    cache = Path(args.cache)
    written = sorted(cache.glob("loop-sm.json*.json"))
    if written:
        return load_compiled_model(written[0])
    model = compile_model_source(Path(args.ufo), cache_dir=cache / "model-cache")
    model = replace(
        model,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            model.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    return load_compiled_model(model.write(cache / "loop-sm.json"))


def _native_dag(on: bool) -> contextlib.AbstractContextManager[None]:
    """``dag_input.NATIVE_DAG`` set to ``on`` (the lane) for the block."""

    from pyamplicol.loops.v2 import dag_input

    return dag_input.native_flags(dag=on)


def _s1p(args: argparse.Namespace, clock: _Clock, record: dict[str, Any]) -> None:
    from pyamplicol.loops.v2 import snapshot
    from pyamplicol.loops.v2.blocks import TreeBlocks
    from pyamplicol.loops.v2.dag_input import loop_request_blocks
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.prepared_catalog_builder import model_tree_kernel_variant
    from pyamplicol.processes.model import build_model_process_ir

    compiled = clock("model", lambda: _load_model(args))
    model = CompiledUFOModel(compiled)
    process = build_model_process_ir(args.process, compiled.ir, color_accuracy="full")
    with _native_dag(args.lane == "native"):
        dag, blocks, fallback = clock(
            "loop-block DAG + blocks",
            lambda: loop_request_blocks(
                process,
                model,
                block_coupling_orders=record["loop_bounds"],
                ignored_particle_ids=tuple(args.ignore),
                kernel_variant=model_tree_kernel_variant(model),
            ),
        )
    tree = TreeBlocks(process, dag, dag, blocks, (), blocks[0].kernel_variant)
    digest = clock("S1' digest", lambda: snapshot.loop_blocks_digest(tree))
    record["fallback"] = fallback
    record["counts"] = _counts(dag, blocks)
    record["digests"] = {"S1'": digest}


def _counts(dag: Any, blocks: tuple[Any, ...]) -> dict[str, int]:
    from pyamplicol.loops.v2 import dag_input

    native = (
        dag
        if isinstance(dag, dag_input.NativeDag)
        else dag_input.NativeDag.from_dag(dag, None)
    )
    currents, interactions, sources = native.counts()
    return {
        "currents": currents,
        "interactions": interactions,
        "sources": sources,
        "blocks": len(blocks),
    }


def _s1pp(args: argparse.Namespace, clock: _Clock, record: dict[str, Any]) -> None:
    from pyamplicol.loops.v2 import snapshot
    from pyamplicol.loops.v2.assemble import (
        SliceRequest,
        prepare_request_model,
        resolve_request,
    )
    from pyamplicol.loops.v2.blocks import (
        build_tree_blocks,
        vetoed_particle_ids,
        with_ward_kernels,
    )
    from pyamplicol.loops.v2.ct_closure import counterterm_program_blocks
    from pyamplicol.loops.v2.families import LoopOrderSelection, add_orders
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.processes.model import build_model_process_ir

    compiled = clock("model", lambda: _load_model(args))
    request = resolve_request(
        compiled,
        SliceRequest(
            args.process,
            record["born_bounds"],
            record["loop_orders"],
            particle_veto=tuple(args.ignore),
        ),
    )
    prepared, continuation, fixed, _scale, _alpha = clock(
        "request model", lambda: prepare_request_model(compiled, request)
    )
    data = json.loads(prepared.structure.native.to_json())
    selection = LoopOrderSelection(
        frozenset(request.corrections),
        frozenset(add_orders(tuple(o.items())) for o in request.exact_loop_orders),
    )
    process = build_model_process_ir(args.process, compiled.ir, color_accuracy="full")

    def loop_blocks() -> tuple[tuple[Any, ...], Any]:
        tree = build_tree_blocks(
            process,
            CompiledUFOModel(compiled),
            born_coupling_orders=dict(request.born_orders),
            block_coupling_orders=request.loop_order_bounds,
            ignored_particle_ids=vetoed_particle_ids(request.particle_veto),
        )
        tree, kernels = with_ward_kernels(tree, compiled, fixed)
        return tree.born_orders, kernels

    with _native_dag(args.blocks == "native"):
        born_orders, compiled = clock("Born + loop blocks + Ward kernels", loop_blocks)
    # The production chain (``LoopSlice.generate``): the CT program depends
    # on no loop family (REQUIREMENTS_ADDENDUM J), so none is generated here.
    with _native_dag(args.lane == "native"):
        program = counterterm_program_blocks(
            compiled,
            prepared,
            request.process,
            data,
            selection,
            born_orders,
            veto=request.particle_veto,
            fixed_parameters=fixed,
            declaration=request.colour_declaration,
            continuation=continuation,
            scope=(
                tuple(sorted(request.born_orders.items())),
                tuple(sorted(request.loop_order_bounds.items())),
                request.compatibility,
            ),
            stage=clock,
        )
    digest = clock("S1'' digest", lambda: snapshot.counterterm_program_digest(program))
    record["fallback"] = program.blocks.dag_fallback
    record["counts"] = {
        **_counts(program.blocks.blocks_dag, program.blocks.blocks),
        "clones": len(program.clones),
    }
    record["digests"] = {"S1''": digest}
    if args.stage == "pct":
        record["_program"] = program


def _pct(args: argparse.Namespace, clock: _Clock, record: dict[str, Any]) -> None:
    from pyamplicol.loops.v2 import dag_input, snapshot
    from pyamplicol.loops.v2.ct_closure import CountertermProgram
    from pyamplicol.loops.v2.native_trees import _CountertermTrees

    lane = args.lane
    args.lane = "native"
    try:
        _s1pp(args, clock, record)
    finally:
        args.lane = lane
    blocks = record.pop("_program")
    with dag_input.native_flags(program=lane == "native"):
        trees = clock(
            "CT interaction program",
            lambda: _CountertermTrees.of(CountertermProgram(blocks, ()), 0),
        )
    digest = clock("P-ct digest", lambda: snapshot.counterterm_trees_digest(trees))
    steps = trees.record["interactions"]["steps"]
    record["counts"]["program rows"] = sum(len(step["interactions"]) for step in steps)
    record["counts"]["kernels"] = len(trees.saved)
    record["digests"]["P-ct"] = digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("s1p", "s1pp", "pct"))
    parser.add_argument("process")
    parser.add_argument("--born", required=True, help="Born bounds, e.g. QCD=5,QED=0")
    parser.add_argument(
        "--loop", help="exact loop orders (default: the Born bounds + 2 in QCD)"
    )
    parser.add_argument("--lane", choices=("python", "native"), default="native")
    parser.add_argument(
        "--blocks",
        choices=("python", "native"),
        default="python",
        help="s1pp, pct: the loop blocks upstream of the CT program in Python, "
        "or native (equal by S1')",
    )
    parser.add_argument(
        "--ignore",
        type=int,
        nargs="*",
        default=[],
        help="ignored particle ids (s1p: signed tree-engine ids; s1pp: the "
        "request's particle_veto, |PDG| codes with both charge states)",
    )
    parser.add_argument("--model", type=Path, help="compiled loop model (JSON)")
    parser.add_argument("--ufo", type=Path, help="loop_sm UFO (with --cache)")
    parser.add_argument("--cache", type=Path, help="model cache directory")
    parser.add_argument("--out", type=Path, help="JSON record")
    parser.add_argument("--log", type=Path, help="stage progress log")
    parser.add_argument(
        "--repeat", type=int, default=1, help="runs in this process (warm: 2+)"
    )
    args = parser.parse_args(argv)
    if args.model is None and (args.ufo is None or args.cache is None):
        parser.error("give --model, or --ufo with --cache")
    born = _orders(args.born)
    loop = (
        _orders(args.loop)
        if args.loop
        else {name: value + (2 if name == "QCD" else 0) for name, value in born.items()}
    )
    record: dict[str, Any] = {
        "stage": args.stage,
        "process": args.process,
        "lane": args.lane,
        "born_bounds": born,
        "loop_orders": loop,
        "loop_bounds": loop,
        "ignored_particle_ids": list(args.ignore),
        "blocks": args.blocks,
    }
    try:
        record["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        record["commit"] = None
    stage = {"s1p": _s1p, "s1pp": _s1pp, "pct": _pct}[args.stage]
    clock = _Clock(args.log)
    start = time.perf_counter()
    stage(args, clock, record)
    record["seconds"] = clock.seconds
    record["wall_seconds"] = round(time.perf_counter() - start, 3)
    # Warm runs: the same stage again in this process (model caches warm);
    # their records must reproduce the cold one.
    warm = []
    for _run in range(args.repeat - 1):
        again: dict[str, Any] = dict(record)
        clock = _Clock(args.log)
        begin = time.perf_counter()
        stage(args, clock, again)
        if again["digests"] != record["digests"]:
            raise SystemExit("a warm run changed the stage digests")
        warm.append(
            {"seconds": clock.seconds, "wall": round(time.perf_counter() - begin, 3)}
        )
    record["warm"] = warm
    from pyamplicol.loops.v2 import dag_input

    record["peak_rss_gb"] = round(_peak_gb(), 3)
    record["decodes"] = dag_input.decode_count()
    text = json.dumps(record, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
