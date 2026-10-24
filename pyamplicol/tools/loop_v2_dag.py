#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Stage-isolated measurements of the loop-request DAG builders (WP3.1 A).

Each run builds one stage of one request with one lane and writes a JSON
record: counts, the stage digest, the wall time of every sub-stage, the
process peak memory and the number of native decodes (rule 9). Python
oracle records are stored once per (process, stage) and compared against the
native lane (``RUST-MIGRATION.md`` §2 rule 6).

Stages:

* ``s1p``: S1', the loop-request block DAG (``GenericDAGCompiler(
  loop_blocks=True)``, default reachability, the request's loop-order
  bounds) and its blocks (``blocks._blocks``).
* ``s1pp``: S1'', the counterterm program's DAG, its blocks and clone lines
  (``ct_closure.counterterm_program_blocks``, the production chain: CT
  lowering, the recursion with its inert rules, ``CountertermBlocks.build``,
  the U(1) leakage check), chained after the request model, the Born and the
  loop blocks. The native lane runs the same chain and replaces only its
  ``CT program`` stage (``dag_input.native_counterterm_blocks`` and
  ``dag_input.u1_leaking_structures``), so the lowering and the recursion
  are the production ones. ``--blocks native`` chains the native loop-block
  DAG (equal to Python's by its S1' digest) into the Ward-kernel query and
  compiles only the Born in Python (rule 6: no oracle run repeats an
  upstream Python stage whose native twin is proven equal).

Lanes: ``python`` (the oracle) or ``native`` (``dag_input.native_blocks``).
The request follows ``tools/loop_v2_generate.py``: ``--born`` bounds, loop
orders = Born + 2 in QCD unless ``--loop`` is given.

Example::

    python tools/loop_v2_dag.py s1p "g g > g g g" --born QCD=3 --lane native \\
        --model loop-sm.json --out s1p.json
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


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

    from pyamplicol.loops.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )
    from pyamplicol.models.loading import compile_model_source

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


def _loop_blocks(
    clock: _Clock,
    lane: str,
    process: Any,
    model: Any,
    bounds: Mapping[str, int],
    ignored: tuple[int, ...],
) -> tuple[Any, Any, tuple[Any, ...]]:
    """``(NativeDag of the DAG, the GenericDAG or None, blocks)``."""

    from pyamplicol.generation.dag_compiler import GenericDAGCompiler
    from pyamplicol.loops.v2 import dag_input
    from pyamplicol.loops.v2.blocks import _blocks
    from pyamplicol.models.prepared_catalog_builder import model_tree_kernel_variant

    variant = model_tree_kernel_variant(model)
    if lane == "python":
        dag = clock(
            "loop-block DAG",
            lambda: GenericDAGCompiler(
                model=model,
                max_coupling_orders=bounds,
                ignored_particle_ids=ignored,
                loop_blocks=True,
            ).compile(process),
        )
        blocks = clock("blocks", lambda: _blocks(dag, model, variant))
        native = clock("encode", lambda: dag_input.NativeDag.from_dag(dag, model))
        return native, dag, blocks
    tables = clock(
        "input tables",
        lambda: dag_input.dag_input(
            process, model, max_coupling_orders=bounds, ignored_particle_ids=ignored
        ),
    )
    handle = clock("loop-block DAG", lambda: dag_input._native().V2Dag.build(tables))
    native = dag_input.NativeDag(handle, process, model)
    blocks = clock("blocks", lambda: native.blocks(variant))
    return native, None, blocks


def _s1p(args: argparse.Namespace, clock: _Clock, record: dict[str, Any]) -> None:
    from pyamplicol.loops.v2 import dag_input
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.processes.model import build_model_process_ir

    compiled = clock("model", lambda: _load_model(args))
    model = CompiledUFOModel(compiled)
    process = build_model_process_ir(args.process, compiled.ir, color_accuracy="full")
    native, _dag, blocks = _loop_blocks(
        clock, args.lane, process, model, record["loop_bounds"], tuple(args.ignore)
    )
    dag_digest = clock("dag digest", native.digest)
    block_digest = clock("blocks digest", lambda: dag_input.blocks_digest(blocks))
    currents, interactions, sources = native.counts()
    record["counts"] = {
        "currents": currents,
        "interactions": interactions,
        "sources": sources,
        "blocks": len(blocks),
    }
    record["digests"] = {
        "dag": dag_digest,
        "blocks": block_digest,
        "S1'": dag_input.stage_digest("S1'", dag_digest, block_digest),
    }


def _native_blocks_chain(
    clock: _Clock, process: Any, compiled: Any, request: Any, fixed: Any
) -> tuple[Any, tuple[Any, ...]]:
    """``(compiled, born_orders)`` of ``with_ward_kernels(build_tree_blocks(...))``
    with the loop-block DAG native: the Born in Python, the Ward-kernel query
    on the native DAG's ``coloured_propagated()`` (the current scan of
    ``ward_breaking_fields``) and the Born DAG."""

    from dataclasses import replace
    from types import SimpleNamespace

    from pyamplicol.generation.dag_compiler import compile_generic_dag
    from pyamplicol.loops.v2 import dag_input
    from pyamplicol.loops.v2.blocks import (
        _born_roots,
        vetoed_particle_ids,
        ward_breaking_fields,
    )
    from pyamplicol.loops.v2.catalogue import compile_tree_kernel_variant
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.prepared_catalog_builder import model_tree_kernel_variant

    model = CompiledUFOModel(compiled)
    ignored = vetoed_particle_ids(request.particle_veto)
    born = clock(
        "Born",
        lambda: compile_generic_dag(
            process,
            model=model,
            max_coupling_orders=dict(request.born_orders),
            ignored_particle_ids=ignored,
        ),
    )
    native = clock(
        "loop-block DAG (native)",
        lambda: dag_input.native_blocks(
            replace(process, color_accuracy="full"),
            model,
            max_coupling_orders=request.loop_order_bounds,
            ignored_particle_ids=ignored,
        ),
    )
    # The current scan of ``ward_breaking_fields`` reads ids and kinds only.
    propagated: Any = SimpleNamespace(
        sources=(),
        currents=tuple(
            SimpleNamespace(index=SimpleNamespace(particle_id=pdg, auxiliary_kind=None))
            for pdg in sorted(native.coloured_propagated())
        ),
    )
    if model_tree_kernel_variant(model) != "full" and ward_breaking_fields(
        (propagated, born), compiled, fixed
    ):
        compiled = compile_tree_kernel_variant(compiled, "full")
    born_orders = tuple(
        sorted({root.coupling_orders for root in _born_roots(born, model)})
    )
    return compiled, born_orders


def _s1pp(args: argparse.Namespace, clock: _Clock, record: dict[str, Any]) -> None:
    from pyamplicol.loops.v2 import dag_input
    from pyamplicol.loops.v2.assemble import SliceRequest, prepare_request_model
    from pyamplicol.loops.v2.blocks import (
        build_tree_blocks,
        vetoed_particle_ids,
        with_ward_kernels,
    )
    from pyamplicol.loops.v2.ct_closure import (
        UnprojectedU1Error,
        counterterm_program_blocks,
    )
    from pyamplicol.loops.v2.families import LoopOrderSelection, add_orders
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.processes.model import build_model_process_ir

    compiled = clock("model", lambda: _load_model(args))
    request = SliceRequest(
        args.process,
        record["born_bounds"],
        record["loop_orders"],
        particle_veto=tuple(args.ignore),
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
    if args.blocks == "python":
        blocks = clock(
            "Born + loop blocks",
            lambda: build_tree_blocks(
                process,
                CompiledUFOModel(compiled),
                born_coupling_orders=dict(request.born_orders),
                block_coupling_orders=request.loop_order_bounds,
                ignored_particle_ids=vetoed_particle_ids(request.particle_veto),
            ),
        )
        blocks, compiled = with_ward_kernels(blocks, compiled, fixed)
        born_orders = blocks.born_orders
    else:
        compiled, born_orders = _native_blocks_chain(
            clock, process, compiled, request, fixed
        )
    recursion: list[Any] = []

    class _NativeProgram(Exception):
        """Raised at the chain's ``CT program`` stage in the native lane."""

    def stage(name: str, work: Callable[[], Any]) -> Any:
        if args.lane == "native" and name == "CT program":
            raise _NativeProgram
        value = clock(name, work)
        if name == "CT recursion":
            recursion.append(value)
        return value

    # The production chain (``LoopSlice.generate``): the CT program depends
    # on no loop family (REQUIREMENTS_ADDENDUM J), so none is generated here.
    try:
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
            stage=stage,
        )
    except _NativeProgram:
        ((ct_recursion, ir),) = recursion
        native_program = clock(
            "CT program",
            lambda: dag_input.native_counterterm_blocks(
                ct_recursion,
                ir,
                ignored_particle_ids=vetoed_particle_ids(request.particle_veto),
            ),
        )
        leaking = clock(
            "U(1) leakage", lambda: dag_input.u1_leaking_structures(native_program)
        )
        if leaking:
            raise UnprojectedU1Error(
                f"counterterm vertices {sorted(str(s.particles) for s in leaking)} "
                "would take the U(1) part of a fermion line's gluon"
            ) from None
        native = native_program.dag
        program_blocks, clones = native_program.blocks, native_program.clones
    else:
        lowered_model = CompiledUFOModel(program.recursion.model.lowered.compiled)
        dag = program.blocks.blocks_dag
        native = clock(
            "encode", lambda: dag_input.NativeDag.from_dag(dag, lowered_model)
        )
        program_blocks, clones = program.blocks.blocks, program.clones
    dag_digest = clock("dag digest", native.digest)
    block_digest = clock(
        "blocks digest", lambda: dag_input.blocks_digest(program_blocks, clones)
    )
    currents, interactions, sources = native.counts()
    record["counts"] = {
        "currents": currents,
        "interactions": interactions,
        "sources": sources,
        "blocks": len(program_blocks),
        "clones": len(clones),
    }
    record["digests"] = {
        "dag": dag_digest,
        "blocks": block_digest,
        "S1''": dag_input.stage_digest("S1''", dag_digest, block_digest),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("s1p", "s1pp"))
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
        help="s1pp: the Born and loop blocks upstream of the CT program in "
        "Python, or the loop blocks native (equal by S1')",
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
    stage = _s1p if args.stage == "s1p" else _s1pp
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
