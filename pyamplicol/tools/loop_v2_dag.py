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
  (``ct_closure.CountertermBlocks.build``), chained after the request
  model, the Born, the loop blocks and the families. Python lane only until
  the native CT builder (A3).

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


def _s1pp(args: argparse.Namespace, clock: _Clock, record: dict[str, Any]) -> None:
    if args.lane != "python":
        raise SystemExit("s1pp: the native CT builder is increment A3")
    from pyamplicol.loops.v2 import dag_input
    from pyamplicol.loops.v2.assemble import SliceRequest, prepare_request_model
    from pyamplicol.loops.v2.blocks import build_tree_blocks, with_ward_kernels
    from pyamplicol.loops.v2.ct_closure import (
        CountertermBlocks,
        lowered_counterterm_model,
        program_order_limits,
    )
    from pyamplicol.loops.v2.ct_recursion import CountertermRecursion
    from pyamplicol.loops.v2.families import (
        LoopOrderSelection,
        ModelLoopCatalogue,
        add_orders,
        generate_families,
    )
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.processes.model import build_model_process_ir

    compiled = clock("model", lambda: _load_model(args))
    request = SliceRequest(
        args.process,
        record["born_bounds"],
        record["loop_orders"],
        ignored_particle_ids=tuple(args.ignore),
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
    blocks = clock(
        "Born + loop blocks",
        lambda: build_tree_blocks(
            process,
            CompiledUFOModel(compiled),
            born_coupling_orders=dict(request.born_orders),
            block_coupling_orders=request.loop_order_bounds,
            ignored_particle_ids=request.ignored_particle_ids,
        ),
    )
    blocks, compiled = with_ward_kernels(blocks, compiled, fixed)
    catalogue = ModelLoopCatalogue(
        data, species_orders=prepared.structure.perturbative_species_orders
    )
    families = clock(
        "families",
        lambda: generate_families(blocks.block_table(data), catalogue, selection),
    )
    scope = (
        tuple(sorted(request.born_orders.items())),
        tuple(sorted(request.loop_order_bounds.items())),
        request.compatibility,
    )
    lowered = clock(
        "CT lowering",
        lambda: lowered_counterterm_model(
            compiled,
            dict(fixed),
            forbidden=request.ignored_particle_ids,
            declaration=request.colour_declaration,
            request=(scope, prepared, continuation),
            order_limits=program_order_limits(
                compiled, blocks.born_orders, selection.loop_orders
            ),
        ),
    )
    recursion = CountertermRecursion.build(
        lowered,
        prepared.counterterms.rules,
        families,
        data,
        selection,
        blocks.born_orders,
        forbidden=request.ignored_particle_ids,
    )
    ir = build_model_process_ir(
        args.process, lowered.lowered.compiled.ir, color_accuracy="full"
    )
    program = clock(
        "CT program",
        lambda: CountertermBlocks.build(
            recursion, ir, ignored_particle_ids=request.ignored_particle_ids
        ),
    )
    lowered_model = CompiledUFOModel(lowered.lowered.compiled)
    dag = program.blocks.blocks_dag
    native = clock("encode", lambda: dag_input.NativeDag.from_dag(dag, lowered_model))
    dag_digest = clock("dag digest", native.digest)
    block_digest = clock(
        "blocks digest",
        lambda: dag_input.blocks_digest(program.blocks.blocks, program.clones),
    )
    record["counts"] = {
        "currents": len(dag.currents),
        "interactions": len(dag.interactions),
        "sources": len(dag.sources),
        "blocks": len(program.blocks.blocks),
        "clones": len(program.clones),
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
        "--ignore", type=int, nargs="*", default=[], help="ignored particle ids"
    )
    parser.add_argument("--model", type=Path, help="compiled loop model (JSON)")
    parser.add_argument("--ufo", type=Path, help="loop_sm UFO (with --cache)")
    parser.add_argument("--cache", type=Path, help="model cache directory")
    parser.add_argument("--out", type=Path, help="JSON record")
    parser.add_argument("--log", type=Path, help="stage progress log")
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
    }
    try:
        record["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        record["commit"] = None
    clock = _Clock(args.log)
    start = time.perf_counter()
    (_s1p if args.stage == "s1p" else _s1pp)(args, clock, record)
    from pyamplicol.loops.v2 import dag_input

    record["seconds"] = clock.seconds
    record["wall_seconds"] = round(time.perf_counter() - start, 3)
    record["peak_rss_gb"] = round(_peak_gb(), 3)
    record["decodes"] = dag_input.decode_count()
    text = json.dumps(record, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
