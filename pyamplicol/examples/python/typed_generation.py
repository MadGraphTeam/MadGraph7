# SPDX-License-Identifier: 0BSD
"""Plan or generate a transactional schema-v3 process artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyamplicol import Generator, ModelSource, ProcessRequest, ProcessSet
from pyamplicol.config import resolve_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "process",
        nargs="+",
        help="Repeat process expressions after '--', or quote each expression.",
    )
    parser.add_argument(
        "--model",
        default="built-in-sm",
        help="built-in-sm, a trusted UFO directory, or a model JSON path",
    )
    parser.add_argument("--restriction")
    parser.add_argument("--color-accuracy", choices=("lc", "nlc", "full"), default="lc")
    parser.add_argument(
        "--mode",
        choices=("error", "append", "replace"),
        default="error",
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    card = {
        "schema_version": 1,
        "action": "generate",
        "model": {
            "source": args.model,
            "restriction": args.restriction,
        },
        "process": {
            "entries": [{"expression": expression} for expression in args.process]
        },
        "color": {"accuracy": args.color_accuracy},
        "generation": {"output": args.output, "mode": args.mode},
    }
    resolution = resolve_config(card)
    config = resolution.effective
    processes = ProcessSet(
        tuple(
            ProcessRequest.parse(entry.expression, name=entry.name)
            for entry in config.process.entries
        )
    )
    model = (
        ModelSource.built_in_sm()
        if args.model == "built-in-sm"
        else ModelSource.from_path(
            args.model,
            restriction=args.restriction,
            simplify=config.model.simplify,
        )
    )

    generator = Generator(resolution)
    plan = generator.plan(processes, model=model)
    for process in plan.concrete_processes:
        print(f"{process.name}: {process.expression}")
    if args.plan_only:
        return 0

    result = generator.generate(
        processes,
        config.generation.output or args.output,
        model=model,
        mode=config.generation.mode,
    )
    print(result.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
