# SPDX-License-Identifier: 0BSD
"""Benchmark an artifact through the typed service facade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pyamplicol import BenchmarkConfig, BenchmarkRunner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--momenta", type=Path)
    parser.add_argument("--target-runtime", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--minimum-samples", type=int, default=5)
    return parser


def _read_points(path: Path | None) -> Any:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = _parser().parse_args()
    config = BenchmarkConfig(
        target_runtime=args.target_runtime,
        batch_size=args.batch_size,
        warmup_runs=args.warmup_runs,
        minimum_samples=args.minimum_samples,
    )
    result = BenchmarkRunner(config).run(
        args.artifact,
        points=_read_points(args.momenta),
    )
    print(
        json.dumps(
            {
                "sample_count": result.sample_count,
                "wall_time_per_point": result.wall_time_per_point,
                "evaluator_time_per_point": result.evaluator_time_per_point,
                "relative_standard_error": (result.uncertainty.relative_standard_error),
                "environment": dict(result.environment),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
