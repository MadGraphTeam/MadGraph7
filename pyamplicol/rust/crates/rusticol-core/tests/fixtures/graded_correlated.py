# SPDX-License-Identifier: 0BSD
"""Fixture of the graded correlated test in ``tests/compiled_graded_colour.rs``.

Usage, from the repository root with the development environment::

    PYTHONPATH=src python \
        rust/crates/rusticol-core/tests/fixtures/graded_correlated.py DIR

writes into DIR the built-in SM trees ``u d > u d`` (graded full colour and
``strict-nlc``) and ``u u~ > g g`` (graded full colour and ``strict-lc``),
compiled, QCD only, each declaring every dipole and Casimir ``T_i . T_j``
(id ``Tij``, 1 <= i <= j <= 4). The subdirectories are named
``<process>-<accuracy>``: ``ud_ud-full``, ``ud_ud-strict-nlc``,
``uux_gg-full`` and ``uux_gg-strict-lc``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pyamplicol import ColorCorrelator, CorrelatorConfig, Generator, ModelSource
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    GenerationConfig,
    ProcessConfig,
    RunConfig,
)

ARTIFACTS = {
    "ud_ud-full": ("u d > u d", "full", True),
    "ud_ud-strict-nlc": ("u d > u d", "strict-nlc", False),
    "uux_gg-full": ("u u~ > g g", "full", True),
    "uux_gg-strict-lc": ("u u~ > g g", "strict-lc", False),
}


def main(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    correlators = CorrelatorConfig(
        color_correlations=tuple(
            ColorCorrelator.dipole(f"T{i}{j}", i, j)
            for i in range(1, 5)
            for j in range(i, 5)
        )
    )
    for name, (process, accuracy, graded) in ARTIFACTS.items():
        config = RunConfig(
            action="generate",
            process=ProcessConfig(
                coupling_order_policy="explicit",
                max_coupling_orders={"QCD": 2, "QED": 0},
            ),
            color=ColorConfig(accuracy=accuracy, graded=graded),
            evaluator=EvaluatorConfig(backend="jit", execution_mode="compiled"),
            generation=GenerationConfig(workers=1, emit_api_bundle=False),
        )
        Generator(config).generate(
            process,
            root / name,
            model=ModelSource.built_in_sm(),
            correlators=correlators,
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
