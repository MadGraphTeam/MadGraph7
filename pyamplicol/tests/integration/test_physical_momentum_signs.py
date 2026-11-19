# SPDX-License-Identifier: 0BSD
"""Every binary64 entry point that takes physical momenta checks energy signs.

A massless Weyl source derives its helicity from chirality and sgn(E), so an
incoming quark fed a negated momentum is the crossed (outgoing) state, not
zero. Momenta with E <= 0, or E not finite, must therefore be refused instead
of returning a finite matrix element of the wrong crossing.
"""

from __future__ import annotations

import importlib.util
import math
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from pyamplicol import Generator, ProcessSet, Runtime
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.models.builtin.validation import generic_validation_point


def _config(*, unions: bool) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc"),
        process=ProcessConfig(
            flavor_scheme=4,
            flavor_blind_strict=unions,
            flavor_unions=unions,
            max_coupling_orders={"QED": 0},
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="recurrence",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    root = tmp_path_factory.mktemp("physical-momentum-signs")
    paths = {"dedicated": root / "dedicated", "union": root / "union"}
    try:
        Generator(_config(unions=False)).generate(
            ProcessSet.from_expressions(("u g > u g",)), paths["dedicated"]
        )
        Generator(_config(unions=True)).generate(
            ProcessSet.from_expressions(("d d > d d", "d u > d u")), paths["union"]
        )
        yield paths
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _point(expression: str) -> list[list[float]]:
    return [
        [float(component) for component in particle.momentum]
        for particle in generic_validation_point(expression)
    ]


def _negated_incoming(point: list[list[float]]) -> list[list[float]]:
    return [[-x for x in leg] if index < 2 else leg for index, leg in enumerate(point)]


def test_negated_incoming_energies_are_refused(artifacts: dict[str, Path]) -> None:
    runtime = Runtime.load(artifacts["dedicated"], process="u g > u g")
    point = _point("u g > u g")
    value = complex(runtime.evaluate([point])[0]).real
    assert math.isfinite(value) and value > 0
    crossed = _negated_incoming(point)
    helicity = runtime.physics.helicities[0].id
    for call in (
        lambda: runtime.evaluate([crossed]),
        lambda: runtime.evaluate([crossed], helicities=(helicity,)),
        lambda: runtime.evaluate_resolved([crossed]),
    ):
        with pytest.raises(Exception, match="energy"):
            call()
    # Unchanged after the refusals.
    assert complex(runtime.evaluate([point])[0]).real == value


@pytest.mark.parametrize("energy", [0.0, -0.0, math.nan, math.inf])
def test_degenerate_energies_are_refused(
    artifacts: dict[str, Path], energy: float
) -> None:
    runtime = Runtime.load(artifacts["dedicated"], process="u g > u g")
    point = _point("u g > u g")
    point[2][0] = energy
    with pytest.raises(Exception, match="energy"):
        runtime.evaluate([point])


def test_assignments_refuse_unsigned_crossings(artifacts: dict[str, Path]) -> None:
    template = Runtime.load(artifacts["union"], process="d d > d d")
    point = _point("d d > d d")
    assert len(template.evaluate_assignments([point], [[0]])[0]) == 1
    with pytest.raises(Exception, match="energy"):
        template.evaluate_assignments([_negated_incoming(point)], [[0]])
