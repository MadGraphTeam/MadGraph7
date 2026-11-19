# SPDX-License-Identifier: 0BSD
"""Every entry point that takes physical momenta checks energy signs.

A massless Weyl source derives its helicity from chirality and sgn(E), so an
incoming quark fed a negated momentum is the crossed (outgoing) state, not
zero. Momenta with E <= 0, or E not finite, must therefore be refused instead
of returning a finite matrix element of the wrong crossing: by the native
binary64 lanes (LC and full colour, selected helicities, orders, assignments,
correlations) and by the Python exact lanes behind higher precisions.
"""

from __future__ import annotations

import importlib.util
import math
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from pyamplicol import ColorCorrelator, CorrelatorConfig, Generator, ProcessSet, Runtime
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


def _config(*, unions: bool, accuracy: str = "lc") -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy=accuracy),
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
    paths = {
        name: root / name
        for name in ("lc", "full", "lc-union", "full-union", "correlated")
    }
    try:
        for accuracy in ("lc", "full"):
            Generator(_config(unions=False, accuracy=accuracy)).generate(
                ProcessSet.from_expressions(("u g > u g",)), paths[accuracy]
            )
            Generator(_config(unions=True, accuracy=accuracy)).generate(
                ProcessSet.from_expressions(("d d > d d", "d u > d u")),
                paths[f"{accuracy}-union"],
            )
        Generator(_config(unions=False)).generate(
            "u g > u g",
            paths["correlated"],
            correlators=CorrelatorConfig(
                color_correlations=(ColorCorrelator.dipole("T12", 1, 2),)
            ),
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


@pytest.mark.parametrize("accuracy", ["lc", "full"])
def test_negated_incoming_energies_are_refused(
    artifacts: dict[str, Path], accuracy: str
) -> None:
    runtime = Runtime.load(artifacts[accuracy], process="u g > u g")
    point = _point("u g > u g")
    value = complex(runtime.evaluate([point])[0]).real
    assert math.isfinite(value) and value > 0
    crossed = _negated_incoming(point)
    helicity = runtime.physics.helicities[0].id
    for call in (
        lambda: runtime.evaluate([crossed]),
        # At full colour a helicity selection routes to the selector companion.
        lambda: runtime.evaluate([crossed], helicities=(helicity,)),
        lambda: runtime.evaluate_resolved([crossed]),
        lambda: runtime.evaluate_orders([crossed]),
    ):
        with pytest.raises(Exception, match="energy"):
            call()
    # Unchanged after the refusals.
    assert complex(runtime.evaluate([point])[0]).real == value


def test_exact_lanes_refuse_negated_incoming_energies(
    artifacts: dict[str, Path],
) -> None:
    runtime = Runtime.load(artifacts["lc"], process="u g > u g")
    point = _point("u g > u g")
    assert complex(runtime.evaluate([point], precision=32)[0]).real > 0
    crossed = _negated_incoming(point)
    for call in (
        lambda: runtime.evaluate([crossed], precision=32),
        lambda: runtime.evaluate_resolved([crossed], precision=32),
    ):
        with pytest.raises(Exception, match="energy"):
            call()


def test_correlated_evaluations_refuse_negated_incoming_energies(
    artifacts: dict[str, Path],
) -> None:
    runtime = Runtime.load(artifacts["correlated"], process="u g > u g")
    point = _point("u g > u g")
    crossed = _negated_incoming(point)
    for precision in (16, 32):
        values = runtime.evaluate_correlated(
            [point], color_correlation="T12", precision=precision
        )
        assert math.isfinite(complex(values[0]).real)
        with pytest.raises(Exception, match="energy"):
            runtime.evaluate_correlated(
                [crossed], color_correlation="T12", precision=precision
            )


@pytest.mark.parametrize("energy", [0.0, -0.0, math.nan, math.inf])
def test_degenerate_energies_are_refused(
    artifacts: dict[str, Path], energy: float
) -> None:
    runtime = Runtime.load(artifacts["lc"], process="u g > u g")
    point = _point("u g > u g")
    point[2][0] = energy
    with pytest.raises(Exception, match="energy"):
        runtime.evaluate([point])


@pytest.mark.parametrize("accuracy", ["lc", "full"])
def test_assignments_refuse_negated_energies_of_an_uncrossed_assignment(
    artifacts: dict[str, Path], accuracy: str
) -> None:
    # Production assignments never cross sides, so every energy must be positive.
    template = Runtime.load(artifacts[f"{accuracy}-union"], process="d d > d d")
    point = _point("d d > d d")
    assert len(template.evaluate_assignments([point], [[0]])[0]) == 1
    with pytest.raises(Exception, match="energy"):
        template.evaluate_assignments([_negated_incoming(point)], [[0]])
