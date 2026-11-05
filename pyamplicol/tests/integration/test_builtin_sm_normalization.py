# SPDX-License-Identifier: 0BSD
"""Tree normalization of the built-in SM against analytic QED and Z exchange.

The built-in stage evaluators include the local vertex couplings but not the
overall g_s and e: the global factor (4 pi alpha_s)^QCD (8 pi alpha)^QED
restores them, with the coupling orders of the Born amplitudes as powers
(``generation.born_orders``). The powers used to be the number of
colour-singlet legs, which equals the QED order only when the process has
coloured legs: every fully colourless process came out (8 pi alpha)^2 = 4 e^4
too small (a factor 27.797 at alpha = 1/132.507), colourless decays such as
z > e+ e- included.
The built-in-vs-UFO comparison is
``test_tree_coupling_orders.test_builtin_sm_matches_the_ufo_sm``.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, Runtime
from pyamplicol.api.errors import GenerationError
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.generation.born_orders import RuntimeNormalizationRequest
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir

# The built-in SM's electroweak inputs (models.builtin.model.BuiltinSMModel).
ALPHA = 1.0 / 132.507
SIN_W = 0.47143025548407230
MZ, WZ = 91.188, 2.441404
_COLOURLESS = ("e+ e- > a a", "mu+ mu- > a a", "e+ e- > mu+ mu-", "z > e+ e-")


def _config(
    execution_mode: str = "recurrence", process: ProcessConfig | None = None
) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="full"),
        process=process or ProcessConfig(),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=execution_mode,
            optimization=EvaluatorOptimizationConfig(cores=1),
        ),
    )


def _points(process: str = "") -> list[list[tuple[float, float, float, float]]]:
    """Massless 2 -> 2 points: on the Z peak, above it and far below it; for
    a decay, the decaying particle at rest at its mass and three angles."""

    if process.startswith("z >"):
        return [
            _decay_point(theta, phi)
            for theta, phi in ((0.7, 0.3), (2.1, 1.1), (1.3, 2.0))
        ]
    points = []
    for energy, theta, phi in ((45.594, 0.7, 0.3), (250.0, 2.1, 1.1), (5.0, 1.3, 2.0)):
        x = energy * math.sin(theta) * math.cos(phi)
        y = energy * math.sin(theta) * math.sin(phi)
        z = energy * math.cos(theta)
        points.append(
            [
                (energy, 0.0, 0.0, energy),
                (energy, 0.0, 0.0, -energy),
                (energy, x, y, z),
                (energy, -x, -y, -z),
            ]
        )
    return points


def _decay_point(theta: float, phi: float) -> list[tuple[float, float, float, float]]:
    energy = MZ / 2
    x = energy * math.sin(theta) * math.cos(phi)
    y = energy * math.sin(theta) * math.sin(phi)
    z = energy * math.cos(theta)
    return [(MZ, 0.0, 0.0, 0.0), (energy, x, y, z), (energy, -x, -y, -z)]


def _invariant(a, b) -> float:
    return (a[0] - b[0]) ** 2 - sum((a[i] - b[i]) ** 2 for i in (1, 2, 3))


def _annihilation_to_photons(point, alpha: float = ALPHA) -> float:
    """l+ l- > a a: e^4 (u/t + t/u), spin average 1/4, identical photons 1/2."""

    t, u = _invariant(point[0], point[2]), _invariant(point[0], point[3])
    return (4 * math.pi * alpha) ** 2 * (u / t + t / u)


def _lepton_pair_with_z(point) -> float:
    """e+ e- > mu+ mu- through the photon and the fixed-width Z.

    Helicity amplitudes of massless f f~ > f' f'~ by vector exchange,
    A_ij = sum_V c_i c'_j / (s - M_V^2 + i M_V W_V): averaged |M|^2 =
    (|A_LL|^2 + |A_RR|^2) u^2 + (|A_LR|^2 + |A_RL|^2) t^2 with
    t = (p_e- - p_mu-)^2 and u = (p_e- - p_mu+)^2.
    """

    e2 = 4 * math.pi * ALPHA
    cos_w = math.sqrt(1 - SIN_W**2)
    charge, isospin = -1.0, -0.5
    left = (isospin - charge * SIN_W**2) / (SIN_W * cos_w)
    right = -charge * SIN_W**2 / (SIN_W * cos_w)
    s = _invariant(point[0], [-c for c in point[1]])
    t, u = _invariant(point[1], point[3]), _invariant(point[1], point[2])
    z = 1 / complex(s - MZ**2, MZ * WZ)

    def amplitude(c: float, c_prime: float) -> complex:
        return e2 * (charge * charge / s + c * c_prime * z)

    same = abs(amplitude(left, left)) ** 2 + abs(amplitude(right, right)) ** 2
    opposite = abs(amplitude(left, right)) ** 2 + abs(amplitude(right, left)) ** 2
    return same * u * u + opposite * t * t


def _z_to_lepton_pair(point) -> float:
    """z > l+ l- (massless): (2/3) MZ^2 e^2 (g_L^2 + g_R^2) / (s_w c_w)^2,
    averaged over the three Z helicities, at any decay angle; g_L = T3 - Q s_w^2
    and g_R = -Q s_w^2 with Q = -1, T3 = -1/2."""

    del point
    cos_w = math.sqrt(1 - SIN_W**2)
    charge, isospin = -1.0, -0.5
    left, right = isospin - charge * SIN_W**2, -charge * SIN_W**2
    return (
        (2 / 3)
        * MZ**2
        * 4
        * math.pi
        * ALPHA
        * (left**2 + right**2)
        / (SIN_W * cos_w) ** 2
    )


_ANALYTIC = {
    "e+ e- > a a": _annihilation_to_photons,
    "mu+ mu- > a a": _annihilation_to_photons,
    "e+ e- > mu+ mu-": _lepton_pair_with_z,
    "z > e+ e-": _z_to_lepton_pair,
}


def _assert_analytic(runtime: Runtime, process: str) -> None:
    points = _points(process)
    for point, value in zip(points, runtime.evaluate(points), strict=True):
        expected = _ANALYTIC[process](point)
        assert value.imag == 0.0
        assert value.real == pytest.approx(expected, rel=1e-12), (process, point)


@pytest.fixture(scope="module")
def colourless_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    artifact = tmp_path_factory.mktemp("builtin-colourless") / "artifact"
    Generator(_config()).generate(
        list(_COLOURLESS), artifact, model=ModelSource.built_in_sm()
    )
    return artifact


@pytest.mark.parametrize("process", _COLOURLESS)
def test_colourless_processes_match_analytic_qed_and_z(
    colourless_artifact: Path, process: str
) -> None:
    _assert_analytic(Runtime.load(colourless_artifact, process=process), process)


def test_runtime_alpha_change_rescales_the_colourless_born(
    colourless_artifact: Path,
) -> None:
    """The runtime recomputes the factor from the stored powers: e+ e- > a a
    is O(alpha^2), not O(alpha^4)."""

    runtime = Runtime.load(colourless_artifact, process="e+ e- > a a")
    runtime.set_model_parameter("normalization.alpha_ew", 2 * ALPHA)
    points = _points()
    for point, value in zip(points, runtime.evaluate(points), strict=True):
        expected = _annihilation_to_photons(point, alpha=2 * ALPHA)
        assert value.real == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("execution_mode", ("eager", "compiled", "on-the-fly"))
def test_every_execution_mode_normalizes_the_colourless_born(
    tmp_path: Path, execution_mode: str
) -> None:
    """Compiled and eager read the orders off the DAG's amplitude roots,
    recurrence and on-the-fly off the closures their limits admit."""

    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    process = "e+ e- > a a"
    Generator(_config(execution_mode)).generate(
        process, tmp_path / "artifact", model=ModelSource.built_in_sm()
    )
    _assert_analytic(Runtime.load(tmp_path / "artifact"), process)


def test_born_amplitudes_of_several_coupling_orders_are_refused() -> None:
    """One global factor cannot weigh Born amplitudes of different orders
    against each other. (The built-in vertex set yields one Born order for
    the processes tried, u u~ > d d~ with QED <= 2 included: its quark lines
    absorb electroweak currents but do not emit them, so the refusal is
    checked on the normalization request directly.)"""

    model = BuiltinSMModel()
    process = build_process_ir("u u~ > d d~")
    request = RuntimeNormalizationRequest(
        process, lambda: ((("QCD", 2),), (("QED", 2),))
    )
    with pytest.raises(GenerationError, match="several coupling orders"):
        model.runtime_normalization_payload(request)
    single = RuntimeNormalizationRequest(process, lambda: ((("QCD", 2),),))
    payload = model.runtime_normalization_payload(single)
    assert (payload["qcd_coupling_power"], payload["electroweak_coupling_power"]) == (
        2,
        0,
    )
