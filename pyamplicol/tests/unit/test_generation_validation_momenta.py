# SPDX-License-Identifier: 0BSD
"""Validation momenta from the model's masses (genericity audit m7).

A scattering takes on-shell beams of the model's masses in its
centre-of-mass frame, a decay the rest frame of the decaying particle;
massless beams keep the historical points bit for bit. pyAmpliCol processes
have two initial particles today, so the decay tests call the builder directly.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from pyamplicol.generation.phase_space import massive_rambo_final_state
from pyamplicol.generation.validation import build_process_validation_point

MASSES = {
    6: 173.0,
    -6: 173.0,
    5: 4.7,
    -5: 4.7,
    24: 80.4,
    -24: 80.4,
    21: 0.0,
    2: 0.0,
    -2: 0.0,
}


class _Model:
    def mass(self, pdg: int) -> float:
        return MASSES[int(pdg)]


def _process(initial, final):
    names = " ".join(map(str, initial)) + " > " + " ".join(map(str, final))
    return SimpleNamespace(initial_pdgs=initial, final_pdgs=final, process=names)


def _point(initial, final, seed=7):
    return build_process_validation_point(
        _process(initial, final), _Model(), process_id="p", seed=seed
    )


def _assert_on_shell_and_conserved(record, n_initial):
    momenta = [m for _pdg, m in record.particles]
    scale = max(abs(x) for m in momenta for x in m)
    for pdg, (e, x, y, z) in record.particles:
        mass = MASSES[pdg]
        assert abs(e * e - x * x - y * y - z * z - mass * mass) <= 1e-12 * scale**2
    for axis in range(4):
        total = sum(m[axis] for m in momenta[:n_initial]) - sum(
            m[axis] for m in momenta[n_initial:]
        )
        assert abs(total) <= 1e-12 * scale


def test_massless_beams_keep_the_historical_point():
    record = _point((2, -2), (6, -6, 21))
    sqrt_s = max(1000.0, 2 * 173.0 + 100.0)
    beam = 0.5 * sqrt_s
    final = massive_rambo_final_state(
        3, sqrt_s=sqrt_s, masses=(173.0, 173.0, 0.0), seed=7
    )
    assert record.particles == (
        (2, (beam, 0.0, 0.0, beam)),
        (-2, (beam, 0.0, 0.0, -beam)),
        *zip((6, -6, 21), final, strict=True),
    )


@pytest.mark.parametrize(
    ("initial", "final", "threshold"),
    [
        ((6, -6), (21, 21), 346.0),
        ((6, 21), (6, 21), 173.0),
        ((2, 6), (6, 6, -6), 519.0),
    ],
)
def test_massive_beams_are_on_shell(initial, final, threshold):
    """On-shell beams at sqrt(s) twice the larger threshold of the process."""

    record = _point(initial, final)
    assert record.available, record.error
    _assert_on_shell_and_conserved(record, 2)
    (_, first), (_, second) = record.particles[:2]
    assert first[3] > 0 > second[3] and first[1:3] == second[1:3] == (0.0, 0.0)
    assert math.isclose(first[0] + second[0], 2.0 * threshold, rel_tol=1e-15)


def test_decays_take_the_rest_frame():
    record = _point((6,), (5, 24, 21))
    assert record.available, record.error
    assert record.particles[0] == (6, (173.0, 0.0, 0.0, 0.0))
    _assert_on_shell_and_conserved(record, 1)


def test_a_closed_decay_has_no_point():
    record = _point((5,), (24, 21))
    assert not record.available
    assert "decay of mass 4.7" in record.error
    assert not _point((24,), (6,)).available  # one final state: no decay
    assert math.isfinite(_point((6,), (5, 24)).particles[1][1][0])
