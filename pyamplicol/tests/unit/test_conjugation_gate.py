# SPDX-License-Identifier: 0BSD
"""The charge-conjugation (C) gate of flavour-blind and flavour-union routing.

A conjugated member is evaluated with the representative's program at the same
momenta and helicities, colour conjugated. That is licensed by C: every allowed
vertex needs a vertex on its antiparticles with the identical Lorentz, colour
and coupling signature. Toy compiled-model IRs exercise the condition without
the model compiler; the compiled UFO SM's QCD is the positive control in
``test_flavor_blind_expansion.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

from pyamplicol.generation.process_grouping_symmetry import VertexSymmetryTable

_PARTICLES = {
    "d": (1, "d~"),
    "d~": (-1, "d"),
    "s": (3, "s~"),
    "s~": (-3, "s"),
    "g": (21, "g"),
    "a": (22, "a"),
    "zp": (32, "zp"),
    "h": (25, "h"),
}
_TABLE = {pdg: SimpleNamespace(name=name) for name, (pdg, _) in _PARTICLES.items()}


def _term(
    particles: str, lorentz: str, color: str, coupling: str, orders: tuple = ()
) -> SimpleNamespace:
    return SimpleNamespace(
        particles=tuple(particles.split()),
        coupling_orders=orders,
        lorentz_expression=lorentz,
        color_expression=color,
        coupling=coupling,
        coupling_expression=coupling,
    )


def _qcd() -> list[SimpleNamespace]:
    # UFO couplings carry the Feynman factor i (GC_11 = complex(0,1)*G).
    return [
        _term("d~ d g", "Gamma(3,2,1)", "T(3,2,1)", "complex(0,1)*G", (("QCD", 1),)),
        _term("s~ s g", "Gamma(3,2,1)", "T(3,2,1)", "complex(0,1)*G", (("QCD", 1),)),
        _term("g g g", "VVV1", "f(1,2,3)", "-G", (("QCD", 1),)),
    ]


def _reason(terms: list[SimpleNamespace]) -> str | None:
    ir = SimpleNamespace(
        particles=[
            SimpleNamespace(name=name, pdg_code=pdg, antiname=anti)
            for name, (pdg, anti) in _PARTICLES.items()
        ],
        couplings=[],
        vertex_terms=terms,
    )
    symmetry = VertexSymmetryTable.from_compiled_ir(ir)  # type: ignore[arg-type]
    return symmetry._conjugation_reason(symmetry.allowed({}), (1, 3), _TABLE)  # type: ignore[arg-type]


def test_qcd_is_charge_conjugation_symmetric() -> None:
    assert _reason(_qcd()) is None


def test_off_diagonal_vector_needs_identical_conjugate_coupling() -> None:
    lorentz, color = "Gamma(3,2,1)", "Identity(1,2)"
    equal = [
        _term("d~ s zp", lorentz, color, "complex(0,1)*gds"),
        _term("s~ d zp", lorentz, color, "complex(0,1)*gds"),
    ]
    assert _reason(_qcd() + equal) is None
    # Hermitian but C-violating: g_sd = conj(g_ds) with a nonzero imaginary part.
    complex_coupling = [
        _term("d~ s zp", lorentz, color, "complex(0,1)*(gr + complex(0,1)*gi)"),
        _term("s~ d zp", lorentz, color, "complex(0,1)*(gr - complex(0,1)*gi)"),
    ]
    reason = _reason(_qcd() + complex_coupling)
    assert reason is not None and "[d~ s zp]" in reason
    assert "coupling or structure" in reason


def test_levi_civita_vertex_refuses_conjugation() -> None:
    epsilon = _term(
        "a a h", "Epsilon(1,2,-1,-2)*P(-1,1)*P(-2,2)", "1", "complex(0,1)*cpodd"
    )
    reason = _reason([*_qcd(), epsilon])
    assert reason is not None and "Levi-Civita" in reason and "[a a h]" in reason
    cp_even = _term(
        "a a h", "P(1,2)*P(2,1) - Metric(1,2)*P(-1,1)*P(-1,2)", "1", "complex(0,1)*c"
    )
    assert _reason([*_qcd(), cp_even]) is None


def test_a_missing_conjugate_vertex_is_still_refused() -> None:
    missing = _term("d~ s zp", "Gamma(3,2,1)", "Identity(1,2)", "complex(0,1)*gds")
    reason = _reason([*_qcd(), missing])
    assert reason is not None and "no allowed image" in reason
