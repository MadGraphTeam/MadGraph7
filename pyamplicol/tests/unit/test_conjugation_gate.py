# SPDX-License-Identifier: 0BSD
"""The charge-conjugation (C) gate of flavour-blind and flavour-union routing.

A conjugated member is evaluated with the representative's program at the same
momenta and helicities, colour conjugated. That is licensed by C: the C image
of every allowed vertex (antiparticles, fermion chain transposed, fixed C
parities of self-conjugate fields) must be an allowed vertex with the same
Lorentz x colour x coupling sum. Toy compiled-model IRs exercise the transform
without the model compiler; a toy JSON model compiled like any UFO model
checks the production structure strings; the compiled UFO SM's QCD is the
positive control in ``test_flavor_blind_expansion.py``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from pyamplicol import ModelSource
from pyamplicol.api.models import _compiled_model_payload
from pyamplicol.generation.process_grouping import particle_table
from pyamplicol.generation.process_grouping_symmetry import (
    SymmetryVertex,
    VertexSymmetryTable,
)
from pyamplicol.models.loading import CompiledModel

# name: (pdg, antiname, UFO spin code)
_PARTICLES = {
    "d": (1, "d~", 2),
    "d~": (-1, "d", 2),
    "s": (3, "s~", 2),
    "s~": (-3, "s", 2),
    "g": (21, "g", 3),
    "a": (22, "a", 3),
    "zp": (32, "zp", 3),
    "h": (25, "h", 1),
    "w+": (24, "w-", 3),
    "w-": (-24, "w+", 3),
}
_TABLE = {pdg: SimpleNamespace(name=name) for name, (pdg, _, _) in _PARTICLES.items()}
_SM = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)
_VVV1 = (
    "P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3)"
    " + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)"
)


def _term(
    particles: str, lorentz: str, color: str, coupling: str, orders: tuple = ()
) -> SimpleNamespace:
    return SimpleNamespace(
        particles=tuple(particles.split()),
        coupling_orders=orders,
        lorentz_source=lorentz,
        lorentz_expression=lorentz,
        color_source=color,
        color_expression=color,
        coupling=coupling,
        coupling_expression=coupling,
    )


def _qcd() -> list[SimpleNamespace]:
    # UFO couplings carry the Feynman factor i (GC_11 = complex(0,1)*G).
    return [
        _term("d~ d g", "Gamma(3,2,1)", "T(3,2,1)", "complex(0,1)*G", (("QCD", 1),)),
        _term("s~ s g", "Gamma(3,2,1)", "T(3,2,1)", "complex(0,1)*G", (("QCD", 1),)),
        _term("g g g", _VVV1, "f(1,2,3)", "-G", (("QCD", 1),)),
    ]


def _reason(terms: list[SimpleNamespace]) -> str | None:
    ir = SimpleNamespace(
        particles=[
            SimpleNamespace(name=name, pdg_code=pdg, antiname=anti, spin=spin)
            for name, (pdg, anti, spin) in _PARTICLES.items()
        ],
        couplings=[],
        vertex_terms=terms,
    )
    symmetry = VertexSymmetryTable.from_compiled_ir(ir)  # type: ignore[arg-type]
    return symmetry._conjugation_reason(symmetry.allowed({}), (1, 3), _TABLE)  # type: ignore[arg-type]


def test_qcd_is_charge_conjugation_symmetric() -> None:
    assert _reason(_qcd()) is None


def test_c_relabels_the_charged_legs_of_a_self_conjugate_vertex() -> None:
    # W+ W- a: C exchanges the W legs (VVV1 is odd) and the photon is C-odd.
    assert _reason([*_qcd(), _term("a w- w+", _VVV1, "1", "complex(0,1)*e")]) is None
    # A structure even under that exchange is C-odd.
    odd = _term("a w- w+", "P(1,2)*Metric(2,3) + P(1,3)*Metric(2,3)", "1", "k")
    reason = _reason([*_qcd(), odd])
    assert reason is not None and "[a w- w+]" in reason


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


def test_c_odd_structures_on_self_conjugate_particles_are_refused() -> None:
    # d^mu(dbar d) Zp_mu is C-even only for a C-even Zp, the vector current
    # only for a C-odd one: no C parity of Zp keeps both.
    density = "P(3,1)*Identity(2,1) + P(3,2)*Identity(2,1)"
    both = [
        _term("d~ d zp", "Gamma(3,2,1)", "Identity(1,2)", "complex(0,1)*gv"),
        _term("d~ d zp", density, "Identity(1,2)", "complex(0,1)*k"),
    ]
    reason = _reason(_qcd() + both)
    assert reason is not None and "[d~ d zp]" in reason
    # The vector current alone and the convective current are C-even.
    convective = "P(3,1)*Identity(2,1) - P(3,2)*Identity(2,1)"
    assert _reason([*_qcd(), both[0]]) is None
    assert (
        _reason([*_qcd(), _term("d~ d zp", convective, "Identity(1,2)", "k")]) is None
    )
    # Tr G{G,G} (the d symbol) is C-odd, unlike the f-type vertex.
    d_symbol = _term("g g g", _VVV1, "d(1,2,3)", "complex(0,1)*dg")
    reason = _reason([*_qcd(), d_symbol])
    assert reason is not None and "[g g g]" in reason


def test_unknown_structures_fail_closed() -> None:
    four_fermion = _term("d~ d s~ s", "Identity(2,1)*Identity(4,3)", "1", "c4")
    reason = _reason([*_qcd(), four_fermion])
    assert reason is not None and "cannot be checked" in reason
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


def test_builtin_vertices_compare_couplings_and_known_self_conjugate_kinds() -> None:
    def vertex(
        particles: tuple[int, ...],
        coupling: tuple[float, float],
        *,
        c_even: bool = True,
    ) -> SymmetryVertex:
        return SymmetryVertex(
            particles=particles,
            orders=(),
            signature=("current", coupling),
            c_signature=coupling,
            c_even=c_even,
        )

    def reason(vertices: list[SymmetryVertex]) -> str | None:
        antiparticles = {1: -1, -1: 1, 3: -3, -3: 3, 32: 32}
        symmetry = VertexSymmetryTable(vertices, antiparticles)
        return symmetry._conjugation_reason(symmetry.allowed({}), (1, 3), _TABLE)  # type: ignore[arg-type]

    pair = [vertex((-1, 3, 32), (1.0, 1.0)), vertex((-3, 1, 32), (1.0, 1.0))]
    assert reason(pair) is None
    differing = [vertex((-1, 3, 32), (1.0, 1.0)), vertex((-3, 1, 32), (1.0, -1.0))]
    found = reason(differing)
    assert found is not None and "coupling or structure" in found
    found = reason([vertex((-1, 1, 32), (1.0, 1.0), c_even=False)])
    assert found is not None and "cannot be checked" in found


@pytest.fixture(scope="module")
def toy_model(tmp_path_factory: pytest.TempPathFactory) -> CompiledModel:
    """The SM plus a C-even (CP-odd) Levi-Civita ``a a H`` vertex and a C-odd
    density coupling ``d~ d a``, each with its own coupling order."""

    root = tmp_path_factory.mktemp("conjugation-toy-model")
    model = json.loads((_SM / "sm.json").read_text(encoding="utf-8"))
    model["orders"] += [
        {"expansion_order": 99, "hierarchy": 3, "name": "NPE"},
        {"expansion_order": 99, "hierarchy": 4, "name": "NPD"},
    ]
    model["couplings"] += [
        {
            "expression": "1𝑖*UFO::{}::aS",  # noqa: RUF001
            "name": f"GC_{order}",
            "orders": [[order, 1]],
            "value": [0.0, 0.118],
        }
        for order in ("NPE", "NPD")
    ]
    model["lorentz_structures"] += [
        {
            "name": "VVS_EPSILON",
            "spins": [3, 3, 1],
            "structure": "UFO::{}::Epsilon(1,2,-1,-2)*UFO::{}::P(-1,1)"
            "*UFO::{}::P(-2,2)",
        },
        {
            "name": "FFV_DENSITY",
            "spins": [2, 2, 3],
            "structure": "UFO::{}::P(3,1)*UFO::{}::Identity(2,1)"
            "+UFO::{}::P(3,2)*UFO::{}::Identity(2,1)",
        },
    ]
    model["vertex_rules"] += [
        {
            "color_structures": ["1"],
            "couplings": [["GC_NPE"]],
            "lorentz_structures": ["VVS_EPSILON"],
            "name": "V_NPE",
            "particles": ["a", "a", "H"],
        },
        {
            "color_structures": ["UFO::{}::Identity(1,2)"],
            "couplings": [["GC_NPD"]],
            "lorentz_structures": ["FFV_DENSITY"],
            "name": "V_NPD",
            "particles": ["d~", "d", "a"],
        },
    ]
    (root / "toy.json").write_text(json.dumps(model), encoding="utf-8")
    shutil.copy(_SM / "restrict_default.json", root / "restrict_default.json")
    handle = ModelSource(
        kind="json",
        path=root / "toy.json",
        restriction=root / "restrict_default.json",
    ).compile(use_cache=False)
    return cast(CompiledModel, _compiled_model_payload(handle))


def test_compiled_structures_are_transformed_in_their_production_form(
    toy_model: CompiledModel,
) -> None:
    # Compiled sources wrap indices (UFO::{}::idx(1,n), UFO::{}::dummy(k)).
    source = next(
        term.lorentz_source
        for term in toy_model.ir.vertex_terms
        if term.vertex == "V_NPE"
    )
    assert "UFO::{}::dummy(1)" in source
    table = particle_table(
        toy_model.ir.particles,
        {parameter.name: parameter for parameter in toy_model.ir.parameters},
    )
    symmetry = VertexSymmetryTable.from_compiled_ir(toy_model.ir)

    def reason(**limits: int) -> str | None:
        return symmetry._conjugation_reason(symmetry.allowed(limits), (1,), table)

    # QCD, ghosts included, is C symmetric.
    assert reason(QCD=1, QED=0, NPE=0, NPD=0) is None
    found = reason(QCD=1, QED=0, NPE=1, NPD=0)
    assert found is not None and "Levi-Civita" in found and "[a a H]" in found
    found = reason(QCD=1, QED=0, NPE=0, NPD=1)
    assert found is not None and "coupling or structure" in found
    assert "[d~ d a]" in found
