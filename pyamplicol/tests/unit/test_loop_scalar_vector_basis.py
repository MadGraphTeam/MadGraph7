# SPDX-License-Identifier: 0BSD
"""Source-key scope of locally tested neutral scalar/vector mixing tensors.

The certificates are structural (genericity audit m5): spins, colours, charge
conjugation (``name``/``antiname``) and the exact tensors, never PDG codes.
"""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.loop_fast


def _particle(pdg, spin, color, neutral):
    name = f"p{pdg}"
    return {
        "fields": {
            "pdg_code": pdg,
            "spin": spin,
            "color": color,
            "name": name,
            "antiname": name if neutral else f"p{-pdg}",
        }
    }


def source(
    entries, *, pdgs=(250, 23), spins=(1, 3), colors=(1, 1), neutral=(True, True)
):
    return SimpleNamespace(
        kernels=tuple(
            SimpleNamespace(
                key=("counterterm_vertices", index, 0, 0, 0),
                kind=kind,
                particle_ids=(0, 1),
                color=color,
                lorentz=lorentz,
            )
            for index, (kind, color, lorentz) in enumerate(entries)
        ),
        raw=SimpleNamespace(
            collections={
                "particles": tuple(
                    _particle(*row)
                    for row in zip(pdgs, spins, colors, neutral, strict=True)
                )
            }
        ),
    )


def test_neutral_scalar_vector_evidence_retains_exact_keys_and_port_scope():
    pytest.importorskip("symbolica")
    from pyamplicol.loops.model_electroweak import (
        neutral_scalar_vector_counterterm_validations,
    )

    records = (("UVmass", "1", "P(2,1)"), ("UVmass", "1", "P(2,2)"))
    # Renumbered neutral species are the same structure.
    for pdgs in ((250, 22), (250, 23), (9000250, 9000023)):
        model = source(records, pdgs=pdgs)
        actual = neutral_scalar_vector_counterterm_validations(model)
        assert tuple(v.source_key for v in actual) == tuple(
            e.key for e in model.kernels
        )
        assert tuple(v.lorentz for v in actual) == ("P(2,1)", "P(2,2)")
        assert all(
            v.particle_pdgs == pdgs and v.particle_spins == (1, 3) for v in actual
        )
    # Same-name or momentum-conservation coincidences do not broaden evidence.
    bad = source(
        (
            ("tree", "1", "P(2,1)"),
            ("UVmass", "2", "P(2,1)"),
            ("UVmass", "1", "P(1,2)"),
            ("UVmass", "1", "P(2,1)+P(2,2)"),
            ("UVmass", "1", "-P(2,1)"),
        )
    )
    assert neutral_scalar_vector_counterterm_validations(bad) == ()
    for overrides in (
        {"pdgs": (251, 24), "neutral": (False, False)},
        {"neutral": (True, False)},
        {"spins": (3, 3)},
        {"colors": (1, 8)},
    ):
        assert (
            neutral_scalar_vector_counterterm_validations(source(records, **overrides))
            == ()
        )


def test_massive_neutral_ffs_evidence_keeps_exact_source_keys_and_ports():
    pytest.importorskip("symbolica")
    from pyamplicol.loops.model_electroweak import (
        massive_neutral_lepton_counterterm_validations,
    )

    def candidate(*, pdgs=(-13, 13, 250), spins=(2, 2, 1), colors=(1, 1, 1)):
        # A fermion and its antiparticle (opposite PDG codes) and a scalar
        # that is its own antiparticle.
        neutral = tuple(spin == 1 for spin in spins)
        model = source((), pdgs=pdgs, spins=spins, colors=colors, neutral=neutral)
        model.kernels = tuple(
            SimpleNamespace(
                key=("counterterm_vertices", index, 0, 0, 0),
                kind="R2",
                particle_ids=(0, 1, 2),
                color="1",
                lorentz=lorentz,
            )
            for index, lorentz in enumerate(
                (
                    "ProjM(2,1)-ProjP(2,1)",
                    "ProjM(2,1)+ProjP(2,1)",
                    "Gamma5(2,1)",  # unadmitted source spelling/basis
                )
            )
        )
        return model

    for pdgs in ((-13, 13, 250), (-13, 13, 25), (-11, 11, 25), (-15, 15, 9000250)):
        model = candidate(pdgs=pdgs)
        result = massive_neutral_lepton_counterterm_validations(model)
        assert tuple(row.source_key for row in result) == tuple(
            row.key for row in model.kernels[:2]
        )
    for overrides in (
        {"pdgs": (13, -13, 250)},
        {"pdgs": (-14, 13, 251)},
        {"spins": (2, 2, 3)},
        {"colors": (3, 3, 1)},
    ):
        assert (
            massive_neutral_lepton_counterterm_validations(candidate(**overrides)) == ()
        )


def test_neutral_ffv_evidence_is_structural_for_every_singlet_fermion_pair():
    """m5: a tau or neutrino QED/Z vertex counterterm gets the muon's
    certificate (any colour-singlet antifermion/fermion pair, the
    antiparticle first, with any self-conjugate vector), never by PDG code."""

    pytest.importorskip("symbolica")
    from pyamplicol.loops.model_electroweak import (
        neutral_lepton_counterterm_validations,
    )

    lorentz = (
        "Gamma(3,2,1)",
        "Gamma(3,2,-1)*ProjM(-1,1)",
        "Gamma5(3,2,1)",  # not a certified FFV basis
    )

    def candidate(
        *,
        pdgs=(-13, 13, 22),
        spins=(2, 2, 3),
        colors=(1, 1, 1),
        neutral=(False, False, True),
    ):
        model = source((), pdgs=pdgs, spins=spins, colors=colors, neutral=neutral)
        model.kernels = tuple(
            SimpleNamespace(
                key=("counterterm_vertices", index, 0, 0, 0),
                kind="UVvertex",
                particle_ids=(0, 1, 2),
                color="1",
                lorentz=structure,
            )
            for index, structure in enumerate(lorentz)
        )
        return model

    controls = {}
    for pdgs in (
        (-13, 13, 22),
        (-15, 15, 22),
        (-15, 15, 23),
        (-12, 12, 23),
        (-15, 15, 9000023),
    ):
        model = candidate(pdgs=pdgs)
        result = neutral_lepton_counterterm_validations(model)
        assert tuple(row.source_key for row in result) == tuple(
            row.key for row in model.kernels[:2]
        )
        assert all(row.particle_pdgs == pdgs for row in result)
        controls[pdgs] = {row.conformance_control for row in result}
    assert len(set(map(frozenset, controls.values()))) == 1
    for overrides in (
        {"pdgs": (15, -15, 22)},  # the fermion first
        {"pdgs": (-15, 13, 22)},  # not a particle/antiparticle pair
        # A charged vector (not its own antiparticle).
        {"pdgs": (-15, 15, 24), "neutral": (False, False, False)},
        {"spins": (2, 2, 1)},
        {"colors": (3, 3, 1)},
    ):
        assert neutral_lepton_counterterm_validations(candidate(**overrides)) == ()
