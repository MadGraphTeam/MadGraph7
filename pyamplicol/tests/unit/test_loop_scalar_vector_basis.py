# SPDX-License-Identifier: 0BSD
"""Source-key scope of locally tested neutral scalar/vector mixing tensors."""

from types import SimpleNamespace

import pytest


def source(entries, *, pdgs=(250, 23), spins=(1, 3), colors=(1, 1)):
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
                    {"fields": {"pdg_code": pdg, "spin": spin, "color": color}}
                    for pdg, spin, color in zip(pdgs, spins, colors, strict=True)
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
    for pdgs in ((250, 22), (250, 23)):
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
        {"pdgs": (23, 250)},
        {"pdgs": (251, 24)},
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
        model = source((), pdgs=pdgs, spins=spins, colors=colors)
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

    for pdgs in ((-13, 13, 250), (-13, 13, 25), (-11, 11, 25)):
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
