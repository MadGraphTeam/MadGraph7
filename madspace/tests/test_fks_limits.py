"""FKSLimits: the soft and collinear limits of the real matrix element from
the Born blocks, as sbornsoft, sborncol_fsr and sborncol_isr in
fks_singular.f.

The reference data (see test_fks_reference.py) store, for each generated
event (also those close to the limits), sreal at its soft, collinear and
soft-collinear counter-events, which is there the limit of xi^2 (1 - y) times
the real matrix element of MadFKS, and at its Born the Born blocks of
pyAmpliCol: the Born, its color-correlated versions and, for a gluon emitter,
the tensor of its spin-correlated version. From these blocks, with the
spin-correlated Born contracted with k_perp of FKSSplittingKernels, FKSLimits
must give the same on the counter-events. This fixes the conventions of the
Born blocks that FKSLimits takes.
"""

import numpy as np
import pytest
from fks_testing import FKS_DATA, FKS_DATA_IDS, configuration, evaluate, reference
from fks_testing import reference_born_mapping as born_mapping

import madspace as ms
from madspace.fks import fks_limits, fks_splitting_kernels

SOFT, COLLINEAR, SOFT_COLLINEAR = 2, 3, 4
TINY = 1e-6  # tiny of sreal


def generated_points(data):
    """The configurations and points of the generated events, those close to
    the limits included."""
    return [
        (conf, point)
        for conf in data["configs"]
        for point in conf["points"] + conf["limit_points"]
        if "born_blocks" in point
    ]


def born_blocks(data, rows):
    """born, color_correlated, spin_correlated and alpha_s of the points, with
    the tensor of the spin-correlated Born (zero where the emitter is not a
    gluon) contracted with k_perp of FKSSplittingKernels at the event, and
    their fks_index."""
    index = np.array([conf["nfks"] - 1 for conf, _ in rows], dtype=np.int32)
    kernels = fks_splitting_kernels(born_mapping(data), data["fks_configurations"])
    *_, k_perp = evaluate(
        kernels,
        np.array([point["p_ev"] for _, point in rows]),
        np.array([point["p_i_hat_ev"] for _, point in rows]),
        np.array([point["p_born"] for _, point in rows]),
        fks_index=index,
    )
    blocks = [point["born_blocks"] for _, point in rows]
    tensors = np.array([b.get("spin_correlated", np.zeros((4, 4))) for b in blocks])
    return (
        np.array([b["born"] for b in blocks]),
        np.array([b["color_correlated"] for b in blocks]),
        np.einsum("bm,bmn,bn->b", k_perp, tensors, k_perp),
        np.full(len(rows), data["born_blocks"]["alpha_s"]),
        index,
    )


def soft_scale(data, rows):
    """4 pi alpha_s iden_comp times the sum of the magnitudes of the terms of
    the soft limit at the soft counter-event, from the eikonal factors of the
    Fortran: the scale of its rounding where the terms cancel."""
    alpha_s = data["born_blocks"]["alpha_s"]
    scale = np.zeros(len(rows))
    for k, (conf, point) in enumerate(rows):
        i_fks = conf["i_fks"]
        correlated = np.array(point["born_blocks"]["color_correlated"])
        for (m, n), e in zip(point["eik_pairs"], point["eik_soft"]):
            b = correlated[m - (m > i_fks) - 1, n - (n > i_fks) - 1]
            scale[k] += (1.0 if m == n else 2.0) * abs(e * b)
        scale[k] *= 4 * np.pi * alpha_s * conf["iden_comp"]
    return scale


def counter_events(rows):
    """The arguments of FKSLimits from the soft, collinear and soft-collinear
    counter-events of the points."""

    def field(key, position=None):
        values = np.array([point[key] for _, point in rows])
        return values if position is None else values[:, position]

    return [
        field("p_cnt", SOFT),
        field("p_i_hat_cnt", SOFT),
        field("y_ev"),
        field("p_cnt", COLLINEAR),
        field("p_cnt", SOFT_COLLINEAR),
    ]


@pytest.mark.parametrize("path", FKS_DATA, ids=FKS_DATA_IDS)
def test_matches_sreal(path):
    data = reference(path)
    limits = fks_limits(born_mapping(data), data["fks_configurations"])
    assert limits.configuration_count() == len(data["configs"])
    rows = generated_points(data)
    *blocks, index = born_blocks(data, rows)
    soft, collinear, soft_collinear = evaluate(
        limits, *counter_events(rows), *blocks, fks_index=index
    )
    sreal = np.array([point["sreal"] for _, point in rows])
    y = np.array([point["y_ev"] for _, point in rows])
    massive = np.array(
        [conf["masses"][conf["j_fks"] - 1] > 0.0 for conf, _ in rows], dtype=bool
    )
    cases = [
        # sreal takes the collinear limit at the soft counter-event of y = 1
        ("soft", soft, sreal[:, 1], (sreal[:, 1] != -1.0) & (1 - y >= TINY)),
        ("collinear", collinear, sreal[:, 2], sreal[:, 2] != -1.0),
        ("soft_collinear", soft_collinear, sreal[:, 3], sreal[:, 3] != -1.0),
    ]
    # the terms of the soft limit cancel for massive emitters
    scales = {"soft": soft_scale(data, rows)}
    for name, ours, theirs, valid in cases:
        assert valid.any() or massive.all(), name
        # the Born of pyAmpliCol and the Fortran agree to 1e-12
        tolerance = 1e-11 * np.abs(theirs) + 1e-13 * scales.get(name, 0.0)
        deviation = np.abs(ours - theirs)
        assert (deviation[valid] <= tolerance[valid]).all(), name
    # zero without the limit: i a quark, or a massive emitter
    soft_limit = np.array([limits.soft(c) for c in index], dtype=bool)
    np.testing.assert_array_equal(soft[~soft_limit], 0.0)
    np.testing.assert_array_equal(collinear[massive], 0.0)
    np.testing.assert_array_equal(soft_collinear[massive], 0.0)


@pytest.mark.parametrize("path", FKS_DATA, ids=FKS_DATA_IDS)
def test_spin_correlations_enter(path):
    """Where the emitter is a gluon and the azimuthal terms do not vanish, the
    collinear limit depends on the azimuth of the emission: with the
    spin-correlated Born at its azimuthal average B / 2 it differs from that
    of MadFKS."""
    data = reference(path)
    limits = fks_limits(born_mapping(data), data["fks_configurations"])
    rows = generated_points(data)
    born, correlated, spin, alpha_s, index = born_blocks(data, rows)
    (_, collinear, _), (_, averaged, _) = (
        evaluate(
            limits,
            *counter_events(rows),
            born,
            correlated,
            spin_correlated,
            alpha_s,
            fks_index=index,
        )
        for spin_correlated in [spin, 0.5 * born]
    )
    gluon = np.array(["spin_correlated" in p["born_blocks"] for _, p in rows])
    if not gluon.any():
        np.testing.assert_array_equal(collinear, averaged)
        return
    sreal = np.array([point["sreal"] for _, point in rows])[:, 2]
    differs = (sreal != -1.0) & (np.abs(averaged - sreal) > 1e-6 * np.abs(sreal))
    assert (differs & gluon).any()
    np.testing.assert_array_equal(collinear[~gluon], averaged[~gluon])


def test_configurations():
    """The soft limit is that of a gluon i, the collinear ones those of a
    massless emitter, and the identical-particle ratio enters both."""
    # e+ e- > t t~ g g with (i, j) = (5, 4) and (5, 2), and e+ e- > t t~ u u~
    # with (i, j) = (5, 4)
    pdgs = [-11, 11, 6, -6, 21, 21]
    colors = [1, 1, 3, -3, 8, 8]
    masses = [0.0, 0.0, 173.0, 173.0, 0.0, 0.0]
    pairs = [[], [], [], [], [], [2, 3, 4]]
    quarks = [-11, 11, 6, -6, 2, -2]
    limits = ms.FKSLimits(
        [
            configuration(5, 4, pdgs, colors, masses=masses, fks_j_from_i=pairs),
            configuration(5, 2, pdgs, colors, masses=masses, fks_j_from_i=pairs),
            configuration(5, 4, quarks, [1, 1, 3, -3, 3, -3], masses=masses),
        ]
    )
    assert [limits.soft(c) for c in range(3)] == [True, True, False]
    assert [limits.collinear(c) for c in range(3)] == [True, False, True]
    assert limits.eikonals().born_pairs(1) == [(2, 2), (3, 2), (3, 3), (4, 2), (4, 3)]


def test_invalid_configurations():
    pdgs = [2, -2, 11, -11, 22]
    colors = [3, -3, 1, 1, 1]
    charges = [2 / 3, -2 / 3, -1.0, 1.0, 0.0]
    pairs = [[], [], [], [], [0, 1, 2, 3]]
    with pytest.raises(ValueError, match="QCD"):
        # a photon i
        ms.FKSLimits([configuration(4, 2, pdgs, colors, charges, fks_j_from_i=pairs)])
    with pytest.raises(ValueError, match="splitting"):
        # a final-state u and an incoming u~ come from no splitting
        ms.FKSLimits([configuration(4, 1, [2, -2, 11, -11, 2], [3, -3, 1, 1, 3])])
