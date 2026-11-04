"""FKSLimits: the soft and collinear limits of the real matrix element from
the Born blocks, as sbornsoft, sborncol_fsr and sborncol_isr in
fks_singular.f, per NLO order.

The reference data (see test_fks_reference.py) store, for each generated
event (also those close to the limits), sreal at its soft, collinear and
soft-collinear counter-events, which is there the limit of xi^2 (1 - y) times
the real matrix element of MadFKS, with its parts per position of the
amp_split arrays, and at its Born the Born blocks of pyAmpliCol: the Born, its
color-correlated versions and, for a gluon or photon emitter, the tensor of
its spin-correlated version. From these blocks, with the spin-correlated Born
contracted with k_perp of FKSSplittingKernels, FKSLimits must give the same
on the counter-events at each NLO order, for QCD and QED splittings. This
fixes the conventions of the Born blocks that FKSLimits takes.
"""

import numpy as np
import pytest
from fks_testing import (
    FKS_DATA,
    FKS_DATA_IDS,
    SPLITTING_TYPES,
    alpha,
    at_nlo_orders,
    configuration,
    evaluate,
    order_positions,
    per_born_order,
    reference,
)
from fks_testing import reference_born_mapping as born_mapping

import madspace as ms
from madspace.fks import fks_configurations, fks_limits, fks_splitting_kernels

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
    """born, color_correlated, spin_correlated (per Born order), alpha_s and
    alpha of the points, with the tensor of the spin-correlated Born (zero
    where the emitter is not a gluon or a photon) contracted with k_perp of
    FKSSplittingKernels at the event, and their fks_index."""
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
        per_born_order([b["born"] for b in blocks], data),
        per_born_order([b["color_correlated"] for b in blocks], data),
        per_born_order(np.einsum("bm,bmn,bn->b", k_perp, tensors, k_perp), data),
        np.full(len(rows), data["born_blocks"]["alpha_s"]),
        np.full(len(rows), alpha(data)),
        index,
    )


def soft_scale(rows):
    """Twice iden_comp times the sum of the magnitudes of the terms of the
    soft limit at the soft counter-event, from the eikonal factors and the
    color- or charge-linked Borns of the Fortran (with the coupling): the
    scale of its rounding where the terms cancel."""
    scale = np.zeros(len(rows))
    for k, (conf, point) in enumerate(rows):
        links = point.get("color_links", [])
        if len(links) != len(point["eik_soft"]):
            continue
        scale[k] = (
            2 * conf["iden_comp"] * np.abs(np.multiply(point["eik_soft"], links)).sum()
        )
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
    orders = data["fks_configurations"]["orders"]
    assert limits.born_order_count() == len(orders["born"])
    assert limits.nlo_order_count() == len(orders["nlo"])
    rows = generated_points(data)
    *blocks, index = born_blocks(data, rows)
    soft, collinear, soft_collinear = evaluate(
        limits, *counter_events(rows), *blocks, fks_index=index
    )
    sreal = np.array([point["sreal"] for _, point in rows])
    sreal_orders = np.array([point["sreal_orders"] for _, point in rows])
    # the real emission and its limits have no part at the Born orders
    born_positions, _ = order_positions(data)
    np.testing.assert_array_equal(sreal_orders[..., born_positions], 0.0)
    theirs = at_nlo_orders(sreal_orders, data)
    y = np.array([point["y_ev"] for _, point in rows])
    massive = np.array(
        [conf["masses"][conf["j_fks"] - 1] > 0.0 for conf, _ in rows], dtype=bool
    )
    cases = [
        # sreal takes the collinear limit at the soft counter-event of y = 1
        ("soft", soft, theirs[:, 1], (sreal[:, 1] != -1.0) & (1 - y >= TINY)),
        ("collinear", collinear, theirs[:, 2], sreal[:, 2] != -1.0),
        ("soft_collinear", soft_collinear, theirs[:, 3], sreal[:, 3] != -1.0),
    ]
    # the terms of the soft limit cancel for massive emitters
    scales = {"soft": soft_scale(rows)[:, None]}
    for name, ours, expected, valid in cases:
        assert valid.any() or massive.all(), name
        assert ours.shape == expected.shape
        # the Born of pyAmpliCol and the Fortran agree to 1e-12
        tolerance = 1e-11 * np.abs(expected) + 1e-13 * scales.get(name, 0.0)
        deviation = np.abs(ours - expected)
        assert (deviation[valid] <= tolerance[valid]).all(), name
    # zero without the limit: i not a gluon or photon, or a massive emitter
    soft_limit = np.array([limits.soft(c) for c in index], dtype=bool)
    np.testing.assert_array_equal(soft[~soft_limit], 0.0)
    np.testing.assert_array_equal(collinear[massive], 0.0)
    np.testing.assert_array_equal(soft_collinear[massive], 0.0)


@pytest.mark.parametrize("path", FKS_DATA, ids=FKS_DATA_IDS)
def test_born_orders_of_the_counterterms(path):
    """The Born order of the counterterms at each NLO order is that of
    amp_split_cnt of the Fortran: the Born at its position for the splitting
    type of the configuration, zero where the counterterms have none."""
    data = reference(path)
    limits = fks_limits(born_mapping(data), data["fks_configurations"])
    configurations = fks_configurations(born_mapping(data), data["fks_configurations"])
    for conf, point in generated_points(data):
        c = conf["nfks"] - 1
        assert limits.qed(c) == (configurations[c].splitting_types == ["QED"])
        born_orders = np.array(point["born_orders"])
        expected = at_nlo_orders(
            born_orders[SPLITTING_TYPES[configurations[c].splitting_types[0]]], data
        )
        born = point["born_blocks"]["born"]
        for b, value in zip(limits.born_index(c), expected):
            assert value == (pytest.approx(born, rel=1e-11) if b >= 0 else 0.0)


@pytest.mark.parametrize("path", FKS_DATA, ids=FKS_DATA_IDS)
def test_spin_correlations_enter(path):
    """Where the emitter is a gluon or a photon and the azimuthal terms do not
    vanish, the collinear limit depends on the azimuth of the emission: with
    the spin-correlated Born at its azimuthal average B / 2 it differs from
    that of MadFKS."""
    data = reference(path)
    limits = fks_limits(born_mapping(data), data["fks_configurations"])
    rows = generated_points(data)
    born, correlated, spin, alpha_s, alpha_qed, index = born_blocks(data, rows)
    (_, collinear, _), (_, averaged, _) = (
        evaluate(
            limits,
            *counter_events(rows),
            born,
            correlated,
            spin_correlated,
            alpha_s,
            alpha_qed,
            fks_index=index,
        )
        for spin_correlated in [spin, 0.5 * born]
    )
    vector = np.array(["spin_correlated" in p["born_blocks"] for _, p in rows])
    if not vector.any():
        np.testing.assert_array_equal(collinear, averaged)
        return
    np.testing.assert_array_equal(collinear[~vector], averaged[~vector])
    if np.allclose(spin[vector], 0.5 * born[vector], rtol=1e-12, atol=0):
        # the Born does not depend on the linear polarization of one photon
        # (a a > e- e+ with massless leptons)
        return
    sreal = at_nlo_orders([point["sreal_orders"] for _, point in rows], data)[:, 2]
    differs = (sreal != 0.0) & (np.abs(averaged - sreal) > 1e-6 * np.abs(sreal))
    assert (differs & vector[:, None]).any()


def test_qed_soft_limit():
    """The soft limit of a photon i is that of the charge-linked Borns
    eta_m Q_m eta_n Q_n B with the QED coupling, with eta = -1 for an incoming
    particle and a half for a massive particle with itself."""
    # u u~ > t t~ a, the photon i forming FKS pairs with all quarks
    pdgs = [2, -2, 6, -6, 22]
    charges = [2 / 3, -2 / 3, 2 / 3, -2 / 3, 0.0]
    masses = [0.0, 0.0, 173.0, 173.0, 0.0]
    pairs = [[], [], [], [], [0, 1, 2, 3]]
    config = configuration(
        4,
        2,
        pdgs,
        [3, -3, 3, -3, 1],
        charges,
        masses,
        fks_j_from_i=pairs,
        splitting_types=["QED"],
    )
    orders = ms.FKSCouplingOrders(
        split_orders=["QCD", "QED"], born=[[0, 4]], nlo=[[0, 6]], max_nlo=[0, 6]
    )
    limits = ms.FKSLimits([config], orders)
    assert limits.soft(0) and limits.qed(0) and not limits.collinear(0)
    rng = np.random.default_rng(3)
    batch = 5
    # momenta and the direction of i, with a soft i
    p = rng.normal(size=(batch, 5, 4))
    p[..., 0] = np.linalg.norm(p[..., 1:], axis=-1) + np.array(masses)
    p_i_hat = p[:, 4].copy()
    p[:, 4] = 0.0
    y = rng.uniform(-0.9, 0.9, batch)
    born = rng.uniform(0.5, 2.0, (batch, 1))
    alpha_s, alpha_qed = np.full(batch, 0.118), np.full(batch, 0.0078)
    eikonals = ms.FKSEikonals([config])
    (eik,) = evaluate(eikonals, p, p_i_hat, y, fks_index=np.zeros(batch))
    expected = np.zeros(batch)
    eta = np.array([-1.0, -1.0, 1.0, 1.0, 1.0])
    for k, (m, n) in enumerate(eikonals.pairs(0)):
        weight = 0.5 if m == n else 1.0
        link = eta[m] * charges[m] * eta[n] * charges[n] * born[:, 0]
        expected += -2 * weight * eik[:, k] * link
    expected *= 4 * np.pi * alpha_qed * config.identical_particle_ratio()
    soft, collinear, _ = evaluate(
        limits,
        p,
        p_i_hat,
        y,
        p,
        p,
        born,
        rng.normal(size=(batch, 1, 4, 4)),
        born / 2,
        alpha_s,
        alpha_qed,
        fks_index=np.zeros(batch),
    )
    np.testing.assert_allclose(soft[:, 0], expected, rtol=1e-14)
    np.testing.assert_array_equal(collinear, 0.0)


def test_coupling_orders():
    """A configuration contributes at the NLO orders of the counterterms of its
    splitting type, Born order plus two powers of its coupling, where these
    are at most the largest NLO orders."""
    # u u~ > t t~ g at the Born orders of u u~ > t t~ with QCD^2 <= 4 and
    # QED^2 <= 4 and QCD corrections
    pdgs = [2, -2, 6, -6, 21]
    colors = [3, -3, 3, -3, 8]
    masses = [0.0, 0.0, 173.0, 173.0, 0.0]
    pairs = [[], [], [], [], [0, 1, 2, 3]]
    gluon = configuration(4, 0, pdgs, colors, masses=masses, fks_j_from_i=pairs)
    orders = ms.FKSCouplingOrders(
        split_orders=["QCD", "QED"],
        born=[[0, 4], [2, 2], [4, 0]],
        nlo=[[0, 6], [2, 4], [4, 2], [6, 0]],
        max_nlo=[6, 4],
    )
    assert [orders.counterterm_order(b, "QCD") for b in range(3)] == [1, 2, 3]
    # QED counterterms above the largest QED order
    assert orders.counterterm_order(0, "QED") == -1
    limits = ms.FKSLimits([gluon], orders)
    assert limits.born_index(0) == [-1, 0, 1, 2]
    batch = 3
    rng = np.random.default_rng(5)
    p = rng.normal(size=(batch, 5, 4))
    p[..., 0] = np.linalg.norm(p[..., 1:], axis=-1) + np.array(masses)
    born = rng.uniform(0.5, 2.0, (batch, 3))
    correlated = rng.normal(size=(batch, 3, 4, 4))
    out = evaluate(
        limits,
        p,
        p[:, 4],
        np.full(batch, 0.3),
        p,
        p,
        born,
        correlated,
        born / 2,
        np.full(batch, 0.118),
        np.full(batch, 0.0078),
        fks_index=np.zeros(batch),
    )
    # each NLO order takes the blocks of its Born order alone
    single = ms.FKSLimits([gluon], ms.FKSCouplingOrders.qcd(1))
    for o, b in enumerate(limits.born_index(0)):
        if b < 0:
            for value in out:
                np.testing.assert_array_equal(value[:, o], 0.0)
            continue
        expected = evaluate(
            single,
            p,
            p[:, 4],
            np.full(batch, 0.3),
            p,
            p,
            born[:, [b]],
            correlated[:, [b]],
            born[:, [b]] / 2,
            np.full(batch, 0.118),
            np.full(batch, 0.0078),
            fks_index=np.zeros(batch),
        )
        for value, single_value in zip(out, expected):
            np.testing.assert_array_equal(value[:, o], single_value[:, 0])
    with pytest.raises(ValueError, match="NLO order"):
        # a QCD counterterm of order (4, 2) that is not an NLO order
        ms.FKSLimits(
            [gluon],
            ms.FKSCouplingOrders(
                split_orders=["QCD", "QED"],
                born=[[2, 2]],
                nlo=[[2, 4]],
                max_nlo=[6, 4],
            ),
        )
    assert ms.FKSCouplingOrders.qcd(2).nlo == [[6]]


def test_configurations():
    """The soft limit is that of a gluon or photon i, the collinear ones those
    of a massless emitter, and the identical-particle ratio enters both."""
    # e+ e- > t t~ g g with (i, j) = (5, 4) and (5, 2), and e+ e- > t t~ u u~
    # with (i, j) = (5, 4)
    pdgs = [-11, 11, 6, -6, 21, 21]
    colors = [1, 1, 3, -3, 8, 8]
    masses = [0.0, 0.0, 173.0, 173.0, 0.0, 0.0]
    pairs = [[], [], [], [], [], [2, 3, 4]]
    quarks = [-11, 11, 6, -6, 2, -2]
    qcd = ms.FKSCouplingOrders.qcd(0)
    limits = ms.FKSLimits(
        [
            configuration(5, 4, pdgs, colors, masses=masses, fks_j_from_i=pairs),
            configuration(5, 2, pdgs, colors, masses=masses, fks_j_from_i=pairs),
            configuration(5, 4, quarks, [1, 1, 3, -3, 3, -3], masses=masses),
        ],
        qcd,
    )
    assert [limits.soft(c) for c in range(3)] == [True, True, False]
    assert [limits.collinear(c) for c in range(3)] == [True, False, True]
    assert [limits.qed(c) for c in range(3)] == [False] * 3
    assert limits.eikonals().born_pairs(1) == [(2, 2), (3, 2), (3, 3), (4, 2), (4, 3)]
    # u u~ > e- e+ a with a photon i and e- e+ > u u~ off a photon (QED)
    qed = ms.FKSCouplingOrders(
        split_orders=["QCD", "QED"], born=[[0, 4]], nlo=[[0, 6]], max_nlo=[0, 6]
    )
    leptons = [2, -2, 11, -11, 22]
    charges = [2 / 3, -2 / 3, -1.0, 1.0, 0.0]
    photon = configuration(
        4,
        2,
        leptons,
        [3, -3, 1, 1, 1],
        charges,
        fks_j_from_i=[[], [], [], [], [0, 1, 2, 3]],
        splitting_types=["QED"],
    )
    pair = configuration(
        4,
        3,
        [-11, 11, 2, 2, -2],
        [1, 1, 3, 3, -3],
        [1.0, -1.0, 2 / 3, 2 / 3, -2 / 3],
        splitting_types=["QED"],
    )
    assert pair.born_pdgs() == [-11, 11, 2, 22]
    limits = ms.FKSLimits([photon, pair], qed)
    assert [limits.soft(c) for c in range(2)] == [True, False]
    assert [limits.qed(c) for c in range(2)] == [True, True]


def test_invalid_configurations():
    pdgs = [2, -2, 11, -11, 22]
    colors = [3, -3, 1, 1, 1]
    charges = [2 / 3, -2 / 3, -1.0, 1.0, 0.0]
    pairs = [[], [], [], [], [0, 1, 2, 3]]
    qcd = ms.FKSCouplingOrders.qcd(0)
    with pytest.raises(ValueError, match="QCD"):
        # a photon i in a QCD splitting
        ms.FKSLimits(
            [configuration(4, 2, pdgs, colors, charges, fks_j_from_i=pairs)], qcd
        )
    with pytest.raises(ValueError, match="split order"):
        # a QED splitting without the split order QED
        ms.FKSLimits(
            [
                configuration(
                    4,
                    2,
                    pdgs,
                    colors,
                    charges,
                    fks_j_from_i=pairs,
                    splitting_types=["QED"],
                )
            ],
            qcd,
        )
    with pytest.raises(ValueError, match="extra counterterm"):
        # a quark pair from a gluon or a photon
        ms.FKSLimits(
            [
                configuration(
                    4,
                    3,
                    [-11, 11, 2, 2, -2],
                    [1, 1, 3, 3, -3],
                    [1.0, -1.0, 2 / 3, 2 / 3, -2 / 3],
                    splitting_types=["QCD", "QED"],
                )
            ],
            ms.FKSCouplingOrders(
                split_orders=["QCD", "QED"],
                born=[[2, 4]],
                nlo=[[4, 4], [2, 6]],
                max_nlo=[4, 6],
            ),
        )
    with pytest.raises(ValueError, match="splitting types"):
        ms.FKSLimits(
            [configuration(4, 2, pdgs, colors, charges, splitting_types=["EW"])], qcd
        )
    with pytest.raises(ValueError, match="splitting"):
        # a final-state u and an incoming u~ come from no splitting
        ms.FKSLimits([configuration(4, 1, [2, -2, 11, -11, 2], [3, -3, 1, 1, 3])], qcd)
    with pytest.raises(ValueError, match="power"):
        ms.FKSCouplingOrders(
            split_orders=["QCD"], born=[[0, 4]], nlo=[[2]], max_nlo=[2]
        )
