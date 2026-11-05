"""FKSSplittingKernels: the factors of the FKS collinear limit, as
sborncol_fsr and sborncol_isr in fks_singular.f.

The reference data (see test_fks_reference.py) store, for each point whose
collinear and soft-collinear counter-events genps_fks.f generates, z and the
Altarelli-Parisi kernels and azimuthal terms of AP_reduced and
Qterms_reduced_timelike/spacelike there, without the couplings, and xij_kperp,
the transverse direction of the emission. The kernels of the configurations of
the Born, from the metadata of the FKS exporter, must give the same on those
momenta. On events of the FKS mappings, z is the momentum fraction of the
counter-event and k_perp is a unit vector orthogonal to the Born emitter and to
the incoming momenta.
"""

import numpy as np
import pytest
from fks_testing import EPS, FKS_DATA, FKS_DATA_IDS, configuration, evaluate, reference
from fks_testing import reference_born_mapping as born_mapping

import madspace as ms
from madspace.fks import fks_configuration_mapping, fks_splitting_kernels

# positions k + 2 of the counter-events k = 1, 2 of genps_fks.f
COUNTER_EVENTS = {"collinear": 3, "soft_collinear": 4}


def dot(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def fortran_points(data):
    """The momenta, momentum of i per unit xi at the event, Born and fks_index
    of the collinear and soft-collinear counter-events of the reference
    points, with the factors of sborncol_fsr/isr."""
    rows = []
    for conf in data["configs"]:
        for point in conf["points"]:
            for k, position in enumerate(COUNTER_EVENTS.values()):
                coll = point["coll_cnt"][k]
                if coll[0] == -1.0:
                    continue
                rows.append(
                    (
                        point["p_cnt"][position],
                        point["p_i_hat_ev"],
                        point["p_born"],
                        conf["nfks"] - 1,
                        coll,
                    )
                )
    return [np.array(column) for column in zip(*rows)]


@pytest.mark.parametrize("path", FKS_DATA, ids=FKS_DATA_IDS)
def test_matches_sborncol(path):
    data = reference(path)
    configurations = data["fks_configurations"]
    kernels = fks_splitting_kernels(born_mapping(data), configurations)
    count = len(configurations["configurations"])
    assert kernels.configuration_count() == count
    rows = fortran_points(data)
    if not rows:
        # only massive emitters
        assert all(
            c["j_fks"] > 2 and c["masses"][c["j_fks"] - 1] > 0 for c in data["configs"]
        )
        return
    momenta, p_i_hat, p_born, fks_index, coll = rows
    z, ap, q_terms, _ = evaluate(
        kernels,
        momenta,
        p_i_hat,
        p_born,
        fks_index=fks_index,
    )
    # z from the momenta, as 1 - E_i / (E_i + E_j) or 1 - 2 E_i / sqrt(s)
    np.testing.assert_allclose(z, coll[:, 0], rtol=0, atol=1e-14)
    # the kernels go like z or 1 - z, whose rounding is that of z
    zz = coll[:, 0]
    rtol = 1e-12 + 10 * EPS / np.maximum(np.minimum(zz, 1 - zz), EPS)
    for ours, column in [
        (ap[:, 0], 2),
        (ap[:, 1], 3),
        (q_terms[:, 0], 4),
        (q_terms[:, 1], 5),
    ]:
        deviation = np.abs(ours - coll[:, column])
        assert (deviation <= rtol * np.abs(coll[:, column])).all(), column


def born_position(conf):
    """Position of the emitter in the Born, 0-based."""
    i_fks, j_fks = conf["i_fks"] - 1, conf["j_fks"] - 1
    return j_fks if j_fks < i_fks else j_fks - 1


@pytest.mark.parametrize("path", FKS_DATA, ids=FKS_DATA_IDS)
def test_transverse_vector_matches_xij_kperp(path):
    """k_perp from the momentum of i per unit xi and the Born of genps_fks.f,
    in the partonic center-of-mass frame of the Born, is xij_kperp up to its
    sign, for every emitter."""
    data = reference(path)
    configurations = data["fks_configurations"]
    kernels = fks_splitting_kernels(born_mapping(data), configurations)
    count = len(configurations["configurations"])
    rows = []
    for conf in data["configs"]:
        for point in conf["points"]:
            if point["s_ij_ev"] == -1.0:
                continue
            rows.append(
                (
                    point["p_ev"],
                    point["p_i_hat_ev"],
                    point["p_born"],
                    conf["nfks"] - 1,
                    point["kperp"],
                    born_position(conf),
                )
            )
    momenta, p_i_hat, p_born, fks_index, kperp, born_j = (
        np.array(c) for c in zip(*rows)
    )
    *_, k_perp = evaluate(
        kernels,
        momenta,
        p_i_hat,
        p_born,
        fks_index=fks_index,
    )
    sign = np.sign((k_perp * kperp).sum(1))[:, None]
    # to the precision of the part of p_i_hat transverse to the axis of the
    # emission, eps / sin(theta): the beam for an incoming emitter (theta from
    # y), the Born emitter i + j for a final-state one, which is closer to i
    # than j is where j is soft
    axis = np.zeros((len(rows), 3))
    axis[:, 2] = 1.0
    final = np.array([c >= 2 for c in born_j])
    axis[final] = p_born[final, born_j[final], 1:]
    p_i = p_i_hat[:, 1:]
    sin_theta = np.linalg.norm(np.cross(p_i, axis), axis=1) / (
        np.linalg.norm(p_i, axis=1) * np.linalg.norm(axis, axis=1)
    )
    np.testing.assert_array_less(
        np.abs(sign * k_perp - kperp).max(1), 20 * EPS / np.maximum(sin_theta, EPS)
    )


@pytest.mark.parametrize(
    "name", ["pp_ttx_P0_gg_ttx", "epem_uuxg_P0_epem_uuxg", "pp_epemj_P0_ug_uemep"]
)
def test_on_mapping_events(name):
    """On events of the FKS mappings (lab frame): at the collinear
    counter-event of a final-state emitter p_i = (1 - z) p_jbar and p_j = z
    p_jbar, of an initial-state one p_i = (1 - z) p_j; z = 1 at the
    soft-collinear one; k_perp^2 = -1 and k_perp is orthogonal to the Born
    emitter and to the incoming Born momenta."""
    data = reference(FKS_DATA[FKS_DATA_IDS.index(name)])
    configurations = data["fks_configurations"]
    born = born_mapping(data)
    mapping = fks_configuration_mapping(born, configurations)
    kernels = fks_splitting_kernels(born, configurations)
    count = len(configurations["configurations"])
    names = list(mapping.counter_event_types().keys())
    rng = np.random.default_rng(37)
    batch = 400
    r = 0.05 + 0.9 * rng.random((batch, mapping.random_dim()))
    index = rng.integers(0, count, batch).astype(np.int32)
    out = dict(zip(names, evaluate(ms.FKSKinematics(mapping), r, fks_index=index)))
    i_fks = np.array([c["i_fks"] - 1 for c in configurations["configurations"]])
    j_fks = np.array([c["j_fks"] - 1 for c in configurations["configurations"]])
    ii, jj = i_fks[index], j_fks[index]
    jbar = np.where(jj < ii, jj, jj - 1)
    rows = np.arange(batch)
    for prefix in ["collinear_", "soft_collinear_"]:
        momenta = out[f"{prefix}momenta"]
        z, _, _, k_perp = evaluate(
            kernels,
            momenta,
            out["p_i_hat"],
            out["born_momenta"],
            fks_index=index,
        )
        ok = out[f"{prefix}det"] > 0
        assert ok.mean() > 0.5
        p_i, p_j = momenta[rows, ii], momenta[rows, jj]
        scale = momenta[:, :2, 0].sum(1)[:, None]
        final = ok & (jj >= 2)
        initial = ok & (jj < 2)
        p_jbar = out["born_momenta"][rows, jbar]
        if prefix == "collinear_":
            np.testing.assert_allclose(
                (p_j - z[:, None] * p_jbar)[final] / scale[final], 0, atol=1e-12
            )
            np.testing.assert_allclose(
                (p_i - (1 - z[:, None]) * p_jbar)[final] / scale[final], 0, atol=1e-12
            )
            np.testing.assert_allclose(
                (p_i - (1 - z[:, None]) * p_j)[initial] / scale[initial], 0, atol=1e-12
            )
        else:
            np.testing.assert_array_equal(z[ok], 1.0)
        incoming = out["born_momenta"][:, 0] + out["born_momenta"][:, 1]
        np.testing.assert_allclose(dot(k_perp, k_perp), -1.0, rtol=1e-12)
        np.testing.assert_allclose(dot(k_perp, p_jbar) / p_jbar[:, 0], 0, atol=1e-12)
        np.testing.assert_allclose(
            dot(k_perp, incoming) / incoming[:, 0], 0, atol=1e-12
        )


def test_splitting_kinds():
    """The splittings of AP_reduced and their factors: final-state g -> g g,
    q -> q g, g -> q q~ and initial-state g -> g g, q -> q g (mother q), q ->
    g q (mother g), g -> q q~ (mother q)."""
    ca, tr, cf = 3.0, 0.5, 4.0 / 3.0
    # u u~ > e- e+ g g with (i, j) = (5, 4), (5, 0) and (4, 0)
    pdgs = [2, -2, 11, -11, 21, 21]
    colors = [3, -3, 1, 1, 8, 8]
    charges = [2 / 3, -2 / 3, -1.0, 1.0, 0.0, 0.0]
    kernels = ms.FKSSplittingKernels(
        [
            configuration(i, j, pdgs, colors, charges)
            for i, j in [(5, 4), (5, 0), (4, 0)]
        ]
    )
    assert [kernels.splitting_kind(c) for c in range(3)] == [0, 2, 2]
    assert kernels.factors(0) == [2 * ca, 0.0, -4 * ca, 0.0]
    assert kernels.factors(1) == pytest.approx([cf, 4 / 9, 0.0, 0.0])
    # g u > e- e+ u with i = 4 the final u and j = 0 the incoming g: the mother
    # is a u~ (g -> u u~)
    pdgs = [21, 2, 11, -11, 2]
    colors = [8, 3, 1, 1, 3]
    charges = [0.0, 2 / 3, -1.0, 1.0, 2 / 3]
    kernels = ms.FKSSplittingKernels([configuration(4, 0, pdgs, colors, charges)])
    assert kernels.splitting_kind(0) == 1
    assert kernels.factors(0) == pytest.approx([tr, 3 * 4 / 9, 0.0, 0.0])
    # u u~ > e- e+ u~ with i = 4 the final u~ and j = 0 the incoming u: the
    # mother is a g (q -> g q), which has azimuthal terms
    pdgs = [2, -2, 11, -11, -2]
    colors = [3, -3, 1, 1, -3]
    charges = [2 / 3, -2 / 3, -1.0, 1.0, -2 / 3]
    kernels = ms.FKSSplittingKernels([configuration(4, 0, pdgs, colors, charges)])
    assert kernels.splitting_kind(0) == 3
    assert kernels.factors(0) == pytest.approx([cf, 4 / 9, -4 * cf, -4 * 4 / 9])
    # e+ e- > g g u u~, final-state g -> u u~ with i = 5 the u~ and j = 4 the u
    pdgs = [-11, 11, 21, 21, 2, -2]
    colors = [1, 1, 8, 8, 3, -3]
    charges = [1.0, -1.0, 0.0, 0.0, 2 / 3, -2 / 3]
    kernels = ms.FKSSplittingKernels([configuration(5, 4, pdgs, colors, charges)])
    assert kernels.splitting_kind(0) == 1
    assert kernels.factors(0) == pytest.approx([tr, 4 / 3, 4 * tr, 16 / 3])


def test_invalid_configurations():
    pdgs = [2, -2, 11, -11, 21, 2]
    colors = [3, -3, 1, 1, 8, 3]
    charges = [2 / 3, -2 / 3, -1.0, 1.0, 0.0, 2 / 3]
    with pytest.raises(ValueError, match="final state"):
        # a final-state quark i with a gluon j
        ms.FKSSplittingKernels([configuration(5, 4, pdgs, colors, charges)])
    with pytest.raises(ValueError, match="final-state"):
        ms.FKSSplittingKernels([configuration(1, 0, pdgs, colors, charges)])
    with pytest.raises(ValueError, match="number of particles"):
        ms.FKSSplittingKernels(
            [
                configuration(5, 4, pdgs, colors, charges),
                configuration(4, 0, pdgs[:-1], colors[:-1], charges[:-1]),
            ]
        )
