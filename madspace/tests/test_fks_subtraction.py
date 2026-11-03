"""FKSSubtraction: the real emission and its local FKS counterterms, as
compute_real_emission and compute_soft_counter_term and its siblings in
fks_singular.f, without the collinear remnants.

The reference data (see test_fks_reference.py) store, for each generated
event, what those routines take: sreal at the event and at the
counter-events, the sector functions, the prefactors with their Jacobians and
symmetry factors, and the parameters of the subtraction. FKSSubtraction must
give their contributions per unit Jacobian (as for FKSPrefactors) from the
momenta of the event and counter-events, the real matrix element of the
Fortran and the Born blocks of pyAmpliCol (see test_fks_limits.py). On events
of the FKS mappings it is the product of FKSSectorFunction, FKSPrefactors and
FKSLimits with the conditions of the counterterms.
"""

import numpy as np
import pytest
from fks_testing import FKS_DATA, FKS_DATA_IDS, evaluate, reference
from fks_testing import reference_born_mapping as born_mapping
from test_fks_limits import born_blocks, generated_points

import madspace as ms
from madspace.fks import (
    fks_configuration_mapping,
    fks_configurations,
    fks_limits,
    fks_prefactors,
    fks_sector_function,
    fks_subtraction,
)

SOFT, COLLINEAR, SOFT_COLLINEAR = 2, 3, 4
TINY = 1e-6  # tiny of sreal and fks_Sij


def madfks_contributions(data, rows):
    """The contributions of compute_real_emission and
    compute_soft_counter_term and its siblings without the collinear
    remnants, per unit Jacobian of the event (with the factor xi) and of the
    counter-events, from the values of the Fortran."""
    xi_s_cut = data["subtraction"]["xi_s_cut"]
    delta_s = data["subtraction"]["delta_s"]
    contributions = []
    for conf, point in rows:
        f = point["prefactors"]
        fx, s_ev, s_cnt = point["sreal"], point["s_ij_ev"], point["s_ij_cnt"]
        xi, xi_hat, y = point["xi_ev"], point["xi_hat_ev"], point["y_ev"]
        xi_max = point["xiimax_cnt"]
        massless = conf["masses"][conf["j_fks"] - 1] == 0.0
        real = soft = collinear = soft_collinear = 0.0
        if f["f_r"] != 0.0 and s_ev > 0.0:
            real = fx[0] * s_ev * f["f_r"]
        if xi_hat * xi_max[SOFT] <= xi_s_cut and s_cnt[0] > 0 and xi <= xi_s_cut:
            soft = -fx[1] * s_cnt[0] * f["f_s"]
        if y > 1 - delta_s and massless and s_cnt[1] > 0:
            collinear = -fx[2] * s_cnt[1] * f["f_c"]
        if (
            xi_hat * xi_max[COLLINEAR] < xi_s_cut
            and y > 1 - delta_s
            and massless
            and s_cnt[2] > 0
            and point["xi_cnt"][COLLINEAR] < xi_s_cut
        ):
            soft_collinear = fx[3] * s_cnt[2] * f["f_sc"]
        jac, jac_cnt = point["jac"], point["jac_cnt"]
        xinorm, xinorm_c = point["xinorm_ev"], point["xinorm_cnt"][COLLINEAR]
        norms = [
            xinorm * jac * xi,
            xinorm * jac_cnt[SOFT],
            xinorm_c * jac_cnt[COLLINEAR],
            xinorm_c * jac_cnt[SOFT_COLLINEAR],
        ]
        values = [real, soft, collinear, soft_collinear]
        contributions.append(
            [v / n if v != 0.0 else 0.0 for v, n in zip(values, norms)]
        )
    return np.array(contributions)


@pytest.mark.parametrize("path", FKS_DATA, ids=FKS_DATA_IDS)
def test_matches_madfks(path):
    data = reference(path)
    subtraction = fks_subtraction(
        born_mapping(data), data["fks_configurations"], **data["subtraction"]
    )
    assert subtraction.configuration_count() == len(data["configs"])
    rows = [
        (conf, point)
        for conf, point in generated_points(data)
        if not point["no_counter_events"]
    ]

    def field(key, position=None):
        values = np.array([point[key] for _, point in rows])
        return values if position is None else values[:, position]

    massless = np.array(
        [conf["masses"][conf["j_fks"] - 1] == 0.0 for conf, _ in rows], dtype=float
    )
    jac_cnt = field("jac_cnt")
    xi, y = field("xi_ev"), field("y_ev")
    # the real matrix element of sreal, which near the limits is their value
    matrix_element = field("sreal")[:, 0] / (xi**2 * (1 - y))
    *blocks, index = born_blocks(data, rows)
    ours = np.stack(
        evaluate(
            subtraction,
            field("p_ev"),
            (field("jac") > 0).astype(float),
            xi,
            field("xi_hat_ev"),
            y,
            field("xiimax_ev"),
            field("p_i_hat_ev"),
            field("p_cnt", SOFT),
            np.maximum(jac_cnt[:, SOFT], 0.0),
            field("p_i_hat_cnt", SOFT),
            field("xiimax_cnt", SOFT),
            field("p_cnt", COLLINEAR),
            np.maximum(jac_cnt[:, COLLINEAR], 0.0) * massless,
            field("p_i_hat_cnt", COLLINEAR),
            field("xi_cnt", COLLINEAR),
            field("xiimax_cnt", COLLINEAR),
            field("p_cnt", SOFT_COLLINEAR),
            np.maximum(jac_cnt[:, SOFT_COLLINEAR], 0.0) * massless,
            field("p_i_hat_cnt", SOFT_COLLINEAR),
            matrix_element,
            *blocks,
            fks_index=index,
        ),
        axis=1,
    )
    theirs = madfks_contributions(data, rows)
    # all contributions occur
    assert (theirs[:, 0] != 0).any() and (theirs[:, 1] != 0).any()
    assert (theirs[:, 2] != 0).any() or not massless.any()
    # below xi = 1e-6 fks_Sij takes the direction of i from the soft
    # counter-event, and sreal at the soft counter-event of y = 1 its
    # collinear limit
    rtol = np.where(xi < TINY, 1e-5, 1e-11)
    names = ["real", "soft", "collinear", "soft_collinear"]
    for k, name in enumerate(names):
        valid = (1 - y >= TINY) if name == "soft" else np.full(len(rows), True)
        deviation = np.abs(ours[valid, k] - theirs[valid, k])
        assert (deviation <= rtol[valid] * np.abs(theirs[valid, k])).all(), name


@pytest.mark.parametrize(
    "name", ["pp_ttx_P0_gg_ttx", "epem_uuxg_P0_epem_uuxg", "pp_epemj_P0_ug_uemep"]
)
def test_on_mapping_events(name):
    """On events of the FKS mappings, FKSSubtraction is the product of
    FKSSectorFunction, FKSPrefactors and FKSLimits with the signs, symmetry
    factors and conditions of the counterterms, finite everywhere and zero
    where an event or counter-event is not generated."""
    data = reference(FKS_DATA[FKS_DATA_IDS.index(name)])
    configurations = data["fks_configurations"]
    born = born_mapping(data)
    parameters = dict(xicut=0.4, xi_s_cut=0.3, delta_s=0.5)
    subtraction = fks_subtraction(born, configurations, **parameters)
    mapping = fks_configuration_mapping(born, configurations)
    count = len(configurations["configurations"])
    names = list(mapping.counter_event_types().keys())
    rng = np.random.default_rng(43)
    batch = 400
    r = 0.05 + 0.9 * rng.random((batch, mapping.random_dim()))
    index = rng.integers(0, count, batch).astype(np.int32)
    out = dict(zip(names, evaluate(ms.FKSKinematics(mapping), r, fks_index=index)))
    # Born blocks of a Born with any colors: one, the color correlations of a
    # reference point over its Born and a spin-correlated Born that is not
    # its average
    point = next(p for _, p in generated_points(data))
    correlated = np.array(point["born_blocks"]["color_correlated"])
    correlated /= point["born_blocks"]["born"]
    me = 1 / (out["xi"] ** 2 * (1 - out["y"]))
    born_args = [
        np.ones(batch),
        np.broadcast_to(correlated, (batch, *correlated.shape)).copy(),
        np.full(batch, 0.3),
        np.full(batch, 0.118),
    ]
    kinematics = [
        out[key]
        for key in [
            "momenta",
            "weight",
            "xi",
            "xi_hat",
            "y",
            "xi_max",
            "p_i_hat",
            "soft_momenta",
            "soft_det",
            "soft_p_i_hat",
            "soft_xi_max",
            "collinear_momenta",
            "collinear_det",
            "collinear_p_i_hat",
            "collinear_xi",
            "collinear_xi_max",
            "soft_collinear_momenta",
            "soft_collinear_det",
            "soft_collinear_p_i_hat",
        ]
    ]
    real, soft, collinear, soft_collinear = evaluate(
        subtraction, *kinematics, me, *born_args, fks_index=index
    )
    # the products of the pieces
    sector = fks_sector_function(born, configurations)
    s = {
        prefix: evaluate(
            sector, out[f"{prefix}momenta"], out[f"{prefix}p_i_hat"], fks_index=index
        )[0]
        for prefix in ["", "soft_", "collinear_", "soft_collinear_"]
    }
    lim_s, lim_c, lim_sc = evaluate(
        fks_limits(born, configurations),
        out["soft_momenta"],
        out["soft_p_i_hat"],
        out["y"],
        out["collinear_momenta"],
        out["soft_collinear_momenta"],
        *born_args,
        fks_index=index,
    )
    _, f_s, f_c, _, f_sc, _ = evaluate(
        fks_prefactors(born, configurations, **parameters),
        out["xi"],
        out["y"],
        out["xi_max"],
        out["collinear_xi"],
        out["collinear_xi_max"],
        out["collinear_momenta"],
        out["soft_collinear_momenta"],
        fks_index=index,
    )
    sym = np.array(
        [c.symmetry_factor() for c in fks_configurations(born, configurations)]
    )[index]
    xi_s_cut, delta_s = parameters["xi_s_cut"], parameters["delta_s"]
    in_y = out["y"] > 1 - delta_s
    xi_hat = out["xi_hat"]
    with_soft = (
        (out["soft_det"] > 0)
        & (xi_hat * out["soft_xi_max"] <= xi_s_cut)
        & (out["xi"] <= xi_s_cut)
    )
    with_collinear = (out["collinear_det"] > 0) & in_y
    with_soft_collinear = (
        (out["soft_collinear_det"] > 0)
        & in_y
        & (xi_hat * out["collinear_xi_max"] < xi_s_cut)
        & (out["collinear_xi"] < xi_s_cut)
    )
    for ok in [with_soft, with_collinear, with_soft_collinear]:
        assert ok.any() and not ok.all()
    expected = [
        np.where(out["weight"] > 0, sym * me * s[""], 0.0),
        np.where(with_soft, -sym * (s["soft_"] * f_s * lim_s), 0.0),
        np.where(with_collinear, -sym * (s["collinear_"] * f_c * lim_c), 0.0),
        np.where(
            with_soft_collinear, sym * (s["soft_collinear_"] * f_sc * lim_sc), 0.0
        ),
    ]
    for ours, theirs in zip([real, soft, collinear, soft_collinear], expected):
        assert np.isfinite(ours).all()
        np.testing.assert_allclose(ours, theirs, rtol=1e-15, atol=0)


def test_symmetry_factors_and_parameters():
    """The symmetry factor counts the FKS pairs of the real process that a
    configuration stands for, and the parameters must be positive."""
    # e+ e- > u u~ g g with (i, j) = (5, 2), (5, 4); e+ e- > u u~ u u~ with
    # (i, j) = (5, 4); g g > t t~ g with (i, j) = (4, 0)
    gluons = ms.FKSConfiguration(
        i_fks=5,
        j_fks=2,
        pdgs=[-11, 11, 2, -2, 21, 21],
        colors=[1, 1, 3, -3, 8, 8],
        charges=[1.0, -1.0, 2 / 3, -2 / 3, 0.0, 0.0],
        masses=[0.0] * 6,
        fks_j_from_i=[[], [], [], [], [], [2, 3, 4]],
    )
    assert gluons.symmetry_factor() == 2.0
    assert gluons.identical_particle_ratio() == 0.5
    assert gluons.born_pdgs() == [-11, 11, 2, -2, 21]
    gluons.j_fks = 4
    assert gluons.symmetry_factor() == 2.0
    quarks = ms.FKSConfiguration(
        i_fks=5,
        j_fks=4,
        pdgs=[-11, 11, 2, -2, 2, -2],
        colors=[1, 1, 3, -3, 3, -3],
        charges=[1.0, -1.0, 2 / 3, -2 / 3, 2 / 3, -2 / 3],
        masses=[0.0] * 6,
        fks_j_from_i=[[], [], [], [], [], [2, 4]],
    )
    assert quarks.symmetry_factor() == 4.0
    assert quarks.identical_particle_ratio() == 0.25
    assert quarks.born_pdgs() == [-11, 11, 2, -2, 21]
    initial = ms.FKSConfiguration(
        i_fks=4,
        j_fks=0,
        pdgs=[21, 21, 6, -6, 21],
        colors=[8, 8, 3, -3, 8],
        charges=[0.0, 0.0, 2 / 3, -2 / 3, 0.0],
        masses=[0.0, 0.0, 173.0, 173.0, 0.0],
        fks_j_from_i=[[], [], [], [], [0, 1, 2, 3]],
    )
    assert initial.symmetry_factor() == 1.0
    assert initial.identical_particle_ratio() == 1.0
    with pytest.raises(ValueError, match="positive"):
        ms.FKSSubtraction([initial], xi_s_cut=0.0)
