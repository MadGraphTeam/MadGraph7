"""FKSSubtraction: the real emission and its local FKS counterterms with the
collinear remnants, as compute_real_emission and compute_soft_counter_term
and its siblings in fks_singular.f.

The reference data (see test_fks_reference.py) store, for each generated
event, what those routines take: sreal at the event and at the
counter-events, the sector functions, the prefactors with their Jacobians and
symmetry factors, the collinear remnants of sreal_deg (also in the DIS
scheme) and the parameters of the subtraction. FKSSubtraction must give
their contributions per unit Jacobian (as for FKSPrefactors) from the momenta
of the event and counter-events, the real matrix element of the Fortran and
the Born blocks of pyAmpliCol (see test_fks_limits.py), also at the events
close to the soft limit, where sreal and fks_Sij switch to the limit and
FKSSubtraction does not use the real matrix element, per NLO order with the
parts of the amp_split arrays. On events of the FKS mappings it is the
product of FKSSectorFunction, FKSPrefactors and FKSLimits with the
conditions of the counterterms and the collinear remnants, and with the
limits at the event close to them.
"""

import numpy as np
import pytest
from fks_testing import FKS_DATA, FKS_DATA_IDS, at_nlo_orders, evaluate, reference
from fks_testing import reference_born_mapping as born_mapping
from test_fks_limits import born_blocks, generated_points, soft_scale

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


KINEMATICS = [
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


def mapping_events(name, seed, r_xi_max=None):
    """The reference data of a subprocess, its Born mapping and FKS
    configurations, and the outputs of FKSKinematics on events of the FKS
    configuration mapping, with r_xi below r_xi_max where it is given, with
    their fks_index."""
    data = reference(FKS_DATA[FKS_DATA_IDS.index(name)])
    configurations = data["fks_configurations"]
    born = born_mapping(data)
    mapping = fks_configuration_mapping(born, configurations)
    count = len(configurations["configurations"])
    names = list(mapping.counter_event_types().keys())
    rng = np.random.default_rng(seed)
    batch = 400
    r = 0.05 + 0.9 * rng.random((batch, mapping.random_dim()))
    if r_xi_max is not None:
        # the random number of xi_hat, the first of the emission
        r[:, -3] = r_xi_max * rng.random(batch)
    index = rng.integers(0, count, batch).astype(np.int32)
    out = dict(zip(names, evaluate(ms.FKSKinematics(mapping), r, fks_index=index)))
    return data, born, configurations, out, index


ALPHA_S, ALPHA = 0.118, 1 / 128.0


def born_blocks_of_any_born(data, batch):
    """Born blocks of a Born with any colors, of its only Born order: one, the
    color correlations of a reference point over its Born, a spin-correlated
    Born that is not its average, alpha_s and alpha."""
    point = next(p for _, p in generated_points(data))
    correlated = np.array(point["born_blocks"]["color_correlated"])
    correlated /= point["born_blocks"]["born"]
    return [
        np.ones((batch, 1)),
        np.broadcast_to(correlated, (batch, 1, *correlated.shape)).copy(),
        np.full((batch, 1), 0.3),
        np.full(batch, ALPHA_S),
        np.full(batch, ALPHA),
    ]


def collinear_remnants(
    born, configurations, out, index, born_args, scales, f, s_c, s_sc, delta_i, dis, n_f
):
    """The collinear remnants of sreal_deg for the events of FKSKinematics,
    in numpy: per unit det of the collinear and soft-collinear
    counter-events, zero for a final-state or massive emitter, with the
    color or charge factors of the splitting type (QCD or QED) and its
    coupling, for one Born and one NLO order."""
    tr = 0.5
    fks = fks_configurations(born, configurations)
    limits = fks_limits(born, configurations)
    kernels = limits.splitting_kernels()
    count = len(fks)
    kind = np.array([kernels.splitting_kind(c) for c in range(count)])[index]
    qed = np.array([limits.qed(c) for c in range(count)])[index]
    # the color or charge factor of AP_reduced of the splitting type
    color = np.array([kernels.factors(c)[int(limits.qed(c))] for c in range(count)])
    color = color[index]
    remnant = np.array([c.initial_state() and c.masses[c.j_fks] == 0.0 for c in fks])
    ratio = np.array([c.identical_particle_ratio() for c in fks])[index]
    beam = np.array([c.j_fks if c.initial_state() else 0 for c in fks])[index]
    born_value, alpha_s, alpha_qed = born_args[0][:, 0], born_args[3], born_args[4]
    coupling = np.where(qed, alpha_qed, alpha_s)
    mu_f = np.where(beam == 0, scales[0], scales[1])
    xi_c = out["collinear_xi"]

    def s_hat(momenta):
        p = momenta[:, 0] + momenta[:, 1]
        return p[:, 0] ** 2 - (p[:, 1:] ** 2).sum(1)

    def pieces(z, s):
        w = 1 - z
        ap = np.choose(
            kind,
            [
                w * w / z + z + z * w * w,
                (z * z + w * w) * w,
                1 + z * z,
                (1 + w * w) * w / z,
            ],
        )
        ap_prime = np.choose(kind, [0 * z, -2 * z * w * w, -w * w, -z * w])
        n = born_value * ratio * 4 * np.pi * coupling / (8 * np.pi**2 * z) * color
        d_xi = n * (ap * np.log(s * delta_i / (2 * mu_f**2)) - ap_prime)
        d_lxi = 2 * n * ap
        k = np.zeros((3, len(z)))
        if dis:
            lx = np.log(z)
            pqg, pgq = z * z + w * w, 1 + z * z
            regular = -pqg * lx + 8 * z * w - 1
            plus = -1.5 - pgq * lx + w * (3 + 2 * z)
            delta = 4.5 + np.pi**2 / 3
            # the factors of xkplus, xklog and xkdelta: those of AP_reduced,
            # with n_f T_R for g -> g g and the sign of q -> g/a q
            factor = np.choose(kind, [-2 * n_f * tr + 0 * z, color, color, -color])
            k_p = factor * np.choose(kind, [w * regular, w * regular, plus, plus])
            k_l = factor * np.choose(kind, [w * pqg, w * pqg, pgq, pgq])
            k_d = factor * np.choose(
                kind, [0 * z, 0 * z, -delta + 0 * z, -delta + 0 * z]
            )
            k = -n / color * np.array([k_p, k_l, k_d])
        return d_xi, d_lxi, k[0], k[1], k[2]

    log_xi = np.log(xi_c)
    d_xi, d_lxi, k_p, k_l, _ = pieces(1 - xi_c, s_hat(out["collinear_momenta"]))
    collinear = f["f_dc"] * (d_xi + k_p + (d_lxi + k_l) * log_xi)
    one = np.ones(len(index))
    d_xi, d_lxi, k_p, k_l, k_d = pieces(one, s_hat(out["soft_collinear_momenta"]))
    f_dsc, f_pdf = f["f_dsc"], f["f_pdf"]
    soft_collinear = (
        -f_dsc[:, 0] * (d_xi + k_p + (d_lxi + k_l) * log_xi)
        - f_dsc[:, 1] * d_xi
        - f_dsc[:, 2] * d_lxi
        + f_pdf[:, 0] * k_d
        + f_pdf[:, 1] * k_p
        + f_pdf[:, 2] * k_l
    )
    with_remnant = remnant[index]
    return (
        np.where(with_remnant & (s_c > 0), collinear, 0.0)[:, None],
        np.where(with_remnant & (s_sc > 0), soft_collinear, 0.0)[:, None],
    )


def fact_scales(data, batch):
    """The factorization scales of the two beams of the dump."""
    return [np.full(batch, np.sqrt(mu_f2)) for mu_f2 in data["scales"]["mu_f2"]]


def madfks_contributions(data, rows):
    """The contributions of compute_real_emission and
    compute_soft_counter_term and its siblings with the collinear remnants
    of sreal_deg at the scales of the dump, per unit Jacobian of the event
    (with the factor xi) and of the counter-events, from the values of the
    Fortran per NLO order, shape (rows, 4, NLO orders)."""
    xi_s_cut = data["subtraction"]["xi_s_cut"]
    delta_s = data["subtraction"]["delta_s"]
    # the coefficients of log(mu_F^2 / QES^2) of the weight lines
    log_mu_f = np.log(data["scales"]["mu_f2"][0] / data["scales"]["qes2"])
    contributions = []
    for conf, point in rows:
        f = point["prefactors"]
        fx = at_nlo_orders(point["sreal_orders"], data)
        remnants = at_nlo_orders(point["collinear_remnants_orders"], data)
        s_ev, s_cnt = point["s_ij_ev"], point["s_ij_cnt"]
        xi, xi_hat, y = point["xi_ev"], point["xi_hat_ev"], point["y_ev"]
        xi_max = point["xiimax_cnt"]
        massless = conf["masses"][conf["j_fks"] - 1] == 0.0
        zero = np.zeros(fx.shape[1])
        real, soft, collinear, soft_collinear = zero, zero, zero, zero
        if f["f_r"] != 0.0 and s_ev > 0.0:
            real = fx[0] * s_ev * f["f_r"]
        if xi_hat * xi_max[SOFT] <= xi_s_cut and s_cnt[0] > 0 and xi <= xi_s_cut:
            soft = -fx[1] * s_cnt[0] * f["f_s"]
        # xi of the collinear counter-event, zero for a massive emitter
        log_xi = np.log(point["xi_cnt"][COLLINEAR]) if massless else 0.0
        if y > 1 - delta_s and massless and s_cnt[1] > 0:
            d_xi, d_lxi, d_mu_f, k_p, k_l, _ = remnants[0]
            remnant = d_xi + k_p + (d_lxi + k_l) * log_xi + d_mu_f * log_mu_f
            collinear = -fx[2] * s_cnt[1] * f["f_c"] + remnant * f["f_dc"]
        if (
            xi_hat * xi_max[COLLINEAR] < xi_s_cut
            and y > 1 - delta_s
            and massless
            and s_cnt[2] > 0
            and point["xi_cnt"][COLLINEAR] < xi_s_cut
        ):
            d_xi, d_lxi, d_mu_f, k_p, k_l, k_d = remnants[1]
            f_dsc, f_pdf = f["f_dsc"], f["f_pdf_scheme"]
            remnant = (
                -(d_xi + k_p + (d_lxi + k_l) * log_xi) * f_dsc[0]
                - (d_xi * f_dsc[1] + d_lxi * f_dsc[2])
                + k_d * f_pdf[0]
                + k_p * f_pdf[1]
                + k_l * f_pdf[2]
                - d_mu_f * f_dsc[3] * log_mu_f
            )
            soft_collinear = fx[3] * s_cnt[2] * f["f_sc"] + remnant
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
            [
                np.divide(v, n, out=np.zeros_like(v), where=v != 0.0)
                for v, n in zip(values, norms)
            ]
        )
    return np.array(contributions)


@pytest.mark.parametrize("path", FKS_DATA, ids=FKS_DATA_IDS)
def test_matches_madfks(path):
    data = reference(path)
    scheme = ms.FKSSubtraction.PdfScheme
    subtraction = fks_subtraction(
        born_mapping(data),
        data["fks_configurations"],
        **data["subtraction"],
        pdf_scheme=scheme.dis if data["pdf_scheme"] == 1 else scheme.msbar,
        n_f=data["n_f"],
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
    # the real matrix element of sreal, not used where it switches to the
    # limits (below tiny, with the soft counter-event for the soft limit)
    switched = ((xi < TINY) & (jac_cnt[:, SOFT] > 0)) | (1 - y < TINY)
    assert switched.any()
    sreal = at_nlo_orders(field("sreal_orders"), data)
    matrix_element = np.where(
        switched[:, None], np.nan, sreal[:, 0] / (xi**2 * (1 - y))[:, None]
    )
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
            *fact_scales(data, len(rows)),
            fks_index=index,
        ),
        axis=1,
    )
    theirs = madfks_contributions(data, rows)
    assert ours.shape == theirs.shape
    # all contributions occur
    assert (theirs[:, 0] != 0).any() and (theirs[:, 1] != 0).any()
    assert (theirs[:, 2] != 0).any() or not massless.any()
    # At an event of an initial-state emitter, sbornsoft takes the energy of
    # j in the eikonal factors in the frame of genps_fks.f, which differs from
    # that of the event by O(xi). The terms of the soft limit cancel for
    # massive emitters. MadFKS's dot sets invariants below 1e-6 GeV^2 to zero,
    # which drops a pair of i and a particle collinear to it from fks_Sij
    # (S = 1 where it is 1 - 6e-9 at a photon i of u u~ > e- e+ a a).
    p = field("p_ev")
    p_i = p[np.arange(len(rows)), [conf["i_fks"] - 1 for conf, _ in rows]]
    invariants = np.abs(
        p_i[:, None, 0] * p[..., 0] - (p_i[:, None, 1:] * p[..., 1:]).sum(-1)
    )
    invariants[invariants == 0.0] = np.inf
    rounded = invariants.min(1) < 1e-6
    initial = np.array([conf["j_fks"] <= 2 for conf, _ in rows])
    rtol = 1e-11 + np.where(switched, 1e-9 + initial * xi, 0) + rounded * 1e-8
    rtol = rtol[:, None]
    # the soft contribution per unit soft limit, times the scale of the limit
    fx_s = sreal[:, 1]
    per_limit = np.divide(theirs[:, 1], fx_s, out=np.zeros_like(fx_s), where=fx_s != 0)
    soft_tolerance = 1e-13 * np.abs(per_limit) * soft_scale(rows)[:, None]
    names = ["real", "soft", "collinear", "soft_collinear"]
    for k, name in enumerate(names):
        tolerance = rtol * np.abs(theirs[:, k]) + (soft_tolerance if k == 1 else 0)
        # sreal at the soft counter-event of y = 1 takes the collinear limit
        valid = (1 - y >= TINY) if name == "soft" else np.full(len(rows), True)
        deviation = np.abs(ours[valid, k] - theirs[valid, k])
        assert (deviation <= tolerance[valid]).all(), name


# processes with every kind of emitter, also of QED splittings
MAPPING_PROCESSES = [
    "pp_ttx_P0_gg_ttx",
    "epem_uuxg_P0_epem_uuxg",
    "pp_epemj_P0_ug_uemep",
    "pp_epem_qed_P0_aa_emep",
    "pp_epema_qed_P0_uux_emepa",
]


@pytest.mark.parametrize("name", MAPPING_PROCESSES)
@pytest.mark.parametrize(
    "pdf_scheme", list(ms.FKSSubtraction.PdfScheme.__members__.values())
)
def test_on_mapping_events(name, pdf_scheme):
    """On events of the FKS mappings, FKSSubtraction is the product of
    FKSSectorFunction, FKSPrefactors and FKSLimits with the signs, symmetry
    factors and conditions of the counterterms, plus the collinear remnants
    of an initial-state emitter with the factorization scale of its beam,
    finite everywhere and zero where an event or counter-event is not
    generated."""
    parameters = dict(xicut=0.4, xi_s_cut=0.3, delta_i=0.7, delta_s=0.5)
    data, born, configurations, out, index = mapping_events(name, 43)
    subtraction = fks_subtraction(
        born, configurations, **parameters, pdf_scheme=pdf_scheme, n_f=4.0
    )
    me = 1 / (out["xi"] ** 2 * (1 - out["y"]))[:, None]
    born_args = born_blocks_of_any_born(data, len(index))
    # factorization scales that differ between the beams
    scales = [np.full(len(index), 90.0), np.full(len(index), 140.0)]
    kinematics = [out[key] for key in KINEMATICS]
    real, soft, collinear, soft_collinear = evaluate(
        subtraction, *kinematics, me, *born_args, *scales, fks_index=index
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
    _, f_s, f_c, f_dc, f_sc, f_dsc, f_pdf = evaluate(
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
    )[index, None]
    s = {key: value[:, None] for key, value in s.items()}
    f_s, f_c, f_sc = f_s[:, None], f_c[:, None], f_sc[:, None]
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
    with_soft, with_collinear, with_soft_collinear = (
        ok[:, None] for ok in [with_soft, with_collinear, with_soft_collinear]
    )
    remnant_c, remnant_sc = collinear_remnants(
        born,
        configurations,
        out,
        index,
        born_args,
        scales,
        dict(f_dc=f_dc, f_dsc=f_dsc, f_pdf=f_pdf),
        s["collinear_"][:, 0],
        s["soft_collinear_"][:, 0],
        parameters["delta_i"],
        pdf_scheme == ms.FKSSubtraction.PdfScheme.dis,
        4.0,
    )
    initial = np.array([c["j_fks"] <= 2 for c in configurations["configurations"]])
    assert (remnant_c[initial[index]] != 0).any() or not initial.any()
    expected = [
        np.where(out["weight"][:, None] > 0, sym * me * s[""], 0.0),
        np.where(with_soft, -sym * (s["soft_"] * f_s * lim_s), 0.0),
        np.where(
            with_collinear, sym * (remnant_c - s["collinear_"] * f_c * lim_c), 0.0
        ),
        np.where(
            with_soft_collinear,
            sym * (s["soft_collinear_"] * f_sc * lim_sc + remnant_sc),
            0.0,
        ),
    ]
    # the remnants in another order of the operations
    for ours, theirs, rtol in zip(
        [real, soft, collinear, soft_collinear], expected, [1e-15, 1e-15, 1e-13, 1e-13]
    ):
        assert np.isfinite(ours).all()
        np.testing.assert_allclose(ours, theirs, rtol=rtol, atol=0)


@pytest.mark.parametrize("name", MAPPING_PROCESSES)
def test_limits_at_the_event(name):
    """Below tiny in xi the real emission is the soft limit at the event over
    xi^2 (1 - y), with the sector function in the direction of i at the soft
    counter-event, and below tiny in 1 - y (which the maps of y do not reach,
    so y is set here) the collinear limit at the event, with that at the
    soft-collinear counter-event where also xi <= tiny. The real matrix
    element is not used there. With tiny = 0 it is used everywhere."""
    data, born, configurations, out, index = mapping_events(name, 47, 2e-3)
    batch = len(index)
    xi = out["xi"]
    assert (xi < TINY).any() and (xi > TINY).any()
    # a y beyond the reach of the maps for every third event
    y = np.where(np.arange(batch) % 3 == 0, 1 - 4e-7, out["y"])
    omy = 1 - y
    massless = np.array(
        [c.masses[c.j_fks] == 0.0 for c in fks_configurations(born, configurations)]
    )[index]
    soft_switch = (omy >= TINY) & (xi < TINY) & (out["soft_det"] > 0)
    switched = (omy < TINY) | soft_switch
    me = 1 / (xi**2 * omy)[:, None]
    born_args = born_blocks_of_any_born(data, batch)
    kinematics = [y if key == "y" else out[key] for key in KINEMATICS]
    scales = [np.full(batch, 100.0)] * 2
    real, *_ = evaluate(
        fks_subtraction(born, configurations),
        *kinematics,
        np.where(switched[:, None], np.nan, me),
        *born_args,
        *scales,
        fks_index=index,
    )
    (plain, *_) = evaluate(
        fks_subtraction(born, configurations, tiny=0.0),
        *kinematics,
        me,
        *born_args,
        *scales,
        fks_index=index,
    )
    sector = fks_sector_function(born, configurations)
    s = {
        direction: evaluate(
            sector, out["momenta"], out[f"{direction}p_i_hat"], fks_index=index
        )[0]
        for direction in ["", "soft_", "soft_collinear_"]
    }
    lim_s, lim_c, _ = evaluate(
        fks_limits(born, configurations),
        out["momenta"],
        out["soft_p_i_hat"],
        y,
        out["momenta"],
        out["momenta"],
        *born_args,
        fks_index=index,
    )
    sym = np.array(
        [c.symmetry_factor() for c in fks_configurations(born, configurations)]
    )[index, None]
    with_soft_collinear = (out["soft_collinear_det"] > 0) & massless
    sector_at_event = np.where(
        xi <= TINY,
        np.where(
            (omy < TINY) & with_soft_collinear,
            s["soft_collinear_"],
            np.where(out["soft_det"] > 0, s["soft_"], s[""]),
        ),
        s[""],
    )
    reduced = (xi**2 * omy)[:, None]
    value = np.where(
        (omy < TINY)[:, None],
        lim_c / reduced,
        np.where(soft_switch[:, None], lim_s / reduced, me),
    )
    assert np.isfinite(real).all()
    np.testing.assert_allclose(
        real, sym * (sector_at_event[:, None] * value), rtol=1e-15
    )
    np.testing.assert_allclose(plain, sym * (s[""][:, None] * me), rtol=1e-15)
    # the limits are those of a soft gluon or photon and a massless emitter
    limits = fks_limits(born, configurations)
    soft_limit = np.array([limits.soft(c) for c in index])
    soft_switched = soft_switch & soft_limit
    assert (real[soft_switched] != 0).any() or not soft_switched.any()
    assert (real[omy < TINY] != 0).any()


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
        ms.FKSSubtraction([initial], ms.FKSCouplingOrders.qcd(2), xi_s_cut=0.0)
    with pytest.raises(ValueError, match="n_f"):
        ms.FKSSubtraction([initial], ms.FKSCouplingOrders.qcd(2), n_f=-1.0)
