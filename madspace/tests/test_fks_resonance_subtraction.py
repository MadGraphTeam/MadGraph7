"""The local FKS counterterms of the resonance-local recoil of an emission off a
final-state parton: FKSEikonals, FKSSplittingKernels, FKSPrefactors, FKSLimits
and FKSSubtraction with resonance, as fks_singular.f of MadGraph7.

The reference data of test_data/fks_resonance (see
fks_reference/make_fks_resonance_reference.py) store, at the points of each
Born configuration, the factors of eikonal_reduced, sborncol_fsr and
compute_prefactors_n1body, the scales of resonance_subtraction_scales, sreal at
the event and the counter-events and the Born and color-linked Borns of MadFKS
at the Born. Where a configuration has a resonance of momentum K, MadGraph7
takes the energies of sborncol_fsr and the
eikonal factors of a massless j in the rest frame of K, and the cuts of the
plus distributions scaled from those of the partonic center-of-mass frame. With
K, zero where there is no resonance, the classes with resonance must give the
same, and the classes without resonance must differ there. The transverse
direction of the emission is that of the rest frame of K boosted with the
momenta, as resonance_collinear_phase transports it for the phase of the
spin-correlated Born. On events of an FKSMapping with a resonance
the cuts follow from its soft counter-event and K.
"""

import numpy as np
import pytest
from fks_testing import (
    EPS,
    FKS_RESONANCE_DATA,
    at_nlo_orders,
    configuration,
    data_id,
    evaluate,
    reference,
    reference_born_mapping,
    resonance_stretch,
)
from test_fks_limits import counter_events, generated_points, soft_scale
from test_fks_resonance import EMITTER_IDS, EMITTERS, SQRT_S, z_topology
from test_fks_subtraction import check_against_madfks

import madspace as ms
from madspace.fks import (
    fks_eikonals,
    fks_limits,
    fks_prefactors,
    fks_splitting_kernels,
)

SOFT, COLLINEAR, SOFT_COLLINEAR = 2, 3, 4
TINY = 1e-6  # tiny of eikonal_reduced and sreal
IDS = [data_id(path) for path in FKS_RESONANCE_DATA]


def dot(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def resonance_momentum(conf, point):
    """K of a point, zero where its configuration has no resonance."""
    if conf["resonance"]["resonance_recoil"]:
        return point["resonance_momentum"]
    return [0.0] * 4


def rows_of(data, keep):
    """(configuration, point) of the points where keep(conf, point)."""
    return [
        (conf, point)
        for conf in data["configs"]
        for point in conf["points"]
        if keep(conf, point)
    ]


def resonant(rows):
    return np.array([conf["resonance"]["resonance_recoil"] for conf, _ in rows])


def momenta_of(rows):
    return np.array([resonance_momentum(conf, point) for conf, point in rows])


def index_of(rows):
    return np.array([conf["nfks"] - 1 for conf, _ in rows], dtype=np.int32)


def test_reference_covers_the_frames():
    """Resonant configurations of a massless and of a massive emitter, and
    configurations without a resonance."""
    kinds = set()
    for path in FKS_RESONANCE_DATA:
        for conf in reference(path)["configs"]:
            massive = conf["masses"][conf["j_fks"] - 1] > 0
            if conf["resonance"]["resonance_recoil"]:
                kinds.add("massive" if massive else "massless")
            else:
                kinds.add("none")
    assert kinds == {"massless", "massive", "none"}


@pytest.mark.parametrize("path", FKS_RESONANCE_DATA, ids=IDS)
def test_eikonals_match_eikonal_reduced(path):
    """The eikonal factors at the soft counter-event, with (p_j p_i_hat) /
    (1 - y) = (K p_j) / 2 for a massless j."""
    data = reference(path)
    born = reference_born_mapping(data)
    rows = rows_of(data, lambda conf, point: bool(point["eik_pairs"]))
    momenta = np.array([point["p_cnt"][SOFT] for _, point in rows])
    p_i_hat = np.array([point["p_i_hat_cnt"][SOFT] for _, point in rows])
    y = np.array([point["y_ev"] for _, point in rows])
    index = index_of(rows)
    eikonals = fks_eikonals(born, data["fks_configurations"], resonance=True)
    (eik,) = evaluate(eikonals, momenta, p_i_hat, y, momenta_of(rows), fks_index=index)
    (global_eik,) = evaluate(
        fks_eikonals(born, data["fks_configurations"]),
        momenta,
        p_i_hat,
        y,
        fks_index=index,
    )
    differs = False
    for k, (conf, point) in enumerate(rows):
        pairs = [(m - 1, n - 1) for m, n in point["eik_pairs"]]
        assert eikonals.pairs(index[k]) == pairs
        size = len(pairs)
        # the stretch of the boost of p_i_hat out of the rest frame of K
        stretch = resonance_stretch([point])[0]
        rtol = (1e-12 if 1 - y[k] >= TINY else 1e-5) * stretch
        np.testing.assert_allclose(eik[k, :size], point["eik_soft"], rtol=rtol)
        np.testing.assert_array_equal(eik[k, size:], 0.0)
        close = np.allclose(global_eik[k, :size], point["eik_soft"], rtol=rtol)
        massless = conf["masses"][conf["j_fks"] - 1] == 0
        if conf["resonance"]["resonance_recoil"] and massless:
            differs |= not close
        else:
            # the frame enters only the pairs with a massless j
            assert close
    assert differs or not any(
        conf["resonance"]["resonance_recoil"] and conf["masses"][conf["j_fks"] - 1] == 0
        for conf in data["configs"]
    )


@pytest.mark.parametrize("path", FKS_RESONANCE_DATA, ids=IDS)
def test_splitting_kernels_match_sborncol_fsr(path):
    """z and the kernels at the collinear and soft-collinear counter-events, with
    the energies E = (K p) / M_K and t = z M_K^2 / 4."""
    data = reference(path)
    rows = []
    for conf in data["configs"]:
        for point in conf["points"]:
            for k, position in enumerate([COLLINEAR, SOFT_COLLINEAR]):
                if point["coll_cnt"][k][0] != -1.0:
                    rows.append((conf, point, position, point["coll_cnt"][k]))
    if not rows:
        # massive emitters only
        assert all(c["masses"][c["j_fks"] - 1] > 0 for c in data["configs"])
        return
    kernels = fks_splitting_kernels(
        reference_born_mapping(data), data["fks_configurations"], resonance=True
    )
    coll = np.array([row[3] for row in rows])
    z, ap, q_terms, _ = evaluate(
        kernels,
        np.array([point["p_cnt"][position] for _, point, position, _ in rows]),
        np.array([point["p_i_hat_ev"] for _, point, _, _ in rows]),
        np.array([point["p_born"] for _, point, _, _ in rows]),
        momenta_of([row[:2] for row in rows]),
        fks_index=index_of([row[:2] for row in rows]),
    )
    np.testing.assert_allclose(z, coll[:, 0], rtol=0, atol=1e-14)
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
    # z at the counter-events is the same in every frame, t = z s / 4 is not
    z_global, ap_global, *_ = evaluate(
        fks_splitting_kernels(reference_born_mapping(data), data["fks_configurations"]),
        np.array([point["p_cnt"][position] for _, point, position, _ in rows]),
        np.array([point["p_i_hat_ev"] for _, point, _, _ in rows]),
        np.array([point["p_born"] for _, point, _, _ in rows]),
        fks_index=index_of([row[:2] for row in rows]),
    )
    np.testing.assert_allclose(z_global, coll[:, 0], rtol=0, atol=1e-14)
    flags = resonant([row[:2] for row in rows])
    np.testing.assert_allclose(ap_global[~flags], ap[~flags], rtol=1e-14)
    assert (np.abs(ap_global[flags] / ap[flags] - 1) > 1e-3).all()


def collinear_phase(k_perp, emitter):
    """The phase of <ij>/[ij] in the collinear limit from the transverse
    direction and the Born emitter, as resonance_collinear_phase
    takes it from the transported vector: k_perp rotated into the frame of the
    emitter (getangles, trp_rotate_invar), -(e^(i phi) (r_1 + i r_2))^2 /
    (r_1^2 + r_2^2) with phi the azimuth of the emitter."""
    n = emitter[:, 1:] / np.linalg.norm(emitter[:, 1:], axis=1)[:, None]
    cth = n[:, 2]
    sth = np.sqrt(np.maximum(1 - cth**2, 0.0))
    phi = np.arctan2(n[:, 1], n[:, 0])
    cph, sph = np.cos(phi), np.sin(phi)
    q1, q2, q3 = k_perp[:, 1], k_perp[:, 2], k_perp[:, 3]
    r1 = q1 * cph * cth + q2 * sph * cth - q3 * sth
    r2 = -q1 * sph + q2 * cph
    z = (cph + 1j * sph) * (r1 + 1j * r2)
    return -z * z / (r1**2 + r2**2)


@pytest.mark.parametrize("path", FKS_RESONANCE_DATA, ids=IDS)
def test_transverse_vector(path):
    """k_perp is orthogonal to the frame (K or the incoming momenta) and to the
    Born emitter, gives the phase xij_aor of the spin-correlated Born at a
    massless emitter and is xij_kperp up to its sign."""
    data = reference(path)
    rows = rows_of(data, lambda conf, point: point["s_ij_ev"] != -1.0)
    kernels = fks_splitting_kernels(
        reference_born_mapping(data), data["fks_configurations"], resonance=True
    )
    p_born = np.array([point["p_born"] for _, point in rows])
    k = momenta_of(rows)
    *_, k_perp = evaluate(
        kernels,
        np.array([point["p_ev"] for _, point in rows]),
        np.array([point["p_i_hat_ev"] for _, point in rows]),
        p_born,
        k,
        fks_index=index_of(rows),
    )
    flags = resonant(rows)
    born_j = np.array(
        [conf["j_fks"] - 1 - (conf["j_fks"] > conf["i_fks"]) for conf, _ in rows]
    )
    emitter = p_born[np.arange(len(rows)), born_j]
    frame = np.where(flags[:, None], k, p_born[:, 0] + p_born[:, 1])
    stretch = resonance_stretch([point for _, point in rows])
    np.testing.assert_allclose(dot(k_perp, k_perp), -1.0, rtol=1e-12)
    np.testing.assert_array_less(
        np.abs(dot(k_perp, emitter)) / emitter[:, 0], 1e-13 * stretch
    )
    np.testing.assert_array_less(
        np.abs(dot(k_perp, frame)) / frame[:, 0], 1e-13 * stretch
    )
    # MadGraph7 sets xij_kperp and xij_aor for a massless emitter only
    massless = np.array([conf["masses"][conf["j_fks"] - 1] == 0 for conf, _ in rows])
    kperp = np.array([point["kperp"] for _, point in rows])
    np.testing.assert_array_equal(kperp[~massless], 0.0)
    if not massless.any():
        return
    # to the precision of the part of p_i_hat transverse to the emitter
    p_i = np.array([point["p_i_hat_ev"] for _, point in rows])[:, 1:]
    axis = emitter[:, 1:]
    sin_theta = np.linalg.norm(np.cross(p_i, axis), axis=1) / (
        np.linalg.norm(p_i, axis=1) * np.linalg.norm(axis, axis=1)
    )
    tolerance = 20 * EPS * stretch**2 / np.maximum(sin_theta, EPS)
    aor = np.array([point["xij_aor"] for _, point in rows])
    phase = collinear_phase(k_perp, emitter)
    np.testing.assert_array_less(
        np.abs(phase - (aor[:, 0] + 1j * aor[:, 1]))[massless], tolerance[massless]
    )
    sign = np.sign((k_perp * kperp).sum(1))[:, None]
    deviation = np.abs(sign * k_perp - kperp).max(1)
    np.testing.assert_array_less(deviation[massless], tolerance[massless])


def resonance_scales(rows):
    """soft, collinear and angular scales of resonance_subtraction_scales at the
    soft counter-event of the points, one where there is no resonance."""
    p_s = np.array([point["p_cnt"][SOFT] for _, point in rows])
    p_i_hat = np.array([point["p_i_hat_cnt"][SOFT] for _, point in rows])
    j = np.array([conf["j_fks"] - 1 for conf, _ in rows])
    p_j = p_s[np.arange(len(rows)), j]
    total = p_s[:, 0] + p_s[:, 1]
    k = momenta_of(rows)
    flags = resonant(rows)
    q = np.where(flags[:, None], k, total)
    ratio = dot(total, total) / dot(q, q)
    soft = ratio * dot(p_i_hat, q) / dot(p_i_hat, total)
    collinear = ratio * dot(p_j, q) / dot(p_j, total)
    return soft, collinear, ratio / collinear**2


@pytest.mark.parametrize("path", FKS_RESONANCE_DATA, ids=IDS)
def test_scales_of_resonance_subtraction_scales(path):
    """The ratios a(p_i_hat) and a(p_j) of xi in the rest frames of K and of the
    incoming momenta at the soft counter-event, and s / (K^2 a(p_j)^2) for
    delta."""
    data = reference(path)
    rows = rows_of(
        data,
        lambda conf, point: "prefactors" in point and not point["no_counter_events"],
    )
    ours = resonance_scales(rows)
    for key, value in zip(["soft", "collinear", "angular"], ours):
        theirs = np.array([point["resonance_scales"][key] for _, point in rows])
        np.testing.assert_allclose(value, theirs, rtol=1e-12, err_msg=key)
    # the scales move the cuts where there is a resonance
    flags = resonant(rows)
    assert (np.abs(ours[0][flags] - 1) > 1e-3).all()
    np.testing.assert_array_equal(ours[0][~flags], 1.0)


@pytest.mark.parametrize("path", FKS_RESONANCE_DATA, ids=IDS)
def test_prefactors_match_compute_prefactors_n1body(path):
    """The prefactors with the cuts xicut a(p_i_hat) in the soft counterterm,
    xicut a(p_j) in the soft-collinear ones and delta s / (K^2 a(p_j)^2) in the
    collinear ones, divided by the Jacobians, symmetry factors and xinorm of
    the Fortran (as test_fks_prefactors.py)."""
    data = reference(path)
    prefactors = fks_prefactors(
        reference_born_mapping(data),
        data["fks_configurations"],
        **data["subtraction"],
        resonance=True,
    )
    rows = rows_of(
        data,
        lambda conf, point: "prefactors" in point and not point["no_counter_events"],
    )
    points = [point for _, point in rows]

    def field(key, position=None):
        values = np.array([p[key] for p in points])
        return values if position is None else values[:, position]

    f_r, f_s, f_c, f_dc, f_sc, f_dsc, f_pdf = evaluate(
        prefactors,
        field("xi_ev"),
        field("y_ev"),
        field("xiimax_ev"),
        field("xi_cnt", COLLINEAR),
        field("xiimax_cnt", COLLINEAR),
        field("p_cnt", COLLINEAR),
        field("p_cnt", SOFT_COLLINEAR),
        field("p_cnt", SOFT),
        field("p_i_hat_cnt", SOFT),
        momenta_of(rows),
        fks_index=index_of(rows),
    )
    fortran = {
        key: np.array([p["prefactors"][key] for p in points])
        for key in ["f_r", "f_s", "f_c", "f_dc", "f_sc", "f_dsc", "f_pdf_scheme"]
    }
    sym, sym_deg = field("fks_symmetry_factor"), field("fks_symmetry_factor_deg")
    jac, jac_cnt = field("jac"), field("jac_cnt")
    xinorm, xinorm_c = field("xinorm_ev"), field("xinorm_cnt", COLLINEAR)
    massless = np.array([conf["masses"][conf["j_fks"] - 1] == 0 for conf, _ in rows])
    norm_c = xinorm_c * jac_cnt[:, COLLINEAR]
    norm_sc = xinorm_c * jac_cnt[:, SOFT_COLLINEAR]
    soft = jac_cnt[:, SOFT] > 0
    collinear = massless & (jac_cnt[:, COLLINEAR] > 0)
    soft_collinear = massless & (jac_cnt[:, SOFT_COLLINEAR] > 0)
    cases = [
        ("f_r", f_r, fortran["f_r"], xinorm * jac * sym, jac > 0),
        ("f_s", f_s, fortran["f_s"], xinorm * jac_cnt[:, SOFT] * sym, soft),
        ("f_c", f_c, fortran["f_c"], norm_c * sym, collinear),
        ("f_dc", f_dc, fortran["f_dc"], norm_c * sym_deg, collinear),
        ("f_sc", f_sc, fortran["f_sc"], norm_sc * sym_deg, soft_collinear),
    ]
    cases += [
        (
            f"f_dsc({k + 1})",
            f_dsc[:, k],
            fortran["f_dsc"][:, k],
            norm_sc * sym_deg,
            soft_collinear,
        )
        for k in range(4)
    ]
    cases += [
        (
            f"f_pdfsch({name})",
            f_pdf[:, k],
            fortran["f_pdf_scheme"][:, k],
            norm_sc * sym_deg,
            soft_collinear,
        )
        for k, name in enumerate(["d", "p", "l"])
    ]
    for key, ours, theirs, norm, valid in cases:
        valid = valid & (norm != 0)
        assert valid.any() or not massless.any(), key
        np.testing.assert_allclose(
            ours[valid], theirs[valid] / norm[valid], rtol=1e-12, err_msg=key
        )
    for ours in [f_c, f_dc, f_sc, *f_dsc.T, *f_pdf.T]:
        np.testing.assert_array_equal(ours[~massless], 0.0)
    # those of the partonic center-of-mass frame differ where there is a
    # resonance
    flags = resonant(rows) & soft
    if flags.any():
        _, f_s_global, *_ = evaluate(
            fks_prefactors(
                reference_born_mapping(data),
                data["fks_configurations"],
                **data["subtraction"],
            ),
            field("xi_ev"),
            field("y_ev"),
            field("xiimax_ev"),
            field("xi_cnt", COLLINEAR),
            field("xiimax_cnt", COLLINEAR),
            field("p_cnt", COLLINEAR),
            field("p_cnt", SOFT_COLLINEAR),
            fks_index=index_of(rows),
        )
        assert (np.abs(f_s_global[flags] / f_s[flags] - 1) > 1e-6).all()


def born_blocks(data, rows):
    """born, color_correlated and spin_correlated of the Born of MadFKS at the
    points (the emitters are quarks, so the spin-correlated Born is zero),
    alpha_s, alpha and fks_index."""
    configurations = data["fks_configurations"]
    count = len(configurations["orders"]["born"])
    n = len(configurations["born_pdgs"])
    born = np.zeros((len(rows), count))
    color = np.zeros((len(rows), count, n, n))
    for k, (_, point) in enumerate(rows):
        blocks = point["born_blocks"]
        for o, block in enumerate(blocks.get("orders", [blocks])):
            born[k, o] = block["born"]
            color[k, o] = block["color_correlated"]
    return (
        born,
        color,
        np.zeros_like(born),
        np.full(len(rows), data["born_blocks"]["alpha_s"]),
        np.full(len(rows), data["gal"] ** 2 / (4 * np.pi)),
    )


def soft_rounding(rows):
    """The growth of the rounding of the terms of the soft limit, which cancel
    for a massive emitter: (E_j / m_j)^2 at the soft counter-event for a
    massive j, from (p_j p_i_hat), times the stretch of the boost out of the
    rest frame of the resonance."""
    factor = resonance_stretch([point for _, point in rows])
    for k, (conf, point) in enumerate(rows):
        j = conf["j_fks"] - 1
        m_j = conf["masses"][j]
        if m_j > 0:
            factor[k] *= (point["p_cnt"][SOFT][j][0] / m_j) ** 2
    return factor


@pytest.mark.parametrize("path", FKS_RESONANCE_DATA, ids=IDS)
def test_limits_match_sreal(path):
    """FKSLimits with K against sreal at the counter-events, also at the
    points close to the limits."""
    data = reference(path)
    limits = fks_limits(
        reference_born_mapping(data), data["fks_configurations"], resonance=True
    )
    assert limits.resonance()
    rows = generated_points(data)
    index = index_of(rows)
    soft, collinear, soft_collinear = evaluate(
        limits,
        *counter_events(rows),
        *born_blocks(data, rows),
        momenta_of(rows),
        fks_index=index,
    )
    sreal = np.array([point["sreal"] for _, point in rows])
    theirs = at_nlo_orders([point["sreal_orders"] for _, point in rows], data)
    y = np.array([point["y_ev"] for _, point in rows])
    massive = np.array([conf["masses"][conf["j_fks"] - 1] > 0 for conf, _ in rows])
    stretch = resonance_stretch([point for _, point in rows])[:, None]
    scale = soft_scale(rows) * soft_rounding(rows)
    cases = [
        ("soft", soft, theirs[:, 1], (sreal[:, 1] != -1.0) & (1 - y >= TINY)),
        ("collinear", collinear, theirs[:, 2], sreal[:, 2] != -1.0),
        ("soft_collinear", soft_collinear, theirs[:, 3], sreal[:, 3] != -1.0),
    ]
    for name, ours, expected, valid in cases:
        assert valid.any() or massive.all(), name
        tolerance = 1e-12 * stretch * np.abs(expected) + 2 * EPS * scale[:, None]
        deviation = np.abs(ours - expected)
        assert (deviation[valid] <= tolerance[valid]).all(), name
    np.testing.assert_array_equal(collinear[massive], 0.0)
    np.testing.assert_array_equal(soft_collinear[massive], 0.0)


@pytest.mark.parametrize("path", FKS_RESONANCE_DATA, ids=IDS)
def test_subtraction_matches_madfks(path):
    """FKSSubtraction with K against the contributions of compute_real_emission
    and compute_soft_counter_term and its siblings (as
    test_fks_subtraction.py)."""
    check_against_madfks(reference(path), resonance=True, rounding=soft_rounding)


@pytest.mark.parametrize("masses,j_fks", EMITTERS, ids=EMITTER_IDS)
def test_prefactors_on_mapping_events(masses, j_fks):
    """On events of an FKSMapping with the Z as resonance (e+ e- > e+ e- q q~
    g, lab frame), the cuts follow from the soft counter-event and K of the
    mapping: f_s and f_c with xicut a(p_i_hat) and delta s / (K^2 a(p_j)^2),
    and a(p_i_hat) is the ratio of xi in the rest frames of K and of the
    incoming momenta at the soft counter-event."""
    xicut, xi_s_cut, delta_o, delta_s = 0.3, 0.5, 0.7, 0.9
    pdgs = [-11, 11, -11, 11, 2, -2, 21]
    colors = [1, 1, 1, 1, 3, -3, 8]
    real_masses = [0.0, 0.0] + list(masses) + [0.0]
    config = configuration(6, j_fks, pdgs, colors, masses=real_masses)
    born = ms.PhaseSpaceMapping(z_topology(masses), SQRT_S, leptonic=True)
    fks = ms.FKSMapping(born, 6, j_fks, resonance=[4, 5, 6])
    names = list(fks.counter_event_types().keys())
    batch = 1000
    r = 0.02 + 0.96 * np.random.default_rng(21).random((batch, fks.random_dim()))
    out = dict(zip(names, evaluate(ms.FKSKinematics(fks), r)))
    ok = out["weight"] > 0
    assert ok.mean() > 0.5
    prefactors = ms.FKSPrefactors(
        [config],
        xicut=xicut,
        xi_s_cut=xi_s_cut,
        delta_o=delta_o,
        delta_s=delta_s,
        resonance=True,
    )
    f_r, f_s, f_c, *_ = evaluate(
        prefactors,
        *[
            out[name]
            for name in (
                "xi",
                "y",
                "xi_max",
                "collinear_xi",
                "collinear_xi_max",
                "collinear_momenta",
                "soft_collinear_momenta",
                "soft_momenta",
                "soft_p_i_hat",
                "resonance_momentum",
            )
        ],
        fks_index=np.zeros(batch, dtype=np.int32),
    )
    k = out["resonance_momentum"]
    p_s, p_i_hat = out["soft_momenta"], out["soft_p_i_hat"]
    total = p_s[:, 0] + p_s[:, 1]
    s, mass2 = dot(total, total), dot(k, k)
    soft_scale = s * dot(p_i_hat, k) / (mass2 * dot(p_i_hat, total))
    coll_scale = s * dot(p_s[:, j_fks], k) / (mass2 * dot(p_s[:, j_fks], total))
    # xi of the soft momentum in the rest frame of K over that of the
    # incoming momenta
    xi_k, xi_p = 2 * dot(p_i_hat, k) / mass2, 2 * dot(p_i_hat, total) / s
    np.testing.assert_allclose(soft_scale, xi_k / xi_p, rtol=1e-13)
    assert (np.abs(soft_scale[ok] - 1) > 1e-3).mean() > 0.9
    xi, omy = out["xi"], 1 - out["y"]
    xi_lim = np.minimum(out["xi_max"], xi_s_cut)
    expected_s = (1 / xi + np.log(xicut * soft_scale / xi_lim) / xi_lim) / omy
    np.testing.assert_allclose(f_r[ok], 1 / (xi * omy)[ok], rtol=1e-14)
    np.testing.assert_allclose(f_s[ok], expected_s[ok], rtol=1e-12)
    if masses[j_fks - 2] > 0:
        np.testing.assert_array_equal(f_c, 0.0)
        return
    angular = s / (mass2 * coll_scale**2)
    in_y = 1 / omy + np.log(delta_o * angular / delta_s) / delta_s
    np.testing.assert_allclose(f_c[ok], (in_y / out["collinear_xi"])[ok], rtol=1e-12)
