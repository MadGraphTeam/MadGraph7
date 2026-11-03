"""FKSPrefactors: the prefactors of the real emission and its FKS
counterterms, as compute_prefactors_n1body in fks_singular.f.

The reference data (see test_fks_reference.py) store, for each generated event,
f_r, f_s, f_c, f_dc, f_sc and f_dsc of compute_prefactors_n1body with unit vegas
weight, with the Jacobians, the symmetry factors and the width of the range of
xi (xinorm) they hold, and the parameters of the subtraction, at their defaults
for most processes and away from them for pp_epemj_params. Divided by those,
the Fortran prefactors are those of madspace, which leaves the width and the
Jacobians to the dets of FKSKinematics and the symmetry factors to the
integrand. On events of the FKS mappings the collinear counter-event shares
xi with the event for a final-state emitter and xi_hat for an initial-state
one.
"""

import numpy as np
import pytest
from fks_testing import FKS_DATA, FKS_DATA_IDS, evaluate, reference
from fks_testing import reference_born_mapping as born_mapping

import madspace as ms
from madspace.fks import fks_configuration_mapping, fks_prefactors

SOFT, COLLINEAR, SOFT_COLLINEAR = 2, 3, 4


def test_reference_parameters():
    """The prefactors are tested at the default parameters and at others."""
    parameters = {tuple(reference(path)["subtraction"].values()) for path in FKS_DATA}
    assert len(parameters) == 2


@pytest.mark.parametrize("path", FKS_DATA, ids=FKS_DATA_IDS)
def test_matches_compute_prefactors_n1body(path):
    data = reference(path)
    configurations = data["fks_configurations"]
    prefactors = fks_prefactors(
        born_mapping(data), configurations, **data["subtraction"]
    )
    count = len(configurations["configurations"])
    assert prefactors.configuration_count() == count
    points, index = [], []
    for conf in data["configs"]:
        for point in conf["points"]:
            if "prefactors" in point and not point["no_counter_events"]:
                points.append(point)
                index.append(conf["nfks"] - 1)

    def field(key, position=None):
        values = np.array([p[key] for p in points])
        return values if position is None else values[:, position]

    f_r, f_s, f_c, f_dc, f_sc, f_dsc = evaluate(
        prefactors,
        field("xi_ev"),
        field("y_ev"),
        field("xiimax_ev"),
        field("xi_cnt", COLLINEAR),
        field("xiimax_cnt", COLLINEAR),
        field("p_cnt", COLLINEAR),
        field("p_cnt", SOFT_COLLINEAR),
        fks_index=np.array(index),
    )
    fortran = {
        key: np.array([p["prefactors"][key] for p in points])
        for key in ["f_r", "f_s", "f_c", "f_dc", "f_sc", "f_dsc"]
    }
    sym, sym_deg = field("fks_symmetry_factor"), field("fks_symmetry_factor_deg")
    jac, jac_cnt = field("jac"), field("jac_cnt")
    xinorm, xinorm_c = field("xinorm_ev"), field("xinorm_cnt", COLLINEAR)
    emitter_mass = {c["nfks"] - 1: c["masses"][c["j_fks"] - 1] for c in data["configs"]}
    massless = np.array([emitter_mass[c] == 0.0 for c in index])
    norm_c = xinorm_c * jac_cnt[:, COLLINEAR]
    norm_sc = xinorm_c * jac_cnt[:, SOFT_COLLINEAR]
    # where the counter-events are generated (a negative Jacobian flags those
    # genps_fks.f could not generate)
    soft = jac_cnt[:, SOFT] > 0
    collinear = massless & (jac_cnt[:, COLLINEAR] > 0)
    soft_collinear = massless & (jac_cnt[:, SOFT_COLLINEAR] > 0)
    cases = [
        ("f_r", f_r, fortran["f_r"], xinorm * jac * sym, jac > 0),
        ("f_s", f_s, fortran["f_s"], xinorm * jac_cnt[:, SOFT] * sym, soft),
        ("f_c", f_c, fortran["f_c"], norm_c * sym, collinear),
        ("f_dc", f_dc, fortran["f_dc"], norm_c * sym_deg, collinear),
        ("f_sc", f_sc, fortran["f_sc"], norm_sc * sym_deg, soft_collinear),
    ] + [
        (
            f"f_dsc({k + 1})",
            f_dsc[:, k],
            fortran["f_dsc"][:, k],
            norm_sc * sym_deg,
            soft_collinear,
        )
        for k in range(4)
    ]
    for key, ours, theirs, norm, valid in cases:
        # where the Fortran prefactor has its Jacobian and symmetry factor
        valid = valid & (norm != 0)
        assert valid.any() or not massless.any(), key
        np.testing.assert_allclose(
            ours[valid], theirs[valid] / norm[valid], rtol=1e-12, err_msg=key
        )
    # no collinear counterterms for a massive emitter
    for ours in [f_c, f_dc, f_sc, *f_dsc.T]:
        np.testing.assert_array_equal(ours[~massless], 0.0)


@pytest.mark.parametrize(
    "name", ["pp_ttx_P0_gg_ttx", "epem_uuxg_P0_epem_uuxg", "pp_epemj_P0_ug_uemep"]
)
def test_counter_event_xi_on_mapping_events(name):
    """The collinear counter-event has the xi of the event for a final-state
    emitter and its xi_hat for an initial-state one, and the prefactors are
    finite on the events of the mappings."""
    data = reference(FKS_DATA[FKS_DATA_IDS.index(name)])
    configurations = data["fks_configurations"]
    born = born_mapping(data)
    mapping = fks_configuration_mapping(born, configurations)
    prefactors = fks_prefactors(born, configurations)
    count = len(configurations["configurations"])
    names = list(mapping.counter_event_types().keys())
    rng = np.random.default_rng(41)
    batch = 400
    r = 0.05 + 0.9 * rng.random((batch, mapping.random_dim()))
    index = rng.integers(0, count, batch).astype(np.int32)
    out = dict(zip(names, evaluate(ms.FKSKinematics(mapping), r, fks_index=index)))
    j_fks = np.array([c["j_fks"] - 1 for c in configurations["configurations"]])
    final = j_fks[index] >= 2
    ok = out["collinear_det"] > 0
    assert ok.mean() > 0.5
    np.testing.assert_allclose(
        out["collinear_xi"][ok & final], out["xi"][ok & final], rtol=1e-14
    )
    np.testing.assert_allclose(
        (out["collinear_xi"] / out["collinear_xi_max"])[ok & ~final],
        out["xi_hat"][ok & ~final],
        rtol=1e-13,
    )
    # the soft counter-event: xi = 0 and the range of the event
    np.testing.assert_array_equal(out["soft_xi"][out["soft_det"] > 0], 0.0)
    values = evaluate(
        prefactors,
        out["xi"],
        out["y"],
        out["xi_max"],
        out["collinear_xi"],
        out["collinear_xi_max"],
        out["collinear_momenta"],
        out["soft_collinear_momenta"],
        fks_index=index,
    )
    for value in values:
        assert np.isfinite(value[ok]).all()


def test_invalid_parameters():
    masses = [[0.0] * 5]
    with pytest.raises(ValueError, match="positive"):
        ms.FKSPrefactors([0], masses, xicut=0.0)
    with pytest.raises(ValueError, match="particle of the real process"):
        ms.FKSPrefactors([5], masses)
    with pytest.raises(ValueError, match="number of particles"):
        ms.FKSPrefactors([0, 1], masses + [[0.0] * 6])
