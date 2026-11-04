"""Consistency of the genps_fks.f reference data in test_data/fks.

The data are produced by fks_reference/make_fks_reference.py and are the
reference for the madspace FKS mappings. These checks only make sure that
the files are complete and that the conventions the mapping tests rely on
hold: momenta in the Born partonic c.m. frame, the radiation-only rerun
reproducing the event, and the counter-events being the expected limits.
The Born blocks of pyAmpliCol must be the Born of MadFKS and its color-linked
versions in their conventions.
"""

import glob
import json
import os

import numpy as np
import pytest

FILES = sorted(
    glob.glob(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_data", "fks", "*.json"
        )
    )
)
METRIC = np.diag([1.0, -1.0, -1.0, -1.0])
SOFT, COLL, SOFT_COLL = 2, 3, 4  # position of counter-event k = 0, 1, 2

pytestmark = pytest.mark.cpu_only


def minkowski_sq(p):
    return np.einsum("...i,ij,...j->...", p, METRIC, p)


def valid_points(data):
    for conf in data["configs"]:
        for point in conf["points"]:
            if point["jac"] > 0 and "rad_jac" in point:
                yield conf, point


@pytest.fixture(params=FILES, ids=[os.path.basename(f)[:-5] for f in FILES])
def data(request):
    with open(request.param) as f:
        return json.load(f)


def test_files_present():
    names = {os.path.basename(f) for f in FILES}
    for prefix in ["epem_uuxg", "epem_ttx", "pp_epem", "pp_ttx"]:
        assert any(n.startswith(prefix) for n in names)


def test_emitter_cases_covered():
    cases = set()
    for path in FILES:
        with open(path) as f:
            data = json.load(f)
        for conf in data["configs"]:
            j = conf["j_fks"]
            if j <= data["nincoming"]:
                cases.add("initial")
            elif conf["masses"][j - 1] > 0:
                cases.add("final_massive")
            else:
                cases.add("final_massless")
    assert cases == {"initial", "final_massive", "final_massless"}


def test_event_kinematics(data):
    nin = data["nincoming"]
    count = 0
    for conf, point in valid_points(data):
        p = np.array(point["p_ev"])
        scale = p[:, 0].max()
        assert np.allclose(p[:nin].sum(0), p[nin:].sum(0), rtol=0, atol=1e-12 * scale)
        assert np.allclose(
            minkowski_sq(p),
            np.array(conf["masses"]) ** 2,
            rtol=0,
            atol=1e-12 * scale**2,
        )
        assert len(point["x"]) == data["ndim"]
        count += 1
    assert count > 0


def test_uncut_rerun(data):
    """The rad_ entries rerun the FKS map on the Born of the point without the
    lower bound tau_lower_bound that the cuts put on the partonic energy of
    the real emission. They give a physical event, and the same one as the
    main dump wherever the bound does not act, which is always the case for a
    final-state emitter (the bound restricts xi and y of initial-state ones)."""
    nin = data["nincoming"]
    for conf, point in valid_points(data):
        p = np.array(point["rad_p_ev"])
        scale = p[:, 0].max()
        assert np.allclose(p[:nin].sum(0), p[nin:].sum(0), rtol=0, atol=1e-12 * scale)
        assert np.allclose(
            minkowski_sq(p),
            np.array(conf["masses"]) ** 2,
            rtol=0,
            atol=1e-12 * scale**2,
        )
        same_xi = abs(point["rad_xi_ev"] - point["xi_ev"]) <= 1e-12
        assert same_xi or conf["j_fks"] <= nin
        if same_xi:
            assert np.allclose(p, point["p_ev"], rtol=0, atol=1e-12 * scale)


def test_bounded_rerun(data):
    """The cut_ entries rerun the FKS map on the Born of the point with the
    lower bound tau_lower_bound kept, and so reproduce the event of the main
    dump. Their radiation Jacobian equals the uncut one for a final-state
    emitter."""
    nin = data["nincoming"]
    for conf, point in valid_points(data):
        assert 0 <= point["tau_lower_bound"] < 1
        p = np.array(point["cut_p_ev"])
        scale = p[:, 0].max()
        assert np.allclose(p, point["p_ev"], rtol=0, atol=1e-12 * scale)
        assert np.allclose(point["cut_xi_ev"], point["xi_ev"], rtol=0, atol=1e-12)
        if conf["j_fks"] > nin:
            assert np.isclose(point["cut_jac"], point["rad_jac"], rtol=1e-12)
    nin = data["nincoming"]
    for conf, point in valid_points(data):
        p_born = np.array(point["p_born"])
        p_ev = np.array(point["p_ev"])
        scale = p_born[:, 0].max()
        assert np.allclose(p_born[:nin, 1:].sum(0), 0.0, atol=1e-12 * scale)
        assert np.isclose(
            minkowski_sq(p_born[:nin].sum(0)), point["shat_born"], rtol=1e-12
        )
        # The event stays in the Born c.m. frame: for an initial-state emitter
        # the incoming partons of the event move with ycm_ev - ycm_born.
        q = p_ev[:nin].sum(0)
        y = 0.5 * np.log((q[0] + q[3]) / (q[0] - q[3]))
        assert np.isclose(y, point["ycm_ev"] - point["ycm_born"], atol=1e-12)


def test_counter_events(data):
    """Counter-events of the uncut FKS map (the rad_ entries)."""
    nin = data["nincoming"]
    for conf, point in valid_points(data):
        i_fks, j_fks = conf["i_fks"] - 1, conf["j_fks"] - 1
        p_born = np.array(point["p_born"])
        born_as_real = np.insert(p_born, i_fks, 0.0, axis=0)
        scale = p_born[:, 0].max()
        if point["rad_jac_cnt"][SOFT] > 0:
            p_soft = np.array(point["rad_p_cnt"][SOFT])
            assert np.allclose(p_soft, born_as_real, rtol=0, atol=1e-10 * scale)
            assert point["rad_xi_cnt"][SOFT] == 0.0
        if point["rad_jac_cnt"][COLL] > 0:
            p_coll = np.array(point["rad_p_cnt"][COLL])
            mother = p_coll[i_fks] + p_coll[j_fks]
            spectators = [k for k in range(len(p_coll)) if k not in (i_fks, j_fks)]
            # xi_hat is shared, xi only for a final-state emitter (for an
            # initial-state one the upper bound of xi depends on y)
            assert np.isclose(
                point["rad_xi_hat_cnt"][COLL], point["rad_xi_hat_ev"], rtol=1e-14
            )
            if j_fks >= nin:
                assert np.isclose(
                    point["rad_xi_cnt"][COLL], point["rad_xi_ev"], rtol=1e-14
                )
                # i and j collinear and summing to the Born emitter
                assert np.allclose(mother, born_as_real[j_fks], atol=1e-10 * scale)
                assert np.allclose(
                    p_coll[spectators], born_as_real[spectators], atol=1e-10 * scale
                )


def sbornsoft_pairs(conf):
    """The pairs (m, n) of sbornsoft, 1-based in the real process: every
    unordered pair of the partners of i, a particle with itself where it is
    massive."""
    partners = conf["fks_j_from_i"][conf["i_fks"] - 1]
    return [
        (m, n)
        for k, m in enumerate(partners)
        for n in partners[: k + 1]
        if m != n or conf["masses"][m - 1] != 0.0
    ]


def test_born_blocks(data):
    """The Born blocks of pyAmpliCol at the Born of every generated event: the
    Born of MadFKS (the QCD part of sborn), color-correlated Borns
    B_mn = <T_m . T_n> B whose sum over n vanishes by color conservation and
    which are the color-linked Borns of sborn_sf over g^2 (twice those for
    m = n), and for a gluon emitter the spin-correlated tensor S, transverse
    to the emitter, with -g_mu,nu S^mu,nu = B, the sum over its physical
    polarizations."""
    g2 = data["g"] ** 2
    assert data["born_blocks"]["alpha_s"] == pytest.approx(g2 / (4 * np.pi), rel=1e-15)
    born_pdgs = data["fks_configurations"]["born_pdgs"]
    count = 0
    for conf in data["configs"]:
        i_fks, j_fks = conf["i_fks"], conf["j_fks"]
        born_j = j_fks if j_fks < i_fks else j_fks - 1
        for point in conf["points"] + conf["limit_points"]:
            assert ("born" in point) == (point["jac"] > 0)
            if "born" not in point:
                continue
            blocks = point["born_blocks"]
            born = blocks["born"]
            # the Fortran in double precision, pyAmpliCol in double-double
            assert born == pytest.approx(point["born"], rel=1e-11)
            correlated = np.array(blocks["color_correlated"])
            scale = np.abs(np.diag(correlated)).max()
            np.testing.assert_allclose(correlated, correlated.T, rtol=0, atol=0)
            np.testing.assert_allclose(correlated.sum(1), 0, atol=1e-13 * scale)
            # color links where i is a gluon
            pairs = sbornsoft_pairs(conf) if conf["pdg"][i_fks - 1] == 21 else []
            assert len(point["color_links"]) == len(pairs)
            for (m, n), link in zip(pairs, point["color_links"]):
                mb, nb = m - (m > i_fks), n - (n > i_fks)
                expected = correlated[mb - 1, nb - 1] * (0.5 if m == n else 1)
                assert abs(link / g2 - expected) <= 1e-12 * scale
            gluon = born_pdgs[born_j - 1] == 21
            assert ("spin_correlated" in blocks) == gluon
            if gluon:
                tensor = np.array(blocks["spin_correlated"])
                np.testing.assert_allclose(tensor, tensor.T, rtol=0, atol=0)
                assert -np.trace(METRIC @ tensor) == pytest.approx(born, rel=1e-12)
                p_j = np.array(point["p_born"][born_j - 1])
                np.testing.assert_allclose(tensor @ p_j, 0, atol=1e-12 * born * p_j[0])
            count += 1
    assert count > 0


def test_limit_points(data):
    """Every configuration has points close to the soft limit, with xi_hat
    close to its lower end stiny = 1e-6 (of the range of xi_hat for a
    massive emitter), where sreal and fks_Sij switch to the soft limit below
    xi = 1e-6, and the Born blocks there."""
    switched = 0
    for conf in data["configs"]:
        points = conf["limit_points"]
        assert len(points) >= 2
        for point in points:
            assert 0 < point["xi_hat_ev"] <= 1.1e-6
            if point["jac"] > 0:
                assert "born_blocks" in point
                switched += point["xi_ev"] < 1e-6
    assert switched > 0
