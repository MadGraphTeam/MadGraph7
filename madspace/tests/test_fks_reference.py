"""Consistency of the genps_fks.f reference data in test_data/fks.

The data are produced by fks_reference/make_fks_reference.py and are the
reference for the madspace FKS mappings. These checks only make sure that
the files are complete and that the conventions the mapping tests rely on
hold: momenta in the Born partonic c.m. frame, the radiation-only rerun
reproducing the event, and the counter-events being the expected limits.
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
        assert np.allclose(point["rad_p_ev"], p, rtol=0, atol=1e-13 * scale)
        assert len(point["x"]) == data["ndim"]
        count += 1
    assert count > 0


def test_born_frame(data):
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
    nin = data["nincoming"]
    for conf, point in valid_points(data):
        i_fks, j_fks = conf["i_fks"] - 1, conf["j_fks"] - 1
        p_born = np.array(point["p_born"])
        born_as_real = np.insert(p_born, i_fks, 0.0, axis=0)
        scale = p_born[:, 0].max()
        if point["rad_jac_cnt"][SOFT] > 0:
            p_soft = np.array(point["p_cnt"][SOFT])
            assert np.allclose(p_soft, born_as_real, rtol=0, atol=1e-10 * scale)
            assert point["xi_cnt"][SOFT] == 0.0
        if point["rad_jac_cnt"][COLL] > 0:
            p_coll = np.array(point["p_cnt"][COLL])
            mother = p_coll[i_fks] + p_coll[j_fks]
            spectators = [k for k in range(len(p_coll)) if k not in (i_fks, j_fks)]
            # xi_hat is shared, xi only for a final-state emitter (for an
            # initial-state one the upper bound of xi depends on y)
            assert np.isclose(point["xi_hat_cnt"][COLL], point["xi_hat_ev"], rtol=1e-14)
            if j_fks >= nin:
                assert np.isclose(point["xi_cnt"][COLL], point["xi_ev"], rtol=1e-14)
                # i and j collinear and summing to the Born emitter
                assert np.allclose(mother, born_as_real[j_fks], atol=1e-10 * scale)
                assert np.allclose(
                    p_coll[spectators], born_as_real[spectators], atol=1e-10 * scale
                )
