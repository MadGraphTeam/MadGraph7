"""The reference data of the fixed-order NLO integrand of MadFKS
(test_data/fks_integrand, from fks_reference/make_fks_integrand_reference.py).

They hold, per point of a short run, the contributions as sigint of
driver_mintFO.f sums them. These tests check that the data are what the FKS
integrand of madspace will be compared with: the weight of a contribution
is its luminosity times (wgt_1 + wgt_2 log(mu_R^2/Q^2) + wgt_3
log(mu_F^2/Q^2)) g_s^n, summed over the subprocesses of parton_iproc, the
n-body part sits at the FKS configuration of the Born and the n+1-body part
at the picked one, and the momenta conserve energy and momentum.
"""

import glob
import json
import math
import os

import numpy as np
import pytest

DATA = sorted(
    glob.glob(
        os.path.join(os.path.dirname(__file__), "test_data", "fks_integrand", "*.json")
    )
)
IDS = [os.path.basename(path)[:-5] for path in DATA]
# MadFKS converts GeV^-2 to pb (conv of include_PDF_and_alphas)
CONV = 389379660.0
NBODY = {2, 3, 14, 15}
N1BODY = {1, 4, 5, 6}


def reference(path):
    with open(path) as f:
        return json.load(f)


def points(data):
    for channel in data["channels"]:
        for point in channel["points"]:
            yield channel, point


def without_pdfs(c):
    """The weight of a contribution before the luminosity."""
    q2, mu_r2, mu_f2 = c["scales2"]
    w1, w2, w3 = c["wgt"]
    scale = w1 + w2 * math.log(mu_r2 / q2) + w3 * math.log(mu_f2 / q2)
    return scale * c["g_strong"] ** c["qcd_power"]


def test_reference_covers_the_integrand():
    """Final-state emitters without PDFs, initial-state emitters with PDFs, a
    massive final state, and every contribution of fixed order."""
    assert {"epem_jj", "pp_epem", "pp_ttx"} <= set(IDS)
    for path in DATA:
        types = {
            c["type"] for _, p in points(reference(path)) for c in p["contributions"]
        }
        assert NBODY | N1BODY <= types, path


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_weights_are_luminosity_times_coefficients(path):
    """wgts(1) = sum over the subprocesses of parton_iproc, and that over the
    weight without PDFs is the luminosity, conv without PDFs. (It can be
    negative: the gluon of NNPDF23 is at x close to 1.)"""
    data = reference(path)
    leptonic = int(float(data["run_card"]["lpp1"])) == 0
    for _, point in points(data):
        for c in point["contributions"]:
            np.testing.assert_allclose(sum(c["parton_iproc"]), c["weight"], rtol=1e-14)
            bare = without_pdfs(c)
            if bare == 0:
                assert c["weight"] == 0
                continue
            luminosity = c["weight"] / bare
            if leptonic:
                np.testing.assert_allclose(luminosity / c["niproc"], CONV, rtol=1e-14)


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_parts_sit_at_their_configurations(path):
    """The n-body part at the FKS configuration of the Born (nfks_born), the
    n+1-body part at the picked one, one pick per point."""
    for channel, point in points(reference(path)):
        assert not point["sum"]
        assert point["real_type"] == 0
        count = len(channel["fks_configurations"]["configurations"])
        assert 1 <= point["nfks_picked"] <= count
        for c in point["contributions"]:
            if c["type"] in NBODY:
                assert c["nfks"] == point["nfks_born"]
            elif c["type"] in N1BODY:
                assert c["nfks"] == point["nfks_picked"]


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_momenta_conserve(path):
    """In the partonic center-of-mass frame of each contribution, with the
    incoming momenta along the beams."""
    for _, point in points(reference(path)):
        for c in point["contributions"]:
            p = np.array(c["momenta"])
            total = p[2:].sum(0) - p[:2].sum(0)
            scale = p[:, 0].max()
            np.testing.assert_allclose(total / scale, 0.0, atol=1e-12)
            np.testing.assert_allclose(p[:2, 1:3], 0.0, atol=1e-12 * scale)


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_born_of_the_point(path):
    """The Born of the phase-space point conserves momentum at sqrt(shat),
    and the Born contribution carries the momentum fractions of that Born."""
    for _, point in points(reference(path)):
        born = point["born"]
        if not born["valid"]:
            continue
        p = np.array(point["p_born"])
        np.testing.assert_allclose(p[:2, 0].sum() ** 2, born["shat"], rtol=1e-12)
        for c in point["contributions"]:
            if c["type"] == 2:
                np.testing.assert_allclose(c["bjx"], born["xbjrk"], rtol=1e-14)


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_matrix_element_blocks(path):
    """The blocks of a point sit at its configurations, sum over the
    split-order positions, and the colour links of e+ e- > q q~ are -C_F B
    g_s^2 (sborn_sf has g_s^2)."""
    data = reference(path)
    count = 0
    for _, point in points(data):
        blocks = point.get("blocks")
        if blocks is None:
            assert not point["born"]["valid"]
            continue
        count += 1
        assert blocks["nfks"] == point["nfks_picked"]
        assert blocks["nbody"]["nfks"] == point["nfks_born"]
        np.testing.assert_allclose(
            sum(blocks["born_orders"]), blocks["born"], rtol=1e-14
        )
        if "real" in blocks:
            np.testing.assert_allclose(
                sum(blocks["real_orders"]), blocks["real"], rtol=1e-14
            )
        for link in blocks["nbody"]["links"]:
            np.testing.assert_allclose(sum(link["orders"]), link["value"], rtol=1e-14)
            if os.path.basename(path) == "epem_jj.json":
                ratio = link["value"] / (link["g"] ** 2 * blocks["born"])
                np.testing.assert_allclose(ratio, -4 / 3, rtol=1e-12)
    assert count > 0
