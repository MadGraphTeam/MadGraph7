"""FKSSoftVirtual: the n-body part of the FKS subtraction without the
virtual, as bornsoftvirtual in fks_singular.f.

The reference data (see test_fks_reference.py) store, at the soft
counter-event of each generated event of a configuration with a gluon i,
bornsoftvirtual without the virtual with two different factorization scales,
a renormalization scale and an Ellis-Sexton scale (bsv_wgt and born_wgt) and
the weight lines of MadFKS (wgtnstmp and the coefficients wgtwnstmpmur and
wgtwnstmpmuf of the logarithms of the scales), and at its Born the Born
blocks of pyAmpliCol. From the Born momenta and blocks, FKSSoftVirtual with
the renormalization scale at the Ellis-Sexton scale must give bsv_wgt minus
born_wgt without the logarithm of the renormalization scale, and the
coefficients of the logarithms of the scales.
"""

import numpy as np
import pytest
from fks_testing import FKS_DATA, FKS_DATA_IDS, evaluate, reference
from fks_testing import reference_born_mapping as born_mapping
from test_fks_limits import generated_points

import madspace as ms
from madspace.fks import fks_configurations, fks_soft_virtual

SOFT = 2
# the power of alpha_s of the Born of the processes of the reference data
ALPHA_S_POWER = {
    "e+ e- > t t~ [QCD]": 0,
    "e+ e- > u u~ g [QCD]": 1,
    "p p > e+ e- [QCD]": 0,
    "p p > e+ e- j [QCD]": 1,
    "p p > t t~ [QCD]": 2,
}


def soft_virtual(data):
    """The FKSSoftVirtual of the reference data, with its parameters."""
    subtraction = data["subtraction"]
    return fks_soft_virtual(
        born_mapping(data),
        data["fks_configurations"],
        ALPHA_S_POWER[data["process"]],
        xicut=subtraction["xicut"],
        delta_o=subtraction["delta_o"],
        n_f=data["n_f"],
    )


def rows_and_arguments(data):
    """The points with bornsoftvirtual and the arguments of FKSSoftVirtual
    there: the Born momenta of the soft counter-event, the Born blocks and the
    scales of the dump, with the renormalization scale at the Ellis-Sexton
    scale."""
    rows = [(c, p) for c, p in generated_points(data) if "soft_virtual" in p]
    momenta = np.array(
        [np.delete(np.array(p["p_cnt"][SOFT]), c["i_fks"] - 1, axis=0) for c, p in rows]
    )
    blocks = [p["born_blocks"] for _, p in rows]
    scales = data["soft_virtual_scales"]
    batch = len(rows)
    return rows, [
        momenta,
        np.array([b["born"] for b in blocks]),
        np.array([b["color_correlated"] for b in blocks]),
        np.full(batch, data["born_blocks"]["alpha_s"]),
        np.full(batch, np.sqrt(scales["qes2"])),
        np.full(batch, np.sqrt(scales["mu_f2"][0])),
        np.full(batch, np.sqrt(scales["mu_f2"][1])),
    ]


@pytest.mark.parametrize("path", FKS_DATA, ids=FKS_DATA_IDS)
def test_matches_bornsoftvirtual(path):
    data = reference(path)
    rows, args = rows_and_arguments(data)
    assert rows
    value, ren, fact = evaluate(soft_virtual(data), *args)
    # the Born of MadFKS (sborn) is that of the Born process, without iden_comp
    bsv, born, nstmp, mur, muf = np.array([p["soft_virtual"] for _, p in rows]).T
    scales = data["soft_virtual_scales"]
    log_mu_r = np.log(scales["mu_r2"] / scales["qes2"])
    log_mu_f = np.log(np.array(scales["mu_f2"]) / scales["qes2"])
    # the Born of pyAmpliCol and the Fortran agree to 1e-12
    np.testing.assert_allclose(args[1], born, rtol=1e-11, atol=0)
    tolerance = 1e-11 * np.abs(born)
    assert (np.abs(value + ren * log_mu_r - (bsv - born)) <= tolerance).all()
    # wgtnstmp drops the logarithms of mu_R and of the factorization scale
    # of the first beam, which wgtwnstmpmuf multiplies for both beams
    nstmp_ours = value - fact.sum(1) * log_mu_f[0]
    assert (np.abs(nstmp_ours - nstmp) <= tolerance).all()
    assert (np.abs(ren - mur) <= tolerance).all()
    assert (np.abs(fact.sum(1) - muf) <= tolerance).all()
    # the two beams take their own factorization scales, which differ
    assert scales["mu_f2"][0] != scales["mu_f2"][1]
    incoming = [np.array(c["particle_type"][:2]) != 1 for c, _ in rows]
    np.testing.assert_array_equal(fact[~np.array(incoming)], 0.0)


@pytest.mark.parametrize("path", FKS_DATA, ids=FKS_DATA_IDS)
def test_any_frame(path):
    """The energies are those of the partonic center-of-mass frame, whatever
    the frame of the momenta."""
    data = reference(path)
    _, args = rows_and_arguments(data)
    rapidity = 0.7
    boost = np.eye(4)
    boost[0, 0] = boost[3, 3] = np.cosh(rapidity)
    boost[0, 3] = boost[3, 0] = np.sinh(rapidity)
    boosted = [np.einsum("mn,bkn->bkm", boost, args[0]), *args[1:]]
    function = soft_virtual(data)
    for ours, theirs in zip(evaluate(function, *args), evaluate(function, *boosted)):
        np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-14)


def test_scales():
    """The value moves with the factorization scales by their coefficients,
    and the coefficient of the renormalization scale is the running of the
    Born."""
    data = reference(FKS_DATA[FKS_DATA_IDS.index("pp_ttx_P0_gg_ttx")])
    rows, args = rows_and_arguments(data)
    function = soft_virtual(data)
    value, ren, fact = evaluate(function, *args)
    moved = list(args)
    moved[5] = 2.0 * args[5]
    moved[6] = 0.3 * args[6]
    value_moved, ren_moved, fact_moved = evaluate(function, *moved)
    np.testing.assert_allclose(ren_moved, ren, rtol=1e-15)
    np.testing.assert_allclose(fact_moved, fact, rtol=1e-15)
    shift = fact @ np.log([4.0, 0.09])
    np.testing.assert_allclose(value_moved, value + shift, rtol=1e-12, atol=0)
    # b gamma_g alpha_s / (2 pi) B with b = 2 and n_f = 4
    gamma_g = (11 * 3 - 2 * data["n_f"]) / 6
    np.testing.assert_allclose(
        ren, 2 * gamma_g * args[3] / (2 * np.pi) * args[1], rtol=1e-14
    )


def test_parameters():
    data = reference(FKS_DATA[FKS_DATA_IDS.index("pp_ttx_P0_gg_ttx")])
    born = born_mapping(data)
    configurations = data["fks_configurations"]
    function = fks_soft_virtual(born, configurations, 2, xicut=0.3, delta_o=0.8)
    assert function.alpha_s_power() == 2
    assert function.particle_count() == 4
    for parameters in [
        dict(alpha_s_power=-1),
        dict(alpha_s_power=2, xicut=0.0),
        dict(alpha_s_power=2, delta_o=-1.0),
        dict(alpha_s_power=2, n_f=-1.0),
    ]:
        with pytest.raises(ValueError):
            ms.FKSSoftVirtual(fks_configurations(born, configurations), **parameters)
    # without a gluon i, there are no color links
    quark_i = [
        c for c in fks_configurations(born, configurations) if c.colors[c.i_fks] != 8
    ]
    if quark_i:
        with pytest.raises(ValueError):
            ms.FKSSoftVirtual(quark_i, alpha_s_power=2)
