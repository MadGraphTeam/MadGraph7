"""FKSSoftVirtual: the n-body part of the FKS subtraction without the
virtual, as bornsoftvirtual in fks_singular.f, per NLO order.

The reference data (see test_fks_reference.py) store, at the soft
counter-event of each generated event of a configuration with a gluon or
photon i, bornsoftvirtual without the virtual with two different
factorization scales, a renormalization scale and an Ellis-Sexton scale
(bsv_wgt and born_wgt) and the weight lines of MadFKS (wgtnstmp and the
coefficients wgtwnstmpmur and wgtwnstmpmuf of the logarithms of the scales),
with their parts per position of the amp_split arrays, and at its Born the
Born blocks of pyAmpliCol. From the Born momenta and blocks, FKSSoftVirtual
with the renormalization scale at the Ellis-Sexton scale must give at each
NLO order the part of bsv_wgt there without the logarithm of the
renormalization scale, and the coefficients of the logarithms of the scales,
for QCD and QED.
"""

import numpy as np
import pytest
from fks_testing import (
    FKS_DATA,
    FKS_DATA_IDS,
    alpha,
    at_nlo_orders,
    born_inputs,
    evaluate,
    reference,
)
from fks_testing import reference_born_mapping as born_mapping
from test_fks_limits import generated_points

import madspace as ms
from madspace.subtraction.fks import (
    fks_configurations,
    fks_coupling_orders,
    fks_soft_virtual,
)

SOFT = 2


def soft_virtual(data):
    """The FKSSoftVirtual of the reference data, with its parameters."""
    subtraction = data["subtraction"]
    return fks_soft_virtual(
        born_mapping(data),
        data["fks_configurations"],
        xicut=subtraction["xicut"],
        delta_o=subtraction["delta_o"],
        n_f=data["n_f"],
        n_l=data["n_l"],
    )


def rows_and_arguments(data):
    """The points with bornsoftvirtual and the arguments of FKSSoftVirtual
    there: the Born momenta of the soft counter-event, the Born blocks, the
    couplings and the scales of the dump, with the renormalization scale at
    the Ellis-Sexton scale."""
    rows = [(c, p) for c, p in generated_points(data) if "soft_virtual" in p]
    if len(data["fks_configurations"]["orders"]["born"]) > 1:
        # the reference has color-correlated Borns per order only where i is
        # a gluon (fortran_orders of make_fks_reference.py)
        rows = [(c, p) for c, p in rows if c["pdg"][c["i_fks"] - 1] == 21]
    momenta = np.array(
        [np.delete(np.array(p["p_cnt"][SOFT]), c["i_fks"] - 1, axis=0) for c, p in rows]
    )
    # the Born orders only, without the extra counterterms
    born, color, _ = born_inputs(data, rows, np.zeros((len(rows), 4)))
    born = born[:, : color.shape[1]]
    scales = data["soft_virtual_scales"]
    batch = len(rows)
    return rows, [
        momenta,
        born,
        color,
        np.full(batch, data["born_blocks"]["alpha_s"]),
        np.full(batch, alpha(data)),
        np.full(batch, np.sqrt(scales["qes2"])),
        np.full(batch, np.sqrt(scales["mu_f2"][0])),
        np.full(batch, np.sqrt(scales["mu_f2"][1])),
    ]


@pytest.mark.parametrize("path", FKS_DATA, ids=FKS_DATA_IDS)
def test_matches_bornsoftvirtual(path):
    data = reference(path)
    rows, args = rows_and_arguments(data)
    assert rows
    function = soft_virtual(data)
    orders = data["fks_configurations"]["orders"]
    assert function.born_order_count() == len(orders["born"])
    assert function.nlo_order_count() == len(orders["nlo"])
    value, ren, fact = evaluate(function, *args)
    # the Born of MadFKS (sborn) is that of the Born process, without iden_comp
    bsv, born, nstmp, mur, muf = np.array([p["soft_virtual"] for _, p in rows]).T
    # per NLO order: amp_split after bornsoftvirtual (the n-body part with
    # the logarithm of the renormalization scale), wgtnstmp, wgtwnstmpmur and
    # wgtwnstmpmuf
    bsv_o, nstmp_o, mur_o, muf_o = np.moveaxis(
        at_nlo_orders([p["soft_virtual_orders"] for _, p in rows], data), 1, 0
    )
    scales = data["soft_virtual_scales"]
    log_mu_r = np.log(scales["mu_r2"] / scales["qes2"])
    log_mu_f = np.log(np.array(scales["mu_f2"]) / scales["qes2"])
    # the Born of pyAmpliCol and the Fortran agree to 1e-12, summed over the
    # Born orders
    np.testing.assert_allclose(args[1].sum(1), born, rtol=1e-11, atol=0)
    tolerance = 1e-11 * np.abs(born)[:, None]
    assert (np.abs(value + ren * log_mu_r - bsv_o) <= tolerance).all()
    # wgtnstmp drops the logarithms of mu_R and of the factorization scale
    # of the first beam, which wgtwnstmpmuf multiplies for both beams
    nstmp_ours = value - fact.sum(2) * log_mu_f[0]
    assert (np.abs(nstmp_ours - nstmp_o) <= tolerance).all()
    assert (np.abs(ren - mur_o) <= tolerance).all()
    assert (np.abs(fact.sum(2) - muf_o) <= tolerance).all()
    # the sums over the orders
    assert (
        np.abs(value.sum(1) + ren.sum(1) * log_mu_r - (bsv - born)) <= tolerance[:, 0]
    ).all()
    np.testing.assert_allclose(nstmp_o.sum(1), nstmp, rtol=1e-14, atol=0)
    # the two beams take their own factorization scales, which differ
    assert scales["mu_f2"][0] != scales["mu_f2"][1]
    # no coefficient for a neutral, colorless incoming particle
    configurations = fks_configurations(born_mapping(data), data["fks_configurations"])
    for (c, _), coefficients in zip(rows, fact):
        config = configurations[c["nfks"] - 1]
        for beam in range(2):
            inert = config.colors[beam] == 1 and config.charges[beam] == 0.0
            if inert and config.pdgs[beam] != 22:
                np.testing.assert_array_equal(coefficients[:, beam], 0.0)


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
    moved[6] = 2.0 * args[6]
    moved[7] = 0.3 * args[7]
    value_moved, ren_moved, fact_moved = evaluate(function, *moved)
    np.testing.assert_allclose(ren_moved, ren, rtol=1e-15)
    np.testing.assert_allclose(fact_moved, fact, rtol=1e-15)
    shift = fact @ np.log([4.0, 0.09])
    np.testing.assert_allclose(value_moved, value + shift, rtol=1e-12, atol=0)
    # b gamma_g alpha_s / (2 pi) B with b = 2 and n_f = 4
    gamma_g = (11 * 3 - 2 * data["n_f"]) / 6
    np.testing.assert_allclose(
        ren[:, 0], 2 * gamma_g * args[3] / (2 * np.pi) * args[1][:, 0], rtol=1e-14
    )


def test_qed_has_no_running():
    """Without QCD counterterms there is no coefficient of the renormalization
    scale, and the QED part scales with alpha."""
    data = reference(FKS_DATA[FKS_DATA_IDS.index("pp_epema_qed_P0_uux_emepa")])
    rows, args = rows_and_arguments(data)
    function = soft_virtual(data)
    value, ren, fact = evaluate(function, *args)
    np.testing.assert_array_equal(ren, 0.0)
    doubled = list(args)
    doubled[4] = 2.0 * args[4]
    # alpha_s does not enter
    doubled[3] = 0.5 * args[3]
    value_doubled, _, fact_doubled = evaluate(function, *doubled)
    np.testing.assert_allclose(value_doubled, 2.0 * value, rtol=1e-14)
    np.testing.assert_allclose(fact_doubled, 2.0 * fact, rtol=1e-14)


def test_tagged_photons():
    """A tagged photon (an identified one, !a!) has no Q terms: tagging the
    photon of the Born of u u~ > e- e+ a removes a B Q_a, with the gamma and
    gamma' of the photon from n_f and n_l, and nothing else."""
    data = reference(FKS_DATA[FKS_DATA_IDS.index("pp_epema_qed_P0_uux_emepa")])
    rows, args = rows_and_arguments(data)
    born = born_mapping(data)
    configurations = fks_configurations(born, data["fks_configurations"])
    orders = fks_coupling_orders(data["fks_configurations"])
    photon = data["fks_configurations"]["born_pdgs"].index(22)
    tagged = []
    for config in configurations:
        # the Born photon in the real process, after i, unless it is the
        # emitter j split into a pair (which a tagged photon never is)
        position = photon if photon < config.i_fks else photon + 1
        assert not config.tagged(position)
        if config.pdgs[position] == 22:
            config.tags = [k == position for k in range(config.particle_count())]
            assert config.tagged(position)
        else:
            assert config.j_fks == position
        tagged.append(config)
    assert configurations[0].pdgs[configurations[0].i_fks] == 22
    subtraction = data["subtraction"]
    parameters = dict(
        xicut=subtraction["xicut"],
        delta_o=subtraction["delta_o"],
        n_f=data["n_f"],
        n_l=data["n_l"],
    )
    value, ren, fact = evaluate(soft_virtual(data), *args)
    value_tagged, ren_tagged, fact_tagged = evaluate(
        ms.FKSSoftVirtual(tagged, orders, **parameters), *args
    )
    # the photon is in the final state, so the coefficients stay
    np.testing.assert_array_equal(ren_tagged, ren)
    np.testing.assert_array_equal(fact_tagged, fact)
    momenta = args[0]
    p_tot = momenta[:, 0] + momenta[:, 1]
    s = p_tot[:, 0] ** 2 - (p_tot[:, 1:] ** 2).sum(1)
    p = momenta[:, photon]
    energy = (p[:, 0] * p_tot[:, 0] - (p[:, 1:] * p_tot[:, 1:]).sum(1)) / np.sqrt(s)
    charges = sum(
        4 / 9 if f % 2 == 0 else 1 / 9 for f in range(1, round(data["n_f"]) + 1)
    )
    gamma = -(3 * charges + data["n_l"]) * 2 / 3
    gamma_prime = -(3 * charges + data["n_l"]) * 23 / 9
    mu_r2 = args[5] ** 2
    l_o = np.log(s * subtraction["delta_o"] / (2 * mu_r2))
    l_e = np.log(2 * energy / np.sqrt(s))
    q_photon = gamma_prime - l_o * gamma - 2 * gamma * l_e
    a = args[4] / (2 * np.pi)
    np.testing.assert_allclose(
        (value - value_tagged)[:, 0], a * args[1][:, 0] * q_photon, rtol=1e-12
    )


def test_parameters():
    data = reference(FKS_DATA[FKS_DATA_IDS.index("pp_ttx_P0_gg_ttx")])
    born = born_mapping(data)
    configurations = data["fks_configurations"]
    function = fks_soft_virtual(born, configurations, xicut=0.3, delta_o=0.8)
    assert function.particle_count() == 4
    assert (function.born_order_count(), function.nlo_order_count()) == (1, 1)
    orders = fks_coupling_orders(configurations)
    for parameters in [
        dict(xicut=0.0),
        dict(delta_o=-1.0),
        dict(n_f=-1.0),
        dict(n_l=-1.0),
    ]:
        with pytest.raises(ValueError):
            ms.FKSSoftVirtual(
                fks_configurations(born, configurations), orders, **parameters
            )
    # without a gluon or photon i, there are no color or charge links
    quark_i = [
        c for c in fks_configurations(born, configurations) if c.colors[c.i_fks] != 8
    ]
    if quark_i:
        with pytest.raises(ValueError, match="gluon or photon"):
            ms.FKSSoftVirtual(quark_i, orders)
