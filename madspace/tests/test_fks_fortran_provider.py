"""The Fortran provider of the MG7 NLO driver (madgraph/mg7/nlo/provider.py
with fks_provider.f of its templates) against the references of
test_data/fks_integrand. At the points of the references its blocks at g = 1
are those MadFKS took there, over their powers of g: the real at the event,
the Born and its azimuthal part, amp2 of the Born diagrams, the colour links
of the pairs sbornsoft takes and the virtual of MadLoop. Its color-correlated
and spin-correlated Borns are those that test_fks_integrand builds from the
blocks of MadFKS, the latter with the phase of k_perp in place of xij_aor. At
the madspace kinematics of these points FKSCrossSection with its blocks gives
the weights it gives with those of MadFKS.

The Borns of the extra counterterms come from an output of g u > u a with
QCD and QED corrections at the orders of p p > a j of test_data/fks
(pp_aj_mix_P0_gu_ua.json), against its blocks of pyAmpliCol and the
counterterm parts of extra_cnt of MadFKS at its Borns.

The provider is compiled in the aMC@NLO outputs of the references, as a
fixed-order run leaves them (make_fks_integrand_reference.py --keep DIR
keeps them, one output per reference, DIR/epem_jj and so on). The tests run
where MADSPACE_FKS_OUTPUTS names that directory and are skipped otherwise.
"""

import atexit
import functools
import math
import os

import numpy as np
import pytest
from fks_testing import born_inputs, evaluate, reference
from madgraph.mg7.nlo.fks import born_topology
from madgraph.mg7.nlo.provider import FortranProvider, azimuthal_phase, fks_blocks
from test_fks_integrand import DATA, IDS, WEIGHTS, channels

from madspace.subtraction.fks import fks_cross_section, fks_splitting_kernels

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
pytestmark = pytest.mark.skipif(
    not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs"
)
# the blocks agree to the rounding of the two evaluations, the virtual to
# that of MadLoop
RTOL = 1e-12
VIRTUAL_RTOL = 1e-10
# the weights of the kinematics of madspace, which are those of MadFKS to the
# rounding of the Born (see test_fks_integrand.RTOL): a real 2e-9 apart at a
# soft event (xi = 6e-5 on e+ e- > j j)
WEIGHT_RTOL = 1e-8
# the output with extra counterterms, a subprocess of the process of the
# reference of test_data/fks with the same configurations, and the rounding
# of pyAmpliCol and MadFKS
EXTRA_OUTPUT = "gu_ua_mix"
EXTRA_SUBPROCESS = "P0_gu_ua"
EXTRA_REFERENCE = os.path.join(
    os.path.dirname(__file__), "test_data", "fks", "pp_aj_mix_P0_gu_ua.json"
)
EXTRA_RTOL = 1e-10


def name_of(path):
    return os.path.basename(path)[:-5]


@functools.cache
def provider(name, subprocess):
    p = FortranProvider(os.path.join(OUTPUTS, name, "SubProcesses", subprocess))
    atexit.register(p.close)
    return p


def qcd_powers(p):
    return np.array([orders[p.qcd_pos] for orders in p.positions])


def points_with_blocks(channel):
    return [point for point in channel["points"] if "blocks" in point]


def assert_close(ours, theirs, rtol):
    """ours and theirs per position, relative to the largest of theirs."""
    scale = np.abs(theirs).max()
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=rtol * scale)


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_real(path):
    """The real at the event of the picked configuration, where MadFKS has
    its weight line, is smatrix_real of MadFKS there over g^n per
    position."""
    count = 0
    for channel in reference(path)["channels"]:
        p = provider(name_of(path), channel["subprocess"])
        for point in points_with_blocks(channel):
            blocks = point["blocks"]
            lines = [c for c in point["contributions"] if c["type"] == 1]
            if "real_orders" not in blocks or not lines:
                continue
            count += 1
            ours = p.real(np.array([lines[0]["momenta"]]), [point["nfks_picked"] - 1])
            theirs = np.array(blocks["real_orders"]) / blocks["g_real"] ** qcd_powers(p)
            assert_close(ours[0], theirs, RTOL)
    assert count > 0


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_born(path):
    """At the Born of MadFKS with the azimuthal part of the picked
    configuration: the Born and its azimuthal part at the positions of the
    counterterms are those of sborn over g^b, the Born per position (the
    Born orders) is the Born at the counterterms, and amp2 of the Born
    diagrams is that of MadFKS up to a power of g."""
    for channel in reference(path)["channels"]:
        p = provider(name_of(path), channel["subprocess"])
        points = points_with_blocks(channel)
        momenta = np.array([point["p_born"] for point in points])
        picked = np.array([point["nfks_picked"] for point in points]) - 1
        born, parts, amp2 = p.born(momenta, picked)
        qcd = p.qcd_pos
        positions = [p.counterterm_position(o, qcd) for o in p.orders["born"]]
        powers = qcd_powers(p) - 2
        for k, point in enumerate(points):
            blocks = point["blocks"]
            g = blocks["g_born"] ** powers
            theirs = np.array(blocks["born_orders"]) / g
            azimuthal = np.array(blocks["azimuthal_orders"]) / g
            assert_close(parts[k, qcd, 0].real, theirs, RTOL)
            assert_close(parts[k, qcd, 1].real, azimuthal[0], RTOL)
            assert_close(parts[k, qcd, 1].imag, azimuthal[1], RTOL)
            for order, position in zip(p.orders["born"], positions):
                assert born[k, p.position(order)] == pytest.approx(
                    parts[k, qcd, 0, position].real, rel=1e-14
                )
            for entry in (blocks["channel"], blocks["nbody"]["channel"]):
                theirs = np.array(entry["amp2"])
                np.testing.assert_allclose(
                    amp2[k] / amp2[k].sum(), theirs / theirs.sum(), rtol=1e-12
                )


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_links(path):
    """sborn_sf of the pairs sbornsoft takes at the configuration of the
    n-body part, those of a massive leg with itself included, is that of
    MadFKS over g^(b + 2) per position."""
    count = 0
    for channel in reference(path)["channels"]:
        p = provider(name_of(path), channel["subprocess"])
        for point in points_with_blocks(channel):
            nbody = point["blocks"]["nbody"]
            if not nbody["links"]:
                continue
            pairs = [(link["m"], link["n"]) for link in nbody["links"]]
            ours = p.links(np.array([point["p_born"]]), nbody["nfks"] - 1, pairs)
            for l, link in enumerate(nbody["links"]):
                count += 1
                theirs = np.array(link["orders"]) / link["g"] ** qcd_powers(p)
                assert_close(ours[0, l], theirs, RTOL)
    assert count > 0


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_correlated_borns(path):
    """The color-correlated Borns over all pairs of Born legs (each pair from
    the first configuration with links that registers it) and the
    spin-correlated Born from xij_kperp of MadFKS with the phase of
    azimuthal_phase are those of test_fks_integrand from the blocks of
    MadFKS. That phase is xij_aor."""
    name = name_of(path)
    for data, channel in zip(reference(path)["channels"], channels(path)):
        p = provider(name, data["subprocess"])
        _, born, color, spin, _ = channel.blocks
        momenta = channel.field("p_born")
        picked = channel.field("nfks_picked") - 1
        scale = np.abs(born).max()
        ours = p.color_correlated(momenta)
        np.testing.assert_allclose(ours, color, rtol=0, atol=RTOL * scale)
        k_perp = channel.field("blocks", "k_perp")
        ours = p.spin_correlated(momenta, k_perp, picked)
        np.testing.assert_allclose(ours, spin, rtol=0, atol=RTOL * scale)
        phases = channel.field("blocks", "azimuthal_phase")
        for k, point in enumerate(channel.points):
            configuration = data["fks_configurations"]["configurations"][picked[k]]
            if configuration["ij_pdg"] != 21 or not phases[k].any():
                continue
            i_fks, j_fks = configuration["i_fks"], configuration["j_fks"]
            mother = np.array(point["p_born"][j_fks - 1 - (j_fks > i_fks)])
            ours = azimuthal_phase(mother[None], k_perp[k : k + 1])[0]
            assert abs(ours - complex(*phases[k])) < 1e-14


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_channels(path):
    """The Born configurations of MadFKS are the channels of amp2, one
    Born diagram each, with a propagator per internal line of the Born, and
    the topology of each maps the Born (born_topology of nlo.fks)."""
    for channel in reference(path)["channels"]:
        p = provider(name_of(path), channel["subprocess"])
        configurations = p.fks_configurations
        n = len(configurations["born_pdgs"])
        channels = p.channels()
        assert len(channels) == p.channel_count
        assert len({c["diagram"] for c in channels}) == len(channels)
        masses = [0.0] * n
        for c in channels:
            assert len(c["forest"]) == len(c["props"]) == n - 1 - p.incoming
            assert all(mass >= 0 and width >= 0 for mass, width in c["props"])
            topology = born_topology(masses, p.incoming, c)
            assert topology is not None


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_virtual(path):
    """The finite part of MadLoop at the Born and the Ellis-Sexton scale of
    the n-body weight lines (of the virtual, type 14) is that of MadFKS over
    g^n per position. The scale qes2 the reference keeps with the virtual is
    that at the end of sigint, of the last kinematics, not that of the
    virtual."""
    count = 0
    for channel in reference(path)["channels"]:
        p = provider(name_of(path), channel["subprocess"])
        for point in points_with_blocks(channel):
            virtual = point["blocks"]["nbody"].get("virtual")
            if virtual is None:
                continue
            count += 1
            (line,) = (c for c in point["contributions"] if c["type"] == 14)
            scale = math.sqrt(line["scales2"][0])
            values, codes = p.virtual(np.array([point["p_born"]]), scale)
            theirs = np.array(virtual["orders"]) / line["g_strong"] ** qcd_powers(p)
            assert_close(values["finite"][0], theirs, VIRTUAL_RTOL)
            # stable in double precision
            assert codes[0] // 100 < 4
    assert count > 0


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_cross_section(path):
    """FKSCrossSection with the real, Born, color-correlated and
    spin-correlated Borns of the provider at the madspace kinematics of the
    points (k_perp from FKSSplittingKernels at the collinear counter-event)
    gives the weights it gives with the blocks of MadFKS, and so those of
    MadFKS (test_fks_integrand.test_weights). The real at the kinematics of
    madspace is that at the kinematics of MadFKS up to their rounding, which
    a soft event enhances, and the cross section is the sum of the parts,
    which cancel."""
    name = name_of(path)
    for data, channel in zip(reference(path)["channels"], channels(path)):
        p = provider(name, data["subprocess"])
        out, index = channel.event
        kernels = fks_splitting_kernels(channel.born, channel.configurations)
        *_, k_perp = evaluate(
            kernels,
            out["collinear_momenta"],
            out["p_i_hat"],
            out["born_momenta"],
            fks_index=index,
        )
        blocks = fks_blocks(p, out, index, k_perp)
        function = fks_cross_section(
            channel.born, channel.configurations, cuts=channel.cuts(), n_f=channel.n_f
        )
        inputs = dict(channel.inputs)
        inputs.update(blocks)
        inputs.update({k: inputs[k] * v for k, v in channel.masks.items()})
        names = list(function.arg_types().keys())
        args = [inputs[n] for n in names if n != "fks_index"]
        ours = evaluate(function, *args, fks_index=index)
        ours = dict(zip(function.return_types().keys(), ours))
        theirs = channel.weights
        parts = sum(np.abs(theirs[key]) for key in WEIGHTS if key != "lo_weight")
        for key, value in theirs.items():
            scale = parts if key == "cross_section" else np.abs(value)
            assert (np.abs(ours[key] - value) <= WEIGHT_RTOL * scale).all(), key


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_couplings(path):
    """Without tagged photons the factors of the contributions are 1 and
    there are no tagged_factors, and the QED coupling of the counterterms is
    gal(1)^2 / (4 pi) of the model, 1 / aEWM1 of the param card."""
    data = reference(path)
    for channel in data["channels"]:
        p = provider(name_of(path), channel["subprocess"])
        tagged, alpha, factors = p.couplings([0, 2, 4])
        assert tagged == 0
        np.testing.assert_array_equal(factors, 1.0)
        aewm1 = data["param_card"]["sminputs"]["1"]
        assert alpha == pytest.approx(1 / aewm1, rel=1e-14)
        assert p.tagged_factors() is None


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_reference_scales(path):
    """The reference scale of the user is user_dynamical_scale of dummy_fct.f,
    by default mur_ref_fixed of the run card the output was compiled with
    (run_card.inc), at kinematics of the real process and of the Born, zero
    where the points are not valid."""
    data = reference(path)
    with open(os.path.join(OUTPUTS, name_of(path), "Source", "run_card.inc")) as f:
        line = next(line for line in f if "MUR_REF_FIXED" in line.upper())
    fixed = float(line.split("=")[1].strip().upper().replace("D", "E"))
    for channel in data["channels"]:
        p = provider(name_of(path), channel["subprocess"])
        points = points_with_blocks(channel)
        born = np.array([point["p_born"] for point in points])
        valid = np.arange(len(points)) % 3 != 0
        np.testing.assert_array_equal(
            p.reference_scales(born, valid), np.where(valid, fixed, 0.0)
        )
        events = [
            c["momenta"]
            for point in points
            for c in point["contributions"]
            if c["type"] == 1
        ]
        if events:
            np.testing.assert_array_equal(p.reference_scales(np.array(events)), fixed)


def test_extra_counterterms():
    """At the Borns of the reference of p p > a j (configurations with a
    quark pair from a gluon and from a photon, whose Born a u > u a is the
    extra counterterm), the Born inputs of the provider (the Born orders,
    then the Born of the extra counterterm of the configuration of the
    point) and their B_perp with k_perp of the point are those of pyAmpliCol
    (fks_testing.born_inputs) over g^b, and its counterterm parts of the
    extra counterterm are those of extra_cnt of MadFKS there."""
    directory = os.path.join(OUTPUTS, EXTRA_OUTPUT, "SubProcesses", EXTRA_SUBPROCESS)
    if not os.path.isdir(directory):
        pytest.skip(f"no output {EXTRA_OUTPUT}")
    data = reference(EXTRA_REFERENCE)
    p = provider(EXTRA_OUTPUT, EXTRA_SUBPROCESS)
    ours_configurations = p.fks_configurations
    theirs_configurations = data["fks_configurations"]
    for key in ("born_pdgs", "extra_counterterms", "orders"):
        assert ours_configurations[key] == theirs_configurations[key], key
    for ours, theirs in zip(
        ours_configurations["configurations"], theirs_configurations["configurations"]
    ):
        for key in ("i_fks", "j_fks", "real_pdgs", "splitting_type", "extra_cnt_index"):
            assert ours[key] == theirs[key], key
    rows = [
        (conf, point)
        for conf in data["configs"]
        for point in conf["points"]
        if "born_blocks" in point
    ]
    momenta = np.array([point["p_born"] for _, point in rows])
    k_perp = np.array([point["kperp"] for _, point in rows])
    fks_index = np.array([conf["nfks"] - 1 for conf, _ in rows])
    extra = fks_index[[p.extra_columns(c) is not None for c in fks_index]]
    assert len(extra) and len(extra) < len(fks_index)
    born, _, spin = born_inputs(data, rows, k_perp)
    orders = theirs_configurations["orders"]
    qcd = orders["split_orders"].index("QCD")
    powers = [o[qcd] for o in orders["born"]]
    powers += [o[qcd] for e in orders["extra_born"] for o in e]
    g = math.sqrt(4 * math.pi * data["born_blocks"]["alpha_s"])
    scale = g ** np.array(powers)
    assert p.tagged_factors() is None
    ours_born, ours_spin = p.born_inputs(momenta, k_perp, fks_index)
    assert ours_born.shape == born.shape == (len(rows), p.born_input_count())
    assert_close(ours_born * scale, born, EXTRA_RTOL)
    # pyAmpliCol has no B_perp for a quark emitter, the provider B / 2
    tensor = spin != 0
    assert tensor[:, -1].any()
    assert_close(np.where(tensor, ours_spin * scale, 0), spin, EXTRA_RTOL)
    np.testing.assert_allclose(ours_spin[~tensor], ours_born[~tensor] / 2, rtol=1e-15)
    # the counterterm parts of the extra counterterm per row of the split
    # orders and position, the coupling of the Born of the extra
    # counterterm divided out
    parts = p.extra(momenta, fks_index)
    has = np.array([p.extra_columns(c) is not None for c in fks_index])
    np.testing.assert_array_equal(parts[~has], 0)
    theirs = np.array([point["extra_born_orders"] for _, point in rows])[has]
    rows_of = [p.qcd_pos, p.qed_pos]
    powers = np.array(
        [
            [o[p.qcd_pos] - 2 * (row == p.qcd_pos) for o in p.positions]
            for row in rows_of
        ]
    )
    ours = parts[has][:, rows_of, 0].real * data["g"] ** powers
    assert_close(ours, theirs, EXTRA_RTOL)
