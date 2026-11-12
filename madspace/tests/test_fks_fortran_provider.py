"""The Fortran test provider (fks_fortran.py with fks_reference/fks_provider.f)
against the references of test_data/fks_integrand. At the points of the
references its blocks at g = 1 are those MadFKS took there, over their powers
of g: the real at the event, the Born and its azimuthal part, amp2 of the
Born diagrams, the colour links of the pairs sbornsoft takes and the virtual
of MadLoop. Its color-correlated and spin-correlated Borns are those that
test_fks_integrand builds from the blocks of MadFKS, the latter with the
phase of k_perp in place of xij_aor. At the madspace kinematics of these
points FKSCrossSection with its blocks gives the weights it gives with those
of MadFKS.

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
from fks_fortran import FortranProvider, azimuthal_phase, fks_blocks
from fks_testing import evaluate, reference
from test_fks_integrand import DATA, IDS, WEIGHTS, channels

from madspace.fks import fks_cross_section, fks_splitting_kernels

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
