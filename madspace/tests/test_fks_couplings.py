"""The scales, the strong coupling and the parton luminosities of the FKS
integrand at each kinematics, point by point against sigint of driver_mintFO.f
(test_data/fks_integrand, see test_fks_integrand.py for the kinematics).

sigint takes the scales of setscales.f at the event for the real emission, at
the soft counter-event (the Born and a soft parton without momentum) for the
soft and soft-collinear counterterms and the n-body part, and at the
collinear counter-event for the collinear counterterm. include_PDF_and_alphas
then multiplies each weight line by g_strong at its scales and by the
luminosity dlum at its momentum fractions and its factorization scale (that
of the first beam for both), the sum of PD = f_1 f_2 over the flavors of the
real matrix element of its configuration (of the Born for the n-body part).

The references run with the built-in NNPDF2.3 NLO of MadFKS (nn23nlo):
NNevolvePDF of NNPDFDriver.f on its grid, polynomial in log x (in x above
0.1) through four nodes and linear in log Q^2, and the 2-loop running of
alfas_functions.f from alpha_s(m_Z) = 0.119. Their ports below give the
couplings and the densities of each weight line of MadFKS from the scales and
the momentum fractions of madspace. ALPHAS of MadFKS interpolates the running
linearly on a grid of 256 nodes per region (alphas_from_grids), within 1e-5
of it. madspace itself reads LHAPDF grids, here the same nodes of NNPDF in
that format, which it interpolates cubically, and alpha_s from a dense grid of
the 2-loop running, both as the MG7 NLO driver writes them
(madgraph/mg7/nlo/grids.py).
"""

import functools
import math
import os

import numpy as np
import pytest
from fks_testing import evaluate, reference
from madgraph.mg7.nlo.grids import (
    COLUMNS,
    NNPDF_GRID,
    PIDS,
    alpha_s_2loop,
    write_alpha_s_grid,
    write_lhapdf_grid,
)
from madspace_testing import default_context
from test_fks_integrand import (
    DATA,
    IDS,
    WEIGHTS,
    channels,
    scale_coefficient,
)

import madspace as ms
from madspace.nlo import nlo_scale
from madspace.nlo.fks import (
    fks_born_flavor_indices,
    fks_cross_section,
    fks_luminosities,
    fks_soft_virtual,
    underlying_born,
)

# the kinematics of the weight lines of add_wgt: the event, the Born (the soft
# counter-event of MadFKS) and the counter-events
KINEMATICS = {
    1: "event",
    2: "born",
    3: "born",
    14: "born",
    15: "born",
    4: "soft",
    5: "collinear",
    6: "soft_collinear",
}
# the kinematics of the scales of each kinematics
SCALES_OF = {
    "event": "event",
    "born": "born",
    "soft": "born",
    "soft_collinear": "born",
    "collinear": "collinear",
}
# the momentum fractions of each kinematics among the outputs of FKSKinematics
FRACTIONS = {
    "event": ("x1", "x2"),
    "born": ("born_x1", "born_x2"),
    "soft": ("soft_x1", "soft_x2"),
    "collinear": ("collinear_x1", "collinear_x2"),
    "soft_collinear": ("soft_collinear_x1", "soft_collinear_x2"),
}
# the Born and its counter-events come back through the boost into the lab
# and the round trip of the Born mapping (see test_fks_integrand.py)
RTOL = 1e-10
# The cubic interpolation of madspace and the polynomial one of NNPDFDriver.f
# on the same nodes give luminosities within 2.4e-3 of each other at the
# points of the references with momentum fractions below 0.3 and scales above
# the first interval of the grid (2 GeV), and within 4e-2 up to x = 0.95,
# where the densities fall towards x = 1, and at the bottom of the grid, where
# both freeze (a scale of mu_F = H_T/4 with the floor of 2 GeV on H_T/2). At
# x = 1 MadFKS has no densities, madspace their rounding.
INTERPOLATION_RTOL = 3e-3
LARGE_X, LOW_SCALE, EDGE_RTOL = 0.3, 2.0, 5e-2


def interpolation_rtol(fractions, scales):
    """The tolerance of the interpolation at the largest momentum fraction and
    the smallest scale of a point."""
    edge = (np.max(np.stack(fractions), axis=0) >= LARGE_X) | (
        np.min(np.stack(scales), axis=0) < LOW_SCALE
    )
    return np.where(edge, EDGE_RTOL, INTERPOLATION_RTOL)


def assert_within_interpolation(ours, theirs, rtol):
    assert (np.abs(ours - theirs) <= rtol * np.abs(theirs) + 1e-15).all()


needs_nnpdf = pytest.mark.skipif(
    not os.path.isfile(NNPDF_GRID), reason="needs the NNPDF grid of MadGraph"
)


class NNPDF:
    """NNevolvePDF of NNPDFDriver.f on its grid (member 0) and fdist of pdf.f."""

    def __init__(self, path):
        with open(path) as f:
            lines = [line.strip() for line in f]
        k = next(i for i, line in enumerate(lines) if "NNPDF20int" in line) + 2
        nx = int(lines[k])
        self.x = np.array([float(v) for v in lines[k + 1 : k + 1 + nx]])
        k += 1 + nx
        nq2 = int(lines[k])
        # the record after nq2 is read and dropped, as the one after the grid
        self.q2 = np.array([float(v) for v in lines[k + 2 : k + 2 + nq2]])
        k += 3 + nq2
        rows = [[float(v) for v in line.split()] for line in lines[k : k + nx * nq2]]
        # (column, ix, iq)
        self.values = np.array(rows).reshape(nx, nq2, -1).transpose(2, 0, 1)

    @staticmethod
    def polint(xa, ya, x):
        """lh_polint (Neville) on each row of xa and ya, at x, with ns
        1-based as the Fortran."""
        count, n = xa.shape
        rows = np.arange(count)
        dif = np.abs(x - xa[:, 0])
        ns = np.ones(count, dtype=int)
        for i in range(n):
            dift = np.abs(x - xa[:, i])
            closer = dift < dif
            ns = np.where(closer, i + 1, ns)
            dif = np.where(closer, dift, dif)
        c, d = ya.copy(), ya.copy()
        y = ya[rows, ns - 1]
        ns = ns - 1
        for m in range(1, n):
            for i in range(n - m):
                ho = xa[:, i] - x
                hp = xa[:, i + m] - x
                w = c[:, i + 1] - d[:, i]
                den = w / (ho - hp)
                d[:, i] = hp * den
                c[:, i] = ho * den
            upper = 2 * ns < n - m
            dy = np.where(
                upper,
                c[rows, np.clip(ns, 0, n - 1)],
                d[rows, np.clip(ns - 1, 0, n - 1)],
            )
            ns = np.where(upper, ns, ns - 1)
            y = y + dy
        return y

    def density(self, pid, x, q):
        """f(x, Q) of the flavor pid, xpdf / x as pdg2pdf, zero for a flavor
        that is not in the grid."""
        x, q = np.asarray(x, dtype=float), np.asarray(q, dtype=float)
        if pid not in COLUMNS:
            return np.zeros_like(x)
        x = np.clip(x, self.x[0], self.x[-1])
        q2 = np.clip(q * q, self.q2[0], self.q2[-1])
        nx, nq2 = len(self.x), len(self.q2)
        ix = np.searchsorted(self.x, x, side="right")
        iq = np.searchsorted(self.q2, q2, side="right")
        # the first nodes, 1-based as IX1A and IX2A
        ix1 = np.where(ix < 2, 1, np.where(ix > nx - 2, nx - 3, ix - 1))
        iq1 = np.where(iq > nq2 - 1, nq2 - 1, iq)
        ixs = ix1[:, None] + np.arange(4) - 1
        iqs = iq1[:, None] + np.arange(2) - 1
        linear = x >= 0.1
        x1 = np.where(linear, x, np.log(x))
        x1a = np.where(linear[:, None], self.x[ixs], np.log(self.x)[ixs])
        x2a = np.log(self.q2)[iqs]
        grid = self.values[COLUMNS[pid] - 1]
        nodes = [
            self.polint(x2a, grid[ixs[:, j][:, None], iqs], np.log(q2))
            for j in range(4)
        ]
        return self.polint(x1a, np.stack(nodes, axis=1), x1) / x


@functools.cache
def alpha_s_tables(asmz):
    """The tables of alphas_from_grids: 256 nodes in log Q from 0.5 GeV to the c
    threshold, from there to the b threshold and from there to 100 TeV, with
    the slope of each interval."""
    edges = [math.log(0.5), math.log(1.42), math.log(4.7), math.log(100000.0)]
    tables = []
    for low, high in zip(edges, edges[1:]):
        q = [low + (high - low) * float(i) / 255.0 for i in range(256)]
        values = [alpha_s_2loop(asmz, math.exp(v)) for v in q]
        slopes = [
            (b - a) / (q2 - q1) for a, b, q1, q2 in zip(values, values[1:], q, q[1:])
        ]
        tables.append((low, high, q, values, slopes))
    return tables


def alpha_s_madfks(asmz, q):
    """ALPHAS of alfas_functions.f: alphas_from_grids, the linear
    interpolation in log Q of the 2-loop running between 0.5 GeV and 100
    TeV."""
    log_q = math.log(q)
    for low, high, nodes, values, slopes in alpha_s_tables(asmz):
        if low <= log_q < high:
            i = int((log_q - low) / (high - low) * 255)
            return values[i] + (log_q - nodes[i]) * slopes[i]
    return alpha_s_2loop(asmz, q)


@functools.cache
def nnpdf():
    return NNPDF(NNPDF_GRID)


# the globals of PdfGrid and AlphaSGrid in the context
GLOBALS = (
    "pdf_logx",
    "pdf_logq2",
    "pdf_coefficients",
    "alpha_s_logq2",
    "alpha_s_coefficients",
)


def delete_globals(context):
    for name in GLOBALS:
        if context.global_exists(name):
            context.delete_global(name)


@pytest.fixture(scope="module")
def grids(tmp_path_factory):
    """The PdfGrid of the NNPDF nodes and the AlphaSGrid of the 2-loop running
    from an alpha_s(m_Z), uploaded to the context at each call, in place of
    those another test may have left there."""
    directory = tmp_path_factory.mktemp("fks_couplings")
    pdf_path = str(directory / "NNPDF23nlo_as_0119_qed_0000.dat")
    write_lhapdf_grid(pdf_path)
    pdf = ms.PdfGrid(pdf_path)
    alpha_s = {}

    def upload(asmz):
        if asmz not in alpha_s:
            path = str(directory / f"alpha_s_{asmz}.info")
            write_alpha_s_grid(path, asmz)
            alpha_s[asmz] = ms.AlphaSGrid(path)
        context = default_context()
        delete_globals(context)
        pdf.initialize_globals(context)
        alpha_s[asmz].initialize_globals(context)
        return pdf, alpha_s[asmz]

    yield upload
    delete_globals(default_context())


@functools.cache
def asmz(path):
    """alpha_s(m_Z) of MadFKS, that of nn23nlo (pdfwrap) with PDFs and of the
    param card without, which has both."""
    return float(reference(path)["param_card"]["sminputs"]["3"])


def beams(channel):
    return (int(channel.run_card["lpp1"]), int(channel.run_card["lpp2"]))


def scale_parameters(run):
    """The scales of the run card by the names of nlo_scale."""
    flag = {"True": True, "False": False}
    return dict(
        dynamical_scale_choice=int(run["dynamical_scale_choice"]),
        fixed_ren_scale=flag[run["fixed_ren_scale"]],
        fixed_fac_scale=flag[run["fixed_fac_scale"]],
        mur_ref_fixed=float(run["mur_ref_fixed"]),
        muf_ref_fixed=float(run["muf_ref_fixed"]),
        mur_over_ref=float(run["mur_over_ref"]),
        muf_over_ref=float(run["muf_over_ref"]),
    )


@functools.cache
def madspace_scales(channel):
    """The outputs of NLOScale at the event, the Born and the collinear
    counter-event."""
    out, _ = channel.event
    configurations = channel.configurations
    scale = nlo_scale(
        len(configurations["born_pdgs"]) + 1,
        configurations.get("nincoming", 2),
        **scale_parameters(channel.run_card),
    )
    born = scale.with_particle_count(len(channel.configurations["born_pdgs"]))
    return {
        "event": evaluate(scale, out["momenta"]),
        "born": evaluate(born, out["born_momenta"]),
        "collinear": evaluate(scale, out["collinear_momenta"]),
    }


def flavors(channel, contribution, kinematics):
    """The incoming partons of the flavors that the luminosity of a weight
    line sums over."""
    configurations = channel.configurations
    if kinematics == "born":
        return [f[:2] for f in configurations["born_flavors"]]
    index = contribution["nfks"] - 1
    real_me = configurations["configurations"][index]["real_me"]
    return [f[:2] for f in configurations["real_flavors"][real_me - 1]]


def weight_lines(path):
    """Each weight line of the references with its channel, point and
    kinematics."""
    for channel in channels(path):
        for k, point in enumerate(channel.points):
            for c in point["contributions"]:
                if c["type"] in KINEMATICS:
                    yield channel, k, c, KINEMATICS[c["type"]]


def densities(channel, kinematics):
    """The momentum fractions and factorization scales of madspace at a
    kinematics."""
    out, _ = channel.event
    _, fact1, fact2 = madspace_scales(channel)[SCALES_OF[kinematics]]
    x1, x2 = (out[name] for name in FRACTIONS[kinematics])
    return x1, x2, fact1, fact2


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_scales(path):
    """NLOScale gives mu_R and mu_F of each weight line at its kinematics, and
    the Ellis-Sexton scale of MadFKS is mu_R. A decay has the mass of the
    decaying particle for mu_R, and for mu_F, which enters nothing (MadFKS
    takes H_T/2 without the first outgoing particle there)."""
    lines = 0
    for channel, k, c, kinematics in weight_lines(path):
        ren, fact1, fact2 = madspace_scales(channel)[SCALES_OF[kinematics]]
        q2, mu_r2, mu_f2 = c["scales2"]
        np.testing.assert_allclose(ren[k] ** 2, mu_r2, rtol=RTOL)
        if channel.incoming == 1:
            run = channel.run_card
            ratio = float(run["muf_over_ref"]) / float(run["mur_over_ref"])
            mu_f2 = mu_r2 * ratio**2
        np.testing.assert_allclose(fact1[k] ** 2, mu_f2, rtol=RTOL)
        np.testing.assert_allclose(fact2[k] ** 2, mu_f2, rtol=RTOL)
        np.testing.assert_allclose(q2, mu_r2, rtol=1e-15)
        lines += 1
    assert lines > 0


@needs_nnpdf
@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_strong_coupling(path, grids):
    """The 2-loop running of MadFKS (linear on its grid) at mu_R of madspace
    is g_strong of each weight line, and RunningCoupling on its grid is that
    running."""
    _, grid = grids(asmz(path))
    running = ms.RunningCoupling(grid)
    for channel, k, c, kinematics in weight_lines(path):
        ren = madspace_scales(channel)[SCALES_OF[kinematics]][0][k]
        np.testing.assert_allclose(
            alpha_s_madfks(asmz(path), ren),
            c["g_strong"] ** 2 / (4 * math.pi),
            rtol=RTOL,
        )
    for channel in channels(path):
        for ren, _, _ in madspace_scales(channel).values():
            (ours,) = evaluate(running, ren)
            expected = [alpha_s_2loop(asmz(path), q) for q in ren]
            np.testing.assert_allclose(ours, expected, rtol=1e-7)


def test_strong_coupling_at_low_scales(grids):
    """RunningCoupling has the running down to 0.5 GeV, where the grids of
    ALPHAS of MadFKS begin, and so at 1 GeV, the floor of the scales (2 GeV)
    times a factor 1/2 of a scale variation: below its first node madspace
    takes alpha_s as zero, which once cut out that region of a run."""
    _, grid = grids(0.119)
    scales = np.array([0.51, 0.7, 1.0, 1.42, 2.0])
    (ours,) = evaluate(ms.RunningCoupling(grid), scales)
    expected = [alpha_s_madfks(0.119, q) for q in scales]
    np.testing.assert_allclose(ours, expected, rtol=2e-4)
    np.testing.assert_allclose(
        ours, [alpha_s_2loop(0.119, q) for q in scales], rtol=1e-5
    )


@needs_nnpdf
@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_luminosities(path):
    """PD of each flavor of each weight line, f_1 f_2 of NNPDF at the momentum
    fractions and the factorization scales of madspace, over the flavors of
    born_flavors and real_flavors, in their order."""
    pdf = nnpdf()
    lines = 0
    for channel, k, c, kinematics in weight_lines(path):
        pairs = flavors(channel, c, kinematics)
        assert len(pairs) == c["niproc"]
        bare = scale_coefficient(c) * c["g_strong"] ** c["qcd_power"]
        if bare == 0:
            continue
        # PD of MadFKS, from parton_iproc = PD conv wgt_wo_pdf, to the
        # rounding of the scale logs of wgt_wo_pdf
        w1, w2, w3 = c["wgt"]
        q2, mu_r2, mu_f2 = c["scales2"]
        size = abs(w1) + abs(w2 * math.log(mu_r2 / q2)) + abs(w3 * math.log(mu_f2 / q2))
        theirs = np.array(c["parton_iproc"]) / (channel.madfks_conv * bare)
        x1, x2, fact1, _ = (v[k : k + 1] for v in densities(channel, kinematics))
        if int(channel.run_card["lpp1"]) == 0:
            ours = np.ones(len(pairs))
        else:
            # the factorization scale of the first beam for both, as sigint
            ours = np.array(
                [
                    pdf.density(a, x1, fact1)[0] * pdf.density(b, x2, fact1)[0]
                    for a, b in pairs
                ]
            )
        tolerance = RTOL * abs(ours).max() * size / abs(scale_coefficient(c))
        np.testing.assert_allclose(ours, theirs, rtol=RTOL, atol=tolerance)
        lines += 1
    assert lines > 0


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_flavors(path):
    """The flavors of each weight line (parton_pdg of set_pdg_codes) are the
    real flavors of its configuration, in the order of real_flavors, and
    their underlying Borns (parton_pdg_uborn) those of underlying_born, which
    for the n-body part are the flavors of the Born."""
    lines = 0
    for channel, _, c, kinematics in weight_lines(path):
        configurations = channel.configurations
        configuration = configurations["configurations"][c["nfks"] - 1]
        reals = configurations["real_flavors"][configuration["real_me"] - 1]
        assert c["parton_pdg"] == reals
        borns = [underlying_born(flavor, configuration) for flavor in reals]
        assert c["parton_pdg_uborn"] == borns
        if kinematics == "born":
            assert borns == configurations["born_flavors"]
        lines += 1
    assert lines > 0
    for channel in channels(path):
        indices = fks_born_flavor_indices(channel.configurations)
        born_flavors = channel.configurations["born_flavors"]
        for k, configuration in enumerate(channel.configurations["configurations"]):
            reals = channel.configurations["real_flavors"][configuration["real_me"] - 1]
            assert [born_flavors[b] for b in indices[k]] == [
                underlying_born(flavor, configuration) for flavor in reals
            ]


def channel_pids(channel):
    """The incoming partons of all flavors of a channel."""
    configurations = channel.configurations
    flavors_ = [configurations["born_flavors"]] + configurations["real_flavors"]
    return sorted({p for flavor in sum(flavors_, []) for p in flavor[:2]})


@needs_nnpdf
@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_luminosity_function(path, grids):
    """NLOLuminosity of the real emission at each of its kinematics and of the
    n-body part, for each flavor and summed, is f_1 f_2 of PartonDensity, and
    within the interpolation the NNPDF luminosity of MadFKS. With a flavor
    index the n-body part takes the underlying Born of the real flavor over
    the number of real flavors with that Born."""
    pdf_grid, _ = grids(asmz(path))
    for channel in channels(path):
        hadrons = beams(channel) != (0, 0)
        grid = pdf_grid if hadrons else None
        summed = fks_luminosities(
            channel.configurations, grid, beams=beams(channel), sum_flavors=True
        )
        flavored = fks_luminosities(channel.configurations, grid, beams=beams(channel))
        born_indices = fks_born_flavor_indices(channel.configurations)
        born_flavors = channel.configurations["born_flavors"]
        _, index = channel.event
        pids = [p for p in channel_pids(channel) if p in PIDS]
        density = ms.PartonDensity(pdf_grid, pids)
        for kinematics in FRACTIONS:
            x1, x2, fact1, fact2 = densities(channel, kinematics)
            nbody = kinematics == "born"
            f1 = f2 = n1 = n2 = None
            if hadrons:
                # within the grid, as NLOLuminosity
                q_range = (pdf_grid.q[0], pdf_grid.q[-1])
                (xf1,) = evaluate(density, x1, np.clip(fact1, *q_range))
                (xf2,) = evaluate(density, x2, np.clip(fact2, *q_range))
                f1 = dict(zip(pids, (xf1 / x1[:, None]).T))
                f2 = dict(zip(pids, (xf2 / x2[:, None]).T))
                n1 = {p: nnpdf().density(p, x1, fact1) for p in pids}
                n2 = {p: nnpdf().density(p, x2, fact2) for p in pids}

            def product(densities1, densities2, pair, k):
                if not hadrons:
                    return 1.0
                a, b = pair
                if a not in pids or b not in pids:
                    return 0.0
                return densities1[a][k] * densities2[b][k]

            # the sum over the flavors
            function = summed[1] if nbody else summed[0]
            fks_index = np.zeros_like(index) if nbody else index
            (ours,) = evaluate(function, x1, x2, fact1, fact2, fks_index=fks_index)
            expected, madfks = np.zeros(len(x1)), np.zeros(len(x1))
            for k in range(len(x1)):
                pairs = flavors(channel, dict(nfks=index[k] + 1), kinematics)
                expected[k] = sum(product(f1, f2, pair, k) for pair in pairs)
                madfks[k] = sum(product(n1, n2, pair, k) for pair in pairs)
            np.testing.assert_allclose(ours, expected, rtol=1e-14)
            rtol = interpolation_rtol([x1, x2], [fact1, fact2])
            assert_within_interpolation(ours, madfks, rtol)
            # each flavor of the event, the underlying Born for the n-body part
            function = flavored[1] if nbody else flavored[0]
            count = max(len(r) for r in channel.configurations["real_flavors"])
            for r in range(count + 1):
                flavor_index = np.full(len(index), r)
                (ours,) = evaluate(
                    function,
                    x1,
                    x2,
                    fact1,
                    fact2,
                    flavor_index=flavor_index,
                    fks_index=index,
                )
                expected = np.zeros(len(x1))
                for k in range(len(x1)):
                    reals = flavors(channel, dict(nfks=index[k] + 1), "event")
                    if r >= len(reals):
                        continue
                    if nbody:
                        b = born_indices[index[k]][r]
                        share = born_indices[index[k]].count(b)
                        pair = born_flavors[b][:2]
                        expected[k] = product(f1, f2, pair, k) / share
                    else:
                        expected[k] = product(f1, f2, reals[r][:2], k)
                np.testing.assert_allclose(ours, expected, rtol=1e-14, atol=0)


def run_with_couplings(channel, function, flavor_index=None):
    """The outputs of an FKSCrossSection with the couplings inside, on the
    inputs of the integrand test, the momentum fractions of madspace and a
    flavor index."""
    out, index = channel.event
    inputs = dict(channel.inputs)
    inputs.update({name: out[name] for names in FRACTIONS.values() for name in names})
    names = [
        name
        for name in function.arg_types().keys()
        if name not in ("flavor_index", "fks_index")
    ]
    outputs = evaluate(
        function,
        *[inputs[name] for name in names],
        flavor_index=flavor_index,
        fks_index=index,
    )
    return dict(zip(function.return_types().keys(), outputs))


@needs_nnpdf
@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_cross_section_with_couplings(path, grids):
    """FKSCrossSection with NLOScale, RunningCoupling and NLOLuminosity inside
    is FKSCrossSection with their outputs as arguments, and within the
    interpolation of the PDFs the weights of MadFKS."""
    pdf_grid, alpha_s_grid = grids(asmz(path))
    running = ms.RunningCoupling(alpha_s_grid)
    for channel in channels(path):
        hadrons = beams(channel) != (0, 0)
        parameters = dict(n_f=channel.n_f, cuts=channel.cuts())
        inside = fks_cross_section(
            channel.born,
            channel.configurations,
            scales=scale_parameters(channel.run_card),
            running_coupling=running,
            pdf_grid=pdf_grid if hadrons else None,
            beams=beams(channel),
            sum_flavors=True,
            **parameters,
        )
        ours = run_with_couplings(channel, inside)
        # the same from the pieces
        out, index = channel.event
        scales = madspace_scales(channel)
        inputs = dict(channel.inputs)
        ren, fact1, fact2 = scales["born"]
        _, coll1, coll2 = scales["collinear"]
        inputs.update(
            ren_scale=ren,
            fact_scale1=fact1,
            fact_scale2=fact2,
            collinear_fact_scale1=coll1,
            collinear_fact_scale2=coll2,
        )
        for name in ("event", "born", "collinear"):
            (inputs[f"{name}_alpha_s"],) = evaluate(running, scales[name][0])
        real, born = fks_luminosities(
            channel.configurations,
            pdf_grid if hadrons else None,
            beams=beams(channel),
            sum_flavors=True,
        )
        for kinematics in FRACTIONS:
            x1, x2, f1, f2 = densities(channel, kinematics)
            function, fks_index = (
                (born, np.zeros_like(index)) if kinematics == "born" else (real, index)
            )
            (inputs[f"{kinematics}_luminosity"],) = evaluate(
                function, x1, x2, f1, f2, fks_index=fks_index
            )
        outside = fks_cross_section(channel.born, channel.configurations, **parameters)
        names = [name for name in outside.arg_types().keys() if name != "fks_index"]
        theirs = evaluate(outside, *[inputs[name] for name in names], fks_index=index)
        for name, value in zip(outside.return_types().keys(), theirs):
            np.testing.assert_allclose(ours[name], value, rtol=1e-13, atol=0)
        # each contribution with the couplings and luminosities of MadFKS
        # (see test_weights of test_fks_integrand.py)
        fractions = [out[name] for names in FRACTIONS.values() for name in names]
        factorization = [scale for _, *fact in scales.values() for scale in fact]
        rtol = interpolation_rtol(fractions, factorization)
        for name in WEIGHTS:
            assert_within_interpolation(ours[name], channel.weights[name], rtol)


@needs_nnpdf
@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_cross_section_per_flavor(path, grids):
    """FKSCrossSection with a flavor index, summed over the flavor of each
    event, is FKSCrossSection summed over the flavors, and each flavor gives
    the weights of its own luminosities. A batch mixes configurations, and
    flavors beyond those of a configuration give zero."""
    pdf_grid, alpha_s_grid = grids(asmz(path))
    running = ms.RunningCoupling(alpha_s_grid)
    for channel in channels(path):
        hadrons = beams(channel) != (0, 0)
        options = dict(
            scales=scale_parameters(channel.run_card),
            running_coupling=running,
            pdf_grid=pdf_grid if hadrons else None,
            beams=beams(channel),
            n_f=channel.n_f,
            cuts=channel.cuts(),
        )
        summed = fks_cross_section(
            channel.born, channel.configurations, sum_flavors=True, **options
        )
        flavored = fks_cross_section(channel.born, channel.configurations, **options)
        assert "flavor_index" in flavored.arg_types().keys()
        assert "flavor_index" not in summed.arg_types().keys()
        expected = run_with_couplings(channel, summed)
        _, index = channel.event
        count = max(len(r) for r in channel.configurations["real_flavors"])
        total = {name: 0.0 for name in expected}
        for r in range(count + 1):
            ours = run_with_couplings(channel, flavored, np.full(len(index), r))
            reals = np.array(
                [len(flavors(channel, dict(nfks=k + 1), "event")) for k in index]
            )
            for name, value in ours.items():
                assert (value[r >= reals] == 0).all()
                total[name] = total[name] + value
        for name, value in expected.items():
            # the sum over the flavors rounds differently, to 1e-12 of the
            # largest weight of the cross section with its cancellations (a
            # massive emitter alone has no collinear weights)
            scale = max(np.abs(value).max(), np.finfo(float).tiny)
            np.testing.assert_allclose(
                total[name] / scale, value / scale, rtol=0, atol=1e-12
            )


def setscales(p, choice, fixed_ren, fixed_fac, ref_fixed, over_ref):
    """mu_R and mu_F of setscales.f at momenta p (batch, n, 4) in their
    partonic center-of-mass frame."""
    final = p[:, 2:]
    if choice == 1:
        pt = np.hypot(final[..., 1], final[..., 2])
        safe = np.where(pt > 0, pt, 1.0)
        et = final[..., 0] * pt / np.sqrt(safe**2 + final[..., 3] ** 2)
        reference = np.where(pt > 0, et, 0.0).sum(-1)
    elif choice == -2:
        reference = np.full(len(p), ref_fixed[0])
    else:
        mt2 = (final[..., 0] + final[..., 3]) * (final[..., 0] - final[..., 3])
        reference = np.sqrt(np.maximum(mt2, 0.0)).sum(-1)
        reference = reference if choice == 2 else reference / 2
    dynamical = np.maximum(ms.NLOScale.min_scale, reference)
    scales = [
        np.full(len(p), fixed) if is_fixed else dynamical
        for fixed, is_fixed in zip(
            ref_fixed, (fixed_ren, fixed_fac, fixed_fac), strict=True
        )
    ]
    return [factor * scale for factor, scale in zip(over_ref, scales, strict=True)]


def random_event(rng, count, n, soft):
    """Massless momenta of count events of n particles in their partonic
    center-of-mass frame, the first outgoing one without momentum where soft,
    and the same boosted along z."""
    outgoing = rng.normal(size=(count, n - 3, 3)) * 50
    if soft:
        outgoing[:, 0] = 0
    outgoing = np.concatenate([outgoing, -outgoing.sum(1, keepdims=True)], axis=1)
    energies = np.linalg.norm(outgoing, axis=-1)
    rs = energies.sum(-1)
    p = np.zeros((count, n, 4))
    p[:, :2, 0] = rs[:, None] / 2
    p[:, 0, 3], p[:, 1, 3] = rs / 2, -rs / 2
    p[:, 2:, 0] = energies
    p[:, 2:, 1:] = outgoing
    rapidity = rng.uniform(-2, 2, size=count)
    cosh, sinh = np.cosh(rapidity)[:, None], np.sinh(rapidity)[:, None]
    lab = p.copy()
    lab[..., 0] = cosh * p[..., 0] + sinh * p[..., 3]
    lab[..., 3] = sinh * p[..., 0] + cosh * p[..., 3]
    return lab, p


SCALE_CASES = {
    "half_mt": dict(choice=-1),
    "half_mt_3": dict(choice=3),
    "mt": dict(choice=2),
    "et": dict(choice=1),
    "fixed_reference": dict(choice=-2),
    "fixed_ren": dict(choice=-1, fixed_ren=True),
    "fixed_fac": dict(choice=1, fixed_fac=True),
    "factors": dict(choice=-1, over_ref=(2.0, 0.5, 0.25)),
}


@pytest.mark.parametrize("case", SCALE_CASES.values(), ids=SCALE_CASES.keys())
@pytest.mark.parametrize("soft", [False, True], ids=["hard", "soft"])
def test_scale_choices(case, soft):
    """NLOScale is setscales.f on random events in the lab (the transverse
    energies in the partonic frame), also with a parton without momentum,
    and the floor holds for soft events."""
    rng = np.random.default_rng(7)
    lab, partonic = random_event(rng, 200, 6, soft)
    # a few events below the floor
    lab[:5] *= 1e-3
    partonic[:5] *= 1e-3
    options = dict(
        choice=case["choice"],
        fixed_ren=case.get("fixed_ren", False),
        fixed_fac=case.get("fixed_fac", False),
        ref_fixed=(91.118, 125.0, 80.0),
        over_ref=case.get("over_ref", (1.0, 1.0, 1.0)),
    )
    scale = ms.NLOScale(
        6,
        dynamical_scale_choice=options["choice"],
        fixed_ren_scale=options["fixed_ren"],
        fixed_fac_scale=options["fixed_fac"],
        mur_ref_fixed=91.118,
        muf1_ref_fixed=125.0,
        muf2_ref_fixed=80.0,
        mur_over_ref=options["over_ref"][0],
        muf1_over_ref=options["over_ref"][1],
        muf2_over_ref=options["over_ref"][2],
    )
    ours = evaluate(scale, lab)
    expected = setscales(partonic, **options)
    for value, reference_value in zip(ours, expected, strict=True):
        assert np.isfinite(value).all()
        np.testing.assert_allclose(value, reference_value, rtol=1e-12)
    if not options["fixed_ren"] and options["choice"] != -2:
        assert (ours[0][:5] == options["over_ref"][0] * ms.NLOScale.min_scale).all()


@pytest.mark.parametrize("case", SCALE_CASES.values(), ids=SCALE_CASES.keys())
def test_decay_scales(case):
    """For a decay the reference scale of every choice is the mass of the
    decaying particle, at rest or boosted, raised to the floor, a fixed scale
    replaces it and each scale takes its factor, as setscales.f for
    nincoming = 1."""
    rng = np.random.default_rng(11)
    mass = np.concatenate([[1.0], rng.uniform(10.0, 200.0, 63)])
    momentum = rng.normal(size=(64, 3)) * 30
    p = np.zeros((64, 4, 4))
    p[:, 0, 0] = np.sqrt(mass**2 + (momentum**2).sum(-1))
    p[:, 0, 1:] = momentum
    p[:, 1:] = rng.normal(size=(64, 3, 4))
    over_ref = case.get("over_ref", (1.0, 1.0, 1.0))
    scale = ms.NLOScale(
        4,
        dynamical_scale_choice=case["choice"],
        fixed_ren_scale=case.get("fixed_ren", False),
        fixed_fac_scale=case.get("fixed_fac", False),
        mur_ref_fixed=91.118,
        muf1_ref_fixed=125.0,
        muf2_ref_fixed=80.0,
        mur_over_ref=over_ref[0],
        muf1_over_ref=over_ref[1],
        muf2_over_ref=over_ref[2],
        incoming=1,
    )
    assert scale.incoming() == 1
    assert scale.with_particle_count(3).incoming() == 1
    dynamical = np.maximum(ms.NLOScale.min_scale, mass)
    fixed = (case.get("fixed_ren", False), case.get("fixed_fac", False))
    expected = [
        over_ref[0] * (91.118 if fixed[0] else dynamical),
        over_ref[1] * (125.0 if fixed[1] else dynamical),
        over_ref[2] * (80.0 if fixed[1] else dynamical),
    ]
    for ours, theirs in zip(evaluate(scale, p), expected, strict=True):
        np.testing.assert_allclose(ours, theirs, rtol=1e-12)


@pytest.mark.parametrize("choice", [0, 10])
def test_user_scale(choice):
    """With the scale of the user (dynamical_scale_choice 0 or 10) the
    reference scale is an argument, raised to the floor, a fixed scale
    replaces it and each scale takes its factor, a decay takes the mass."""
    rng = np.random.default_rng(13)
    lab, _ = random_event(rng, 64, 5, False)
    reference = np.concatenate([[0.5], rng.uniform(1.0, 500.0, 63)])
    scale = ms.NLOScale(
        5,
        dynamical_scale_choice=choice,
        fixed_fac_scale=True,
        muf1_ref_fixed=125.0,
        muf2_ref_fixed=80.0,
        mur_over_ref=2.0,
        muf2_over_ref=0.5,
    )
    assert scale.user()
    assert list(scale.arg_types().keys()) == ["momenta", "reference_scale"]
    assert scale.with_particle_count(4).user()
    ren, fact1, fact2 = evaluate(scale, lab, reference)
    np.testing.assert_allclose(
        ren, 2 * np.maximum(ms.NLOScale.min_scale, reference), rtol=1e-15
    )
    np.testing.assert_array_equal(fact1, 125.0)
    np.testing.assert_array_equal(fact2, 40.0)
    assert not ms.NLOScale(5).user()
    decay = ms.NLOScale(4, dynamical_scale_choice=choice, incoming=1)
    p = np.zeros((2, 4, 4))
    p[:, 0, 0] = [173.0, 80.0]
    ren, _, _ = evaluate(decay, p, np.array([10.0, 20.0]))
    np.testing.assert_allclose(ren, [173.0, 80.0], rtol=1e-15)


@needs_nnpdf
@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_cross_section_with_user_scale(path, grids):
    """FKSCrossSection with the scale of the user takes the reference scales
    of the event, of the soft counter-event (for the Born) and of the
    collinear counter-event as arguments: with those of the scale choice of
    the run it gives what it gives with that choice."""
    pdf_grid, alpha_s_grid = grids(asmz(path))
    running = ms.RunningCoupling(alpha_s_grid)
    for channel in channels(path):
        hadrons = beams(channel) != (0, 0)
        scales = scale_parameters(channel.run_card)
        common = dict(
            running_coupling=running,
            pdf_grid=pdf_grid if hadrons else None,
            beams=beams(channel),
            sum_flavors=True,
            n_f=channel.n_f,
            cuts=channel.cuts(),
        )
        born, configurations = channel.born, channel.configurations
        theirs = run_with_couplings(
            channel, fks_cross_section(born, configurations, scales=scales, **common)
        )
        user = fks_cross_section(
            born,
            configurations,
            scales=dict(scales, dynamical_scale_choice=10),
            **common,
        )
        names = list(user.arg_types().keys())
        assert names[-4:] == [
            "reference_scale",
            "soft_reference_scale",
            "collinear_reference_scale",
            "fks_index",
        ]
        # the reference scales of the choice of the run, raised to the floor
        reference = nlo_scale(
            len(configurations["born_pdgs"]) + 1,
            configurations.get("nincoming", 2),
            dynamical_scale_choice=scales["dynamical_scale_choice"],
            mur_ref_fixed=scales["mur_ref_fixed"],
        )
        out, index = channel.event
        inputs = dict(channel.inputs)
        inputs.update({n: out[n] for names in FRACTIONS.values() for n in names})
        inputs["reference_scale"] = evaluate(reference, out["momenta"])[0]
        inputs["collinear_reference_scale"] = evaluate(
            reference, out["collinear_momenta"]
        )[0]
        inputs["soft_reference_scale"] = evaluate(
            reference.with_particle_count(len(configurations["born_pdgs"])),
            out["born_momenta"],
        )[0]
        outputs = evaluate(
            user, *[inputs[n] for n in names if n != "fks_index"], fks_index=index
        )
        for name, value in zip(user.return_types().keys(), outputs):
            np.testing.assert_allclose(value, theirs[name], rtol=1e-14, atol=0)


@needs_nnpdf
@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_scale_variations(path, grids):
    """FKSCrossSection with scale variations gives at the factors (1, 1) its
    cross section, and at (xi_R, xi_F) that with mur_over_ref and
    muf_over_ref times them where the virtual moves to the Ellis-Sexton scale
    xi_R mu_R: V' = V + SV(mu_R, mu_F) + c_R log xi_R^2 + (c_F1 + c_F2)
    log xi_F^2 - SV(xi_R mu_R, xi_F mu_F) with the soft-virtual part SV and
    its coefficients of FKSSoftVirtual. That is the reweighting of MadFKS,
    whose virtual stays at the Ellis-Sexton scale of the central scales."""
    pdf_grid, alpha_s_grid = grids(asmz(path))
    running = ms.RunningCoupling(alpha_s_grid)
    pairs = [(1.0, 1.0), (2.0, 1.0), (0.5, 1.0), (1.0, 2.0), (1.0, 0.5), (2.0, 0.5)]
    for channel in channels(path):
        hadrons = beams(channel) != (0, 0)
        scales = scale_parameters(channel.run_card)
        common = dict(
            running_coupling=running,
            pdf_grid=pdf_grid if hadrons else None,
            beams=beams(channel),
            sum_flavors=True,
            n_f=channel.n_f,
            cuts=channel.cuts(),
        )
        born, configurations = channel.born, channel.configurations
        varied = fks_cross_section(
            born, configurations, scales=scales, scale_variations=pairs, **common
        )
        assert varied.scale_variations() == pairs
        ours = run_with_couplings(channel, varied)
        variations = ours["scale_variations"]
        assert variations.shape == (len(channel.points), len(pairs))
        np.testing.assert_array_equal(variations[:, 0], ours["cross_section"])
        # the soft-virtual part at the central scales and at those of a pair
        out, index = channel.event
        inputs = dict(channel.inputs)
        inputs.update({n: out[n] for names in FRACTIONS.values() for n in names})
        b = len(configurations["orders"]["born"])
        ren, fact1, fact2 = madspace_scales(channel)["born"]
        soft_virtual = fks_soft_virtual(born, configurations, n_f=channel.n_f)
        count = len(index)

        def at(ren_factor, fact_factor):
            return evaluate(
                soft_virtual,
                out["born_momenta"],
                inputs["born"][:, :b],
                inputs["color_correlated"],
                np.full(count, 1 / (4 * math.pi)),
                inputs["alpha"],
                ren * ren_factor,
                fact1 * fact_factor,
                fact2 * fact_factor,
            )

        central, ren_coefficient, fact_coefficients = at(1.0, 1.0)
        for k, (ren_factor, fact_factor) in enumerate(pairs):
            moved = fks_cross_section(
                born,
                configurations,
                scales=dict(
                    scales,
                    mur_over_ref=scales["mur_over_ref"] * ren_factor,
                    muf_over_ref=scales["muf_over_ref"] * fact_factor,
                ),
                **common,
            )
            shifted, _, _ = at(ren_factor, fact_factor)
            virtual = inputs["virtual"] + central - shifted
            virtual = virtual + ren_coefficient * math.log(ren_factor**2)
            virtual = virtual + fact_coefficients.sum(-1) * math.log(fact_factor**2)
            names = [n for n in moved.arg_types().keys() if n != "fks_index"]
            values = dict(inputs, virtual=virtual)
            theirs, *_ = evaluate(moved, *[values[n] for n in names], fks_index=index)
            scale = np.abs(theirs).max()
            np.testing.assert_allclose(
                variations[:, k], theirs, rtol=1e-11, atol=1e-11 * scale
            )


def scaled_lhapdf_grid(source, path, factor, header=()):
    """The LHAPDF grid source with its densities times factor, and the lines
    of header (keys of a member) in place of those of its own header."""
    with open(source) as f:
        lines = f.read().splitlines()
    keys = {line.split(":")[0] for line in header}
    out, block = list(header), None
    for line in lines:
        if line.strip() == "---":
            block = 0
        elif block is None:
            if line.split(":")[0] in keys:
                continue
        else:
            # the nodes in x and Q and the flavors, then the densities
            block += 1
            if block > 3:
                line = " ".join(f"{float(v) * factor:.17e}" for v in line.split())
        out.append(line)
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def delete_prefixed_globals(context, prefixes):
    for prefix in prefixes:
        for name in GLOBALS:
            if context.global_exists(f"{prefix}.{name}"):
                context.delete_global(f"{prefix}.{name}")


@needs_nnpdf
@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_pdf_variations(path, grids, tmp_path):
    """FKSCrossSection with PDF variations gives with a member of the central
    densities and alpha_s its cross section, with the densities times 1.1
    the cross section times 1.21 (every part goes with the luminosity), and
    with another running of alpha_s the cross section with that running,
    each member on the globals of its own prefix."""
    hadronic = [c for c in channels(path) if all(abs(b) == 1 for b in beams(c))]
    if not hadronic:
        pytest.skip("lepton beams have no PDF variations")
    pdf_grid, alpha_s_grid = grids(asmz(path))
    running = ms.RunningCoupling(alpha_s_grid)
    central = str(tmp_path / "central.dat")
    write_lhapdf_grid(central)
    scaled = str(tmp_path / "scaled.dat")
    scaled_lhapdf_grid(central, scaled, 1.1)
    other = str(tmp_path / "alpha_s.info")
    write_alpha_s_grid(other, asmz(path) + 0.005)
    grids_of = {
        "member_a": (pdf_grid, alpha_s_grid),
        "member_b": (ms.PdfGrid(scaled), alpha_s_grid),
        "member_c": (pdf_grid, ms.AlphaSGrid(other)),
    }
    context = default_context()
    delete_prefixed_globals(context, grids_of)
    try:
        members = []
        for prefix, (grid, alpha_s) in grids_of.items():
            grid.initialize_globals(context, prefix)
            alpha_s.initialize_globals(context, prefix)
            members.append((ms.RunningCoupling(alpha_s, prefix), grid, prefix))
        for channel in hadronic:
            common = dict(
                pdf_grid=pdf_grid,
                beams=beams(channel),
                sum_flavors=True,
                n_f=channel.n_f,
                cuts=channel.cuts(),
            )
            born, configurations = channel.born, channel.configurations
            scales = scale_parameters(channel.run_card)
            varied = fks_cross_section(
                born,
                configurations,
                scales=scales,
                running_coupling=running,
                pdf_variations=members,
                scale_variations=[(2.0, 0.5)],
                **common,
            )
            assert varied.pdf_variation_count() == len(members)
            ours = run_with_couplings(channel, varied)
            variations = ours["pdf_variations"]
            sigma = ours["cross_section"]
            assert variations.shape == (len(channel.points), len(members))
            assert ours["scale_variations"].shape == (len(channel.points), 1)
            atol = 1e-13 * np.abs(sigma).max()
            np.testing.assert_allclose(variations[:, 0], sigma, rtol=1e-14, atol=0)
            np.testing.assert_allclose(
                variations[:, 1], 1.21 * sigma, rtol=1e-12, atol=atol
            )
            with_other = fks_cross_section(
                born,
                configurations,
                scales=scales,
                running_coupling=members[2][0],
                **common,
            )
            theirs = run_with_couplings(channel, with_other)["cross_section"]
            atol = 1e-13 * np.abs(theirs).max()
            np.testing.assert_allclose(variations[:, 2], theirs, rtol=1e-13, atol=atol)
            if np.abs(sigma).max() > 0:
                assert not np.allclose(theirs, sigma, rtol=1e-3)
        with pytest.raises(ValueError, match="need scales"):
            fks_cross_section(born, configurations, pdf_variations=members)
        with pytest.raises(ValueError, match="one set for both beams"):
            fks_cross_section(
                born,
                configurations,
                scales=scales,
                running_coupling=running,
                pdf_variations=members,
                **dict(common, pdf_grid2=pdf_grid),
            )
    finally:
        delete_prefixed_globals(context, grids_of)


def test_scale_errors():
    with pytest.raises(ValueError, match="dynamical_scale_choice 4"):
        ms.NLOScale(5, dynamical_scale_choice=4)
    with pytest.raises(ValueError, match="an outgoing particle"):
        ms.NLOScale(2)
    with pytest.raises(ValueError, match="an outgoing particle"):
        ms.NLOScale(1, incoming=1)
    with pytest.raises(ValueError, match="or one"):
        ms.NLOScale(5, incoming=3)
    with pytest.raises(TypeError, match="unknown scale parameter"):
        nlo_scale(5, scalefact=1.0)


@needs_nnpdf
def test_luminosity_flavors(grids):
    """A flavor without density (a lepton of a proton) adds nothing, lepton
    beams count each flavor once, the luminosity picks the group of the event,
    and with all_groups it stacks the luminosity of each group."""
    pdf_grid, _ = grids(0.118)
    flavors_ = [[[2, -2], [4, -4]], [[21, -2], [11, -2]], [[2, 21]]]
    x1, x2 = np.array([0.1, 0.2, 0.3]), np.array([0.05, 0.01, 0.4])
    q = np.array([91.0, 200.0, 50.0])
    index = np.array([0, 1, 2])
    (ours,) = evaluate(
        ms.NLOLuminosity(flavors_, pdf_grid, sum_flavors=True),
        x1,
        x2,
        q,
        q,
        fks_index=index,
    )
    density = ms.PartonDensity(pdf_grid, [-4, -2, 2, 4, 21])
    (xf1,) = evaluate(density, x1, q)
    (xf2,) = evaluate(density, x2, q)
    f1 = dict(zip([-4, -2, 2, 4, 21], (xf1 / x1[:, None]).T))
    f2 = dict(zip([-4, -2, 2, 4, 21], (xf2 / x2[:, None]).T))
    expected = [
        f1[2][0] * f2[-2][0] + f1[4][0] * f2[-4][0],
        f1[21][1] * f2[-2][1],
        f1[2][2] * f2[21][2],
    ]
    np.testing.assert_allclose(ours, expected, rtol=1e-15)
    # every group at once, from one evaluation of the densities
    stacked = ms.NLOLuminosity(flavors_, pdf_grid, sum_flavors=True, all_groups=True)
    assert stacked.all_groups() and stacked.group_count() == 3
    assert list(stacked.arg_types().keys()) == [
        "x1",
        "x2",
        "fact_scale1",
        "fact_scale2",
    ]
    (groups,) = evaluate(stacked, x1, x2, q, q)
    assert groups.shape == (3, 3)
    for group in range(3):
        (one,) = evaluate(
            ms.NLOLuminosity(flavors_, pdf_grid, sum_flavors=True),
            x1,
            x2,
            q,
            q,
            fks_index=np.full(3, group),
        )
        np.testing.assert_array_equal(groups[:, group], one)
    # one flavor of each event, with factors, beyond the flavors zero
    factors = [[0.5, 2.0], [1.0, 3.0], [4.0]]
    (picked,) = evaluate(
        ms.NLOLuminosity(flavors_, pdf_grid, factors=factors),
        x1,
        x2,
        q,
        q,
        flavor_index=np.array([1, 0, 1]),
        fks_index=index,
    )
    np.testing.assert_allclose(
        picked, [2.0 * f1[4][0] * f2[-4][0], f1[21][1] * f2[-2][1], 0.0], rtol=1e-15
    )
    # leptons count one, and so does a beam without densities
    (leptons,) = evaluate(
        ms.NLOLuminosity([[[-11, 11], [-11, 11]], [[-11, 11]]], sum_flavors=True),
        x1,
        x2,
        q,
        q,
        fks_index=np.array([0, 1, 0]),
    )
    np.testing.assert_array_equal(leptons, [2.0, 1.0, 2.0])
    (dis,) = evaluate(
        ms.NLOLuminosity(
            [[[11, 2], [11, 4]]], pdf_grid, densities=[False, True], sum_flavors=True
        ),
        x1,
        x2,
        q,
        q,
        fks_index=np.zeros(3, dtype=int),
    )
    np.testing.assert_allclose(dis, f2[2] + f2[4], rtol=1e-15)


@needs_nnpdf
def test_luminosity_errors(grids):
    pdf_grid, _ = grids(0.118)
    with pytest.raises(ValueError, match="no groups of flavors"):
        ms.NLOLuminosity([])
    with pytest.raises(ValueError, match="without flavors"):
        ms.NLOLuminosity([[]])
    with pytest.raises(ValueError, match="two incoming partons"):
        ms.NLOLuminosity([[[1, 2, 3]]])
    with pytest.raises(ValueError, match="needs one for the first"):
        ms.NLOLuminosity([[[1, -1]]], pdf_grid2=pdf_grid)
    with pytest.raises(ValueError, match="a beam with densities"):
        ms.NLOLuminosity([[[1, -1]]], pdf_grid, densities=[False, False])
    with pytest.raises(ValueError, match="factors need the groups"):
        ms.NLOLuminosity([[[1, -1]]], factors=[[1.0], [1.0]])
    with pytest.raises(ValueError, match="a factor for each flavor"):
        ms.NLOLuminosity([[[1, -1]]], factors=[[1.0, 2.0]])
    with pytest.raises(ValueError, match="all groups sums over their flavors"):
        ms.NLOLuminosity([[[1, -1]]], all_groups=True)
    configurations = {"configurations": []}
    with pytest.raises(ValueError, match="older FKS exporter"):
        fks_luminosities(configurations)
    configurations.update(born_flavors=[], real_flavors=[])
    with pytest.raises(ValueError, match="not protons, antiprotons, photons or"):
        fks_luminosities(configurations, pdf_grid, beams=(5, 5))
    with pytest.raises(ValueError, match="hadron beams need a PdfGrid"):
        fks_luminosities(configurations, beams=(1, 1))
    for beams in ((0, 0), (2, 2)):
        with pytest.raises(ValueError, match="hadron beams need a PdfGrid"):
            fks_luminosities(configurations, pdf_grid, beams=beams)
    for beams in ((0, 1), (2, 1), (2, 0)):
        with pytest.raises(ValueError, match="of two kinds"):
            fks_luminosities(configurations, pdf_grid, beams=beams)


def test_underlying_born():
    """underlying_born as set_pdg_codes: q -> q g, g -> q q~ in the final
    state, q -> g q and g -> q q~ in the initial state, and a missing Born."""
    final = dict(i_fks=5, j_fks=4, splitting_type=["QCD"])
    assert underlying_born([1, -1, 21, 2, 21], final) == [1, -1, 21, 2]
    assert underlying_born([1, -1, 21, 2, -2], final) == [1, -1, 21, 21]
    initial = dict(i_fks=5, j_fks=1, splitting_type=["QCD"])
    assert underlying_born([2, -1, 11, -12, 2], initial) == [21, -1, 11, -12]
    assert underlying_born([21, -1, 11, -12, -2], initial) == [2, -1, 11, -12]
    assert (
        underlying_born([1, -1, 11, -11, 1], dict(initial, splitting_type=["QED"]))[0]
        == 22
    )
    with pytest.raises(ValueError, match="no mother"):
        underlying_born([2, -1, 11, -12, 1], initial)
    configurations = {
        "born_pdgs": [2, -2, 11, -11],
        "born_flavors": [[2, -2, 11, -11]],
        "real_flavors": [[[21, -2, 11, -11, -4]]],
        "configurations": [dict(initial, real_me=1)],
    }
    with pytest.raises(ValueError, match="is not a flavor of the Born"):
        fks_born_flavor_indices(configurations)
