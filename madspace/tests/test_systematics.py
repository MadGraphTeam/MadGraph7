"""Scale/PDF systematics computed at combine time (SystematicsCalculator).

The PDF-dependent tests need an LHAPDF data directory holding
NNPDF40MC_lo_as_01180 (LHAPDF_DATA_PATH, or the paths of the lhapdf module);
the scalar interpolation is checked against the runtime kernels, and against
LHAPDF itself when the python module is available.
"""

import json
import os
import re

import numpy as np
import pytest
from pytest import approx

import madspace as ms

PDF_SET = "NNPDF40MC_lo_as_01180"


def _pdf_dir():
    paths = []
    if os.environ.get("LHAPDF_DATA_PATH"):
        paths += os.environ["LHAPDF_DATA_PATH"].split(os.pathsep)
    try:
        import lhapdf

        lhapdf.setVerbosity(0)
        paths += list(lhapdf.paths())
    except ImportError:
        pass
    for path in paths:
        if os.path.isfile(os.path.join(path, PDF_SET, f"{PDF_SET}_0000.dat")):
            return path
    return None


PDF_DIR = _pdf_dir()
pytestmark = pytest.mark.skipif(
    PDF_DIR is None, reason=f"pdf set {PDF_SET} required to run this test"
)


def grid_files(set_name=PDF_SET, member=0):
    return (
        os.path.join(PDF_DIR, set_name, f"{set_name}_{member:04d}.dat"),
        os.path.join(PDF_DIR, set_name, f"{set_name}.info"),
    )


@pytest.fixture(scope="module")
def grids():
    grid_file, info_file = grid_files()
    return ms.PdfGrid(grid_file), ms.AlphaSGrid(info_file)


def sample_points(grid, n, seed=1):
    rng = np.random.default_rng(seed)
    x = 10 ** rng.uniform(np.log10(grid.x[0]) + 1e-6, 0, n)
    q = 10 ** rng.uniform(
        np.log10(grid.q[0]) + 1e-6, np.log10(grid.q[-1]) - 1e-6, n
    )
    return x, q


def test_scalar_pdf_matches_kernel(grids):
    """PdfGrid.interpolate reproduces the runtime PartonDensity kernel."""
    grid, _ = grids
    ctx = ms.default_context()
    grid.initialize_globals(ctx, "syst")
    pids = [-2, -1, 21, 1, 2, 3, 4, 5]
    pdf = ms.PartonDensity(grid, pids, False, "syst")
    x, q = sample_points(grid, 500)
    kernel = np.asarray(pdf(x, q))
    for j, pid in enumerate(pids):
        index = grid.pid_index(pid)
        scalar = np.array([grid.interpolate(index, xi, qi) for xi, qi in zip(x, q)])
        assert scalar == approx(kernel[:, j], rel=1e-12, abs=1e-14)


def test_scalar_alpha_s_matches_kernel(grids):
    """AlphaSGrid.interpolate reproduces the runtime RunningCoupling kernel."""
    _, alpha_s = grids
    ctx = ms.default_context()
    alpha_s.initialize_globals(ctx, "syst")
    coupling = ms.RunningCoupling(alpha_s, "syst")
    rng = np.random.default_rng(2)
    q = 10 ** rng.uniform(
        np.log10(alpha_s.q[0]) + 1e-6, np.log10(alpha_s.q[-1]) - 1e-6, 500
    )
    kernel = np.asarray(coupling(q))
    scalar = np.array([alpha_s.interpolate(qi) for qi in q])
    assert scalar == approx(kernel, rel=1e-12)
    assert alpha_s.interpolate(91.188) == approx(0.118, rel=1e-3)


def test_scalar_matches_lhapdf(grids):
    """Both interpolations agree with LHAPDF where the python module exists."""
    lhapdf = pytest.importorskip("lhapdf")
    lhapdf.setVerbosity(0)
    try:
        ref = lhapdf.mkPDF(PDF_SET, 0)
    except RuntimeError:
        pytest.skip("lhapdf cannot load %s" % PDF_SET)
    grid, alpha_s = grids
    x, q = sample_points(grid, 200, seed=3)
    for pid in [21, 2, -1, 5]:
        index = grid.pid_index(pid)
        for xi, qi in zip(x, q):
            assert grid.interpolate(index, xi, qi) == approx(
                ref.xfxQ(pid, xi, qi), rel=1e-6, abs=1e-9
            )
    for qi in q:
        assert alpha_s.interpolate(qi) == approx(ref.alphasQ(qi), rel=1e-6)


def make_config(**kwargs):
    grid_file, info_file = grid_files()
    config = ms.SystematicsConfig()
    config.mur = [0.5, 1.0, 2.0]
    config.muf = [0.5, 1.0, 2.0]
    config.nominal_set_name = PDF_SET
    config.nominal_lhaid = 338500
    config.nominal_error_type = "replicas"
    config.nominal_description = "test set <x> => y"
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def make_events(grid, alpha_s, n=50, seed=4):
    """Random u u > u u like events with the columns the combined event buffer
    carries; the PDF product is the nominal one, as the integrand stores it."""
    rng = np.random.default_rng(seed)
    x1 = 10 ** rng.uniform(-3, -0.3, n)
    x2 = 10 ** rng.uniform(-3, -0.3, n)
    scale = 10 ** rng.uniform(1.5, 3, n)
    weight = rng.uniform(0.5, 1.5, n)
    up = grid.pid_index(2)
    product = np.array(
        [grid.interpolate(up, a, s) * grid.interpolate(up, b, s) for a, b, s in zip(x1, x2, scale)]
    )
    return dict(
        event_weight=list(weight),
        subprocess_index=[0] * n,
        flavor_index=[0] * n,
        ren_scale=list(scale),
        x1=list(x1),
        fact_scale1=list(scale),
        x2=list(x2),
        fact_scale2=list(scale),
        partial_weight_product=list(product),
    )


def test_scale_variations(grids):
    """The LO scale reweighting formula, checked against a direct evaluation."""
    grid, alpha_s = grids
    config = make_config()
    args = [ms.SubprocessSystArgs(qcd_power=2, beam_pdgs=[[2, 2]])]
    calc = ms.SystematicsCalculator(config, args, grid, alpha_s)
    variations = calc.variations
    assert len(variations) == 8
    assert [(v.mur, v.muf) for v in variations] == [
        (0.5, 0.5), (0.5, 1.0), (0.5, 2.0), (1.0, 0.5),
        (1.0, 2.0), (2.0, 0.5), (2.0, 1.0), (2.0, 2.0),
    ]
    assert calc.weight_ids == list(range(1, 9))
    events = make_events(grid, alpha_s)
    weights = calc.weights(**events)
    up = grid.pid_index(2)
    for i in range(len(events["event_weight"])):
        w0 = events["event_weight"][i]
        mu = events["ren_scale"][i]
        x1, x2 = events["x1"][i], events["x2"][i]
        for var, w in zip(variations, weights[i]):
            r_alpha = (alpha_s.interpolate(var.mur * mu) / alpha_s.interpolate(mu)) ** 2
            r_pdf = (
                grid.interpolate(up, x1, var.muf * mu) * grid.interpolate(up, x2, var.muf * mu)
            ) / (grid.interpolate(up, x1, mu) * grid.interpolate(up, x2, mu))
            assert w == approx(w0 * r_alpha * r_pdf, rel=1e-12)


def test_identities(grids):
    """Trivial variations give exactly the nominal weight."""
    grid, alpha_s = grids
    events = make_events(grid, alpha_s, n=20)
    # no QCD coupling: mu_R variations are the identity
    config = make_config(mur=[0.5, 1.0, 2.0], muf=[1.0])
    calc = ms.SystematicsCalculator(
        config, [ms.SubprocessSystArgs(qcd_power=0, beam_pdgs=[[2, 2]])], grid, alpha_s
    )
    for w0, row in zip(events["event_weight"], calc.weights(**events)):
        assert row == [w0, w0]
    # leptonic beams: mu_F variations are dropped, mu_R ones remain
    config = make_config(has_pdf=False, pdf_members=[])
    calc = ms.SystematicsCalculator(
        config, [ms.SubprocessSystArgs(qcd_power=2, beam_pdgs=[[11, -11]])], None, alpha_s
    )
    assert [(v.mur, v.muf) for v in calc.variations] == [(0.5, 1.0), (2.0, 1.0)]


def test_non_uniform_qcd_power_drops_mur(grids):
    grid, alpha_s = grids
    config = make_config()
    calc = ms.SystematicsCalculator(
        config, [ms.SubprocessSystArgs(qcd_power=-1, beam_pdgs=[[2, -2]])], grid, alpha_s
    )
    assert [(v.mur, v.muf) for v in calc.variations] == [(1.0, 0.5), (1.0, 2.0)]
    assert any("renormalisation scale" in w for w in calc.warnings)


def test_pdf_member_and_header(grids):
    """A PDF member variation (here the nominal member as a fake second set)
    reweights by the PDF ratio, and the <initrwgt> text follows the
    systematics.py conventions."""
    grid, alpha_s = grids
    grid_file, info_file = grid_files()
    other = ms.PdfMemberSpec(
        set_name="OtherSet", set_lhaid=900000, member=1, grid_file=grid_file,
        info_file=info_file, error_type="hessian", description="a => b > c",
    )
    same = ms.PdfMemberSpec(
        set_name=PDF_SET, set_lhaid=338500, member=1, grid_file=grid_file,
        info_file=info_file, error_type="replicas", description="",
    )
    config = make_config(mur=[1.0], muf=[1.0], pdf_members=[same, other])
    calc = ms.SystematicsCalculator(
        config, [ms.SubprocessSystArgs(qcd_power=2, beam_pdgs=[[2, 2]])], grid, alpha_s
    )
    # nominal-set member requested -> the group starts with the nominal member
    assert [(v.pdf_index) for v in calc.variations] == [-1, 0, 1]
    events = make_events(grid, alpha_s, n=10)
    for w0, row in zip(events["event_weight"], calc.weights(**events)):
        # same grid file everywhere: every ratio is exactly one
        assert row == approx([w0, w0, w0], rel=1e-12)
    header = calc.initrwgt()
    assert '<weightgroup name="%s" combine="replicas">' % PDF_SET in header
    assert '<weightgroup name="OtherSet" combine="hessian"> # 900000: a ; b .gt. c' in header
    assert '<weight id="1" MUR="1.0" MUF="1.0" PDF="338500" >  </weight>' in header
    assert '<weight id="2" MUR="1.0" MUF="1.0" PDF="338501" > PDF=338500 MemberID=1 </weight>' in header
    assert '<weight id="3" MUR="1.0" MUF="1.0" PDF="900001" > PDF=900000 MemberID=1 </weight>' in header
    assert header.count("<weightgroup") == header.count("</weightgroup>") == 2
    summary = json.loads(calc.summary())
    assert [v["id"] for v in summary["variations"]] == [1, 2, 3]
    assert summary["variations"][2]["pdf_set"] == "OtherSet"


def test_lhe_event_rwgt_format():
    event = ms.LHEEvent()
    event.particles = [ms.LHEParticle(pdg_id=21, status_code=-1, energy=1.0, p_z=1.0)]
    event.rwgt_ids = [1, 2]
    event.rwgt = [1.5e3, -2.0]
    text = event.format()
    assert "<rwgt>\n<wgt id='1'> +1.5000000e+03 </wgt>\n<wgt id='2'> -2.0000000e+00 </wgt>\n</rwgt>\n</event>" in text


def test_config_json_roundtrip():
    grid_file, info_file = grid_files()
    config = make_config(write_inputs=True, together=False)
    config.pdf_members = [
        ms.PdfMemberSpec(PDF_SET, 338500, 1, grid_file, info_file, "replicas", "d")
    ]
    text = config.to_json()
    back = ms.SystematicsConfig.from_json(text)
    assert back.mur == config.mur and back.muf == config.muf
    assert back.write_inputs and not back.together
    assert back.pdf_members[0].grid_file == grid_file
    args = ms.SubprocessSystArgs(qcd_power=2, beam_pdgs=[[2, 2], [21, 2]])
    back_args = ms.SubprocessSystArgs.from_json(args.to_json())
    assert back_args.qcd_power == 2 and back_args.beam_pdgs == [[2, 2], [21, 2]]


def random_momenta(n, seed=5):
    """2 -> 2 massless momenta (E px py pz), incoming along z."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        e1, e2 = rng.uniform(50, 800, 2)
        p_in = [[e1, 0.0, 0.0, e1], [e2, 0.0, 0.0, -e2]]
        # back-to-back in the partonic frame, boosted along z
        s = 4 * e1 * e2
        pcm = np.sqrt(s) / 2
        theta, phi = np.arccos(rng.uniform(-1, 1)), rng.uniform(0, 2 * np.pi)
        px, py, pz = pcm * np.sin(theta) * np.cos(phi), pcm * np.sin(theta) * np.sin(phi), pcm * np.cos(theta)
        beta = (e1 - e2) / (e1 + e2)
        gamma = 1 / np.sqrt(1 - beta**2)
        def boost(e, pz):
            return gamma * (e + beta * pz), gamma * (pz + beta * e)
        ea, pza = boost(pcm, pz)
        eb, pzb = boost(pcm, -pz)
        out.append(p_in + [[ea, px, py, pza], [eb, -px, -py, pzb]])
    return out


def test_dynamical_scales(grids):
    """Dynamical scale variations: the scales from the momenta match their
    definitions, and the weights follow the LO formula with the new scale."""
    grid, alpha_s = grids
    momenta = random_momenta(20)
    for p in momenta[:5]:
        p = np.array(p)
        fin = p[2:]
        et = sum(e * np.sqrt((px**2 + py**2) / (px**2 + py**2 + pz**2)) for e, px, py, pz in fin)
        ht = sum(np.sqrt(max(0.0, e**2 - pz**2)) for e, px, py, pz in fin)
        shat = np.sqrt((p[0, 0] + p[1, 0]) ** 2 - (p[0, 3] + p[1, 3]) ** 2)
        assert ms.SystematicsCalculator.dynamical_scale(1, p.tolist()) == approx(et, rel=1e-12)
        assert ms.SystematicsCalculator.dynamical_scale(2, p.tolist()) == approx(ht, rel=1e-12)
        assert ms.SystematicsCalculator.dynamical_scale(3, p.tolist()) == approx(ht / 2, rel=1e-12)
        assert ms.SystematicsCalculator.dynamical_scale(4, p.tolist()) == approx(shat, rel=1e-12)

    config = make_config(mur=[0.5, 1.0], muf=[1.0, 2.0], dyn_scales=[3, 4])
    calc = ms.SystematicsCalculator(
        config, [ms.SubprocessSystArgs(qcd_power=2, beam_pdgs=[[2, 2]])], grid, alpha_s
    )
    # (mur, muf) grid at the generated scale minus the nominal point, then the
    # full grid for each dynamical choice
    assert [(v.mur, v.muf, v.dyn) for v in calc.variations] == [
        (0.5, 1.0, -1), (0.5, 2.0, -1), (1.0, 2.0, -1),
        (0.5, 1.0, 3), (0.5, 2.0, 3), (1.0, 1.0, 3), (1.0, 2.0, 3),
        (0.5, 1.0, 4), (0.5, 2.0, 4), (1.0, 1.0, 4), (1.0, 2.0, 4),
    ]
    header = calc.initrwgt()
    assert 'DYN_SCALE="3" PDF="338500" > dyn_scale_choice=HT/2' in header
    events = make_events(grid, alpha_s, n=20)
    events["momenta"] = momenta
    weights = calc.weights(**events)
    up = grid.pid_index(2)
    for i in range(20):
        w0, mu = events["event_weight"][i], events["ren_scale"][i]
        x1, x2 = events["x1"][i], events["x2"][i]
        nominal_pdf = grid.interpolate(up, x1, mu) * grid.interpolate(up, x2, mu)
        for var, w in zip(calc.variations, weights[i]):
            mu_dyn = mu if var.dyn == -1 else ms.SystematicsCalculator.dynamical_scale(var.dyn, momenta[i])
            r_alpha = (alpha_s.interpolate(var.mur * mu_dyn) / alpha_s.interpolate(mu)) ** 2
            r_pdf = grid.interpolate(up, x1, var.muf * mu_dyn) * grid.interpolate(up, x2, var.muf * mu_dyn) / nominal_pdf
            assert w == approx(w0 * r_alpha * r_pdf, rel=1e-12)


def test_event_histograms(grids):
    """Event-sample histograms: bin sums equal the mean weight per column,
    the scale envelope brackets the nominal, binomial errors."""
    grid, alpha_s = grids
    n = 200
    momenta = random_momenta(n, seed=6)
    pt = np.array([np.hypot(p[2][1], p[2][2]) for p in momenta])
    ctx = ms.Context(device=ms.cpu_device(), thread_count=1)
    pids = [2, 2, 2, 2]
    obs = ms.ObservableValues([
        ms.Observable(pids, observable="pt", select_pids=[[2]], order_observable="pt",
                      order_indices=[1], name="jet-pt"),
        ms.Observable(pids, observable="sqrt_s", select_pids=[], name="sqrt_s"),
    ])
    specs = [ms.EventHistogramSpec("jet-pt", 0.0, 500.0, 10),
             ms.EventHistogramSpec("sqrt_s", 0.0, 2000.0, 8)]
    hists = ms.EventHistograms(ctx, specs, [ms.SubprocessObservables(obs, 4)])
    rng = np.random.default_rng(7)
    weight = np.full(n, 3.0)
    syst = np.stack([weight * 0.8, weight * 1.3, weight], axis=1)  # two scale-like columns + nominal copy
    hists.fill(list(weight), [0] * n, momenta, syst.tolist())
    data = json.loads(hists.to_json())
    assert [h["name"] for h in data] == ["jet-pt", "sqrt_s"]
    jet = data[0]
    assert sum(jet["bin_values"]) == approx(3.0)
    assert [sum(w["bin_values"]) for w in jet["weights"]] == approx([2.4, 3.9, 3.0])
    # binning of the leading-pt observable against numpy
    counts, _ = np.histogram(pt, bins=10, range=(0.0, 500.0))
    assert jet["bin_values"][1:-1] == approx(list(3.0 * counts / n))
    # binomial error of an unweighted sample
    m = counts[counts.argmax()]
    assert jet["bin_errors"][1 + counts.argmax()] == approx(3.0 * np.sqrt(m * (1 - m / n)) / n)
    # a systematics calculator adds the envelope: fake one with two scale variations
    config = make_config(mur=[0.5, 1.0, 2.0], muf=[1.0], pdf_members=[])
    calc = ms.SystematicsCalculator(
        config, [ms.SubprocessSystArgs(qcd_power=2, beam_pdgs=[[2, 2]])], grid, alpha_s
    )
    assert calc.weight_count == 2
    hists2 = ms.EventHistograms(ctx, specs, [ms.SubprocessObservables(obs, 4)])
    hists2.fill(list(weight), [0] * n, momenta, syst[:, :2].tolist())
    data2 = json.loads(hists2.to_json(calc))
    env = data2[0]["scale_envelope"]
    for lo, nom, hi in zip(env["low"], data2[0]["bin_values"], env["high"]):
        assert lo <= nom + 1e-12 and hi >= nom - 1e-12
    assert [w["id"] for w in data2[0]["weights"]] == [1, 2]
