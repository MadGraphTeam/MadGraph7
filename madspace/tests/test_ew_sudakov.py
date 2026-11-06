"""EWSudakov: the Sudakov approximation of the EW corrections, as
ewsudakov_wrapper.f and ewsudakov_functions.f of MadGraph.

The reference data (see fks_reference/make_ewsudakov_reference.py) store,
for Born points of several processes, the matrix elements that
sudakov_wrapper takes per helicity combination (the base amplitude at the
LO1 and the LO2, its parameter derivatives, the interferences), the
coefficients of ewsudakov_functions.f per helicity combination and per
interference in every mode, and what sudakov_wrapper gives with them, per
helicity combination and summed. EWSudakov must give the coefficients (from
unit matrix elements) and the corrections from the matrix elements of the
Fortran in every mode.
"""

import glob
import math
import os

import numpy as np
import pytest
from fks_testing import evaluate, reference

import madspace as ms
from madspace.fks import ew_sudakov, ew_sudakov_process

DATA = sorted(
    glob.glob(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "test_data",
            "ewsudakov",
            "*.json",
        )
    )
)
IDS = [os.path.basename(p)[:-5] for p in DATA]


def parameters(data):
    """The EWSudakovParameters of reference data."""
    p = data["params"]
    scheme = data["fks_configurations"]["ewsudakov"]["scheme"]
    return ms.EWSudakovParameters(
        mw=p["mw"],
        mz=p["mz"],
        mh=p["mh"],
        mt=p["mt"],
        alpha=p["gal"] ** 2 / (4 * math.pi),
        gf=p["gf"],
        scheme=ms.EWSudakovScheme(scheme),
        mb=p["mb"],
    )


def modes(data):
    """The modes of reference data, in their order."""
    return [
        ms.EWSudakovMode(sud_mod=m, s_to_rij=bool(s), rij_ge_mw=bool(r))
        for m, s, r in data["modes"]
    ]


def generator(data):
    return ew_sudakov(data["fks_configurations"], parameters(data), modes(data))


def inputs(data, sudakov, point, rows):
    """The inputs of EWSudakov for a batch of rows at a point, all matrix
    elements zero."""
    n = sudakov.particle_count()
    h = sudakov.helicity_count()
    t = max(len(sudakov.interference_terms()), 1)
    p = data["params"]
    return dict(
        momenta=np.broadcast_to(np.array(point["momenta"]), (rows, n, 4)).copy(),
        diagonal=np.zeros((rows, h)),
        lo2=np.zeros((rows, h)),
        derivatives=np.zeros((rows, h, 6)),
        interference=np.zeros((rows, t, 2)),
        alpha_s=np.full(rows, p["g"] ** 2 / (4 * math.pi)),
        ren_scale=np.full(rows, p["scale"]),
    )


def run(sudakov, args):
    keys = ["momenta", "diagonal", "lo2", "derivatives", "interference"]
    keys += ["alpha_s", "ren_scale"]
    return evaluate(sudakov, *[args[k] for k in keys])[0]


def per_mode(rows, data, count, start):
    """The complex values of rows [mode, helicity, ...] of the dump as an
    array (modes, helicities + 1, count), zero where there is no row."""
    out = np.zeros((len(data["modes"]), count[0], count[1]), dtype=complex)
    for row in rows:
        values = np.array(row[start:])
        out[row[0], row[1]] = values[0::2] + 1j * values[1::2]
    return out


def matrix_elements(data, sudakov, point):
    """The matrix elements of the Fortran at a point as inputs of EWSudakov,
    in a batch of one row."""
    args = inputs(data, sudakov, point, 1)
    for h, hel in enumerate(point["helicities"]):
        args["diagonal"][0, h] = hel["diag"]
        args["lo2"][0, h] = hel["lo2"]
        args["derivatives"][0, h] = hel["derivatives"]
    position = {term: k for k, term in enumerate(sudakov.interference_terms())}
    for h, t, re, im in point["interferences"]:
        k = position.get((h - 1, t - 1))
        if k is not None:
            args["interference"][0, k] = [re, im]
    return args


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_bases_and_terms(path):
    """The base amplitude of each helicity combination is that of the
    Fortran, and the interference terms are those with a coefficient in a
    mode at some point, in the order of the helicities and interferences."""
    data = reference(path)
    sudakov = generator(data)
    point = data["points"][0]
    assert list(sudakov.helicity_bases()) == [h["base"] for h in point["helicities"]]
    terms = list(sudakov.interference_terms())
    assert terms == sorted(terms)
    nonzero = {
        (row[1] - 1, row[2] - 1)
        for point in data["points"]
        for row in point["interference_coefficients"]
    }
    assert nonzero <= set(terms)
    for h, t in terms:
        base = sudakov.helicity_bases()[h]
        assert (
            data["fks_configurations"]["ewsudakov"]["interference"][t]["base"] == base
        )
    # a matrix element the Fortran takes is one of the inputs
    for point in data["points"]:
        for h, t, re, im in point["interferences"]:
            if (h - 1, t - 1) not in terms:
                coefficients = [
                    row[3:]
                    for row in point["interference_coefficients"]
                    if row[1:3] == [h, t]
                ]
                assert not any(any(c) for c in coefficients)


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_coefficients_match_madgraph(path):
    """The coefficients of get_lsc_diag + get_ssc_n_diag + get_xxc_diag
    (their real parts, which multiply the real |M_h|^2) and of the
    interferences (complex), GET_QCD_LO2 and the parameter renormalisation per
    helicity combination, from unit matrix elements."""
    data = reference(path)
    sudakov = generator(data)
    info = data["fks_configurations"]["ewsudakov"]
    h = sudakov.helicity_count()
    terms = sudakov.interference_terms()
    for point in data["points"]:
        rows = 2 * h + 2 * len(terms) + h
        args = inputs(data, sudakov, point, rows)
        for k in range(h):
            args["diagonal"][k, k] = 1.0
            args["lo2"][h + k, k] = 1.0
        for k in range(len(terms)):
            args["interference"][2 * h + 2 * k, k] = [1.0, 0.0]
            args["interference"][2 * h + 2 * k + 1, k] = [0.0, 1.0]
        offset = 2 * h + 2 * len(terms)
        for k, hel in enumerate(point["helicities"]):
            args["derivatives"][offset + k, k] = hel["derivatives"]
        out = run(sudakov, args)

        count = (h + 1, 3)
        coefficients = per_mode(point["coefficients"], data, count, 2)
        diagonal = 2 * coefficients.sum(axis=2).real[:, 1:]
        scale = np.abs(coefficients).sum(axis=2)[:, 1:]
        if not info["has_lo1"]:
            diagonal[:] = 0.0
        np.testing.assert_allclose(
            out[:h].T, diagonal, rtol=0, atol=1e-13 * scale.max()
        )

        qcd = per_mode(point["qcd_coefficients"], data, (h + 1, 1), 2)[:, 1:, 0]
        np.testing.assert_allclose(out[h : 2 * h].T, 2 * qcd.real, rtol=1e-13, atol=0)

        expected = np.zeros((len(data["modes"]), len(terms)), dtype=complex)
        position = {term: k for k, term in enumerate(terms)}
        for m, hh, t, re, im in point["interference_coefficients"]:
            expected[m, position[(hh - 1, t - 1)]] = re + 1j * im
        got = out[2 * h : offset : 2].T - 1j * out[2 * h + 1 : offset : 2].T
        np.testing.assert_allclose(
            got / 2, expected, rtol=0, atol=1e-13 * max(np.abs(expected).max(), 1e-3)
        )

        # the parameter renormalisation of each helicity, 2 Re of
        # amp_split_ewsud_par of sudakov_wrapper for that helicity alone
        lo1 = per_mode(point["lo1"], data, (h + 1, 4), 2)
        par = 2 * lo1[:, 1:, 3].real
        np.testing.assert_allclose(
            out[offset:].T, par, rtol=0, atol=1e-13 * max(np.abs(par).max(), 1e-30)
        )


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_corrections_match_madgraph(path):
    """The corrections in every mode from the matrix elements of the Fortran
    are the weights of compute_ewsudakov (without f_b / g^k), per helicity
    combination and summed."""
    data = reference(path)
    sudakov = generator(data)
    h = sudakov.helicity_count()
    for point in data["points"]:
        args = matrix_elements(data, sudakov, point)
        out = run(sudakov, args)[0]
        weight = np.array(point["weight"])
        # the size of the parts, for the tolerance
        lo1 = per_mode(point["lo1"], data, (h + 1, 4), 2)
        lo2 = per_mode(point["lo2"], data, (h + 1, 2), 2)
        size = 2 * (
            np.abs(lo1[:, 1:]).sum(axis=(1, 2)) + np.abs(lo2[:, 1:]).sum(axis=(1, 2))
        )
        np.testing.assert_allclose(out, weight, rtol=0, atol=1e-12 * size.max())

        # each helicity combination alone
        rows = inputs(data, sudakov, point, h)
        for k in range(h):
            rows["diagonal"][k, k] = args["diagonal"][0, k]
            rows["lo2"][k, k] = args["lo2"][0, k]
            rows["derivatives"][k, k] = args["derivatives"][0, k]
        for k, (hh, _) in enumerate(sudakov.interference_terms()):
            rows["interference"][hh, k] = args["interference"][0, k]
        out = run(sudakov, rows)
        expected = 2 * (lo1[:, 1:].sum(axis=2).real + lo2[:, 1:].sum(axis=2).real)
        np.testing.assert_allclose(out.T, expected, rtol=0, atol=1e-12 * size.max())


def test_coverage():
    """The reference data have every kind of amplitude, both schemes, a LO2,
    the modes of compute_ewsudakov and points where rij_ge_mw matters."""
    kinds, schemes, lo2, clamped = set(), set(), False, False
    for path in DATA:
        data = reference(path)
        info = data["fks_configurations"]["ewsudakov"]
        kinds |= {amp["type"] for amp in info["interference"]}
        kinds |= {"goldstone"} if info["goldstone"] else set()
        schemes.add(info["scheme"])
        lo2 = lo2 or (
            info["has_lo2"]
            and any(any(h["lo2"] for h in p["helicities"]) for p in data["points"])
        )
        clamped = clamped or any(
            p["weight"][6] != p["weight"][7] for p in data["points"]
        )
    assert kinds == {"goldstone", "cew", "iz1", "iz2", "ipm2"}
    assert schemes == {"gmu", "alpha_mz"}
    assert lo2 and clamped


def test_scheme_and_entry_checks():
    data = reference(DATA[0])
    configurations = data["fks_configurations"]
    p = parameters(data)
    other = ms.EWSudakovScheme(
        "alpha_mz" if configurations["ewsudakov"]["scheme"] == "gmu" else "gmu"
    )
    wrong = ms.EWSudakovParameters(
        mw=p.mw, mz=p.mz, mh=p.mh, mt=p.mt, alpha=p.alpha, gf=p.gf, scheme=other
    )
    with pytest.raises(ValueError, match="scheme"):
        ew_sudakov(configurations, wrong)
    without = {k: v for k, v in configurations.items() if k != "ewsudakov"}
    with pytest.raises(ValueError, match="ewsudakov"):
        ew_sudakov_process(without)
    # the default modes are those of compute_ewsudakov
    assert ew_sudakov(configurations, p).mode_count() == 2


def test_invalid_processes():
    p = ms.EWSudakovParameters(mw=80.4, mz=91.2, mh=125.0, mt=173.0, alpha=1 / 132.5)
    hel = [[1, -1, 1, -1]]
    ok = ms.EWSudakovProcess(pdgs=[2, -2, 1, -1], helicities=hel)
    assert ms.EWSudakov(ok, p).interference_terms() == []
    bad = [
        ms.EWSudakovProcess(pdgs=[2, -2, 1, -1], helicities=[[1, -1, 1]]),
        ms.EWSudakovProcess(pdgs=[2, -2, 1, -1], helicities=[]),
        ms.EWSudakovProcess(pdgs=[2, -2, 1, -1], helicities=hel, incoming=1),
        ms.EWSudakovProcess(
            pdgs=[2, -2, 1, -1],
            helicities=hel,
            interference=[
                ms.EWSudakovAmplitude(type="ipm2", legs=[0], old_pdgs=[2], new_pdgs=[1])
            ],
        ),
        ms.EWSudakovProcess(
            pdgs=[2, -2, 1, -1],
            helicities=hel,
            interference=[
                ms.EWSudakovAmplitude(
                    type="cew", base=1, legs=[2], old_pdgs=[1], new_pdgs=[2]
                )
            ],
        ),
        ms.EWSudakovProcess(
            pdgs=[2, -2, 1, -1],
            helicities=hel,
            goldstone=[
                ms.EWSudakovAmplitude(
                    type="goldstone", legs=[7], old_pdgs=[24], new_pdgs=[251]
                )
            ],
        ),
    ]
    for process in bad:
        with pytest.raises(ValueError):
            ms.EWSudakov(process, p)
    with pytest.raises(ValueError, match="sud_mod"):
        ms.EWSudakov(ok, p, [ms.EWSudakovMode(sud_mod=2)])
    with pytest.raises(ValueError, match="modes"):
        ms.EWSudakov(ok, p, [])
    with pytest.raises(ValueError):
        ms.EWSudakov(
            ok, ms.EWSudakovParameters(mw=91.3, mz=91.2, mh=125.0, mt=173.0, alpha=0.01)
        )
