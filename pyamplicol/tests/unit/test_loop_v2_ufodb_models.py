# SPDX-License-Identifier: 0BSD
"""The QCD+QED loop UFOs as UFOmodeldb ships them (m-sm, package MS-UFODB;
user 2026-10-30: https://github.com/mg5amcnlo/UFOmodeldb is the ground truth).

* The shipped loop_qcd_qed_sm and loop_qcd_qed_sm_Gmu are UFOmodeldb's
  (commit 6ca4b120) with the user's model patch of UFOmodeldb issue 1 /
  mg5amcnlo#475 (user rule 2026-10-30 ~12:45; package MS-UFO-PATCH):
  R2_G0tt_{d,s,b} split by loop content ([W+, G+, q] and [G+, q]; the
  coefficient sum is kept), an exactly unitary CKM from the same Wolfenstein
  inputs, the G_mu charge counterterm's b-mass term. UFOmodeldb's unpatched
  models (``*_ufodb_unpatched_ufo``, controls) and MG5_aMC's online-database
  downloads (which differ from those in one declaration, the R2_ZZHH
  coupling: UFOmodeldb's fix 1f13c48f / 6ca4b120) have the same shipped
  ``[loop.model]`` profile.
* R2_ZZHH enters the R2 vertices Z Z H H and Z Z G0 G0 only. A counterterm
  vertex is a tree vertex inserted once (ADDENDUM J) and the bare loops use
  tree couplings only, so a request none of whose trees can hold such a
  vertex generates the same plan on either model. Of the M-SM campaign rows
  only SC10 (z z > h h) can (``_reaches``, a necessary condition).
* Recola (R2 terms of its own) equals MadLoop on the UFOmodeldb model for
  SC10 and not MadLoop on the MG5_aMC download: the fix is right.
* The defect the patch fixes: with a massive b, UFOmodeldb's G_mu charge
  counterterm omits its b-mass term (ST04's Recola offset, exactly); patched,
  eCoup_a0_UV_EW + eCoup_DGmua0_UV_EW is the full G_mu charge counterterm
  eCoup_Gmu_UV_EW, and unchanged at MB = 0.
* SC10 equals MadLoop in quadruple precision on the UFOmodeldb model, and the
  difference to MadLoop on the MG5_aMC download is the R2_ZZHH term alone
  (nightly; the Python reference path of the slice: the native plan of this
  process is refused at load, MS-VALIDATE finding F1).
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from fractions import Fraction
from itertools import product
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
CAMPAIGN = ROOT / "tools" / "loop_reference" / "msm_campaign.json"
SPEC = ROOT / "tools" / "loop_reference" / "required_processes.json"
MODELS = ("loop_qcd_qed_sm", "loop_qcd_qed_sm_Gmu")
# UFOmodeldb's models without the issue-1 patch (conftest controls).
UNPATCHED = {
    "loop_qcd_qed_sm": "loop_qcd_qed_sm_ufodb_unpatched_ufo",
    "loop_qcd_qed_sm_Gmu": "loop_gmu_ufodb_unpatched_ufo",
}
COUPLING = "R2_ZZHH"
# The coupling's factor before (MG5_aMC's online database) and after
# UFOmodeldb's fix.
MG5_FACTOR = "(2+2*cw**2 +40*cw**4 -4*cw**6)"
UFODB_FACTOR = "(1+2*cw**2 +40*cw**4 -4*cw**6)"


def _shipped(name: str) -> Path:
    from pyamplicol._internal.packaged_models import packaged_ufo_model

    path = packaged_ufo_model(name)
    assert path is not None, name
    return path


def _mg5_download(unpatched: Path, directory: Path) -> Path:
    """MG5_aMC's online-database model: UFOmodeldb's unpatched model with
    R2_ZZHH as the database has it."""

    target = directory / unpatched.name
    shutil.copytree(
        unpatched, target, ignore=shutil.ignore_patterns("__pycache__", "*.pkl")
    )
    couplings = target / "CT_couplings.py"
    text = couplings.read_text(encoding="utf-8")
    assert text.count(UFODB_FACTOR) == 1
    couplings.write_text(text.replace(UFODB_FACTOR, MG5_FACTOR), encoding="utf-8")
    return target


def _coupling(raw, name: str) -> dict:
    (record,) = [
        r for r in raw.to_dict()["collections"]["couplings"] if r["name"] == name
    ]
    return record


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", MODELS)
def test_shipped_model_is_patched_ufodb_and_older_copies_share_its_profile(
    request, tmp_path, name
):
    """The shipped model's R2_ZZHH is UFOmodeldb's; MG5_aMC's download
    differs from UFOmodeldb's unpatched model in that coupling record alone;
    the shipped (patched), unpatched and download declarations take the same
    shipped profile, in that digest order."""

    from pyamplicol.loops.shipped_profiles import (
        shipped_loop_profile,
        source_declarations_digest,
    )
    from pyamplicol.models.loop_import import load_raw_loop_model

    unpatched = Path(request.getfixturevalue(UNPATCHED[name]))
    shipped = load_raw_loop_model(_shipped(name), restriction="full")
    ufodb = load_raw_loop_model(unpatched, restriction="full")
    mg5 = load_raw_loop_model(_mg5_download(unpatched, tmp_path), restriction="full")
    for raw in (shipped, ufodb):
        assert UFODB_FACTOR in _coupling(raw, COUPLING)["fields"]["value"]["expression"]
    assert MG5_FACTOR in _coupling(mg5, COUPLING)["fields"]["value"]["expression"]
    a, b = ufodb.to_dict()["collections"], mg5.to_dict()["collections"]
    assert {key for key in a if a[key] != b[key]} == {"couplings"}
    changed = [
        (x["name"], y["name"])
        for x, y in zip(a["couplings"], b["couplings"], strict=True)
        if x != y
    ]
    assert changed == [(COUPLING, COUPLING)]
    profile = shipped_loop_profile(shipped)
    assert profile is not None
    assert shipped_loop_profile(ufodb) is profile
    assert shipped_loop_profile(mg5) is profile
    digests = [source_declarations_digest(raw) for raw in (shipped, ufodb, mg5)]
    # The shipped digest first; UFOmodeldb 60ee58bb's (the b-flavour R2 split
    # alone) is the second.
    assert profile.source_digests[0] == digests[0]
    assert len(profile.source_digests) == 4
    assert profile.source_digests[2:] == tuple(digests[1:])


# ------------------------------------------------- which requests reach it


def _particles(raw) -> dict[str, dict]:
    return {r["name"]: r["fields"] for r in raw.to_dict()["collections"]["particles"]}


def _triality(fields: dict) -> int:
    return {3: 1, -3: 2}.get(fields["color"], 0)


def _reaches(legs, externals, particles) -> bool:
    """Whether a tree of the outgoing ``externals`` (particle names) can hold
    the vertex of the outgoing fields ``legs`` once: a necessary condition.
    The externals split into one non-empty branch per leg. A one-particle
    branch is the leg's particle or antiparticle (the model has no two-point
    tree vertex); a longer one is a tree current of the leg's field, so its
    fermion number parity, electric charge and colour triality match the
    field's up to conjugation."""

    def fermion(f):
        return f["spin"] % 2 == 0

    def matches(leg, branch):
        field = particles[leg]
        if len(branch) == 1:
            return branch[0] in (leg, field["antiname"])
        members = [particles[name] for name in branch]
        charge = sum(round(3 * f["charge"]) for f in members)  # in units of e/3
        triality = sum(_triality(f) for f in members) % 3
        return (
            sum(map(fermion, members)) % 2 == fermion(field)
            and abs(charge) == abs(round(3 * field["charge"]))
            and triality in (_triality(field), (-_triality(field)) % 3)
        )

    for assignment in product(range(len(legs)), repeat=len(externals)):
        branches = [
            [e for e, a in zip(externals, assignment, strict=True) if a == leg]
            for leg in range(len(legs))
        ]
        if all(branches) and all(
            matches(leg, branch) for leg, branch in zip(legs, branches, strict=True)
        ):
            return True
    return False


def _row_externals(row, runs, spec, particles) -> list[list[str]]:
    """The outgoing externals (particle names) of a campaign row's processes:
    its runs' PDG codes, else its process text."""

    from pyamplicol.processes.model import resolve_model_particle_name

    by_pdg = {fields["pdg_code"]: name for name, fields in particles.items()}
    processes = []
    for run_id in row["runs"]:
        run = runs[run_id]
        pdgs = run.get("pdgs") or spec[run["process"]]["pdgs"]
        processes.append([by_pdg[p] for p in pdgs])
    if not processes:
        text = row["process"].split("(")[0].split("/")[0]
        # without the annotations (orders, "CMS")
        tokens = [
            t for t in text.replace(">", " ").split() if "=" not in t and t != "CMS"
        ]
        aliases = {"mu-": "m-", "mu+": "m+"}  # MG5_aMC's muon labels
        processes.append(
            [
                resolve_model_particle_name(
                    aliases.get(t, t), tuple(particles), model_name="G_mu"
                )
                for t in tokens
            ]
        )
    # all outgoing: the two incoming legs as their antiparticles
    return [
        [particles[n]["antiname"] if i < 2 else n for i, n in enumerate(names)]
        for names in processes
    ]


@pytest.mark.loop_fast
def test_only_sc10_reaches_the_r2_zzhh_vertices():
    """R2_ZZHH is a counterterm coupling of the R2 vertices Z Z H H and
    Z Z G0 G0 only, and no tree vertex has two legs; of every M-SM campaign
    row on the QCD+QED models only SC10 (z z > h h) can hold one of them in a
    tree, so every other row's plan is the same on the MG5_aMC download and
    on UFOmodeldb's model."""

    from pyamplicol.models.loop_import import load_raw_loop_model

    raw = load_raw_loop_model(_shipped("loop_qcd_qed_sm_Gmu"), restriction="full")
    collections = raw.to_dict()["collections"]
    particles = _particles(raw)

    def uses(record):
        return any(
            ref["name"] == COUPLING
            for _, ref in record["fields"]["couplings"]["entries"]
        )

    assert not any(uses(v) for v in collections["vertices"])
    assert min(len(v["fields"]["particles"]) for v in collections["vertices"]) >= 3
    vertices = {
        v["name"]: [p["name"] for p in v["fields"]["particles"]]
        for v in collections["counterterm_vertices"]
        if uses(v)
    }
    assert {k: Counter(v) for k, v in vertices.items()} == {
        "V_R2ZZHH": Counter({"Z": 2, "H": 2}),
        "V_R2ZZG0G0": Counter({"Z": 2, "G0": 2}),
    }
    assert all(
        v["fields"]["type"] == "R2"
        for v in collections["counterterm_vertices"]
        if uses(v)
    )

    campaign = json.loads(CAMPAIGN.read_text())
    runs = {r["id"]: r for r in campaign["runs"]}
    spec = {p["name"]: p for p in json.loads(SPEC.read_text())["processes"]}
    codes = campaign["restriction_codes"]
    reached, checked = set(), 0
    for row in campaign["rows"]:
        if not codes[row["restriction_code"]]["model"].startswith("loop_qcd_qed_sm"):
            continue  # loop_sm has no such coupling
        for externals in _row_externals(row, runs, spec, particles):
            checked += 1
            if any(_reaches(legs, externals, particles) for legs in vertices.values()):
                reached.add(row["row"])
    assert checked >= 70
    assert reached == {"SC10"}


# ------------------------------------------------------- SC10 = MadLoop QP


def _sc10(ufo, loop_native):
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import compile_model_source

    fixed = {"MB": "0", "ymb": "0", "MTA": "0", "ymtau": "0"}
    fixed |= {name: "0" for name in ("WT", "WW", "WZ", "WH")}
    return LoopSlice.generate(
        compile_model_source(ufo, require_supported=False),
        SliceRequest(
            "z z > h h",
            {"QCD": 0, "QED": 2},
            {"QCD": 0, "QED": 4},
            corrections=("QED",),
            fixed_parameters=fixed,
            scale_parameter="MU_R",
        ),
        loop_native,
    )


@pytest.mark.loop_fast
def test_recola_decides_the_r2_zzhh_fix():
    """Recola's R2 terms do not come from the UFO: on SC10 it equals MadLoop
    on UFOmodeldb's model (to 2.2e-11 of the finite part) and not MadLoop on
    the MG5_aMC download (3.7 % to 2.7 times the finite part)."""

    fixtures = {
        key: json.loads((FIXTURES / f"required/msm/{key}.json").read_text())
        for key in ("zz_hh_ew", "zz_hh_ew_mg5")
    }
    ufodb, mg5 = fixtures["zz_hh_ew"], fixtures["zz_hh_ew_mg5"]
    assert ufodb["validated_providers"] == ["madloop", "recola"]
    assert ufodb["points"] == mg5["points"]
    for row, control in zip(ufodb["rows"], mg5["rows"], strict=True):
        recola = row["providers"]["recola"]["finite"]
        madloop = row["providers"]["madloop"]["finite"]
        download = control["providers"]["madloop"]["finite"]
        assert abs(madloop - recola) <= 1e-10 * abs(recola)
        assert abs(download - recola) >= 1e-2 * abs(recola)


def _source_values(ufo: Path, card: dict, tag: str):
    """(CT parameter values by (name, epsilon power), the parameters'
    namespace, a coupling evaluator) of a UFO at ``card`` (binary64; the
    UFO's own Python source, ``tools/loop_reference/ufo_source_oracle.py``)."""

    from tools.loop_reference import ufo_source_oracle as oracle

    modules = oracle.import_ufo(ufo, f"_pyamplicol_test_ufodb_{tag}")
    values, environment = oracle.ufo_source_values(
        modules, card, exact=False, wrap=None
    )
    scope = {**environment, **{k: v for (k, p), v in values.items() if p == 0}}

    def coupling(source: str) -> complex:
        return complex(eval(oracle.compiled(source, False), scope))

    return values, environment, coupling, modules[3]


# The UFO cards (decimal strings) of the checks below: restrict_ckm's
# Wolfenstein inputs, massive d, s, b.
CKM_CARD = {"lamWS": "0.2253", "AWS": "0.808", "rhoWS": "0.132", "etaWS": "0.341"}
MASS_CARD = {"MD": "0.3", "MS": "0.5", "MB": "4.5", "ymb": "4.5"}


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", MODELS)
def test_the_issue_1_patch_changes_what_it_says(request, name):
    """The shipped model against UFOmodeldb's unpatched one: the records that
    differ are V_ttG02, the R2_G0tt couplings (R2_G0tt_{d,s,b} split into
    _WG, label [W+, G+, q], and _G, label [G+, q]; each pair sums to the
    unpatched coupling at massive quarks and a complex CKM), the CKM
    parameters (now unitary to rounding; the unpatched one is off by
    O(lambda^4)) and, for G_mu only, the two charge counterterms."""

    import cmath

    from pyamplicol.models.loop_import import load_raw_loop_model

    unpatched = Path(request.getfixturevalue(UNPATCHED[name]))
    new = load_raw_loop_model(_shipped(name), restriction="full").to_dict()
    old = load_raw_loop_model(unpatched, restriction="full").to_dict()

    def plain(item):
        # Record ids and source lines shift with the patch's added records.
        if isinstance(item, dict):
            return {
                k: plain(v) for k, v in item.items() if k not in ("id", "value_source")
            }
        return [plain(v) for v in item] if isinstance(item, list) else item

    def values(raw, collection):
        return {r["name"]: plain(r["fields"]) for r in raw["collections"][collection]}

    def changed(collection):
        a, b = values(old, collection), values(new, collection)
        return {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}

    split = {f"R2_G0tt_{q}" for q in "dsb"}
    pieces = {f"{c}_{part}" for c in split for part in ("WG", "G")}
    assert changed("couplings") == split | pieces
    assert changed("counterterm_vertices") == {"V_ttG02"}
    ckm = {f"CKM{i}{j}" for i in "123" for j in "123"}
    assert changed("parameters") == ckm | {"CKMc12", "CKMc13", "CKMc23", "CKMs13e"}
    charge = {"eCoup_a0_UV_EW", "eCoup_DGmua0_UV_EW"}
    assert changed("counterterm_parameters") == (charge if "Gmu" in name else set())
    for key in ("vertices", "particles", "lorentz_structures", "propagators"):
        assert values(old, key) == values(new, key), key
    (vertex,) = [
        r["fields"]
        for r in new["collections"]["counterterm_vertices"]
        if r["name"] == "V_ttG02"
    ]
    labels = [sorted(p["name"] for p in group[0]) for group in vertex["loop_particles"]]
    expected = [["Z", "t"]]
    for q in "dsb":
        expected += [sorted(["W+", "G+", q]), sorted(["G+", q])]
    assert labels == expected

    card = {**CKM_CARD, **MASS_CARD}
    _, env, coupling, couplings = _source_values(_shipped(name), card, f"new_{name}")
    old_sources = {r["name"]: r for r in old["collections"]["couplings"]}
    for c in split:
        whole = old_sources[c]["fields"]["value"]["expression"]
        parts = getattr(couplings, f"{c}_WG").value, getattr(couplings, f"{c}_G").value
        assert coupling(whole) != 0
        total = coupling(parts[0]) + coupling(parts[1])
        assert abs(total - coupling(whole)) <= 1e-15 * abs(coupling(whole)), c

    def unitarity(environment):
        v = [[environment[f"CKM{i}{j}"] for j in "123"] for i in "123"]
        return max(
            abs(sum(v[i][k] * v[j][k].conjugate() for k in range(3)) - (i == j))
            for i in range(3)
            for j in range(3)
        )

    _, env_old, _, _ = _source_values(unpatched, card, f"old_{name}")
    assert unitarity(env) <= 1e-15
    assert 1e-4 < unitarity(env_old) < 1e-2
    assert cmath.isclose(env["CKM12"], env["lamWS"] * env["CKMc13"])


@pytest.mark.loop_fast
def test_gmu_charge_counterterm_has_its_b_mass_term(loop_gmu_ufodb_unpatched_ufo):
    """With a massive b (MB = 4.5), the patched G_mu model's charge
    counterterms eCoup_a0_UV_EW + eCoup_DGmua0_UV_EW (what its 186 CT
    couplings use) equal the full G_mu charge counterterm eCoup_Gmu_UV_EW in
    both Laurent coefficients, while UFOmodeldb's unpatched model misses its
    b-mass term dMB_eCoup_UV_EW (-2.37010e-5 at MB = 4.5, any mu_R): the
    defect behind ST04's Recola offset (-4 dMB_eCoup_UV_EW x Born =
    +9.48039e-5 x Born on ``required/udx_tbx_ew_mb.json``, MadLoop on the
    unpatched model). At MB = 0 both counterterms are the unpatched ones
    exactly."""

    import re

    from pyamplicol.models.loop_import import load_raw_loop_model

    unpatched = Path(loop_gmu_ufodb_unpatched_ufo)
    shipped = _shipped("loop_qcd_qed_sm_Gmu")
    raw = load_raw_loop_model(unpatched, restriction="full")
    sources = [
        r["fields"]["value_source"]["argument"]
        for r in raw.to_dict()["collections"]["couplings"]
        if r["fields"]["value_source"]["file"] == "CT_couplings.py"
    ]
    assert sum("(eCoup_a0_UV_EW+eCoup_DGmua0_UV_EW)" in v for v in sources) == 186
    assert not any(re.search(r"\beCoup_Gmu_UV_EW\b", v) for v in sources)

    a0, dg, gmu, b = (
        "eCoup_a0_UV_EW",
        "eCoup_DGmua0_UV_EW",
        "eCoup_Gmu_UV_EW",
        "dMB_eCoup_UV_EW",
    )
    fixture = json.loads((FIXTURES / "required/udx_tbx_ew_mb.json").read_text())
    card = {**fixture["parameters"], **fixture["fixed_parameters"]}
    assert card["MB"] == "4.5"
    new, _, _, _ = _source_values(shipped, card, "charge_new")
    old, _, _, _ = _source_values(unpatched, card, "charge_old")
    assert abs(old[b, 0] - -2.3700982771e-5) <= 1e-9 * abs(old[b, 0])
    for power in (-1, 0):
        full = new[gmu, power]
        assert abs(new[a0, power] + new[dg, power] - full) <= 1e-13 * abs(full)
        missing = full - old[a0, power] - old[dg, power]
        assert abs(missing - (old[b, 0] if power == 0 else 0)) <= 1e-13 * abs(full)
    zero = {**card, "MB": "0", "ymb": "0"}
    new0, _, _, _ = _source_values(shipped, zero, "charge_new0")
    old0, _, _, _ = _source_values(unpatched, zero, "charge_old0")
    for key in ((a0, -1), (a0, 0), (dg, -1), (dg, 0)):
        assert new0[key] == old0[key], key

    offsets = [
        (r["providers"]["madloop"]["finite"] - r["providers"]["recola"]["finite"])
        / r["providers"]["madloop"]["born"]
        for r in fixture["rows"]
    ]
    assert all(abs(x - 4 * 2.3700982771e-5) <= 1e-6 * x for x in offsets), offsets


@pytest.mark.loop_nightly
def test_sc10_equals_madloop_on_both_models(
    loop_native, loop_gmu_ufodb_unpatched_ufo, tmp_path
):
    """z z > h h [virt=QED] on the shipped G_mu model (UFOmodeldb's with the
    issue-1 patch, which SC10 does not reach) equals MadLoop QP on UFOmodeldb's
    model (``required/msm/zz_hh_ew.json``; binary64, its metric: relative,
    Laurent floor |Born| alpha/2pi, 1e-8), and on the MG5_aMC download
    (``zz_hh_ew_mg5.json``) likewise; the change between the models is the R2
    component alone and equals MadLoop's change to 1e-10 of it (about 9
    min)."""

    import math

    reference = {
        model: json.loads((FIXTURES / f"required/msm/{key}.json").read_text())
        for model, key in (("ufodb", "zz_hh_ew"), ("mg5", "zz_hh_ew_mg5"))
    }
    assert reference["ufodb"]["points"] == reference["mg5"]["points"]
    ufos = {
        "ufodb": _shipped("loop_qcd_qed_sm_Gmu"),
        "mg5": _mg5_download(Path(loop_gmu_ufodb_unpatched_ufo), tmp_path),
    }
    first = reference["ufodb"]
    alpha = float(first["derived_parameters"]["alpha"])
    fixed = first["fixed_parameters"]
    parameters = {k: v for k, v in first["parameters"].items() if k not in fixed}
    values = {}
    for model, ufo in ufos.items():
        slice_ = _sc10(ufo, loop_native)
        assert all(fixed[k] == v for k, v in slice_.request.fixed_parameters.items())
        values[model] = [
            slice_.evaluate(point, str(Fraction(mu) ** 2), parameters=parameters)
            for point, mu in zip(first["points"], first["mu_r"], strict=True)
        ]
    for k, (ours_u, ours_m) in enumerate(
        zip(values["ufodb"], values["mg5"], strict=True)
    ):
        (born,) = ours_u.born.values()
        (sector,) = ours_u.components["virtual"]
        floor = abs(complex(born)) * alpha / (2 * math.pi)
        for model, ours in (("ufodb", ours_u), ("mg5", ours_m)):
            row = reference[model]["rows"][k]
            assert abs(complex(born).real - row["born"]) <= 1e-12 * row["born"]
            laurent = ours.components["virtual"][sector].coefficients
            for i, key in enumerate(("finite", "single_pole", "double_pole")):
                found = complex(laurent[i]).real
                error = abs(found - row[key]) / max(abs(found), abs(row[key]), floor)
                assert error <= 1e-8, (model, k, key, found, row[key])
        for name in ("bare", "uv"):
            assert (
                ours_u.components[name][sector].coefficients
                == ours_m.components[name][sector].coefficients
            ), (k, name)
        change = complex(ours_u.components["r2"][sector].coefficients[0]) - complex(
            ours_m.components["r2"][sector].coefficients[0]
        )
        madloop = (
            reference["ufodb"]["rows"][k]["finite"]
            - reference["mg5"]["rows"][k]["finite"]
        )
        assert change.real != 0
        assert abs(change.real - madloop) <= 1e-10 * abs(madloop), (k, change, madloop)


# ------------------------------------- the rows the patch changes (nightly)

# Run (fixture required/msm/<run>.json, MadLoop QP and Recola on the patched
# model) -> restriction of the shipped G_mu model.
PATCHED_ROWS = {
    "ttx_ttx_ew": "default",  # TT04: R2_G0tt_b, now inserted by MadLoop
    "ttx_zh_ew": "default",  # SC09: likewise
    "gt_tttx_ew": "default",  # TT05: likewise (double-double)
    "udx_tbx_ew_mb": "with_b_mass",  # ST04: the G_mu b-mass charge counterterm
    "gb_ttxb_ew_mb": "with_b_mass",  # TT06: both (double-double)
}
# Rows evaluated in double-double: binary64 loses up to 1e-4 to m_b^2/s
# cancellations (TT06) and 1e-9 (TT05), which their stability estimates flag.
DOUBLE_DOUBLE = {"gt_tttx_ew", "gb_ttxb_ew_mb"}


@pytest.mark.loop_nightly
@pytest.mark.parametrize("run", sorted(PATCHED_ROWS))
def test_patched_rows_equal_madloop_and_recola(run, loop_native):
    """On the shipped (patched) G_mu model, TT04, SC09, TT05, ST04 and TT06
    equal MadLoop QP on the patched model and Recola, Born and every Laurent
    coefficient (native executor, binary64 or double-double; the fixtures'
    metric: relative, Laurent floor |Born| alpha/2pi; 1e-9, the vanishing QED
    double pole of t t~ > z h in binary64 cancellations reaching 2e-10).
    Before the patch MadLoop missed Recola by 2.5 % (TT04), 7.7 % (SC09) and
    1e-3 (ST04), and omitted R2_G0tt_b on TT05. About 3 to 6 min each."""

    import argparse
    import math

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision
    from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan
    from pyamplicol.models.loading import compile_model_source
    from tools import loop_v2_generate

    reference = json.loads((FIXTURES / f"required/msm/{run}.json").read_text())
    assert reference["validated_providers"] == ["madloop", "recola"]
    compiled = compile_model_source(
        _shipped("loop_qcd_qed_sm_Gmu"),
        restriction=PATCHED_ROWS[run],
        require_supported=False,
    )
    request, _ignored = loop_v2_generate._fixture_request(
        reference, compiled, argparse.Namespace()
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    executor = NativeExecutor(
        slice_, forward=False, exported=export_plan(slice_, forward=False)
    )
    results = executor.evaluate(
        reference["points"],
        [str(Fraction(mu) ** 2) for mu in reference["mu_r"]],
        parameters=dict(reference["parameters"]),
        precision=Precision.double_double() if run in DOUBLE_DOUBLE else None,
    )
    alpha = float(reference["derived_parameters"]["alpha"])
    keys = ("born", "finite", "single_pole", "double_pole")
    for index, (result, row) in enumerate(zip(results, reference["rows"], strict=True)):
        mine = [float(complex(x).real) for x in result.totals[:4]]
        for provider in ("madloop", "recola"):
            values = row["providers"][provider]
            floor = abs(values["born"]) * alpha / (2 * math.pi)
            for k, (found, key) in enumerate(zip(mine, keys, strict=True)):
                want = values[key]
                scale = max(abs(found), abs(want), floor if k else 0.0)
                assert abs(found - want) <= 1e-9 * scale, (run, index, provider, key)


@pytest.mark.loop_fast
def test_a_model_writing_values_after_their_constructors_is_a_listed_refusal(tmp_path):
    """The UFOmodeldb issue-1 patch as published extends the G_mu charge
    counterterms by module-level ``X.value[k] += '...'``; ufo-model-loader then
    certifies no value of the model, which one-loop requests need: the
    listed refusal ``ufo-value-writes`` (naming the fold tool), not an
    untyped error. The shipped model has the appends folded."""

    from pyamplicol.loops.scalar_frontend import ScalarFunctionLibrary
    from pyamplicol.models.loop_import import load_raw_loop_model
    from pyamplicol.models.source_refusals import (
        ModelSourceUnsupported,
        listed_model_source_refusal_keys,
    )

    assert "ufo-value-writes" in listed_model_source_refusal_keys()
    shipped = _shipped("loop_qcd_qed_sm_Gmu")
    ScalarFunctionLibrary(load_raw_loop_model(shipped, restriction="full"))
    target = tmp_path / "loop_qcd_qed_sm_Gmu_written"
    shutil.copytree(shipped, target, ignore=shutil.ignore_patterns("__pycache__"))
    with (target / "CT_parameters.py").open("a", encoding="utf-8") as stream:
        stream.write("\neCoup_a0_UV_EW.value[0] += ''\n")
    raw = load_raw_loop_model(target, restriction="full")
    with pytest.raises(ModelSourceUnsupported) as caught:
        ScalarFunctionLibrary(raw)
    assert caught.value.key == "ufo-value-writes"
    assert "fold_ufo_value_appends.py" in str(caught.value)
