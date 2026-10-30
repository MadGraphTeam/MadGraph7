# SPDX-License-Identifier: 0BSD
"""The QCD+QED loop UFOs as UFOmodeldb ships them (m-sm, package MS-UFODB;
user 2026-10-30: https://github.com/mg5amcnlo/UFOmodeldb is the ground truth).

* The shipped loop_qcd_qed_sm and loop_qcd_qed_sm_Gmu are UFOmodeldb's
  (commit 6ca4b120). MG5_aMC's online-database downloads differ from them in
  one declaration, the R2_ZZHH coupling (UFOmodeldb's fix 1f13c48f /
  6ca4b120); both have the same shipped ``[loop.model]`` profile.
* R2_ZZHH enters the R2 vertices Z Z H H and Z Z G0 G0 only. A counterterm
  vertex is a tree vertex inserted once (ADDENDUM J) and the bare loops use
  tree couplings only, so a request none of whose trees can hold such a
  vertex generates the same plan on either model. Of the M-SM campaign rows
  only SC10 (z z > h h) can (``_reaches``, a necessary condition).
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


def _mg5_download(name: str, directory: Path) -> Path:
    """The shipped model with R2_ZZHH as MG5_aMC's online database has it."""

    target = directory / name
    shutil.copytree(_shipped(name), target)
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
def test_shipped_model_is_ufodb_and_the_mg5_download_shares_its_profile(tmp_path, name):
    """The shipped model's R2_ZZHH is UFOmodeldb's; the MG5_aMC download's
    declarations differ in that coupling record alone and take the same
    shipped profile (by its second digest)."""

    from pyamplicol.loops.shipped_profiles import (
        shipped_loop_profile,
        source_declarations_digest,
    )
    from pyamplicol.models.loop_import import load_raw_loop_model

    ufodb = load_raw_loop_model(_shipped(name), restriction="full")
    mg5 = load_raw_loop_model(_mg5_download(name, tmp_path), restriction="full")
    assert UFODB_FACTOR in _coupling(ufodb, COUPLING)["fields"]["value"]["expression"]
    assert MG5_FACTOR in _coupling(mg5, COUPLING)["fields"]["value"]["expression"]
    a, b = ufodb.to_dict()["collections"], mg5.to_dict()["collections"]
    assert {key for key in a if a[key] != b[key]} == {"couplings"}
    changed = [
        (x["name"], y["name"])
        for x, y in zip(a["couplings"], b["couplings"], strict=True)
        if x != y
    ]
    assert changed == [(COUPLING, COUPLING)]
    profile = shipped_loop_profile(ufodb)
    assert profile is not None and shipped_loop_profile(mg5) is profile
    assert profile.source_digests == (
        source_declarations_digest(ufodb),
        source_declarations_digest(mg5),
    )


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


@pytest.mark.loop_nightly
def test_sc10_equals_madloop_on_both_models(loop_native, tmp_path):
    """z z > h h [virt=QED] on the shipped (UFOmodeldb) G_mu model equals
    MadLoop QP on it (``required/msm/zz_hh_ew.json``; binary64, its metric:
    relative, Laurent floor |Born| alpha/2pi, 1e-8), and on the MG5_aMC
    download (``zz_hh_ew_mg5.json``) likewise; the change between the models
    is the R2 component alone and equals MadLoop's change to 1e-10 of it
    (about 9 min)."""

    import math

    reference = {
        model: json.loads((FIXTURES / f"required/msm/{key}.json").read_text())
        for model, key in (("ufodb", "zz_hh_ew"), ("mg5", "zz_hh_ew_mg5"))
    }
    assert reference["ufodb"]["points"] == reference["mg5"]["points"]
    ufos = {
        "ufodb": _shipped("loop_qcd_qed_sm_Gmu"),
        "mg5": _mg5_download("loop_qcd_qed_sm_Gmu", tmp_path),
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
