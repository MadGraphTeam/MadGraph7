# SPDX-License-Identifier: 0BSD
"""Stored one-loop processes whose certified-scaleless companion's Born
couplings call the model's helper functions (m-lit lit-jj04).

The native parameter program of a plan (``native_trees._parameter_program``)
compiles its outputs with the scalar library's helper functions
(``<namespace>::value_N`` / ``diagnostic_N``: the UFO function library's
``complexconjugate``, ``re``, ...). It registered them only for the plans its
counterterm and field couplings read, but the Born couplings call them too: a
CKM coupling's ``complexconjugate`` (loop_sm, even at CKM = 1), G_mu's
``re()`` through ``Gfbar``. The certified-scaleless companion
(``LoopSlice.scaleless``, which the "uv" and "ir" pole selections run) is
exported without counterterms, so it registered none, and Symbolica refused
its evaluator ("function value_N(..) is not in a parameter map"): storing
such a request failed, typed, with nothing stored and no wrong value, and so
did the in-memory "uv" / "ir" selections with native point inputs. Every
plan library of which an output calls an unregistered helper is now
registered whole, which changes no program that compiled before.

* ``u u~ > d d~ g`` [QCD] with both Born orders (caps QCD 3, QED 2) on
  loop_sm/no_b_mass and on loop_qcd_qed_sm_Gmu stores with its companion,
  and the stored "all", "uv" and "ir" selections equal the in-memory slice
  with Python point inputs (``NativeExecutor(inputs="python")``: no native
  parameter program) per sector within the repository's native-vs-Python
  input bound (``test_loop_v2_ew_lines.TOLERANCE``);
* nightly: M-SM row JJ04 (the same request with [QCD, QED] corrections)
  stored equals MadLoop (``required/msm/uux_ddxg_ew.json``) in the
  fixture's metric and tolerance, and its stored selections equal the
  in-memory ones within the two binary64 evaluations' own estimates (the
  1e-12 bound is below this request's binary64 accuracy).
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MADLOOP = ROOT / "tests" / "fixtures" / "loop" / "required" / "msm" / "uux_ddxg_ew.json"
PROCESS = "u u~ > d d~ g"
# The native-vs-Python point-input bound of the repository
# (``test_loop_v2_ew_lines.TOLERANCE``), of each sector's virtual scale.
TOLERANCE = 1e-12
WIDTHS = {"WT": "0", "WW": "0", "WZ": "0", "WH": "0"}
SELECTIONS = ("all", "uv", "ir")


def _madloop() -> dict:
    return json.loads(MADLOOP.read_text())


def _card(output: Path, model: str, restriction: str, loop_orders, fixed) -> str:
    fixed_text = ", ".join(f"{name} = {json.dumps(v)}" for name, v in fixed.items())
    return f"""
action = "generate"
[model]
source = {json.dumps(model)}
restriction = {json.dumps(restriction)}
[process]
entries = [{{expression = {json.dumps(PROCESS)}, name = "p"}}]
loop_coupling_orders = {json.dumps(list(loop_orders))}
coupling_order_policy = "explicit"
max_coupling_orders = {{ QCD = 3, QED = 2 }}
[loop.preparation]
fixed_parameters = {{{fixed_text}}}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
"""


def _store(root: Path, monkeypatch, model, restriction, loop_orders, fixed):
    """The request stored through the public Generator, with the slice it
    generated (kept from ``artifact.generate_process``, which nothing else
    keeps)."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.loops.v2 import artifact

    generated = []
    original = artifact.generate_process

    def generate(*args, **kwargs):
        found = original(*args, **kwargs)
        generated.append(found)
        return found

    monkeypatch.setattr(artifact, "generate_process", generate)
    root.mkdir(parents=True, exist_ok=True)
    card = root / "card.toml"
    card.write_text(_card(root / "artifact", model, restriction, loop_orders, fixed))
    Generator(resolve_config(card)).generate(
        ProcessSet((ProcessRequest.parse(PROCESS, name="p"),)), root / "artifact"
    )
    monkeypatch.setattr(artifact, "generate_process", original)
    (process,) = generated
    return root / "artifact", process.slice


def _sector(order) -> tuple:
    return tuple(sorted((name, power) for name, power in order if power))


def _stored(artifact: Path, poles: str, points, scales, parameters, checks):
    """{sector: [(Laurent, absolute estimate or None) per point]} of the
    stored process' selection (``checks``: the stability checks or None)."""

    from pyamplicol import Runtime

    runtime = Runtime.load(artifact, process="p", model_parameters=parameters)
    result = runtime.evaluate(
        points, mu_r2=scales, poles=poles, stability_checks=checks
    )
    return {
        _sector(sector.order): [
            (
                [float(x) for x in result.values[i][j]],
                None
                if checks is None
                else [float(x) for x in result.stability[i].absolute[j]],
            )
            for i in range(len(points))
        ]
        for j, sector in enumerate(result.sectors)
    }


def _companion_selections_equal(
    artifact, slice_, points, scales, parameters=None, estimated=False
):
    """The stored process has its companion, and its "all", "uv" and "ir"
    selections equal the in-memory slice's with Python point inputs, per
    point, sector and Laurent coefficient: within ``TOLERANCE`` of the
    sector's scale (the largest in-memory coefficient), or with
    ``estimated`` within the two binary64 evaluations' own absolute
    estimates (recommended stability checks), E_stored + E_in-memory.
    Returns the worst difference per selection, on the sector's scale (and
    with ``estimated`` on the estimate)."""

    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS

    execution = json.loads(
        (artifact / "processes" / "p" / "execution.json").read_text()
    )
    assert execution["scaleless_plan_sha256"] is not None
    assert slice_.scaleless is not None
    memory = NativeExecutor(slice_, forward=False, inputs="python")
    public = VirtualStabilityChecks(1, True, rescaled=True) if estimated else None
    worst = {}
    for poles in SELECTIONS:
        stored = _stored(artifact, poles, points, scales, parameters, public)
        found = memory.evaluate(
            points,
            scales,
            parameters=parameters,
            poles=poles,
            stability=RECOMMENDED_CHECKS if estimated else None,
        )
        worst[poles] = [0.0, 0.0]
        for i, result in enumerate(found):
            virtual = {
                _sector(sector): (
                    [float(getattr(x, "real", x)) for x in laurent.coefficients],
                    None
                    if not estimated
                    else result.stability.absolute[sector].coefficients,
                )
                for sector, laurent in result.components["virtual"].items()
            }
            assert set(virtual) == set(stored), (poles, i)
            for sector, (theirs, their_error) in virtual.items():
                mine, my_error = stored[sector][i]
                scale = max(map(abs, theirs))
                for k, (a, b) in enumerate(zip(mine, theirs, strict=True)):
                    bound = (
                        my_error[k] + their_error[k] if estimated else TOLERANCE * scale
                    )
                    assert abs(a - b) <= bound, (poles, i, sector, k, a, b, bound)
                    worst[poles][0] = max(worst[poles][0], abs(a - b) / scale)
                    if estimated and bound:
                        worst[poles][1] = max(worst[poles][1], abs(a - b) / bound)
    return worst


def _points() -> tuple[list, list[str]]:
    """The MadLoop fixture's u u~ > d d~ g points and mu_R^2 (exact texts)."""

    fixture = _madloop()
    scales = [str(Fraction(mu) ** 2) for mu in fixture["mu_r"]]
    return fixture["points"], scales


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("model", "restriction"),
    [("loop_sm", "no_b_mass"), ("loop_qcd_qed_sm_Gmu", "default")],
    ids=["loop_sm", "gmu"],
)
def test_companion_born_couplings_calling_helpers_store(
    loop_native, tmp_path, monkeypatch, model, restriction
):
    """u u~ > d d~ g [QCD] with both Born orders (QCD^3 and QCD QED^2) has a
    certified-scaleless companion whose Born couplings call helpers
    (loop_sm: a CKM coupling's complexconjugate; G_mu: re() through Gfbar):
    it stores, and its stored selections equal the in-memory ones."""

    artifact, slice_ = _store(
        tmp_path, monkeypatch, model, restriction, ("QCD",), WIDTHS
    )
    points, scales = _points()
    worst = _companion_selections_equal(artifact, slice_, points, scales)
    print(f"{model}: stored vs in-memory (Python inputs), worst per selection {worst}")


# ------------------------------------------------------------------ nightly

# M-SM row JJ04 (tests/fixtures/loop/msm/manifest.json): G_mu, Feynman gauge,
# explicit caps QCD 3 QED 2, [QCD, QED], the fixture's fixed parameters.
JJ04_FIXED = {
    "MB": "0",
    "ymb": "0",
    "MTA": "0",
    "ymtau": "0",
    "WT": "0",
    "WW": "0",
    "WZ": "0",
    "WH": "0",
}


def _fixture_metric(a: float, b: float, unit: float) -> float:
    """The fixture's error metric, |a-b| / max(|a|, |b|, |Born| coupling /
    2 pi) (``error_metric``), with the sector's unit of
    ``references.sector_pole_error``: the largest |Born(partner)| alpha_c /
    2 pi over the corrections c whose Born sector the loop sector has."""

    return abs(a - b) / max(abs(a), abs(b), unit)


@pytest.mark.loop_nightly
def test_jj04_stored_equals_madloop(loop_native, tmp_path, monkeypatch):
    """JJ04 (u u~ > d d~ g, [QCD, QED], caps QCD 3 QED 2, G_mu) stored: the
    finite part and both poles of every sector at the fixture's four points
    equal MadLoop in the fixture's metric within its tolerance, and the
    stored "all" / "uv" / "ir" selections equal the in-memory ones with
    Python point inputs within the evaluations' own binary64 estimates
    (recommended stability checks). Cost: about 3.5 min at ~10 GB."""

    fixture = _madloop()
    assert set(JJ04_FIXED.items()) <= set(fixture["fixed_parameters"].items())
    artifact, slice_ = _store(
        tmp_path,
        monkeypatch,
        "loop_qcd_qed_sm_Gmu",
        "default",
        ("QCD", "QED"),
        JJ04_FIXED,
    )
    points, scales = _points()
    # MW is derived in the G_mu scheme (the fixture records its value).
    parameters = {k: v for k, v in fixture["parameters"].items() if k != "MW"}
    stored = {
        sector: [laurent for laurent, _ in rows]
        for sector, rows in _stored(
            artifact, "all", points, scales, parameters, None
        ).items()
    }
    couplings = {
        "QCD": float(fixture["parameters"]["aS"]),
        "EW": float(fixture["derived_parameters"]["alpha"]),
    }
    worst = 0.0
    for i, row in enumerate(fixture["rows"]):
        born = {(b["QCD"], b["QED"]): b["born"] for b in row["sectors"]["born"]}
        loops = row["sectors"]["loop"]
        assert {_sector({"QCD": s["QCD"], "QED": s["QED"]}.items()) for s in loops} == (
            set(stored)
        )
        for loop in loops:
            mine = stored[_sector({"QCD": loop["QCD"], "QED": loop["QED"]}.items())][i]
            unit = max(
                abs(born[(partner["QCD"], partner["QED"])])
                * couplings[name]
                / (2 * math.pi)
                for name, partner in loop["born_sectors"].items()
                if partner
            )
            for a, key in zip(
                mine, ("finite", "single_pole", "double_pole"), strict=True
            ):
                error = _fixture_metric(a, loop[key], unit)
                assert error <= fixture["tolerance"], (i, loop["QCD"], key, a)
                worst = max(worst, error)
    print(f"JJ04 vs MadLoop: worst {worst:.2e} (fixture metric)")
    # The 1e-12 bound of the CI-tier tests is below this request's binary64
    # accuracy (stored 'uv' at point 1, QCD^4 QED^4: 1.1e-12 of the sector,
    # estimated 3.8e-11): the bound is the evaluations' own estimates.
    selections = _companion_selections_equal(
        artifact, slice_, points, scales, parameters, estimated=True
    )
    print(
        f"JJ04 stored vs in-memory (Python inputs), worst per selection: {selections}"
    )
