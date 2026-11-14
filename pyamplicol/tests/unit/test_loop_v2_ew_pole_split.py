# SPDX-License-Identifier: 0BSD
"""MS-POLES (m-sm/PLAN.md §3.2): the electroweak counterterms' poles split by
the UFOs' own regulators (``loops.v2.poles``, "Models with pole regulators").

MadLoop's electroweak UFOs write every pole of their CT parameters with
``epsUV`` and ``epsIR`` (UFO/SLHA TECHNICAL 1 and 2). In the ``"uv"`` and
``"ir"`` selections the counterterm and field components' UV parts are the
counterterm program at the IR regulator 0 and their IR parts the rest, with
no remainder, so ``bare_UV + CT_UV = 0`` is a test here.

Oracle (MS-REF): MadLoop in quadruple precision on loop_qcd_qed_sm_Gmu's
``restrict_with_technical`` card at its own masses (MT = ymt = 174.3,
MH = 120, every width 0 by ``fixed_parameters``), at ``epsIR = 1``
(``required/msm/uux_emep_ew_tech``, ``gg_ttx_ew_tech``) and at ``epsIR = 0``
(``..._ir0``, the same export and points: its counterterms hold their UV
parts only, and the bare loop does not read the regulators). Per point and
squared-order sector, in binary64 with native point inputs:

* ``poles="all"`` equals MadLoop at both cards (the total poles; ``"all"``
  and ``"none"`` never form the split, so their values are those of the
  slice before MS-POLES bit for bit);
* the counterterms' UV part (``uv`` + ``r2`` of the ``"uv"`` parts) equals
  MadLoop's UV-CT component at ``epsIR = 0``: MadLoop's virtual there less
  the bare loop (MadLoop's own bare loop from the component run for
  g g > t t~, the engine's for u u~ > e- e+); their IR part equals MadLoop's
  ``epsIR = 1`` minus ``epsIR = 0`` virtual;
* bare + counterterm UV poles vanish (1e-12 of the bare UV pole);
* the finite part is the same in every selection, and the two origins of
  every component add up to its combined poles.

DY01 (u u~ > e- e+, QCD and QED corrections) runs through the public API
(stored artifact): the Python runtime's selections, the SDK's plan v4
runtime bitwise equal to them, and the resolved outputs, whose pole parts
add up to the inclusive ones. TT01 (g g > t t~) runs on the generated slice:
native and Python point inputs, and the resolved outputs of the slice's
observables (Python inputs: the counterterm twin of each point). Nightly:
each generation takes about 3-6 min and 5-11 GB.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_nightly

ROOT = Path(__file__).resolve().parents[2]
REFERENCES = ROOT / "tests/fixtures/loop/required/msm"
# The fixtures' metric (|a - b| / max(|a|, |b|, |Born| alpha / 2 pi)); MadLoop
# QP against binary64 agrees to about 1e-14 here (rule 1's 1e-12 band).
TOLERANCE = 1e-10
ALPHA = 0.0075467722613403515
POINTS = (0, 1)
GMU_FIXED = {
    "MB": "0",
    "ymb": "0",
    "MTA": "0",
    "ymtau": "0",
    "WT": "0",
    "WW": "0",
    "WZ": "0",
    "WH": "0",
}


def _reference(name: str) -> dict:
    return json.loads((REFERENCES / f"{name}.json").read_text())


def _sector(entry) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((o, int(entry[o])) for o in ("QCD", "QED") if entry[o]))


def _metric(value: float, reference: float, born: float) -> float:
    floor = abs(born) * ALPHA / (2 * math.pi)
    scale = max(abs(value), abs(reference), floor)
    return abs(value - reference) / scale if scale else 0.0


def _madloop(name: str, index: int) -> dict:
    """Per sector the MadLoop (finite, single, double) and the point's Born."""

    row = _reference(name)["rows"][index]
    return row["born"], {
        _sector(s): (s["finite"], s["single_pole"], s["double_pole"])
        for s in row["sectors"]["loop"]
    }


def _check_split(
    totals, uv_parts, ir_parts, *, born, sectors, bare_madloop=None, label=""
):
    """The MS-POLES identities and the MadLoop oracle of one point.

    ``totals[eps]``: per sector the ``"all"`` Laurent at ``epsIR = eps``;
    ``uv_parts`` / ``ir_parts``: per component and sector the ``"uv"`` and
    ``"ir"`` parts (``(finite, single, double)``) at ``epsIR = 1``;
    ``sectors[eps]``: MadLoop's per sector; ``bare_madloop``: MadLoop's
    bare loop per sector (or None: the engine's)."""

    for eps in ("1", "0"):
        for sector, expected in sectors[eps].items():
            mine = totals[eps][sector]
            for k in range(3):
                error = _metric(mine[k], expected[k], born)
                assert error <= TOLERANCE, (label, eps, sector, k, mine, expected)
    for sector in sectors["1"]:
        whole = totals["1"][sector]
        for name, parts in uv_parts.items():
            u, i = parts[sector], ir_parts[name][sector]
            # Every part keeps the finite part of "all".
            assert u[0] == i[0], (label, name, sector)
            if name == "virtual":
                assert u[0] == whole[0], (label, sector)
                for k in (1, 2):
                    assert _metric(u[k] + i[k], whole[k], born) <= 1e-14
        bare = uv_parts["bare"][sector][1] + ir_parts["bare"][sector][1]
        if bare_madloop is not None:
            # The engine's bare loop is MadLoop's (independent of the split).
            assert _metric(bare, bare_madloop[sector], born) <= TOLERANCE
            bare = bare_madloop[sector]
        ct_uv = uv_parts["uv"][sector][1] + uv_parts["r2"][sector][1]
        ct_ir = ir_parts["uv"][sector][1] + ir_parts["r2"][sector][1]
        madloop_uv = sectors["0"][sector][1] - bare
        madloop_ir = sectors["1"][sector][1] - sectors["0"][sector][1]
        assert _metric(ct_uv, madloop_uv, born) <= TOLERANCE, (label, sector)
        assert _metric(ct_ir, madloop_ir, born) <= TOLERANCE, (label, sector)
        # UV finiteness is a test now (no remainder takes it).
        bare_uv = uv_parts["bare"][sector][1]
        assert abs(uv_parts["virtual"][sector][1]) <= 1e-12 * abs(bare_uv), (
            label,
            sector,
            uv_parts["virtual"][sector],
            bare_uv,
        )
        # The double pole is infrared.
        assert all(uv_parts[name][sector][2] == 0 for name in ("uv", "r2"))


def _parameters(name: str, eps_ir: str) -> dict[str, str]:
    reference = _reference(name)
    return {**reference["parameters"], "epsIR": eps_ir}


def _scales(name: str, indices) -> list[str]:
    mu = _reference(name)["mu_r"]
    return [str(Fraction(mu[i]) ** 2) for i in indices]


# --------------------------------------------------------------- DY01, public


DY01 = "uux_emep_ew_tech"


def _dy01_card(ufo: Path, output: Path) -> str:
    fixed = ", ".join(f'{name} = "{value}"' for name, value in GMU_FIXED.items())
    return f"""
action = "generate"
[model]
source = {json.dumps(str(ufo))}
restriction = "with_technical"
[process]
entries = [{{expression = "u u~ > e- e+", name = "dy"}}]
loop_coupling_orders = ["QCD", "QED"]
coupling_order_policy = "minimal"
[color]
accuracy = "full"
contraction = "direct"
[loop]
r2_strategy = "model"
[loop.model]
loop_coupling_orders = ["QCD", "QED"]
uv_complete = true
model_r2_complete = true
[loop.model.scheme]
dimensional_regularization = "HV"
external_states = "physical-four-dimensional"
gamma5 = "source-UFO-four-dimensional-prescription"
renormalization = "source-UFO mass-dependent decoupling"
mass_scheme = "real-on-shell"
finite_restoration = "source-UFO-local-rules"
model_convention = "loop_qcd_qed_sm_Gmu lhv=1 supplied QCD and QED rules"
[loop.preparation]
fixed_parameters = {{{fixed}}}
compatibility = "gmu_particle_anti"
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
"""


@pytest.fixture(scope="module")
def dy01_artifact(loop_gmu_ufo, loop_native, tmp_path_factory):
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    root = tmp_path_factory.mktemp("dy01")
    card = root / "card.toml"
    card.write_text(_dy01_card(loop_gmu_ufo, root / "artifact"))
    Generator(resolve_config(card)).generate(
        ProcessSet((ProcessRequest.parse("u u~ > e- e+", name="dy"),)),
        root / "artifact",
    )
    return root / "artifact"


def _public_sectors(result) -> list[tuple[tuple[str, int], ...]]:
    return [
        tuple(sorted((name, power) for name, power in sector.order if power))
        for sector in result.sectors
    ]


def test_dy01_technical_card_pole_split_equals_madloop(dy01_artifact, loop_native):
    """u u~ > e- e+ (QCD + QED) through the public API: the stored native
    record names the regulators; every MS-POLES identity and the MadLoop
    oracle per point and sector; the SDK runtime gives the Python API's
    values bitwise in every selection; the resolved outputs' pole parts add
    up to the inclusive ones."""

    from pyamplicol import Runtime
    from pyamplicol.loops.v2.artifact import read_native_process

    pacbin = next(dy01_artifact.rglob("loop-runtime.pacbin"))
    from pyamplicol.loops.v2.artifact import SCALELESS_PLAN_MEMBER, read_plan

    stored = read_native_process(
        pacbin, read_plan(pacbin), read_plan(pacbin, SCALELESS_PLAN_MEMBER)
    )
    assert stored.record["pole_regulators"] == {"uv": "epsUV", "ir": "epsIR"}
    # The conventions record which split the "uv" and "ir" parts use.
    conventions = Runtime.load(dy01_artifact).inspect()["conventions"]
    assert conventions["pole_split"] == "regulators"
    assert conventions["pole_regulators"] == {"uv": "epsUV", "ir": "epsIR"}
    reference = _reference(DY01)
    points = [reference["points"][i] for i in POINTS]
    scales = _scales(DY01, POINTS)
    runtimes = {
        eps: Runtime.load(dy01_artifact, model_parameters=_parameters(DY01, eps))
        for eps in ("1", "0")
    }
    values = {
        (eps, poles): runtimes[eps].evaluate(points, mu_r2=scales, poles=poles)
        for eps in ("1", "0")
        for poles in ("all", "none", "uv", "ir")
    }
    sectors_of = _public_sectors(values["1", "all"])
    for k, index in enumerate(POINTS):
        born, madloop = _madloop(DY01, index)
        madloop_0 = _madloop(DY01 + "_ir0", index)[1]
        totals = {
            eps: {
                sector: tuple(float(x) for x in values[eps, "all"].values[k][s])
                for s, sector in enumerate(sectors_of)
            }
            for eps in ("1", "0")
        }
        parts = {}
        for poles in ("uv", "ir"):
            lane = values["1", poles].pole_parts[k]
            for origin in ("uv", "ir"):
                found = {
                    name: {
                        sector: tuple(float(x) for x in laurent[s])
                        for s, sector in enumerate(sectors_of)
                    }
                    for name, laurent in lane.origin(origin).items()
                }
                if poles == "uv":
                    parts[origin] = found
                else:
                    # Both selections report the same parts.
                    assert found == parts[origin], (origin, index)
        _check_split(
            totals,
            parts["uv"],
            parts["ir"],
            born=born,
            sectors={"1": madloop, "0": madloop_0},
            label=f"DY01 point {index}",
        )
        # The finite part is the same in every selection.
        finite = [values["1", "all"].values[k][s][0] for s in range(len(sectors_of))]
        for poles in ("none", "uv", "ir"):
            assert [
                values["1", poles].values[k][s][0] for s in range(len(sectors_of))
            ] == finite, poles
    # The SDK's plan v4 runtime: bitwise the Python API's, both cards.
    flat = [[str(x) for momentum in point for x in momentum] for point in points]
    for eps in ("1", "0"):
        for poles in ("all", "none", "uv", "ir"):
            native = loop_native.evaluate_v2_artifact(
                str(dy01_artifact),
                flat,
                scales,
                process="dy",
                parameters=_parameters(DY01, eps),
                poles=poles,
            )
            width = 1 if poles == "none" else 3
            python = values[eps, poles]
            for index, (result, failures) in enumerate(native):
                assert not failures, failures
                born, virtual = result
                assert born == list(python.born_values[index])
                assert virtual == [
                    list(laurent)[:width] for laurent in python.values[index]
                ], (eps, poles, index)
    # The resolved outputs (per helicity) split their poles the same way:
    # their parts add up to the inclusive ones.
    resolved = runtimes["1"].evaluate_resolved(points[:1], mu_r2=scales[:1], poles="uv")
    inclusive = values["1", "uv"].pole_parts[0]
    lane = resolved.pole_parts[0]
    for origin in ("uv", "ir"):
        for name, laurent in inclusive.origin(origin).items():
            summed = lane.origin(origin)[name]
            for s in range(len(sectors_of)):
                whole = [float(x) for x in laurent[s]]
                total = [0.0, 0.0, 0.0]
                for value in _resolved_entries(summed[s]):
                    total = [
                        a + complex(b).real for a, b in zip(total, value, strict=True)
                    ]
                scale = max(abs(x) for x in whole) or 1.0
                assert all(
                    abs(a - b) <= 1e-12 * scale
                    for a, b in zip(total, whole, strict=True)
                ), (origin, name, s, total, whole)


def _resolved_entries(axis):
    """The Laurent coefficients of one sector's resolved axes (nested tuples
    down to ``LaurentCoefficients``)."""

    from pyamplicol.api.virtual_results import LaurentCoefficients

    if isinstance(axis, LaurentCoefficients):
        yield axis
        return
    for item in axis:
        yield from _resolved_entries(item)


# ----------------------------------------------------------- TT01, the slice


TT01 = "gg_ttx_ew_tech"


@pytest.fixture(scope="module")
def tt01_slice(loop_gmu_ufo, loop_native, tmp_path_factory):
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import compile_model_source

    compiled = compile_model_source(
        loop_gmu_ufo,
        restriction="with_technical",
        require_supported=False,
        cache_dir=tmp_path_factory.mktemp("tt01-model"),
    )
    request = SliceRequest(
        "g g > t t~",
        {"QCD": 2, "QED": 0},
        ({"QCD": 4, "QED": 0}, {"QCD": 2, "QED": 2}),
        corrections=("QCD", "QED"),
        fixed_parameters=GMU_FIXED,
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )
    return LoopSlice.generate(compiled, request, loop_native)


def _parts(result) -> dict[str, dict[str, dict]]:
    return {
        origin: {
            name: {
                sector: tuple(float(x) for x in laurent.coefficients)
                for sector, laurent in values.items()
            }
            for name, values in result.pole_origins[origin].items()
        }
        for origin in ("uv", "ir")
    }


def test_tt01_technical_card_pole_split_equals_madloop(tt01_slice):
    """g g > t t~ (QCD + QED) on the slice: the MS-POLES identities and the
    MadLoop oracle with MadLoop's own bare loop (component run), native
    point inputs; the Python point inputs (the counterterm twin at the IR
    regulator 0) give the same parts; the stored native process the same
    values bitwise; the resolved outputs of the slice's observables add up
    to the inclusive parts."""

    from pyamplicol.loops.v2.native_process import NativeProcess
    from pyamplicol.loops.v2.observables import NativeObservables, ObservableSlice
    from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan

    slice_ = tt01_slice
    assert slice_.pole_regulators.record() == {"uv": "epsUV", "ir": "epsIR"}
    native = NativeExecutor(slice_, forward=False, inputs="native")
    components = json.loads((REFERENCES / "components" / f"{TT01}.json").read_text())
    reference = _reference(TT01)
    points = [reference["points"][i] for i in POINTS]
    scales = _scales(TT01, POINTS)
    runs = {
        (eps, poles): native.evaluate(
            points, scales, parameters=_parameters(TT01, eps), poles=poles
        )
        for eps in ("1", "0")
        for poles in ("all", "uv", "ir")
    }
    for k, index in enumerate(POINTS):
        born, madloop = _madloop(TT01, index)
        madloop_0 = _madloop(TT01 + "_ir0", index)[1]
        bare = {
            tuple(
                sorted(
                    (o, int(p))
                    for o, p in (
                        part.split("=") for part in key.replace("^2", "").split(",")
                    )
                    if int(p)
                )
            ): entry["rows"][index]["bare"]["single_pole"]
            for key, entry in components["sectors"].items()
        }
        totals = {
            eps: {
                sector: tuple(float(x) for x in laurent.coefficients)
                for sector, laurent in runs[eps, "all"][k].components["virtual"].items()
            }
            for eps in ("1", "0")
        }
        uv, ir = _parts(runs["1", "uv"][k]), _parts(runs["1", "ir"][k])
        # Both selections report the same parts.
        assert uv == ir
        _check_split(
            totals,
            uv["uv"],
            uv["ir"],
            born=born,
            sectors={"1": madloop, "0": madloop_0},
            bare_madloop=bare,
            label=f"TT01 point {index}",
        )
    # Python point inputs: the counterterm twin of the point (trees in
    # Python at the IR regulator 0) against the native tree inputs.
    python = NativeExecutor(
        slice_, forward=False, exported=export_plan(slice_, forward=False)
    )
    (mine,) = python.evaluate(
        points[:1], scales[:1], parameters=_parameters(TT01, "1"), poles="uv"
    )
    for origin, parts in _parts(mine).items():
        for name, values in parts.items():
            for sector, value in values.items():
                expected = _parts(runs["1", "uv"][0])[origin][name][sector]
                scale = max(abs(x) for x in expected) or 1.0
                assert all(
                    abs(a - b) <= 1e-12 * scale
                    for a, b in zip(value, expected, strict=True)
                ), (origin, name, sector, value, expected)
    # The stored native process names the regulators and evaluates bitwise.
    process = NativeProcess.of(native)
    assert process.record["pole_regulators"] == {"uv": "epsUV", "ir": "epsIR"}
    loaded = NativeProcess.from_members(process.plan_json, process.members())
    stored = loaded.executor(slice_.native)
    for poles in ("uv", "ir"):
        (again,) = stored.evaluate(
            points[:1], scales[:1], parameters=_parameters(TT01, "1"), poles=poles
        )
        assert _parts(again) == _parts(runs["1", poles][0])
        assert again.totals == runs["1", poles][0].totals
    # The resolved outputs of the slice's observables (Python inputs): per
    # helicity the counterterm twin's interferences; they add up.
    observables = ObservableSlice.of(slice_)
    resolved = NativeObservables.resolved(observables, "helicity")
    (columns,) = resolved.evaluate(
        points[:1], scales[:1], parameters=_parameters(TT01, "1"), poles="uv"
    )
    inclusive = _parts(runs["1", "uv"][0])
    for origin in ("uv", "ir"):
        for name, values in inclusive[origin].items():
            for sector, whole in values.items():
                total = [0.0, 0.0, 0.0]
                for result in columns.values():
                    laurent = result.pole_origins[origin][name][sector]
                    total = [
                        a + complex(b).real
                        for a, b in zip(total, laurent.coefficients, strict=True)
                    ]
                scale = max(abs(x) for x in whole) or 1.0
                assert all(
                    abs(a - b) <= 1e-12 * scale
                    for a, b in zip(total, whole, strict=True)
                ), (origin, name, sector, total, whole)
