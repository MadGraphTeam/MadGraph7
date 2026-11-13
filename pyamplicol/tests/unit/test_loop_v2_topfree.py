# SPDX-License-Identifier: 0BSD
"""Top-free requests against massless five-flavour QCD (NJet, N_f = 5).

``SliceRequest.particle_veto`` removes a particle (|PDG|, both orientations)
from the tree currents, the loop lines, the counterterms and the field
insertions. With the top excluded (and MB = 0), loop_sm's virtual is
massless QCD with five flavours and MS-bar alpha_s, NJet's scheme, so every
Laurent coefficient compares directly with ``njet_nf5_topfree.json`` (NJet
double-double at pyAmpliCol's exact points, several renormalisation scales).
This is the scheme the g g > g g g g g finite part needs against
``gg_ggggg_njet.json``.

Before the loop lines honoured the exclusion, the closed top loops stayed in
while their decoupling counterterm and field insertion were dropped: the
single pole exceeded NJet's by (alpha_s/4pi)(4/3) B at every multiplicity.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol.loops.v2 import assemble, generator_input
from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan
from tools.loop_reference.njet.njet_fixture import merge_metric

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "loop"
    / "required"
    / "njet_nf5_topfree.json"
)
COEFFICIENTS = ("born", "finite", "single_pole", "double_pole")


@pytest.fixture(scope="module")
def controls() -> dict:
    return json.loads(FIXTURE.read_text())


def _slice(compiled_path, native, name: str, controls: dict) -> LoopSlice:
    from pyamplicol.models.loading import load_compiled_model

    entry = controls["processes"][name]
    process = " ".join(
        t for t in entry["process"].split(" [")[0].split() if "=" not in t
    )
    return LoopSlice.generate(
        load_compiled_model(compiled_path),
        SliceRequest(
            process,
            entry["born_orders"],
            entry["loop_orders"],
            fixed_parameters={"MB": "0"},
            particle_veto=(6,),
        ),
        native,
    )


@contextlib.contextmanager
def _no_coloured_leg_check() -> Iterator[None]:
    """Skip the slice's coloured-leg stability gate for massless-QCD
    fixtures (no top coupling to check), restored on exit."""

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            assemble._PointEvaluation, "_stable_coloured_legs", lambda self: False
        )
        yield


def _errors(slice_: LoopSlice, controls: dict, name: str, indices, precision=None):
    entry = controls["processes"][name]
    alphas = controls["parameters"]["aS"]
    executor = NativeExecutor(
        slice_, forward=False, exported=export_plan(slice_, forward=False)
    )
    rows = {row["index"]: row for row in entry["rows"]}
    out = {}
    # The fixture's Born is massless QCD: no top coupling to check here.
    with _no_coloured_leg_check():
        for index in indices:
            scale = str(int(entry["mu_r"][index]) ** 2)
            (result,) = executor.evaluate(
                [entry["points"][index]],
                [scale],
                parameters={"aS": alphas},
                precision=precision,
            )
            mine = [float(x) for x in result.totals[:4]]
            ref = [float(rows[index][key]) for key in COEFFICIENTS]
            floor = abs(ref[0]) * float(alphas) / (2 * math.pi)
            out[index] = {
                key: merge_metric(a, b, 0.0 if key == "born" else floor)
                for key, a, b in zip(COEFFICIENTS, mine, ref, strict=True)
            }
    return out


@pytest.mark.loop_physics
def test_top_free_gg_gg_is_massless_five_flavour_qcd(
    loop_sm_compiled_model, loop_native, controls
):
    slice_ = _slice(loop_sm_compiled_model, loop_native, "gg_gg", controls)
    pdg = {entry["name"]: abs(entry["pdg_code"]) for entry in slice_.data["particles"]}
    species = {pdg[name] for name in slice_.loop_catalogue.loop_species()}
    assert 6 not in species and {1, 2, 3, 4, 5, 21} <= species
    errors = _errors(slice_, controls, "gg_gg", range(10))
    # Measured <= 2.1e-14 over the ten points (three scales).
    assert max(max(row.values()) for row in errors.values()) < 1e-12, errors


@pytest.mark.loop_physics
def test_top_free_gg_ggg_is_massless_five_flavour_qcd(
    loop_sm_compiled_model, loop_native, controls
):
    slice_ = _slice(loop_sm_compiled_model, loop_native, "gg_ggg", controls)
    # Point 4 is ill-conditioned in binary64 (finite part 2.3e-8 off, the
    # rotation check estimates 1.3e-7); DoubleFloat resolves it to 3e-14.
    errors = _errors(slice_, controls, "gg_ggg", (0, 1, 2))
    assert max(max(row.values()) for row in errors.values()) < 1e-11, errors
    errors = _errors(
        slice_, controls, "gg_ggg", (4,), precision=Precision.double_double()
    )
    assert max(errors[4].values()) < 1e-12, errors


# ------------------------------------------------- the veto on the native lane


@pytest.fixture(scope="module")
def vetoed(loop_sm_compiled_model, loop_native):
    """Slices of the native lane's veto controls, with and without the veto:
    g g > g g without the top (NJet N_f = 5) and u u~ > e- e+ without the Z
    (the legacy engine's photon-only Drell-Yan, ``legacy/dy.json``)."""

    from pyamplicol.models.loading import load_compiled_model
    from tests.unit.test_loop_v2_slice import _request as legacy_request

    compiled = load_compiled_model(loop_sm_compiled_model)
    requests = {
        "gg_gg": SliceRequest(
            "g g > g g",
            {"QCD": 2, "QED": 0},
            {"QCD": 4, "QED": 0},
            fixed_parameters={"MB": "0"},
            particle_veto=(6,),
        ),
        "dy": legacy_request("dy"),
    }
    built: dict[tuple[str, bool], LoopSlice] = {}

    def get(name: str, veto: bool = True) -> LoopSlice:
        if (name, veto) not in built:
            request = requests[name]
            if not veto:
                request = dataclasses.replace(request, particle_veto=())
            built[name, veto] = LoopSlice.generate(compiled, request, loop_native)
        return built[name, veto]

    return get


@pytest.mark.loop_physics
@pytest.mark.parametrize(("name", "vetoed_pdg"), [("gg_gg", 6), ("dy", 23)])
def test_particle_veto_reaches_the_native_generator(vetoed, name, vetoed_pdg):
    """RUST-MIGRATION §4 item 1(c): the native generator follows the public
    ``particle_veto``. Its families and schedules of the vetoed request equal
    the Python stages' on the same (vetoed) block table and loop catalogue;
    the veto removes the particle's loop lines and blocks (fewer loop
    routings than the unvetoed request), its tree currents (Born, loop
    blocks, counterterm program) and every counterterm term with it in the
    loop; the native
    per-point inputs (tree program, Born and counterterm programs, field
    insertions) give every component of the Python inputs within 1e-12."""

    from pyamplicol.loops.v2.families import LoopOrderSelection, generate_families
    from pyamplicol.loops.v2.schedules import build_schedules
    from tests.unit.test_loop_v2_generate import (
        assert_same_families,
        assert_same_schedules,
    )
    from tests.unit.test_loop_v2_native_trees import _assert_same

    slice_ = vetoed(name)
    assert slice_.generation is not None  # the native stages ran
    vetoed_names = {
        p["name"] for p in slice_.data["particles"] if abs(p["pdg_code"]) == vetoed_pdg
    }
    assert vetoed_names and not vetoed_names & set(slice_.loop_catalogue._species)
    program = slice_.counterterm_program
    for dag in (
        slice_.blocks.blocks_dag,
        slice_.blocks.born_dag,
        program.blocks.blocks.blocks_dag,
    ):
        assert vetoed_pdg not in {abs(int(c.index.particle_id)) for c in dag.currents}
    recursion = program.recursion
    for rule_name, rule in recursion.rules.items():
        for index in recursion.admission[rule_name]:
            assert vetoed_pdg not in rule.alternatives[index]

    table = slice_.blocks.block_table(slice_.data)
    loop = assemble.add_orders(tuple(slice_.request.loop_orders.items()))
    selection = LoopOrderSelection(
        frozenset(slice_.request.corrections), frozenset({loop})
    )
    families = generate_families(table, slice_.loop_catalogue, selection)
    assert_same_families(slice_.families, families)
    assert_same_schedules(
        slice_.plan,
        build_schedules(
            families,
            slice_.loop_catalogue,
            slice_.colour,
            policy=assemble._frame_policy(generator_input.leg_rows(slice_.blocks)),
        ),
    )
    full = vetoed(name, veto=False)

    def routings(families):
        return sum(len(family.closures) for family in families.families.values())

    assert routings(slice_.families) < routings(full.families)

    native = NativeExecutor(slice_, forward=False, inputs="native")
    python = NativeExecutor(slice_, forward=False, exported=native.exported)
    assert native.exported.trees.program["counterterm_trees"] is not None
    points, scales = _veto_points(name)
    parameters = _veto_parameters(name)
    mine = native.evaluate(points, scales, parameters=parameters)
    _assert_same(mine, python.evaluate(points, scales, parameters=parameters), 1e-12)
    # The veto changes the virtual (the vetoed particle contributed).
    unvetoed = NativeExecutor(full, forward=False, inputs="native").evaluate(
        points[:1], scales[:1], parameters=parameters
    )
    assert mine[0].totals[1] != unvetoed[0].totals[1]


def _veto_points(name: str) -> tuple[list, list[str]]:
    if name == "dy":
        from tests.unit.test_loop_v2_slice import _fixture

        fixture = _fixture("dy")
        return fixture["points"], list(fixture["scales_squared"])
    entry = json.loads(FIXTURE.read_text())["processes"][name]
    return entry["points"], [str(int(mu) ** 2) for mu in entry["mu_r"]]


def _veto_parameters(name: str) -> dict[str, str]:
    from tests.unit.test_loop_v2_slice import DRELL_YAN_PARAMETERS

    if name == "dy":
        return DRELL_YAN_PARAMETERS
    return {"aS": json.loads(FIXTURE.read_text())["parameters"]["aS"]}


@pytest.mark.loop_physics
def test_vetoed_native_lane_equals_the_references(vetoed, controls):
    """The native lane with the veto against the vetoed physics: g g > g g
    without the top equals NJet's N_f = 5 virtual (ten points, three scales)
    and u u~ > e- e+ without the Z equals the legacy engine's photon-only
    Drell-Yan (Born and every component), both in binary64."""

    from tests.unit.test_loop_v2_slice import _assert_legacy, _fixture

    entry = controls["processes"]["gg_gg"]
    rows = {row["index"]: row for row in entry["rows"]}
    points, scales = _veto_points("gg_gg")
    with _no_coloured_leg_check():
        results = NativeExecutor(
            vetoed("gg_gg"), forward=False, inputs="native"
        ).evaluate(points, scales, parameters=_veto_parameters("gg_gg"))
    alphas = float(controls["parameters"]["aS"])
    worst = 0.0
    for index, result in enumerate(results):
        mine = [float(x) for x in result.totals[:4]]
        reference = [float(rows[index][key]) for key in COEFFICIENTS]
        floor = abs(reference[0]) * alphas / (2 * math.pi)
        for key, a, b in zip(COEFFICIENTS, mine, reference, strict=True):
            worst = max(worst, merge_metric(a, b, 0.0 if key == "born" else floor))
    # Measured with Python inputs: <= 2.1e-14 (test above).
    assert worst < 1e-12, worst

    fixture = _fixture("dy")
    points, scales = _veto_points("dy")
    results = NativeExecutor(vetoed("dy"), forward=False, inputs="native").evaluate(
        points, scales, parameters=_veto_parameters("dy")
    )
    reference = fixture["results"]["arbitrary-200"]
    for index, result in enumerate(results):
        _assert_legacy(reference, index, result, Decimal("1e-12"))
