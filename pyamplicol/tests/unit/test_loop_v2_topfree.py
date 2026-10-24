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

import json
import math
from pathlib import Path

import pytest

from pyamplicol.loops.v2 import assemble
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


def _errors(slice_: LoopSlice, controls: dict, name: str, indices, precision=None):
    entry = controls["processes"][name]
    alphas = controls["parameters"]["aS"]
    executor = NativeExecutor(
        slice_, forward=False, exported=export_plan(slice_, forward=False)
    )
    rows = {row["index"]: row for row in entry["rows"]}
    out = {}
    # The fixture's Born is massless QCD: no top coupling to check here.
    checked = assemble._PointEvaluation._stable_coloured_legs
    assemble._PointEvaluation._stable_coloured_legs = lambda self: False
    try:
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
    finally:
        assemble._PointEvaluation._stable_coloured_legs = checked
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
