# SPDX-License-Identifier: 0BSD
"""Derived helicity rows and row links of plan v4 (WP3.1 I11).

``export_plan`` writes per step the legs it leaves open and per content its
legs and row helicities; the native loader derives the schedule-B steps'
helicity rows and row links from them and the Born rows
(``loop_v2::plan::links``). With ``links=True`` the exporter also writes
them as before, and the loader checks its derivation against them. Checks:

* the derivation equals the written rows and links on processes with
  quark lines, gluons and massive tops, and a corrupted link is refused;
* the derived plan runs the same program: family numerators bitwise equal.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan
from tools.loop_reference.exact_points import generate

PARAMETERS = {"aS": "0.118", "MT": "173"}
PROCESSES = {
    "gg_gg": ("g g > g g", {"QCD": 2, "QED": 0}, [0, 0, 0, 0]),
    "uux_uuxg": ("u u~ > u u~ g", {"QCD": 3, "QED": 0}, [0] * 5),
    "gg_ggg": ("g g > g g g", {"QCD": 3, "QED": 0}, [0] * 5),
    "gg_ttx": ("g g > t t~", {"QCD": 2, "QED": 0}, [0, 0, 173, 173]),
}


@pytest.fixture(scope="module")
def slices(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict[str, LoopSlice] = {}

    def build(name: str) -> LoopSlice:
        if name not in built:
            process, born, _masses = PROCESSES[name]
            loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
            request = SliceRequest(process, born, loop, fixed_parameters={"MB": "0"})
            built[name] = LoopSlice.generate(compiled, request, loop_native)
        return built[name]

    return build


def _edges(program: dict):
    for family in program["families"]:
        for step in family["steps"]:
            yield step, step["edges"]


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", list(PROCESSES))
def test_derived_links_equal_the_written_links(slices, name):
    """The loader's derivation equals the exporter's rows and links, and the
    derived plan's numerators equal the written plan's bitwise."""

    slice_ = slices(name)
    written = export_plan(slice_, forward=False, links=True)
    derived = export_plan(slice_, forward=False)
    assert all("rows" in step for step, _ in _edges(written.program))
    assert not any(
        "rows" in step or any("links" in edge for edge in edges)
        for step, edges in _edges(derived.program)
    )
    assert len(derived.json()) < len(written.json())
    point = [[str(x) for x in p] for p in generate(PROCESSES[name][2], count=1)[0]]
    numerators = []
    for exported in (written, derived):
        executor = NativeExecutor(slice_, forward=False, exported=exported)
        # Loading the written plan checks the derived rows and links.
        executor.native("double")
        (point_numerators,) = executor.family_numerators(
            [point], ["1000000"], parameters=PARAMETERS
        )
        numerators.append(point_numerators)
    assert numerators[0].keys() == numerators[1].keys()
    for key, value in numerators[0].items():
        assert value.columns == numerators[1][key].columns


@pytest.mark.loop_physics
def test_a_corrupted_link_is_refused(slices):
    slice_ = slices("uux_uuxg")
    written = export_plan(slice_, forward=False, links=True)
    program = json.loads(written.json())
    edge = next(
        edge
        for _step, edges in _edges(program)
        for edge in edges
        if len(edge["links"]) >= 6 and edge["links"][1] != edge["links"][4]
    )
    edge["links"][1], edge["links"][4] = edge["links"][4], edge["links"][1]
    corrupted = replace(written, program=program)
    executor = NativeExecutor(slice_, forward=False, exported=corrupted)
    with pytest.raises(Exception, match="row links differ from the derived links"):
        executor.native("double")
