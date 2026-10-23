# SPDX-License-Identifier: 0BSD
"""``particle_veto`` (REQUIREMENTS_ADDENDUM J): a vetoed particle leaves every
generation step of a v2 virtual.

It is removed from every tree vertex (Born, loop blocks, counterterm program:
the tree engine's ``ignored_particle_ids``), from the loop species and loop
vertices (``ModelLoopCatalogue``), and every counterterm or external-field
term whose loop content holds it is dropped (``ct_recursion.
admitted_alternatives``, ``fields.external_field_insertions``).

Oracle: with loop_sm, ``particle_veto = [t]`` is massless five-flavour QCD
(no top loop, no top decoupling term), which NJet computes
(``tests/fixtures/loop/required/njet_nf5_controls.json``, NJet 3.1.2 in
double-double; its Nf=5 values agree with OpenLoops' to 3e-13).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "loop"
    / "required"
    / "njet_nf5_controls.json"
)
TOP = 6
# Relative, per Laurent coefficient with the floor alpha_s / (2 pi) Born (the
# provider comparisons of test_loop_v2_slice); binary64 against double-double.
TOLERANCE = 1e-12
PARAMETERS = {"aS": "0.118", "MT": "173"}
LAURENT = ("finite", "single_pole", "double_pole")


def _controls(name: str) -> dict:
    return json.loads(FIXTURE.read_text())["processes"][name]


def _request(name: str):
    from pyamplicol.loops.v2.assemble import SliceRequest

    control = _controls(name)
    born = control["born_orders"]
    return SliceRequest(
        control["process"],
        born,
        {"QCD": born["QCD"] + 2, "QED": 0},
        fixed_parameters={"MB": "0"},
        particle_veto=(TOP,),
    )


def _assert_top_free(slice_) -> None:
    """No top current in the trees or the CT program, no top loop species,
    no admitted top loop content in the counterterms and field factors."""

    program = slice_.counterterm_program
    for dag in (
        slice_.blocks.blocks_dag,
        slice_.blocks.born_dag,
        program.blocks.blocks.blocks_dag,
    ):
        assert TOP not in {abs(int(c.index.particle_id)) for c in dag.currents}
    assert not {"t", "t~"} & set(slice_.loop_catalogue._species)
    recursion = program.recursion
    assert recursion.model.veto == {TOP}
    for name, rule in recursion.rules.items():
        assert recursion.admission[name]
        for index in recursion.admission[name]:
            assert TOP not in rule.alternatives[index]
    # The gluon's top-loop field term and the top-loop UV rules are gone.
    assert slice_.fields
    assert all(
        TOP not in item.rule.alternatives[item.alternative] for item in slice_.fields
    )
    assert not any(rule.alternatives == ((TOP,),) for rule in recursion.rules.values())


def _assert_equals_njet(name: str, index: int, result) -> None:
    row = _controls(name)["rows"][index]
    (born,) = (float(value) for value in result.born.values())
    (laurent,) = result.components["virtual"].values()
    floor = abs(born) * float(PARAMETERS["aS"]) / (2 * math.pi)
    assert abs(born - float(row["born"])) <= TOLERANCE * abs(born)
    for pole, mine in zip(LAURENT, laurent.coefficients, strict=True):
        reference = float(row[pole])
        scale = max(abs(reference), floor)
        assert abs(float(mine) - reference) <= TOLERANCE * scale, (index, pole)


@pytest.mark.loop_fast
def test_veto_removes_loop_species_and_counterterm_loop_content(
    loop_sm_compiled_model,
):
    """The loop catalogue has no top species and no loop vertex with a top;
    every loop_sm counterterm alternative with a top loop is dropped and no
    other one."""

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.v2.ct_recursion import admitted_alternatives
    from pyamplicol.loops.v2.families import ModelLoopCatalogue
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    prepared = prepare_loop_request_model(compiled, fixed_parameters={"MB": "0"})
    structure = prepared.structure
    data = json.loads(structure.native.to_json())
    species = structure.perturbative_species_orders
    full = ModelLoopCatalogue(data, species_orders=species)
    vetoed = ModelLoopCatalogue(data, species_orders=species, veto=(TOP,))
    assert {"t", "t~"} <= set(full._species)
    assert set(full._species) - set(vetoed._species) == {"t", "t~"}
    dropped = 0
    for rule in prepared.counterterms.rules.values():
        kept = admitted_alternatives(rule, (TOP,))
        tops = {i for i, a in enumerate(rule.alternatives) if TOP in a}
        assert set(kept) == set(range(len(rule.alternatives))) - tops
        assert admitted_alternatives(rule) == tuple(range(len(rule.alternatives)))
        dropped += len(tops)
    assert dropped


@pytest.mark.loop_physics
def test_top_veto_equals_njet_five_flavours_gg_gg(loop_sm_compiled_model, loop_native):
    """g g > g g with particle_veto = [t]: Born, poles and finite part equal
    NJet's Nf=5 virtual at its ten points (four of them rotated), binary64."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    slice_ = LoopSlice.generate(compiled, _request("gg_gg"), loop_native)
    _assert_top_free(slice_)
    control = _controls("gg_gg")
    for index, (point, mu) in enumerate(
        zip(control["points"], control["mu_r"], strict=True)
    ):
        result = slice_.evaluate(
            point,
            str(int(mu) ** 2),
            parameters=PARAMETERS,
            precision=Precision.double(),
        )
        _assert_equals_njet("gg_gg", index, result)


@pytest.mark.loop_nightly
def test_top_veto_equals_njet_five_flavours_gg_gggg(
    loop_sm_compiled_model, loop_native
):
    """g g > g g g g with particle_veto = [t] at NJet's first point, the
    complete virtual through the native executor (binary64). Generation and
    the point's Python inputs (the counterterm program's rows) take about
    half an hour and 8 GB."""

    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    slice_ = LoopSlice.generate(compiled, _request("gg_gggg"), loop_native)
    _assert_top_free(slice_)
    control = _controls("gg_gggg")
    (result,) = NativeExecutor(slice_, forward=False).evaluate(
        [control["points"][0]],
        [str(int(control["mu_r"][0]) ** 2)],
        parameters=PARAMETERS,
    )
    _assert_equals_njet("gg_gggg", 0, result)
