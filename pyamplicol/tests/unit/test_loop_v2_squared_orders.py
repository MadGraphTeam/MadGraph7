# SPDX-License-Identifier: 0BSD
"""Squared-coupling-order sector selection (``process.squared_coupling_orders``,
MG5_aMC's ``QCD^2<=4``; ux-process).

A selection keeps the squared-order sectors that pass every constraint: the
kept sectors are the result's only sectors (and its explicit sum), each equal
to the same sector of the unrestricted run at the same points; generation
skips the loop orders no kept sector pairs; a selection keeping no sector is
the listed ``squared-order-selection`` refusal. Born-interfered (u u~ > d d~,
QCD corrections to QCD and photon Borns: three sectors) and loop-induced
(g g > a a)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop" / "required" / "li"
MU_R2 = "10000"


def _card(root: Path, expression: str, process: str, extra: str, generation: str):
    from pyamplicol.config import resolve_config

    root.mkdir(parents=True, exist_ok=True)
    card = root / "card.toml"
    card.write_text(
        f"""action = "generate"
[model]
source = "loop_sm"
restriction = "no_b_mass"
[process]
entries = [{{expression = {json.dumps(expression)}, name = "p"}}]
loop_coupling_orders = ["QCD"]
{process}
[loop]
r2_strategy = "model"
{extra}
[generation]
output = "artifact"
emit_api_bundle = false
{generation}
"""
    )
    return resolve_config(card)


def _generate(
    root: Path, expression: str, process: str = "", extra: str = "", generation=""
):
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet

    config = _card(root, expression, process, extra, generation)
    output = Path(config.effective.generation.output)
    Generator(config).generate(
        ProcessSet((ProcessRequest.parse(expression, name="p"),)), output
    )
    return output


def _points(artifact: Path) -> list:
    record = json.loads(
        (artifact / "processes" / "p" / "validation-momenta.json").read_text()
    )
    return [[entry["momentum"] for entry in point] for point in record["points"]]


def _sectors(artifact: Path, points) -> dict:
    from pyamplicol import Runtime

    evaluation = Runtime.load(artifact).evaluate(points, mu_r2=MU_R2)
    return {
        sector.order: [values[index] for values in evaluation.values]
        for index, sector in enumerate(evaluation.sectors)
    }, evaluation


def _same(left, right) -> None:
    for a, b in zip(left, right, strict=True):
        for x, y in zip(a, b, strict=True):
            assert float(x) == pytest.approx(float(y), rel=1e-10, abs=1e-300)


UUX = 'coupling_order_policy = "explicit"\nmax_coupling_orders = { QCD = 2, QED = 2 }'


@pytest.fixture(scope="module")
def unrestricted(tmp_path_factory):
    artifact = _generate(tmp_path_factory.mktemp("all"), "u u~ > d d~", UUX)
    points = _points(artifact)
    return artifact, points, _sectors(artifact, points)[0]


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("selection", "kept"),
    [
        (["QCD^2<=4"], ((("QCD", 2), ("QED", 4)), (("QCD", 4), ("QED", 2)))),
        (["QCD^2==6"], ((("QCD", 6),),)),
        (["QED^2>=2", "QED^2<4"], ((("QCD", 4), ("QED", 2)),)),
    ],
)
def test_selected_sectors_equal_the_unrestricted_ones(
    unrestricted, tmp_path, selection, kept
):
    _artifact, points, reference = unrestricted
    assert len(reference) == 3
    artifact = _generate(
        tmp_path,
        "u u~ > d d~",
        UUX + f"\nsquared_coupling_orders = {json.dumps(selection)}",
    )
    sectors, evaluation = _sectors(artifact, points)
    assert tuple(sorted(sectors)) == kept
    for order, values in sectors.items():
        _same(values, reference[order])
    # The total sums the kept sectors only.
    for point, summed in enumerate(evaluation.sum_orders()):
        expected = math.fsum(float(sectors[order][point][0]) for order in kept)
        assert float(summed[0]) == pytest.approx(expected, rel=1e-12)
    # Generation skips the loop orders no kept sector pairs (the dry run
    # lists the request's loop orders and its selection).
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet

    config = _card(
        tmp_path / "plan",
        "u u~ > d d~",
        UUX + f"\nsquared_coupling_orders = {json.dumps(selection)}",
        "",
        "",
    )
    plan = Generator(config).plan(
        ProcessSet((ProcessRequest.parse("u u~ > d d~", name="p"),))
    )
    (process,) = plan.estimated_coverage["processes"]
    assert list(process["squared_coupling_orders"]) == sorted(selection)
    loops = process["loop_orders"]
    loops = [loops] if isinstance(loops, Mapping) else list(loops)
    expected = 1 if selection == ["QCD^2==6"] else 2
    assert len(loops) == expected


@pytest.mark.loop_physics
def test_a_selection_keeping_no_sector_is_refused(tmp_path):
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    with pytest.raises(LoopRequestUnsupported) as refused:
        _generate(
            tmp_path,
            "u u~ > d d~",
            UUX + '\nsquared_coupling_orders = ["QCD^2>6"]',
        )
    assert refused.value.key == "squared-order-selection"
    assert "QCD^2>6" in str(refused.value)


@pytest.mark.loop_physics
def test_a_loop_induced_selection(tmp_path):
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    fixture = json.loads((FIXTURES / "li_gg_aa_nf5.json").read_text())
    fixed = ", ".join(
        f"{key} = {json.dumps(value)}"
        for key, value in fixture["scheme_data"]["fixed_parameters"].items()
    )
    extra = f"[loop.preparation]\nfixed_parameters = {{{fixed}}}"
    veto = 'particle_veto = ["t"]'
    reference = _generate(tmp_path / "all", "g g > a a", "", extra, veto)
    points = _points(reference)[:1]
    all_sectors, _ = _sectors(reference, points)
    selected = _generate(
        tmp_path / "sel",
        "g g > a a",
        'squared_coupling_orders = ["QED^2==4", "QCD^2<=4"]',
        extra,
        veto,
    )
    sectors, evaluation = _sectors(selected, points)
    assert evaluation.contribution == "loop-induced"
    assert sectors.keys() == all_sectors.keys()
    for order, values in sectors.items():
        _same(values, all_sectors[order])
    with pytest.raises(LoopRequestUnsupported) as refused:
        _generate(
            tmp_path / "none",
            "g g > a a",
            'squared_coupling_orders = ["QED^2<4"]',
            extra,
            veto,
        )
    assert refused.value.key == "squared-order-selection"
