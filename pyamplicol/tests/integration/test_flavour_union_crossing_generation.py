# SPDX-License-Identifier: 0BSD
"""Generated crossed flavour unions (``process.flavor_union_crossing``).

Leading colour with crossing: one template per all-outgoing content. Every
crossed member of the generated artifact, evaluated as a template assignment
at ``q[r] = sigma_r * p[permutation[r]]``, must equal its own flavour-class
program at 1e-12 (totals, per helicity relabelled by rho and pi, per LC flow),
and must find a target route in a full-colour artifact whose unions stay side
aware (the reweight adapter routes by event flavour).

``p p > j j`` always runs; ``p p > j j j`` and ``p p > j j j j`` only with
``PYAMPLICOL_RUN_FLAVOUR_UNION=pp3j,pp4j`` (either or both).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from pyamplicol import Generator, ProcessSet, Runtime
from pyamplicol.artifacts.process_groups import read_process_groups
from pyamplicol.artifacts.umami_reweight import (
    _route,
    _route_index,
    _target_routes,
    load_umami_metadata,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.generation.process_grouping import crossing_signs
from tests.integration.test_flavour_union_crossing import (
    _close,
    _member_points,
    _signed,
    _strings,
)

_OPT_IN_ENV = "PYAMPLICOL_RUN_FLAVOUR_UNION"
# name: (expression, programs with crossing, of which unions)
_SETS = {
    "pp2j": ("p p > j j", 3, 2),
    "pp3j": ("p p > j j j", 3, 2),
    "pp4j": ("p p > j j j j", 4, 3),
}


def _config(accuracy: str, *, unions: bool, crossing: bool, bundle: bool) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy=accuracy),
        process=ProcessConfig(
            flavor_scheme=4,
            flavor_blind_strict=True,
            flavor_unions=unions,
            flavor_union_crossing=crossing,
            max_coupling_orders={"QED": 0},
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=bundle,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="recurrence",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


@pytest.fixture(scope="module", params=sorted(_SETS))
def artifacts(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[str, dict[str, Path]]]:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    name = request.param
    if name != "pp2j" and name not in os.environ.get(_OPT_IN_ENV, "").split(","):
        pytest.skip(f"set {_OPT_IN_ENV}={name} to run it")
    expression = _SETS[name][0]
    root = tmp_path_factory.mktemp(f"crossing-{name}")
    jobs = {
        "crossing": _config("lc", unions=True, crossing=True, bundle=True),
        "reference": _config("lc", unions=False, crossing=False, bundle=False),
        "full": _config("full", unions=True, crossing=False, bundle=True),
    }
    paths = {}
    try:
        for job, config in jobs.items():
            paths[job] = root / job
            Generator(config).generate(
                ProcessSet.from_expressions((expression,)), paths[job]
            )
        yield name, paths
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _crossed_groups(artifact: Path) -> list[tuple[str, str, list[object]]]:
    group_map = read_process_groups(artifact)
    assert group_map is not None
    return [
        (
            group.representative.process_id,
            group.representative.expression,
            [member for member in group.members[1:] if member.crossed],
        )
        for group in group_map.groups.values()
        if group.pattern_pairs is not None
    ]


def test_crossing_generates_one_program_per_all_outgoing_content(
    artifacts: tuple[str, dict[str, Path]],
) -> None:
    name, paths = artifacts
    _, programs, unions = _SETS[name]
    raw = json.loads((paths["crossing"] / "artifact.json").read_text("utf-8"))
    assert len(raw["processes"]) == programs
    groups = json.loads((paths["crossing"] / "process-groups.json").read_text("utf-8"))
    assert groups["schema_version"] == 2
    crossed = _crossed_groups(paths["crossing"])
    assert len(crossed) == unions
    assert all(members for _, _, members in crossed)
    # the side-aware full-colour artifact keeps schema 1
    plain = json.loads((paths["full"] / "process-groups.json").read_text("utf-8"))
    assert plain["schema_version"] == 1


def test_crossed_members_equal_their_own_programs(
    artifacts: tuple[str, dict[str, Path]],
) -> None:
    _, paths = artifacts
    checked = 0
    for template_id, template_expression, members in _crossed_groups(paths["crossing"]):
        template = Runtime.load(paths["crossing"], process=template_id)
        index_of = {
            entry["process_id"]: i
            for i, entry in enumerate(template.flavour_union_assignments)
        }
        for member in members:
            index = index_of[member.process_id]
            entry = template.flavour_union_assignments[index]
            perm = [int(value) for value in entry["permutation"]]
            sigma = [int(value) for value in entry["momentum_signs"]]
            rho = [int(value) for value in entry["helicity_signs"]]
            # the core re-derived the generator's sigma and rho
            assert perm == list(member.permutation)
            assert sigma == list(member.crossing_momentum_signs)
            assert rho == list(member.crossing_helicity_signs)
            assert tuple(sigma) == crossing_signs(
                perm,
                len(template_expression.split(">")[0].split()),
                member.initial_count,
            )
            reference = Runtime.load(paths["reference"], process=member.expression)
            points = _member_points(member.expression)
            signed = [_signed(point, perm, sigma) for point in points]
            totals = [
                row[0] for row in template.evaluate_assignments(signed, [[index]] * 4)
            ]
            expected = [complex(value).real for value in reference.evaluate(points)]
            for total, value in zip(totals, expected, strict=True):
                _close(total, value, abs(value), (member.expression, "total"))
            member_helicity = {
                tuple(h.values): i for i, h in enumerate(reference.physics.helicities)
            }
            mapped = []
            for helicity in template.physics.helicities:
                vector = [0] * len(perm)
                for r, value in enumerate(helicity.values):
                    vector[perm[r]] = rho[r] * value
                mapped.append(member_helicity[tuple(vector)])
            assert sorted(mapped) == list(range(len(member_helicity)))
            member_flows = {
                _strings(flow.word, member.expression): i
                for i, flow in enumerate(reference.physics.color_flows)
            }
            flows = [
                member_flows[
                    _strings([perm[x - 1] + 1 for x in flow.word], member.expression)
                ]
                for flow in template.physics.color_flows
            ]
            assert sorted(flows) == list(range(len(member_flows)))
            rows = template.evaluate_assignments(
                signed, [[index]] * 4, output="helicity-flow"
            )
            resolved = reference.evaluate_resolved(points)
            for point, row in enumerate(rows):
                values = resolved.values[point]
                scale = max(abs(complex(v).real) for line in values for v in line)
                for h, line in enumerate(row[0]):
                    for c, value in enumerate(line):
                        own = complex(values[mapped[h]][flows[c]]).real
                        _close(value, own, scale, (member.expression, point, h, c))
            checked += 1
    assert checked > 0


def test_every_crossed_member_has_a_full_colour_target_route(
    artifacts: tuple[str, dict[str, Path]],
) -> None:
    _, paths = artifacts
    index = _route_index(_target_routes(load_umami_metadata(paths["full"])), 2)
    routed = 0
    for _, _, members in _crossed_groups(paths["crossing"]):
        for member in members:
            process, _artifact, legs, factor, assignment, key = _route(
                index, member.pdgs, 2
            )
            assert sorted(legs) == list(range(len(member.pdgs)))
            assert factor > 0 and (assignment < 0 or isinstance(key, str))
            assert len(process["pdgs"]) == len(member.pdgs)
            routed += 1
    assert routed > 0


def test_crossed_members_export_as_template_references(
    artifacts: tuple[str, dict[str, Path]], tmp_path: Path
) -> None:
    """The CLI exports each crossed member as a reference to its template's program
    (v2, sigma crossing, rho helicity labels, own flows, signs and factor); its
    reference values equal the member's own separately generated program."""
    from pyamplicol.cli.recurrence_export import parse_export_recurrence

    name, paths = artifacts
    kept = os.environ.get("PYAMPLICOL_RECURRENCE_EXPORT_ARTIFACTS")
    if kept and name == "pp2j":
        # CI's Rust parity test interprets the crossed exports bit for bit as well.
        destination = Path(kept) / "flavour-union-crossing"
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(paths["crossing"], destination)
    crossed = [
        (template_id, member)
        for template_id, _, members in _crossed_groups(paths["crossing"])
        for member in members
    ]
    points = {
        member.process_id: _member_points(member.expression) for _, member in crossed
    }
    references = tmp_path / "points.json"
    references.write_text(json.dumps(points), encoding="utf-8")
    output = tmp_path / "export.json"
    parse_export_recurrence(
        [
            str(paths["crossing"]),
            "-o",
            str(output),
            "--process",
            crossed[0][1].process_id,
            "--reference-points",
            str(references),
        ]
        + [
            option
            for _, member in crossed[1:]
            for option in ("--process", member.process_id)
        ]
    ).run()
    combined = json.loads(output.read_text(encoding="utf-8"))
    assert combined["schema_version"] == 3
    assert combined["execution_semantics"] == "rusticol-batched-f64-v2"
    full = {
        (entry["process_key"], entry["colour_id"]): entry
        for entry in combined["programs"]
        if "program_reference" not in entry
    }
    entries = [entry for entry in combined["programs"] if "program_reference" in entry]
    checked = 0
    for template_id, member in crossed:
        template = Runtime.load(paths["crossing"], process=template_id)
        own = [entry for entry in entries if entry["process_key"] == member.process_id]
        assert len(own) == len(template.physics.color_flows)
        perm, sigma = list(member.permutation), list(member.crossing_momentum_signs)
        reference = Runtime.load(paths["reference"], process=member.expression)
        member_helicity = {h.id: i for i, h in enumerate(reference.physics.helicities)}
        member_flows = {
            _strings(flow.word, member.expression): i
            for i, flow in enumerate(reference.physics.color_flows)
        }
        resolved = reference.evaluate_resolved(points[member.process_id])
        for entry in own:
            target = entry["program_reference"]
            assert target["process_key"] == template_id
            program = full[(target["process_key"], target["colour_id"])]
            assert entry["crossing"] == [
                {"program_leg": r, "caller_leg": leg, "sign": float(s)}
                for r, (leg, s) in enumerate(zip(perm, sigma, strict=True))
            ]
            assert [leg["pdg"] for leg in entry["caller_legs"]] == list(member.pdgs)
            assert entry["group_factor"] == {
                "numerator": member.factor.numerator,
                "denominator": member.factor.denominator,
            }
            signs = [0] * len(program["pattern_signs"])
            for pattern, sign in zip(
                member.patterns or (), member.pattern_signs or (), strict=True
            ):
                signs[pattern] = sign
            assert entry["pattern_signs"] == signs
            # template helicity order; ids are the member's own labels
            assert len(entry["helicity_outputs"]) == len(program["helicity_outputs"])
            columns = [member_helicity[h["id"]] for h in entry["helicity_outputs"]]
            word = [int(label) for label in entry["colour_id"][5:].split(",")]
            flow = member_flows[_strings(word, member.expression)]
            values = entry["reference_values"]["values"]
            for point, row in enumerate(values):
                line = resolved.values[point]
                scale = max(abs(complex(v).real) for col in line for v in col)
                for k, value in enumerate(row):
                    expected = complex(line[columns[k]][flow]).real
                    _close(value, expected, scale, (member.expression, point, k))
            checked += 1
    assert checked > 0


def test_umami_runtime_is_the_program_of_a_crossed_member(
    artifacts: tuple[str, dict[str, Path]],
) -> None:
    """As the recurrence-export docs say: the program is the runtime's
    ``(process_id, color_id)`` at ``runtime.permutation``/``momentum_signs``,
    never the member's own flow mapped through pi (identical-orbit members run
    another template flow at a relabelled gather)."""
    _, paths = artifacts
    members = {
        member.process_id: member
        for _, _, crossed in _crossed_groups(paths["crossing"])
        for member in crossed
    }
    templates: dict[str, Runtime] = {}
    references: dict[str, Runtime] = {}
    checked = swapped = 0
    for document in load_umami_metadata(paths["crossing"]):
        for channel in document["channels"]:
            for process in channel["processes"]:
                for entry in process["members"]:
                    runtime = entry["runtime"]
                    member = members.get(entry["process_id"])
                    if member is None or "momentum_signs" not in runtime:
                        continue
                    template_id = runtime["process_id"]
                    template = templates.setdefault(
                        template_id,
                        Runtime.load(paths["crossing"], process=template_id),
                    )
                    flow_ids = [
                        "flow:" + ",".join(map(str, flow.word))
                        for flow in template.physics.color_flows
                    ]
                    axis = flow_ids.index(runtime["color_id"])
                    assert runtime["color_axis"] == axis
                    perm = list(member.permutation)
                    # the flow that program_reference would name
                    own = [int(x) for x in entry["color_id"][5:].split(",")]
                    mapped = [
                        "flow:" + ",".join(str(perm[x - 1] + 1) for x in flow.word)
                        for flow in template.physics.color_flows
                    ]
                    swapped += (
                        flow_ids[mapped.index(entry["color_id"])]
                        != (runtime["color_id"])
                    )
                    reference = references.setdefault(
                        member.expression,
                        Runtime.load(paths["reference"], process=member.expression),
                    )
                    flows = {
                        _strings(flow.word, member.expression): i
                        for i, flow in enumerate(reference.physics.color_flows)
                    }
                    column = flows[_strings(own, member.expression)]
                    points = _member_points(member.expression)
                    perm_r, signs_r = runtime["permutation"], runtime["momentum_signs"]
                    signed = [_signed(point, perm_r, signs_r) for point in points]
                    rows = template.evaluate_assignments(
                        signed,
                        [[runtime["assignment"]]] * len(points),
                        output="helicity-flow",
                    )
                    resolved = reference.evaluate_resolved(points)
                    for point, row in enumerate(rows):
                        value = sum(line[axis] for line in row[0])
                        expected = sum(
                            complex(line[column]).real
                            for line in resolved.values[point]
                        )
                        label = (entry["process_id"], point)
                        _close(value, expected, abs(expected), label)
                    checked += 1
    assert checked > 0
    # pp2j has d d~ > g g style members whose second flow is an orbit copy
    assert swapped > 0
