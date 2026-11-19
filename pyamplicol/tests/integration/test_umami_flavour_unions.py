# SPDX-License-Identifier: 0BSD
"""UMAMI and reweight providers of flavour-union artifacts.

A union member is evaluated as an assignment of its template's runtime. Its
matrix elements, colour tags, ratios and flavour-switch weights must equal
those of an artifact generated per flavour (strict flavour-blind classes):
conjugated members, members with exchanged beams and a six-quark union
included. The full-colour target may have flavour unions too.
"""

from __future__ import annotations

import ctypes as ct
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pyamplicol import Generator, ModelSource, ProcessSet, Runtime
from pyamplicol.artifacts.umami_reweight import (
    load_umami_metadata,
    write_reweight_provider,
)
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
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
from pyamplicol.models.builtin.validation import generic_validation_point
from tests.integration.test_umami_api import (
    _canonical_tags,
    _config_executable,
    _library,
    _unavailable,
)
from tests.integration.test_umami_flavor_switch import IN, OUT

EXPRESSIONS = (
    "d d~ > d d~ g g",
    "u u~ > d d~ g g",
    "d~ d > d d~ g g",
    "u~ u > s s~ g g",
    "d d > d d g g",
    "u d > u d g g",
    "u~ d~ > u~ d~ g g",
    "g g > d d~ g g",
    "g g > u u~ g g",
    "d d~ > d d d~ d~",
    "d d~ > u s u~ s~",
    "u u~ > d d d~ d~",
    "d~ d > u u~ s s~",
)
NAMES = tuple(f"p{index}" for index in range(len(EXPRESSIONS)))
NAMES_OF = {21: "g", 1: "d", 2: "u", 3: "s", 4: "c"}
SWITCH_LINES = 3


def _config(accuracy: str, unions: bool) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy=accuracy),
        process=ProcessConfig(
            flavor_scheme=4,
            flavor_blind_strict=True,
            flavor_unions=unions,
            max_coupling_orders={"QED": 0},
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=True,
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


def _make(directory: Path, config_executable: str, *arguments: str) -> None:
    # The installed SDK: the environment's PYTHONPATH, not the source tree.
    completed = subprocess.run(
        [
            "make",
            "-C",
            str(directory),
            f"RUSTICOL_CONFIG={config_executable}",
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    config_executable = _config_executable()
    if shutil.which("make") is None or shutil.which(os.environ.get("CC", "cc")) is None:
        _unavailable("UMAMI integration requires make and a C compiler")
    root = tmp_path_factory.mktemp("umami-unions")
    artifacts = {
        "union": root / "lc-union",
        "flavour": root / "lc-flavour",
        "full": root / "full",
        "full-union": root / "full-union",
    }
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        for kind, path in artifacts.items():
            Generator(
                _config(
                    "full" if kind.startswith("full") else "lc",
                    kind in ("union", "full-union"),
                )
            ).generate(
                ProcessSet.from_expressions(EXPRESSIONS, names=NAMES),
                path,
                model=ModelSource.from_path(prepared_model),
            )
    result = {"artifacts": artifacts, "umami": {}, "reweight": {}}
    # (LC reference, full-colour target) of each reweight provider
    providers = {
        "union": ("union", "full"),
        "flavour": ("flavour", "full"),
        "union-full": ("union", "full-union"),
        "flavour-full": ("flavour", "full-union"),
    }
    for kind in ("union", "flavour"):
        build = root / f"build-{kind}"
        _make(artifacts[kind] / "API/umami", config_executable, f"BUILD_DIR={build}")
        result["umami"][kind] = [
            (data, _library(build / data["provider"]["library"]))
            for data in load_umami_metadata(artifacts[kind])
        ]
    for kind, (reference, target) in providers.items():
        output = root / f"reweight-{kind}"
        ids = write_reweight_provider(
            artifacts[reference], artifacts[target], output, switch_lines=SWITCH_LINES
        )
        _make(output, config_executable)
        result["reweight"][kind] = {
            identifier: (
                output,
                json.loads((output / f"flavor_switch_{identifier}.json").read_text()),
                _library(output / f"build/libreweight_{identifier}.so"),
            )
            for identifier in ids
        }
    yield result


def _handle(library, artifact):
    handle = ct.c_void_p()
    assert library.umami_initialize(ct.byref(handle), os.fsencode(artifact)) == 0
    return handle


def _points(count: int = 3):
    """Massless 2 -> 4 points (any flavour of the set), rotated about the beam."""

    base = [
        tuple(map(float, p.momentum)) for p in generic_validation_point("g g > g g g g")
    ]
    points = []
    for index in range(count):
        angle = 0.7 + 1.3 * index
        cos, sin = math.cos(angle), math.sin(angle)
        points.append(
            tuple((e, cos * x - sin * y, sin * x + cos * y, z) for e, x, y, z in base)
        )
    return points


def _members(documents):
    """(process id, canonical physical tags) -> (provider, channel, flavour,
    entry, member) for every exported physical contribution."""

    result = {}
    for provider, (data, _) in enumerate(documents):
        for channel, description in enumerate(data["channels"]):
            for flavour, entry in enumerate(description["processes"]):
                for member in entry["members"]:
                    tags = _canonical_tags(data["color_flows"][member["color_flows"]])
                    result.setdefault(
                        (member["process_id"], tags),
                        (provider, channel, flavour, entry, member),
                    )
    return result


def _to_input(entry, member, values):
    """Physical-leg values (momenta, helicities) in the entry's input legs."""

    mapped = [None] * len(values)
    for j, slot in enumerate(entry["runtime"]["permutation"]):
        mapped[slot] = values[member["runtime"]["permutation"][j]]
    return tuple(mapped)


def _to_physical(entry, member, values):
    physical = [None] * len(values)
    for j, slot in enumerate(entry["runtime"]["permutation"]):
        physical[member["runtime"]["permutation"][j]] = values[slot]
    return tuple(physical)


def _call(library, handle, events, *, helicity=None, sample=False):
    """One UMAMI call: events are (input momenta, channel, flavour)."""

    count, n = len(events), len(events[0][0])
    momenta = (ct.c_double * (4 * n * count))(
        *(
            float(events[event][0][leg][component])
            for component in range(4)
            for leg in range(n)
            for event in range(count)
        )
    )
    inputs = {
        IN["UMAMI_IN_MOMENTA"]: momenta,
        IN["UMAMI_IN_FLAVOR_INDEX"]: (ct.c_int * count)(*(e[2] for e in events)),
        IN["UMAMI_IN_CHANNEL_INDEX"]: (ct.c_uint * count)(*(e[1] for e in events)),
        IN["UMAMI_IN_RANDOM_HELICITY"]: (ct.c_double * count)(*[0.37] * count),
    }
    if helicity is not None:
        inputs[IN["UMAMI_IN_HELICITY_INDEX"]] = (ct.c_int * count)(*helicity)
    outputs = {
        OUT["UMAMI_OUT_MATRIX_ELEMENT"]: (ct.c_double * count)(),
        OUT["UMAMI_OUT_COLOR_INDEX"]: (ct.c_int * count)(),
    }
    if sample:
        outputs[OUT["UMAMI_OUT_HELICITY_INDEX"]] = (ct.c_int * count)()
    status = library.umami_matrix_element(
        handle,
        count,
        count,
        0,
        len(inputs),
        (ct.c_int * len(inputs))(*inputs),
        (ct.c_void_p * len(inputs))(
            *(ct.cast(v, ct.c_void_p) for v in inputs.values())
        ),
        len(outputs),
        (ct.c_int * len(outputs))(*outputs),
        (ct.c_void_p * len(outputs))(
            *(ct.cast(v, ct.c_void_p) for v in outputs.values())
        ),
    )
    assert status == 0
    values = [list(v) for v in outputs.values()]
    return values if sample else values[:2]


def test_union_artifacts_route_members_as_assignments(built):
    groups = json.loads(
        (built["artifacts"]["union"] / "process-groups.json").read_text()
    )
    members = [
        member
        for group in groups["groups"]
        for member in group["members"]
        if "flavour_union_patterns" in member
    ]
    assert any(member["conjugated"] for member in members)
    assert any(member["permutation"][:2] == [1, 0] for member in members)
    assert any(len([p for p in member["pdgs"] if p != 21]) == 6 for member in members)
    runtimes = [
        entry["runtime"]
        for data, _ in built["umami"]["union"]
        for channel in data["channels"]
        for entry in channel["processes"]
    ]
    assert any(runtime.get("assignment", 0) > 0 for runtime in runtimes)


def _partition(documents):
    """Per provider channel, the physical flavours of each entry."""

    return [
        sorted(
            sorted(
                tuple(data["pdg_ids"][term["pdg_ids"]])
                for term in entry["matrix_elements"]
            )
            for entry in channel["processes"]
        )
        for data, _ in documents
        for channel in data["channels"]
    ]


def test_union_partition_equals_per_flavour(built):
    # a template's own flavour shares its entry with all-pattern members
    assert _partition(built["umami"]["union"]) == _partition(built["umami"]["flavour"])


def test_union_matrix_elements_equal_per_flavour_artifacts(built):
    unions, flavours = built["umami"]["union"], built["umami"]["flavour"]
    expected = _members(flavours)
    union_members = _members(unions)
    assert set(union_members) == set(expected)
    artifacts = built["artifacts"]
    handles = {
        "union": [_handle(library, artifacts["union"]) for _, library in unions],
        "flavour": [_handle(library, artifacts["flavour"]) for _, library in flavours],
    }
    try:
        points = _points()
        mixed = {provider: [] for provider in range(len(unions))}
        for key, (provider, channel, flavour, entry, member) in union_members.items():
            other = expected[key]
            for point in points:
                ours = _call(
                    unions[provider][1],
                    handles["union"][provider],
                    [(_to_input(entry, member, point), channel, flavour)],
                )
                theirs = _call(
                    flavours[other[0]][1],
                    handles["flavour"][other[0]],
                    [(_to_input(other[3], other[4], point), other[1], other[2])],
                )
                assert ours[0][0] == pytest.approx(
                    theirs[0][0], rel=1e-12, abs=0.0
                ), key
                # LHE colour tags of the event, compared in physical legs
                tags = unions[provider][0]["color_flows"][ours[1][0]]
                other_tags = flavours[other[0]][0]["color_flows"][theirs[1][0]]
                assert _canonical_tags(_to_physical(entry, member, tags)) == (
                    _canonical_tags(_to_physical(other[3], other[4], other_tags))
                ), key
                mixed[provider].append(
                    ((_to_input(entry, member, point), channel, flavour), ours[0][0])
                )
            # fixed helicities, mapped through the physical legs
            data, other_data = unions[provider][0], flavours[other[0]][0]
            for label in entry["runtime"]["helicity_export"][:4]:
                physical = _to_physical(entry, member, data["helicities"][label])
                other_label = other_data["helicities"].index(
                    list(_to_input(other[3], other[4], physical))
                )
                point = _to_input(entry, member, points[0])
                ours = _call(
                    unions[provider][1],
                    handles["union"][provider],
                    [(point, channel, flavour)],
                    helicity=[label],
                )
                theirs = _call(
                    flavours[other[0]][1],
                    handles["flavour"][other[0]],
                    [(_to_input(other[3], other[4], points[0]), other[1], other[2])],
                    helicity=[other_label],
                )
                assert ours[0][0] == pytest.approx(theirs[0][0], rel=1e-12, abs=0.0)
        # every flavour and flow of a provider in one call: the same values
        for provider, events in mixed.items():
            order = np.random.default_rng(5).permutation(len(events))
            batch = [events[i][0] for i in order]
            values = _call(unions[provider][1], handles["union"][provider], batch)[0]
            for value, index in zip(values, order, strict=True):
                assert value == pytest.approx(events[index][1], rel=1e-13, abs=0.0)
            sampled = _call(
                unions[provider][1], handles["union"][provider], batch, sample=True
            )
            assert sampled[0] == pytest.approx(values, rel=1e-13, abs=0.0)
            assert all(label >= 0 for label in sampled[2])
    finally:
        for kind, documents in (("union", unions), ("flavour", flavours)):
            for (_, library), handle in zip(documents, handles[kind], strict=True):
                assert library.umami_free(handle) == 0


def _densities():
    rng = np.random.default_rng(3)
    densities = rng.uniform(0.2, 2.0, size=(2, 13))
    densities[:, [0, 12]] = 0.0
    return densities


def _ratio_call(
    library,
    handle,
    events,
    densities,
    *,
    switch,
    random_flavor=None,
    random_helicity=None,
):
    """Events are (input momenta, channel, flavour, helicity label)."""

    count, n = len(events), len(events[0][0])
    assert (
        library.umami_set_parameter(handle, b"flavor_switch", float(switch), 0.0) == 0
    )
    momenta = (ct.c_double * (4 * n * count))(
        *(
            float(events[event][0][leg][k])
            for k in range(4)
            for leg in range(n)
            for event in range(count)
        )
    )
    pdfs = (ct.c_double * (26 * count))(
        *(
            densities[leg, column]
            for column in range(13)
            for leg in range(2)
            for _ in range(count)
        )
    )
    random_flavor = [0.0] * count if random_flavor is None else random_flavor
    random_helicity = [0.5] * count if random_helicity is None else random_helicity
    inputs = {
        IN["UMAMI_IN_MOMENTA"]: momenta,
        IN["UMAMI_IN_FLAVOR_INDEX"]: (ct.c_int * count)(*(e[2] for e in events)),
        IN["UMAMI_IN_HELICITY_INDEX"]: (ct.c_int * count)(*(e[3] for e in events)),
        IN["UMAMI_IN_CHANNEL_INDEX"]: (ct.c_uint * count)(*(e[1] for e in events)),
        IN["UMAMI_IN_PARTON_DENSITIES"]: pdfs,
        IN["UMAMI_IN_RANDOM_FLAVOR"]: (ct.c_double * count)(*random_flavor),
        IN["UMAMI_IN_RANDOM_HELICITY"]: (ct.c_double * count)(*random_helicity),
        IN["UMAMI_IN_RANDOM_COLOR"]: (ct.c_double * count)(*[0.5] * count),
    }
    names = [
        "UMAMI_OUT_MATRIX_ELEMENT_RATIO",
        "UMAMI_OUT_FLAVOR_INDEX",
        "UMAMI_OUT_HELICITY_INDEX",
        "UMAMI_OUT_COLOR_INDEX",
    ]
    outputs = [(ct.c_double * count)()] + [(ct.c_int * count)() for _ in range(3)]
    status = library.umami_matrix_element(
        handle,
        count,
        count,
        0,
        len(inputs),
        (ct.c_int * len(inputs))(*inputs),
        (ct.c_void_p * len(inputs))(
            *(ct.cast(v, ct.c_void_p) for v in inputs.values())
        ),
        len(outputs),
        (ct.c_int * len(outputs))(*(OUT[name] for name in names)),
        (ct.c_void_p * len(outputs))(*(ct.cast(v, ct.c_void_p) for v in outputs)),
    )
    assert status == 0
    return [np.array(list(o)) for o in outputs]


def _flavour_events(documents, provider, point):
    """One unswitched event per (channel, flavour) of a provider: (key, event)
    with the key (process id, physical helicity) of its first member."""

    data = documents[provider][0]
    events = []
    for channel, description in enumerate(data["channels"]):
        for flavour, entry in enumerate(description["processes"]):
            if not entry["members"]:
                continue  # a phase-space partner copy
            member = entry["members"][0]
            for label in entry["runtime"]["helicity_export"][::7][:3]:
                physical = _to_physical(entry, member, data["helicities"][label])
                events.append(
                    (
                        (member["process_id"], physical),
                        (_to_input(entry, member, point), channel, flavour, label),
                    )
                )
    return events


def _process(pdgs):
    names = [NAMES_OF[abs(p)] + ("~" if p < 0 else "") for p in pdgs]
    return " ".join(names[:2]) + " > " + " ".join(names[2:])


@pytest.mark.parametrize(
    ("kind", "per_flavour"),
    [("union", "flavour"), ("union-full", "flavour"), ("flavour-full", "flavour")],
)
def test_union_ratios_and_switch_weights_equal_per_flavour_artifacts(
    built, kind, per_flavour
):
    """LC unions, full-colour unions and both against per-flavour artifacts."""

    artifacts = built["artifacts"]
    reference = "flavour" if kind == "flavour-full" else "union"
    unions, flavours = built["umami"][reference], built["umami"][per_flavour]
    assert len(unions) == len(flavours) == 1
    ((_, union),) = built["reweight"][kind].items()
    union_dir, union_sidecar, union_library = union
    ((_, (flavour_dir, _, flavour_library)),) = built["reweight"][per_flavour].items()
    assert union_sidecar["targets"]
    union_handle = _handle(union_library, union_dir)
    flavour_handle = _handle(flavour_library, flavour_dir)
    lc_library = unions[0][1]
    lc_handle = _handle(lc_library, artifacts[reference])
    try:
        densities = _densities()
        for point in _points(2):
            ours = _flavour_events(unions, 0, point)
            theirs = dict(_flavour_events(flavours, 0, point))
            events = [event for _, event in ours]
            other = [theirs[key] for key, _ in ours]
            # Helicities with numerically zero amplitudes (1e-30 of the largest)
            # have rounding-noise ratios that depend on the identical-leg order.
            lc = np.array(
                _call(
                    lc_library,
                    lc_handle,
                    [event[:3] for event in events],
                    helicity=[event[3] for event in events],
                )[0]
            )
            kept = (lc == 0.0) | (lc > 1e-20 * lc.max())
            for switch in (False, True):
                # unswitched events (random flavour 0): the ratio, times the
                # summed flavours over the generated ones when switching
                a = _ratio_call(
                    union_library, union_handle, events, densities, switch=switch
                )
                b = _ratio_call(
                    flavour_library, flavour_handle, other, densities, switch=switch
                )
                assert (a[1] == -1).all() and (b[1] == -1).all()
                assert a[0][kept] == pytest.approx(b[0][kept], rel=1e-12, abs=0.0)
                assert (a[0] > 0).any()
        # switched events: the target's full colour over its leading colour
        point = _points(1)[0]
        lc = {}
        full = {}
        event = _flavour_events(unions, 0, point)[0][1]
        base = _ratio_call(
            union_library, union_handle, [event], densities, switch=False
        )[0][0]
        stay = _ratio_call(
            union_library, union_handle, [event], densities, switch=True
        )[0][0]
        enhancement = stay / base
        assert enhancement > 1.0
        count = 600
        grid = (np.arange(count) + 0.5) / count
        random_flavor = 1 / enhancement + grid * (1 - 1 / enhancement)
        random_helicity = np.modf(np.arange(count) * (math.sqrt(5) - 1) / 2)[0]
        frequencies = {}
        for library, handle, sidecar in (
            (union_library, union_handle, union_sidecar),
            (
                flavour_library,
                flavour_handle,
                next(iter(built["reweight"][per_flavour].values()))[1],
            ),
        ):
            ratio, target, helicity, color = _ratio_call(
                library,
                handle,
                [event] * count,
                densities,
                switch=True,
                random_flavor=list(random_flavor),
                random_helicity=list(random_helicity),
            )
            assert (target >= 0).all()
            counted = {}
            for k in target:
                pdgs = tuple(sidecar["targets"][k])
                counted[pdgs] = counted.get(pdgs, 0) + 1
            frequencies[library is union_library] = counted
            data = (unions if library is union_library else flavours)[0][0]
            offset = sidecar["color_flow_offset"]
            for value, k, label, c in zip(ratio, target, helicity, color, strict=True):
                pdgs = tuple(sidecar["targets"][k])
                if pdgs not in lc:
                    # the per-flavour artifacts' own processes
                    lc[pdgs] = Runtime.load(
                        artifacts["flavour"], process=_process(pdgs)
                    )
                    full[pdgs] = Runtime.load(artifacts["full"], process=_process(pdgs))
                vector = data["helicities"][label]
                identifier = "h:" + ",".join(f"{v:+d}" for v in vector)
                resolved = lc[pdgs].evaluate_resolved((point,))
                index = list(lc[pdgs].physics.helicity_ids).index(identifier)
                row = float(np.array(resolved.values)[0].real[index].sum())
                fc = complex(
                    full[pdgs].evaluate((point,), helicities=[identifier])[0]
                ).real
                assert value == pytest.approx(enhancement * fc / row, rel=1e-12), pdgs
                tags = (
                    sidecar["color_flows"][c - offset]
                    if c >= offset
                    else data["color_flows"][c]
                )
                for pdg, (colour, anticolour) in zip(pdgs, tags, strict=True):
                    if pdg > 0 and pdg != 21:
                        assert colour and not anticolour
                    elif pdg < 0:
                        assert anticolour and not colour
        # every target is drawn as often (a grid of random numbers; their
        # order differs, so a share below the grid step may be missed)
        for pdgs in set(frequencies[True]) | set(frequencies[False]):
            difference = frequencies[True].get(pdgs, 0) - frequencies[False].get(
                pdgs, 0
            )
            assert abs(difference) <= 2, pdgs
    finally:
        assert union_library.umami_free(union_handle) == 0
        assert flavour_library.umami_free(flavour_handle) == 0
        assert lc_library.umami_free(lc_handle) == 0


def test_switch_shares_cover_every_flavour_once(built):
    """The union catalogue lists the same orderings (flavour, beam order, final
    order) and weights as the per-flavour one, and fewer native calls."""

    from pyamplicol.artifacts.umami_reweight import switch_catalogue

    catalogues = {}
    for kind in ("union", "flavour"):
        ((data, _),) = built["umami"][kind]
        groups = json.loads(
            (built["artifacts"][kind] / "process-groups.json").read_text()
        )
        catalogues[kind] = switch_catalogue(data, groups, SWITCH_LINES)

    def orderings(catalogue):
        result = {}
        targets = iter(catalogue["target_pdgs"])
        for ordering in catalogue["orderings"]:
            evaluation = catalogue["evaluations"][ordering["evaluation"]]
            key = (tuple(ordering["pdf"]), ordering["target"] >= 0)
            if ordering["target"] >= 0:
                key += (tuple(next(targets)),)
            result.setdefault(key, []).append(
                (ordering["weight"], evaluation["process"])
            )
        return result

    union, flavour = orderings(catalogues["union"]), orderings(catalogues["flavour"])
    assert {k for k in union if k[1]} == {k for k in flavour if k[1]}
    assert sum(len(v) for v in union.values()) == sum(len(v) for v in flavour.values())
    assert len(catalogues["union"]["calls"]) < len(catalogues["flavour"]["calls"])


def test_relabelled_switch_calls_equal_their_evaluations(built):
    """With the union's pattern table, the switch evaluates every template leg
    type once: an evaluation at legs l is its assignment relabelled by
    pi[r] = c^-1[l[r]] at the call's legs c. Each must equal the assignment at
    its own legs (the phase-5 call), with the same orderings."""

    from pyamplicol.artifacts.umami_reweight import switch_catalogue

    ((data, _),) = built["umami"]["union"]
    artifact = built["artifacts"]["union"]
    groups = json.loads((artifact / "process-groups.json").read_text())
    assert any("flavour_union_pattern_pairs" in group for group in groups["groups"])
    plain = json.loads(json.dumps(groups))
    for group in plain["groups"]:
        group.pop("flavour_union_pattern_pairs", None)
    relabelled = switch_catalogue(data, groups, SWITCH_LINES)
    unrelabelled = switch_catalogue(data, plain, SWITCH_LINES)
    assert relabelled["evaluations"] == unrelabelled["evaluations"]
    assert relabelled["orderings"] == unrelabelled["orderings"]
    assert relabelled["targets"] == unrelabelled["targets"]
    assert len(relabelled["calls"]) < len(unrelabelled["calls"])
    assert any(call.get("permutations") for call in relabelled["calls"])
    runtimes = {}
    points = _points(2)
    for call in relabelled["calls"]:
        if call["assignments"] is None:
            continue
        process = data["runtime_processes"][call["process"]]
        runtime = runtimes.setdefault(
            call["process"], Runtime.load(artifact, process=process["id"])
        )
        permutations = call["permutations"] or [None] * len(call["assignments"])
        ids = [
            assignment if permutation is None
            else runtime.relabel_assignment(assignment, permutation)
            for assignment, permutation in zip(
                call["assignments"], permutations, strict=True
            )
        ]
        gathered = [tuple(p[slot] for slot in call["legs"]) for p in points]
        values = runtime.evaluate_assignments(gathered, [ids] * len(points))
        for evaluation, entry in zip(call["evaluations"], call["entries"], strict=True):
            own = relabelled["evaluations"][evaluation]
            at_legs = [tuple(p[slot] for slot in own["legs"]) for p in points]
            expected = runtime.evaluate_assignments(
                at_legs, [[own["assignment"]]] * len(points)
            )
            assert [row[entry] for row in values] == pytest.approx(
                [row[0] for row in expected], rel=1e-12
            )
