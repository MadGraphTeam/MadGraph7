# SPDX-License-Identifier: 0BSD
"""UMAMI and reweight providers of flavour-union artifacts.

A union member is evaluated as an assignment of its template's runtime. Its
matrix elements, colour tags, ratios and flavour-switch weights must equal
those of an artifact generated per flavour (strict flavour-blind classes):
conjugated members, members with exchanged beams and a six-quark union
included. The full-colour target may have flavour unions too. With
process.flavor_union_crossing, leading-colour members may also be crossed
(sign-flipped momenta, helicities flipped by rho). A selector-only full-colour
union target (MG7 with generation.selector_only_full_colour) gives the same
ratios.
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
from pyamplicol.artifacts.umami_c import helicity_export
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


def _config(
    accuracy: str, unions: bool, crossing: bool = False, selector_only: bool = False
) -> RunConfig:
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
            emit_api_bundle=True,
            selector_only_full_colour=selector_only,
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
        "crossing": root / "lc-crossing",
        "full": root / "full",
        "full-union": root / "full-union",
        "full-union-selector": root / "full-union-selector",
    }
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        for kind, path in artifacts.items():
            Generator(
                _config(
                    "full" if kind.startswith("full") else "lc",
                    kind in ("union", "full-union", "crossing", "full-union-selector"),
                    crossing=kind == "crossing",
                    selector_only=kind == "full-union-selector",
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
        "crossing": ("crossing", "full"),
        "crossing-full": ("crossing", "full-union"),
        "union-full-selector": ("union", "full-union-selector"),
    }
    for kind in ("union", "flavour", "crossing"):
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


def test_selector_only_union_target_has_no_main_schedule(built):
    """Every class drops the main schedule; union templates evaluate their
    members on a pattern-resolved companion."""
    artifact = built["artifacts"]["full-union-selector"]
    pattern_resolved = []
    for process in json.loads((artifact / "artifact.json").read_text())["processes"]:
        execution = json.loads(
            (artifact / "processes" / process["id"] / "execution.json").read_text()
        )
        capabilities = execution["required_runtime_capabilities"]
        assert "rusticol.recurrence-selector-only-fc.v1" in capabilities
        assert execution["plan"] is None, process["id"]
        resolution = execution["helicity_selector_companion"].get("pattern_resolution")
        assert (resolution is not None) == (
            "rusticol.recurrence-pattern-resolved-companion.v1" in capabilities
        )
        pattern_resolved.append(resolution is not None)
    assert any(pattern_resolved) and not all(pattern_resolved)


def _partition(documents, ordered=False):
    """Per provider channel, the physical flavours of each entry (in export
    order with ``ordered``, the flavour sampling order of MG7)."""

    arrange = list if ordered else sorted
    return [
        arrange(
            arrange(
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
    assert _partition(built["umami"]["union"], ordered=True) == _partition(
        built["umami"]["flavour"], ordered=True
    )


# g q > q Q Q~: the shape of every partition split left at pp > 3j, 4j and 5j
GQ_EXPRESSIONS = (
    "g u > d u d~",
    "g u > u s s~",
    "g u > u c c~",
    "g u > u u u~",
    "g d > d u u~",
    "g d > d s s~",
    "g d > d c c~",
    "g d > d d d~",
)


PARTITION_CASES = {
    "gq": GQ_EXPRESSIONS,
    "pp3j": ("p p > j j j",),
    "pp4j": ("p p > j j j j",),
}
# pp3j and pp4j run only with PYAMPLICOL_RUN_FLAVOUR_UNION=pp3j,pp4j (either or both)
PARTITION_OPT_IN = {"pp3j", "pp4j"}


@pytest.fixture(scope="module", params=tuple(PARTITION_CASES))
def partitions(request, tmp_path_factory):
    opted = os.environ.get("PYAMPLICOL_RUN_FLAVOUR_UNION", "").split(",")
    if request.param in PARTITION_OPT_IN and request.param not in opted:
        pytest.skip(f"set PYAMPLICOL_RUN_FLAVOUR_UNION={request.param} to run it")
    expressions = PARTITION_CASES[request.param]
    root = tmp_path_factory.mktemp(f"umami-unions-{request.param}")
    result = {}
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        for kind, unions in (("union", True), ("flavour", False)):
            Generator(_config("lc", unions)).generate(
                ProcessSet.from_expressions(
                    expressions,
                    names=tuple(f"q{index}" for index in range(len(expressions))),
                ),
                root / kind,
                model=ModelSource.from_path(prepared_model),
            )
            result[kind] = _partition(
                [(data, None) for data in load_umami_metadata(root / kind)],
                ordered=True,
            )
    return result


def test_union_partition_equals_per_flavour_sets(partitions):
    # members of one flavour class share its union route (g u > d u d~ included),
    # and entries and their terms are exported in the same order
    assert partitions["union"] == partitions["flavour"]


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
            for label in helicity_export(data, entry["runtime"])[:4]:
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
    random_color=None,
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
    random_color = [0.5] * count if random_color is None else random_color
    inputs = {
        IN["UMAMI_IN_MOMENTA"]: momenta,
        IN["UMAMI_IN_FLAVOR_INDEX"]: (ct.c_int * count)(*(e[2] for e in events)),
        IN["UMAMI_IN_HELICITY_INDEX"]: (ct.c_int * count)(*(e[3] for e in events)),
        IN["UMAMI_IN_CHANNEL_INDEX"]: (ct.c_uint * count)(*(e[1] for e in events)),
        IN["UMAMI_IN_PARTON_DENSITIES"]: pdfs,
        IN["UMAMI_IN_RANDOM_FLAVOR"]: (ct.c_double * count)(*random_flavor),
        IN["UMAMI_IN_RANDOM_HELICITY"]: (ct.c_double * count)(*random_helicity),
        IN["UMAMI_IN_RANDOM_COLOR"]: (ct.c_double * count)(*random_color),
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


def _flavour_events(documents, provider, point, every=False):
    """One unswitched event per (channel, flavour) of a provider: (key, event)
    with the key (process id, physical helicity) of its first member; a few
    exported helicities each, or all of them with ``every``."""

    data = documents[provider][0]
    events = []
    for channel, description in enumerate(data["channels"]):
        for flavour, entry in enumerate(description["processes"]):
            if not entry["members"]:
                continue  # a phase-space partner copy
            member = entry["members"][0]
            exported = helicity_export(data, entry["runtime"])
            for label in exported if every else exported[::7][:3]:
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


# the LC reference artifact of each reweight provider
_REFERENCE = {
    "union": "union",
    "union-full": "union",
    "flavour-full": "flavour",
    "crossing": "crossing",
    "crossing-full": "crossing",
    "union-full-selector": "union",
}


@pytest.mark.parametrize(
    ("kind", "per_flavour"),
    [
        ("union", "flavour"),
        ("union-full", "flavour"),
        ("flavour-full", "flavour"),
        ("crossing", "flavour"),
        ("crossing-full", "flavour"),
        ("union-full-selector", "flavour"),
    ],
)
def test_union_ratios_and_switch_weights_equal_per_flavour_artifacts(
    built, kind, per_flavour
):
    """LC unions (crossed or not), full-colour unions and both against
    per-flavour artifacts."""

    artifacts = built["artifacts"]
    reference = _REFERENCE[kind]
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
            theirs = dict(_flavour_events(flavours, 0, point, every=True))
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
        # the first event with a leading colour of its own and switch targets,
        # whose input legs carry the point unrelabelled (as the targets below)
        for _, event in _flavour_events(unions, 0, point):
            if not all(
                np.array_equal(a, b) for a, b in zip(event[0], point, strict=True)
            ):
                continue
            base = _ratio_call(
                union_library, union_handle, [event], densities, switch=False
            )[0][0]
            stay = _ratio_call(
                union_library, union_handle, [event], densities, switch=True
            )[0][0]
            if base > 0.0 and stay > base:
                break
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


def _assert_switched_draws_equal(union, flavour, umami, point, reference="union"):
    """union and flavour are (directory, sidecar, library) of reweight providers
    over the LC UMAMI documents umami[reference] and umami["flavour"]."""

    (union_dir, union_sidecar, union_library) = union
    (flavour_dir, flavour_sidecar, flavour_library) = flavour
    union_handle = _handle(union_library, union_dir)
    flavour_handle = _handle(flavour_library, flavour_dir)
    try:
        densities = _densities()
        theirs = dict(_flavour_events(umami["flavour"], 0, point, every=True))
        checked = 0
        for key, event in _flavour_events(umami[reference], 0, point):
            other = theirs[key]
            if not all(
                np.array_equal(a, b) and np.array_equal(a, c)
                for a, b, c in zip(event[0], other[0], point, strict=True)
            ):
                continue
            base, stay = (
                _ratio_call(union_library, union_handle, [event], densities, switch=s)[
                    0
                ][0]
                for s in (False, True)
            )
            if not (base > 0.0 and stay > base * (1 + 1e-9)):
                continue
            enhancement = stay / base
            count = 300
            grid = (np.arange(count) + 0.5) / count
            randoms = dict(
                random_flavor=list(1 / enhancement + grid * (1 - 1 / enhancement)),
                random_helicity=list(
                    np.modf(np.arange(count) * (math.sqrt(5) - 1) / 2)[0]
                ),
                random_color=list(np.modf(np.arange(count) * math.sqrt(2))[0]),
            )
            draws = []
            for library, handle, sidecar, documents, ev in (
                (union_library, union_handle, union_sidecar, umami[reference], event),
                (flavour_library, flavour_handle, flavour_sidecar, umami["flavour"], other),
            ):
                data = documents[0][0]
                ratio, target, helicity, color = _ratio_call(
                    library, handle, [ev] * count, densities, switch=True, **randoms
                )
                assert (target >= 0).all()
                assert len(data["color_flows"]) == sidecar["color_flow_offset"]
                flows = data["color_flows"] + sidecar["color_flows"]
                draws.append((
                    ratio,
                    [
                        (
                            tuple(sidecar["targets"][k]),
                            tuple(data["helicities"][h]),
                            _canonical_tags(flows[c]),
                        )
                        for k, h, c in zip(target, helicity, color, strict=True)
                    ],
                ))
            assert draws[0][1] == draws[1][1]
            assert draws[0][0] == pytest.approx(draws[1][0], rel=1e-12, abs=0.0)
            checked += 1
            if checked == 3:
                break
        assert checked
    finally:
        assert union_library.umami_free(union_handle) == 0
        assert flavour_library.umami_free(flavour_handle) == 0


@pytest.mark.parametrize("kind", ["union", "union-full", "crossing", "crossing-full"])
def test_switched_draws_equal_per_flavour_artifacts(built, kind):
    """The same random numbers switch an event to the same target, helicity and
    colour flow with the same ratio, with unions (crossed too) and per flavour."""

    ((union),) = built["reweight"][kind].values()
    ((flavour),) = built["reweight"]["flavour"].values()
    _assert_switched_draws_equal(
        union, flavour, built["umami"], _points(1)[0], _REFERENCE[kind]
    )


def test_crossed_reweight_tables_carry_their_signs(built):
    """The crossing artifact has crossed members; their reference rows and switch
    evaluations carry momentum signs, the targets none."""

    groups = json.loads(
        (built["artifacts"]["crossing"] / "process-groups.json").read_text()
    )
    crossed = {
        member["process_id"]
        for group in groups["groups"]
        for member in group["members"]
        if "crossing_momentum_signs" in member
    }
    assert crossed
    ((output, _, _),) = built["reweight"]["crossing"].values()
    from pyamplicol.artifacts.umami_reweight_data import decode_reweight_data

    tables = decode_reweight_data(next(output.glob("reweight_*.rwd")).read_bytes())
    assert any(row.reference_signs for rows in tables.channels for row in rows)
    assert any(e.signs for e in tables.switch.evaluations)
    from pyamplicol.artifacts.umami_reweight import switch_catalogue

    ((data, _),) = built["umami"]["crossing"]
    catalogue = switch_catalogue(data, groups, SWITCH_LINES)
    signed = [e for e in catalogue["evaluations"] if "signs" in e]
    assert signed and {e["assignment_key"] for e in signed} <= crossed
    for evaluation in signed:
        for r, slot in enumerate(evaluation["legs"]):
            crossed_slot = (r < 2) != (slot < 2)
            assert evaluation["signs"][r] == (-1 if crossed_slot else 1)
            assert evaluation["helicity_signs"][r] == 1 or crossed_slot
    # the full-colour targets stay unsigned
    with pytest.raises(ValueError, match="reweight targets with crossed"):
        write_reweight_provider(
            built["artifacts"]["flavour"],
            built["artifacts"]["crossing"],
            output.parent / "reweight-crossed-target",
        )


def test_reweight_provider_without_signs_is_refused(built, tmp_path):
    """Reference rows that lose the signs of crossed assignments fail at load
    (the core's crossing of the assignment differs)."""

    from dataclasses import replace

    from pyamplicol.artifacts.umami_reweight_data import (
        decode_reweight_data,
        write_reweight_data,
    )

    output = tmp_path / "stripped"
    (identifier,) = write_reweight_provider(
        built["artifacts"]["crossing"],
        built["artifacts"]["full"],
        output,
        switch_lines=SWITCH_LINES,
    )
    tables = decode_reweight_data((output / f"reweight_{identifier}.rwd").read_bytes())
    count = sum(row.reference_signs is not None for rows in tables.channels for row in rows)
    assert count
    for rows in tables.channels:
        rows[:] = [replace(row, reference_signs=None) for row in rows]
    write_reweight_data(output, tables)
    _make(output, _config_executable())
    library = _library(output / f"build/libreweight_{identifier}.so")
    handle = ct.c_void_p()
    if library.umami_initialize(ct.byref(handle), os.fsencode(output)) != 0:
        return
    try:
        ((data, _),) = built["umami"]["crossing"]
        point = _points(1)[0]
        events = [
            (point, channel, flavour, helicity_export(data, entry["runtime"])[0])
            for channel, description in enumerate(data["channels"])
            for flavour, entry in enumerate(description["processes"])
            if "momentum_signs" in entry["runtime"]
        ]
        assert events
        with pytest.raises(AssertionError):
            _ratio_call(library, handle, events, _densities(), switch=False)
    finally:
        assert library.umami_free(handle) == 0


@pytest.mark.parametrize("kind", ["union", "crossing"])
def test_switch_shares_cover_every_flavour_once(built, kind):
    """The union catalogue lists the same orderings (flavour, beam order, final
    order) and weights as the per-flavour one, and fewer native calls."""

    from pyamplicol.artifacts.umami_reweight import switch_catalogue

    catalogues = {}
    for name in (kind, "flavour"):
        ((data, _),) = built["umami"][name]
        groups = json.loads(
            (built["artifacts"][name] / "process-groups.json").read_text()
        )
        catalogues["union" if name == kind else name] = switch_catalogue(
            data, groups, SWITCH_LINES
        )

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


@pytest.mark.parametrize("kind", ["union", "crossing"])
def test_relabelled_switch_calls_equal_their_evaluations(built, kind):
    """With the union's pattern table, the switch evaluates every template leg
    type once: an evaluation at legs l is its assignment relabelled by
    pi[r] = c^-1[l[r]] at the call's legs c. Each must equal the assignment at
    its own legs (the phase-5 call), with the same orderings. Every relabelling
    keeps template sides and particles (the core's symmetry check), so a call's
    requests share its crossing."""

    from pyamplicol.artifacts.umami_reweight import switch_catalogue

    ((data, _),) = built["umami"][kind]
    artifact = built["artifacts"][kind]
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

    def gather(point, evaluation):
        signs = evaluation.get("signs") or [1] * len(evaluation["legs"])
        return tuple(
            tuple(s * v for v in point[slot])
            for s, slot in zip(signs, evaluation["legs"], strict=True)
        )

    for call in relabelled["calls"]:
        if call["assignments"] is None:
            continue
        process = data["runtime_processes"][call["process"]]
        pdgs = process["pdgs"]
        for permutation in call["permutations"] or ():
            # Python copy of the core's validate_symmetry
            assert sorted(permutation) == list(range(len(pdgs)))
            assert all(
                (r < 2) == (image < 2) and pdgs[r] == pdgs[image]
                for r, image in enumerate(permutation)
            )
        first = relabelled["evaluations"][call["evaluations"][0]]
        assert all(
            relabelled["evaluations"][e].get("signs") == first.get("signs")
            for e in call["evaluations"]
        )
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
        gathered = [gather(p, first) for p in points]
        values = runtime.evaluate_assignments(gathered, [ids] * len(points))
        for evaluation, entry in zip(call["evaluations"], call["entries"], strict=True):
            own = relabelled["evaluations"][evaluation]
            at_legs = [gather(p, own) for p in points]
            expected = runtime.evaluate_assignments(
                at_legs, [[own["assignment"]]] * len(points)
            )
            assert [row[entry] for row in values] == pytest.approx(
                [row[0] for row in expected], rel=1e-12
            )


# t t~ j: the switch exchanges the massless partons only, the tops keep their slots
TT_EXPRESSIONS = (
    "g g > t t~ g",
    "u u~ > t t~ g",
    "u~ u > t t~ g",
    "d d~ > t t~ g",
    "d~ d > t t~ g",
    "g u > t t~ u",
    "u g > t t~ u",
    "g d > t t~ d",
    "d g > t t~ d",
    "g u~ > t t~ u~",
    "u~ g > t t~ u~",
)
TT_SWITCH_LINES = 2
# t t~ with up to two light-quark lines: the targets have two light lines
TT_LINES_EXPRESSIONS = (
    "g g > t t~ g g",
    "u u~ > t t~ g g",
    "u~ u > t t~ g g",
    "d d~ > t t~ g g",
    "d~ d > t t~ g g",
    *(
        f"{a} {a}~ > t t~ {b} {b}~" if beams else f"{a}~ {a} > t t~ {b} {b}~"
        for a in ("u", "d", "s")
        for b in ("u", "d", "s")
        for beams in (True, False)
    ),
)


def _massive_providers(tmp_path_factory, name, expressions, switch_lines):
    config_executable = _config_executable()
    if shutil.which("make") is None or shutil.which(os.environ.get("CC", "cc")) is None:
        _unavailable("UMAMI integration requires make and a C compiler")
    root = tmp_path_factory.mktemp(name)
    artifacts = {kind: root / kind for kind in ("union", "flavour", "full")}
    names = tuple(f"t{index}" for index in range(len(expressions)))
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        for kind, path in artifacts.items():
            Generator(
                _config("full" if kind == "full" else "lc", kind == "union")
            ).generate(
                ProcessSet.from_expressions(expressions, names=names),
                path,
                model=ModelSource.from_path(prepared_model),
            )
    result = {"artifacts": artifacts, "umami": {}, "reweight": {}}
    for kind in ("union", "flavour"):
        build = root / f"build-{kind}"
        _make(artifacts[kind] / "API/umami", config_executable, f"BUILD_DIR={build}")
        result["umami"][kind] = [
            (data, _library(build / data["provider"]["library"]))
            for data in load_umami_metadata(artifacts[kind])
        ]
        output = root / f"reweight-{kind}"
        (identifier,) = write_reweight_provider(
            artifacts[kind], artifacts["full"], output, switch_lines=switch_lines
        )
        _make(output, config_executable)
        result["reweight"][kind] = (
            output,
            json.loads((output / f"flavor_switch_{identifier}.json").read_text()),
            _library(output / f"build/libreweight_{identifier}.so"),
        )
    return result


@pytest.fixture(scope="module")
def massive(tmp_path_factory):
    return _massive_providers(
        tmp_path_factory, "umami-unions-tt", TT_EXPRESSIONS, TT_SWITCH_LINES
    )


@pytest.fixture(scope="module")
def massive_lines(tmp_path_factory):
    return _massive_providers(
        tmp_path_factory, "umami-unions-ttqq", TT_LINES_EXPRESSIONS, 3
    )


def _massive_points(count: int):
    """On-shell g g > t t~ g points (the physical leg order of every flavour)."""

    from pyamplicol.generation.phase_space import massive_rambo_final_state

    mt = 173.0
    final = massive_rambo_final_state(3, sqrt_s=1000.0, masses=(mt, mt, 0.0), seed=7)
    base = [(500.0, 0.0, 0.0, 500.0), (500.0, 0.0, 0.0, -500.0)]
    base += [tuple(map(float, p)) for p in final]
    points = []
    for index in range(count):
        angle = 0.7 + 1.3 * index
        cos, sin = math.cos(angle), math.sin(angle)
        points.append(
            tuple((e, cos * x - sin * y, sin * x + cos * y, z) for e, x, y, z in base)
        )
    return points


def test_switch_keeps_massive_quarks_in_their_slots(massive):
    from pyamplicol.artifacts.umami_reweight import switch_catalogue

    for kind in ("union", "flavour"):
        ((data, _),) = massive["umami"][kind]
        groups = json.loads(
            (massive["artifacts"][kind] / "process-groups.json").read_text()
        )
        catalogue = switch_catalogue(data, groups, TT_SWITCH_LINES)
        massive_slots = [m > 0 for m in data["provider"]["masses"]]
        assert massive_slots == [False, False, True, True, False]
        assert catalogue["target_pdgs"]
        for pdgs in catalogue["target_pdgs"]:
            assert [abs(p) == 6 for p in pdgs] == massive_slots, pdgs
        for evaluation in catalogue["evaluations"]:
            pdgs = data["runtime_processes"][evaluation["process"]]["pdgs"]
            for leg, slot in enumerate(evaluation["legs"]):
                assert (abs(pdgs[leg]) == 6) == massive_slots[slot], evaluation
        # the sidecar lists the same targets
        assert massive["reweight"][kind][1]["targets"] == catalogue["target_pdgs"]


def test_massive_switch_ratios_equal_with_and_without_unions(massive):
    unions, flavours = massive["umami"]["union"], massive["umami"]["flavour"]
    union_dir, union_sidecar, union_library = massive["reweight"]["union"]
    flavour_dir, _, flavour_library = massive["reweight"]["flavour"]
    union_handle = _handle(union_library, union_dir)
    flavour_handle = _handle(flavour_library, flavour_dir)
    lc_library = unions[0][1]
    lc_handle = _handle(lc_library, massive["artifacts"]["union"])
    try:
        densities = _densities()
        for point in _massive_points(2):
            ours = _flavour_events(unions, 0, point)
            theirs = dict(_flavour_events(flavours, 0, point, every=True))
            events = [event for _, event in ours]
            other = [theirs[key] for key, _ in ours]
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
                a = _ratio_call(
                    union_library, union_handle, events, densities, switch=switch
                )
                b = _ratio_call(
                    flavour_library, flavour_handle, other, densities, switch=switch
                )
                assert (a[1] == -1).all() and (b[1] == -1).all()
                assert a[0][kept] == pytest.approx(b[0][kept], rel=1e-12, abs=0.0)
                assert (a[0] > 0).any()
        # switched events, t t~ and t~ t in the event's slots: the target's full
        # colour over its LC, with the event's t momentum a t in every target
        point = _massive_points(1)[0]
        data = unions[0][0]
        by_order = {}
        for _, event in _flavour_events(unions, 0, point):
            entry = data["channels"][event[1]]["processes"][event[2]]
            row = data["pdg_ids"][entry["matrix_elements"][0]["pdg_ids"]]
            if tuple(row[2:4]) in by_order:
                continue
            base, stay = (
                _ratio_call(union_library, union_handle, [event], densities, switch=s)[
                    0
                ][0]
                for s in (False, True)
            )
            # the first event of each order with a ratio of its own
            if base > 0.0:
                by_order[tuple(row[2:4])] = (event, row, stay / base)
        assert set(by_order) == {(6, -6), (-6, 6)}
        lc_runtimes, full_runtimes = {}, {}
        names = {**NAMES_OF, 6: "t"}
        drawn = set()
        for event, row, enhancement in by_order.values():
            assert enhancement > 1.0
            count = 400
            grid = (np.arange(count) + 0.5) / count
            random_flavor = list(1 / enhancement + grid * (1 - 1 / enhancement))
            random_helicity = list(
                np.modf(np.arange(count) * (math.sqrt(5) - 1) / 2)[0]
            )
            ratio, target, helicity, _ = _ratio_call(
                union_library,
                union_handle,
                [event] * count,
                densities,
                switch=True,
                random_flavor=random_flavor,
                random_helicity=random_helicity,
            )
            assert (target >= 0).all()
            for value, k, label in zip(ratio, target, helicity, strict=True):
                pdgs = tuple(union_sidecar["targets"][k])
                drawn.add(pdgs)
                assert pdgs[2:4] == tuple(row[2:4]), (row, pdgs)
                if pdgs not in lc_runtimes:
                    text = [names[abs(p)] + ("~" if p < 0 else "") for p in pdgs]
                    process = " ".join(text[:2]) + " > " + " ".join(text[2:])
                    lc_runtimes[pdgs] = Runtime.load(
                        massive["artifacts"]["flavour"], process=process
                    )
                    full_runtimes[pdgs] = Runtime.load(
                        massive["artifacts"]["full"], process=process
                    )
                vector = data["helicities"][label]
                identifier = "h:" + ",".join(f"{v:+d}" for v in vector)
                # the target's legs are the event slots
                resolved = lc_runtimes[pdgs].evaluate_resolved((event[0],))
                index = list(lc_runtimes[pdgs].physics.helicity_ids).index(identifier)
                lc_value = float(np.array(resolved.values)[0].real[index].sum())
                fc = full_runtimes[pdgs].evaluate((event[0],), helicities=[identifier])
                fc = complex(fc[0]).real
                assert value == pytest.approx(enhancement * fc / lc_value, rel=1e-12)
        assert len(drawn) > 2
    finally:
        assert union_library.umami_free(union_handle) == 0
        assert flavour_library.umami_free(flavour_handle) == 0
        assert lc_library.umami_free(lc_handle) == 0


def test_massive_switched_draws_equal_with_and_without_unions(massive_lines):
    """t t~ events switched to targets with two light-quark lines and the t line."""
    from pyamplicol.generation.phase_space import massive_rambo_final_state

    mt = 173.0
    final = massive_rambo_final_state(4, sqrt_s=1000.0, masses=(mt, mt, 0.0, 0.0), seed=7)
    point = ((500.0, 0.0, 0.0, 500.0), (500.0, 0.0, 0.0, -500.0))
    point += tuple(tuple(map(float, p)) for p in final)
    _assert_switched_draws_equal(
        massive_lines["reweight"]["union"],
        massive_lines["reweight"]["flavour"],
        massive_lines["umami"],
        point,
    )
