# SPDX-License-Identifier: 0BSD
"""Native checks of the flavour switch of the UMAMI ratio providers.

Events of the flavours with fewer than two quark lines are switched to those
with two: the provider sums the leading colour of every flavour at the event
(both beam orientations, every final-state ordering) with the parton densities
passed in. Checked against the Python runtimes, with a default and a
selector-only full-colour target.
"""

from __future__ import annotations

import ctypes as ct
import itertools
import json
import math
import os
import re
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from pyamplicol import Generator, ModelSource, ProcessSet, Runtime
from pyamplicol.artifacts.umami_c import helicity_export
from pyamplicol.artifacts.umami_reweight import quark_lines, write_reweight_provider
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
    RunConfig,
)
from tests.integration.test_umami_api import (
    _config_executable,
    _library,
    _runtime_point,
    _unavailable,
)

EXPRESSIONS = ("g g > g g", "d d~ > g g", "g g > d d~", "d d~ > u u~", "u d > u d")
NAMES = ("gg", "dd_gg", "gg_dd", "dd_uu", "ud_ud")
NAMES_OF = {21: "g", 1: "d", 2: "u"}


def _keys(prefix: str) -> dict[str, int]:
    header = (files("pyamplicol.assets.api_templates") / "umami/umami.h").read_text()
    body = header.split(f"}} {prefix};")[0].rsplit("typedef enum {", 1)[1]
    names = re.findall(r"^\s*(UMAMI_\w+),", body, re.MULTILINE)
    return {name: index for index, name in enumerate(names)}


IN, OUT = _keys("UmamiInputKey"), _keys("UmamiOutputKey")


def _generate(artifact: Path, accuracy: str, *, selector_only: bool = False) -> None:
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy=accuracy),
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
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        Generator(config).generate(
            ProcessSet.from_expressions(EXPRESSIONS, names=NAMES),
            artifact,
            model=ModelSource.from_path(prepared_model),
        )


@pytest.fixture(scope="module", params=(False, True), ids=("default", "selector-only"))
def switch(request, tmp_path_factory):
    config_executable = _config_executable()
    if shutil.which("make") is None or shutil.which(os.environ.get("CC", "cc")) is None:
        _unavailable("UMAMI integration requires make and a C compiler")
    root = tmp_path_factory.mktemp("umami-switch")
    lc, full, output = root / "lc", root / "full", root / "reweight"
    _generate(lc, "lc")
    _generate(full, "full", selector_only=request.param)
    plans = [
        json.loads((full / "processes" / p["id"] / "execution.json").read_text())
        for p in json.loads((full / "artifact.json").read_text())["processes"]
    ]
    assert any(plan["plan"] is None for plan in plans) == request.param
    assert write_reweight_provider(lc, full, output, switch_lines=2) == ["p0"]
    completed = subprocess.run(
        ["make", "-C", str(output), f"RUSTICOL_CONFIG={config_executable}"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    data = json.loads((lc / "API/umami/metadata.json").read_text())
    sidecar = json.loads((output / "flavor_switch_p0.json").read_text())
    library = _library(output / "build/libreweight_p0.so")
    handle = ct.c_void_p()
    assert library.umami_initialize(ct.byref(handle), os.fsencode(output)) == 0
    yield lc, full, data, sidecar, library, handle
    assert library.umami_free(handle) == 0


def _process(pdgs):
    names = [NAMES_OF[abs(p)] + ("~" if p < 0 else "") for p in pdgs]
    return " ".join(names[:2]) + " > " + " ".join(names[2:])


def _point(lc):
    class _Artifact:
        pass

    provider = _Artifact()
    provider.artifact = lc
    return _runtime_point(provider, "gg")


def _densities():
    rng = np.random.default_rng(3)
    densities = rng.uniform(0.2, 2.0, size=(2, 13))
    densities[:, [0, 12]] = 0.0
    return densities


def _summed(lc, data, point, densities):
    """Generated and dropped leading colour of all flavours at the point."""
    flavours = set()
    for process in data["runtime_processes"]:
        pdgs = process["pdgs"]
        for initial in {tuple(pdgs[:2]), tuple(pdgs[1::-1])}:
            flavours.add((initial, tuple(sorted(pdgs[2:]))))
    totals = {}
    for initial, final in flavours:
        value = 0.0
        for order in set(itertools.permutations(final)):
            runtime = Runtime.load(lc, process=_process([*initial, *order]))
            value += complex(runtime.evaluate((point,))[0]).real
        weight = math.prod(math.factorial(final.count(p)) for p in set(final))
        pdf = densities[0, 6 if initial[0] == 21 else initial[0] + 6]
        pdf *= densities[1, 6 if initial[1] == 21 else initial[1] + 6]
        lines = quark_lines([*initial, *final])
        totals[lines >= 2] = totals.get(lines >= 2, 0.0) + pdf * weight * value / 2
    return totals[False], totals[True]


def _call(library, handle, point, densities, random_flavor, random_helicity, channel,
          flavour, helicity, switch_on=True, random_color=None):
    count = len(random_flavor)
    n = len(point)
    assert library.umami_set_parameter(handle, b"flavor_switch", float(switch_on), 0.0) == 0
    momenta = (ct.c_double * (4 * n * count))(
        *(point[leg][k] for k in range(4) for leg in range(n) for _ in range(count))
    )
    pdfs = (ct.c_double * (26 * count))(
        *(densities[leg, column] for column in range(13) for leg in range(2)
          for _ in range(count))
    )
    inputs = {
        IN["UMAMI_IN_MOMENTA"]: momenta,
        IN["UMAMI_IN_FLAVOR_INDEX"]: (ct.c_int * count)(*[flavour] * count),
        IN["UMAMI_IN_HELICITY_INDEX"]: (ct.c_int * count)(*[helicity] * count),
        IN["UMAMI_IN_CHANNEL_INDEX"]: (ct.c_uint * count)(*[channel] * count),
        IN["UMAMI_IN_PARTON_DENSITIES"]: pdfs,
        IN["UMAMI_IN_RANDOM_FLAVOR"]: (ct.c_double * count)(*random_flavor),
        IN["UMAMI_IN_RANDOM_HELICITY"]: (ct.c_double * count)(*random_helicity),
        IN["UMAMI_IN_RANDOM_COLOR"]: (ct.c_double * count)(
            *([0.5] * count if random_color is None else random_color)
        ),
    }
    names = [
        "UMAMI_OUT_MATRIX_ELEMENT_RATIO", "UMAMI_OUT_FLAVOR_INDEX",
        "UMAMI_OUT_HELICITY_INDEX", "UMAMI_OUT_COLOR_INDEX",
    ]
    outputs = [(ct.c_double * count)()] + [(ct.c_int * count)() for _ in range(3)]
    status = library.umami_matrix_element(
        handle, count, count, 0, len(inputs),
        (ct.c_int * len(inputs))(*inputs),
        (ct.c_void_p * len(inputs))(*(ct.cast(v, ct.c_void_p) for v in inputs.values())),
        len(outputs), (ct.c_int * len(outputs))(*(OUT[name] for name in names)),
        (ct.c_void_p * len(outputs))(*(ct.cast(v, ct.c_void_p) for v in outputs)),
    )
    assert status == 0
    return [np.array(list(o)) for o in outputs]


def _entry(data):
    for channel, description in enumerate(data["channels"]):
        for flavour, entry in enumerate(description["processes"]):
            process = data["runtime_processes"][entry["runtime"]["process_index"]]
            if process["id"] == "gg":
                return channel, flavour, entry
    raise AssertionError("no g g > g g entry")


def test_stay_is_the_ratio_times_the_summed_flavours(switch):
    lc, _, data, _, library, handle = switch
    point, densities = _point(lc), _densities()
    channel, flavour, entry = _entry(data)
    helicity = helicity_export(data, entry["runtime"])[3]
    generated, dropped = _summed(lc, data, point, densities)
    assert dropped > 0.0
    off = _call(library, handle, point, densities, [0.0], [0.5], channel, flavour,
                helicity, switch_on=False)
    on = _call(library, handle, point, densities, [0.0], [0.5], channel, flavour, helicity)
    assert on[1][0] == -1 and on[2][0] == -1 and on[3][0] == -1
    assert on[0][0] == pytest.approx(off[0][0] * (1 + dropped / generated), rel=1e-9)


def test_switched_targets_and_helicities_follow_the_leading_colour(switch):
    lc, full, data, sidecar, library, handle = switch
    point, densities = _point(lc), _densities()
    channel, flavour, entry = _entry(data)
    generated, dropped = _summed(lc, data, point, densities)
    enhancement = 1 + dropped / generated
    count = 4000
    # random numbers on a grid over the switching range, helicities on a grid
    grid = (np.arange(count) + 0.5) / count
    random_flavor = 1 / enhancement + grid * (1 - 1 / enhancement)
    # equidistributed within the block of events of every target
    random_helicity = np.modf(np.arange(count) * (math.sqrt(5) - 1) / 2)[0]
    ratio, target, helicity, color = _call(
        library, handle, point, densities, random_flavor, random_helicity, channel,
        flavour, helicity_export(data, entry["runtime"])[0],
    )
    assert (target >= 0).all()
    shares = {}
    for k, pdgs in enumerate(sidecar["targets"]):
        runtime = Runtime.load(lc, process=_process(pdgs))
        weight = math.prod(math.factorial(pdgs[2:].count(p)) for p in set(pdgs[2:])) / 2
        pdf = densities[0, pdgs[0] + 6] * densities[1, pdgs[1] + 6]
        shares[k] = pdf * weight * complex(runtime.evaluate((point,))[0]).real
    total = sum(shares.values())
    assert total == pytest.approx(dropped, rel=1e-9)
    for k, share in shares.items():
        assert abs((target == k).sum() - count * share / total) <= 2
    color_offset = sidecar["color_flow_offset"]
    for k in np.unique(target):
        pdgs = sidecar["targets"][k]
        lc_runtime = Runtime.load(lc, process=_process(pdgs))
        fc_runtime = Runtime.load(full, process=_process(pdgs))
        ids = list(lc_runtime.physics.helicity_ids)
        values = np.array(lc_runtime.evaluate_resolved((point,)).values)[0].sum(-1)
        chosen = target == k
        for e in np.unique(helicity[chosen]):
            vector = data["helicities"][e]
            identifier = "h:" + ",".join(f"{v:+d}" if v else "0" for v in vector)
            index = ids.index(identifier)
            # frequencies follow the leading colour of the helicities
            expected = chosen.sum() * values[index] / values.sum()
            assert abs((helicity[chosen] == e).sum() - expected) <= 3 + 0.03 * expected
            fc = complex(fc_runtime.evaluate((point,), helicities=[identifier])[0]).real
            assert ratio[chosen & (helicity == e)] == pytest.approx(
                enhancement * fc / values[index], rel=1e-9
            )
        tags = [
            (sidecar["color_flows"][c - color_offset] if c >= color_offset
             else data["color_flows"][c])
            for c in np.unique(color[chosen])
        ]
        for flow in tags:
            for pdg, (colour, anticolour) in zip(pdgs, flow):
                if pdg > 0 and pdg != 21:
                    assert colour and not anticolour
                elif pdg < 0:
                    assert anticolour and not colour


def test_charge_conjugate_totals_keep_the_switch(switch, tmp_path, monkeypatch):
    """Orderings reading the total of their charge conjugate (pure QCD) give the
    same ratios and draws; the targets keep evaluations no call writes."""
    import pyamplicol.artifacts.umami_reweight as module
    from pyamplicol.artifacts.umami_reweight_data import decode_reweight_data

    lc, full, data, _, library, handle = switch
    old = Path(library._name).parents[1]
    # the builtin model has no alpha_s proof; its 2 -> 2 flavours are pure QCD
    monkeypatch.setattr(
        module, "_pure_qcd", lambda p: all(module._is_qcd_parton(x) for x in p["pdgs"])
    )
    monkeypatch.setattr(module, "SWITCH_C_DEDUP", True)
    output = tmp_path / "dedup"
    assert write_reweight_provider(lc, full, output, switch_lines=2) == ["p0"]
    completed = subprocess.run(
        ["make", "-C", str(output), f"RUSTICOL_CONFIG={_config_executable()}"],
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    before = decode_reweight_data((old / "reweight_p0.rwd").read_bytes()).switch
    after = decode_reweight_data((output / "reweight_p0.rwd").read_bytes()).switch
    assert len(after.calls) < len(before.calls)
    assert after.targets == before.targets and after.evaluations == before.evaluations
    written = {e for call in after.calls for e in call.evaluations}
    assert set(after.targets.evaluation) - written
    dedup = _library(output / "build/libreweight_p0.so")
    dedup_handle = ct.c_void_p()
    assert dedup.umami_initialize(ct.byref(dedup_handle), os.fsencode(output)) == 0
    point, densities = _point(lc), _densities()
    channel, flavour, entry = _entry(data)
    count = 400
    random_flavor = (np.arange(count) + 0.5) / count * 1.2
    random_helicity = np.modf(np.arange(count) * (math.sqrt(5) - 1) / 2)[0]
    results = [
        _call(lib, h, point, densities, random_flavor, random_helicity, channel, flavour,
              helicity_export(data, entry["runtime"])[0], random_color=random_helicity)
        for lib, h in ((library, handle), (dedup, dedup_handle))
    ]
    assert (results[0][1] >= 0).any()
    np.testing.assert_allclose(results[1][0], results[0][0], rtol=1e-12)
    for k in (1, 2, 3):
        assert (results[1][k] == results[0][k]).all()
    assert dedup.umami_free(dedup_handle) == 0


def _flow_strings(pdgs, flow):
    """Open colour strings of LHE tags (fundamental first), as in a runtime flow id."""
    # all outgoing: incoming legs swap colour and anticolour
    pairs = [(a, c) if leg < 2 else (c, a) for leg, (c, a) in enumerate(flow)]
    strings = []
    for leg, (colour, anticolour) in enumerate(pairs):
        if not colour or anticolour:
            continue
        string, tag = [leg + 1], colour
        while tag:
            following = next(k for k, (_, a) in enumerate(pairs) if a == tag)
            string.append(following + 1)
            tag = pairs[following][0]
        strings.append(tuple(string))
    return tuple(sorted(strings))


def _id_strings(pdgs, color_id):
    fundamental = {
        leg + 1 for leg, pdg in enumerate(pdgs) if pdg != 21 and (pdg > 0) == (leg >= 2)
    }
    strings = []
    for label in map(int, color_id.removeprefix("flow:").split(",")):
        if label in fundamental or not strings:
            strings.append([])
        strings[-1].append(label)
    return tuple(sorted(map(tuple, strings)))


def test_switched_colours_follow_the_leading_colour_flows(switch):
    """For a fixed target and helicity, a grid of random colours draws each colour
    flow in proportion to its leading-colour value at that helicity."""
    lc, _, data, sidecar, library, handle = switch
    point, densities = _point(lc), _densities()
    channel, flavour, entry = _entry(data)
    generated, dropped = _summed(lc, data, point, densities)
    enhancement = 1 + dropped / generated
    flows = data["color_flows"] + sidecar["color_flows"]
    count, cases, multi = 2000, set(), 0
    grid = list((np.arange(count) + 0.5) / count)
    for u in np.linspace(0.02, 0.98, 25):
        for random_helicity in (0.1, 0.45, 0.8):
            fixed = 1 / enhancement + u * (1 - 1 / enhancement)
            ratio, target, helicity, color = _call(
                library, handle, point, densities, [fixed] * count,
                [random_helicity] * count, channel, flavour,
                helicity_export(data, entry["runtime"])[0], random_color=grid,
            )
            assert len(set(target)) == 1 and len(set(helicity)) == 1
            k, e = int(target[0]), int(helicity[0])
            if (k, e) in cases:
                continue
            cases.add((k, e))
            pdgs = sidecar["targets"][k]
            runtime = Runtime.load(lc, process=_process(pdgs))
            resolved = runtime.evaluate_resolved((point,))
            vector = data["helicities"][e]
            identifier = "h:" + ",".join(f"{v:+d}" if v else "0" for v in vector)
            values = np.array(resolved.values)[0].real[
                list(resolved.helicity_ids).index(identifier)
            ]
            expected = {
                _id_strings(pdgs, c): value
                for c, value in zip(resolved.color_ids, values, strict=True)
            }
            drawn = {}
            for c in color:
                key = _flow_strings(pdgs, flows[c])
                drawn[key] = drawn.get(key, 0) + 1
            assert set(drawn) <= {key for key, value in expected.items() if value > 0}
            total = sum(expected.values())
            for key, value in expected.items():
                assert abs(drawn.get(key, 0) - count * value / total) <= 2, (pdgs, key)
            multi += sum(value > 0 for value in expected.values()) > 1
    assert multi


def _rebuilt(source: Path, target: Path, data: bytes | None, header: bool) -> Path:
    """A copy of a provider with the data file ``data`` (None: removed); with
    ``header``, its header bound to that data and the library rebuilt."""

    from pyamplicol.artifacts.umami_reweight_data import (
        content_sha256,
        decode_reweight_data,
        reweight_data_header,
    )

    shutil.copytree(source, target)
    path = target / "reweight_p0.rwd"
    if data is None:
        path.unlink()
        return target
    if header:
        tables = decode_reweight_data(path.read_bytes())
        (target / "reweight_p0.h").write_text(
            reweight_data_header(tables, path.name, content_sha256(data))
        )
    path.write_bytes(data)
    if header:
        completed = subprocess.run(
            ["make", "-C", str(target), f"RUSTICOL_CONFIG={_config_executable()}"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
    return target


def _initialize(directory: Path) -> tuple[int, str]:
    """umami_initialize in a child process: its status and stderr (no crash)."""

    import sys

    code = (
        "import ctypes as ct, sys; lib = ct.CDLL(sys.argv[1]); h = ct.c_void_p(); "
        "s = lib.umami_initialize(ct.byref(h), sys.argv[2].encode()); "
        "s == 0 and lib.umami_free(h); print(s)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(directory / "build/libreweight_p0.so"),
         str(directory)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return int(completed.stdout), completed.stderr


def _field(data: bytes, tag: str, row: int, field: int, value: int) -> bytes:
    import struct

    from pyamplicol.artifacts.umami_reweight_data import (
        reseal_reweight_data,
        section_table,
    )

    offset, count, size = section_table(data)[tag]
    assert row < count, tag
    patched = bytearray(data)
    struct.pack_into("<i", patched, offset + row * size + 4 * field, value)
    return reseal_reweight_data(bytes(patched))


def _helicity_value(data: bytes) -> bytes:
    """The first value of flavour 0's reference helicity map out of range."""

    import struct

    from pyamplicol.artifacts.umami_reweight_data import (
        reseal_reweight_data,
        section_table,
    )

    table = section_table(data)
    ref = struct.unpack_from("<I", data, table["FLAV"][0] + 16)[0]
    patched = bytearray(data)
    struct.pack_into("<i", patched, table["I32P"][0] + 4 * ref, 1 << 20)
    return reseal_reweight_data(bytes(patched))


def _pool_end(data: bytes) -> int:
    from pyamplicol.artifacts.umami_reweight_data import section_table

    return section_table(data)["I32P"][1]


def _calls(data: bytes, change) -> bytes:
    """The data re-encoded with the switch calls ``change(calls, orderings)``."""

    import dataclasses

    from pyamplicol.artifacts.umami_reweight_data import (
        decode_reweight_data,
        encode_reweight_data,
    )

    tables = decode_reweight_data(data)
    sw = tables.switch
    calls = change(list(sw.calls), sw.orderings)
    return encode_reweight_data(
        dataclasses.replace(tables, switch=dataclasses.replace(sw, calls=calls))
    )


def _plain_pair(calls, orderings):
    from pyamplicol.artifacts.umami_reweight_data import SwitchCall

    return [*calls, SwitchCall((0, 1), None, None, None)]


def _uncovered(calls, orderings):
    return [c for c in calls if orderings.evaluation[0] not in c.evaluations]


# (corrupted data or None, header and library rebuilt for it, message)
CORRUPTIONS = {
    "flipped byte": (
        lambda d: d[:-5] + bytes([d[-5] ^ 0x40]) + d[-4:], False, "checksum"
    ),
    "truncated": (lambda d: d[:-8], False, "file size"),
    "newer format": (
        lambda d: d[:8] + (2).to_bytes(4, "little") + d[12:], False, "format version 2"
    ),
    "missing": (lambda d: None, False, "cannot read the reweight data"),
    "stale library": (lambda d: _field(d, "FLAV", 0, 13, 7), False, "digest"),
    "target evaluation": (lambda d: _field(d, "STGT", 0, 0, 1 << 20), True, "target"),
    "ordering target": (lambda d: _field(d, "SORD", 0, 1, 1 << 20), True, "ordering"),
    "colour length": (
        lambda d: _field(d, "STGT", 0, 2, _pool_end(d)), True, "target"
    ),
    "assignment without key": (
        lambda d: _field(d, "FLAV", 0, 8, 0), True, "flavour"
    ),
    "call without evaluations": (lambda d: _field(d, "SCAL", 0, 0, 0), True, "call"),
    "helicity map value": (_helicity_value, True, "flavour"),
    "plain call of two evaluations": (
        lambda d: _calls(d, _plain_pair), True, "invalid switch call"
    ),
    "ordering of an unwritten evaluation": (
        lambda d: _calls(d, _uncovered), True, "invalid switch ordering"
    ),
}


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_a_corrupted_data_file_fails_the_load(switch, tmp_path, name):
    """umami_initialize refuses damaged, mismatched or out-of-range data with a
    message; semantic damage behind a matching digest is caught by validation."""

    library = switch[4]
    output = Path(library._name).parents[1]
    patch, header, message = CORRUPTIONS[name]
    data = patch((output / "reweight_p0.rwd").read_bytes())
    copy = _rebuilt(output, tmp_path / "copy", data, header)
    status, stderr = _initialize(copy)
    assert status != 0
    assert "umami: " in stderr and message in stderr, stderr
