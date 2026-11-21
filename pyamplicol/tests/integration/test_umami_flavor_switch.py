# SPDX-License-Identifier: 0BSD
"""Native checks of the flavour switch of the UMAMI ratio providers.

Events of the flavours with fewer than two quark lines are switched to those
with two: the provider sums the leading colour of every flavour at the event
(both beam orientations, every final-state ordering) with the parton densities
passed in. Checked against the Python runtimes.
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


def _generate(artifact: Path, accuracy: str) -> None:
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy=accuracy),
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
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        Generator(config).generate(
            ProcessSet.from_expressions(EXPRESSIONS, names=NAMES),
            artifact,
            model=ModelSource.from_path(prepared_model),
        )


@pytest.fixture(scope="module")
def switch(tmp_path_factory):
    config_executable = _config_executable()
    if shutil.which("make") is None or shutil.which(os.environ.get("CC", "cc")) is None:
        _unavailable("UMAMI integration requires make and a C compiler")
    root = tmp_path_factory.mktemp("umami-switch")
    lc, full, output = root / "lc", root / "full", root / "reweight"
    _generate(lc, "lc")
    _generate(full, "full")
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
          flavour, helicity, switch_on=True):
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
        IN["UMAMI_IN_RANDOM_COLOR"]: (ct.c_double * count)(*[0.5] * count),
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
    helicity = entry["runtime"]["helicity_export"][3]
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
        flavour, entry["runtime"]["helicity_export"][0],
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
