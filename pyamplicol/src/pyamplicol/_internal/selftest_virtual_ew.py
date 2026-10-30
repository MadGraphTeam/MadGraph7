# SPDX-License-Identifier: 0BSD
"""The installed self-test's electroweak virtual from a prepared artifact.

The wheel ships a portable v2 virtual artifact (``assets/selftest_virtual/
ew``): e- e+ > e- e+ with QED corrections on the shipped
loop_qcd_qed_sm (alpha(MZ) scheme, default restriction, every width zero;
MG5_aMC: ``e- e+ > e- e+ QCD=0 QED=2 [virt=QED]``), generated through the
public route (``generate_ew_self_test_artifact``: ``model.source =
"loop_qcd_qed_sm"``, its shipped ``[loop.model]`` profile and source repair,
native point inputs). ``pyamplicol self-test`` loads it, with no generation
and no regeneration at load, and evaluates it in binary64 at MadLoop's three
points; Born, finite part and both poles must agree with MadLoop 3.7.0's
accurate-mode values (``expected.json``, from
``tests/fixtures/loop/required/emep_emep_ew.json``) to ``RELATIVE_TOLERANCE``.
"""

from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path
from typing import Any

PROCESS = "e- e+ > e- e+"
NAME = "selftest_virtual_ew"
MODEL = "loop_qcd_qed_sm"
CORRECTIONS = ("QED",)
FIXED_PARAMETERS = {
    "MB": "0",
    "ymb": "0",
    "MTA": "0",
    "ymtau": "0",
    "WT": "0",
    "WW": "0",
    "WZ": "0",
    "WH": "0",
}
# Binary64 against MadLoop's accurate mode (its own accuracy <= 1.4e-15);
# the installation check, not the physics validation (tests and campaigns).
RELATIVE_TOLERANCE = 1.0e-9


def fixture_root() -> Path:
    """The installed self-test directory (``artifact`` and ``expected.json``)."""

    return Path(
        str(resources.files("pyamplicol").joinpath("assets", "selftest_virtual", "ew"))
    )


def card(output: Path, cache_dir: Path | None = None) -> dict[str, object]:
    """The public-route generation card of the self-test artifact."""

    model: dict[str, object] = {"source": MODEL}
    if cache_dir is not None:
        model["cache_dir"] = str(cache_dir)
    return {
        "action": "generate",
        "model": model,
        "process": {
            "entries": [{"expression": PROCESS, "name": NAME}],
            "loop_coupling_orders": list(CORRECTIONS),
            "coupling_order_policy": "minimal",
        },
        "loop": {
            "r2_strategy": "model",
            "preparation": {"fixed_parameters": dict(FIXED_PARAMETERS)},
        },
        "generation": {
            "output": str(output),
            "emit_api_bundle": False,
            "validation": {"post_build_validation": False},
        },
        "output": {"progress": "off"},
    }


def generate_ew_self_test_artifact(output: Path, cache_dir: Path | None = None):
    """Generate the self-test artifact into ``output`` (the shipped copy is
    this output; regenerate it when the artifact format changes)."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    output = Path(output)
    config = resolve_config(card(output, cache_dir), base_dir=output.parent)
    processes = ProcessSet((ProcessRequest.parse(PROCESS, name=NAME),))
    return Generator(config).generate(processes, output)


def evaluate(artifact: Path, expected: dict[str, Any]) -> list[tuple[float, ...]]:
    """(Born, finite, single pole, double pole) of the artifact at the
    expectation's points, each summed over its squared-order sectors."""

    from pyamplicol import Runtime

    runtime = Runtime.load(artifact, process=NAME, mute_warnings=True)
    points = [[list(map(float, p)) for p in point] for point in expected["points"]]
    result = runtime.evaluate(points, mu_r2=[float(x) for x in expected["mu_r2"]])
    values = []
    for born, laurent in zip(result.born_values, result.values, strict=True):
        totals = [sum((complex(value) for value in born), 0j)]
        totals += [
            sum((complex(getattr(sector, name)) for sector in laurent), 0j)
            for name in ("finite", "single_pole", "double_pole")
        ]
        for value in totals:
            if abs(value.imag) > RELATIVE_TOLERANCE * abs(value.real):
                raise ValueError(f"the self-test virtual is not real: {value!r}")
        values.append(tuple(value.real for value in totals))
    return values


def run_prepared_virtual_self_test(root: Path | None = None) -> tuple[str, float]:
    """Evaluate the shipped prepared EW virtual against MadLoop; return a
    summary and the largest relative deviation. Raises on any failure."""

    root = fixture_root() if root is None else Path(root)
    expected = json.loads((root / "expected.json").read_text(encoding="utf-8"))
    values = evaluate(root / "artifact", expected)
    worst = 0.0
    for index, (mine, reference) in enumerate(
        zip(values, expected["madloop"], strict=True)
    ):
        for name, value, target in zip(
            ("Born", "finite", "single pole", "double pole"),
            mine,
            reference,
            strict=True,
        ):
            deviation = abs(value - target) / abs(target)
            if not math.isfinite(deviation) or deviation > RELATIVE_TOLERANCE:
                raise ValueError(
                    f"point {index} {name} is {value!r}, MadLoop {target!r} "
                    f"(relative {deviation:.2e} > {RELATIVE_TOLERANCE:.0e})"
                )
            worst = max(worst, deviation)
    return (
        f"{PROCESS} [virt=QED] on the shipped {MODEL} from its prepared artifact: "
        f"Born and Laurent coefficients = MadLoop to {worst:.1e} at "
        f"{len(values)} points",
        worst,
    )


__all__ = [
    "RELATIVE_TOLERANCE",
    "card",
    "evaluate",
    "fixture_root",
    "generate_ew_self_test_artifact",
    "run_prepared_virtual_self_test",
]
