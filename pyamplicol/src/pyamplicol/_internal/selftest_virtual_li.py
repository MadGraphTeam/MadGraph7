# SPDX-License-Identifier: 0BSD
"""The installed self-test's loop-induced virtual from a prepared artifact.

The wheel ships a portable loop-induced artifact (``assets/selftest_virtual/
li``): g g > h at one loop in QCD on the shipped loop_sm (no_b_mass, every
width zero; MG5_aMC: ``g g > h [sqrvirt=QCD]``), |A_loop|^2 of a process
without a Born amplitude (m-lit), generated through the public route
(``generate_li_self_test_artifact``). ``pyamplicol self-test`` loads it,
with no generation and no regeneration at load, and evaluates it in binary64
on the default reduction backend at MadLoop's three points
(``expected.json``, from ``tests/fixtures/loop/required/li/li_gg_h.json``):
a loop-induced ``VirtualEvaluation`` (empty Born fields), whose finite part
agrees with MadLoop 3.7.0 to ``RELATIVE_TOLERANCE`` and whose computed poles
vanish to it (the finiteness cross-check, m-lit Q36). Binary64 evaluation
of a stored artifact never calls Symbolica: the check runs without
``SYMBOLICA_LICENSE`` (``tests/unit/test_loop_v2_induced_api.py``).
"""

from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path
from typing import Any

PROCESS = "g g > h"
NAME = "selftest_virtual_li"
MODEL = "loop_sm"
RESTRICTION = "no_b_mass"
CORRECTIONS = ("QCD",)
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
# Binary64 against MadLoop's accurate mode (its own accuracy <= 6e-13; the
# analytic value agrees to 6e-13); the installation check, not the physics
# validation (tests and campaigns).
RELATIVE_TOLERANCE = 1.0e-9


def fixture_root() -> Path:
    """The installed self-test directory (``artifact`` and ``expected.json``)."""

    return Path(
        str(resources.files("pyamplicol").joinpath("assets", "selftest_virtual", "li"))
    )


def card(output: Path, cache_dir: Path | None = None) -> dict[str, object]:
    """The public-route generation card of the self-test artifact."""

    model: dict[str, object] = {"source": MODEL, "restriction": RESTRICTION}
    if cache_dir is not None:
        model["cache_dir"] = str(cache_dir)
    return {
        "action": "generate",
        "model": model,
        "process": {
            "entries": [{"expression": PROCESS, "name": NAME}],
            "loop_coupling_orders": list(CORRECTIONS),
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


def generate_li_self_test_artifact(output: Path, cache_dir: Path | None = None) -> Any:
    """Generate the self-test artifact into ``output`` (the shipped copy is
    this output; regenerate it when the artifact format or the package
    version changes)."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    output = Path(output)
    config = resolve_config(card(output, cache_dir), base_dir=output.parent)
    processes = ProcessSet((ProcessRequest.parse(PROCESS, name=NAME),))
    return Generator(config).generate(processes, output)


def evaluate(artifact: Path, expected: dict[str, Any]) -> Any:
    """The artifact's ``VirtualEvaluation`` at the expectation's points and
    parameters (binary64, every pole, the default reduction backend)."""

    from pyamplicol import Runtime

    runtime = Runtime.load(
        artifact,
        process=NAME,
        model_parameters=expected["parameters"],
        mute_warnings=True,
    )
    return runtime.evaluate(expected["points"], mu_r2=expected["mu_r2"])


def run_prepared_loop_induced_self_test(
    root: Path | None = None,
) -> tuple[str, float]:
    """Evaluate the shipped loop-induced artifact against MadLoop; return a
    summary and the largest relative deviation. Raises on any failure."""

    root = fixture_root() if root is None else Path(root)
    expected = json.loads((root / "expected.json").read_text(encoding="utf-8"))
    result = evaluate(root / "artifact", expected)
    if result.contribution != "loop-induced" or result.born_orders:
        raise ValueError("the self-test result is not a loop-induced virtual")
    worst = 0.0
    for index, (born, laurent, target) in enumerate(
        zip(result.born_values, result.values, expected["madloop"], strict=True)
    ):
        if born:
            raise ValueError(f"point {index} has Born values {born!r}")
        finite = math.fsum(float(sector.finite) for sector in laurent)
        deviation = abs(finite - target) / abs(target)
        if not math.isfinite(deviation) or deviation > RELATIVE_TOLERANCE:
            raise ValueError(
                f"point {index} |A_loop|^2 is {finite!r}, MadLoop {target!r} "
                f"(relative {deviation:.2e} > {RELATIVE_TOLERANCE:.0e})"
            )
        for name in ("single_pole", "double_pole"):
            pole = math.fsum(float(getattr(sector, name)) for sector in laurent)
            if not abs(pole) <= RELATIVE_TOLERANCE * abs(finite):
                raise ValueError(
                    f"point {index} {name} {pole!r} does not vanish against "
                    f"the finite part {finite!r} (the finiteness cross-check)"
                )
        worst = max(worst, deviation)
    reduction = result.diagnostics[0].reduction
    backend = "" if reduction is None else f" on {reduction.effective}"
    return (
        f"{PROCESS} [sqrvirt=QCD] (loop-induced) on the shipped {MODEL} from its "
        f"prepared artifact{backend}: |A_loop|^2 = MadLoop to {worst:.1e} at "
        f"{len(result.values)} points, poles vanish",
        worst,
    )


__all__ = [
    "RELATIVE_TOLERANCE",
    "card",
    "evaluate",
    "fixture_root",
    "generate_li_self_test_artifact",
    "run_prepared_loop_induced_self_test",
]
