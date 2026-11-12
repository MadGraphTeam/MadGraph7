# SPDX-License-Identifier: 0BSD
"""The installed self-test's one-loop virtual on the shipped loop_sm.

``pyamplicol self-test`` generates u u~ > e- e+ with QCD corrections and the
Z vetoed (MG5_aMC: ``u u~ > e- e+ / z QCD=0 QED=2 [virt=QCD]``) on the
shipped MG5_aMC 3.7.0 loop_sm, restriction no_b_mass, through the public
route (``model.source = "loop_sm"``: the shipped ``[loop.model]`` profile,
full colour by default, v2), in a temporary directory, and evaluates it in
binary64 at MadLoop's three exact points at mu_R = 1000 GeV. Born, finite
part and both poles must agree with MadLoop 3.7.0's accurate-mode values
(``tests/fixtures/loop/drell-yan-control.json``, its reported accuracy at most
5.7e-12) to ``RELATIVE_TOLERANCE``. A generation or licence failure fails the
check with its reason (the Symbolica licence state is always reported).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory

PROCESS = "u u~ > e- e+"
RESTRICTION = "no_b_mass"
VETO = ("Z",)
MU_R2 = 1.0e6
POINTS = (
    ((500, 0, 0, 500), (500, 0, 0, -500), (500, 300, 0, 400), (500, -300, 0, -400)),
    ((650, 0, 0, 650), (650, 0, 0, -650), (650, 600, 0, 250), (650, -600, 0, -250)),
    ((500, 0, 0, 500), (500, 0, 0, -500), (500, 0, 400, -300), (500, 0, -400, 300)),
)
# MadLoop 3.7.0, loop_sm-no_b_mass (alpha_s 0.118, aEWM1 132.507, MT 173,
# MB 0), accurate mode: (Born, finite, single pole, double pole) per point.
MADLOOP = (
    (
        0.002185154556821645,
        0.00010229932249500759,
        -0.00016415128639310988,
        -0.00010943419092873994,
    ),
    (
        0.0015295135806876875,
        0.00012135035900169152,
        -7.47050101318556e-05,
        -7.659919555554748e-05,
    ),
    (
        0.0018120793885838043,
        8.483358450804345e-05,
        -0.00013612545700892074,
        -9.075030467261367e-05,
    ),
)
# Binary64 against MadLoop's accurate mode (its own accuracy <= 5.7e-12);
# the installation check, not the physics validation (tests and campaigns).
RELATIVE_TOLERANCE = 1.0e-9


def _card(root: Path) -> dict[str, object]:
    return {
        "action": "generate",
        "model": {
            "source": "loop_sm",
            "restriction": RESTRICTION,
            "cache_dir": str(root / "model-cache"),
        },
        "process": {
            "entries": [{"expression": PROCESS, "name": "selftest_virtual"}],
            "loop_coupling_orders": ["QCD"],
            "coupling_order_policy": "minimal",
        },
        "loop": {"r2_strategy": "model"},
        "generation": {
            "output": str(root / "artifact"),
            "emit_api_bundle": False,
            "particle_veto": list(VETO),
            "validation": {"post_build_validation": False},
        },
        "output": {"progress": "off"},
    }


def run_virtual_self_test() -> tuple[str, float]:
    """Generate and evaluate the self-test virtual; return a summary and the
    largest relative deviation from MadLoop. Raises on any failure."""

    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    started = time.perf_counter()
    with TemporaryDirectory(prefix="pyamplicol-selftest-virtual-") as directory:
        root = Path(directory)
        config = resolve_config(_card(root), base_dir=root)
        processes = ProcessSet(
            (ProcessRequest.parse(PROCESS, name="selftest_virtual"),)
        )
        Generator(config).generate(processes, root / "artifact")
        runtime = Runtime.load(root / "artifact", process="selftest_virtual")
        points = [[list(map(float, p)) for p in point] for point in POINTS]
        result = runtime.evaluate(points, mu_r2=[MU_R2] * len(points))
        accuracy = runtime.inspect()["color_accuracy"]
    worst = 0.0
    for index, (born, values, reference) in enumerate(
        zip(result.born_values, result.values, MADLOOP, strict=True)
    ):
        ((mine_born,),) = (born,)
        ((laurent,),) = (values,)
        mine = (mine_born, laurent.finite, laurent.single_pole, laurent.double_pole)
        for name, value, expected in zip(
            ("Born", "finite", "single pole", "double pole"),
            mine,
            reference,
            strict=True,
        ):
            actual = complex(value)
            if abs(actual.imag) > RELATIVE_TOLERANCE * abs(expected):
                raise ValueError(f"point {index} {name} is not real: {actual!r}")
            deviation = abs(actual.real - expected) / abs(expected)
            if not math.isfinite(deviation) or deviation > RELATIVE_TOLERANCE:
                raise ValueError(
                    f"point {index} {name} is {actual.real!r}, MadLoop "
                    f"{expected!r} (relative {deviation:.2e} > "
                    f"{RELATIVE_TOLERANCE:.0e})"
                )
            worst = max(worst, deviation)
    seconds = time.perf_counter() - started
    return (
        f"{PROCESS} / {' '.join(VETO)} [virt=QCD] on the shipped loop_sm-"
        f"{RESTRICTION} ({accuracy} colour): Born and Laurent coefficients = "
        f"MadLoop to {worst:.1e} at {len(POINTS)} points; generated in "
        f"{seconds:.0f} s",
        worst,
    )


__all__ = ["MADLOOP", "POINTS", "RELATIVE_TOLERANCE", "run_virtual_self_test"]
