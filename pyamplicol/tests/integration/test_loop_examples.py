# SPDX-License-Identifier: 0BSD
"""Smoke tests of the four packaged one-loop examples (m-lit lit-examples B7):
each example's walkthrough (``examples/python/loop_walkthrough.py``) runs end
to end from a fresh ``pyamplicol examples copy`` workspace at a tiny point
count: ``pyamplicol check`` of the card, generation with the correlator
declaration, evaluation on both reduction backends, the per-stage profile in
the checked and the unchecked setting, correlated evaluation, helicity
selection, the partial weights' sum check and the native loop client
(compiled with the examples' Makefile, run with SYMBOLICA_LICENSE unset).

The test decides on the capability: with a native extension built without
TIRust, 'auto' serves a loop-induced example on NinjaRS (its reason
recorded) and an explicit 'tirust' arm prints its typed refusal.

Every example passes ``check`` (exit 0). Example 2 (u u~ > d d~ e+ e-, every
order, complex-mass scheme) is also compared with Recola 2.3.0 on the
walkthrough's own artifact (``tests/fixtures/loop/uux-ddxepem-recola-cms.json``):
the Born per order, both poles of every sector and the finite parts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_physics

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ("loop_uux_ttxgg", "loop_uux_ddxepem", "loop_gg_zgg", "loop_gg_epemmpmm")
LOOP_INDUCED = {"loop_gg_zgg", "loop_gg_epemmpmm"}
# Born-interfered examples with a leading-colour variant (helicity selection).
LC_VARIANTS = {"loop_uux_ttxgg": "loop_uux_ttxgg_lc_generate.toml"}
RECOLA = ROOT / "tests" / "fixtures" / "loop" / "uux-ddxepem-recola-cms.json"
STEPS = (
    "=== 1. pyamplicol check of the card",
    "=== 2. Generation",
    "=== 3. Evaluation on both reduction backends",
    "=== 4. Profile",
    "=== 5. Colour- and spin-correlated evaluation",
    "=== 6. Helicity selection",
    "=== 7. Partial weights per squared coupling order",
    "=== 8. The same calls through the native SDK",
    "=== Times",
)


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    source = str(ROOT / "src")
    path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not path else f"{source}{os.pathsep}{path}"
    return environment


def _loop_sdk() -> tuple[str, str]:
    """The loop SDK flags: the installed wheel's (``rusticol-config``) or, in
    a source checkout, the staged native build's loop SDK library."""

    config = shutil.which("rusticol-config")
    if config is not None:
        cflags = subprocess.run(
            [config, "--cflags"], check=True, capture_output=True, text=True
        ).stdout.strip()
        libs = subprocess.run(
            [config, "--libs"], check=True, capture_output=True, text=True
        ).stdout.strip()
        return cflags, libs
    info = ROOT / ".artifacts" / "source-runtime" / "_build_info.json"
    if not info.is_file():
        pytest.fail(f"{info} missing: stage a native build first")
    build = Path(json.loads(info.read_text())["native_build"]["directory"])
    if not (build / "librusticol_loop_sdk.so").is_file():
        pytest.fail(f"{build} has no librusticol_loop_sdk.so")
    include = ROOT / "rust" / "loop-reduction" / "sdk" / "include"
    return f"-I{include}", f"-L{build} -lrusticol_loop_sdk -Wl,-rpath,{build}"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("loop-examples") / "workspace"
    subprocess.run(
        [sys.executable, "-m", "pyamplicol", "examples", "copy", str(root)],
        check=True,
        env=_environment(),
        capture_output=True,
    )
    make = shutil.which("make")
    if make is None:
        pytest.fail("no make on PATH for the native loop client")
    cflags, libs = _loop_sdk()
    subprocess.run(
        [make, "-C", str(root / "native"), "loop"],
        check=True,
        env={**_environment(), "LOOP_SDK_CFLAGS": cflags, "LOOP_SDK_LIBS": libs},
        capture_output=True,
    )
    return root, root / "native" / "loop_client"


@pytest.fixture(scope="module")
def walkthrough(workspace):
    root, client = workspace
    done: dict[str, subprocess.CompletedProcess] = {}

    def run(name: str) -> subprocess.CompletedProcess:
        if name not in done:
            command = [
                sys.executable,
                "python/loop_walkthrough.py",
                f"{name}_generate.toml",
                "--points",
                "1",
                "--batch",
                "4",
                "--samples",
                "1",
                "--color",
                "never",
                "--native-client",
                str(client),
            ]
            if name in LC_VARIANTS:
                command += ["--lc-card", LC_VARIANTS[name]]
            done[name] = subprocess.run(
                command,
                cwd=root,
                env=_environment(),
                capture_output=True,
                text=True,
                check=False,
            )
        return done[name]

    return run


def _tirust() -> bool:
    """Whether the staged native extension serves TIRust (asked in a child
    process: this test process loads no native extension, so its memory
    does not add to the examples')."""

    found = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pyamplicol.loops.native import load_loop_native\n"
            "from pyamplicol.loops.v2.reduction import native_backends\n"
            "print('tirust' in native_backends(load_loop_native()))",
        ],
        env=_environment(),
        capture_output=True,
        text=True,
        check=True,
    )
    return found.stdout.strip().splitlines()[-1] == "True"


@pytest.mark.parametrize("name", EXAMPLES)
def test_the_example_runs_end_to_end(walkthrough, name):
    found = walkthrough(name)
    output = found.stdout
    # The walkthrough exits with its check's exit code.
    assert found.returncode == 0, found.stderr[-4000:]
    assert "check exit code 0" in output, output[-4000:]
    for step in STEPS:
        assert step in output, (step, output[-4000:], found.stderr[-4000:])
    # Both settings of the per-stage profile.
    assert output.count("Time per stage") == 2
    assert "unchecked (timing mode, no accuracy certificate)" in output
    assert "Correlated request: total" in output
    oracle = (
        "the per-helicity calls summed"
        if name in LOOP_INDUCED
        else "the correlated born total"
    )
    assert f"sum over orders vs {oracle}: PASS" in output
    # The native client, without a Symbolica licence.
    assert "SYMBOLICA_LICENSE unset" in output
    assert "auto (checked):" in output and "auto (unchecked):" in output
    assert "colour histories:" in output
    tir = _tirust()
    if name in LOOP_INDUCED:
        expected = "tirust" if tir else "ninjars"
        assert f"backend: {expected} (requested auto)" in output
        assert "time one / summed" in output and "one helicity / summed call" in output
        assert "explicit ninjars" in output
    else:
        assert "backend: ninjars (requested auto)" in output
        assert "explicit tirust" in output
        if not tir:
            assert "explicit tirust: refused" in output
    if name in LC_VARIANTS:
        assert "time one / summed" in output
    elif name not in LOOP_INDUCED:
        assert "exists at leading colour only" in output


@pytest.mark.parametrize("name", EXAMPLES)
def test_the_example_passes_check(walkthrough, name):
    output = walkthrough(name).stdout
    assert "check exit code 0" in output, output[-6000:]


_RECOLA_EVALUATION = """
import json, sys
from pyamplicol import Runtime
from pyamplicol.api import VirtualStabilityChecks
from pyamplicol.loops.v2.stability import RESCUE_TARGET
artifact, points, scales = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
def name(order):
    return " ".join(f"{k}^{v}" for k, v in order if v)
checks = VirtualStabilityChecks(1, True, rescaled=True)
result = Runtime.load(artifact).evaluate(points, mu_r2=scales, stability_checks=checks)
print(json.dumps({"target": float(RESCUE_TARGET), "rows": [
    {
        "born": {
            name(order): complex(value).real
            for order, value in zip(result.born_orders, result.born_values[i])
        },
        "virtual": {
            name(s.order): [complex(x).real for x in result.values[i][j]]
            for j, s in enumerate(result.sectors)
        },
        "estimate": {
            name(s.order): [float(x) for x in result.stability[i].absolute[j]]
            for j, s in enumerate(result.sectors)
        },
    }
    for i in range(len(points))
]}))
"""


def test_example_2_equals_recola(walkthrough, workspace):
    """Example 2 (the walkthrough's artifact, complex-mass scheme) against
    Recola 2.3.0 at its four points (fixture provenance; m-lit lit-ew-blocks,
    the block colour-order fix of the e^6 loop): the Born per squared order to
    1e-12; both poles of every sector within ten times the evaluation's own
    estimate E plus 1e-12 of the sector's pole scale (Recola's larger pole
    coefficient); each sector's finite part within the fixture's CMS
    convention level of its point (the two codes' complex-mass counterterm
    prescriptions, measured on the neutral-only helicity rows; at least the
    default target accuracy ``stability.RESCUE_TARGET``, read in the child
    process like the values). Before the fix the
    QED^10 and QCD^2 QED^8 poles were off by 3e-3 - 1.8 and the finite parts
    by 0.09 - 2.5."""

    assert walkthrough("loop_uux_ddxepem").returncode == 0
    root, _client = workspace
    fixture = json.loads(RECOLA.read_text())
    points = json.loads(
        (ROOT / "examples" / "data" / "loop_uux_ddxepem_momenta.json").read_text()
    )
    found = subprocess.run(
        [
            sys.executable,
            "-c",
            _RECOLA_EVALUATION,
            str(root / "artifacts" / "loop_uux_ddxepem"),
            json.dumps(points),
            json.dumps(fixture["mu_r2"]),
        ],
        env=_environment(),
        capture_output=True,
        text=True,
        check=True,
    )
    evaluated = json.loads(found.stdout.strip().splitlines()[-1])
    for point, (row, born, virtual, levels) in enumerate(
        zip(
            evaluated["rows"],
            fixture["born"],
            fixture["virtual"],
            fixture["finite_levels"],
            strict=True,
        )
    ):
        assert set(row["born"]) == set(born)
        for order, expected in born.items():
            assert abs(row["born"][order] - expected) <= 1e-12 * abs(expected), (
                point,
                order,
            )
        assert set(row["virtual"]) == set(virtual)
        for sector, expected in virtual.items():
            value, estimate = row["virtual"][sector], row["estimate"][sector]
            scale = max(abs(expected[1]), abs(expected[2]))
            for k in (1, 2):
                bound = 10 * estimate[k] + 1e-12 * scale
                assert abs(value[k] - expected[k]) <= bound, (point, sector, k)
            level = max(levels[sector], evaluated["target"])
            deviation = abs(value[0] - expected[0]) / abs(expected[0])
            assert deviation <= level, (point, sector, deviation, level)
