# SPDX-License-Identifier: 0BSD
"""TIRust as the loop-induced default at a target accuracy (m-lit F3,
lit-tirust-default; USER 2026-11-08 ~14:40 / ~15:40, DECISIONS 15:19 / 15:32).

* The one ``"auto"`` definition (Rust ``reduction_policy``, R7) takes TIRust
  for loop-induced processes with NinjaRS as its per-point fallback, NinjaRS
  for Born-interfered ones; an explicit backend and a TIR-only artifact have
  no fallback; a native without TIRust takes NinjaRS with the reason. One
  helper ``auto_default`` is what the inspect record and the SDK capabilities
  read.
* The target accuracy: its default is the native policy's (1e-5, the
  rescue's ``"relative"`` target), one validator and one refusal text for
  the configuration, the CLI, the Python API and the SDKs.
* The rescue stack: (precision bits, reduction) levels in non-decreasing
  precision, a repeated pair refused; the loop-induced default stack.

The evaluations (the fallback on the LI-O1 rows, the loop-induced default
stack's records, the C ABI) are in ``test_loop_v2_induced_observables.py``
and ``test_loop_v2_sdk_induced.py``, which generate those rows.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.loop_fast

ALL = ("double", "double-double", "arbitrary")


class _Native:
    """A native extension's capability query (``reduction_backends``)."""

    def __init__(self, served):
        self._served = served

    def reduction_backends(self):
        return {name: list(carriers) for name, carriers in self._served.items()}


def _choice(requested, kind, carrier="double", native=None, admitted=None):
    from pyamplicol.loops.v2.reduction import SUMMED, resolve

    return resolve(
        requested,
        contribution=kind,
        carrier=carrier,
        native=native or _Native({"ninjars": ALL, "tirust": ALL}),
        route=SUMMED,
        admitted=admitted,
    )


def test_auto_takes_tirust_for_loop_induced_with_the_per_point_fallback(loop_native):
    from pyamplicol.loops.v2.reduction import text

    for carrier in ALL:
        found = _choice("auto", "loop-induced", carrier)
        assert (found.requested, found.effective, found.fallback) == (
            "auto",
            "tirust",
            "ninjars",
        ), carrier
        assert found.reason.startswith("auto: TIRust, the loop-induced default; ")
        assert "per-point fallback" in found.reason
        found = _choice("auto", "virtual", carrier)
        assert (found.effective, found.fallback) == ("ninjars", None)
        # Explicit backends: as requested, never a fallback.
        for backend in ("ninjars", "tirust"):
            for kind in ("virtual", "loop-induced"):
                found = _choice(backend, kind, carrier)
                assert (found.effective, found.reason, found.fallback) == (
                    backend,
                    "requested",
                    None,
                )
    # A TIR-only artifact: TIRust, no fallback (NinjaRS refused typed).
    only = {"tirust"}
    for kind in ("virtual", "loop-induced"):
        found = _choice("auto", kind, admitted=only)
        assert (found.effective, found.fallback) == ("tirust", None)
        assert found.reason == "auto: TIRust, the only backend the artifact admits"
    # A native built without TIRust (tir off): NinjaRS with its reason.
    off = _Native({"ninjars": ALL})
    found = _choice("auto", "loop-induced", native=off)
    assert (found.effective, found.fallback) == ("ninjars", None)
    assert "TIRust, the loop-induced default, is unavailable" in found.reason
    # The reason texts are the native policy's.
    assert text("served", "tirust", "ninjars", "x") == (
        "auto: NinjaRS, the per-point fallback: TIRust refused or failed the point (x)"
    )
    assert text("failed", "ninjars", "", "y") == "NinjaRS fallback: y"
    assert text("cross", detail="z").endswith("other reduction backend: z")


def test_one_auto_default_helper_and_the_policy_record(loop_native):
    from pyamplicol.loops.v2.reduction import DEFAULTS, auto_default, policy

    assert auto_default("loop-induced") == DEFAULTS["loop-induced"] == "tirust"
    assert auto_default("virtual") == DEFAULTS["virtual"] == "ninjars"
    assert auto_default("virtual", {"tirust"}) == "tirust"
    assert auto_default("virtual", {"ninjars", "tirust"}) == "ninjars"
    with pytest.raises(ValueError, match="unknown contribution"):
        auto_default("tree")
    record = policy()
    assert record["fallback"] == "ninjars"
    for route in record["routes"].values():
        assert sorted(route["auto"]) == ["ninjars", "tirust"]


def test_the_target_accuracy_has_one_default_and_one_validator(loop_native):
    from pyamplicol.api.virtual_results import VirtualReduction
    from pyamplicol.config import ConfigurationError
    from pyamplicol.config.models import EvaluationConfig, reduction_target
    from pyamplicol.loops.v2.reduction import default_target, policy
    from pyamplicol.loops.v2.stability import RESCUE_TARGET

    # The native policy's one definition, the rescue's "relative" target.
    assert default_target() == 1e-5 == RESCUE_TARGET
    refusal = policy()["target"]["refusal"]
    assert reduction_target(1e-7) == 1e-7
    assert reduction_target(2) == 2.0
    for bad in (0, 0.0, -1e-5, math.nan, math.inf, True, "1e-5", None):
        with pytest.raises(ValueError) as refused:
            reduction_target(bad)
        assert str(refused.value) == refusal
        if bad is not None:  # None: the default target
            with pytest.raises(ValueError, match="relative accuracy"):
                VirtualReduction("auto", "tirust", "x", target=bad)
    assert EvaluationConfig(reduction_target=1e-6).reduction_target == 1e-6
    assert EvaluationConfig().reduction_target is None
    with pytest.raises(ConfigurationError, match=r"evaluation\.reduction_target"):
        EvaluationConfig(reduction_target=-1.0)
    # A fallback is the other backend of an "auto" request.
    with pytest.raises(ValueError, match="other backend"):
        VirtualReduction("auto", "tirust", "x", fallback="tirust")
    with pytest.raises(ValueError, match="only an 'auto' request"):
        VirtualReduction("tirust", "tirust", "x", fallback="ninjars")


def test_the_cli_and_the_run_card_set_the_target(tmp_path):
    from pyamplicol.cli.parser import build_parser

    parser = build_parser()
    found = parser.parse_args(
        [
            "evaluate",
            "artifact",
            "--reduction-target",
            "1e-7",
            "--loop-reduction",
            "tirust",
        ]
    )
    options = vars(found)
    assert options["evaluation.reduction_target"] == 1e-7
    assert options["evaluation.reduction"] == "tirust"


def test_the_rescue_stack_takes_precision_and_backend_levels():
    from pyamplicol.api.virtual_results import VirtualRescueLevel, VirtualRescuePolicy
    from pyamplicol.loops.v2.rescue import default_stack, loop_induced_stack

    # A same-precision level on the other backend is allowed.
    stack = VirtualRescuePolicy(
        (
            VirtualRescueLevel("double", reduction="tirust"),
            VirtualRescueLevel("double", reduction="ninjars"),
            VirtualRescueLevel("double-double"),
        )
    )
    assert [level.reduction for level in stack.levels] == ["tirust", "ninjars", "auto"]
    # A repeated (precision, backend) pair, or a lower precision, is refused.
    with pytest.raises(ValueError, match="repeated"):
        VirtualRescuePolicy(
            (VirtualRescueLevel("double"), VirtualRescueLevel("double"))
        )
    with pytest.raises(ValueError, match="repeated"):
        VirtualRescuePolicy(
            (
                VirtualRescueLevel("arbitrary", 40, reduction="ninjars"),
                VirtualRescueLevel("arbitrary", 40, reduction="ninjars"),
            )
        )
    with pytest.raises(ValueError, match="non-decreasing"):
        VirtualRescuePolicy(
            (VirtualRescueLevel("double-double"), VirtualRescueLevel("double"))
        )
    # The loop-induced default stack: TIRust b64 -> NinjaRS b64 -> the default
    # ladder's double-double and 57-digit levels ("auto"), every target 1e-5.
    default = VirtualRescuePolicy()
    found = loop_induced_stack()
    assert [
        (level.arithmetic, level.precision, level.reduction) for level in found.levels
    ] == [
        ("double", None, "tirust"),
        ("double", None, "ninjars"),
        ("double-double", None, "auto"),
        ("arbitrary", 57, "auto"),
    ]
    assert found.levels[0] == default.levels[0].__class__(
        **{**_fields(default.levels[0]), "reduction": "tirust"}
    )
    assert found.levels[2:] == default.levels[1:]
    # It replaces the default policy only for a loop-induced "auto" call whose
    # binary64 "auto" has the NinjaRS fallback.
    assert default_stack(default, "loop-induced", "auto", "ninjars") == found
    assert default_stack(default, "loop-induced", "ninjars", None) is default
    assert default_stack(default, "loop-induced", "tirust", None) is default
    assert default_stack(default, "loop-induced", "auto", None) is default
    assert default_stack(default, "virtual", "auto", None) is default
    assert default_stack(stack, "loop-induced", "auto", "ninjars") is stack
    assert default_stack(None, "loop-induced", "auto", "ninjars") is None


def _fields(level):
    from dataclasses import fields

    return {item.name: getattr(level, item.name) for item in fields(level)}
