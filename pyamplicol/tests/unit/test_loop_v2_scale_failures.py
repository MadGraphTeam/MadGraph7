# SPDX-License-Identifier: 0BSD
"""A mu_R^2 at which a counterterm coupling is not finite fails its own points.

The native parameter program refuses a whole batch when a counterterm coupling
is not finite at one point's scale in the lane's carrier (``mu_r2 = "1e-400"``
underflows in binary64). The v2 runtimes re-run such a batch per exact scale
(``runtime.virtual.isolate_scale_failures``), so with ``on_failure="return"``
(and with the default "best-effort", which has no value to offer there) only
the points of that scale fail, with a typed per-point diagnostic, and the
other points keep the values they have without it (HANDOFF §9.16). A
non-positive ``mu_r2`` stays an invalid input of the whole call.
"""

from __future__ import annotations

import json

import pytest

from pyamplicol.api.errors import EvaluationError

POINTS = [
    [[500, 0, 0, 500], [500, 0, 0, -500], [500, 300, 0, 400], [500, -300, 0, -400]],
    [[500, 0, 0, 500], [500, 0, 0, -500], [500, 0, 400, 300], [500, 0, -400, -300]],
]
PARAMETERS = {"aS": "0.118", "MT": "173"}
GOOD = ("1000000", "250000")
BAD = "1e-400"
NOT_FINITE = "a counterterm coupling's diagnostic is not finite"

CARD = """
action = "generate"
[model]
source = {model}
[process]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[color]
accuracy = "full"
contraction = "direct"
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{MB = "0"}}
[loop.model]
loop_coupling_orders = ["QCD"]
uv_complete = true
model_r2_complete = true
[loop.model.scheme]
dimensional_regularization = "HV"
external_states = "physical-four-dimensional"
gamma5 = "source-UFO-four-dimensional-prescription"
renormalization = "source-UFO mass-dependent decoupling"
mass_scheme = "real-on-shell"
finite_restoration = "source-UFO-local-rules"
model_convention = "loop_sm lhv=1 supplied QCD rules"
[generation]
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


@pytest.fixture(scope="module")
def runtime(loop_sm_compiled_model, loop_native, tmp_path_factory):
    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    tmp = tmp_path_factory.mktemp("v2-scale-failures")
    card = tmp / "card.toml"
    card.write_text(CARD.format(model=json.dumps(str(loop_sm_compiled_model))))
    output = tmp / "artifact"
    Generator(resolve_config(card)).generate(
        ProcessSet((ProcessRequest.parse("u u~ > d d~", name="uudd"),)), output
    )
    return Runtime.load(output, process="uudd", model_parameters=PARAMETERS)


def _assert_failed_alone(result, bad: set[int]) -> None:
    for index, diagnostic in enumerate(result.diagnostics):
        if index not in bad:
            assert diagnostic.status == "success", diagnostic
            continue
        assert diagnostic.status == "failed"
        assert result.values[index] is None
        (failure,) = diagnostic.failures
        assert failure.point_id == index and failure.stage == "v2-evaluation"
        assert failure.detail.startswith("mu_r2 = 1E-400: ")
        assert NOT_FINITE in failure.detail


@pytest.mark.loop_physics
def test_a_non_finite_counterterm_coupling_fails_only_its_points(runtime):
    """Inclusive (``loops.v2.artifact``) and resolved (``runtime.virtual``)
    evaluations, with and without poles: the bad scale's point fails alone,
    the others equal a batch without it bitwise; the default best-effort
    policy returns it failed (no value exists) with a "failed" warning as
    data; ``raise`` names the point."""

    points = [POINTS[0], POINTS[0], POINTS[1]]
    scales = [GOOD[0], BAD, GOOD[1]]
    clean_points = [POINTS[0], POINTS[1]]
    for poles in ("all", "none"):
        clean = runtime.evaluate(clean_points, mu_r2=list(GOOD), poles=poles)
        mixed = runtime.evaluate(points, mu_r2=scales, poles=poles, on_failure="return")
        _assert_failed_alone(mixed, {1})
        assert [mixed.values[0], mixed.values[2]] == list(clean.values)
        assert [mixed.born_values[0], mixed.born_values[2]] == list(clean.born_values)
    resolved_clean = runtime.evaluate_resolved(clean_points, mu_r2=list(GOOD))
    resolved = runtime.evaluate_resolved(points, mu_r2=scales, on_failure="return")
    _assert_failed_alone(resolved, {1})
    assert [resolved.values[0], resolved.values[2]] == list(resolved_clean.values)
    # A batch of the bad scale only: every point returned, failed.
    alone = runtime.evaluate(POINTS, mu_r2=BAD, on_failure="return")
    _assert_failed_alone(alone, {0, 1})
    # The default policy ("best-effort", m-lit ux-besteffort) never raises
    # for a point: the bad scale's point has no value at all (no unchecked
    # reduction can give one without a finite coupling), so it stays failed
    # with its reason, returned as data in one "failed" warning; the other
    # points are the clean batch's values bitwise.
    default = runtime.evaluate(points, mu_r2=scales, poles="none")
    _assert_failed_alone(default, {1})
    assert default.diagnostics[1].best_effort is None
    assert [default.values[0], default.values[2]] == list(clean.values)  # "none"
    (warning,) = (w for w in default.warnings if w.source == "numerical")
    assert (warning.kind, warning.count, warning.points) == ("failed", 1, (1,))
    assert NOT_FINITE in warning.message
    # The opt-in strict mode raises and names the point.
    with pytest.raises(EvaluationError, match=r"failed at point 1: mu_r2 = 1E-400: "):
        runtime.evaluate(points, mu_r2=scales, on_failure="raise")
    # A non-positive scale is an invalid input of the whole call.
    with pytest.raises(ValueError, match="mu_r2 must be positive"):
        runtime.evaluate(points, mu_r2=[GOOD[0], "0", GOOD[1]], on_failure="return")


@pytest.mark.loop_fast
def test_scale_isolation_keeps_point_failures_and_other_errors(loop_native):
    """``isolate_scale_failures`` without a native batch: per-scale reruns
    map their results and point failures back to the caller's indices, one
    failing scale fails its points only, and any other error propagates,
    also one with the same text that is not the native typed refusal
    (``NonFiniteCountertermError``)."""

    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.runtime.virtual import isolate_scale_failures

    del loop_native
    typed = load_loop_native().NonFiniteCountertermError
    assert issubclass(typed, ValueError)
    scales = ["4", "1e-400", "9", "4.0", "1E-400"]
    calls: list[list[int]] = []

    def run(indices: list[int]) -> tuple[list, dict[int, list[str]]]:
        calls.append(indices)
        if any(scales[i].lower() == "1e-400" for i in indices):
            raise typed(f"loop v2 trees: {NOT_FINITE} (helper)")
        found = [f"value {i}" for i in indices]
        failed = {
            k: ["lane 0: ImpreciseResult"] for k, i in enumerate(indices) if i == 2
        }
        for k in failed:
            found[k] = None
        return found, failed

    results, failures = isolate_scale_failures(run, scales)
    assert calls == [[0, 1, 2, 3, 4], [0, 3], [1, 4], [2]]
    assert results == ["value 0", None, None, "value 3", None]
    assert failures == {
        1: [f"mu_r2 = 1e-400: loop v2 trees: {NOT_FINITE} (helper)"],
        4: [f"mu_r2 = 1e-400: loop v2 trees: {NOT_FINITE} (helper)"],
        2: ["lane 0: ImpreciseResult"],
    }
    # One scale: no rerun.
    calls.clear()
    _, failures = isolate_scale_failures(run, ["1e-400", "1e-400"])
    assert calls == [[0, 1]] and set(failures) == {0, 1}

    def other(indices: list[int]) -> tuple[list, dict[int, list[str]]]:
        raise ValueError("loop v2 trees: unknown parameter")

    with pytest.raises(ValueError, match="unknown parameter"):
        isolate_scale_failures(other, scales)

    def untyped(indices: list[int]) -> tuple[list, dict[int, list[str]]]:
        raise ValueError(f"loop v2 trees: {NOT_FINITE} (helper)")

    with pytest.raises(ValueError, match="diagnostic is not finite"):
        isolate_scale_failures(untyped, scales)
