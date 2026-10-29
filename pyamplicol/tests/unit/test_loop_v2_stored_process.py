# SPDX-License-Identifier: 0BSD
"""A stored (prepared) plan v4 process regenerates nothing on any call.

REQUIREMENTS_ADDENDUM G / HANDOFF §0.2 (d): loading an artifact never
regenerates the process, and neither does any call of the API contract on
the loaded process. The call kinds of one stored process run in one fresh
interpreter under the generation sentinel of
``tools/loop_artifact_portability.py`` (``GenerationSentinel``: every call of
slice, plan, tree-program or model generation code is counted from before
the first pyAmpliCol import to the end of the last call, and attributed to
the call it ran in), and each must count zero:

* the inclusive virtual in every pole selection (``all``, ``none``, ``uv``,
  ``ir``: the stored companion of the certified-scaleless families, with the
  split ``pole_parts``), with rotation and reduction-direction stability
  checks, in double-double and arbitrary precision, after
  ``set_model_parameters``, and its per-coupling-order partials;
* an exact flavour member served by its representative;
* colour-correlated (also with the companion's columns in ``uv``, finite
  only, with stability checks and in double-double), spin-correlated (also
  in ``uv``) and helicity-resolved outputs (also in ``ir``) and the
  correlation catalogue (the stored observable columns,
  ``native_observables``);
* the same calls of a leading-colour process (its colour-pruned slice, the
  pruned companion and its column twins), and nightly of a five-leg
  next-to-leading-colour one (``test_nlc_five_leg_stored_process_regenerates
  _nothing``).

Every call's values in the fresh interpreter equal those of the generating
test process's own load (bitwise; extended precision as decimal text).
Binary64 observable calls also run without a licence while another process
holds the restricted Symbolica permit (they touch no Symbolica state).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OBSERVABLES = ROOT / "tests" / "fixtures" / "loop" / "legacy" / "observables"
_tool_spec = importlib.util.spec_from_file_location(
    "loop_artifact_portability", ROOT / "tools" / "loop_artifact_portability.py"
)
portability = importlib.util.module_from_spec(_tool_spec)
_tool_spec.loader.exec_module(portability)

SCHEME = """
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
"""


def _record(name: str) -> dict:
    if name == "gg_plain":
        # The g g > g g record's points, without declared observables.
        return {**_record("gg_gg"), "histories": [], "spin_classes": []}
    return json.loads((OBSERVABLES / f"{name}.json").read_text())


def _record_path(name: str) -> Path:
    """The record whose points and parameters a call of ``name`` reads."""

    return OBSERVABLES / ("gg_gg.json" if name == "gg_plain" else f"{name}.json")


def _card(
    model: Path, output: Path, entries, process: str = "", accuracy: str = "full"
) -> str:
    processes = ", ".join(
        f"{{expression={json.dumps(expression)}, name={json.dumps(name)}}}"
        for name, expression in entries
    )
    return f"""
action = "generate"
[model]
source = {json.dumps(str(model))}
[process]
entries = [{processes}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
{process}
[color]
accuracy = "{accuracy}"
contraction = "direct"
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{MB = "0"}}
{SCHEME}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
samples = 1
"""


def _generate(model: Path, name: str, root: Path, process: str = "") -> Path:
    """The artifact of observables record ``name`` (its processes, colour
    histories, spin classes and colour accuracy) through the public API."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.correlators import CorrelatorConfig

    record = _record(name)
    entries = tuple((key, expression) for key, expression in record["processes"])
    card = root / "card.toml"
    card.write_text(
        _card(
            model,
            root / "artifact",
            entries,
            process,
            record.get("accuracy", "full"),
        )
    )
    correlators = CorrelatorConfig.from_json_dict(
        {
            "color_correlations": record["histories"],
            "spin_correlations": record["spin_classes"],
        }
    )
    Generator(resolve_config(card)).generate(
        ProcessSet(
            tuple(
                ProcessRequest.parse(expression, name=key)
                for key, expression in entries
            )
        ),
        root / "artifact",
        correlators=correlators,
    )
    return root / "artifact"


@pytest.fixture(scope="module")
def stored(loop_sm_compiled_model, loop_sm_colour_model, loop_native, tmp_path_factory):
    """The artifacts of the g g > g g observables record (colour histories
    and spin classes declared) in full and leading colour (``gg_gg_lc``), of
    d d~ > g g with its exact flavour member s s~ > g g, and of g g > g g
    without declared observables (``gg_plain``: the Born colour metric and
    the resolved columns only), generated through the public API."""

    out = {}
    for name, process in (
        ("gg_gg", ""),
        ("ddgg_ssgg", "flavor_blind = true"),
        ("gg_plain", ""),
        ("gg_gg_lc", ""),
    ):
        model = loop_sm_colour_model if name == "gg_gg_lc" else loop_sm_compiled_model
        out[name] = _generate(model, name, tmp_path_factory.mktemp(name), process)
    return out


# One call kind per case: (artifact, process, call).
CALLS = {
    "all": ("gg_gg", "gg", "evaluate"),
    "none": ("gg_gg", "gg", "evaluate-none"),
    "uv": ("gg_gg", "gg", "evaluate-uv"),
    "ir": ("gg_gg", "gg", "evaluate-ir"),
    "stability": ("gg_gg", "gg", "evaluate-stability"),
    "double-double": ("gg_gg", "gg", "evaluate-double-double"),
    "arbitrary": ("gg_gg", "gg", "evaluate-arbitrary"),
    "parameters": ("gg_gg", "gg", "evaluate-parameters"),
    "orders": ("gg_gg", "gg", "evaluate-orders"),
    "member": ("ddgg_ssgg", "ssgg", "evaluate"),
    "colour": ("gg_gg", "gg", "correlated-colour"),
    "colour-uv": ("gg_gg", "gg", "correlated-uv"),
    "colour-stability": ("gg_gg", "gg", "correlated-stability"),
    "colour-double-double": ("gg_gg", "gg", "correlated-double-double"),
    "member-colour": ("ddgg_ssgg", "ssgg", "correlated-last"),
    "spin": ("gg_gg", "gg", "correlated-spin"),
    "many": ("gg_gg", "gg", "correlated-many"),
    "resolved": ("gg_gg", "gg", "resolved"),
    "catalogue": ("gg_gg", "gg", "catalogue"),
    # Without declared observables: the catalogue, the Born colour metric's
    # correlated output and the resolved outputs.
    "plain-catalogue": ("gg_plain", "gg", "catalogue"),
    "plain-born": ("gg_plain", "gg", "correlated-born"),
    "plain-resolved": ("gg_plain", "gg", "resolved"),
    # Pole-selected resolved, spin and finite-only correlated calls.
    "resolved-ir": ("gg_gg", "gg", "resolved-ir"),
    "spin-uv": ("gg_gg", "gg", "correlated-spin-uv"),
    "colour-none": ("gg_gg", "gg", "correlated-none"),
    # Leading colour: the colour-pruned slice, its pruned companion and the
    # companion's column twins.
    "lc-ir": ("gg_gg_lc", "gg", "evaluate-ir"),
    "lc-colour-uv": ("gg_gg_lc", "gg", "correlated-uv"),
    "lc-spin": ("gg_gg_lc", "gg", "correlated-spin"),
    "lc-resolved-ir": ("gg_gg_lc", "gg", "resolved-ir"),
}

# Runs call kinds of one stored process in turn (argv: artifact, process or
# "", the calls as JSON, record) under one generation sentinel and prints per
# call its generation calls (those counted while it ran; the imports' under
# "") and values (binary64 bits as hexadecimal literals).
_PROBE = r"""
import json, sys
from pathlib import Path
spec_path, artifact, process, calls, record_path = sys.argv[1:6]
import importlib.util
spec = importlib.util.spec_from_file_location("portability", spec_path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
record = json.loads(Path(record_path).read_text())
process = process or None
out = {}
with tool.GenerationSentinel() as sentinel:
    from pyamplicol import Runtime
    from pyamplicol.api import VirtualStabilityChecks
    from pyamplicol.correlators import CorrelatedRequest
    from decimal import Decimal
    out[""] = {"calls": dict(sentinel.calls), "values": None}
    for call in json.loads(calls):
        before = dict(sentinel.calls)
        values = run(Runtime, artifact, process, call, record)
        counted = {
            name: count - before.get(name, 0)
            for name, count in sentinel.calls.items()
            if count != before.get(name, 0)
        }
        out[call] = {"calls": counted, "values": values}
print(json.dumps(out))
"""

_RUN = r"""
def bits(value):
    if value is None:
        return None
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, complex):
        return [value.real.hex(), value.imag.hex()]
    if hasattr(value, "finite"):
        return [bits(value.finite), bits(value.single_pole), bits(value.double_pole)]
    if isinstance(value, (list, tuple)):
        return [bits(v) for v in value]
    if hasattr(value, "real") and hasattr(value, "imag"):
        return [bits(float(value.real)), bits(float(value.imag))]
    return str(value)


def run(Runtime, artifact, process, call, record):
    from pyamplicol.api import VirtualStabilityChecks
    from pyamplicol.correlators import CorrelatedRequest
    from decimal import Decimal

    parameters = record["parameters"]
    runtime = Runtime.load(artifact, process=process, model_parameters=parameters)
    points, scales = record["points"][:1], record["scales_squared"][:1]
    spin = {
        int(leg): tuple(
            (Decimal(re), Decimal(im)) for re, im in record["spin_vectors"][k]
        )
        for k, leg in enumerate(["3", "4"])
    }
    if call == "evaluate":
        result = runtime.evaluate(points, mu_r2=scales)
    elif call.startswith("evaluate-") and call[9:] in ("none", "uv", "ir"):
        result = runtime.evaluate(points, mu_r2=scales, poles=call[9:])
        if call[9:] != "none":
            return [bits(result.values), bits(result.pole_part("uv", "bare"))]
    elif call == "evaluate-stability":
        checks = VirtualStabilityChecks(1, reduction_direction=True)
        result = runtime.evaluate(points, mu_r2=scales, stability_checks=checks)
        return [bits(result.values), [e.relative.hex() for e in result.stability]]
    elif call == "evaluate-double-double":
        result = runtime.evaluate(points, mu_r2=scales, arithmetic="double-double")
        return [str(v) for point in result.values for laurent in point for v in laurent]
    elif call == "evaluate-arbitrary":
        result = runtime.evaluate(
            points, mu_r2=scales, arithmetic="arbitrary", precision=32
        )
        return [str(v) for point in result.values for laurent in point for v in laurent]
    elif call == "evaluate-parameters":
        runtime.set_model_parameters({"aS": "0.13"})
        result = runtime.evaluate(points, mu_r2=scales)
    elif call == "evaluate-orders":
        result = runtime.evaluate(points, mu_r2=scales)
        order = result.sectors[0].order
        return bits(result.for_order(order).values)
    elif call == "correlated-colour":
        result = runtime.evaluate_correlated(
            points, color_correlation="T12", mu_r2=scales
        )
    elif call == "correlated-born":
        result = runtime.evaluate_correlated(
            points, color_correlation="born", mu_r2=scales, poles="uv"
        )
        return [bits(result.values), bits(result.pole_part("uv", "bare"))]
    elif call == "correlated-last":
        # The record's last declared history (a directed one).
        result = runtime.evaluate_correlated(
            points, color_correlation=record["histories"][-1]["id"], mu_r2=scales
        )
    elif call == "correlated-uv":
        result = runtime.evaluate_correlated(
            points, color_correlation="T12", mu_r2=scales, poles="uv"
        )
        return [bits(result.values), bits(result.pole_part("uv", "bare"))]
    elif call == "correlated-stability":
        checks = VirtualStabilityChecks(1, reduction_direction=True)
        result = runtime.evaluate_correlated(
            points, color_correlation="T12", mu_r2=scales, stability_checks=checks
        )
        return [bits(result.values), [e.relative.hex() for e in result.stability]]
    elif call == "correlated-double-double":
        result = runtime.evaluate_correlated(
            points, color_correlation="T12", mu_r2=scales, arithmetic="double-double"
        )
        return [str(v) for point in result.values for laurent in point for v in laurent]
    elif call == "correlated-spin":
        result = runtime.evaluate_correlated_many(
            points, {"spin": CorrelatedRequest("born", spin)}, mu_r2=scales
        )["spin"]
    elif call == "correlated-spin-uv":
        result = runtime.evaluate_correlated_many(
            points, {"spin": CorrelatedRequest("born", spin)}, mu_r2=scales, poles="uv"
        )["spin"]
        return [bits(result.values), bits(result.pole_part("uv", "bare"))]
    elif call == "correlated-none":
        result = runtime.evaluate_correlated(
            points, color_correlation="T12", mu_r2=scales, poles="none"
        )
    elif call == "correlated-many":
        values = runtime.evaluate_correlated_many(
            points,
            {
                name: CorrelatedRequest(name, {})
                for name in ("casimir", "T12", "T34")
            },
            mu_r2=scales,
        )
        return {name: bits(value.values) for name, value in sorted(values.items())}
    elif call == "resolved":
        result = runtime.evaluate_resolved(points, mu_r2=scales)
    elif call == "resolved-ir":
        result = runtime.evaluate_resolved(points, mu_r2=scales, poles="ir")
    elif call == "catalogue":
        return [item.id for item in runtime.available_color_correlations()]
    else:
        raise ValueError(call)
    return [bits(result.born_values), bits(result.values)]
"""


def _calls(
    artifact: Path,
    process: str | None,
    calls: list[str],
    record: Path,
    env: dict[str, str] | None = None,
) -> dict:
    """``calls`` of one stored process in one fresh interpreter (``_PROBE``):
    per call (and ``""``, the imports) its generation calls and values."""

    script = _RUN + _PROBE
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT / "tools" / "loop_artifact_portability.py"),
            str(artifact),
            process or "",
            json.dumps(calls),
            str(record),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
        env=env,
    )
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-4000:]
    assert "restricted Symbolica" not in run.stdout + run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


def _case(name: str):
    return pytest.param(name, marks=(pytest.mark.loop_physics,), id=name)


@pytest.fixture(scope="module")
def fresh(stored):
    """Every call kind of ``CALLS`` in a fresh interpreter, one per stored
    process (``_calls``): per call kind its generation calls and values, and
    per stored process the imports' generation calls."""

    groups: dict[tuple[str, str], list[str]] = {}
    for artifact, process, call in CALLS.values():
        calls = groups.setdefault((artifact, process), [])
        if call not in calls:
            calls.append(call)
    out = {}
    for (artifact, process), calls in groups.items():
        found = _calls(stored[artifact], process, calls, _record_path(artifact))
        out[(artifact, process)] = found
    return out


@pytest.mark.parametrize("name", [_case(name) for name in CALLS])
def test_stored_process_call_regenerates_nothing(stored, fresh, name):
    artifact, process, call = CALLS[name]
    assert fresh[(artifact, process)][""]["calls"] == {}
    found = fresh[(artifact, process)][call]
    assert found["calls"] == {}, found["calls"]
    # The same call in this process (which generated the artifact) gives
    # the same bits.
    namespace: dict = {}
    exec(_RUN, namespace)
    from pyamplicol import Runtime

    here = namespace["run"](Runtime, stored[artifact], process, call, _record(artifact))
    assert json.loads(json.dumps(here)) == found["values"]


# The call kinds of the five-leg NLC process (its record declares dipoles,
# no spin classes).
NLC_CALLS = (
    "evaluate",
    "evaluate-none",
    "evaluate-uv",
    "evaluate-ir",
    "correlated-colour",
    "correlated-uv",
    "correlated-none",
    "resolved",
    "resolved-ir",
    "catalogue",
)


@pytest.mark.loop_nightly
def test_nlc_five_leg_stored_process_regenerates_nothing(
    loop_sm_colour_model, loop_native, tmp_path
):
    """u u~ > g g g at next-to-leading colour with its record's dipoles (the
    colour-pruned slice, its companion and the companion's column twins
    rebuilt for the directed histories, ``StoredObservables.export``): every
    call kind in a fresh interpreter regenerates nothing and gives the bits
    of the generating process's own load."""

    artifact = _generate(loop_sm_colour_model, "uux_ggg_nlc", tmp_path)
    record = _record_path("uux_ggg_nlc")
    found = _calls(artifact, "uuggg", list(NLC_CALLS), record)
    assert found[""]["calls"] == {}
    namespace: dict = {}
    exec(_RUN, namespace)
    from pyamplicol import Runtime

    for call in NLC_CALLS:
        assert found[call]["calls"] == {}, (call, found[call]["calls"])
        here = namespace["run"](
            Runtime, artifact, "uuggg", call, _record("uux_ggg_nlc")
        )
        assert json.loads(json.dumps(here)) == found[call]["values"], call


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("artifact", "process"), [("gg_gg", "gg"), ("ddgg_ssgg", "ddgg")]
)
def test_rust_runtime_equals_the_python_api(stored, loop_native, artifact, process):
    """The SDK's plan v4 runtime (``rusticol_loop_reduction::v2_runtime::
    V2Runtime``, through ``evaluate_v2_artifact``: no Python generation or
    assembly) gives bitwise the Python API's binary64 Born and virtual in
    every pole selection (the split ones with the stored companion of the
    certified-scaleless families), at the default and at caller-set model
    parameters."""

    from pyamplicol import Runtime
    from pyamplicol.loops.v2.artifact import SCALELESS_PLAN_MEMBER, read_plan

    pacbin = next(stored[artifact].rglob(f"{process}/loop-runtime.pacbin"))
    # The split selections run the stored companion here.
    assert read_plan(pacbin, SCALELESS_PLAN_MEMBER) is not None
    record = _record(artifact)
    points, scales = record["points"], record["scales_squared"]
    flat = [[str(x) for momentum in point for x in momentum] for point in points]
    for parameters in (record["parameters"], {**record["parameters"], "aS": "0.13"}):
        runtime = Runtime.load(
            stored[artifact], process=process, model_parameters=parameters
        )
        finite = None
        for poles in ("all", "none", "uv", "ir"):
            python = runtime.evaluate(points, mu_r2=scales, poles=poles)
            native = loop_native.evaluate_v2_artifact(
                str(stored[artifact]),
                flat,
                [str(s) for s in scales],
                process=process,
                parameters=parameters,
                poles=poles,
            )
            width = 1 if poles == "none" else 3
            for index, (values, failures) in enumerate(native):
                assert not failures, failures
                born, virtual = values
                assert born == list(python.born_values[index])
                assert virtual == [
                    list(laurent)[:width] for laurent in python.values[index]
                ], (index, poles)
            # The finite part is the same in every selection.
            here = [[laurent[0] for laurent in values[1]] for values, _ in native]
            assert finite is None or here == finite, poles
            finite = here


@pytest.mark.loop_physics
def test_stored_metrics_equal_the_oracle(
    loop_sm_compiled_model, loop_native, monkeypatch
):
    """The directed metrics of the stored columns (C's native colour cells;
    a swapped metric equal to its direct one stored once) equal the
    oracle's per-pair overlaps (``DirectedColour.overlap``) in the stored
    table form, on every dipole of u u~ > g g g; and the column sets' seed
    records, projected natively (``plan_export.NATIVE_COLUMN_RECORDS``:
    package C's ``project_records`` on the directed tables' native cells),
    are the bytes of the exporter's Python projection (the oracle), with the
    same Born components registered in the same order, for the process's
    sets and its companion's twins."""

    from pyamplicol.correlators import CorrelatorConfig
    from pyamplicol.loops.v2 import native_observables as no
    from pyamplicol.loops.v2 import plan_export
    from pyamplicol.loops.v2.artifact import generate_process
    from pyamplicol.loops.v2.assemble import SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    assert plan_export.NATIVE_COLUMN_RECORDS
    projected = []
    set_records = plan_export._set_records

    def records(exporter, columns):
        before = dict(exporter.born_components)
        native = set_records(exporter, columns)
        after = list(exporter.born_components)
        registered = exporter.born_components
        exporter.born_components = dict(before)
        try:
            oracle = exporter.seed_records(columns)
            assert list(exporter.born_components) == after
        finally:
            exporter.born_components = registered
        for value, expected in zip(native[:2], oracle[:2], strict=True):
            assert isinstance(value, plan_export.RawJSON)
            assert value == json.dumps(expected, separators=(",", ":")).encode()
        assert native[2] == oracle[2]
        projected.append(len(columns))
        return native

    monkeypatch.setattr(plan_export, "_set_records", records)
    checked = []
    cells = no._Metrics.cells

    def compare(self, column, swapped):
        key, found = cells(self, column, swapped)
        if found is not None:
            directed = self.observables.directed(column.history)
            oracle = no._cells(
                self.basis,
                self.ket_literals,
                lambda left, right: directed.overlap(left, right, swapped=swapped),
            )
            assert found == oracle, (column.history.id, swapped)
            checked.append(key != (column.history.id, swapped))
        return key, found

    monkeypatch.setattr(no._Metrics, "cells", compare)
    legs = range(1, 6)
    dipoles = [
        {
            "id": f"T{i}{j}",
            "bra": [{"kind": "emit-gluon", "emitter": i, "emitted": -1}],
            "ket": [{"kind": "emit-gluon", "emitter": j, "emitted": -1}],
        }
        for i in legs
        for j in legs
        if i < j
    ]
    generate_process(
        load_compiled_model(loop_sm_compiled_model),
        SliceRequest(
            "u u~ > g g g",
            {"QCD": 3, "QED": 0},
            {"QCD": 5, "QED": 0},
            fixed_parameters={"MB": "0"},
        ),
        loop_native,
        CorrelatorConfig.from_json_dict(
            {"color_correlations": dipoles, "spin_correlations": []}
        ),
    )
    # Both directions of all ten dipoles; each swap is its direct metric.
    assert len(checked) == 20 and sum(checked) == 10
    # The correlated set (the Born metric and ten self-adjoint dipoles, one
    # column each, per sector) and its companion twin; the resolved columns
    # at full colour read the plan's own seed records.
    assert projected == [11, 11]


def _dipoles(legs) -> list[dict]:
    return [
        {
            "id": f"T{i}{j}",
            "bra": [{"kind": "emit-gluon", "emitter": i, "emitted": -1}],
            "ket": [{"kind": "emit-gluon", "emitter": j, "emitted": -1}],
        }
        for i in legs
        for j in legs
        if i < j
    ]


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "accuracy", ["full", pytest.param("nlc", marks=pytest.mark.loop_nightly)]
)
def test_self_adjoint_dipoles_take_one_column(
    loop_sm_compiled_model,
    loop_sm_colour_model,
    loop_native,
    monkeypatch,
    tmp_path,
    accuracy,
):
    """A dipole T_i.T_j is self-adjoint (``observables.SELF_ADJOINT_HISTORIES``:
    its swapped seed table equals its direct one, so its two directions are
    equal on every tensor): one seed column (2 Re) instead of two, in the
    process's and the companion's column sets, and its swapped metric is not
    evaluated. Every correlated output of u u~ > g g g with all ten dipoles
    declared, in every pole selection, binary64 and double-double, is
    bitwise that of the two-column form (the oracle), at full colour and at
    NLC (whose companion is rebuilt for the histories and takes the slice's
    decisions)."""

    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.correlators import CorrelatedRequest, CorrelatorConfig
    from pyamplicol.loops.v2 import observables, plan_export

    set_records = plan_export._set_records
    projected: list[int] = []

    def records(exporter, columns):
        projected.append(len(columns))
        return set_records(exporter, columns)

    monkeypatch.setattr(plan_export, "_set_records", records)
    record = _record("uux_ggg" if accuracy == "full" else "uux_ggg_nlc")
    model = loop_sm_compiled_model if accuracy == "full" else loop_sm_colour_model
    dipoles = _dipoles(range(1, 6))
    correlators = CorrelatorConfig.from_json_dict(
        {"color_correlations": dipoles, "spin_correlations": []}
    )
    widths = {}
    values = {}
    for adjoint in (False, True):
        monkeypatch.setattr(observables, "SELF_ADJOINT_HISTORIES", adjoint)
        root = tmp_path / str(adjoint)
        root.mkdir()
        card = root / "card.toml"
        card.write_text(
            _card(model, root / "artifact", [("p", "u u~ > g g g")], "", accuracy)
        )
        Generator(resolve_config(card)).generate(
            ProcessSet((ProcessRequest.parse("u u~ > g g g", name="p"),)),
            root / "artifact",
            correlators=correlators,
        )
        # The first projected set is the process's correlated set.
        widths[adjoint] = projected[0]
        projected.clear()
        runtime = Runtime.load(
            root / "artifact", process="p", model_parameters=record["parameters"]
        )
        requests = {h["id"]: CorrelatedRequest(h["id"]) for h in dipoles}
        out = []
        for poles in ("all", "uv", "ir", "none"):
            for arithmetic in ("double", "double-double"):
                results = runtime.evaluate_correlated_many(
                    record["points"][:1],
                    requests,
                    mu_r2=record["scales_squared"][:1],
                    poles=poles,
                    arithmetic=arithmetic,
                )
                for key in sorted(results):
                    result = results[key]
                    out.append([str(v) for point in result.values for v in point])
                    if poles in ("uv", "ir"):
                        out.append(repr(result.pole_parts))
        values[adjoint] = out
    # The Born metric and ten dipoles: 21 lanes, 11 with one column each.
    assert (widths[False], widths[True]) == (21, 11)
    assert values[True] == values[False]


# Disables pyAmpliCol's Symbolica library key in the native code
# (``pyamplicol::DISABLE_VARIABLE``, crate ``rust/crates/pyamplicol``).
NO_NATIVE_KEY = {"PYAMPLICOL_NATIVE_LIBRARY_KEY": "off"}


@pytest.mark.loop_physics
def test_binary64_observable_calls_touch_no_symbolica_state(stored, tmp_path):
    """The stored observable columns evaluate binary64 correlated,
    spin-correlated and resolved calls on the stored plane applications and
    exact rationals only: each runs in a fresh interpreter without a licence,
    with pyAmpliCol's native library key disabled, while another process
    holds the restricted Symbolica permit of a private lock directory (any
    Symbolica state access would abort), and gives the licensed call's bits."""

    import os

    lock = tmp_path / "symbolica-locks"
    lock.mkdir()
    lock.chmod(0o700)
    with portability.RestrictedPermit(lock_dir=str(lock)) as permit:
        assert permit.record["verified"], permit.record
        env = {
            key: value
            for key, value in os.environ.items()
            if key != "SYMBOLICA_LICENSE"
        }
        env = permit.environment({**env, **NO_NATIVE_KEY})
        names = ("many", "spin", "resolved")
        artifact, process = CALLS[names[0]][:2]
        assert all(CALLS[name][:2] == (artifact, process) for name in names)
        calls = [CALLS[name][2] for name in names]
        unlicensed = _calls(
            stored[artifact], process, calls, _record_path(artifact), env
        )
    licensed = _calls(stored[artifact], process, calls, _record_path(artifact))
    for call in calls:
        assert unlicensed[call]["calls"] == {}, (call, unlicensed[call]["calls"])
        assert unlicensed[call]["values"] == licensed[call]["values"], call
