# SPDX-License-Identifier: 0BSD
"""A stored (prepared) plan v4 process regenerates nothing on any call.

REQUIREMENTS_ADDENDUM G / HANDOFF §0.2 (d): loading an artifact never
regenerates the process, and neither does any call of the API contract on
the loaded process. Each call kind runs in a fresh interpreter under the
generation sentinel of ``tools/loop_artifact_portability.py``
(``GenerationSentinel``: every call of slice, plan, tree-program or model
generation code is counted from before the first pyAmpliCol import to the
end of the call), and must count zero:

* the inclusive virtual in every pole selection (``all``, ``none``, ``uv``,
  ``ir``: the stored companion of the certified-scaleless families), with
  rotation and reduction-direction stability checks, in double-double, after
  ``set_model_parameters``, and its per-coupling-order partials;
* an exact flavour member served by its representative;
* colour-correlated, spin-correlated and helicity-resolved outputs and the
  correlation catalogue.

Every call's binary64 values in the fresh interpreter equal those of the
generating test process's own load (bitwise).
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

# Correlated and resolved calls still regenerate the slice on first use: their
# columns are not stored natively yet.
OBSERVABLE_COLUMNS = (
    "native observable columns are not stored yet (generation-scale-2 "
    "continuation): correlated and resolved calls regenerate the slice"
)


def _record(name: str) -> dict:
    return json.loads((OBSERVABLES / f"{name}.json").read_text())


def _card(model: Path, output: Path, entries, process: str = "") -> str:
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
accuracy = "full"
contraction = "direct"
[loop]
r2_strategy = "model"
engine = "v2"
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


@pytest.fixture(scope="module")
def stored(loop_sm_compiled_model, loop_native, tmp_path_factory):
    """The artifacts of the g g > g g observables record (colour histories
    and spin classes declared) and of d d~ > g g with its exact flavour
    member s s~ > g g, generated through the public API."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.correlators import CorrelatorConfig

    out = {}
    for name, process in (("gg_gg", ""), ("ddgg_ssgg", "flavor_blind = true")):
        record = _record(name)
        entries = tuple((key, expression) for key, expression in record["processes"])
        root = tmp_path_factory.mktemp(name)
        card = root / "card.toml"
        card.write_text(
            _card(loop_sm_compiled_model, root / "artifact", entries, process)
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
        out[name] = root / "artifact"
    return out


# One call kind per case: (artifact, process, call).
CALLS = {
    "all": ("gg_gg", "gg", "evaluate"),
    "none": ("gg_gg", "gg", "evaluate-none"),
    "uv": ("gg_gg", "gg", "evaluate-uv"),
    "ir": ("gg_gg", "gg", "evaluate-ir"),
    "stability": ("gg_gg", "gg", "evaluate-stability"),
    "double-double": ("gg_gg", "gg", "evaluate-double-double"),
    "parameters": ("gg_gg", "gg", "evaluate-parameters"),
    "orders": ("gg_gg", "gg", "evaluate-orders"),
    "member": ("ddgg_ssgg", "ssgg", "evaluate"),
    "colour": ("gg_gg", "gg", "correlated-colour"),
    "spin": ("gg_gg", "gg", "correlated-spin"),
    "many": ("gg_gg", "gg", "correlated-many"),
    "resolved": ("gg_gg", "gg", "resolved"),
    "catalogue": ("gg_gg", "gg", "catalogue"),
}
OBSERVABLE_CALLS = ("colour", "spin", "many", "resolved", "catalogue")

# Runs one call kind (argv: artifact, process or "", call, record) under the
# generation sentinel and prints its generation calls and values (binary64
# bits as hexadecimal literals).
_PROBE = r"""
import json, sys
from pathlib import Path
spec_path, artifact, process, call, record_path = sys.argv[1:6]
import importlib.util
spec = importlib.util.spec_from_file_location("portability", spec_path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
record = json.loads(Path(record_path).read_text())
process = process or None
with tool.GenerationSentinel() as sentinel:
    from pyamplicol import Runtime
    from pyamplicol.api import VirtualStabilityChecks
    from pyamplicol.correlators import CorrelatedRequest
    from decimal import Decimal
    values = run(Runtime, artifact, process, call, record)
print(json.dumps({"calls": dict(sentinel.calls), "values": values}))
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
    elif call == "correlated-spin":
        result = runtime.evaluate_correlated_many(
            points, {"spin": CorrelatedRequest("born", spin)}, mu_r2=scales
        )["spin"]
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
    elif call == "catalogue":
        return [item.id for item in runtime.available_color_correlations()]
    else:
        raise ValueError(call)
    return [bits(result.born_values), bits(result.values)]
"""


def _call(artifact: Path, process: str | None, call: str, record: Path) -> dict:
    script = _RUN + _PROBE
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT / "tools" / "loop_artifact_portability.py"),
            str(artifact),
            process or "",
            call,
            str(record),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


def _case(name: str):
    marks = ()
    if name in OBSERVABLE_CALLS:
        marks = (pytest.mark.xfail(reason=OBSERVABLE_COLUMNS, strict=True),)
    return pytest.param(name, marks=(pytest.mark.loop_physics, *marks), id=name)


@pytest.mark.parametrize("name", [_case(name) for name in CALLS])
def test_stored_process_call_regenerates_nothing(stored, name):
    artifact, process, call = CALLS[name]
    fresh = _call(stored[artifact], process, call, OBSERVABLES / f"{artifact}.json")
    assert fresh["calls"] == {}, fresh["calls"]
    # The same call in this process (which generated the artifact) gives
    # the same bits.
    namespace: dict = {}
    exec(_RUN, namespace)  # noqa: S102 - the probe's own call helpers
    from pyamplicol import Runtime

    here = namespace["run"](
        Runtime, stored[artifact], process, call, _record(artifact)
    )
    assert json.loads(json.dumps(here)) == fresh["values"]


@pytest.mark.loop_physics
@pytest.mark.parametrize(("artifact", "process"), [("gg_gg", "gg"), ("ddgg_ssgg", "ddgg")])
def test_rust_runtime_equals_the_python_api(stored, loop_native, artifact, process):
    """The SDK's plan v4 runtime (``rusticol_loop_reduction::v2_runtime::
    V2Runtime``, through ``evaluate_v2_artifact``: no Python generation or
    assembly) gives bitwise the Python API's binary64 Born and virtual, with
    and without poles, at the default and at caller-set model parameters."""

    from pyamplicol import Runtime

    record = _record(artifact)
    points, scales = record["points"], record["scales_squared"]
    flat = [[str(x) for momentum in point for x in momentum] for point in points]
    for parameters in (record["parameters"], {**record["parameters"], "aS": "0.13"}):
        runtime = Runtime.load(
            stored[artifact], process=process, model_parameters=parameters
        )
        for poles in (True, False):
            python = runtime.evaluate(
                points, mu_r2=scales, poles="all" if poles else "none"
            )
            native = loop_native.evaluate_v2_artifact(
                str(stored[artifact]),
                flat,
                [str(s) for s in scales],
                process=process,
                parameters=parameters,
                poles=poles,
            )
            for index, (values, failures) in enumerate(native):
                assert not failures, failures
                born, virtual = values
                assert born == list(python.born_values[index])
                width = 3 if poles else 1
                assert virtual == [
                    list(laurent)[:width] for laurent in python.values[index]
                ], (index, poles)
