# SPDX-License-Identifier: 0BSD
"""The SDK's plan v4 runtime behaves exactly like the Python API for model
parameters and scales (second genericity audit, G2-M2 and G2-m3).

* **Parameters (G2-M2).** On g g > t t~ (WT fixed to zero at generation, the
  v2 convention for a coloured width in tree currents), the Python API
  (``Runtime.load(..., model_parameters=...)``, ``set_model_parameters``),
  the native runtime (``V2Runtime`` through ``evaluate_v2_artifact``), the C
  ABI (a C client of ``rusticol_virtual_open`` / ``_update_parameters``,
  which is the Rust SDK's ``Runtime``) and the slice's own evaluation (the
  reference executor's twin, ``LoopSlice.evaluate``) refuse the same inputs
  with the same message: the fixed WT at another value, the renormalisation
  scale MU_R, an unknown name, a value that is not a finite decimal of
  ASCII text (a fraction, other digits, an infinity), an empty name (the
  Python API's ``TypeError``); a mapping with one refused entry changes
  nothing,
  and one with several is refused with the same message everywhere (values
  first, then names, by name). A complex value is the Python API's
  ``CompatibilityError`` (C status 3). The stored process's own executor
  applies the rule too, with the model's names from its record. An external
  colour-scheme constant that the colour declaration fixes (toy: loop_sm
  with ``lhv`` external) is in the fixed set of every client and of the
  slice, and its refusal says the model preparation fixed it. Accepted
  values (WT at its fixed zero,
  MT, aS, a parameter the process does not read) give bitwise the Python
  API's binary64 values in every client, at points exact in binary64 and on
  shell at their top mass.
  The C++ and Fortran wrappers compile against the typed errors
  (``VirtualError::status``, ``update_parameters(..., error_message=...)``)
  and report them; the native runtime and the Python API's native executor
  keep their tree and closure stages on their own thread pool
  (``threads=1`` starts no global rayon pool). The Rust SDK's own control
  (``sdk/tests/v4_parameters.rs``, its error kinds) runs on this file's
  artifact in ``test_the_rust_sdk_follows_the_python_api`` (nightly).
* **Scale-dependent batches (G2-m3).** Generation records which outputs of
  the parameter program depend on the scale (``native_trees._scale_outputs``).
  loop_sm's NLO couplings read MU_R in the counterterms only, so its Born
  couplings, constants and masses are scale independent and a batch of
  several scales runs as one; a toy running-coupling loop_sm
  (``loop_v2_ufo_variants.running_coupling_ufo``) lists its Born couplings
  (and the loop kernel constants they enter), the Python API and the native
  runtime partition a batch of two scales by
  mu_R^2 (every point bitwise its single-point value), and the slice's
  native executor (both input lanes) and the Python API's resolved calls,
  which take one batch as given, refuse it naming them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SDK = ROOT / "rust" / "loop-reduction" / "sdk"

pytestmark = pytest.mark.loop_physics

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

# g g > t t~ points exact in binary64 (dyadic components) and on shell at
# MT = 173 (loop_sm's default; E^2 - |p|^2 = 173^2 exactly) and MT = 150.
POINT_173 = [
    ["180.91015625", "0", "0", "180.91015625"],
    ["180.91015625", "0", "0", "-180.91015625"],
    ["180.91015625", "0", "31.74609375", "42.328125"],
    ["180.91015625", "0", "-31.74609375", "-42.328125"],
]
POINT_150 = [
    ["250", "0", "0", "250"],
    ["250", "0", "0", "-250"],
    ["250", "0", "120", "160"],
    ["250", "0", "-120", "-160"],
]
# g g > g g, massless.
GG_POINT = [
    ["500", "0", "0", "500"],
    ["500", "0", "0", "-500"],
    ["500", "0", "300", "400"],
    ["500", "0", "-300", "-400"],
]


# Two points of one momentum configuration at two scales (one batch).
SCALES = ["1000000", "490000"]

# Refused updates and the Python API's message for each (the native runtime
# and the C ABI give the same text).
REFUSED = [
    (
        {"WT": "1.4915"},
        "parameter 'WT' was fixed to 0 at generation ([loop.preparation] "
        "fixed_parameters); regenerate the artifact to change it",
    ),
    (
        {"MU_R": "100"},
        "'MU_R' is the renormalisation scale: pass it per point as mu_r2",
    ),
    ({"WX": "1"}, "unknown external parameter 'WX'"),
    # One refused entry refuses the mapping: MT is not applied.
    ({"MT": "150", "WX": "1"}, "unknown external parameter 'WX'"),
    # Several refused entries: every client checks them by name.
    (
        {"WT": "1.4915", "MU_R": "100"},
        "'MU_R' is the renormalisation scale: pass it per point as mu_r2",
    ),
    (
        {"WX": "1", "MU_R": "100"},
        "'MU_R' is the renormalisation scale: pass it per point as mu_r2",
    ),
    # Values are decimal numbers of ASCII text, read before the names (no
    # fractions, no other digits: Arabic-Indic 173 here).
    ({"aS": "59/500"}, "parameter 'aS' real part is not a decimal number"),
    (
        {"MT": "\u0661\u0667\u0663"},
        "parameter 'MT' real part is not a decimal number",
    ),
    ({"WX": "1", "aS": "Infinity"}, "parameter 'aS' real part must be finite"),
]

# An empty name: the Python API's TypeError (the C ABI's invalid argument,
# the Rust SDK's InvalidArgument), in the same words, before every value.
EMPTY_NAME = (
    {"": "1", "MT": "one"},
    "model parameter names must be nonempty strings",
)


def _card(
    model: Path,
    output: Path,
    expression: str,
    name: str,
    accuracy: str = "full",
    scheme: str = SCHEME,
) -> str:
    return f"""
action = "generate"
[model]
source = {json.dumps(str(model))}
[process]
entries = [{{expression={json.dumps(expression)}, name={json.dumps(name)}}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[color]
accuracy = "{accuracy}"
contraction = "direct"
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{MB = "0", WT = "0"}}
{scheme}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
samples = 1
"""


def _generate(
    model: Path, root: Path, expression: str, name: str, **options: str
) -> Path:
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    card = root / "card.toml"
    card.write_text(_card(model, root / "artifact", expression, name, **options))
    Generator(resolve_config(card)).generate(
        ProcessSet((ProcessRequest.parse(expression, name=name),)),
        root / "artifact",
    )
    return root / "artifact"


@pytest.fixture(scope="module")
def ggtt(loop_sm_compiled_model, loop_native, tmp_path_factory):
    """The plan v4 artifact of g g > t t~ with WT (and MB) fixed to zero."""

    return _generate(
        loop_sm_compiled_model, tmp_path_factory.mktemp("ggtt"), "g g > t t~", "ggtt"
    )


def _staged_sdk() -> Path:
    info = ROOT / ".artifacts" / "source-runtime" / "_build_info.json"
    if not info.is_file():
        pytest.fail(f"{info} missing: stage a native build first")
    build = Path(json.loads(info.read_text())["native_build"]["directory"])
    if not (build / "librusticol_loop_sdk.so").is_file():
        pytest.fail(f"{build} has no librusticol_loop_sdk.so")
    return build


def _compiler(variable: str, name: str) -> str:
    found = os.environ.get(variable) or shutil.which(name)
    if found is None:
        pytest.fail(f"no {name} on PATH for the SDK clients; set {variable}")
    return found


CLIENT = r"""
/* ./client artifact process points.txt open|- [update ...]: opens the
 * artifact with the open parameters (NULL for "-") and the default R2
 * strategy (NULL: "auto"), then applies each update in turn. Per step it
 * prints "<step> <status> <message>", then, while the handle is open, one
 * "values" line per point: the Born per Born order and the virtual per
 * sector and epsilon power as C99 hexadecimal literals. */
#include "rusticol_virtual.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void evaluate(rusticol_virtual_handle *runtime, size_t points, size_t legs,
                     const double *momenta, const double *scales) {
  rusticol_virtual_shape_info shape;
  rusticol_virtual_batch_request request = {momenta, points * legs * 4, points, NULL,
                                            scales, points, 0, NULL, NULL};
  rusticol_virtual_result *result = NULL;
  rusticol_virtual_result_shape_info info;
  if (rusticol_virtual_shape(runtime, &shape) != RUSTICOL_VIRTUAL_OK ||
      rusticol_virtual_evaluate(runtime, &request, &result) != RUSTICOL_VIRTUAL_OK ||
      rusticol_virtual_result_shape(result, &info) != RUSTICOL_VIRTUAL_OK) {
    printf("error %s\n", rusticol_virtual_last_error_message());
    exit(1);
  }
  double *born = malloc((info.born_values + 1) * sizeof(double));
  double *values = malloc((info.virtual_values + 1) * sizeof(double));
  if (!born || !values ||
      rusticol_virtual_result_copy_born(result, born, info.born_values) ||
      rusticol_virtual_result_copy_virtual(result, values, info.virtual_values)) {
    printf("error copy\n");
    exit(1);
  }
  for (size_t p = 0; p < info.successful_points; p++) {
    printf("values");
    for (size_t o = 0; o < shape.born_orders; o++)
      printf(" %a", born[p * shape.born_orders + o]);
    for (size_t s = 0; s < shape.virtual_orders * 3; s++)
      printf(" %a", values[p * shape.virtual_orders * 3 + s]);
    printf("\n");
  }
  free(born);
  free(values);
  rusticol_virtual_result_free(result);
}

int main(int argc, char **argv) {
  if (argc < 5) return 2;
  FILE *input = fopen(argv[3], "r");
  size_t points = 0, legs = 0;
  if (!input || fscanf(input, "%zu %zu", &points, &legs) != 2) return 2;
  double *momenta = malloc(points * legs * 4 * sizeof(double));
  double *scales = malloc(points * sizeof(double));
  for (size_t p = 0; p < points; p++) {
    for (size_t i = 0; i < legs * 4; i++)
      if (fscanf(input, "%la", &momenta[p * legs * 4 + i]) != 1) return 2;
    if (fscanf(input, "%la", &scales[p]) != 1) return 2;
  }
  fclose(input);
  rusticol_virtual_handle *runtime = NULL;
  const char *open = strcmp(argv[4], "-") ? argv[4] : NULL;
  int status = rusticol_virtual_open(argv[1], argv[2], open, NULL, &runtime);
  printf("open %d %s\n", status, status ? rusticol_virtual_last_error_message() : "");
  if (status) return 0;
  evaluate(runtime, points, legs, momenta, scales);
  for (int k = 5; k < argc; k++) {
    uint64_t revision = 0;
    status = rusticol_virtual_update_parameters(runtime, argv[k], &revision);
    printf("update %d %s\n", status,
           status ? rusticol_virtual_last_error_message() : "");
    evaluate(runtime, points, legs, momenta, scales);
  }
  rusticol_virtual_free(runtime);
  free(momenta);
  free(scales);
  return 0;
}
"""


@pytest.fixture(scope="module")
def c_client(tmp_path_factory):
    """The C client above, compiled against the staged build's loop SDK."""

    build = _staged_sdk()
    root = tmp_path_factory.mktemp("c-client")
    source = root / "client.c"
    source.write_text(CLIENT)
    subprocess.run(
        [
            _compiler("CC", "cc"),
            "-std=c11",
            "-O2",
            f"-I{SDK / 'include'}",
            str(source),
            f"-L{build}",
            "-lrusticol_loop_sdk",
            f"-Wl,-rpath,{build}",
            "-o",
            str(root / "client"),
        ],
        check=True,
    )
    return root / "client"


def _overrides(parameters: dict[str, str]) -> str:
    """The C ABI's exact parameter JSON: name -> [real, imaginary]."""

    return json.dumps({name: [value, "0"] for name, value in parameters.items()})


def _run_client(client, artifact, process, point, scales, steps, tmp_path):
    """The C client's steps: [(step, status, message, values per point)],
    the values as floats (bitwise the C doubles)."""

    feed = tmp_path / f"points-{len(list(tmp_path.iterdir()))}.txt"
    lines = [f"{len(scales)} {len(point)}"]
    for scale in scales:
        flat = [float(c).hex() for momentum in point for c in momentum]
        lines.append(" ".join([*flat, float(scale).hex()]))
    feed.write_text("\n".join(lines) + "\n")
    run = subprocess.run(
        [str(client), str(artifact), process, str(feed), *steps],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    out: list[list] = []
    for line in run.stdout.splitlines():
        kind, _, rest = line.partition(" ")
        if kind == "values":
            out[-1][3].append([float.fromhex(x) for x in rest.split()])
        else:
            assert kind in ("open", "update"), line
            status, _, message = rest.partition(" ")
            out.append([kind, int(status), message, []])
    return out


def _python_values(runtime, point, scales) -> list[list[float]]:
    result = runtime.evaluate([point] * len(scales), mu_r2=scales)
    return [
        [*born, *(c for laurent in virtual for c in laurent)]
        for born, virtual in zip(result.born_values, result.values, strict=True)
    ]


def _native_values(loop_native, artifact, process, point, scales, parameters):
    """``V2Runtime`` through ``evaluate_v2_artifact`` (no Python assembly)."""

    flat = [str(c) for momentum in point for c in momentum]
    found = loop_native.evaluate_v2_artifact(
        str(artifact),
        [flat] * len(scales),
        list(scales),
        process=process,
        parameters=parameters,
    )
    out = []
    for values, failures in found:
        assert not failures, failures
        born, virtual = values
        out.append([*born, *(c for laurent in virtual for c in laurent)])
    return out


def test_refused_parameters_are_the_python_apis_in_every_client(
    ggtt, loop_native, c_client, tmp_path
):
    from pyamplicol import Runtime

    runtime = Runtime.load(ggtt, process="ggtt")
    before = _python_values(runtime, POINT_173, SCALES)
    for parameters, message in REFUSED:
        # Python: at load and as an update; a refused update changes nothing.
        with pytest.raises(ValueError) as refused:
            Runtime.load(ggtt, process="ggtt", model_parameters=parameters)
        assert str(refused.value) == message
        with pytest.raises(ValueError) as refused:
            runtime.set_model_parameters(parameters)
        assert str(refused.value) == message
        assert _python_values(runtime, POINT_173, SCALES) == before, parameters
        # The native runtime: the same refusal (ValueError, the same text).
        with pytest.raises(ValueError) as refused:
            _native_values(loop_native, ggtt, "ggtt", POINT_173, SCALES, parameters)
        assert str(refused.value) == message
    # The C ABI: open refuses each mapping; on an open handle every update is
    # refused with the same status and text and leaves the values unchanged.
    for parameters, message in REFUSED:
        (opened,) = _run_client(
            c_client,
            ggtt,
            "ggtt",
            POINT_173,
            SCALES,
            [_overrides(parameters)],
            tmp_path,
        )
        assert opened[:3] == ["open", 1, message], opened
    steps = _run_client(
        c_client,
        ggtt,
        "ggtt",
        POINT_173,
        SCALES,
        ["-", *(_overrides(parameters) for parameters, _ in REFUSED)],
        tmp_path,
    )
    assert steps[0][:2] == ["open", 0]
    assert steps[0][3] == before
    for step, (_parameters, message) in zip(steps[1:], REFUSED, strict=True):
        assert step[:3] == ["update", 1, message], step
        assert step[3] == before
    # A complex value: the Python API's CompatibilityError, the C ABI's
    # runtime-error status (a Compatibility error, as "direct" R2), the same
    # text; it is checked after every value is read, before the names.
    from pyamplicol.api.errors import CompatibilityError

    for parameters in ({"aS": ("0.118", "0.1")}, {"WX": "1", "aS": ("0.118", "0.1")}):
        with pytest.raises(CompatibilityError) as refused:
            Runtime.load(ggtt, process="ggtt", model_parameters=parameters)
        assert str(refused.value) == (
            "parameter 'aS': v2 loop runtimes take real external parameters"
        )
        (opened,) = _run_client(
            c_client,
            ggtt,
            "ggtt",
            POINT_173,
            SCALES,
            [
                json.dumps(
                    {
                        name: list(value) if isinstance(value, tuple) else [value, "0"]
                        for name, value in parameters.items()
                    }
                )
            ],
            tmp_path,
        )
        assert opened[:3] == ["open", 3, str(refused.value)], opened


def test_an_empty_parameter_name_is_refused_in_every_client(
    ggtt, loop_native, c_client, tmp_path
):
    """An empty name is the Python API's ``TypeError`` (the stored view's
    and the slice's too), the C ABI's invalid argument (status 1) and the
    native runtime's ``ValueError`` (``evaluate_v2_artifact`` raises every
    typed refusal so), with the same text, before any value is read."""

    from pyamplicol import Runtime

    parameters, message = EMPTY_NAME
    with pytest.raises(TypeError) as refused:
        Runtime.load(ggtt, process="ggtt", model_parameters=parameters)
    assert str(refused.value) == message
    with pytest.raises(ValueError) as refused:
        _native_values(loop_native, ggtt, "ggtt", POINT_173, SCALES, parameters)
    assert str(refused.value) == message
    (opened,) = _run_client(
        c_client,
        ggtt,
        "ggtt",
        POINT_173,
        SCALES,
        [_overrides(parameters)],
        tmp_path,
    )
    assert opened[:3] == ["open", 1, message], opened
    process = _stored_process(ggtt, "ggtt")
    with pytest.raises(TypeError) as refused:
        process.executor(loop_native).evaluate(
            [POINT_173], SCALES[:1], parameters=parameters
        )
    assert str(refused.value) == message


def test_accepted_parameters_are_bitwise_the_python_apis_in_every_client(
    ggtt, loop_native, c_client, tmp_path
):
    from pyamplicol import Runtime

    # At MT = 173 (the default): the fixed WT at its fixed value (another
    # exact spelling), a parameter the process does not read (MH), aS.
    defaults = Runtime.load(ggtt, process="ggtt")
    cases = [{}, {"WT": "0.0"}, {"MH": "130"}, {"aS": "0.13"}]
    steps = _run_client(
        c_client,
        ggtt,
        "ggtt",
        POINT_173,
        SCALES,
        ["-", *(_overrides(case) for case in cases[1:])],
        tmp_path,
    )
    runtime = Runtime.load(ggtt, process="ggtt")
    applied: dict[str, str] = {}
    for case, step in zip(cases, steps, strict=True):
        assert step[1] == 0, step
        runtime.set_model_parameters(case)
        applied.update(case)
        python = _python_values(runtime, POINT_173, SCALES)
        assert step[3] == python, case
        native = _native_values(loop_native, ggtt, "ggtt", POINT_173, SCALES, applied)
        assert native == python, case
    # WT at zero and an unread parameter change nothing; aS does.
    default = _python_values(defaults, POINT_173, SCALES)
    assert [step[3] for step in steps[:3]] == [default] * 3
    assert steps[3][3] != default
    # MT = 150 at a point on shell there, from the open call.
    mass = {"MT": "150"}
    (opened,) = _run_client(
        c_client, ggtt, "ggtt", POINT_150, SCALES, [_overrides(mass)], tmp_path
    )
    assert opened[1] == 0, opened
    python = _python_values(
        Runtime.load(ggtt, process="ggtt", model_parameters=mass), POINT_150, SCALES
    )
    assert opened[3] == python
    assert _native_values(loop_native, ggtt, "ggtt", POINT_150, SCALES, mass) == python


def test_slice_evaluation_refuses_the_same_parameters(
    loop_sm_compiled_model, loop_native
):
    """The slice's own per-point evaluation (the reference executor's twin
    of the stored runtimes) applies the same rule."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    slice_ = LoopSlice.generate(
        compiled,
        SliceRequest(
            "g g > t t~",
            {"QCD": 2, "QED": 0},
            {"QCD": 4, "QED": 0},
            corrections=("QCD",),
            fixed_parameters={"MB": "0", "WT": "0"},
        ),
        loop_native,
    )
    assert slice_.scale_parameter == "MU_R"
    for parameters, message in REFUSED:
        with pytest.raises(ValueError) as refused:
            slice_.evaluate(POINT_173, SCALES[0], parameters=parameters)
        assert str(refused.value) == message
    accepted = slice_.evaluate(
        POINT_173, SCALES[0], parameters={"WT": "0", "MH": "130"}
    )
    reference = slice_.evaluate(POINT_173, SCALES[0])
    assert accepted.components == reference.components


def _stored_process(artifact: Path, process: str):
    """The stored native process of ``process`` (``native_process.NativeProcess``)."""

    from pyamplicol.loops.v2.artifact import read_native_process, read_plan

    (pacbin,) = artifact.rglob(f"{process}/loop-runtime.pacbin")
    return read_native_process(pacbin, read_plan(pacbin))


def test_stored_process_executor_refuses_the_same_parameters(
    ggtt, loop_sm_compiled_model, loop_native
):
    """The stored process's own executor (``NativeProcess.executor``, the
    benchmark's ``--load`` path; the public runtime validates first) applies
    the same rule: it refuses every mapping above with the Python API's
    text, given the model's external parameters or alone (the record holds
    the model's names), so a misspelt name is never dropped."""

    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    externals = {r.name for r in compiled.ir.parameters if r.nature == "external"}
    process = _stored_process(ggtt, "ggtt")
    # loop_sm's preparation fixes nothing besides the card's (same spelling).
    assert process.record["fixed_parameters"] == {"MB": "0", "WT": "0"}
    assert process.record["preparation_fixed_parameters"] == []
    assert process.record["externals"] == sorted(externals)
    executor = process.executor(loop_native, externals=externals)
    bare = process.executor(loop_native)
    for parameters, message in REFUSED:
        for runner in (executor, bare):
            with pytest.raises(ValueError) as refused:
                runner.evaluate([POINT_173], SCALES[:1], parameters=parameters)
            assert str(refused.value) == message
    (accepted,) = executor.evaluate(
        [POINT_173], SCALES[:1], parameters={"WT": "0", "MH": "130"}
    )
    (reference,) = executor.evaluate([POINT_173], SCALES[:1])
    assert accepted.components == reference.components


@pytest.fixture(scope="module")
def scheme_constant(loop_sm_ufo, loop_native, tmp_path_factory):
    """g g > g g at LC of loop_sm with its colour-scheme constant ``lhv``
    external (``loop_v2_ufo_variants.external_scheme_constant_ufo``) and the
    committed QCD colour declaration (``examples/loop/source_colour``),
    generated through the public API; and the compiled model."""

    import importlib.util

    from .loop_v2_ufo_variants import external_scheme_constant_ufo

    root = tmp_path_factory.mktemp("scheme")
    ufo = external_scheme_constant_ufo(loop_sm_ufo, root / "loop_sm_lhv")
    script = ROOT / "examples/loop/source_colour/prepare_model.py"
    spec = importlib.util.spec_from_file_location("_loop_colour_example", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.prepare(ufo, root / "loop-sm-lhv.json")
    return _generate(model, root, "g g > g g", "gg", accuracy="lc", scheme=""), model


LHV_FIXED = (
    "parameter 'lhv' was fixed to 1 at generation by the model preparation "
    "(a scheme constant, e.g. of the colour declaration); it is not a "
    "run-time parameter"
)


def test_a_scheme_constant_the_colour_declaration_fixes_is_refused(
    scheme_constant, loop_native, c_client, tmp_path
):
    """An external colour-scheme constant is fixed at generation by the
    colour declaration (``source_color_fixed_parameters``), not by the card:
    the native record holds it with the card's fixed parameters
    (``LoopSlice.fixed_parameters``; ``preparation_fixed_parameters`` names
    it, so the refusal does not point to the card), and the Python API, the
    native runtime and the C ABI refuse it at another value (before, they
    accepted lhv = 2 and evaluated trees whose counterterm colour
    continuation was built at lhv = 1) and accept it at its value, bitwise
    unchanged."""

    from pyamplicol import Runtime

    artifact, _model = scheme_constant
    record = _stored_process(artifact, "gg").record
    assert record["fixed_parameters"] == {"MB": "0", "WT": "0", "lhv": "1"}
    assert record["preparation_fixed_parameters"] == ["lhv"]
    runtime = Runtime.load(artifact, process="gg")
    before = _python_values(runtime, GG_POINT, SCALES)
    refused = {"lhv": "2"}
    with pytest.raises(ValueError) as error:
        Runtime.load(artifact, process="gg", model_parameters=refused)
    assert str(error.value) == LHV_FIXED
    with pytest.raises(ValueError) as error:
        runtime.set_model_parameters(refused)
    assert str(error.value) == LHV_FIXED
    with pytest.raises(ValueError) as error:
        _native_values(loop_native, artifact, "gg", GG_POINT, SCALES, refused)
    assert str(error.value) == LHV_FIXED
    runtime.set_model_parameters({"lhv": "1.0"})
    assert _python_values(runtime, GG_POINT, SCALES) == before
    native = _native_values(loop_native, artifact, "gg", GG_POINT, SCALES, None)
    assert native == before
    assert (
        _native_values(loop_native, artifact, "gg", GG_POINT, SCALES, {"lhv": "1.0"})
        == before
    )
    (opened,) = _run_client(
        c_client, artifact, "gg", GG_POINT, SCALES, [_overrides(refused)], tmp_path
    )
    assert opened[:3] == ["open", 1, LHV_FIXED], opened
    steps = _run_client(
        c_client,
        artifact,
        "gg",
        GG_POINT,
        SCALES,
        ["-", _overrides(refused), _overrides({"lhv": "1.0"})],
        tmp_path,
    )
    assert [step[:3] for step in steps] == [
        ["open", 0, ""],
        ["update", 1, LHV_FIXED],
        ["update", 0, ""],
    ]
    assert [step[3] for step in steps] == [before] * 3


def test_the_slice_fixes_the_scheme_constant_its_preparation_fixes(
    scheme_constant, loop_native
):
    """The slice (the reference executor's twin) applies the same fixed set:
    its ``fixed_parameters`` hold the declaration's lhv besides the request's,
    lhv = 2 is refused with the same text and lhv = 1.0 changes nothing."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import (
        resolve_loop_color_declaration,
    )

    _artifact, model = scheme_constant
    compiled = load_compiled_model(model)
    slice_ = LoopSlice.generate(
        compiled,
        SliceRequest(
            "g g > g g",
            {"QCD": 2, "QED": 0},
            {"QCD": 4, "QED": 0},
            fixed_parameters={"MB": "0", "WT": "0"},
            accuracy="lc",
            colour_declaration=resolve_loop_color_declaration(compiled.raw_loop_model),
        ),
        loop_native,
    )
    assert slice_.fixed_parameters == {"MB": "0", "WT": "0", "lhv": "1"}
    assert slice_.preparation_fixed_parameters == {"lhv"}
    with pytest.raises(ValueError) as error:
        slice_.evaluate(GG_POINT, SCALES[0], parameters={"lhv": "2"})
    assert str(error.value) == LHV_FIXED
    accepted = slice_.evaluate(GG_POINT, SCALES[0], parameters={"lhv": "1.0"})
    assert accepted.components == slice_.evaluate(GG_POINT, SCALES[0]).components


def test_cxx_and_fortran_clients_report_the_typed_refusal(ggtt, tmp_path):
    """The C++ wrapper's ``VirtualError`` carries the C status and the
    Fortran ``update_parameters`` returns the refusal's text: both wrappers
    and their clients compile against the staged loop SDK and, on g g > t t~,
    report setting MU_R as RUSTICOL_VIRTUAL_INVALID_ARGUMENT with the Python
    API's text."""

    build = _staged_sdk()
    cxx = tmp_path / "client.cpp"
    cxx.write_text(
        """
#include "rusticol_virtual.hpp"
#include <iostream>
int main(int argc, char **argv) {
    if (argc < 3) return 2;
    rusticol::VirtualRuntime runtime(argv[1], argv[2]);
    try {
        runtime.update_parameters("{\\"MU_R\\": [\\"100\\", \\"0\\"]}");
    } catch (const rusticol::VirtualError &error) {
        std::cout << error.status() << " " << error.what() << "\\n";
        return error.status() == RUSTICOL_VIRTUAL_INVALID_ARGUMENT ? 0 : 1;
    }
    return 1;
}
"""
    )
    subprocess.run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            f"-I{SDK / 'include'}",
            str(cxx),
            f"-L{build}",
            "-lrusticol_loop_sdk",
            f"-Wl,-rpath,{build}",
            "-o",
            str(tmp_path / "client-cpp"),
        ],
        check=True,
    )
    fortran = tmp_path / "client.f90"
    fortran.write_text(
        """
program client
  use, intrinsic :: iso_c_binding
  use rusticol_virtual
  implicit none
  type(virtual_runtime) :: runtime
  integer(c_int) :: ierr
  integer(c_int64_t) :: revision
  character(:), allocatable :: message
  character(len=4096) :: path, process
  call get_command_argument(1, path)
  call get_command_argument(2, process)
  call runtime%load(trim(path), trim(process), ierr=ierr)
  if (ierr /= 0) stop 2
  call runtime%update_parameters('{"MU_R": ["100", "0"]}', revision, ierr=ierr, &
    error_message=message)
  print '(i0,1x,a)', ierr, message
  if (ierr /= 1) stop 1
end program
"""
    )
    gfortran = _compiler("FC", "gfortran")
    subprocess.run(
        [
            gfortran,
            "-std=f2018",
            "-c",
            str(SDK / "fortran" / "rusticol_virtual.f90"),
            "-J",
            str(tmp_path),
            "-o",
            str(tmp_path / "rusticol_virtual.o"),
        ],
        check=True,
    )
    subprocess.run(
        [
            gfortran,
            "-std=f2018",
            f"-I{tmp_path}",
            str(fortran),
            str(tmp_path / "rusticol_virtual.o"),
            f"-L{build}",
            "-lrusticol_loop_sdk",
            f"-Wl,-rpath,{build}",
            "-o",
            str(tmp_path / "client-fortran"),
        ],
        check=True,
    )
    message = "'MU_R' is the renormalisation scale: pass it per point as mu_r2"
    for client in ("client-cpp", "client-fortran"):
        run = subprocess.run(
            [str(tmp_path / client), str(ggtt), "ggtt"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert run.returncode == 0, (client, run.stdout, run.stderr)
        assert run.stdout.strip() == f"1 {message}", (client, run.stdout)


# ------------------------------------------------------------ scale batches


def _compiled_variant(ufo: Path, root: Path) -> Path:
    """A loop_sm variant compiled as ``loop_sm_compiled_model`` is (the
    structural scalar-arity view), written next to it."""

    from dataclasses import replace

    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    model = compile_model_source(ufo, cache_dir=root / "model-cache")
    model = replace(
        model,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            model.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    return model.write(root / f"{ufo.name}.json")


@pytest.fixture(scope="module")
def running(loop_sm_ufo, loop_native, tmp_path_factory):
    """g g > g g of the toy running-coupling loop_sm, generated through the
    public API."""

    from .loop_v2_ufo_variants import running_coupling_ufo

    root = tmp_path_factory.mktemp("running")
    path = _compiled_variant(
        running_coupling_ufo(loop_sm_ufo, root / "loop_sm_running"), root
    )
    return _generate(path, root, "g g > g g", "gg"), path


def test_a_renamed_scale_parameter_is_refused_by_its_role(
    loop_sm_ufo, loop_native, tmp_path
):
    """The scale parameter is the one generation resolves (UFO/SLHA code
    LOOP 1), not a name: on the renamed-model harness's ``scale`` variant
    (``MU_R`` -> ``muR``, same code) the Python API and the native runtime
    refuse ``muR`` as the renormalisation scale and ``MU_R`` as unknown."""

    from pyamplicol import Runtime

    from .loop_renamed_models import write_variant

    write_variant(loop_sm_ufo, tmp_path / "loop_sm_scale", "scale")
    path = _compiled_variant(tmp_path / "loop_sm_scale", tmp_path)
    artifact = _generate(path, tmp_path, "g g > g g", "gg")
    runtime = Runtime.load(artifact, process="gg")
    for parameters, message in (
        (
            {"muR": "100"},
            "'muR' is the renormalisation scale: pass it per point as mu_r2",
        ),
        ({"MU_R": "100"}, "unknown external parameter 'MU_R'"),
    ):
        with pytest.raises(ValueError) as refused:
            runtime.set_model_parameters(parameters)
        assert str(refused.value) == message
        with pytest.raises(ValueError) as refused:
            _native_values(loop_native, artifact, "gg", GG_POINT, SCALES, parameters)
        assert str(refused.value) == message


def _trees(artifact: Path, process: str):
    """The stored tree program of ``process`` (``native_trees.NativeTrees``)."""

    return _stored_process(artifact, process).trees


def test_nlo_couplings_read_the_scale_in_the_counterterms_only(ggtt):
    from pyamplicol.loops.v2.native_trees import scale_dependent_shared

    trees = _trees(ggtt, "ggtt")
    record = trees.program["parameters"]
    names = [name for _o, name in record["scale_outputs"]]
    # Named as the plane check names outputs (``_output_label``): counterterm
    # couplings and coefficients (and their diagnostics) only, none of the
    # trees' model couplings, Born couplings, kernel constants or masses.
    assert names and not any(
        name.startswith(("model ", "Born coupling", "kernel constant", "mass "))
        for name in names
    ), names
    assert any(name.startswith("ct-model coupling") for name in names), names
    assert record["names"][record["scale"]] == "MU_R"
    shared = {*record["couplings"], *record["constants"]}
    assert not shared & {o for o, _name in record["scale_outputs"]}
    assert scale_dependent_shared(trees) == ()


def test_running_coupling_batches_are_partitioned_by_scale(running, loop_native):
    from pyamplicol import Runtime
    from pyamplicol.loops.v2.native_trees import scale_dependent_shared

    artifact, _model = running
    dependent = scale_dependent_shared(_trees(artifact, "gg"))
    # The Born couplings run (and the loop kernels' constants with them).
    assert any(name.startswith("Born coupling") for name in dependent), dependent
    assert all(
        name.startswith(("Born coupling", "kernel constant")) for name in dependent
    )
    # Every one by its key (coupling, or the constant's value form), none by
    # an output index alone.
    assert not any("(output" in name for name in dependent), dependent
    runtime = Runtime.load(artifact, process="gg")
    batch = _python_values(runtime, GG_POINT, SCALES)
    singles = [_python_values(runtime, GG_POINT, [scale])[0] for scale in SCALES]
    assert batch == singles
    assert batch[0] != batch[1]
    native = _native_values(loop_native, artifact, "gg", GG_POINT, SCALES, None)
    assert native == batch
    # Interleaved scales come back in the caller's order.
    mixed = SCALES + SCALES[::-1]
    assert _python_values(runtime, GG_POINT, mixed) == [*singles, *singles[::-1]]
    assert _native_values(loop_native, artifact, "gg", GG_POINT, mixed, None) == [
        *singles,
        *singles[::-1],
    ]


def test_a_mixed_scale_batch_of_a_running_coupling_is_refused_by_name(
    running, loop_native
):
    """The slice's native executor evaluates one batch as given: a batch of
    two scales of the running-coupling model is refused, naming the Born
    couplings that depend on the scale (typed, not a per-point mismatch)."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.models.loading import load_compiled_model

    artifact, model = running
    slice_ = LoopSlice.generate(
        load_compiled_model(model),
        SliceRequest(
            "g g > g g",
            {"QCD": 2, "QED": 0},
            {"QCD": 4, "QED": 0},
            fixed_parameters={"MB": "0"},
        ),
        loop_native,
    )
    assert slice_.request.corrections == ("QCD",)
    executor = NativeExecutor(slice_, inputs="native")
    with pytest.raises(
        ValueError,
        match=r"Born coupling GC_\d+.*depend on the "
        r"renormalisation scale MU_R: a batch takes one mu_R\^2",
    ):
        executor.evaluate([GG_POINT, GG_POINT], SCALES)
    one = executor.evaluate([GG_POINT], SCALES[:1])
    assert len(one) == 1
    # The Python-input lane (the oracle) names the values that differ.
    oracle = NativeExecutor(slice_, inputs="python")
    with pytest.raises(
        ValueError,
        match=r"kernel constant .* depend on the renormalisation scale: the "
        r"points of one batch need equal kernel constants and masses",
    ) as refused:
        oracle.evaluate([GG_POINT, GG_POINT], SCALES)
    # Named as the native record names them (``_output_label``: the value's
    # form), not by the key's repr.
    assert "'param'" not in str(refused.value), str(refused.value)
    # So do the Python API's colour-correlated and helicity-resolved calls,
    # which take one scale per call (``docs/user/configuration.md``).
    from pyamplicol import Runtime

    runtime = Runtime.load(artifact, process="gg")
    with pytest.raises(ValueError, match=r"Born coupling GC_\d+.*depend on the"):
        runtime.evaluate_resolved([GG_POINT, GG_POINT], mu_r2=SCALES)


# ------------------------------------------------- request and policy defaults


def _particle(name, *, spin=3, color=1):
    from types import SimpleNamespace

    return SimpleNamespace(
        name=name,
        antiname=name,
        auxiliary_kind=None,
        spin=spin,
        color=color,
        charge=0.0,
        mass="ZERO",
        width="ZERO",
        ghost_number=0,
        goldstoneboson=False,
    )


def test_slice_request_corrections_default_to_the_strong_correction():
    """``SliceRequest.corrections`` has no order-name default (second
    genericity audit, m7): unset, it is the model's strong correction, the
    order the gluon's minimal vertices carry whatever its name; a model
    without one is refused with a typed error; explicit corrections stand."""

    from dataclasses import replace
    from types import SimpleNamespace

    from pyamplicol.api.errors import GenerationError
    from pyamplicol.loops.v2.assemble import SliceRequest, resolve_request

    def model(*vertices):
        return SimpleNamespace(
            ir=SimpleNamespace(
                particles=(_particle("g", color=8), _particle("a")),
                parameters=(),
                vertex_terms=tuple(
                    SimpleNamespace(particles=particles, coupling_orders=orders)
                    for particles, orders in vertices
                ),
            )
        )

    request = SliceRequest("g g > g g", {"GS": 2}, {"GS": 4})
    assert request.corrections is None
    strong = model((("g", "g", "g"), (("GS", 1),)))
    assert resolve_request(strong, request).corrections == ("GS",)
    with pytest.raises(GenerationError, match="names no correction orders"):
        resolve_request(model(), request)
    explicit = replace(request, corrections=("GS", "EW"))
    assert resolve_request(model(), explicit) is explicit


def test_the_cheapest_frame_policy_requires_the_legs_helicity_rows():
    """No count per leg is assumed (second genericity audit, m8): the native
    generation's cheapest frame policy refuses a call without ``leg_rows``
    (the Rust policy refuses the payload, ``schedules.rs``), and
    ``assemble._FRAME_POLICY`` is gone."""

    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.generator_input import NativeGeneration
    from pyamplicol.loops.v2.schedules import STRUCTURED

    assert not hasattr(assemble, "_FRAME_POLICY")
    generation = NativeGeneration.__new__(NativeGeneration)
    generation.legs = (1, 2, 3, 4)
    with pytest.raises(ValueError, match="needs the helicity rows of every leg"):
        generation._policy(STRUCTURED, None, [], None)
    policy = generation._policy(STRUCTURED, None, [], (2, 2, 1, 3))
    assert policy["kind"] == "cheapest" and policy["leg_rows"] == [2, 2, 1, 3]


THREADS_SCRIPT = r"""
import sys
import time
from pyamplicol.loops.native import load_loop_native

def threads():
    with open("/proc/self/status") as status:
        return int(next(l for l in status if l.startswith("Threads:")).split()[1])

artifact, point = sys.argv[1], sys.argv[2:]
native = load_loop_native()
before = threads()
native.evaluate_v2_artifact(artifact, [point] * 32, ["1000000"] * 32, threads=1)
# The call's own pool is dropped on return, but rayon does not join its
# worker: wait (bounded) for it to exit. A global pool's workers never do.
deadline = time.monotonic() + 30
while threads() > before and time.monotonic() < deadline:
    time.sleep(0.05)
print(before, threads())
"""


@pytest.mark.skipif(not Path("/proc/self/status").is_file(), reason="needs procfs")
def test_the_native_runtime_honours_its_thread_bound(ggtt, tmp_path):
    """``V2Runtime``'s tree and closure stages run on the executor's pool
    (its ``threads`` bound), not on rayon's global pool: an evaluation with
    ``threads=1`` in a fresh interpreter starts no global pool (whose workers
    would outlive the call; before, eight with ``RAYON_NUM_THREADS=8``). The
    call's own one-thread pool is dropped on return; its worker exits shortly
    after (rayon signals, it does not join), so the count is read once it
    settles (bounded wait)."""

    import sys

    script = tmp_path / "threads.py"
    script.write_text(THREADS_SCRIPT)
    point = [c for momentum in POINT_173 for c in momentum]
    run = subprocess.run(
        [sys.executable, str(script), str(ggtt), *point],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "RAYON_NUM_THREADS": "8"},
        check=False,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    before, after = map(int, run.stdout.split()[-2:])
    assert after == before, (before, after)


EXECUTOR_THREADS_SCRIPT = r"""
import sys
import time
from pathlib import Path
from pyamplicol.loops.native import load_loop_native
from pyamplicol.loops.v2.artifact import read_native_process, read_plan

def threads():
    with open("/proc/self/status") as status:
        return int(next(l for l in status if l.startswith("Threads:")).split()[1])

artifact, point = Path(sys.argv[1]), sys.argv[2:]
(pacbin,) = artifact.rglob("ggtt/loop-runtime.pacbin")
process = read_native_process(pacbin, read_plan(pacbin))
native = load_loop_native()
before = threads()
executor = process.executor(native, threads=1)
momenta = [[point[4 * i : 4 * i + 4] for i in range(4)]] * 32
executor.evaluate(momenta, ["1000000"] * 32)
deadline = time.monotonic() + 30
while threads() > before + 1 and time.monotonic() < deadline:
    time.sleep(0.05)
print(before, threads())
"""


@pytest.mark.skipif(not Path("/proc/self/status").is_file(), reason="needs procfs")
def test_the_python_apis_native_executor_honours_its_thread_bound(ggtt, tmp_path):
    """The Python API's native executor (``NativeProcess.executor(native,
    threads=1)``, the stored runtime and the benchmark's ``--load`` path)
    binds its batch and runs its tree and closure stages on its own pool
    (``V2Executor``'s, one worker here), as the SDK runtime does: with
    ``RAYON_NUM_THREADS=8`` an evaluation adds at most that worker (before,
    rayon's global pool too: 2 -> 11 threads)."""

    import sys

    script = tmp_path / "executor_threads.py"
    script.write_text(EXECUTOR_THREADS_SCRIPT)
    point = [c for momentum in POINT_173 for c in momentum]
    run = subprocess.run(
        [sys.executable, str(script), str(ggtt), *point],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "RAYON_NUM_THREADS": "8"},
        check=False,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    before, after = map(int, run.stdout.split()[-2:])
    assert after <= before + 1, (before, after)


@pytest.mark.loop_nightly
def test_the_rust_sdk_follows_the_python_api(ggtt):
    """The Rust SDK's own control (``sdk/tests/v4_parameters.rs``: the same
    refusals with their error kinds, ``ModelParameter``, ``InvalidArgument``
    and ``Compatibility``, atomic updates, "auto" and "direct" R2) on this
    file's g g > t t~ artifact. It builds the SDK's test in release into
    ``PYAMPLICOL_SDK_CARGO_TARGET_DIR`` (a private target directory: about
    20 min cold, seconds on one that holds a native build made with
    ``tools/developer/build_loop_native.py``, whose release settings it
    uses); it is not run without one."""

    from tools.developer.build_loop_native import _CODEGEN_DEFAULTS

    target = os.environ.get("PYAMPLICOL_SDK_CARGO_TARGET_DIR")
    if not target:
        pytest.skip(
            "opt-in: set PYAMPLICOL_SDK_CARGO_TARGET_DIR to a private cargo "
            "target directory to build and run the Rust SDK's control"
        )
    shared = os.environ.get("CARGO_TARGET_DIR")
    if shared and Path(shared).resolve() == Path(target).resolve():
        pytest.fail("PYAMPLICOL_SDK_CARGO_TARGET_DIR is the shared CARGO_TARGET_DIR")
    env = {k: v for k, v in os.environ.items() if k != "CARGO_TARGET_DIR"}
    env = {**_CODEGEN_DEFAULTS, **env, "PYAMPLICOL_V4_GGTT_ARTIFACT": str(ggtt)}
    run = subprocess.run(
        [
            "cargo",
            "test",
            "--locked",
            "--release",
            "--target-dir",
            target,
            "-p",
            "rusticol-loop-sdk",
            "--test",
            "v4_parameters",
            "--",
            "--ignored",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
        check=False,
    )
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
    assert "test result: ok. 1 passed" in run.stdout, run.stdout[-4000:]
