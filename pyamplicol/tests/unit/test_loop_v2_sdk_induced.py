# SPDX-License-Identifier: 0BSD
"""The native SDK serves loop-induced processes (m-lit LIT-SDK, PLAN §3.5,
§2.a.10; RULED Q36, Q41, Q8).

On the two LI-O1 rows (g g > h g g and u u~ > h g g, full colour, their
observables declared; ``loop_v2_induced_observables_rows``):

* an existing client opens the loop-induced artifact directly (no opt-in)
  and reads empty Born counts: zero Born orders and Born colour components
  in the shape, zero Born values in every result; the metadata names the
  kind and the interference (``contribution_kind`` "loop-induced",
  ``interference`` "conj(loop)*loop");
* every ordinary value equals the Python API's (``Runtime.evaluate``):
  binary64 bitwise (the executor's amplitudes and the counterterm kets added
  in Python's order, squared by the one native square of Q41), double-double
  and arbitrary precision equal as ``Decimal`` values (squared in
  ``v2_decimal`` in the canonical order); the PyO3 runtime
  (``evaluate_v2_artifact``, the same Rust code) as well;
* the pole selections "uv" and "ir" are refused in the Python API's words
  (its ``CompatibilityError``), and nothing changes;
* the reduction setter (``rusticol_virtual_set_reduction``) takes "auto",
  "ninjars" and "tirust" with the Python API's requested / effective /
  reason record (the one policy, ``reduction_policy``); an explicit TIRust
  call equals the Python API's TIRust call (bitwise where served, the same
  failed statuses and reasons where it fails closed); an unknown backend is
  an invalid argument.
"""

from __future__ import annotations

import ctypes
import json
import math
from decimal import Decimal
from pathlib import Path

import pytest

from tests.unit.loop_v2_induced_observables_rows import ROWS, generate
from tests.unit.test_loop_v2_induced import _fixture
from tests.unit.test_loop_v2_sdk_extras import (
    _Axes,
    _error,
    _Request,
    _ResultShape,
    _sdk,
)

pytestmark = pytest.mark.loop_physics

STATUS_INVALID_ARGUMENT = 1
STATUS_RUNTIME_ERROR = 3
SUCCESS_STATUS = 4  # RUSTICOL_VIRTUAL_ACCURACY_NOT_ESTABLISHED
FAILED_STATUS = 3


@pytest.fixture(scope="module")
def artifacts(loop_sm_ufo, loop_native, tmp_path_factory):
    return {
        name: generate(tmp_path_factory.mktemp(name), name, expression)
        for name, expression in ROWS.items()
    }


def _inputs(name: str, count: int = 2):
    """The fixture's first ``count`` points rounded to binary64 (the C ABI's
    inputs; the Python API reads their exact values) and their mu_R^2."""

    fixture = _fixture(name)
    points = [
        [[float(Decimal(c)) for c in momentum] for momentum in point]
        for point in fixture["points"][:count]
    ]
    scales = [float(Decimal(mu)) ** 2 for mu in fixture["mu_r"][:count]]
    return points, scales


def _exact(points, scales):
    return (
        [[[str(Decimal(c)) for c in p] for p in point] for point in points],
        [str(Decimal(s)) for s in scales],
    )


def _json(sdk, function, handle) -> dict:
    needed = ctypes.c_size_t()
    assert function(handle, None, ctypes.c_size_t(0), ctypes.byref(needed)) == 0
    buffer = ctypes.create_string_buffer(needed.value)
    assert function(handle, buffer, needed, ctypes.byref(needed)) == 0
    return json.loads(buffer.value)


def _open(sdk, artifact: Path):
    handle = ctypes.c_void_p()
    status = sdk.rusticol_virtual_open(
        str(artifact).encode(), b"p", None, None, ctypes.byref(handle)
    )
    assert status == 0, _error(sdk)
    return handle


def _evaluate(sdk, handle, points, scales) -> dict:
    """One ordinary C ABI evaluation: its shape, values, statuses, decimal
    record and diagnostics."""

    flat = [c for point in points for p in point for c in p]
    momenta = (ctypes.c_double * len(flat))(*flat)
    mu = (ctypes.c_double * len(scales))(*scales)
    request = _Request(momenta, len(flat), len(points), None, mu, len(scales))
    result = ctypes.c_void_p()
    status = sdk.rusticol_virtual_evaluate(
        handle, ctypes.byref(request), ctypes.byref(result)
    )
    assert status == 0, _error(sdk)
    try:
        shape = _ResultShape()
        assert sdk.rusticol_virtual_result_shape(result, ctypes.byref(shape)) == 0
        born = (ctypes.c_double * max(shape.born_values, 1))()
        loop = (ctypes.c_double * max(shape.virtual_values, 1))()
        statuses = (ctypes.c_uint32 * len(points))()
        assert (
            sdk.rusticol_virtual_result_copy_born(result, born, shape.born_values) == 0
        )
        assert (
            sdk.rusticol_virtual_result_copy_virtual(result, loop, shape.virtual_values)
            == 0
        )
        assert (
            sdk.rusticol_virtual_result_copy_statuses(result, statuses, len(points))
            == 0
        )
        return {
            "shape": shape,
            "virtual": list(loop)[: shape.virtual_values],
            "statuses": list(statuses),
            "decimal": _json(sdk, sdk.rusticol_virtual_result_decimal_json, result),
            "diagnostics": _json(
                sdk, sdk.rusticol_virtual_result_diagnostics_json, result
            ),
        }
    finally:
        sdk.rusticol_virtual_result_free(result)


def _python_flat(result) -> list:
    """The Python API's values in the SDK's layout (absent poles NaN)."""

    out = []
    for point in result.values:
        if point is None:
            continue
        for laurent in point:
            out.extend(math.nan if x is None else x for x in laurent)
    return out


def _same(found: list, expected: list) -> bool:
    return len(found) == len(expected) and all(
        (math.isnan(a) and math.isnan(float(b))) or a == b
        for a, b in zip(found, expected, strict=True)
    )


@pytest.mark.parametrize("name", sorted(ROWS))
def test_loop_induced_sdk_values_equal_the_python_api(artifacts, loop_native, name):
    from pyamplicol import Runtime

    artifact = artifacts[name]
    runtime = Runtime.load(artifact, process="p")
    points, scales = _inputs(name)
    exact_points, exact_scales = _exact(points, scales)
    sdk = _sdk()
    handle = _open(sdk, artifact)
    try:
        axes = _Axes()
        assert sdk.rusticol_virtual_shape(handle, ctypes.byref(axes)) == 0
        assert axes.born_orders == 0 and axes.born_color_components == 0
        assert axes.virtual_orders == len(
            runtime.evaluate(exact_points[:1], mu_r2=exact_scales[:1]).values[0]
        )
        metadata = _json(sdk, sdk.rusticol_virtual_metadata_json, handle)
        assert metadata["contribution_kind"] == "loop-induced"
        assert metadata["interference"] == "conj(loop)*loop"
        assert metadata["born_orders"] == []
        capabilities = _json(sdk, sdk.rusticol_virtual_capabilities_json, handle)
        assert capabilities["contribution_kind"] == "loop-induced"
        assert capabilities["pole_selections"] == ["all", "none"]
        for poles in ("all", "none"):
            assert sdk.rusticol_virtual_set_poles(handle, poles.encode()) == 0
            found = _evaluate(sdk, handle, points, scales)
            expected = runtime.evaluate(exact_points, mu_r2=exact_scales, poles=poles)
            assert found["shape"].born_values == 0
            assert found["statuses"] == [SUCCESS_STATUS] * len(points)
            assert _same(found["virtual"], _python_flat(expected)), (poles, name)
            assert found["decimal"] is None
            # The PyO3 runtime: the same Rust code.
            native = loop_native.evaluate_v2_artifact(
                str(artifact),
                [[c for p in point for c in p] for point in exact_points],
                exact_scales,
                process="p",
                poles=poles,
            )
            assert [failures for _values, failures in native] == [[], []]
            assert [values[0] for values, _f in native] == [[], []]
            flat = [
                x for values, _f in native for laurent in values[1] for x in laurent
            ]
            width = 3 if poles == "all" else 1
            assert flat == [
                x
                for point in expected.values
                for laurent in point
                for x in laurent[:width]
            ]
        # Extended carriers: the Python API's Decimal values, one point.
        assert sdk.rusticol_virtual_set_poles(handle, b"all") == 0
        for arithmetic, digits in (("double-double", 0), ("arbitrary", 32)):
            status = sdk.rusticol_virtual_set_arithmetic(
                handle, arithmetic.encode(), ctypes.c_uint32(digits)
            )
            assert status == 0, _error(sdk)
            found = _evaluate(sdk, handle, points[:1], scales[:1])
            options = {"arithmetic": arithmetic}
            if digits:
                options["precision"] = digits
            expected = runtime.evaluate(
                exact_points[:1], mu_r2=exact_scales[:1], **options
            )
            literals = found["decimal"]["virtual"]
            values = [x for laurent in expected.values[0] for x in laurent]
            assert found["decimal"]["born"] == []
            assert [Decimal(x) for x in literals] == [Decimal(x) for x in values]
            assert found["virtual"] == [float(Decimal(x)) for x in literals]
    finally:
        sdk.rusticol_virtual_free(handle)


def test_loop_induced_split_poles_are_refused_in_the_python_apis_words(artifacts):
    from pyamplicol import Runtime
    from pyamplicol.api.errors import CompatibilityError

    artifact = artifacts["li_gg_hgg"]
    runtime = Runtime.load(artifact, process="p")
    points, scales = _inputs("li_gg_hgg", 1)
    exact_points, exact_scales = _exact(points, scales)
    sdk = _sdk()
    handle = _open(sdk, artifact)
    try:
        for poles in ("uv", "ir"):
            with pytest.raises(CompatibilityError) as refused:
                runtime.evaluate(exact_points, mu_r2=exact_scales, poles=poles)
            status = sdk.rusticol_virtual_set_poles(handle, poles.encode())
            assert status == STATUS_RUNTIME_ERROR
            assert str(refused.value) in _error(sdk)
        # Nothing changed: the handle still evaluates every pole.
        found = _evaluate(sdk, handle, points, scales)
        assert found["diagnostics"]["poles"] == "all"
        assert len(found["virtual"]) == 3 * found["shape"].axes.virtual_orders
    finally:
        sdk.rusticol_virtual_free(handle)


@pytest.mark.parametrize("name", sorted(ROWS))
def test_the_reduction_setter_records_the_python_apis_choice(
    artifacts, loop_native, name
):
    from pyamplicol import Runtime

    artifact = artifacts[name]
    runtime = Runtime.load(artifact, process="p")
    points, scales = _inputs(name)
    exact_points, exact_scales = _exact(points, scales)
    tir = "tirust" in json.loads(loop_native.reduction_backends())
    sdk = _sdk()
    handle = _open(sdk, artifact)
    try:
        default = _evaluate(sdk, handle, points, scales)
        expected = runtime.evaluate(exact_points, mu_r2=exact_scales)
        record = expected.diagnostics[0].reduction
        assert default["diagnostics"]["reduction"] == {
            "requested": record.requested,
            "effective": record.effective,
            "reason": record.reason,
        }
        capabilities = _json(sdk, sdk.rusticol_virtual_capabilities_json, handle)
        # What the library serves, as the Python API's inspect lists it.
        assert capabilities["reduction"]["native"] == {
            backend: sorted(carriers)
            for backend, carriers in json.loads(
                loop_native.reduction_backends()
            ).items()
        }
        assert capabilities["reduction"]["auto_default"] == "tirust"
        for backend in ("auto", "ninjars", "tirust"):
            status = sdk.rusticol_virtual_set_reduction(handle, backend.encode())
            if backend == "tirust" and not tir:
                assert status == STATUS_RUNTIME_ERROR
                assert "not built into this native extension" in _error(sdk)
                continue
            assert status == 0, _error(sdk)
            found = _evaluate(sdk, handle, points, scales)
            expected = runtime.evaluate(
                exact_points,
                mu_r2=exact_scales,
                reduction=backend,
                on_failure="return",
            )
            record = expected.diagnostics[0].reduction
            assert found["diagnostics"]["reduction"] == {
                "requested": record.requested,
                "effective": record.effective,
                "reason": record.reason,
            }
            served = [value is not None for value in expected.values]
            assert found["statuses"] == [
                SUCCESS_STATUS if ok else FAILED_STATUS for ok in served
            ]
            assert _same(found["virtual"], _python_flat(expected)), backend
            failures = found["diagnostics"]["failures"]
            for index, ok in enumerate(served):
                if not ok:
                    assert "; ".join(failures[str(index)]) == (
                        expected.diagnostics[index].detail
                    )
        assert sdk.rusticol_virtual_set_reduction(handle, b"golem") == (
            STATUS_INVALID_ARGUMENT
        )
        assert "unknown reduction backend" in _error(sdk)
    finally:
        sdk.rusticol_virtual_free(handle)


@pytest.mark.parametrize(
    ("arithmetic", "digits"), [("double-double", 0), ("arbitrary", 32)]
)
def test_tirust_in_extended_precision_follows_the_capability_query(
    artifacts, loop_native, arithmetic, digits
):
    """DD/AP on TIRust (m-lit lit-tir-mp's carriers; DECISIONS 21:50 R2):
    where the native extension reports TIRust at the carrier
    (``reduction_backends()``), the SDK's TIRust values equal the Python
    API's TIRust values (``Decimal`` values; the same failed statuses where
    it fails closed); where it does not, both refuse an explicit TIRust call
    typed with the same message (the Python API's ``ReductionUnavailable``)
    and ``auto`` keeps NinjaRS with the same reason. Runs unchanged once
    lit-tir-mp's carriers land."""

    from pyamplicol import Runtime
    from pyamplicol.api.errors import ReductionUnavailable

    served = json.loads(loop_native.reduction_backends())
    if "tirust" not in served:
        pytest.skip("this native extension is built without TIRust")
    name = "li_gg_hgg"
    artifact = artifacts[name]
    runtime = Runtime.load(artifact, process="p")
    points, scales = _inputs(name, 1)
    exact_points, exact_scales = _exact(points, scales)
    options = {"arithmetic": arithmetic, **({"precision": digits} if digits else {})}
    sdk = _sdk()
    handle = _open(sdk, artifact)
    try:
        status = sdk.rusticol_virtual_set_arithmetic(
            handle, arithmetic.encode(), ctypes.c_uint32(digits)
        )
        assert status == 0, _error(sdk)
        assert sdk.rusticol_virtual_set_reduction(handle, b"tirust") == 0
        if arithmetic not in served["tirust"]:
            with pytest.raises(ReductionUnavailable) as refused:
                runtime.evaluate(
                    exact_points, mu_r2=exact_scales, reduction="tirust", **options
                )
            flat = [c for point in points for p in point for c in p]
            momenta = (ctypes.c_double * len(flat))(*flat)
            mu = (ctypes.c_double * 1)(*scales)
            request = _Request(momenta, len(flat), 1, None, mu, 1)
            result = ctypes.c_void_p()
            status = sdk.rusticol_virtual_evaluate(
                handle, ctypes.byref(request), ctypes.byref(result)
            )
            assert status == STATUS_RUNTIME_ERROR
            assert str(refused.value) in _error(sdk)
            assert sdk.rusticol_virtual_set_reduction(handle, b"auto") == 0
            found = _evaluate(sdk, handle, points, scales)
            expected = runtime.evaluate(exact_points, mu_r2=exact_scales, **options)
            record = expected.diagnostics[0].reduction
            assert found["diagnostics"]["reduction"]["reason"] == record.reason
            return
        found = _evaluate(sdk, handle, points, scales)
        expected = runtime.evaluate(
            exact_points,
            mu_r2=exact_scales,
            reduction="tirust",
            on_failure="return",
            **options,
        )
        if expected.values[0] is None:
            assert found["statuses"] == [FAILED_STATUS]
            return
        literals = found["decimal"]["virtual"]
        values = [x for laurent in expected.values[0] for x in laurent]
        assert [Decimal(x) for x in literals] == [Decimal(x) for x in values]
    finally:
        sdk.rusticol_virtual_free(handle)


# ------------------------------------------------ observables (increment 2)
#
# The loop-induced resolved and correlated outputs (m-lit LIT-SDK, PLAN §3.5
# Done; DECISIONS 2026-11-07 21:50 R7): the C, C++, Fortran and UMAMI
# clients of ``test_loop_v2_sdk_extras`` (and the safe Rust ``Runtime``,
# opt-in as there) run every call of ``examples/extras_v4*`` on the two LI-O1
# rows and equal the Python API's ``evaluate``, ``evaluate_correlated_many``
# (one request per history, ``born`` the colour metric: |A|^2) and
# ``evaluate_resolved`` (per helicity row, one colour-contracted component):
# binary64 bitwise, double-double and arbitrary precision as ``Decimal``
# values; the "uv" and "ir" calls are refused in the Python API's words. The
# Python API's binary64 outputs run the same Rust code
# (``induced_observables.NATIVE_BINARY64``), bitwise its ``Decimal`` oracle.

from tests.unit.test_loop_v2_sdk_extras import (  # noqa: E402
    CALLS,
    _bits,
    _compare,
    _flat,
    _laurent,
    _run,
    clients,  # noqa: F401 (the compiled clients fixture)
)

# The spin source of the first declared gluon class: a literal complex
# vector, not transverse (the output is that of its transverse projection).
SPIN = [(0.5, 0.25), (0.75, -0.125), (0.375, 1.0), (-0.625, 0.5)]


def _case(name: str, runtime) -> dict:
    """The clients' inputs of row ``name``: two fixture points and their
    mu_R^2 in binary64, and a spin source on the first declared class."""

    del runtime
    points, scales = _inputs(name)
    classes = _spin_classes(name)
    assert classes, name
    return {"points": points, "scales": scales, "spin": {classes[0][0]: SPIN}}


def _spin_classes(name: str) -> list[list[int]]:
    """The row's declared spin classes (``loop_v2_induced_observables_rows.
    generate``: every gluon alone)."""

    from tests.unit.loop_v2_induced_observables_rows import gluons

    return [[leg] for leg in gluons(_fixture(name)["pdgs"])]


def _induced_references(runtime, case: dict) -> tuple[dict, dict]:
    """The Python API's values of every client call in the SDK layouts
    (``test_loop_v2_sdk_extras.references``), and per refused call the
    Python API's message."""

    from pyamplicol.api.errors import CompatibilityError
    from pyamplicol.correlators import CorrelatedRequest

    histories = [item.id for item in runtime.available_color_correlations()]
    assert histories[0] == "born" and len(histories) > 1
    points, scales = _exact(case["points"], case["scales"])
    spin = {
        leg: [(str(Decimal(re)), str(Decimal(im))) for re, im in vector]
        for leg, vector in case["spin"].items()
    }
    calls, refused = {}, {}
    for label, arithmetic, digits, poles, kind, count in CALLS:
        options = {"mu_r2": scales[:count], "poles": poles}
        if arithmetic != "double":
            options["arithmetic"] = arithmetic
        if digits is not None:
            options["precision"] = digits
        batch = points[:count]
        if poles in ("uv", "ir"):
            with pytest.raises(CompatibilityError) as error:
                runtime.evaluate(batch, **options)
            refused[label] = str(error.value)
            continue
        if kind == 0:
            result = runtime.evaluate(batch, **options)
            born = [v for b in result.born_values for v in b]
            virtual = [x for v in result.values for lc in v for x in _laurent(lc)]
        elif kind == 3:
            result = runtime.evaluate_resolved(batch, **options)
            born = [x for b in result.born_values for o in b for r in o for x in r]
            virtual = [
                x
                for v in result.values
                for sector in v
                for row in sector
                for lc in row
                for x in _laurent(lc)
            ]
        else:
            vectors = spin if kind == 2 else {}
            found = runtime.evaluate_correlated_many(
                batch, {h: CorrelatedRequest(h, vectors) for h in histories}, **options
            )
            born, virtual = [], []
            for p in range(count):
                for h in histories:
                    born += list(found[h].born_values[p])
                    virtual += [x for lc in found[h].values[p] for x in _laurent(lc)]
        assert born == [], (label, born)
        complex_layout = kind in (1, 2)
        calls[label] = {
            "born": _flat(born, complex_layout),
            "virtual": _flat(virtual, complex_layout),
            "complex": complex_layout,
        }
    return calls, refused


def _induced_feed(case: dict, path: Path, *, bits: bool) -> Path:
    """``test_loop_v2_sdk_extras._feed`` with the row's own scales."""

    def text(value: float) -> str:
        return str(_bits(float(value))) if bits else float(value).hex()

    lines = [f"{len(case['points'])} {len(case['points'][0])}"]
    for point, scale in zip(case["points"], case["scales"], strict=True):
        lines.append(" ".join([text(c) for p in point for c in p] + [text(scale)]))
    lines.append(str(len(case["spin"])))
    for leg in sorted(case["spin"]):
        lines.append(str(leg))
        lines += [f"{text(re)} {text(im)}" for re, im in case["spin"][leg]]
    path.write_text("\n".join(lines) + "\n")
    return path


def _compare_induced(label: str, expected: tuple[dict, dict], found: dict) -> None:
    values, refused = expected
    assert sorted(found) == sorted([*values, *refused]), label
    for call, message in refused.items():
        assert message in found[call].get("refused", ""), (label, call, found[call])
    _compare(label, values, {call: found[call] for call in values})


@pytest.fixture(scope="module")
def induced_cases(artifacts):
    from pyamplicol import Runtime

    out = {}
    for name in sorted(ROWS):
        runtime = Runtime.load(artifacts[name], process="p")
        case = _case(name, runtime)
        out[name] = (case, _induced_references(runtime, case))
    return out


@pytest.mark.parametrize("language", ["c", "cpp", "fortran", "umami"])
def test_loop_induced_sdk_observables_equal_the_python_api(
    language,
    artifacts,
    induced_cases,
    clients,  # noqa: F811 (the imported fixture)
    tmp_path,
):
    for name, (case, expected) in induced_cases.items():
        feed = _induced_feed(case, tmp_path / f"{name}.txt", bits=language == "fortran")
        found = _run(clients[language], artifacts[name], "p", feed)
        _compare_induced(f"{language}:{name}", expected, found)


def test_loop_induced_sdk_announces_its_observables(artifacts):
    """The capabilities list the resolved and correlated outputs: ``born``
    and the declared histories, the declared spin classes, one helicity row
    per plan row and one colour-contracted component; the Born counts stay
    zero."""

    from pyamplicol import Runtime

    sdk = _sdk()
    for name in sorted(ROWS):
        runtime = Runtime.load(artifacts[name], process="p")
        handle = _open(sdk, artifacts[name])
        try:
            capabilities = _json(sdk, sdk.rusticol_virtual_capabilities_json, handle)
            assert capabilities["resolved"] and capabilities["correlations"]
            assert capabilities["observables_unavailable"] is None
            assert capabilities["correlation_histories"] == [
                item.id for item in runtime.available_color_correlations()
            ]
            assert capabilities["spin_correlation_classes"] == _spin_classes(name)
            assert capabilities["resolved_color_ids"] == ["full"]
            axes = _Axes()
            assert sdk.rusticol_virtual_shape(handle, ctypes.byref(axes)) == 0
            assert axes.resolved_available == 1
            assert axes.born_orders == 0 and axes.born_color_components == 0
            assert axes.helicities == len(capabilities["helicity_rows"])
        finally:
            sdk.rusticol_virtual_free(handle)


def test_python_binary64_observables_run_the_native_implementation(artifacts):
    """The Python API's binary64 loop-induced outputs (the native forms,
    spin coefficients and projected vectors the SDK shares) are bitwise its
    ``Decimal`` oracle (``NATIVE_BINARY64 = False``) on both rows: correlated
    with and without a spin source, resolved."""

    from pyamplicol import Runtime
    from pyamplicol.correlators import CorrelatedRequest
    from pyamplicol.loops.v2 import induced_observables as io

    def values(runtime, case) -> list:
        points, scales = _exact(case["points"], case["scales"])
        spin = {
            leg: [(str(Decimal(re)), str(Decimal(im))) for re, im in vector]
            for leg, vector in case["spin"].items()
        }
        histories = [item.id for item in runtime.available_color_correlations()]
        out = []
        for vectors in ({}, spin):
            found = runtime.evaluate_correlated_many(
                points,
                {h: CorrelatedRequest(h, vectors) for h in histories},
                mu_r2=scales,
            )
            out += [
                complex(x)
                for h in histories
                for point in found[h].values
                for lc in point
                for x in _laurent(lc)
            ]
        resolved = runtime.evaluate_resolved(points, mu_r2=scales)
        out += [
            x
            for v in resolved.values
            for sector in v
            for row in sector
            for lc in row
            for x in _laurent(lc)
        ]
        return out

    for name in sorted(ROWS):
        native = Runtime.load(artifacts[name], process="p")
        case = _case(name, native)
        assert io.NATIVE_BINARY64
        found = values(native, case)
        io.NATIVE_BINARY64 = False
        try:
            oracle = values(Runtime.load(artifacts[name], process="p"), case)
        finally:
            io.NATIVE_BINARY64 = True
        assert len(found) == len(oracle) > 0
        for k, (a, b) in enumerate(zip(found, oracle, strict=True)):
            a, b = complex(a), complex(b)
            assert _bits(a.real) == _bits(b.real) and _bits(a.imag) == _bits(b.imag), (
                name,
                k,
                a,
                b,
            )


@pytest.mark.loop_nightly
def test_the_rust_sdk_observables_follow_the_python_api(
    artifacts, induced_cases, tmp_path
):
    """The safe Rust ``Runtime`` (``sdk/tests/v4_extras.rs``) on both rows,
    compared as the other clients (opt-in as
    ``test_loop_v2_sdk_extras.test_the_rust_sdk_extras_follow_the_python_api``:
    ``PYAMPLICOL_SDK_CARGO_TARGET_DIR``, a private cargo target directory)."""

    import os
    import subprocess

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
    root = Path(__file__).resolve().parents[2]
    mask = 2**64 - 1
    for name, (case, expected) in induced_cases.items():
        feed = tmp_path / f"{name}.json"
        feed.write_text(
            json.dumps(
                {
                    "artifact": str(artifacts[name]),
                    "process": "p",
                    "points": [
                        [_bits(float(c)) & mask for p in point for c in p]
                        + [_bits(scale) & mask]
                        for point, scale in zip(
                            case["points"], case["scales"], strict=True
                        )
                    ],
                    "spin_legs": sorted(case["spin"]),
                    "spin": [
                        [
                            _bits(float(x)) & mask
                            for pair in case["spin"][leg]
                            for x in pair
                        ]
                        for leg in sorted(case["spin"])
                    ],
                }
            )
        )
        output = tmp_path / f"{name}.out"
        env = {k: v for k, v in os.environ.items() if k != "CARGO_TARGET_DIR"}
        env = {
            **_CODEGEN_DEFAULTS,
            **env,
            "PYAMPLICOL_V4_EXTRAS_FEED": str(feed),
            "PYAMPLICOL_V4_EXTRAS_OUTPUT": str(output),
        }
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
                "v4_extras",
                "--",
                "--ignored",
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
            check=False,
        )
        assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
        from tests.unit.test_loop_v2_sdk_extras import _parse

        _compare_induced(f"rust:{name}", expected, _parse(output.read_text()))
