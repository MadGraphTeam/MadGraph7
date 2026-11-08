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
    _agree,
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


def _record(record, expected) -> dict:
    """The SDK diagnostics' call-level reduction record of the Python API's
    ``VirtualReduction`` (its fallback, and how many points it evaluated
    again: the lanes recording another backend than the call's)."""

    moved = [
        lane
        for lane in expected.diagnostics
        if lane.reduction is not None and lane.reduction.effective != record.effective
    ]
    return {
        "requested": record.requested,
        "effective": record.effective,
        "reason": record.reason,
        "fallback": record.fallback,
        "fallback_points": len(moved),
    }


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
            assert _agree(values, literals), (arithmetic, values, literals)
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
        assert default["diagnostics"]["reduction"] == _record(record, expected)
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
            assert found["diagnostics"]["reduction"] == _record(record, expected)
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


# The row whose extended-precision executor runs take minutes per client (its
# double-double and arbitrary-precision calls run in C and the Rust control;
# every wrapper transports the same C ABI values, and the other row runs every
# call in every language).
SLOW_EXTENDED = "li_gg_hgg"
DOUBLE_CALLS = {label for label, arithmetic, *_rest in CALLS if arithmetic == "double"}


@pytest.mark.parametrize("language", ["c", "cpp", "fortran", "umami"])
def test_loop_induced_sdk_observables_equal_the_python_api(
    language,
    artifacts,
    induced_cases,
    clients,  # noqa: F811 (the imported fixture)
    tmp_path,
):
    for name, (case, (values, refused)) in induced_cases.items():
        feed = _induced_feed(case, tmp_path / f"{name}.txt", bits=language == "fortran")
        if name == SLOW_EXTENDED and language != "c":
            found = _run(clients[language], artifacts[name], "p", feed, ("binary64",))
            values = {k: v for k, v in values.items() if k in DOUBLE_CALLS}
            refused = {k: v for k, v in refused.items() if k in DOUBLE_CALLS}
        else:
            found = _run(clients[language], artifacts[name], "p", feed)
        _compare_induced(f"{language}:{name}", (values, refused), found)


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


def test_the_c_abi_falls_back_per_point_at_a_stated_target(artifacts, loop_native):
    """F3 (c) through the C ABI: u u~ > h g g point 0 at the target whose
    calibrated acceptance is half its method-choice lanes' deviation
    (``rusticol_virtual_set_reduction_target``): "auto" serves it on NinjaRS
    (bitwise the explicit NinjaRS call's value, the diagnostics' "fallbacks"
    naming it with the fallback's reason, "fallback_points" 1), the other
    points on TIRust (bitwise the explicit TIRust call); explicit "tirust"
    fails it typed; a target that is not finite and positive is an invalid
    argument and changes nothing. The Python API at the same target records
    the same backends and values."""

    import re

    from pyamplicol import Runtime

    if "tirust" not in json.loads(loop_native.reduction_backends()):
        pytest.skip("the staged native extension is built without TIRust")
    name = "li_uux_hgg"
    artifact = artifacts[name]
    points, scales = _inputs(name, 3)
    exact_points, exact_scales = _exact(points, scales)
    runtime = Runtime.load(artifact, process="p")
    tiny = runtime.evaluate(
        exact_points[:1],
        mu_r2=exact_scales[:1],
        reduction="tirust",
        reduction_target=1e-300,
        on_failure="return",
    )
    ratio = max(
        float(d) / float(s)
        for d, s in re.findall(
            r"deviation ([0-9.eE+-]+) of term scale ([0-9.eE+-]+)",
            tiny.diagnostics[0].detail,
        )
    )
    target = ratio / 2 / loop_native.tir_acceptance(1.0)
    sdk = _sdk()
    set_target = sdk.rusticol_virtual_set_reduction_target
    set_target.argtypes = [ctypes.c_void_p, ctypes.c_double]
    handle = _open(sdk, artifact)
    try:
        for bad in (0.0, -1.0, math.inf, math.nan):
            assert set_target(handle, bad) == STATUS_INVALID_ARGUMENT
            assert "finite positive number" in _error(sdk)
        capabilities = _json(sdk, sdk.rusticol_virtual_capabilities_json, handle)
        assert capabilities["reduction"]["target"] == 1e-5
        assert set_target(handle, target) == 0, _error(sdk)
        runs = {}
        for backend in ("ninjars", "tirust", "auto"):
            assert sdk.rusticol_virtual_set_reduction(handle, backend.encode()) == 0
            runs[backend] = _evaluate(sdk, handle, points, scales)
        auto, tirust, ninjars = runs["auto"], runs["tirust"], runs["ninjars"]
        assert tirust["statuses"] == [FAILED_STATUS, SUCCESS_STATUS, SUCCESS_STATUS]
        assert auto["statuses"] == [SUCCESS_STATUS] * 3
        width = len(auto["virtual"]) // 3
        assert auto["virtual"][:width] == ninjars["virtual"][:width]
        assert auto["virtual"][width:] == tirust["virtual"]
        record = auto["diagnostics"]["reduction"]
        assert (record["effective"], record["fallback"], record["fallback_points"]) == (
            "tirust",
            "ninjars",
            1,
        )
        (served,) = auto["diagnostics"]["fallbacks"].items()
        assert served[0] == "0" and served[1]["effective"] == "ninjars"
        assert served[1]["reason"].startswith("auto: NinjaRS, the per-point fallback")
        assert "refused by its check" in served[1]["reason"]
        assert tirust["diagnostics"]["fallbacks"] == {}
        assert tirust["diagnostics"]["reduction"]["fallback"] is None
        expected = runtime.evaluate(
            exact_points, mu_r2=exact_scales, reduction_target=target
        )
        assert [lane.reduction.effective for lane in expected.diagnostics] == [
            "ninjars",
            "tirust",
            "tirust",
        ]
        assert expected.diagnostics[0].reduction.reason == served[1]["reason"]
        assert _same(auto["virtual"], _python_flat(expected))
    finally:
        sdk.rusticol_virtual_free(handle)


FALLBACK_CLIENT = r"""
// ./client artifact process input.txt target arithmetic: one "auto" evaluation
// of a loop-induced artifact through the C++ wrapper at the target accuracy
// `target` in `arithmetic` ("double" or "double-double"): the statuses, the
// virtual values as bit patterns and the diagnostics JSON, one line each.
#include "rusticol_virtual.hpp"
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 6) return 2;
  std::ifstream in(argv[3]);
  std::size_t points = 0, legs = 0;
  in >> points >> legs;
  std::vector<double> momenta(points * legs * 4), scales(points);
  std::string token;
  for (std::size_t p = 0; p < points; ++p) {
    for (std::size_t i = 0; i < legs * 4; ++i) {
      in >> token;
      momenta[p * legs * 4 + i] = std::strtod(token.c_str(), nullptr);
    }
    in >> token;
    scales[p] = std::strtod(token.c_str(), nullptr);
  }
  try {
    rusticol::VirtualRuntime runtime(argv[1], std::string(argv[2]));
    runtime.set_reduction_target(std::strtod(argv[4], nullptr));
    runtime.set_arithmetic(argv[5]);
    auto result = runtime.evaluate(momenta, points, scales);
    std::printf("statuses");
    for (auto status : result.statuses()) std::printf(" %u", status);
    std::printf("\nvirtual");
    for (double value : result.virtual_values()) {
      std::uint64_t bits;
      std::memcpy(&bits, &value, sizeof bits);
      std::printf(" %" PRIu64, bits);
    }
    std::printf("\ndiagnostics %s\n", result.diagnostics_json().c_str());
  } catch (const rusticol::VirtualError &error) {
    std::printf("error %d %s\n", error.status(), error.what());
    return 1;
  }
  return 0;
}
"""


def test_the_sdk_fallback_and_target_need_no_symbolica_licence(
    artifacts, loop_native, tmp_path
):
    """Done 5 of m-lit lit-tirust-default (USER 2026-11-08 ~15:25: no SDK
    call may need a user Symbolica licence): a fresh process with
    ``SYMBOLICA_LICENSE`` removed and ``PYAMPLICOL_NATIVE_LIBRARY_KEY=off``,
    while another process holds the restricted Symbolica permit, sets the
    target accuracy and runs "auto" in binary64 on u u~ > h g g at the
    target that refuses point 0's method-choice lanes: the per-point NinjaRS
    fallback serves point 0, and the output equals the licensed run's. A
    double-double "auto" call without ``SYMBOLICA_LICENSE`` (the package
    library key on, as every SDK double-double call) runs on TIRust with its
    fallback armed and equals the licensed run's."""

    import importlib.util
    import os
    import re
    import subprocess

    from pyamplicol import Runtime
    from tests.unit.test_loop_v2_sdk_extras import SDK, _compiler, _staged_sdk

    if "tirust" not in json.loads(loop_native.reduction_backends()):
        pytest.skip("the staged native extension is built without TIRust")
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "loop_artifact_portability", root / "tools" / "loop_artifact_portability.py"
    )
    assert spec is not None and spec.loader is not None
    portability = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portability)
    name = "li_uux_hgg"
    artifact = artifacts[name]
    points, scales = _inputs(name, 3)
    exact_points, exact_scales = _exact(points, scales)
    tiny = Runtime.load(artifact, process="p").evaluate(
        exact_points[:1],
        mu_r2=exact_scales[:1],
        reduction="tirust",
        reduction_target=1e-300,
        on_failure="return",
    )
    ratio = max(
        float(d) / float(s)
        for d, s in re.findall(
            r"deviation ([0-9.eE+-]+) of term scale ([0-9.eE+-]+)",
            tiny.diagnostics[0].detail,
        )
    )
    target = ratio / 2 / loop_native.tir_acceptance(1.0)
    build = _staged_sdk()
    (tmp_path / "client.cpp").write_text(FALLBACK_CLIENT)
    subprocess.run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-O2",
            f"-I{SDK / 'include'}",
            str(tmp_path / "client.cpp"),
            f"-L{build}",
            "-lrusticol_loop_sdk",
            f"-Wl,-rpath,{build}",
            "-lm",
            "-o",
            str(tmp_path / "client"),
        ],
        check=True,
        cwd=tmp_path,
    )
    lines = [f"{len(points)} {len(points[0])}"]
    for point, scale in zip(points, scales, strict=True):
        lines.append(" ".join(v.hex() for v in [*(c for p in point for c in p), scale]))
    (tmp_path / "input.txt").write_text("\n".join(lines) + "\n")

    def run(env: dict[str, str], arithmetic: str, chosen: float) -> list[str]:
        done = subprocess.run(
            [
                str(tmp_path / "client"),
                str(artifact),
                "p",
                str(tmp_path / "input.txt"),
                repr(chosen),
                arithmetic,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env=env,
        )
        output = done.stdout[-3000:] + done.stderr[-3000:]
        assert done.returncode == 0, output
        assert "restricted Symbolica" not in output
        return done.stdout.splitlines()

    lock = tmp_path / "symbolica-locks"
    lock.mkdir()
    lock.chmod(0o700)
    base = {k: v for k, v in os.environ.items() if k != "SYMBOLICA_LICENSE"}
    with portability.RestrictedPermit(lock_dir=str(lock)) as permit:
        assert permit.record["verified"], permit.record
        unlicensed = run(
            permit.environment({**base, "PYAMPLICOL_NATIVE_LIBRARY_KEY": "off"}),
            "double",
            target,
        )
    licensed = run(dict(os.environ), "double", target)
    assert unlicensed == licensed
    statuses, _values, diagnostics = licensed
    assert statuses == "statuses" + " 4" * len(points)
    record = json.loads(diagnostics.split(" ", 1)[1])
    assert record["reduction"]["effective"] == "tirust"
    assert record["reduction"]["fallback_points"] == 1
    assert record["fallbacks"]["0"]["effective"] == "ninjars"
    # Double-double, no user licence (the package key on).
    keyed = run(base, "double-double", 1e-5)
    assert keyed == run(dict(os.environ), "double-double", 1e-5)
    record = json.loads(keyed[2].split(" ", 1)[1])
    assert (record["reduction"]["effective"], record["reduction"]["fallback"]) == (
        "tirust",
        "ninjars",
    )
    assert record["arithmetic"] == "double-double"


def test_a_structural_tirust_refusal_falls_back_at_the_default_target(
    loop_native, tmp_path
):
    """F3 (c), Done 1 (ii) (1), at the DEFAULT target (1e-5): a refusal no
    acceptance governs. g g > a a g (LI-33) at its first frozen stream points:
    TIRust's conditioning gates refuse a row and then its method choice
    refuses it too ("refused by the gates, then by the method choice"), so
    explicit "tirust" fails the point typed in the Python API and the C ABI;
    "auto" serves it on NinjaRS (its value bitwise the explicit NinjaRS one,
    its record NinjaRS and the fallback's reason), the C ABI's diagnostics
    naming it under "fallbacks", bitwise the Python API's values."""

    from fractions import Fraction

    from pyamplicol import Runtime
    from tools.loop_benchmark import streams

    if "tirust" not in json.loads(loop_native.reduction_backends()):
        pytest.skip("the staged native extension is built without TIRust")
    artifact = generate(tmp_path / "aag", "li_gg_aag", "g g > a a g")
    stream = streams.load("li_gg_aag")
    count = 4
    exact_points = [
        [[str(Decimal(float(Fraction(c)))) for c in leg] for leg in point]
        for point in stream["points"][:count]
    ]
    exact_scales = [
        str(Decimal(float(Fraction(mu) ** 2))) for mu in stream["mu_r"][:count]
    ]
    runtime = Runtime.load(artifact, process="p")
    options = {"mu_r2": exact_scales, "on_failure": "return"}
    tirust = runtime.evaluate(exact_points, reduction="tirust", **options)
    ninjars = runtime.evaluate(exact_points, reduction="ninjars", **options)
    auto = runtime.evaluate(exact_points, **options)
    refused = [
        p
        for p, lane in enumerate(tirust.diagnostics)
        if "refused by the gates, then by the method choice" in lane.detail
    ]
    assert refused, [lane.detail[:200] for lane in tirust.diagnostics]
    for p, lane in enumerate(auto.diagnostics):
        if tirust.values[p] is None:
            assert lane.reduction.effective == "ninjars", p
            assert (
                "refused by the gates, then by the method choice"
                in (lane.reduction.reason)
                or p not in refused
            )
            assert repr(auto.values[p]) == repr(ninjars.values[p]), p
        else:
            assert lane.reduction.effective == "tirust", p
            assert repr(auto.values[p]) == repr(tirust.values[p]), p
    # The C ABI: the same backends and values.
    points = [[[float(c) for c in leg] for leg in point] for point in exact_points]
    scales = [float(Fraction(s)) for s in exact_scales]
    sdk = _sdk()
    handle = _open(sdk, artifact)
    try:
        found = _evaluate(sdk, handle, points, scales)
        record = found["diagnostics"]
        assert record["reduction"]["fallback"] == "ninjars"
        moved = sorted(int(key) for key in record["fallbacks"])
        assert moved == [p for p, v in enumerate(tirust.values) if v is None]
        for key in record["fallbacks"]:
            assert record["fallbacks"][key]["effective"] == "ninjars"
        assert found["statuses"] == [SUCCESS_STATUS] * count
        assert sdk.rusticol_virtual_set_reduction(handle, b"tirust") == 0
        explicit = _evaluate(sdk, handle, points, scales)
        assert [s == FAILED_STATUS for s in explicit["statuses"]] == [
            v is None for v in tirust.values
        ]
    finally:
        sdk.rusticol_virtual_free(handle)
    assert _same(found["virtual"], _python_flat(auto))
