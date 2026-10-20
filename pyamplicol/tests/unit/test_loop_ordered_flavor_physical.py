# SPDX-License-Identifier: 0BSD
"""Actual complete-family d/s banks, without regenerating physical graphs."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from .test_loop_generation_spin_flavor_runtime import _vectors
from .test_loop_native_directed import _blocks, _values
from .test_loop_ordered_projection_physical import _close_component
from .test_loop_physical_correlations import _conjugate, _decimals

RETAINED_CONTROL = None
NAMES = (
    "born",
    "virtual",
    "bare",
    "uv",
    "r2",
    "uv_local",
    "uv_external_fields",
    "first",
    "second",
)


@pytest.fixture(scope="module")
def retained():
    if RETAINED_CONTROL is None:
        pytest.skip("actual retained independent d/s formal owners required")
    return RETAINED_CONTROL


def _points():
    path = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    return json.loads(path.read_text(), parse_float=str, parse_int=str)["points"]


def _no_new_algebra(monkeypatch):
    from pyamplicol.loops import directed_color_projection, prepared_correlations

    def forbidden(*args, **kwargs):
        raise AssertionError("retained complete-family rebind repeated colour algebra")

    for module, names in (
        (
            directed_color_projection,
            ("prepare_directed_color_bank", "project_directed_color_columns"),
        ),
        (
            prepared_correlations,
            (
                "prepare_directed_complete",
                "prepare_directed_bare",
                "prepare_directed_model",
                "_saved",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


@pytest.mark.parametrize("name", ("ddgg", "ssgg"))
@pytest.mark.parametrize(
    "arithmetic,precision,tolerance,bit",
    (
        ("double", None, "3e-9", 20),
        ("double-double", None, "3e-24", 70),
        ("arbitrary", 70, "3e-54", 170),
    ),
)
def test_complete_formal_full_matches_independent_complex_history_reference(
    retained,
    monkeypatch,
    name,
    arithmetic,
    precision,
    tolerance,
    bit,
):
    from pyamplicol.loops.spin_sources import NativeSpinSourcePlan

    key = name, "full"
    if key not in retained["prepared"]:
        pytest.skip("this independently prepared member is not ready yet")
    options = dict(
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine="fast" if arithmetic == "double" else "generic",
    )
    model = retained["models"][key].rebind(
        **options, coefficient_engine="interpreted", projector_engine="interpreted"
    )
    prepared = retained["prepared"][key]
    bank = prepared.bare.color_bank
    bank.require_owner(model)
    families = bank.projection.families
    assert [len(f.histories) for f in families] == [1, 4, 28]
    assert [f.order for f in families] == [0, 1, 2]
    assert bank.histories == prepared.bare.histories
    for item in retained["classes"][key].values():
        item.require_owner(model)
        assert item.bare.color_bank is bank
    ordinary_formal = model.bare.formal_color_metrics
    assert families[0].born_degrees == ordinary_formal.born_degrees
    assert families[0].virtual_degrees == ordinary_formal.virtual_degrees
    _no_new_algebra(monkeypatch)
    reference = model.bare.backend.NativeLoopProgram(
        str(retained["reference"] / f"independent-{name}.pacbin"),
        parameters={"aS": ("0.118", "0"), "MT": ("173", "0"), "MU_R": ("1000", "0")},
        **options,
    )
    point, ids, scales = _points()[:1], [2**64 - 1], ["1000000"]
    a, b = _vectors()
    sources = NativeSpinSourcePlan(model.bare, ((3,), (4,), (3, 4)))
    with localcontext() as context:
        context.prec = 120
        tolerance = Decimal(tolerance)
        alpha = str(Decimal("0.118") * (1 + Decimal(2) ** -bit))
        # The old independent model fixes MB=ymb=0. Match that actual parameter
        # convention using the new owner's supported numerical bindings.
        model.bare.model_state.update(
            {"MB": "0", "ymb": "0", "aS": alpha, "MT": "173", "MU_R": "1000"}
        )
        reference.source_state.update({"aS": (alpha, "0")})
        snapshot = model.bare.model_state.snapshot()
        classes = {
            joint: plan for joint, plan in retained["classes"][key].items() if joint
        }
        for joint, plan in {(): prepared, **classes}.items():
            vectors = {leg: [a if leg == 3 else b] for leg in joint}
            actual = plan.bind(model).evaluate_decimal(
                point,
                scales,
                point_ids=ids,
                spin_sources=sources.prepare_decimal(vectors, ids),
            )
            expected = reference.evaluate_directed(
                point,
                scales,
                point_ids=ids,
                spin_vectors=vectors,
            )
            assert expected.spin_legs == list(joint)
            assert expected.history_ids == ["born", "casimir", "ordered", "swapped"]
            assert not expected.failures and expected.point_ids == ids
            values = {label: _decimals(getattr(actual, label)) for label in NAMES}
            for label, block in zip(NAMES, _blocks(expected)[:-1], strict=True):
                target = _values(block, values[label].shape[:-1])
                _close_component(values[label], target, tolerance)
            for label in NAMES[:-2]:
                _close_component(
                    values[label][:, 2],
                    _conjugate(values[label][:, 3]),
                    tolerance,
                    scale=values[label],
                )
            _close_component(
                values["virtual"], values["first"] + values["second"], tolerance
            )
            _close_component(
                values["virtual"],
                values["bare"] + values["uv"] + values["r2"],
                tolerance,
            )
            _close_component(
                values["second"][:, 2],
                _conjugate(values["first"][:, 3]),
                tolerance,
                scale=values["first"],
            )
            imaginary = max(abs(x) for x in values["virtual"][:, 2, :, :, 1].flat)
            assert imaginary > Decimal("1e-6")
            wrong = values["first"][:, 2] + _conjugate(values["first"][:, 2])
            assert max(
                abs(x) for x in (values["virtual"][:, 2] - wrong).flat
            ) > Decimal("1e-6")
            assert all(
                lane == ids[0] and status == "success"
                for row in actual.statuses
                for lane, status in row
            )
            retained["records"][name, "full", arithmetic, joint] = values
            print(
                "COMPLETE_FULL_PHYSICAL_NONHERMITIAN",
                name,
                arithmetic,
                joint,
                "max_imaginary",
                str(imaginary),
                flush=True,
            )
        assert model.bare.model_state.snapshot() is snapshot
        # A shaped empty source request uses the retained plan, not a fresh bank.
        if arithmetic == "double":
            joint = (3, 4)
            empty = (
                retained["classes"][key][joint]
                .bind(model)
                .evaluate_decimal(
                    np.empty((0, 4, 4), dtype=str),
                    [],
                    point_ids=[],
                    spin_sources=sources.prepare_decimal({3: a, 4: b}, []),
                )
            )
            assert empty.virtual.shape == (
                0,
                4,
                len(model.bare.order_layout.virtual),
                3,
                2,
            )
            assert empty.statuses == ()
