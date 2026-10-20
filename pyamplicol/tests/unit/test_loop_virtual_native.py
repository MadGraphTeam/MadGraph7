# SPDX-License-Identifier: 0BSD
"""Delivery lifecycle over an actual verified, source-owned virtual artifact."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_fast


def test_native_virtual_clear_keeps_parameter_owner_and_immutable_results(
    saved_loop_fixture, loop_native
):
    native = loop_native
    path = str(saved_loop_fixture("source-public-v3/gg"))
    handle = native.NativeVirtualRuntime(path, process="gg")
    metadata = json.loads(handle.metadata_json())
    assert metadata["physics"]["contribution_kind"] == "virtual"
    assert metadata["physics"]["epsilon_powers"] == [0, -1, -2]
    assert metadata["plan"]["r2_strategy"] == "model"
    assert handle.requested_r2_strategy == "auto"
    assert handle.r2_strategy == "model"
    assert handle.arithmetic == "double" and handle.precision_bits == 53

    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/gluon_scattering_qcd.json"
    )
    point = [
        [str(value) for value in leg]
        for leg in json.loads(fixture.read_text())["points"][0]
    ]
    first = handle.evaluate([point], ["1000000"], point_ids=[71])
    retained = first.component_batches[0].to_decimal()
    assert first.point_ids == [71]
    assert first.component_batches[0].columns == 3
    assert first.source_revision == 0

    state = handle.source_state
    old = state.snapshot()
    revision = handle.set_model_parameters(
        {"aS": ["0.121000000000000000000000000001", "0"]}
    )
    assert revision == 1
    changed = handle.evaluate([point], ["1000000"], point_ids=[71])
    assert changed.source_revision == 1
    assert changed.component_batches[0].to_decimal() != retained
    assert old.revision == 0
    assert first.component_batches[0].to_decimal() == retained

    handle.clear()
    assert handle.source_revision == 1
    for requested in ("direct", "off"):
        with pytest.raises(ValueError, match=r"R2|r2_strategy"):
            handle.evaluate([], [], r2_strategy=requested)
    # Updates after clear must still target the original parameter owner.
    assert handle.source_state.snapshot().revision == 1
    assert handle.evaluate([], [], r2_strategy="model").source_revision == 1
    after = handle.evaluate([point], ["1000000"], point_ids=[71])
    assert (
        after.component_batches[0].to_decimal()
        == changed.component_batches[0].to_decimal()
    )
    assert first.component_batches[0].to_decimal() == retained
    assert state.update({"aS": ["0.123000000000000000000000000001", "0"]}).revision == 2
    assert handle.source_revision == 2

    before = handle.parameter_overrides
    with pytest.raises(ValueError, match="unknown"):
        handle.set_model_parameters({"not_a_parameter": ["1", "0"]})
    assert handle.parameter_overrides == before and handle.source_revision == 2
    rebound = handle.rebind(arithmetic="double-double")
    assert rebound.precision_bits == 106
    assert rebound.parameter_overrides == before
    assert rebound.source_revision == 0
    handle.set_model_parameters({"aS": ["0.125", "0"]})
    assert rebound.parameter_overrides == before
    assert rebound.evaluate([], []).source_revision == 0

    with pytest.raises(native.VirtualCompatibilityError, match="direct R2"):
        handle.rebind(r2_strategy="direct")
