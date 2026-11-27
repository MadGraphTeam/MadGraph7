# SPDX-License-Identifier: 0BSD
"""The loop_sm source colour declaration (``models.loop_color_declaration``):
actual source collection keys and group labels, no assumed SM continuation."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_physics


def test_actual_charged_declaration_keys_and_negative_scope(loop_sm_colour_model):
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import (
        LoopColorDeclaration,
        resolve_loop_color_declaration,
    )

    # Prepared now from the committed example, not a saved compiled snapshot
    # whose compiler fingerprint goes stale.
    compiled = load_compiled_model(loop_sm_colour_model)
    raw = compiled.raw_loop_model
    declaration = resolve_loop_color_declaration(raw)
    stored = LoopColorDeclaration.from_dict(
        json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "examples/loop/source_colour/loop_sm_qcd_subset.json"
            ).read_text()
        )
    )
    assert declaration == stored
    # Actual source collection IDs and group labels are part of the explicit
    # provider input. Both charged entries are independently retained source data.
    expected = {
        ("vertices", 34, 0, 0, None): ("V_31", "Identity(1,2)", "GC_12"),
        ("vertices", 43, 0, 0, None): ("V_40", "Identity(1,2)", "GC_39"),
        ("counterterm_vertices", 36, 0, 0, 0): ("V_R2uxdW", "Identity(1,2)", "R2_uxdW"),
        ("counterterm_vertices", 27, 0, 0, 0): ("V_R2dxuW", "Identity(1,2)", "R2_dxuW"),
        ("counterterm_vertices", 2, 0, 0, 0): ("V_R2GDD", "T(3,2,1)", "R2_GQQ"),
        ("counterterm_vertices", 3, 0, 0, 0): ("V_R2GUU", "T(3,2,1)", "R2_GQQ"),
        ("counterterm_vertices", 9, 0, 0, 0): ("V_R2DD", "Identity(1,2)", "R2_QQq"),
        ("counterterm_vertices", 10, 0, 0, 0): ("V_R2UU", "Identity(1,2)", "R2_QQq"),
    }
    for vertex_id, name in ((59, "V_UVGDD"), (60, "V_UVGUU")):
        for group, suffix in enumerate(("q", "c", "b", "t", "g")):
            expected["counterterm_vertices", vertex_id, 0, 0, group] = (
                name,
                "T(3,2,1)",
                f"UV_GQQ{suffix}",
            )
    assert set(expected) <= set(declaration.kernel_keys)
    for (collection, index, color, lorentz, group), (
        name,
        tensor,
        coupling,
    ) in expected.items():
        record = raw.collections[collection][index]
        assert record["name"] == name
        assert record["fields"]["color"][color] == tensor
        entries = record["fields"]["couplings"]["entries"]
        key = [color, lorentz] if group is None else [color, lorentz, group]
        assert (
            next(
                value
                for indices, value in entries
                if tuple(indices["items"]) == tuple(key)
            )["name"]
            == coupling
        )
    assert declaration.parameter_identities == {
        "CA": "Nc",
        "CF": "(Nc**2-1)/(2*Nc)",
        "TF": "0.5",
        "Ncol": "Nc",
    }
    assert declaration.scheme_parameter_values == {"lhv": "1"}
