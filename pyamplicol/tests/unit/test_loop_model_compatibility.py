# SPDX-License-Identifier: 0BSD
"""The explicit scalar-arity view (test support) changes one source reference only,
found structurally."""

import pytest

from pyamplicol.models.loop_import import RawLoopModel, load_raw_loop_model
from tools.loop_reference.model_compatibility import prepare_loop_sm_scalar_declarations

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def raw(loop_sm_ufo):
    pytest.importorskip("symbolica")
    path = str(loop_sm_ufo)
    return load_raw_loop_model(path)


def test_scalar_arity_view_preserves_source_and_every_other_declaration(raw):
    original = raw.to_dict()
    view = prepare_loop_sm_scalar_declarations(raw)
    assert view.original is raw and raw.to_dict() == original
    expected = raw.to_dict()
    replacement = raw.record("lorentz_structures", "SSSS1")
    vertex = raw.record("vertices", "V_1001")
    expected["collections"]["vertices"][vertex["id"]]["fields"]["lorentz"] = [
        dict(
            kind="reference",
            collection="lorentz_structures",
            id=replacement["id"],
            name="SSSS1",
        )
    ]
    expected["metadata"]["loop_sm_scalar_arity_compatibility"] = dict(
        view.diagnostics[0]
    )
    assert view.derived.to_dict() == expected
    with pytest.raises(ValueError, match="already derived"):
        prepare_loop_sm_scalar_declarations(view.derived)


def test_explicit_scalar_view_passes_strict_full_neutral_scope(raw):
    from symbolica import is_licensed

    from pyamplicol.loops.compiler import prepare_loop_request_model

    if not is_licensed():
        pytest.skip("source generation requires licensed Symbolica")
    options = dict(
        born_order_bounds={"QCD": 2, "QED": 2},
        loop_order_bounds={"QCD": 4, "QED": 2},
        fixed_parameters={"MB": "0", "ymb": "0", "MTA": "0", "ymtau": "0"},
    )
    view = prepare_loop_sm_scalar_declarations(raw)
    # The production source repair (``repair_source_declarations``, m-qcd
    # MQ-MODEL) prepares the stock source as the view does.
    for source in (raw, view.derived):
        prepared = prepare_loop_request_model(source, **options)
        vertex = prepared.structure.native.vertex_rule("V_1001")
        assert len(vertex.particles) == 4
        tensor = prepared.structure.native.lorentz_structure(
            vertex.lorentz_structures[0]
        )
        assert tuple(tensor.spins) == (1, 1, 1, 1)


@pytest.mark.parametrize("change", ["arity", "old_tensor", "new_tensor", "new_spins"])
def test_scalar_arity_view_rejects_other_source_declarations(raw, change):
    payload = raw.to_dict()
    vertex = payload["collections"]["vertices"][raw.record("vertices", "V_1001")["id"]][
        "fields"
    ]
    old = payload["collections"]["lorentz_structures"][
        raw.record("lorentz_structures", "SSS1")["id"]
    ]["fields"]
    new = payload["collections"]["lorentz_structures"][
        raw.record("lorentz_structures", "SSSS1")["id"]
    ]["fields"]
    if change == "arity":
        vertex["particles"].pop()
    elif change == "old_tensor":
        old["structure"] = "P(-1,1)*P(-1,2)"
    elif change == "new_tensor":
        new["structure"] = "2"
    else:
        new["spins"] = [1, 1, 1, 3]
    with pytest.raises(ValueError, match=r"arity mismatch|no unique constant tensor"):
        prepare_loop_sm_scalar_declarations(RawLoopModel.from_dict(payload))


def test_scalar_arity_view_is_structural(raw):
    """Other scalar species, renamed tensors and vertex: the same repair."""

    payload = raw.to_dict()
    vertex = raw.record("vertices", "V_1001")
    record = payload["collections"]["vertices"][vertex["id"]]
    particle = raw.record("particles", "H")
    record["fields"]["particles"][0] = dict(
        kind="reference", collection="particles", id=particle["id"], name="H"
    )
    record["name"] = "V_renamed"
    view = prepare_loop_sm_scalar_declarations(RawLoopModel.from_dict(payload))
    (diagnostic,) = view.diagnostics
    assert diagnostic["vertex"] == [vertex["id"], "V_renamed"]
    assert diagnostic["derived_lorentz"][1] == "SSSS1"
    # Byte-identical to the diagnostic saved artifacts record.
    assert diagnostic["reason"] == (
        "four scalar species require the existing four-slot constant tensor"
    )
