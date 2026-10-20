# SPDX-License-Identifier: 0BSD
"""Exact ordinary-source relationships must enter CT attachment witnesses."""

import json
import re
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

from pyamplicol.loops.counterterms import (
    _ordinary_bose_slots,
    _rooted_current_key,
    prepare_counterterm_model,
)
from pyamplicol.loops.model import build_loop_model
from pyamplicol.loops.model_expressions import parse_exact_ufo
from pyamplicol.loops.scalar_frontend import lower_scalar_program
from pyamplicol.loops.vertex_symmetry import symmetric_boson_slots
from pyamplicol.models.loop_import import capture_raw_loop_model

pytestmark = pytest.mark.loop_fast


@pytest.fixture
def hep():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("local dependency vertex proof requires licensed Symbolica")
    return pytest.importorskip("symbolica.community.hepkit")


def _split_vertex(hep, *, independent=False, diagnostic=False):
    from ufo_model_loader.symbolica_processing import wrap_indices

    payload = json.loads(hep.Model.qcd_qed().to_json())
    raw = ModuleType("owned_ordinary_bose")
    raw.all_particles = [SimpleNamespace(**p) for p in payload["particles"]]
    gluon = next(p for p in raw.all_particles if p.pdg_code == 21)
    raw.all_parameters = [
        SimpleNamespace(name=name, nature="external", type="real", value="2")
        for name in ("G", "H")
    ]
    raw.all_parameters.append(
        SimpleNamespace(name="z", nature="external", type="complex", value="1j")
    )
    common = "(G if z < 0 else 2*G)" if diagnostic else "G"
    minus = SimpleNamespace(name="OWNED_MINUS", value=f"-({common})", order={"QCD": 1})
    plus = SimpleNamespace(
        name="OWNED_PLUS", value="H" if independent else common, order={"QCD": 1}
    )
    raw.all_couplings = [minus, plus]
    structures = (
        "P(3,1)*Metric(1,2)",
        "P(3,2)*Metric(1,2)",
        "P(2,1)*Metric(1,3)",
        "P(2,3)*Metric(1,3)",
        "P(1,2)*Metric(2,3)",
        "P(1,3)*Metric(2,3)",
    )
    raw.all_lorentz = [
        SimpleNamespace(name=f"OwnedVVV{i}", spins=[3, 3, 3], structure=value)
        for i, value in enumerate(structures)
    ]
    coefficients = (minus, plus, plus, minus, minus, plus)
    raw.all_vertices = [
        SimpleNamespace(
            name="V_36",
            particles=[gluon] * 3,
            color=["f(1,2,3)"],
            lorentz=raw.all_lorentz,
            couplings={(0, i): value for i, value in enumerate(coefficients)},
        )
    ]
    captured = capture_raw_loop_model(raw)
    program = lower_scalar_program(captured)
    source = build_loop_model(captured, scalar_program=program)
    definitions = {value.name: value for value in program.definitions.values()}
    for coefficient in raw.all_couplings:
        payload["couplings"].append(
            {
                "name": coefficient.name,
                "expression": definitions[
                    coefficient.name
                ].value.expression.to_canonical_string(),
                "orders": [["QCD", 1]],
                "value": None,
            }
        )
    payload["lorentz_structures"].extend(
        {
            "name": value.name,
            "spins": value.spins,
            "structure": wrap_indices(
                parse_exact_ufo(value.structure)
            ).to_canonical_string(),
        }
        for value in raw.all_lorentz
    )
    rule = next(value for value in payload["vertex_rules"] if value["name"] == "V_36")
    rule["lorentz_structures"] = [value.name for value in raw.all_lorentz]
    rule["couplings"] = [[value.name for value in coefficients]]
    native = hep.Model.from_json(json.dumps(payload))
    return prepare_counterterm_model(source, native)


def _current_pair(hep, native):
    current = hep.FeynmanDiagram.from_dot(
        native,
        'digraph {ext[style=invis];v[int_id="V_36"];'
        'ext->v[particle="g"];ext->v[particle="g"];ext->v[particle="g"];}',
    )
    encoded = re.sub(
        r"target_slot=([02])",
        lambda m: f"target_slot={2 - int(m.group(1))}",
        current.to_dot(),
    )
    return current, hep.FeynmanDiagram.from_dot(native, encoded)


def test_owned_split_vertex_recovers_rooted_attachment_identity(hep):
    prepared = _split_vertex(hep)
    raw = prepared.source.raw.to_json()
    assert symmetric_boson_slots(prepared.native, ["V_36"]) == {}
    classes = _ordinary_bose_slots(prepared, ["V_36"])
    assert classes == {"V_36": (0, 0, 0)}
    left, right = _current_pair(hep, prepared.native)
    assert _rooted_current_key(left, {0, 1}) != _rooted_current_key(right, {0, 1})
    assert _rooted_current_key(left, {0, 1}, classes) == _rooted_current_key(
        right, {0, 1}, classes
    )
    assert _rooted_current_key(left, {0, 1}, classes) != _rooted_current_key(
        right, {0, 2}, classes
    )
    assert prepared.source.raw.to_json() == raw


def test_equal_external_defaults_do_not_prove_independent_couplings(hep):
    prepared = _split_vertex(hep, independent=True)
    parameters = prepared.source.raw.collections["parameters"]
    assert parameters[0]["fields"]["value"] == parameters[1]["fields"]["value"]
    assert _ordinary_bose_slots(prepared, ["V_36"]) == {}


@pytest.mark.parametrize("damage", ["missing", "foreign", "override"])
def test_paired_source_binding_damage_is_a_hard_error(hep, damage):
    prepared = _split_vertex(hep)
    if damage == "missing":
        prepared = replace(
            prepared,
            source=replace(
                prepared.source,
                kernels=tuple(
                    entry
                    for entry in prepared.source.kernels
                    if entry.coupling.name != "OWNED_PLUS"
                ),
            ),
        )
    elif damage == "foreign":
        prepared = replace(
            prepared,
            source=replace(
                prepared.source,
                kernels=tuple(
                    replace(entry, coupling=replace(entry.coupling))
                    for entry in prepared.source.kernels
                ),
            ),
        )
    else:
        payload = json.loads(prepared.native.to_json())
        coupling = next(
            value for value in payload["couplings"] if value["name"] == "OWNED_PLUS"
        )
        coupling["expression"] = "2*UFO::G"
        prepared = replace(prepared, native=hep.Model.from_json(json.dumps(payload)))
    with pytest.raises(ValueError, match=r"source binding|scalar owner"):
        _ordinary_bose_slots(prepared, ["V_36"])


def test_local_proof_preserves_active_source_diagnostics(hep):
    prepared = _split_vertex(hep, diagnostic=True)
    program = prepared.source.scalar_program
    diagnostics = tuple(value.diagnostic for value in program.definitions.values())
    assert any(value != 0 for value in diagnostics)
    assert _ordinary_bose_slots(prepared, ["V_36"]) == {"V_36": (0, 0, 0)}
    assert prepared.source.scalar_program is program
    assert (
        tuple(value.diagnostic for value in program.definitions.values()) == diagnostics
    )


def test_unpaired_catalogue_retains_literal_native_proof(hep):
    prepared = _split_vertex(hep)
    unpaired = replace(prepared, source=replace(prepared.source, scalar_program=None))
    assert _ordinary_bose_slots(unpaired, ["V_36"]) == {}
    assert _ordinary_bose_slots(prepared, ["V_36"]) == {"V_36": (0, 0, 0)}
