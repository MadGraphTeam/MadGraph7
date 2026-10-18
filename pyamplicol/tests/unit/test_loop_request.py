# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json

import pytest

from pyamplicol.loops.request import prepare_loop_external_request


@pytest.fixture(scope="module")
def model():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native model controls require licensed Symbolica")
    from pyamplicol.loops.control_models import qcd_qed_control_model

    return qcd_qed_control_model()


@pytest.mark.parametrize(
    "incoming,outgoing,kinds,spin,color,identical",
    [
        ((2, -2), (11, -11), ("u", "v_bar", "u_bar", "v"), 4, 9, 1),
        (
            (2, -2),
            (21, 21, 21),
            ("u", "v_bar", "epsilon_bar", "epsilon_bar", "epsilon_bar"),
            4,
            9,
            6,
        ),
        (
            (21, 21),
            (21, 21),
            ("epsilon", "epsilon", "epsilon_bar", "epsilon_bar"),
            4,
            64,
            2,
        ),
        ((-2, 21), (-2, 21), ("v_bar", "epsilon", "v", "epsilon_bar"), 4, 24, 1),
    ],
)
def test_request_derives_flow_and_reference_normalization(
    model, incoming, outgoing, kinds, spin, color, identical
):
    request = prepare_loop_external_request(model, incoming, outgoing)
    assert request.external_pdgs == incoming + outgoing
    assert request.state_kinds == kinds
    assert request.states.initial_spin_average == spin
    assert request.initial_color_average == color
    assert request.identical_final_factor == identical
    assert request.initial_average == spin * color * identical
    assert (
        prepare_loop_external_request(model, request.incoming, request.outgoing)
        == request
    )
    assert (
        json.loads(json.dumps(request.to_dict()))["normalization"]["divisor"]
        == request.initial_average
    )


def test_scalar_and_structural_massive_vector_have_physical_spin_averages(model):
    from symbolica.community import hepkit as hep

    payload = json.loads(model.to_json())
    gluon = next(p for p in payload["particles"] if p["pdg_code"] == 21)
    for name, pdg, spin in (("probe_s", 991, 1), ("probe_v", 992, 3)):
        payload["particles"].append(
            dict(
                gluon,
                name=name,
                antiname=name,
                pdg_code=pdg,
                spin=spin,
                color=1,
                mass="MT",
                width="ZERO",
            )
        )
    native = hep.Model.from_json(json.dumps(payload))
    request = prepare_loop_external_request(native, (991, 992), (991, 992))
    assert request.state_kinds == ("scalar", "epsilon", "scalar", "epsilon_bar")
    assert request.states.initial_spin_average == request.initial_average == 3
    assert [leg.offset for leg in request.states.legs] == [0, 1, 5, 6]
    assert request.states.legs[1].helicities == (-1, 0, 1)
    assert not request.states.legs[1].massless


def test_request_rejects_invalid_and_ghost_states(model):
    from symbolica.community.hepkit import ModelError

    with pytest.raises(ModelError):
        prepare_loop_external_request(model, (99999999,), (21,))
    with pytest.raises(TypeError, match="sequence"):
        prepare_loop_external_request(model, "u", (21,))
    with pytest.raises(ValueError, match="nonempty"):
        prepare_loop_external_request(model, (), (21,))
    ghost = next(p for p in model.particles if p.spin == -1)
    with pytest.raises(NotImplementedError, match="external"):
        prepare_loop_external_request(model, (ghost.name,), (21,))
