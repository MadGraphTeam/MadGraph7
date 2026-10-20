# SPDX-License-Identifier: 0BSD
"""Projected source observables use an actual common-degree identity bank."""

from types import SimpleNamespace

import pytest

from pyamplicol.color.connections import (
    COLOR_CONNECTION_CONVENTION,
    ColorConnection,
    ColorLeg,
    EmitGluon,
)
from pyamplicol.loops.physical_correlations import (
    PROJECTED_IDENTITY_CONVENTION,
    directed_color_convention,
)
from pyamplicol.loops.physical_model_correlations import require_directed_model_scope

pytestmark = pytest.mark.loop_fast


def _owner(accuracy, *, projected=True):
    bare = SimpleNamespace(
        color_accuracy=accuracy,
        _formal_color_projected=projected,
        _color_metrics_revision=int(projected),
        born_color_degrees=(4,),
        virtual_color_degrees=(5,),
        order_layout=SimpleNamespace(
            born=(None,), virtual=(None,), correction_names=("QCD",)
        ),
    )
    return SimpleNamespace(
        bare=bare,
        r2_strategy="model",
        color_accuracy=accuracy,
        color_projection=object() if projected else None,
    )


def _histories():
    legs = (ColorLeg(0, 8), ColorLeg(1, 8))
    identity = ColorConnection(legs)
    charged = ColorConnection(legs, (EmitGluon(0, -1),))
    return ((identity, identity),), ((charged, charged),)


@pytest.mark.parametrize("accuracy", ["lc", "nlc", "full"])
def test_formally_projected_models_only_admit_identity_histories(accuracy):
    model = _owner(accuracy)
    identity, charged = _histories()
    assert (
        directed_color_convention(model.bare, identity) == PROJECTED_IDENTITY_CONVENTION
    )
    require_directed_model_scope(model, identity)
    # Multiple differently named identity observables can share the same bank;
    # there is no one-history or one-source-class cap.
    require_directed_model_scope(model, identity * 3)
    for histories in ((), charged, (*identity, *charged)):
        with pytest.raises(NotImplementedError, match="identity histories"):
            require_directed_model_scope(model, histories)
    model.color_projection = None
    with pytest.raises(ValueError, match="projection owners differ"):
        require_directed_model_scope(model, identity)


def test_fixed_full_colour_and_unsupported_incomplete_projection_are_distinct():
    identity, charged = _histories()
    model = _owner("full", projected=False)
    assert directed_color_convention(model.bare, charged) == COLOR_CONNECTION_CONVENTION
    require_directed_model_scope(model, charged)
    model.bare._color_metrics_revision = 1
    with pytest.raises(NotImplementedError, match="unmodified full colour"):
        require_directed_model_scope(model, charged)
    with pytest.raises(ValueError, match="no formal projection"):
        require_directed_model_scope(_owner("lc", projected=False), identity)


@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
def test_public_preflight_defers_projected_history_proof_to_actual_owner(accuracy):
    from pyamplicol.color.correlator_matrices import ColorCorrelator
    from pyamplicol.correlators import CorrelatorConfig
    from pyamplicol.generation.loop_service import _check_correlators

    backend = SimpleNamespace(
        _run_config=SimpleNamespace(
            color=SimpleNamespace(accuracy=accuracy, contraction="direct")
        )
    )
    _check_correlators(backend, CorrelatorConfig(spin_correlations=((3,), (3, 4))))
    _check_correlators(
        backend, CorrelatorConfig(color_correlations=(ColorCorrelator("identity"),))
    )
    _check_correlators(
        backend,
        CorrelatorConfig(color_correlations=(ColorCorrelator.dipole("T", 1, 2),)),
    )
    _check_correlators(backend, CorrelatorConfig(all_color_through_order=1))
    with pytest.raises(TypeError, match="CorrelatorConfig"):
        _check_correlators(backend, object())
