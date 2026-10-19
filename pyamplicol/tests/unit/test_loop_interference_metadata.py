# SPDX-License-Identifier: 0BSD
"""The authenticated observable declaration agrees with its execution lane."""

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from pyamplicol.generation.loop_artifact import (
    DIRECTED_INTERFERENCE,
    ORDINARY_INTERFERENCE,
    LoopExecutionMetadata,
    LoopSchemeDeclaration,
)
from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector
from pyamplicol.loops.physical_orders import BornSquaredSector, PhysicalOrderLayout


def _metadata(**options):
    born, loop = (("QCD", 2),), (("QCD", 4),)
    return LoopExecutionMetadata(
        LoopSchemeDeclaration(
            dimensional_regularization="declared test scheme",
            external_states="four-dimensional",
            gamma5="none",
            renormalization="fixture",
            mass_scheme="fixture",
            finite_restoration="fixture",
            model_convention="metadata test only",
        ),
        PhysicalOrderLayout(
            (BornSquaredSector((("QCD", 4),), ((born, born),)),),
            (VirtualOrderSector((("QCD", 6),), (BornLoopPair(born, loop),)),),
            ("QCD",),
        ),
        "model",
        ("model",),
        ("double", "double-double", "arbitrary"),
        True,
        **options,
    )


@pytest.mark.parametrize(
    "correlations,expected",
    [(False, ORDINARY_INTERFERENCE), (True, DIRECTED_INTERFERENCE)],
)
def test_default_and_explicit_observable_declarations(correlations, expected):
    inferred = _metadata(correlations=correlations)
    explicit = _metadata(correlations=correlations, interference=expected)
    assert inferred == explicit
    assert asdict(inferred)["interference"] == expected
    assert inferred.epsilon_powers == (0, -1, -2)


@pytest.mark.parametrize("correlations", [False, True])
def test_explicit_wrong_observable_is_not_reinterpreted(correlations):
    opposite = ORDINARY_INTERFERENCE if correlations else DIRECTED_INTERFERENCE
    for value in (opposite, "", "unknown", False):
        with pytest.raises(ValueError, match="interference declaration"):
            _metadata(correlations=correlations, interference=value)


@pytest.mark.parametrize("correlations", [None, 0, 1, "directed"])
def test_default_does_not_coerce_correlation_capability(correlations):
    with pytest.raises(TypeError, match="capabilities must be booleans"):
        _metadata(correlations=correlations)


def test_changing_lane_requires_a_new_declaration():
    ordinary = _metadata()
    with pytest.raises(ValueError, match="interference declaration"):
        replace(ordinary, correlations=True)
    directed = replace(ordinary, correlations=True, interference=None)
    assert directed.interference == DIRECTED_INTERFERENCE
    assert ordinary.interference == ORDINARY_INTERFERENCE


@pytest.mark.parametrize("directed", [False, True])
def test_saved_observable_metadata_reaches_native_and_inspect(directed):
    """Use freshly published existing plans; no numerical execution is needed."""
    location = os.environ.get("PYAMPLICOL_PROJECTED_FLAVOR_FIXTURES")
    if not location:
        pytest.skip("freshly published projected flavour artifacts required")
    from pyamplicol import Runtime, _rusticol
    from pyamplicol._internal.versions import verify_native_module

    verify_native_module(_rusticol)
    kind = "spin" if directed else "ordinary"
    artifact = Path(location) / f"lc-{kind}-mapped-artifact"
    native = _rusticol._loop.NativeVirtualRuntime(str(artifact), process="ssgg")
    metadata = json.loads(native.metadata_json())
    expected = DIRECTED_INTERFERENCE if directed else ORDINARY_INTERFERENCE
    assert metadata["physics"]["interference"] == expected
    assert metadata["physics"]["correlations"] is directed
    assert (metadata["plan"].get("directed_projection") is not None) is directed
    assert Runtime.load(artifact, process="ssgg").inspect()["interference"] == expected
