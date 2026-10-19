# SPDX-License-Identifier: 0BSD
"""Saved actual charged-vector sources retain all three physical states."""

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from .test_loop_portable import _rewrite


@pytest.fixture(scope="module")
def massive_sources():
    fixture = os.environ.get("PYAMPLICOL_MASSIVE_VECTOR_FIXTURES")
    if not fixture:
        pytest.skip("retained actual charged-vector source artifacts required")
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    verify_native_module(_rusticol)
    assert load_loop_native() is _rusticol._loop
    return Path(fixture), _rusticol._loop


def test_saved_charged_source_identity_states_and_reciprocity(
    massive_sources, tmp_path
):
    root, native = massive_sources
    source = root / "outgoing-spin.pacbin"
    native.PortableLoopPlan(str(source))
    actual = json.loads((root / "model-vector-contracts.json").read_text())

    def malformed_pair(data):
        # Substitute the actual opposite charged SourceIR in the second vector
        # port, then break its declared reciprocal relation. This is a malformed
        # cold descriptor control, not another physical scattering process.
        data["external_pdgs"][3] = -24
        data["state_legs"][3]["massless"] = False
        for saved in data["spin_projections"]:
            if 4 in saved["legs"]:
                contract = deepcopy(actual["-24"])
                contract["identity"]["species_id"] = "foreign-species"
                saved["source_contracts"][saved["legs"].index(4)] = contract

    def different_class_identity(data):
        saved = data["spin_projections"][-1]
        saved["source_contracts"][0]["identity"]["species_id"] = "foreign-species"

    mutations = (
        (lambda d: d["state_legs"][2].update(massless=True), "physical vector"),
        (
            lambda d: d["spin_projections"][0]["source_contracts"][0]["states"].pop(1),
            "physical vector",
        ),
        (
            lambda d: d["spin_projections"][0]["source_contracts"][0]["identity"].update(
                anti_pdg_label=24
            ),
            "antiparticle relation",
        ),
        (different_class_identity, "across saved joint classes"),
        (malformed_pair, "non-involutive particle/antiparticle"),
    )
    for index, (mutate, message) in enumerate(mutations):
        path = tmp_path / f"invalid-{index}.pacbin"
        _rewrite(source, path, mutate)
        with pytest.raises(ValueError, match=message):
            native.PortableLoopPlan(str(path))
