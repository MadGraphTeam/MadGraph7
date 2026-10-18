# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import replace

import pytest

from pyamplicol.loops.counterterms import match_counterterms, prepare_counterterm_model
from pyamplicol.loops.graphs import build_loop_process
from pyamplicol.loops.model import build_loop_model
from pyamplicol.loops.model_gluons import prepare_gluon_counterterm_basis
from pyamplicol.loops.wavefunction import match_external_field_counterterms
from pyamplicol.models.loop_import import load_raw_loop_model


@pytest.fixture(scope="module")
def source():
    s = pytest.importorskip("symbolica")
    if not s.is_licensed():
        pytest.skip("native local gluon rules require licensed Symbolica")
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not path:
        pytest.skip("set PYAMPLICOL_NLO_UFO_REFERENCE to loop_sm")
    from symbolica.community import hepkit as h

    return build_loop_model(load_raw_loop_model(path)), h


def test_source_gluon_basis_matches_exact_native_ordinary_representatives(source):
    model, h = source
    native = h.Model.qcd_qed()
    basis = prepare_gluon_counterterm_basis(model, native)
    assert {
        (r.native_rule, r.color_index, r.lorentz_index)
        for r in basis.ordinary_representatives
    } == {("V_36", 0, 0), ("V_37", 0, 0), ("V_37", 1, 1), ("V_37", 2, 2)}
    assert len(basis.entries) == 88 and len(basis.primitive_validations) == 66
    assert Counter(v.conformance_control for v in basis.primitive_validations) == {
        "local-four-gluon-r2-aloha-and-mg-color-control": 57,
        "local-gluon-two-point-aloha-control": 9,
    }
    assert not basis.formal_nc_continuation_validated
    assert not basis.scalar_coefficients_validated
    prepared = prepare_counterterm_model(
        model,
        native,
        entries=basis.entries,
        ordinary_basis_representatives=basis.ordinary_representatives,
        primitive_validations=basis.primitive_validations,
    )
    assert all(rule.has_validated_tensor_basis for rule in prepared.rules.values())


@pytest.mark.parametrize(
    "kind",
    ["source_ordinary", "source_counterterm", "native_lorentz", "native_coupling"],
)
def test_gluon_control_rejects_modified_tensor_or_scalar_correspondence(source, kind):
    model, h = source
    native = h.Model.qcd_qed()
    if kind.startswith("source"):
        entry = next(
            entry
            for entry in model.kernels
            if entry.vertex_name == ("V_3" if kind == "source_ordinary" else "V_R2RGA")
        )
        changed = replace(entry, lorentz="2*(" + entry.lorentz + ")")
        model = replace(
            model,
            kernels=tuple(
                changed if candidate is entry else candidate
                for candidate in model.kernels
            ),
        )
    else:
        payload = json.loads(native.to_json())
        if kind == "native_lorentz":
            value = next(
                value
                for value in payload["lorentz_structures"]
                if value["name"] == "VVV1"
            )
            value["structure"] = "2*(" + value["structure"] + ")"
        else:
            value = next(
                value for value in payload["couplings"] if value["name"] == "GC_10"
            )
            value["expression"] = "2*(" + value["expression"] + ")"
        native = h.Model.from_json(json.dumps(payload))
    with pytest.raises(
        (ValueError, NotImplementedError), match=r"control|representative"
    ):
        prepare_gluon_counterterm_basis(model, native)


def test_gluon_process_ct_and_external_field_multiplicities(source):
    model, h = source
    native = h.Model.qcd_qed()
    basis = prepare_gluon_counterterm_basis(model, native)
    prepared = prepare_counterterm_model(
        model,
        native,
        entries=basis.entries,
        ordinary_basis_representatives=basis.ordinary_representatives,
        primitive_validations=basis.primitive_validations,
    )
    loops = build_loop_process(
        prepared.process(["g", "g"], ["g", "g"]),
        ordinary_model=native,
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
    )
    matched = match_counterterms(
        prepared,
        prepared.process(["g", "g"], ["g", "g"], counterterms=True),
        loops,
        selected_corrections=["QCD"],
    )
    assert len(loops.born) == 4 and len(loops.loops) == 186
    # Six cubic attachment sites, three internal gauge lines, and one quartic
    # site, each with six quark plus one gauge-loop alternative. Gauge source
    # groups include their ghost contribution; no extra ghost multiplicity.
    assert Counter(
        insertion.rule.couplings[0][1].vertex_name for insertion in matched.insertions
    ) == {"V_R23G": 42, "V_UV1eps3G": 42, "V_R2GG": 21, "V_R2RGA": 7, "V_UV1eps4G": 7}
    assert all(
        insertion.rule.has_validated_tensor_basis for insertion in matched.insertions
    )
    fields = match_external_field_counterterms(
        model, loops, selected_corrections=["QCD"]
    )
    assert len(fields.insertions) == 48
    assert Counter(insertion.rule.alternatives for insertion in fields.insertions) == {
        ((4,),): 16,
        ((5,),): 16,
        ((6,),): 16,
    }
    assert all(
        len(
            {
                insertion.external_index
                for insertion in fields.insertions
                if insertion.born.id == born.id
                and insertion.rule.loop_group_index == group
            }
        )
        == 4
        for born in loops.born
        for group in range(3)
    )
    without_top = match_external_field_counterterms(
        model, loops, selected_corrections=["QCD"], forbidden_particles={6}
    )
    assert len(without_top.insertions) == 32
