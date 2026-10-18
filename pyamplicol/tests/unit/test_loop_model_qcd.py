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
from pyamplicol.loops.model_qcd import prepare_qcd_counterterm_basis
from pyamplicol.loops.wavefunction import match_external_field_counterterms
from pyamplicol.models.loop_import import load_raw_loop_model


@pytest.fixture(scope="module")
def source():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native QCD tensor controls require licensed Symbolica")
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not path:
        pytest.skip("set PYAMPLICOL_NLO_UFO_REFERENCE to loop_sm")
    from symbolica.community import hepkit as h

    return build_loop_model(load_raw_loop_model(path)), h


def _prepare(source, native, basis, **overrides):
    options = dict(
        entries=basis.entries,
        ordinary_basis_representatives=basis.ordinary_representatives,
        primitive_validations=basis.primitive_validations,
        tensor_representatives=basis.tensor_representatives,
    )
    options.update(overrides)
    return prepare_counterterm_model(source, native, **options)


def test_quark_gluon_ct_and_field_multiplicities(source):
    from ufo_model_loader.symbolica_processing import wrap_indices

    from pyamplicol.loops.model_expressions import parse_exact_ufo

    model, h = source
    native = h.Model.qcd_qed()
    basis = prepare_qcd_counterterm_basis(model, native)
    assert len(basis.ordinary_representatives) == 5
    assert len(basis.tensor_representatives) == 1
    representative = basis.tensor_representatives[0]
    assert representative.source_validation.lorentz == "P(-1,1)*Gamma(-1,2,1)"
    assert representative.native_lorentz == "P(-1,1)*Gamma(-1,1,2)"
    assert not basis.fermion_two_point_insertion_validated
    assert not basis.formal_nc_continuation_validated
    assert not basis.scalar_coefficients_validated
    prepared = _prepare(model, native, basis)
    loops = build_loop_process(
        prepared.process(["u", "u~"], ["g", "g"]),
        ordinary_model=native,
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
    )
    assert len(loops.born) == 3
    matched = match_counterterms(
        prepared,
        prepared.process(["u", "u~"], ["g", "g"], counterterms=True),
        loops,
        selected_corrections=["QCD"],
    )
    counts = Counter(i.rule.entries[0].vertex_name for i in matched.insertions)
    assert counts == {
        "V_R2GUU": 5,
        "V_UVGUU": 35,
        "V_R23G": 7,
        "V_UV1eps3G": 7,
        "V_R2GG": 7,
        "V_R2UU": 2,
    }
    assert all(i.rule.has_validated_tensor_basis for i in matched.insertions)
    prepared_rules = {
        r["name"]: r for r in json.loads(prepared.native.to_json())["vertex_rules"]
    }
    for insertion in matched.insertions:
        if insertion.rule.entries[0].vertex_name == "V_R2UU":
            assert insertion.rule.tensor_representatives == (representative,)
            assert (
                insertion.rule.entries[0].lorentz
                == representative.source_validation.lorentz
            )
            lorentz_name = prepared_rules[insertion.rule.name]["lorentz_structures"][0]
            assert (
                prepared.native.lorentz_structure(lorentz_name).structure
                - wrap_indices(parse_exact_ufo(representative.native_lorentz))
            ).expand() == 0
    fields = match_external_field_counterterms(
        model, loops, selected_corrections=["QCD"]
    )
    assert len(fields.insertions) == 18
    assert Counter(i.rule.alternatives for i in fields.insertions) == {
        ((4,),): 6,
        ((5,),): 6,
        ((6,),): 6,
    }


@pytest.mark.parametrize(
    "kind", ["source_vertex", "source_slash", "native_vertex", "native_coupling"]
)
def test_quark_catalogue_rejects_changed_source_or_calibrated_native_basis(
    source, kind
):
    model, h = source
    native = h.Model.qcd_qed()
    if kind.startswith("source"):
        entry = next(
            e
            for e in model.kernels
            if (
                e.kind == "tree" and e.particle_names == ("u~", "u", "G")
                if kind == "source_vertex"
                else e.vertex_name == "V_R2UU"
            )
        )
        changed = replace(entry, lorentz="2*(" + entry.lorentz + ")")
        model = replace(
            model, kernels=tuple(changed if e is entry else e for e in model.kernels)
        )
    else:
        payload = json.loads(native.to_json())
        if kind == "native_vertex":
            entry = next(
                e for e in payload["lorentz_structures"] if e["name"] == "FFV1"
            )
            entry["structure"] = "2*(" + entry["structure"] + ")"
        else:
            entry = next(e for e in payload["couplings"] if e["name"] == "GC_11")
            entry["expression"] = "2*(" + entry["expression"] + ")"
        native = h.Model.from_json(json.dumps(payload))
    with pytest.raises(
        (NotImplementedError, ValueError), match=r"control|representative"
    ):
        prepare_qcd_counterterm_basis(model, native)


def test_explicit_tensor_representative_rejects_stale_source_and_duplicates(source):
    model, h = source
    native = h.Model.qcd_qed()
    basis = prepare_qcd_counterterm_basis(model, native)
    representative = basis.tensor_representatives[0]
    invalid = replace(
        representative,
        source_validation=replace(
            representative.source_validation, lorentz="Identity(1,2)"
        ),
    )
    with pytest.raises(ValueError, match="source lorentz"):
        _prepare(model, native, basis, tensor_representatives=[invalid])
    with pytest.raises(ValueError, match="duplicate"):
        _prepare(
            model,
            native,
            basis,
            tensor_representatives=[representative, representative],
        )
    with pytest.raises(ValueError, match="calibration"):
        replace(representative, calibration="")
    with pytest.raises(ValueError, match="native Lorentz"):
        replace(representative, native_lorentz="")
