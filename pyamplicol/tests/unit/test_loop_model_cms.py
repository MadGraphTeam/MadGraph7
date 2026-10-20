# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import cmath
import io
import json
import tokenize
from decimal import Decimal, localcontext

import pytest

from pyamplicol.loops.model_cms import (
    _rename_expression,
    prepare_gmu_complex_mass_declarations,
)
from pyamplicol.models.loop_import import RawLoopModel, load_raw_loop_model

pytestmark = pytest.mark.loop_fast


def test_cms_identifier_rewrite_preserves_numeric_lexemes_and_other_tokens():
    source = "MW+MW2+cmath.MW+'MW'+1.0000000000000000000000000001e-200"
    result = _rename_expression(source, {"MW": "CMASS_MW"})

    def tokens(text):
        return [
            (x.type, x.string)
            for x in tokenize.generate_tokens(io.StringIO(text).readline)
        ]

    original = tokens(source)
    expected = list(original)
    expected[0] = (tokenize.NAME, "CMASS_MW")
    assert tokens(result) == expected


@pytest.fixture(scope="module")
def cms(loop_gmu_ufo):
    path = str(loop_gmu_ufo)
    source = load_raw_loop_model(path, restriction="full")
    original = source.to_dict()
    result = prepare_gmu_complex_mass_declarations(source)
    assert result.original is source and source.to_dict() == original
    return result


def test_derived_source_retains_records_and_one_mass_for_all_multiplet_partners(cms):
    raw = cms.derived
    for collection in (
        "vertices",
        "counterterm_vertices",
        "lorentz_structures",
        "functions",
    ):
        assert raw.collections[collection] == cms.original.collections[collection]
    assert (
        len(raw.collections["parameters"])
        == len(cms.original.collections["parameters"]) + 4
    )
    assert raw.record("particles", "G-")["fields"]["goldstoneboson"] is True
    assert cms.original.record("particles", "G-")["fields"]["goldstoneboson"] is False
    for record in raw.collections["particles"]:
        original = cms.original.collections["particles"][record["id"]]
        mass = original["fields"]["mass"]["name"]
        assert record["fields"]["mass"]["name"] == cms.mass_aliases.get(mass, mass)
        # Ghost widths remain original metadata; their mass itself is continued.
        assert record["fields"]["width"] == original["fields"]["width"]
    for mass, alias in cms.mass_aliases.items():
        record = raw.record("parameters", alias)
        value = raw.exact_value("parameters", record["id"])["expression"]
        assert mass in value and alias not in value
        assert record["fields"]["type"] == "complex"
        assert raw.record("parameters", mass)["fields"]["nature"] == "external"
        assert raw.record("parameters", mass)["fields"]["type"] == "real"
    assert raw.record("parameters", "sw")["fields"]["type"] == "complex"
    assert raw.record("parameters", "ymt")["fields"]["nature"] == "internal"
    assert cms.tied_yukawas["ymt"] == "CMASS_MT"
    assert "lhablock" not in raw.record("parameters", "ymt")["fields"]
    copied = RawLoopModel.from_dict(json.loads(json.dumps(raw.to_dict())))
    assert copied.to_dict() == raw.to_dict()
    with pytest.raises(ValueError, match="already derived"):
        prepare_gmu_complex_mass_declarations(raw)
    changed = cms.original.to_dict()
    changed["collections"]["parameters"][cms.original.record("parameters", "MW")["id"]][
        "fields"
    ]["type"] = "complex"
    with pytest.raises(NotImplementedError, match="real external"):
        prepare_gmu_complex_mass_declarations(RawLoopModel.from_dict(changed))


@pytest.fixture(scope="module")
def structure(cms):
    pytest.importorskip("symbolica")
    from pyamplicol.loops.covariant_model import build_covariant_structure

    return build_covariant_structure(
        cms.derived, fixed_parameters={"Me": "0", "MM": "0"}
    )


def test_structural_propagators_use_the_same_derived_mass_symbols(structure):
    from symbolica import S

    for pdg, alias in (
        (23, "CMASS_MZ"),
        (24, "CMASS_MW"),
        (9000003, "CMASS_MW"),
        (251, "CMASS_MW"),
        (6, "CMASS_MT"),
    ):
        assert structure.native.particle_by_pdg(pdg).mass == S(f"UFO::{alias}")
    assert structure.native.particle_by_pdg(11).is_massless
    payload = json.loads(structure.native.to_json())
    top = next(p for p in payload["propagators"] if p["particle"] == "t")
    assert "CMASS_MT" in top["numerator"] and "CMASS_MT" in top["denominator"]
    ghost = next(p for p in payload["propagators"] if p["particle"] == "ghWp")
    assert "CMASS_MW" in ghost["denominator"]
    assert "ymt" not in structure.required_bindings


@pytest.mark.parametrize(
    "arithmetic,precision,bit,tolerance",
    [
        ("double", None, 0, "3e-12"),
        ("double-double", None, 80, "2e-27"),
        ("arbitrary", 80, 180, "1e-72"),
    ],
)
def test_native_pole_masses_weak_angle_and_source_ct_continuation(
    loop_native, structure, arithmetic, precision, bit, tolerance
):
    from pyamplicol.loops.scalar_bindings import NativeScalarBindings

    outputs = (
        "CMASS_MW",
        "CMASS_MZ",
        "sw",
        "cw",
        "ee",
        "ymt",
        "CMASS_MT",
        "CMSParam",
        "cWcft_UV_EW_R",
    )
    plan = structure.scalar_bindings(tuple((name, 0) for name in outputs))
    with localcontext() as context:
        context.prec = 110
        mass = Decimal("80.419") + (Decimal(2) ** -bit if bit else 0)
        width = Decimal("2.09291")
        runtime = NativeScalarBindings(
            plan,
            arithmetic=arithmetic,
            precision=precision,
            parameters={
                "MW": str(mass),
                "WW": str(width),
                "MZ": "91.188",
                "WZ": "2.49877",
                "MT": "173.3",
                "WT": "1.37758",
                "Gf": "0.0000116639",
                "MU_R": "1000",
            },
        )
        values = runtime.values.to_decimal()
        real, imaginary = map(Decimal, values[0])
        assert abs(real * real - imaginary * imaginary - mass * mass) < Decimal(
            tolerance
        )
        assert abs(2 * real * imaginary + mass * width) < Decimal(tolerance)
        assert values[5] == values[6]
        assert tuple(map(Decimal, values[7])) == (1, 0)
        assert Decimal(values[2][1]) != 0 and Decimal(values[3][1]) != 0
        assert abs(Decimal(values[4][1])) < Decimal(tolerance)
    mw, mz, sw, cw, ee = [complex(*map(float, pair)) for pair in values[:5]]
    assert abs(sw**2 - (1 - mw**2 / mz**2)) < 2e-15
    expected = -(ee**2 * sw**2 * (-1 + 2 * cmath.log(1000**2 / mz**2))) / (
        72 * cw**2 * cmath.pi**2
    )
    assert abs(complex(*map(float, values[8])) - expected) < 1e-17
    assert abs(expected.imag) > 1e-9
    with pytest.raises(ValueError, match="unknown external"):
        runtime.update({"ymt": "190"})
    # No current-width specialization: zero-width updates reuse the same plan.
    zero = runtime.update({"WW": "0", "WZ": "0", "WT": "0"}).values.to_decimal()
    assert all(Decimal(zero[i][1]) == 0 for i in (0, 1, 2, 3, 4, 5, 6))
