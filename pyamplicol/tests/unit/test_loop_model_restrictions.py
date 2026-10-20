# SPDX-License-Identifier: 0BSD
# ruff: noqa: E501
from __future__ import annotations

from decimal import Decimal, localcontext

import pytest

from pyamplicol.loops.model_restrictions import (
    _exact_card_entries,
    prepare_loop_restrictions,
)
from pyamplicol.models.loop_import import load_raw_loop_model

pytestmark = pytest.mark.loop_fast


@pytest.fixture
def source(tmp_path):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("exact source controls require licensed Symbolica")
    path = tmp_path / "restriction_fixture"
    path.mkdir()
    (path / "__init__.py").write_text(
        "from . import parameters, particles\n"
        "from .object_library import all_parameters, all_particles\ngauge=[1]\n"
    )
    (path / "object_library.py").write_text(
        "all_parameters=[]\nall_particles=[]\n"
        "class Parameter:\n"
        " def __init__(self,name,nature,type,value,lhablock=None,lhacode=None):\n"
        "  self.name=name;self.nature=nature;self.type=type;self.value=value\n"
        "  self.lhablock=lhablock;self.lhacode=lhacode;all_parameters.append(self)\n"
        "class Particle:\n"
        " def __init__(self,pdg_code,name,antiname,spin,color,mass,width,texname,antitexname,charge):\n"
        "  self.pdg_code=pdg_code;self.name=name;self.antiname=antiname\n"
        "  self.spin=spin;self.color=color;self.mass=mass;self.width=width\n"
        "  self.texname=texname;self.antitexname=antitexname;self.charge=charge\n"
        "  all_particles.append(self)\n"
    )
    (path / "parameters.py").write_text(
        "from .object_library import Parameter\n"
        "ZERO=Parameter(name='ZERO',nature='internal',type='real',value='0')\n"
        "M=Parameter(name='M',nature='external',type='real',value=2.0,lhablock='MASS',lhacode=[991])\n"
        "W=Parameter(name='W',nature='external',type='real',value=0.0,lhablock='DECAY',lhacode=[991])\n"
        "P=Parameter(name='P',nature='external',type='real',value=0.1,lhablock='INPUT',lhacode=[1])\n"
        "T=Parameter(name='T',nature='external',type='real',value=1.0,lhablock='INPUT',lhacode=[2])\n"
        "Z=Parameter(name='Z',nature='external',type='real',value=0.0,lhablock='INPUT',lhacode=[3])\n"
        "X=Parameter(name='X',nature='external',type='complex',value='complex(2,3)',lhablock='MIX',lhacode=[1,2])\n"
    )
    (path / "particles.py").write_text(
        "from .object_library import Particle\nfrom . import parameters as P\n"
        "s=Particle(pdg_code=991,name='s',antiname='s',spin=1,color=1,mass=P.M,width=P.W,texname='s',antitexname='s',charge=0)\n"
    )
    (path / "restrict_default.dat").write_text(
        "BLOCK MASS # names/comments do not define the binding\n 991 -0.0D+00\n"
        "BLOCK INPUT\n 1 +1.000000000000000000000000000001\n 2 1D-400\n"
        "BLOCK MIX\n 1 2 -0.125\nBLOCK IMMIX\n 1 2 +0.375\n"
        "DECAY 991 0.0 # not a branching-ratio table\n"
    )
    return path


def test_restriction_exact_defaults_and_structural_zero_are_distinct(source):
    from pyamplicol.loops.covariant_model import build_covariant_structure

    raw = load_raw_loop_model(source)
    before = raw.to_dict()
    prepared = prepare_loop_restrictions(raw)
    assert raw.to_dict() == before and prepared.original is raw
    assert dict(prepared.fixed_parameters) == dict.fromkeys(("M", "W", "Z"), ("0", "0"))
    assert (
        prepared.derived.exact_value("parameters", 4)["expression"]
        == "complex(1E-400,0)"
    )
    structure = build_covariant_structure(prepared.derived)
    assert structure.native.particle("s").is_massless
    assert "M" not in structure.required_bindings and "T" in structure.required_bindings
    with pytest.raises(ValueError, match="contradicts"):
        build_covariant_structure(prepared.derived, fixed_parameters={"M": "1"})
    with pytest.raises(NotImplementedError, match="unapplied"):
        build_covariant_structure(raw)
    unsimplified = prepare_loop_restrictions(raw, simplify=False)
    assert not unsimplified.fixed_parameters
    assert (
        not build_covariant_structure(unsimplified.derived)
        .native.particle("s")
        .is_massless
    )
    with pytest.raises(ValueError, match="already"):
        prepare_loop_restrictions(prepared.derived)


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, "3e-16"),
        ("double-double", None, "2e-32"),
        ("arbitrary", 90, "2e-88"),
    ],
)
def test_restricted_scalar_defaults_keep_original_digits_in_native_carriers(
    loop_native, source, arithmetic, precision, tolerance
):
    from pyamplicol.loops.covariant_model import build_covariant_structure
    from pyamplicol.loops.scalar_bindings import NativeScalarBindings

    prepared = prepare_loop_restrictions(load_raw_loop_model(source))
    structure = build_covariant_structure(prepared.derived)
    state = NativeScalarBindings(
        structure.scalar_bindings((("P", 0), ("X", 0), ("M", 0))),
        arithmetic=arithmetic,
        precision=precision,
    )
    values = state.values.to_decimal()
    with localcontext() as context:
        context.prec = 110
        assert abs(
            Decimal(values[0][0]) - Decimal("1.000000000000000000000000000001")
        ) < Decimal(tolerance)
        if arithmetic != "double":
            assert Decimal(values[0][0]) != 1
    assert tuple(map(Decimal, values[1])) == (Decimal("-0.125"), Decimal("0.375"))
    assert tuple(map(Decimal, values[2])) == (0, 0)
    with pytest.raises(ValueError, match="fixed during generation"):
        state.update({"M": "1"})


@pytest.mark.parametrize(
    "card,match",
    [
        (
            "BLOCK X\n1 1.0000000000000000000001\n1 1.0000000000000000000002\n",
            "duplicate",
        ),
        ("BLOCK X\n1 1\nBLOCK X\n2 2\n", "duplicate"),
        ("BLOCK X Q=100\n1 1\n", "scale-dependent"),
        ("BLOCK X\n1 scan:[1,2]\n", "decimal"),
        ("BLOCK X\n1 NaN\n", "decimal"),
        ("BLOCK X\n1 1 \\\n2\n", "decimal|unsupported"),
        ("DECAY 1 Auto\n", "decimal"),
        ("DECAY 1 0\n1.0 2 2 -2\n", "decay tables"),
        ("unrecognized initial line\nBLOCK X\n1 2\n", "unsupported"),
        ("BLOCK X\n1 garbage 2\n", "address|unsupported"),
        ("BLOCK X garbage\n1 2\n", "header|unsupported"),
        ("BLOCK X\n1 1_000\n", "decimal"),
        ("BLOCK X\n1_0 2\n", "address"),
    ],
)
def test_unsupported_or_ambiguous_cards_fail_without_binary64_certification(
    card, match
):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _exact_card_entries(card)


def test_restriction_uses_source_lha_identity_and_rejects_unknown_entries(source):
    card = source / "restrict_default.dat"
    card.write_text("BLOCK UNKNOWN\n 1 0 # M\n")
    with pytest.raises(ValueError, match="no external"):
        prepare_loop_restrictions(load_raw_loop_model(source))
    card.write_text("BLOCK IMINPUT\n 1 0\n")
    with pytest.raises(ValueError, match="complex parameter"):
        prepare_loop_restrictions(load_raw_loop_model(source))


def test_absent_card_preserves_raw_identity(source):
    (source / "restrict_default.dat").unlink()
    raw = load_raw_loop_model(source)
    prepared = prepare_loop_restrictions(raw)
    assert prepared.derived is prepared.original is raw
    assert not prepared.fixed_parameters


@pytest.mark.parametrize("block", ["MIX", "IMMIX"])
def test_partial_complex_card_cannot_erase_an_unspecified_component(source, block):
    (source / "restrict_default.dat").write_text(f"BLOCK {block}\n1 2 4\n")
    raw = load_raw_loop_model(source)
    original = raw.to_dict()
    with pytest.raises(NotImplementedError, match="both real and imaginary"):
        prepare_loop_restrictions(raw)
    assert raw.to_dict() == original
