# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import shutil
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import pyamplicol
from pyamplicol.models.contracts import CompiledModelIR
from pyamplicol.models.loading import (
    CompiledModel,
    ModelCompileOptions,
    _load_external_model,
)
from pyamplicol.models.loop_import import (
    RawLoopModel,
    capture_raw_loop_model,
    load_raw_loop_model,
)

pytestmark = pytest.mark.loop_fast


def _nlo_module() -> ModuleType:
    module = ModuleType("test_nlo")
    zero = SimpleNamespace(name="ZERO", nature="internal", type="real", value="0")
    dz = SimpleNamespace(
        name="dZ",
        nature="internal",
        type="complex",
        value={
            -2: "g**2",
            -1: "complex(0,1)*g**2",
            0: "g**2*reglog(M**2/muR**2)",
        },
    )
    scalar = SimpleNamespace(name="GC_UV", value="g*dZ", order={"QCD": 3})
    series = SimpleNamespace(
        name="GC_series", value={-1: "g**3", 0: "2*g**3"}, order={"QCD": 3}
    )
    gluon = SimpleNamespace(name="g", pdg_code=21, spin=3, mass=zero, width=zero)
    ghost = SimpleNamespace(name="gh", pdg_code=9000001, spin=-1, mass=zero, width=zero)
    goldstone = SimpleNamespace(
        name="G0",
        pdg_code=250,
        spin=1,
        mass=zero,
        width=zero,
        goldstoneboson=False,
        GoldstoneBoson=True,
    )
    gluon.counterterm = {(1, 0): {-1: "dZ"}}
    gluon.loop_particles = [[ghost]]
    lorentz = SimpleNamespace(
        name="VVV", spins=[3, 3, 3], structure="P(1,2)*Metric(2,3)"
    )
    vertex = SimpleNamespace(
        name="V_duplicate",
        type="UV",
        particles=[gluon, gluon, gluon],
        color=["f(1,2,3)"],
        lorentz=[lorentz],
        loop_particles=[[[gluon], [ghost]], [[gluon, ghost]]],
        couplings={(0, 0, 0): scalar, (0, 0, 1): series},
    )
    second = SimpleNamespace(
        **{**vars(vertex), "type": "R2", "particles": [goldstone, gluon, gluon]}
    )
    order = SimpleNamespace(
        name="QCD", expansion_order=99, hierarchy=1, perturbative_expansion=1
    )
    module.all_parameters = [zero]
    module.all_CTparameters = [dz]
    module.all_couplings = [scalar, series]
    module.all_particles = [gluon, ghost, goldstone]
    module.all_lorentz = [lorentz]
    module.all_vertices = []
    module.all_CTvertices = [vertex, second]
    module.all_orders = [order]
    module.gauge = [0, 1]
    module.regularization_scheme = "declared-test-scheme"
    return module


def test_preserves_laurent_references_ct_indices_and_loop_groups() -> None:
    module = _nlo_module()
    captured = capture_raw_loop_model(module)
    data = captured.to_dict()
    parameters = data["collections"]["counterterm_parameters"]
    assert parameters[0]["fields"]["value"] == {
        "kind": "laurent",
        "terms": [
            [power, value] for power, value in module.all_CTparameters[0].value.items()
        ],
    }
    # A scalar expression containing a CT parameter is NOT an epsilon-zero term.
    assert captured.record("couplings", "GC_UV")["fields"]["value"] == {
        "kind": "expression",
        "expression": "g*dZ",
    }
    vertices = data["collections"]["counterterm_vertices"]
    assert [item["id"] for item in vertices] == [0, 1]
    assert [item["name"] for item in vertices] == ["V_duplicate", "V_duplicate"]
    with pytest.raises(ValueError, match=r"ambiguous.*record ID"):
        captured.record("counterterm_vertices", "V_duplicate")
    entries = vertices[0]["fields"]["couplings"]["entries"]
    assert [entry[0] for entry in entries] == [
        {"kind": "tuple", "items": [0, 0, 0]},
        {"kind": "tuple", "items": [0, 0, 1]},
    ]
    assert [entry[1]["name"] for entry in entries] == ["GC_UV", "GC_series"]
    groups = vertices[0]["fields"]["loop_particles"]
    assert [
        [[p["name"] for p in alternative] for alternative in group] for group in groups
    ] == [[["g"], ["gh"]], [["g", "gh"]]]
    assert data["collections"]["particles"][0]["fields"]["counterterm"]["entries"][0][
        0
    ] == {"kind": "tuple", "items": [1, 0]}
    assert captured.capabilities["ghost_particles"] == ("gh",)
    assert captured.capabilities["goldstone_particles"] == ("G0",)
    assert captured.capabilities["perturbative_orders"] == ("QCD",)
    assert "loop_runtime_supported" not in captured.capabilities
    assert captured.capabilities["model_r2_complete"] is None
    assert captured.metadata["regularization_scheme"] == "declared-test-scheme"

    module.all_CTparameters[0].value[-1] = "changed after capture"
    assert RawLoopModel.from_dict(json.loads(json.dumps(data))).to_dict() == data
    assert captured.to_dict() == data
    with pytest.raises(TypeError):
        captured.record("couplings", "GC_UV")["fields"]["value"]["expression"] = "0"


def test_custom_propagators_and_ufo2_function_types_are_preserved() -> None:
    module = _nlo_module()
    module.all_particles[0].propagator = SimpleNamespace(
        name="custom", numerator="-complex(0,1)*Metric(1,2)", denominator="P(1)**2-M**2"
    )
    module.all_functions = [
        SimpleNamespace(
            name="f", arguments=("z",), argstype=(complex,), expression="z.conjugate()"
        )
    ]
    captured = capture_raw_loop_model(module)
    assert (
        captured.record("particles", "g")["fields"]["propagator"]["collection"]
        == "propagators"
    )
    assert (
        captured.record("propagators", "custom")["fields"]["denominator"]
        == "P(1)**2-M**2"
    )
    argstype = captured.to_dict()["collections"]["functions"][0]["fields"]["argstype"]
    assert argstype == {
        "kind": "tuple",
        "items": [{"kind": "type", "name": "builtins.complex"}],
    }


def _write_ufo(path: Path, expression: str) -> Path:
    path.mkdir(parents=True)
    (path / "object_library.py").write_text(
        "from types import SimpleNamespace\n"
        "all_CTparameters = [SimpleNamespace(name='dZ', nature='internal', "
        f"type='real', value={{-1: {expression!r}}})]\n",
        encoding="utf-8",
    )
    (path / "__init__.py").write_text(
        "import object_library\nall_CTparameters = object_library.all_CTparameters\n",
        encoding="utf-8",
    )
    return path


def test_same_name_ufo_imports_are_isolated_and_restrictions_remain_unapplied(
    tmp_path: Path, monkeypatch
) -> None:
    first = _write_ufo(tmp_path / "first" / "same_name", "g**2")
    second = _write_ufo(tmp_path / "second" / "same_name", "h**2")
    restriction = "BLOCK MASS\n  6  0.0\n"
    (first / "restrict_default.dat").write_text(restriction, encoding="utf-8")
    previous_path = sys.path[:]
    sentinel = ModuleType("object_library")
    monkeypatch.setitem(sys.modules, "object_library", sentinel)
    for path, expression in ((first, "g**2"), (second, "h**2"), (first, "g**2")):
        model = load_raw_loop_model(path)
        assert model.record("counterterm_parameters", "dZ")["fields"]["value"][
            "terms"
        ] == ((-1, expression),)
        assert sys.modules["object_library"] is sentinel
        assert "same_name" not in sys.modules
        assert sys.path == previous_path
    assert model.metadata["restriction"] == {
        "applied": False,
        "name": "default",
        "card": restriction,
    }
    assert not tuple(tmp_path.rglob("*.pyc"))
    assert (
        load_raw_loop_model(first, restriction="none").metadata["restriction"]["card"]
        is None
    )


def test_failed_ufo_capture_restores_import_environment(tmp_path: Path) -> None:
    source = _write_ufo(tmp_path / "broken_ufo", "g**2")
    (source / "__init__.py").write_text(
        "import object_library\nraise RuntimeError('broken UFO')\n", encoding="utf-8"
    )
    previous_path = sys.path[:]
    previous_bytecode = sys.dont_write_bytecode
    with pytest.raises(RuntimeError, match="broken UFO"):
        load_raw_loop_model(source)
    assert "broken_ufo" not in sys.modules
    assert sys.path == previous_path
    assert sys.dont_write_bytecode == previous_bytecode


def test_capture_precedes_tree_loader_loss(tmp_path: Path, monkeypatch) -> None:
    from ufo_model_loader import commands

    from pyamplicol.models import loading

    source = _write_ufo(tmp_path / "before_loss", "g**2")

    class TreeModel:
        raw_ufo_error = None
        parameters = ()
        couplings = ()

        def wrap_indices_in_lorentz_structures(self):
            pass

        def get_model_functions(self):
            return {}

        def get_model_variables(self):
            return {}

        def to_json(self, _look):
            return '{"name":"tree-only","propagators":[]}'

    def normalize(*_args, **_kwargs):
        from ufo_model_loader.raw import (
            capture_raw_ufo_model,
            imported_ufo_module,
        )

        # The authoritative loader captures before deriving its tree view.
        model = TreeModel()
        with imported_ufo_module(source) as module:
            model.raw_ufo = capture_raw_ufo_model(module)
            module.all_CTparameters[0].value.clear()
        return model, {}

    monkeypatch.setattr(commands, "load_model", normalize)
    monkeypatch.setattr(loading, "_classify_external_propagators", lambda _model: {})
    result = _load_external_model(source, options=ModelCompileOptions())
    assert result["model"]["name"] == "tree-only"
    assert RawLoopModel.from_dict(result["raw_loop_model"]).record(
        "counterterm_parameters", "dZ"
    )["fields"]["value"]["terms"] == ((-1, "g**2"),)


def test_unsupported_extra_metadata_keeps_tree_import_but_rejects_loop_capture(
    tmp_path: Path,
) -> None:
    packaged = (
        Path(pyamplicol.__file__).parent / "assets" / "models" / "ufo" / "scalars"
    )
    source = tmp_path / "tree_with_extra_metadata"
    shutil.copytree(packaged, source, ignore=shutil.ignore_patterns("__pycache__"))
    with (source / "__init__.py").open("a", encoding="utf-8") as stream:
        stream.write("\nextra_metadata = object()\n")

    # The real normalized loader accepts this model and ignores the extra field.
    result = _load_external_model(source, options=ModelCompileOptions())
    assert result["model"]["name"] == "tree_with_extra_metadata"
    assert result["model"]["vertex_rules"]
    assert "raw_loop_model" not in result
    assert "model.extra_metadata" in result["raw_loop_import_error"]
    with pytest.raises(ValueError, match=r"model\.extra_metadata"):
        load_raw_loop_model(source)


def test_numeric_source_precision_is_not_certified(tmp_path: Path) -> None:
    digits = "0.123456789012345678901234567890123456789"
    source = _write_ufo(tmp_path / "numeric_literals", digits)
    with (source / "object_library.py").open("a", encoding="utf-8") as stream:
        stream.write(
            f"all_parameters = [SimpleNamespace(name='literal', value={digits})]\n"
        )
    model = load_raw_loop_model(source)
    literal = model.record("parameters", "literal")["fields"]["value"]["expression"]
    assert literal == repr(float(digits))
    assert literal != digits
    assert model.record("counterterm_parameters", "dZ")["fields"]["value"]["terms"] == (
        (-1, digits),
    )
    assert model.capabilities["expression_strings_preserved"] is True
    assert model.capabilities["numeric_literal_source_preserved"] is False
    assert "arbitrary_precision_inputs_validated" not in model.capabilities


def _write_literal_ufo(path: Path, declarations: str, *, constructor_suffix="") -> Path:
    path.mkdir()
    classes = []
    for name, registry in (
        ("Parameter", "all_parameters"),
        ("CTParameter", "all_CTparameters"),
        ("Coupling", "all_couplings"),
    ):
        classes.append(
            f"{registry} = []\nclass {name}:\n"
            "    def __init__(self, name, value):\n"
            "        self.name = name\n        self.value = value\n"
            f"        {registry}.append(self)\n{constructor_suffix}"
        )
    (path / "object_library.py").write_text("\n".join(classes), encoding="utf-8")
    (path / "parameters.py").write_text(
        "from .object_library import Parameter, CTParameter, Coupling\n" + declarations,
        encoding="utf-8",
    )
    (path / "__init__.py").write_text(
        "from . import object_library, parameters\n"
        "from .object_library import all_parameters, all_CTparameters, all_couplings\n",
        encoding="utf-8",
    )
    return path


def test_direct_literals_retain_sixty_digits_and_laurent_coefficients(
    tmp_path: Path,
) -> None:
    from symbolica import E

    digits = "0." + "1234567890" * 6
    source = _write_literal_ufo(
        tmp_path / "literal_ufo",
        f"a = Parameter(name='same', value={digits})\n"
        "b = Parameter(name='same', value=-3.123456789012345678901e-17)\n"
        f"ct = CTParameter(name='ct', value={{-1: {digits}, 0: 'a**2'}})\n"
        f"c = Coupling(name='c', value={digits})\n",
    )
    model = load_raw_loop_model(source)
    assert model.exact_value("parameters", 0)["expression"] == digits
    assert (
        model.exact_value("parameters", 1)["expression"]
        == "-3.123456789012345678901e-17"
    )
    assert model.exact_value("counterterm_parameters", 0)["terms"] == (
        (-1, digits),
        (0, "a**2"),
    )
    assert model.exact_value("couplings", 0)["expression"] == digits
    source_record = model.collections["parameters"][0]["fields"]["value_source"]
    assert source_record == {
        "kind": "source-literal",
        "exact": True,
        "file": "parameters.py",
        "line": 2,
        "argument": digits,
    }
    # Standard exact decimal conversion feeds Symbolica's rational parser. This
    # deliberately avoids the UFO loader's float-based pseudo-float conversion.
    exact = Fraction(Decimal(model.exact_value("parameters", 0)["expression"]))
    expression = E(f"{exact.numerator}/{exact.denominator}")
    assert expression == E(f"{digits[2:]}/10^60")
    rounded = Fraction(float(digits))
    assert expression != E(f"{rounded.numerator}/{rounded.denominator}")
    assert model.capabilities["numeric_literal_source_preserved"] is True
    assert "arbitrary_precision_inputs_validated" not in model.capabilities
    restored = RawLoopModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.exact_value("parameters", 0) == model.exact_value("parameters", 0)


@pytest.mark.parametrize(
    "declarations, reason",
    [
        ("a = Parameter(name='a', value=1.0/3.0)\n", "computed value"),
        (
            "def make():\n    return Parameter(name='a', value=0.1)\na = make()\n",
            "direct module declaration",
        ),
        (
            "a = Parameter(name='a', value=0.123456789012345678901)\n"
            "a.value = 0.123456789012345678902\n",
            "dynamic value writes",
        ),
        (
            "a = Parameter(name='a', value={-1: 0.123456789012345678901})\n"
            "a.value[-1] = 0.123456789012345678902\n",
            "dynamic value writes",
        ),
        (
            "a = Parameter(name='a', value=0.1)\n"
            "b = Parameter(name='b', value=0.2)\na = b\n",
            "no longer binds",
        ),
        (
            "a = Parameter(name='a', value=str(0.123456789012345678901))\n",
            "computed value",
        ),
        (
            "a = Parameter(name='a', value=0.123456789012345678901)\n"
            "class Mutator:\n"
            "    @staticmethod\n"
            "    def change():\n"
            "        a.value = 0.123456789012345678902\n"
            "Mutator.change()\n",
            "dynamic value writes",
        ),
    ],
)
def test_ambiguous_computed_and_mutated_values_are_not_exact(
    tmp_path: Path, declarations: str, reason: str
) -> None:
    source = _write_literal_ufo(tmp_path / "ambiguous_ufo", declarations)
    model = load_raw_loop_model(source)
    assert model.capabilities["numeric_literal_source_preserved"] is False
    with pytest.raises(ValueError, match=reason):
        model.exact_value("parameters", 0)


def test_constructor_cannot_silently_replace_literal_value(tmp_path: Path) -> None:
    source = _write_literal_ufo(
        tmp_path / "changed_constructor",
        "a = Parameter(name='a', value=0.1)\n",
        constructor_suffix="        self.value = self.value + 0.0\n",
    )
    model = load_raw_loop_model(source)
    with pytest.raises(ValueError, match="changed or overridden"):
        model.exact_value("parameters", 0)


def test_composed_counterterm_strings_preserve_source_and_laurent_orders(
    tmp_path: Path,
) -> None:
    digits = "0." + "1234567890" * 6
    source = _write_literal_ufo(
        tmp_path / "composed_ct",
        f"a = CTParameter(name='a', value={{-1: '{digits}*g', 0: 'log(m)'}})\n"
        "b = CTParameter(name='b', value={-1: '(' + a.value[-1] + ')/2', "
        "0: 'x+' + a.value[0]})\n"
        "c = CTParameter(name='c', value={-1: b.value[-1], 0: b.value[0] + '+y'})\n"
        "d = Coupling(name='d', value='g*(' + c.value[-1] + ')')\n",
    )
    model = load_raw_loop_model(source)
    assert model.exact_value("counterterm_parameters", 1)["terms"] == (
        (-1, f"({digits}*g)/2"),
        (0, "x+log(m)"),
    )
    assert model.exact_value("counterterm_parameters", 2)["terms"] == (
        (-1, f"({digits}*g)/2"),
        (0, "x+log(m)+y"),
    )
    assert model.exact_value("couplings", 0)["expression"] == f"g*(({digits}*g)/2)"
    assert model.collections["couplings"][0]["fields"]["value_source"]["kind"] == (
        "source-string-composition"
    )
    assert model.capabilities["numeric_literal_source_preserved"] is True
    assert RawLoopModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model


@pytest.mark.parametrize(
    "declarations, reason",
    [
        (
            "a = CTParameter(name='a', value={-1: str(0.123456789012345678901)})\n"
            "b = Coupling(name='b', value='g*' + a.value[-1])\n",
            "referenced string declaration is not certified",
        ),
        (
            "a = CTParameter(name='a', value={-1: '0.123456789012345678901'})\n"
            "b = Coupling(name='b', value='g*' + a.value[-1])\n"
            "a = CTParameter(name='replacement', value={-1: '2'})\n",
            "referenced declaration is no longer exact",
        ),
        (
            "a = CTParameter(name='a', value={-1: '0.123456789012345678901'})\n"
            "b = Coupling(name='b', value='g*' + a.value[-1])\n"
            "saved = a.value\nsaved[-1] = '2'\n",
            "referenced declaration is no longer exact",
        ),
        (
            "a = CTParameter(name='a', value={1: 'g'})\n"
            "b = Coupling(name='b', value=a.value[True])\n",
            "integer Laurent power",
        ),
        (
            "a = CTParameter(name='a', value={-1: 'g'})\n"
            "index = -1\nb = Coupling(name='b', value=a.value[index])\n",
            "malformed node",
        ),
        (
            "a = CTParameter(name='a', value={-1: 0.123456789012345678901})\n"
            "b = Coupling(name='b', value=a.value[-1])\n",
            "not a source string",
        ),
        (
            "b = Coupling(name='b', value='g*' + str(0.123456789012345678901))\n",
            "not a certified string composition",
        ),
    ],
)
def test_composed_strings_reject_uncertified_or_changed_dependencies(
    tmp_path: Path, declarations: str, reason: str
) -> None:
    source = _write_literal_ufo(tmp_path / "unproven_composition", declarations)
    model = load_raw_loop_model(source)
    assert model.collections["couplings"][0]["fields"]["value_source"]["kind"] == (
        "unsupported"
    )
    with pytest.raises(ValueError, match=reason):
        model.exact_value("couplings", 0)


def test_explicit_exact_overrides_do_not_modify_imported_tree_values(
    tmp_path: Path,
) -> None:
    from pyamplicol.models.loop_import import imported_ufo_module

    digits = "0." + "1234567890" * 6
    source = _write_literal_ufo(
        tmp_path / "override_ufo", "a = Parameter(name='a', value=1.0/3.0)\n"
    )
    with imported_ufo_module(source) as module:
        original = module.all_parameters[0].value
        for override, expected in (
            (digits, digits),
            (Decimal(digits), digits),
            (Fraction(1, 3), "(1/3)"),
        ):
            model = capture_raw_loop_model(
                module, value_overrides={("parameters", 0): override}
            )
            assert model.exact_value("parameters", 0)["expression"] == expected
            assert module.all_parameters[0].value is original
        with pytest.raises(ValueError, match="must use exact"):
            capture_raw_loop_model(
                module, value_overrides={("parameters", 0): float(digits)}
            )


def test_import_observer_restores_existing_profiler_after_failure(
    tmp_path: Path,
) -> None:
    source = _write_literal_ufo(
        tmp_path / "profile_failure", "raise RuntimeError('deliberate failure')\n"
    )
    previous = sys.getprofile()
    events = []

    def profiler(frame, event, _argument):
        if event == "call" and frame.f_code.co_filename == str(
            source / "parameters.py"
        ):
            events.append(frame.f_code.co_name)

    try:
        sys.setprofile(profiler)
        with pytest.raises(RuntimeError, match="deliberate failure"):
            load_raw_loop_model(source)
        assert sys.getprofile() is profiler
        assert events == ["<module>"]
    finally:
        sys.setprofile(previous)


def test_real_tree_loading_retains_rounded_tree_value_and_exact_sidecar(
    tmp_path: Path,
) -> None:
    packaged = (
        Path(pyamplicol.__file__).parent / "assets" / "models" / "ufo" / "scalars"
    )
    source = tmp_path / "tree_with_exact_sidecar"
    shutil.copytree(packaged, source, ignore=shutil.ignore_patterns("__pycache__"))
    parameters = source / "parameters.py"
    digits = "0." + "1234567890" * 6
    parameters.write_text(
        parameters.read_text().replace("value=1.0,", f"value={digits},", 1)
    )
    result = _load_external_model(
        source, options=ModelCompileOptions(restriction="none")
    )
    raw = RawLoopModel.from_dict(result["raw_loop_model"])
    record = raw.record("parameters", "lam")
    assert raw.exact_value("parameters", record["id"])["expression"] == digits
    tree = next(
        parameter
        for parameter in result["model"]["parameters"]
        if parameter["name"] == "lam"
    )
    assert tree["value"] == [float(digits), 0.0]
    assert result["model"]["vertex_rules"]


def test_compiled_model_serialization_retains_raw_sidecar(tmp_path: Path) -> None:
    raw = capture_raw_loop_model(_nlo_module())
    empty_ir = CompiledModelIR(
        name="test",
        orders=(),
        parameters=(),
        particles=(),
        couplings=(),
        propagators=(),
        vertex_terms=(),
        oriented_kernels=(),
        direct_contractions=(),
        closure_contractions=(),
    )
    compiled = CompiledModel(
        source={},
        producer={},
        model={"name": "test"},
        ir=empty_ir,
        parameter_defaults={},
        capabilities={},
        issues=(),
        phase_timings={},
        conversion_seconds=0.0,
        raw_loop_model=raw,
    )
    written = compiled.write(tmp_path / "model")
    restored = CompiledModel.from_dict(
        json.loads(written.read_text()), validate_fingerprint=False
    )
    assert restored.raw_loop_model.to_dict() == raw.to_dict()
    payload = compiled.to_dict()
    # Retained internal development artifacts use the same source records.
    payload["raw_loop_model"]["kind"] = "pyamplicol-raw-loop-model"
    legacy = CompiledModel.from_dict(payload, validate_fingerprint=False)
    assert legacy.raw_loop_model.to_dict() == raw.to_dict()
    del payload["raw_loop_model"]
    assert (
        CompiledModel.from_dict(payload, validate_fingerprint=False).raw_loop_model
        is None
    )


def test_local_loop_sm_regression(loop_sm_ufo) -> None:
    source = str(loop_sm_ufo)
    model = load_raw_loop_model(source)
    assert len(model.collections["counterterm_parameters"]) == 21
    assert len(model.collections["counterterm_vertices"]) == 103
    assert model.record("counterterm_parameters", "G_UVg")["fields"]["value"][
        "terms"
    ] == ((-1, "-((G**2)/(2.0*48.0*cmath.pi**2))*11.0*CA"),)
    assert model.record("couplings", "UV_3Gg")["fields"]["value"] == {
        "kind": "expression",
        "expression": "-G_UVg*G",
    }
    r2 = model.record("counterterm_vertices", "V_R2RGA")["fields"]
    assert "d(-1,1,3)*d(-1,2,4)" in r2["color"]
    assert len(r2["loop_particles"]) == 2
    assert model.capabilities["perturbative_orders"] == ("QCD",)
    assert model.capabilities["ghost_particles"] == ("gh", "gh~")
    assert model.capabilities["goldstone_particles"] == ("G0", "G+", "G-")


def test_local_full_ew_duplicate_vertex_names_are_preserved(
    loop_qcd_qed_sm_ufo,
) -> None:
    source = str(loop_qcd_qed_sm_ufo)
    model = load_raw_loop_model(source)
    vertices = [
        record
        for record in model.collections["counterterm_vertices"]
        if record["name"] == "V_R2AGmGp"
    ]
    # The released UFO gives A G- G+ and Z G- G+ the same original name.
    assert len(vertices) == 2
    assert vertices[0]["id"] != vertices[1]["id"]
    assert [vertex["fields"]["particles"][0]["name"] for vertex in vertices] == [
        "A",
        "Z",
    ]
    coupling_names = [
        vertex["fields"]["couplings"]["entries"][0][1]["name"] for vertex in vertices
    ]
    assert coupling_names == ["R2_AGmGp", "R2_ZGmGp"]
    assert len(model.collections["counterterm_parameters"]) == 105
    assert len(model.collections["counterterm_vertices"]) == 499
    assert model.capabilities["perturbative_orders"] == ("QCD", "QED")
    # Released CT expressions include forty literal-string compositions and
    # previously declared Laurent coefficient reads. No numeric reconstruction
    # from the imported Python value is needed for these declarations.
    for collection in ("parameters", "counterterm_parameters", "couplings"):
        for record in model.collections[collection]:
            model.exact_value(collection, record["id"])
    assert model.capabilities["numeric_literal_source_preserved"] is True
    assert RawLoopModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model
