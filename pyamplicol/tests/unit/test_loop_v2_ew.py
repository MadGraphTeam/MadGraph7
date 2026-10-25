# SPDX-License-Identifier: 0BSD
"""Electroweak counterterms in the tree recursion (WP3.2 EWI-0, EWI-3).

The G_mu UFO (``loop_qcd_qed_sm_Gmu_py3``) writes its UV counterterm
parameters with the electroweak helpers ``re``, ``recms``, ``regsqrt``,
``grreglog``, ``reglogp``/``reglogm``, ``arg`` and a rebound (CMS)
``reglog``. ``ct_recursion.CoefficientLowering`` keeps every such CT scalar a
parameter-program reference: the couplings are linear in them, and the
counterterm program binds them at the card from the scalars' exact bodies,
every helper a function of the model's scalar library
(``ct_recursion.coefficient_plan``). Oracles, model-level (no process):

* the UFO's own Python evaluation of ``parameters.py`` and
  ``CT_parameters.py`` (imported under a private name, ``cmath``, the
  ``cond`` branch not taken never evaluated) at the card of the
  ``gg_ttx_ew`` fixture: every Laurent coefficient of every CT parameter in
  binary64; every coefficient the plan proves zero is zero there;
* an mpmath transliteration of the same source (every number literal exact)
  for the plan at double-double and arbitrary precision (nightly);
* the branch side: the CT values with every helper argument on or within
  1e-12 relative of a branch cut moved to the other side equal the
  unflipped ones (real masses take only real parts);
* the referenced couplings: each CT coupling's Laurent coefficients, formed
  on the reference parameters and bound with the plan, equal the UFO's
  coupling expression at the oracle's CT values;
* the digest rule: loop_sm's ``cond``, ``reglog`` and ``complexconjugate``
  are UFO-standard (resolved by name, so loop_sm plans are unchanged), the
  G_mu ``reglog`` is not;
* hygiene (M2): the lowering forms no infinity in a branch that is not
  taken, refuses a coefficient that is not finite, and decides the scalar
  library's predicates of exact numbers (the CMS switch ``CMSParam``).

The ladder (nightly): one strict xfail per electroweak request, each naming
the blocker its generation reaches (PLAN wp32 §2); an increment that lifts
a blocker flips its rungs.
"""

from __future__ import annotations

import ast
import cmath
import importlib
import importlib.util
import math
import sys
import warnings

import pytest

pytestmark = pytest.mark.loop_physics

# The fixture whose card the CT values are compared at (point 0's scale).
CARD_FIXTURE = "required/gg_ttx_ew"
# Relative agreement with the UFO's binary64 evaluation (both round in
# binary64: measured 3.6e-15 against 60-digit mpmath for the plan, 3.7e-15
# for the UFO's own evaluation).
DOUBLE_TOLERANCE = 1e-13


def _gmu_card(loop_reference) -> tuple[dict[str, str], dict[str, str]]:
    """(the fixture's input parameters with MU_R, its fixed parameters)."""

    fixture = loop_reference(CARD_FIXTURE)
    card = {**fixture["parameters"], "MU_R": fixture["mu_r"][0]}
    return card, dict(fixture["fixed_parameters"])


@pytest.fixture(scope="module")
def gmu(loop_gmu_ufo, loop_reference):
    """The G_mu model's restricted source scalar program, its fixed
    parameters (the default restriction's and the fixture's, without the
    fixture's WTau, which the UFO lacks) and the fixture card."""

    from types import SimpleNamespace

    from pyamplicol.loops.model_restrictions import (
        prepare_loop_restrictions,
        restriction_fixed_parameters,
    )
    from pyamplicol.loops.scalar_frontend import lower_scalar_program
    from pyamplicol.models.loading import compile_model_source

    compiled = compile_model_source(loop_gmu_ufo, require_supported=False)
    derived = prepare_loop_restrictions(compiled.raw_loop_model).derived
    card, fixed = _gmu_card(loop_reference)
    externals = {
        record["name"]
        for record in derived.collections["parameters"]
        if record["fields"].get("nature") == "external"
    }
    fixed = restriction_fixed_parameters(
        derived, {name: value for name, value in fixed.items() if name in externals}
    )
    return SimpleNamespace(
        compiled=compiled,
        raw=derived,
        program=lower_scalar_program(derived),
        fixed=fixed,
        card=card,
    )


@pytest.fixture(scope="module")
def gmu_ufo(loop_gmu_ufo):
    """The UFO's ``parameters``, ``CT_parameters`` and ``function_library``
    modules, imported as a private package (no bytecode written)."""

    name = "_pyamplicol_test_ufo_gmu"
    spec = importlib.util.spec_from_file_location(
        name,
        loop_gmu_ufo / "__init__.py",
        submodule_search_locations=[str(loop_gmu_ufo)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with warnings.catch_warnings():
            # The UFO's texnames hold invalid escape sequences.
            warnings.simplefilter("ignore", DeprecationWarning)
            spec.loader.exec_module(package)
            modules = tuple(
                importlib.import_module(f"{name}.{part}")
                for part in ("parameters", "CT_parameters", "function_library")
            )
            importlib.import_module(f"{name}.CT_couplings")
        yield modules
    finally:
        sys.dont_write_bytecode = previous
        for module in [m for m in sys.modules if m == name or m.startswith(f"{name}.")]:
            del sys.modules[module]


@pytest.fixture(scope="module")
def gmu_plan(gmu):
    """The coefficient plan of every CT parameter's Laurent coefficients."""

    from pyamplicol.loops.v2.ct_recursion import coefficient_plan

    outputs = []
    for record in gmu.raw.collections["counterterm_parameters"]:
        source = gmu.raw.exact_value("counterterm_parameters", record["id"])
        terms = source["terms"] if source["kind"] == "laurent" else ((0, "0"),)
        outputs += [(record["name"], int(power)) for power, _text in terms]
    return coefficient_plan(gmu.program, outputs, gmu.fixed)


# ---------------------------------------------------------------- UFO oracle


class _Source(ast.NodeTransformer):
    """A UFO expression for the oracle: ``cond(c, a, b)`` becomes the lazy
    ``a if c == 0 else b`` (the branch not taken is never evaluated, as in
    the lowering), and with ``exact`` every number literal is read exactly
    (``NUMBER('<literal>')``, ``IMAGINARY('<literal>')``)."""

    def __init__(self, source: str, exact: bool):
        self.source, self.exact = source, exact

    def _number(self, text: str) -> ast.expr:
        return ast.Call(ast.Name("NUMBER", ast.Load()), [ast.Constant(text)], [])

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "cond":
            condition, if_zero, otherwise = node.args
            zero = self._number("0") if self.exact else ast.Constant(0.0)
            return ast.IfExp(
                ast.Compare(condition, [ast.Eq()], [zero]), if_zero, otherwise
            )
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if not self.exact or type(node.value) not in (int, float, complex):
            return node
        text = ast.get_source_segment(self.source, node) or repr(node.value)
        if type(node.value) is complex:
            return ast.Call(
                ast.Name("IMAGINARY", ast.Load()), [ast.Constant(text[:-1])], []
            )
        return self._number(text)


def _compiled(source: str, exact: bool):
    tree = _Source(source, exact).visit(ast.parse(source, mode="eval"))
    return compile(ast.fix_missing_locations(tree), "<ufo>", "eval")


class _Library:
    """``cmath`` and ``math`` of the mpmath transliteration."""

    def __init__(self, mp):
        self._mp, self.pi = mp, mp.pi

    def __getattr__(self, name):
        return getattr(self._mp, name)


def _ufo_values(modules, card, *, digits=None, wrap=None) -> dict:
    """{(CT parameter, epsilon power): value} and the parameters' values of
    the UFO's own source at ``card`` (decimal strings; other external
    parameters at the UFO's values): Python's ``cmath`` in binary64, or
    mpmath at ``digits``. ``wrap(name, function)`` may replace a helper."""

    if digits is not None:
        import mpmath

        with mpmath.workdps(digits):
            return _ufo_source_values(modules, card, exact=True, wrap=wrap)
    return _ufo_source_values(modules, card, exact=False, wrap=wrap)


def _ufo_source_values(modules, card, *, exact, wrap) -> dict:
    parameters, counterterms, functions = modules
    if exact:
        import mpmath

        mp = mpmath.mp
        environment = {
            "cmath": _Library(mp),
            "math": _Library(mp),
            "NUMBER": mp.mpf,
            "IMAGINARY": lambda text: mp.mpc(0, mp.mpf(text)),
            "complex": lambda real, imag=0: mp.mpc(real, imag),
        }
        number = mp.mpf
    else:
        environment = {"cmath": cmath, "math": math, "complex": complex}
        number = float
    # Later definitions rebind a name (the G_mu UFO's CMS reglog), as the
    # module's own attributes do.
    for function in functions.all_functions:
        arguments = function.arguments
        arguments = arguments if isinstance(arguments, tuple) else (arguments,)
        body = eval(
            _compiled(f"lambda {','.join(arguments)}: {function.expr}", exact),
            environment,
        )
        environment[function.name] = (
            wrap(function.name, body) if wrap is not None else body
        )
    for parameter in parameters.all_parameters:
        if parameter.nature == "external":
            value = number(card.get(parameter.name, repr(parameter.value)))
            complex_type = parameter.type == "complex"
            environment[parameter.name] = (
                environment["complex"](value) if complex_type else value
            )
    for parameter in parameters.all_parameters:
        if parameter.nature == "internal":
            environment[parameter.name] = eval(
                _compiled(parameter.value, exact), environment
            )
    values = {}
    for parameter in counterterms.all_CTparameters:
        terms = (
            parameter.value
            if isinstance(parameter.value, dict)
            else {0: parameter.value}
        )
        for power, source in terms.items():
            values[parameter.name, int(power)] = eval(
                _compiled(source, exact), environment
            )
    return values, environment


def _oracle_card(gmu) -> dict[str, str]:
    """The fixture card with every fixed parameter, as decimal strings."""

    return {**gmu.card, **{name: real for name, (real, _imag) in gmu.fixed.items()}}


def _plan_values(plan, card, **options) -> dict:
    from pyamplicol.loops.scalar_bindings import NativeScalarBindings

    state = NativeScalarBindings(
        plan,
        parameters={k: v for k, v in card.items() if k in plan.external_names},
        **options,
    )
    return dict(zip(plan.outputs, state.values.to_decimal(), strict=True))


def _relative(value, reference) -> float:
    """|value - reference| / |reference|, or |value| for a zero reference."""

    scale = abs(reference)
    return float(abs(value - reference) / scale) if scale else float(abs(value))


# --------------------------------------------------------------------- tests


def test_counterterm_scalars_equal_the_ufo_python_evaluation(gmu, gmu_ufo, gmu_plan):
    """Every Laurent coefficient of the G_mu CT parameters from the plan
    (binary64, native scalar bindings) equals the UFO's own Python
    evaluation at the fixture card; the coefficients the plan proves zero
    at the fixed parameters are zero there, and none is singular."""

    found = gmu_plan
    assert not found.nonfinite
    reference, _environment = _ufo_values(gmu_ufo, _oracle_card(gmu))
    assert set(reference) == set(found.plan.outputs) | found.zero
    assert found.zero and all(reference[key] == 0 for key in found.zero)
    values = _plan_values(found.plan, gmu.card, arithmetic="double")
    worst = max(
        (_relative(complex(float(re), float(im)), reference[key]), key)
        for key, (re, im) in values.items()
    )
    assert worst[0] <= DOUBLE_TOLERANCE, worst
    # Discriminating: the values span the poles and finite parts of the
    # electroweak UV counterterms, not only zeros.
    assert sum(1 for value in reference.values() if abs(value) > 1e-3) > 50


def test_counterterm_branch_sides_do_not_matter_at_real_masses(gmu, gmu_ufo):
    """Every argument of a branch-cut helper (``regsqrt``, the logarithms,
    ``grreglog``, ``arg``) on the negative real axis or within 1e-12 of it
    (relative) is moved to the other side of the cut: the CT values are
    unchanged, since real masses keep only real parts. Many arguments are
    on or near the cut (discriminating)."""

    flipped = []

    def wrap(name, function):
        if name not in {"regsqrt", "reglog", "reglogp", "reglogm", "grreglog", "arg"}:
            return function

        def side(z):
            z = complex(z)
            if z.real < 0 and abs(z.imag) <= 1e-12 * abs(z.real):
                flipped.append(z)
                return complex(z.real, -z.imag if z.imag else -0.0)
            return z

        if name == "grreglog":
            return lambda switch, z1, z2: function(switch, side(z1), side(z2))
        return lambda z: function(side(z))

    card = _oracle_card(gmu)
    reference, _environment = _ufo_values(gmu_ufo, card)
    moved, _environment = _ufo_values(gmu_ufo, card, wrap=wrap)
    assert len(flipped) > 100
    worst = max((_relative(moved[key], reference[key]), key) for key in reference)
    assert worst[0] <= DOUBLE_TOLERANCE, worst


def test_referenced_couplings_equal_the_ufo_couplings(gmu, gmu_ufo, gmu_plan):
    """Each G_mu CT coupling that reads a CT parameter needing the scalar
    library is lowered on the reference parameters (``CoefficientLowering``),
    linearly; bound with the plan and the UFO's parameter values, every
    Laurent coefficient equals the UFO's coupling expression at the
    oracle's CT coefficients (sampled every 7th coupling; all couplings
    nightly), and every reference it reads is a plan output or a zero."""

    _check_referenced_couplings(gmu, gmu_ufo, gmu_plan, every=7)


def _check_referenced_couplings(gmu, gmu_ufo, gmu_plan, *, every: int) -> None:
    from types import SimpleNamespace

    from symbolica import Expression

    from pyamplicol.loops.v2.ct_recursion import CoefficientLowering

    builtins = (Expression.PI, Expression.E, Expression.I)
    card = _oracle_card(gmu)
    reference, environment = _ufo_values(gmu_ufo, card)
    lowering = CoefficientLowering(gmu.compiled, gmu.fixed)
    bound = {
        key: complex(float(re), float(im))
        for key, (re, im) in _plan_values(
            gmu_plan.plan, gmu.card, arithmetic="double"
        ).items()
    }
    bound.update(dict.fromkeys(gmu_plan.zero, 0j))
    ct_names = {name for name, _power in reference}
    couplings = gmu_ufo[0].__name__.rsplit(".", 1)[0] + ".CT_couplings"
    sources = {
        c.name: c.value for c in importlib.import_module(couplings).all_couplings
    }
    checked = 0
    for record in gmu.raw.collections["couplings"][::every]:
        entry = SimpleNamespace(
            coupling=gmu.program.definitions["couplings", record["id"]],
            scalar_program=gmu.program,
        )
        referenced = lowering._referenced(entry)
        if referenced is None:
            continue
        source = sources[record["name"]]
        for power in (0, -1, -2):
            value = lowering(entry, power)
            constants = {}
            for symbol in value.get_all_symbols(False):
                if symbol in builtins:
                    continue
                name = symbol.get_name().rsplit("::", 1)[-1]
                if name in lowering.references:
                    scalar, shift, part = lowering.references[name]
                    number = bound[scalar, shift]
                    constants[symbol] = number.real if part == "re" else number.imag
                else:
                    constants[symbol] = complex(environment[name])
            mine = complex(value.evaluate(constants))
            # The UFO coupling at the oracle's eps^power CT coefficients
            # (it is linear in them), and the binary64 rounding scale: the
            # sum of the magnitudes of its terms, one CT parameter at a time.
            program = _compiled(source, False)
            read = [name for name in ct_names if name in source]
            local = {**environment, **dict.fromkeys(ct_names, 0)}
            free = eval(program, local)
            expected, scale = (0 if power else free), abs(free)
            for name in read:
                local[name] = reference.get((name, power), 0)
                term = eval(program, local) - free
                local[name] = 0
                expected += term
                scale += abs(term)
            assert abs(mine - expected) <= DOUBLE_TOLERANCE * scale, (
                record["name"],
                power,
                mine,
                expected,
            )
        checked += 1
    assert checked > 20, checked
    assert lowering.references and all(
        lowering.references[name][:2] in set(gmu_plan.plan.outputs) | gmu_plan.zero
        for name in lowering.references
        if lowering.references[name][0] in ct_names
    )


# Arguments of the helper tests: every quadrant, the negative real axis
# (above, and 1e-30 off either side of the cut), the positive one and zero.
_HELPER_POINTS = (
    ("2", "3"),
    ("-2", "3"),
    ("-2", "-3"),
    ("2", "-3"),
    ("-2", "0"),
    ("-2", "1e-30"),
    ("-2", "-1e-30"),
    ("3", "0"),
    ("0", "0"),
)


def _helper_cases():
    """(helper, arguments) of every G_mu function-library helper: unary ones
    at every point, the switch-taking ones (``cond``, ``recms``, ``crecms``,
    ``grreglog``) with their switch 0 and 1 and the second argument at every
    point paired with a fixed first one."""

    unary = ("re", "im", "complexconjugate", "reglog", "reglogp", "reglogm")
    unary += ("regsqrt", "arg")
    cases = [(name, (z,)) for name in unary for z in _HELPER_POINTS]
    for switch in (("0", "0"), ("1", "0")):
        cases += [("cond", (switch, ("5", "1"), z)) for z in _HELPER_POINTS]
        cases += [
            (name, (switch, z)) for name in ("recms", "crecms") for z in _HELPER_POINTS
        ]
        cases += [
            ("grreglog", (switch, z1, z2))
            for z1 in _HELPER_POINTS[:7]
            for z2 in (("-1", "2"), ("-1", "-2"), ("1", "1"))
        ]
    return cases


def test_scalar_library_helpers_equal_the_ufo_function_library(gmu, gmu_ufo):
    """Every G_mu helper body as the scalar library lowers it (the bound
    body, a function of the native parameter program) evaluated natively
    equals the UFO's own ``function_library`` (Python ``cmath``) in binary64
    and an mpmath transliteration of it in double-double and at 40 digits,
    in every quadrant and on both sides of the cut (the CMS reglog's
    -2 pi i above the negative axis, grreglog's sheet switches)."""

    import mpmath

    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.loops.native import load_loop_native

    library = gmu.program.library
    backend = load_loop_native()
    functions = gmu_ufo[2]
    cases = _helper_cases()
    width = max(len(arguments) for _name, arguments in cases)
    inputs = tuple(symbols.qualified(f"pyamplicol_test_ew::x{i}") for i in range(width))
    names = [f"x{i}" for i in range(width)]
    expressions = tuple(
        library.lower(
            f"{name}({','.join(names[: len(arguments)])})",
            locals=dict(zip(names, inputs, strict=False)),
        )
        for name, arguments in cases
    )
    state = library.evaluator(expressions, inputs).save()
    with mpmath.workdps(60):
        exact = {}
        for function in functions.all_functions:
            arguments = function.arguments
            arguments = arguments if isinstance(arguments, tuple) else (arguments,)
            exact[function.name] = eval(
                _compiled(f"lambda {','.join(arguments)}: {function.expr}", True),
                {
                    "cmath": _Library(mpmath.mp),
                    "math": _Library(mpmath.mp),
                    "NUMBER": mpmath.mpf,
                    "IMAGINARY": lambda text: mpmath.mpc(0, mpmath.mpf(text)),
                },
            )
        references = [
            exact[name](*(mpmath.mpc(mpmath.mpf(re), mpmath.mpf(im)) for re, im in a))
            for name, a in cases
        ]
    python = {f.name: f for f in functions.all_functions}  # the last binding wins
    for arithmetic, precision, tolerance in (
        ("double", None, 4e-16),
        ("double-double", None, 1e-30),
        ("arbitrary", 40, 1e-38),
    ):
        evaluator = backend.CoefficientEvaluator(
            state, arithmetic=arithmetic, precision=precision
        )
        for row, (name, arguments) in enumerate(cases):
            padded = [*arguments, *(("0", "0"),) * (width - len(arguments))]
            batch = backend.NativeBatch.from_decimal(
                padded, 1, width, arithmetic=arithmetic, precision=precision
            )
            re, im = evaluator.evaluate_batch(batch).to_decimal()[2 * row]
            if arithmetic == "double":
                function = python[name]
                args = function.arguments
                args = args if isinstance(args, tuple) else (args,)
                body = eval(
                    _compiled(f"lambda {','.join(args)}: {function.expr}", False),
                    {"cmath": cmath, "math": math},
                )
                reference = complex(
                    body(*(complex(float(x), float(y)) for x, y in arguments))
                )
                error = abs(complex(float(re), float(im)) - reference)
                scale = max(abs(reference), 1.0)
            else:
                with mpmath.workdps(60):
                    reference = references[row]
                    value = mpmath.mpc(mpmath.mpf(re), mpmath.mpf(im))
                    error = float(abs(value - reference))
                    scale = float(max(abs(reference), 1))
            assert error <= tolerance * scale, (arithmetic, name, arguments, re, im)


def test_standard_helpers_keep_loop_sm_by_name(loop_sm_compiled_model, gmu):
    """The digest rule: loop_sm's cond, reglog and complexconjugate are the
    UFO-standard bodies (resolved by name, so its plans are unchanged); the
    G_mu UFO rebinds reglog (CMS sheets), which stays a library function, and
    its electroweak helpers are never resolved by name."""

    from pyamplicol.loops.scalar_frontend import ScalarFunctionLibrary
    from pyamplicol.loops.v2.ct_recursion import standard_helpers
    from pyamplicol.models.loading import load_compiled_model

    loop_sm = load_compiled_model(loop_sm_compiled_model).raw_loop_model
    assert sorted(standard_helpers(ScalarFunctionLibrary(loop_sm)).values()) == [
        "complexconjugate",
        "cond",
        "reglog",
    ]
    library = gmu.program.library
    assert sorted(standard_helpers(library).values()) == ["complexconjugate", "cond"]
    assert {"reglog", "regsqrt", "recms", "re", "grreglog"} <= set(library.helpers)


def test_lowering_never_forms_an_untaken_branch(loop_sm_compiled_model, capfd):
    """M2 hygiene: with MB fixed to 0, cond(MB, 1, 1/MB^2) resolves to 1
    without forming the infinity of the branch not taken (no Symbolica
    warning), reglog(MB^2/MU_R^2) is 0, and a coefficient that is not finite
    in a taken branch (1/MB) is refused."""

    from symbolica import E

    from pyamplicol.loops.scalar_frontend import ScalarFunctionLibrary
    from pyamplicol.loops.v2.ct_recursion import (
        CoefficientLowering,
        CountertermCouplingError,
    )
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    library = ScalarFunctionLibrary(compiled.raw_loop_model)
    lowering = CoefficientLowering(compiled, {"MB": "0"})
    cond, reglog = library.helpers["cond"][0], library.helpers["reglog"][0]
    mb, mu = E("UFO::{}::MB"), E("UFO::{}::MU_R")
    capfd.readouterr()
    assert lowering._finite(lowering._resolve(cond(mb, E("1"), mb**-2))) == E("1")
    assert lowering._finite(lowering._resolve(reglog(mb**2 / mu**2))) == E("0")
    assert "Created" not in capfd.readouterr().err
    with pytest.raises(CountertermCouplingError, match="not finite"):
        lowering._finite(lowering._resolve(cond(mu, E("1"), mb**-1)))


def test_scalar_predicates_of_numbers_are_decided():
    """The scalar library's predicates and conditionals of exact numbers are
    decided: the G_mu UFO's CMS switch at its real-mass value,
    if(equal(CMSParam = 0, 1), 1, 0), is zero, so a cond on it takes its
    branch without an assumption; an inexact one stays undecided."""

    from symbolica import E

    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.loops.v2.ct_recursion import CoefficientLowering, decide_constant

    choose = symbols.qualified("symbolica::if")
    equal, less = (
        symbols.qualified(f"pyamplicol_scalar_v1::{name}") for name in ("equal", "less")
    )
    off = choose(equal(E("0"), E("1")), E("1"), E("0"))
    on = choose(equal(E("1"), E("1")), E("1"), E("0"))
    assert [decide_constant(value) for value in (off, on, off * E("2*𝜋"))] == [
        "zero",
        "non-zero",
        "zero",
    ]
    assert decide_constant(choose(less(E("𝜋"), E("4")), E("1"), E("0"))) == "undecided"
    lowering = CoefficientLowering.__new__(CoefficientLowering)
    lowering._helpers = {0: "cond"}
    lowering._zero = E("0")
    lowering.assumptions = set()
    helper = "pyamplicol_scalar_model_test::value_0"
    taken = lowering._resolve(E(f"{helper}(x, 1, 2)").replace(E("x"), off))
    assert taken == E("1") and not lowering.assumptions


def test_validate_builds_the_electroweak_fixture_request(gmu, loop_reference):
    """``tools/loop_v2_generate.py validate`` on an EW fixture: QCD and QED
    corrections, the G_mu compatibility profile and scale parameter, the
    fixture's fixed parameters without the providers' WTau (recorded)."""

    import argparse

    from tools import loop_v2_generate

    reference = loop_reference(CARD_FIXTURE)
    request, ignored = loop_v2_generate._fixture_request(
        reference, gmu.compiled, argparse.Namespace()
    )
    assert request.corrections == ("QCD", "QED")
    assert (request.compatibility, request.scale_parameter) == (
        "gmu_particle_anti",
        "MU_R",
    )
    assert ignored == ["WTau"]
    assert {k: request.fixed_parameters[k] for k in ("WT", "WW", "WZ", "WH")} == {
        k: "0" for k in ("WT", "WW", "WZ", "WH")
    }
    assert request.process == "g g > t t~"


# ------------------------------------------------------------------ nightly


@pytest.mark.loop_nightly
@pytest.mark.parametrize("carrier", ["double-double", "arbitrary"])
def test_counterterm_scalars_equal_mpmath(gmu, gmu_ufo, gmu_plan, carrier):
    """The plan's CT values in the native double-double carrier and at the
    working digits of a 60-digit evaluation equal an mpmath transliteration
    of the UFO's source (every literal exact) at 64 and 140 digits: to 1e-28
    and 1e-55 relative; the plan's zeros are zero there."""

    import mpmath

    from pyamplicol.loops.v2.assemble import Precision

    if carrier == "double-double":
        options, digits, tolerance = {"arithmetic": "double-double"}, 64, "1e-28"
    else:
        working = Precision.arbitrary(60).working_digits
        options = {"arithmetic": "arbitrary", "precision": working}
        digits, tolerance = 2 * working, "1e-55"
    reference, _environment = _ufo_values(gmu_ufo, _oracle_card(gmu), digits=digits)
    values = _plan_values(gmu_plan.plan, gmu.card, **options)
    with mpmath.workdps(digits):
        # Relative error; absolute where the source value is exactly zero at
        # the card (a coefficient the plan cannot prove zero symbolically).
        worst = max(
            (
                _relative(mpmath.mpc(mpmath.mpf(re), mpmath.mpf(im)), reference[key]),
                key,
            )
            for key, (re, im) in values.items()
        )
        assert worst[0] <= float(tolerance), worst
        assert all(reference[key] == 0 for key in gmu_plan.zero)


@pytest.mark.loop_nightly
def test_every_referenced_coupling_equals_the_ufo_coupling(gmu, gmu_ufo, gmu_plan):
    """``test_referenced_couplings_equal_the_ufo_couplings`` on every
    coupling of the model."""

    _check_referenced_couplings(gmu, gmu_ufo, gmu_plan, every=1)


# ------------------------------------------------------------------- ladder

# Blockers (PLAN wp32 §2): the exception and message each request stops at.
B2 = ("LoopKernelCatalogueError", r"declares no loop species")
B2_RUNTIME = ("ValueError", r"lowers to no kernel")
B3 = ("LoopKernelCatalogueError", r"loop line -?\d+ has a unitary-gauge propagator")
B5 = ("FermionSignError", r"has more than one fermion line")
B6 = ("NotImplementedError", r"is not closed under conjugation")
_PHOTON_ONLY = (23, 24, 25, 250, 251)  # Z, W, H and the Goldstones vetoed
# Electroweak requests need every propagating width fixed to zero outside
# the complex-mass scheme (``ew_policy``, the user's decision [U8]).
_ZERO_WIDTHS = {"WT": "0", "WW": "0", "WZ": "0", "WH": "0"}
_G_MU_FIXED = {"MB": "0", "ymb": "0", "MTA": "0", "ymtau": "0", **_ZERO_WIDTHS}
_MASSLESS_POINT = [
    ["500", "0", "0", "500"],
    ["500", "0", "0", "-500"],
    ["500", "240", "320", "300"],
    ["500", "-240", "-320", "-300"],
]
_GB_TW_POINT = [
    ["500", "0", "0", "500"],
    ["500", "0", "0", "-500"],
    [
        "511.73089221950000000",
        "148.1849175986458016361389900087303105804",
        "355.6438022367499239267335760209527453930",
        "288.9605893173593131904710305170241056318",
    ],
    [
        "488.26910778050000000",
        "-148.1849175986458016361389900087303105804",
        "-355.6438022367499239267335760209527453930",
        "-288.9605893173593131904710305170241056318",
    ],
]


def _rung(name, model, process, born, loop, blocker, **options):
    """One ladder rung: a strict xfail raising its blocker's exception."""

    import pyamplicol.loops.v2.catalogue as catalogue
    import pyamplicol.loops.v2.conventions as conventions

    errors = {
        "LoopKernelCatalogueError": catalogue.LoopKernelCatalogueError,
        "FermionSignError": conventions.FermionSignError,
        "ValueError": ValueError,
        "NotImplementedError": NotImplementedError,
    }
    return pytest.param(
        model,
        process,
        born,
        loop,
        blocker,
        options,
        id=name,
        marks=pytest.mark.xfail(
            strict=True, raises=errors[blocker[0]], reason=f"blocker: {blocker[1]}"
        ),
    )


LADDER = [
    # Photon-only QED corrections of a lepton process: the families' photon
    # and lepton species are not in the kernel catalogue (B2, WP3.2 EWI-1).
    _rung(
        "qq_eemm_photon",
        "loop_qcd_qed_sm",
        "e- e+ > m- m+",
        {"QED": 2, "QCD": 0},
        {"QED": 4, "QCD": 0},
        B2,
        corrections=("QED",),
        particle_veto=_PHOTON_ONLY,
        fixed_parameters=_ZERO_WIDTHS,
    ),
    _rung(
        "sm_epem_uux",
        "loop_sm",
        "e+ e- > u u~",
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
        B2,
    ),
    # Q3: photon-only QED corrections of u u~ > d d~ generate (EWI-3); the
    # photon loop line lowers to no kernel (B2 at runtime). Once it does,
    # the bare and counterterm UV poles must cancel.
    _rung(
        "uudd_qed_photon_uv",
        "loop_qcd_qed_sm",
        "u u~ > d d~",
        {"QCD": 2, "QED": 0},
        {"QCD": 2, "QED": 2},
        B2_RUNTIME,
        corrections=("QED",),
        particle_veto=_PHOTON_ONLY,
        point=_MASSLESS_POINT,
        poles="uv",
        fixed_parameters=_ZERO_WIDTHS,
    ),
    _rung(
        "uudd_qed",
        "loop_qcd_qed_sm",
        "u u~ > d d~",
        {"QCD": 2, "QED": 0},
        {"QCD": 2, "QED": 2},
        B3,
        corrections=("QED",),
        fixed_parameters=_ZERO_WIDTHS,
    ),
    _rung(
        "uudd_qcd_qed",
        "loop_qcd_qed_sm",
        "u u~ > d d~",
        {"QCD": 2, "QED": 0},
        ({"QCD": 4, "QED": 0}, {"QCD": 2, "QED": 2}),
        B3,
        corrections=("QCD", "QED"),
        fixed_parameters=_ZERO_WIDTHS,
    ),
    _rung(
        "gb_twm",
        "loop_sm",
        "g b > t w-",
        {"QCD": 1, "QED": 1},
        {"QCD": 3, "QED": 1},
        B5,
        fixed_parameters={"MB": "0", "WT": "0"},
        point=_GB_TW_POINT,
    ),
    _rung(
        "udx_epve",
        "loop_sm",
        "u d~ > e+ ve",
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
        B6,
    ),
    # The G_mu ladder of PLAN wp32 §1 (real masses, widths zero).
    _rung(
        "gmu_eemm",
        "loop_gmu",
        "e- e+ > m- m+",
        {"QED": 2, "QCD": 0},
        {"QED": 4, "QCD": 0},
        B3,
        corrections=("QED",),
        fixed_parameters=_G_MU_FIXED,
    ),
    _rung(
        "gmu_gg_ttx",
        "loop_gmu",
        "g g > t t~",
        {"QCD": 2, "QED": 0},
        {"QCD": 2, "QED": 2},
        B3,
        corrections=("QCD", "QED"),
        fixed_parameters=_G_MU_FIXED,
    ),
    _rung(
        "gmu_uux_emepg",
        "loop_gmu",
        "u u~ > e- e+ g",
        {"QCD": 1, "QED": 2},
        {"QCD": 1, "QED": 4},
        B3,
        corrections=("QCD", "QED"),
        fixed_parameters=_G_MU_FIXED,
    ),
    _rung(
        "gmu_gg_ttxemep",
        "loop_gmu",
        "g g > t t~ e- e+",
        {"QCD": 2, "QED": 2},
        {"QCD": 2, "QED": 4},
        B3,
        corrections=("QCD", "QED"),
        fixed_parameters=_G_MU_FIXED,
    ),
    _rung(
        "gmu_gg_ttxemepg",
        "loop_gmu",
        "g g > t t~ e- e+ g",
        {"QCD": 3, "QED": 2},
        {"QCD": 3, "QED": 4},
        B3,
        corrections=("QCD", "QED"),
        fixed_parameters=_G_MU_FIXED,
    ),
]


@pytest.fixture(scope="module")
def ladder_models(request):
    """Compiled models of the ladder by name, compiled on first use."""

    from pyamplicol.models.loading import compile_model_source, load_compiled_model

    models = {}

    def get(name):
        if name not in models:
            if name == "loop_sm":
                path = request.getfixturevalue("loop_sm_compiled_model")
                models[name] = load_compiled_model(path)
            else:
                ufo = request.getfixturevalue(f"{name}_ufo")
                models[name] = compile_model_source(ufo, require_supported=False)
        return models[name]

    return get


@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    ("model", "process", "born", "loop", "blocker", "options"), LADDER
)
def test_electroweak_ladder_reaches_its_blocker(
    ladder_models, loop_native, model, process, born, loop, blocker, options
):
    """Each rung generates past the CT lowering (WP3.2 EWI-3) and stops at
    its named blocker; a rung whose blocker is lifted runs its check (the UV
    poles cancel) and flips its strict xfail."""

    import re

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest

    options = dict(options)
    point, poles = options.pop("point", None), options.pop("poles", "all")
    if model != "loop_sm":
        options.setdefault("compatibility", "gmu_particle_anti")
        options.setdefault("scale_parameter", "MU_R")
    request = SliceRequest(process, born, loop, **options)
    try:
        slice_ = LoopSlice.generate(ladder_models(model), request, loop_native)
        if point is not None:
            result = slice_.evaluate(
                point, "1000000", parameters={"aS": "0.118", "MT": "173"}, poles=poles
            )
    except Exception as error:
        if type(error).__name__ == blocker[0] and re.search(blocker[1], str(error)):
            raise
        raise AssertionError(f"stopped before its blocker: {error!r}") from error
    if poles == "uv":
        # Bare UV poles + counterterm UV poles = 0 per sector and coefficient
        # (REQUIREMENTS_ADDENDUM I): renormalisation is complete.
        for sector, bare in result.components["bare"].items():
            counterterms = result.components["uv"][sector]
            for index in (1, 2):
                total = float(bare.coefficients[index]) + float(
                    counterterms.coefficients[index]
                )
                assert abs(total) <= 1e-10 * max(
                    abs(float(bare.coefficients[index])), 1e-300
                ), (sector, index, bare, counterterms)
