# SPDX-License-Identifier: 0BSD
"""Electroweak counterterms in the tree recursion (WP3.2 EWI-0, EWI-3).

The electroweak UFOs (G_mu, ``loop_qcd_qed_sm_Gmu_py3``, and MG5_aMC's
``loop_qcd_qed_sm``) write their UV counterterm parameters with the
electroweak helpers ``re``, ``recms``, ``regsqrt``, ``grreglog``,
``reglogp``/``reglogm``, ``arg`` and a rebound (CMS) ``reglog``.
``ct_recursion.CoefficientLowering`` keeps every such CT scalar a
parameter-program reference: the couplings are linear in them, and the
counterterm program binds them at the card from the scalars' exact bodies,
every helper a function of the model's scalar library
(``ct_recursion.coefficient_plan``). Oracles, model-level (no process), on
both UFOs:

* the UFO's own Python evaluation of ``parameters.py`` and
  ``CT_parameters.py`` (imported under a private name, ``cmath``, the
  ``cond`` branch not taken never evaluated) at the card of the
  ``gg_ttx_ew`` fixture: every Laurent coefficient of every CT parameter in
  binary64; every coefficient the plan proves zero is zero there;
* an mpmath transliteration of the same source (every number literal exact)
  for the plan at double-double and arbitrary precision (nightly);
* MadLoop's own values (the binding convention): every Laurent piece of
  every CT parameter MG5_aMC's Fortran model library writes, in double and
  quadruple precision (``madloop-ct-parameters.json``,
  ``tools/loop_reference/madloop_ct_parameters.py``), for the plan in
  binary64, double-double and arbitrary precision. This pins the branch
  side: at real masses some helper arguments lie exactly on a branch cut,
  their side changes the Higgs two-point CTs by O(1), and MadLoop's helpers
  give a vanishing imaginary part of either sign the +0 side; the -0 side
  differs from MadLoop's values (discriminating).

Further checks:

* the referenced couplings (G_mu): each CT coupling's Laurent coefficients, formed
  on the reference parameters and bound with the plan, equal the UFO's
  coupling expression at the oracle's CT values;
* the digest rule: loop_sm's ``cond``, ``reglog`` and ``complexconjugate``
  are UFO-standard (resolved by name, so loop_sm plans are unchanged), the
  G_mu ``reglog`` is not;
* hygiene (M2): the lowering forms no infinity in a branch that is not
  taken, refuses a coefficient that is not finite, and decides the scalar
  library's predicates of exact numbers (the CMS switch ``CMSParam``).

The ladder (nightly): one rung per electroweak request, a strict xfail
naming the blocker its generation reaches (PLAN wp32 §2) or, once its
blockers are lifted, a plain test that generates and runs its check; an
increment that lifts a blocker flips its rungs.
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
# MadLoop's own CT values at ``_oracle_card`` (both UFOs).
MADLOOP_FIXTURE = "madloop-ct-parameters"
# Relative agreement with MadLoop's quadruple precision (measured: 6.0e-31
# for the double-double plan, 2.8e-33 for the arbitrary-precision one).
QUADRUPLE_TOLERANCE = 1e-28


def _gmu_card(loop_reference) -> tuple[dict[str, str], dict[str, str]]:
    """(the fixture's input parameters with MU_R, its fixed parameters)."""

    fixture = loop_reference(CARD_FIXTURE)
    card = {**fixture["parameters"], "MU_R": fixture["mu_r"][0]}
    return card, dict(fixture["fixed_parameters"])


def _ew_model(ufo, loop_reference):
    """An electroweak UFO's restricted source scalar program, its fixed
    parameters (the default restriction's and those of the fixture's fixed
    parameters the model has: the G_mu UFO lacks the fixture's WTau) and the
    fixture card."""

    from types import SimpleNamespace

    from pyamplicol.loops.model_restrictions import (
        prepare_loop_restrictions,
        restriction_fixed_parameters,
    )
    from pyamplicol.loops.scalar_frontend import lower_scalar_program
    from pyamplicol.models.loading import compile_model_source

    compiled = compile_model_source(ufo, require_supported=False)
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
def gmu(loop_gmu_ufo, loop_reference):
    """The G_mu UFO (``_ew_model``) at the fixture card."""

    return _ew_model(loop_gmu_ufo, loop_reference)


@pytest.fixture(scope="module")
def qcd_qed(loop_qcd_qed_sm_ufo, loop_reference):
    """MG5_aMC's loop_qcd_qed_sm (alpha(M_Z) scheme, ``_ew_model``) at the
    same card (its own aEWM1; the card's Gf is no parameter of it)."""

    return _ew_model(loop_qcd_qed_sm_ufo, loop_reference)


# The UFO modules the oracle reads.
_UFO_PARTS = ("parameters", "CT_parameters", "function_library", "CT_couplings")


def _import_ufo(path, name) -> tuple:
    """The UFO's ``parameters``, ``CT_parameters``, ``function_library`` and
    ``CT_couplings`` modules, imported privately (no bytecode written): a
    UFO with relative imports (the G_mu ``_py3`` conversion) as the package
    ``name``; one with Python-2-style sibling imports (``from object_library
    import ...``, loop_qcd_qed_sm) from its directory as top-level modules.
    Once imported, ``sys.path`` and ``sys.modules`` are restored (a module
    of the same name loaded before is put back): the modules live on in the
    returned objects only."""

    relative = "from .object_library" in (path / "function_library.py").read_text()
    siblings = {source.stem for source in path.glob("*.py")}

    def ours(key: str) -> bool:
        if relative:
            return key == name or key.startswith(f"{name}.")
        return key in siblings

    shadowed = {key: sys.modules.pop(key) for key in list(sys.modules) if ours(key)}
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with warnings.catch_warnings():
            # The UFO's texnames hold invalid escape sequences.
            warnings.simplefilter("ignore", DeprecationWarning)
            if relative:
                spec = importlib.util.spec_from_file_location(
                    name, path / "__init__.py", submodule_search_locations=[str(path)]
                )
                package = importlib.util.module_from_spec(spec)
                sys.modules[name] = package
                spec.loader.exec_module(package)
                prefix = f"{name}."
            else:
                sys.path.insert(0, str(path))
                prefix = ""
            return tuple(
                importlib.import_module(f"{prefix}{part}") for part in _UFO_PARTS
            )
    finally:
        sys.dont_write_bytecode = previous
        if not relative:
            sys.path.remove(str(path))
        for key in [key for key in sys.modules if ours(key)]:
            del sys.modules[key]
        sys.modules.update(shadowed)


@pytest.fixture(scope="module")
def gmu_ufo(loop_gmu_ufo):
    """The G_mu UFO's oracle modules (``_import_ufo``)."""

    return _import_ufo(loop_gmu_ufo, "_pyamplicol_test_ufo_gmu")


@pytest.fixture(scope="module")
def qcd_qed_ufo(loop_qcd_qed_sm_ufo):
    """loop_qcd_qed_sm's oracle modules (``_import_ufo``)."""

    return _import_ufo(loop_qcd_qed_sm_ufo, "_pyamplicol_test_ufo_qcd_qed")


def _counterterm_outputs(raw) -> list[tuple[str, int]]:
    """(name, epsilon power) of every CT parameter's Laurent coefficient."""

    outputs = []
    for record in raw.collections["counterterm_parameters"]:
        source = raw.exact_value("counterterm_parameters", record["id"])
        terms = source["terms"] if source["kind"] == "laurent" else ((0, "0"),)
        outputs += [(record["name"], int(power)) for power, _text in terms]
    return outputs


def _plan(model):
    from pyamplicol.loops.v2.ct_recursion import coefficient_plan

    return coefficient_plan(model.program, _counterterm_outputs(model.raw), model.fixed)


@pytest.fixture(scope="module")
def gmu_plan(gmu):
    """The coefficient plan of every G_mu CT parameter's Laurent
    coefficients."""

    return _plan(gmu)


@pytest.fixture(scope="module")
def qcd_qed_plan(qcd_qed):
    """The coefficient plan of every loop_qcd_qed_sm CT parameter's Laurent
    coefficients."""

    return _plan(qcd_qed)


# The CT-value oracles run on both electroweak UFOs.
EW_MODELS = ("gmu", "qcd_qed")


@pytest.fixture(scope="module", params=EW_MODELS)
def ew(request):
    """(model, its UFO oracle modules, its coefficient plan) of each
    electroweak UFO of ``EW_MODELS``."""

    from types import SimpleNamespace

    name = request.param
    return SimpleNamespace(
        name=name,
        model=request.getfixturevalue(name),
        ufo=request.getfixturevalue(f"{name}_ufo"),
        plan=request.getfixturevalue(f"{name}_plan"),
    )


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
    parameters, counterterms, functions, _couplings = modules
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


def _oracle_card(model) -> dict[str, str]:
    """The fixture card with every fixed parameter and every other external
    parameter at the restricted model's value (the restriction card's: e.g.
    loop_qcd_qed_sm's aEWM1 128.93, not its source default 127.9), as
    decimal strings."""

    defaults = {
        name: repr(real)
        for name, (real, _imag) in model.compiled.parameter_defaults.items()
    }
    fixed = {name: real for name, (real, _imag) in model.fixed.items()}
    return {**defaults, **model.card, **fixed}


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


def test_counterterm_scalars_equal_the_ufo_python_evaluation(ew):
    """Every Laurent coefficient of the CT parameters of each electroweak
    UFO (G_mu, loop_qcd_qed_sm) from the plan (binary64, native scalar
    bindings) equals the UFO's own Python evaluation at the fixture card;
    the coefficients the plan proves zero at the fixed parameters are zero
    there, and none is singular."""

    found = ew.plan
    assert not found.nonfinite
    reference, _environment = _ufo_values(ew.ufo, _oracle_card(ew.model))
    assert set(reference) == set(found.plan.outputs) | found.zero
    assert found.zero and all(reference[key] == 0 for key in found.zero)
    values = _plan_values(found.plan, ew.model.card, arithmetic="double")
    worst = max(
        (_relative(complex(float(re), float(im)), reference[key]), key)
        for key, (re, im) in values.items()
    )
    assert worst[0] <= DOUBLE_TOLERANCE, worst
    # Discriminating: the values span the poles and finite parts of the
    # electroweak UV counterterms, not only zeros.
    assert sum(1 for value in reference.values() if abs(value) > 1e-3) > 50


# The branch-cut helpers of the electroweak UFOs.
_BRANCH_HELPERS = frozenset(
    ("regsqrt", "reglog", "reglogp", "reglogm", "grreglog", "arg")
)
# CT parameters whose helpers take an argument exactly on the cut at the
# card, so the side changes them at real masses (measured on both UFOs:
# HWcft_UV_EW by 80 %, HMass2_UV_EW by 1.7 %; the minimum asserted).
_SIDE_SENSITIVE = {("HWcft_UV_EW", 0): 0.5, ("HMass2_UV_EW", 0): 0.01}


def _on_cut(zero: float):
    """(wrap, moved): ``wrap(name, function)`` gives every argument of a
    branch-cut helper exactly on the negative real axis (imaginary part
    zero, of either sign) the imaginary part ``zero`` (+0.0 or -0.0) and
    records it in ``moved``; arguments off the axis, among them the UFOs'
    deliberate +-1e-10 i regulators, are left alone."""

    moved = []

    def side(z):
        z = complex(z)
        if z.real < 0 and z.imag == 0:
            moved.append(z)
            return complex(z.real, zero)
        return z

    def wrap(name, function):
        if name not in _BRANCH_HELPERS:
            return function
        if name == "grreglog":
            return lambda switch, z1, z2: function(switch, side(z1), side(z2))
        return lambda z: function(side(z))

    return wrap, moved


def _madloop(ew, loop_reference) -> dict:
    """MadLoop's record of ``ew``'s UFO (``MADLOOP_FIXTURE``), whose card is
    the oracle card on every parameter MadLoop's restricted model reads."""

    from decimal import Decimal

    found = loop_reference(MADLOOP_FIXTURE)["models"][ew.name]
    card = _oracle_card(ew.model)
    assert found["card"]
    for name, value in found["card"].items():
        assert Decimal(value) == Decimal(card[name]), name
    return found


def _complex_decimal(pair, digits: int = 60):
    """A (real, imaginary) pair of decimal strings as an mpmath number."""

    import mpmath

    with mpmath.workdps(digits):
        return mpmath.mpc(mpmath.mpf(pair[0]), mpmath.mpf(pair[1]))


# (plan carrier options, MadLoop precision, tolerance); MadLoop's quadruple
# precision reads the card through a binary64 buffer (its lha_read.f), so the
# plan is bound at the card's binary64 values there.
_MADLOOP_CARRIERS = {
    "double": ({"arithmetic": "double"}, "double", DOUBLE_TOLERANCE),
    "double-double": (
        {"arithmetic": "double-double"},
        "quadruple",
        QUADRUPLE_TOLERANCE,
    ),
    "arbitrary": ({"arithmetic": "arbitrary"}, "quadruple", QUADRUPLE_TOLERANCE),
}


@pytest.mark.parametrize("carrier", list(_MADLOOP_CARRIERS))
def test_counterterm_scalars_equal_madloop(ew, loop_reference, carrier):
    """MadLoop's own values are the convention (user, slice-status DECISIONS
    2026-10-25 ~14:10 and ~16:45): every Laurent piece of every CT parameter
    MG5_aMC's Fortran model library writes for the UFO (``MADLOOP_FIXTURE``:
    143 on G_mu, 139 on loop_qcd_qed_sm) equals the plan's coefficient, in
    binary64 against MadLoop's double precision and in double-double and
    arbitrary precision (the working digits of a 60-digit evaluation)
    against its quadruple precision; a piece the plan proves zero is zero
    there.

    Branch sides: at real masses some helper arguments lie exactly on a
    branch cut (e.g. ``regsqrt(MZ^2 (MZ^2 - 4 MW^2))``), and their side
    changes the Higgs two-point CTs by O(1). MadLoop's helpers give a
    vanishing imaginary part of either sign the +0 side, which
    ``ct_recursion._RealArguments`` takes in every lane; the UFO's own
    Python evaluation with those arguments on the -0 side differs from
    MadLoop's values (``_SIDE_SENSITIVE``, discriminating)."""

    from decimal import Decimal

    import mpmath

    from pyamplicol.loops.v2.assemble import Precision

    found = _madloop(ew, loop_reference)
    options, precision, tolerance = _MADLOOP_CARRIERS[carrier]
    if carrier == "arbitrary":
        options = {**options, "precision": Precision.arbitrary(60).working_digits}
    card = _oracle_card(ew.model)
    if precision == "quadruple":
        card = {name: str(Decimal(float(value))) for name, value in card.items()}
    values = _plan_values(ew.plan.plan, card, **options)
    reference = {
        (name, int(power)): pieces[precision]
        for name, powers in found["values"].items()
        for power, pieces in powers.items()
    }
    assert len(reference) == found["pieces"] > 100
    assert set(reference) <= set(values) | ew.plan.zero
    with mpmath.workdps(60):
        worst = max(
            (
                _relative(
                    _complex_decimal(values.get(key, ("0", "0"))),
                    _complex_decimal(pair),
                ),
                key,
            )
            for key, pair in reference.items()
        )
    assert worst[0] <= tolerance, worst
    if carrier != "double":
        return
    wrap, _moved = _on_cut(-0.0)
    minus, _environment = _ufo_values(ew.ufo, _oracle_card(ew.model), wrap=wrap)
    for key, minimum in _SIDE_SENSITIVE.items():
        madloop = complex(*(float(part) for part in reference[key]))
        assert _relative(minus[key], madloop) > minimum, key


def test_both_reglog_conventions_agree_at_real_masses(gmu, gmu_ufo):
    """[U3]: the UFO-standard principal reglog (MadGraph's exporter, the
    ``loops.model_cms`` "ufo-standard" convention) and the G_mu source's CMS
    override (log - 2 pi i for Re z < 0 < Im z, the "source" convention the
    model binds) give equal CT values at real masses, although reglog meets
    the override's quadrant at the card (discriminating)."""

    hits = []

    def wrap(name, function):
        if name != "reglog":
            return function

        def principal(z):
            z = complex(z)
            if z.real < 0.0 and z.imag > 0.0:
                hits.append(z)
            return 0.0 if z == 0 else cmath.log(z)

        return principal

    card = _oracle_card(gmu)
    source, _environment = _ufo_values(gmu_ufo, card)
    standard, _environment = _ufo_values(gmu_ufo, card, wrap=wrap)
    assert hits
    worst = max((_relative(standard[key], source[key]), key) for key in source)
    assert worst[0] <= DOUBLE_TOLERANCE, worst


def test_model_cms_reglog_conventions_agree_at_real_masses(gmu, gmu_plan):
    """[U3] through ``loops.model_cms`` and the production lowering: the
    model's reglog record under each convention of
    ``prepare_gmu_complex_mass_declarations`` ("source": the G_mu override,
    unchanged; "ufo-standard": the principal log), bound in the real-mass
    G_mu model, gives the same CT values at the fixture card in binary64
    (``coefficient_plan``, every helper a scalar-library function) and the
    same structural zeros. The transform's own model enables CMSParam,
    where the conventions differ (eCoup_Gmu_UV_EW and eCoup_DGmua0_UV_EW at
    zero widths): WP3.2 EWI-10."""

    from pyamplicol.loops.model_cms import prepare_gmu_complex_mass_declarations
    from pyamplicol.loops.scalar_frontend import lower_scalar_program
    from pyamplicol.loops.v2.ct_recursion import coefficient_plan
    from pyamplicol.models.loop_import import RawLoopModel

    reglog = gmu.raw.metadata["function_bindings"]["reglog"]
    records = {
        convention: prepare_gmu_complex_mass_declarations(
            gmu.raw, reglog_convention=convention
        ).derived.to_dict()["collections"]["functions"][reglog]
        for convention in ("source", "ufo-standard")
    }
    payload = gmu.raw.to_dict()
    assert records["source"] == payload["collections"]["functions"][reglog]
    assert records["ufo-standard"] != records["source"]
    payload["collections"]["functions"][reglog] = records["ufo-standard"]
    raw = RawLoopModel.from_dict(payload)
    standard = coefficient_plan(
        lower_scalar_program(raw), _counterterm_outputs(raw), gmu.fixed
    )
    assert standard.zero == gmu_plan.zero and not standard.nonfinite
    card = _oracle_card(gmu)
    source = _plan_values(gmu_plan.plan, card)
    values = _plan_values(standard.plan, card)
    assert values.keys() == source.keys()
    worst = max(
        (_relative(complex(*map(float, values[k])), complex(*map(float, v))), k)
        for k, v in source.items()
    )
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
    sources = {c.name: c.value for c in gmu_ufo[3].all_couplings}
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

    def warned() -> bool:
        # Symbolica logs "Created infinity ..." on the process's stdout.
        captured = capfd.readouterr()
        return "Created" in captured.out + captured.err

    warned()
    assert lowering._finite(lowering._resolve(cond(mb, E("1"), mb**-2))) == E("1")
    assert lowering._finite(lowering._resolve(reglog(mb**2 / mu**2))) == E("0")
    assert not warned()
    # The detector's control: forming the branch not taken eagerly warns.
    assert (mb**-2).replace(mb, E("0")).is_finite() is False
    assert warned()
    with pytest.raises(CountertermCouplingError, match="not finite"):
        lowering._finite(lowering._resolve(cond(mu, E("1"), mb**-1)))


def test_scalar_predicates_of_numbers_are_decided(loop_sm_compiled_model):
    """The scalar library's predicates and conditionals of exact numbers are
    decided: the G_mu UFO's CMS switch at its real-mass value,
    if(equal(CMSParam = 0, 1), 1, 0), is zero, so a UFO-standard cond on it
    (loop_sm's) takes its branch without an assumption; an inexact one stays
    undecided."""

    from symbolica import E

    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.loops.scalar_frontend import ScalarFunctionLibrary
    from pyamplicol.loops.v2.ct_recursion import CoefficientLowering, decide_constant
    from pyamplicol.models.loading import load_compiled_model

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
    compiled = load_compiled_model(loop_sm_compiled_model)
    cond = ScalarFunctionLibrary(compiled.raw_loop_model).helpers["cond"][0]
    lowering = CoefficientLowering(compiled, {})
    taken = lowering._resolve(cond(off, E("1"), E("2")))
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


def test_validate_carriers_and_refused_native_inputs(tmp_path, monkeypatch):
    """``tools/loop_v2_generate.py validate``: ``--carrier`` f64, dd and ap
    (``--digits``) select binary64, double-double and arbitrary precision
    (the older ``--precision`` still reads); with ``--inputs native`` a
    process whose trees cannot be exported natively is a failure (exit 1,
    ``point_inputs`` "regenerated-at-load" with the refusal recorded), never
    a fallback to Python inputs."""

    import argparse
    import json
    from types import SimpleNamespace

    from pyamplicol.loops import native as native_module
    from pyamplicol.loops.v2 import plan_export
    from pyamplicol.loops.v2.assemble import LoopSlice, Precision
    from tools import loop_v2_generate

    def carrier(**options):
        return loop_v2_generate._carrier(argparse.Namespace(**options))

    assert carrier(carrier="f64") == ("f64", None)
    assert carrier(carrier="dd") == ("dd", Precision.double_double())
    assert carrier(carrier="ap", digits=40) == ("ap", Precision.arbitrary(40))
    assert carrier(precision="dd") == ("dd", Precision.double_double())
    assert carrier(precision="double") == ("f64", None)

    class Refused:
        def __init__(self, *args, **options):
            raise NotImplementedError("a Weyl-line counterterm root term")

    model = SimpleNamespace(raw_loop_model=None, parameter_defaults={"MB": "4.7"})
    monkeypatch.setattr(loop_v2_generate, "_fixture_model", lambda *a: model)
    monkeypatch.setattr(LoopSlice, "generate", lambda *a, **k: object())
    monkeypatch.setattr(native_module, "load_loop_native", lambda: None)
    monkeypatch.setattr(plan_export, "NativeExecutor", Refused)
    out = tmp_path / "record.json"
    arguments = ["validate", "gg_ttx", "--inputs", "native", "--carrier", "dd"]
    assert loop_v2_generate.main([*arguments, "--json", str(out)]) == 1
    record = json.loads(out.read_text())
    assert record["passed"] is False and record["carrier"] == "dd"
    assert record["point_inputs"] == "regenerated-at-load"
    assert record["native_refusal"] == "a Weyl-line counterterm root term"


def _eager_kernel_proof(remapped, coupling, dimensions):
    """The kernel-equivalence proof as first written: every signature of
    every input order and sign formed whole, the least one taken
    (``compiler_entry._kernel_equivalence_proof`` before its lazy form)."""

    import hashlib
    import json

    candidates, oriented = [], {}
    for order, components in remapped.items():
        oriented[order] = tuple((c * coupling).expand() for c in components)
        for sign in (1, -1):
            signature = json.dumps(
                {
                    "input_dimensions": [dimensions[order[0]], dimensions[order[1]]],
                    "output_dimension": dimensions[2],
                    "components": [
                        (sign * c).expand().to_canonical_string()
                        for c in oriented[order]
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            candidates.append((signature, order, (float(sign), 0.0)))
    signature, order, factor = min(candidates, key=lambda c: c[0])
    exchange = None
    if dimensions[0] == dimensions[1]:
        for sign in (1, -1):
            if all(
                (s - sign * d).expand().to_canonical_string() == "0"
                for d, s in zip(oriented[(0, 1)], oriented[(1, 0)], strict=True)
            ):
                exchange = (float(sign), 0.0)
                break
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    return digest, factor, order, exchange


def test_lazy_kernel_proof_equals_the_eager_one():
    """The tree compiler's kernel-equivalence proof forms the canonical
    strings of the candidates' components lazily (WP3.2 M1: they dominate
    the counterterm-lowered model's compile on the G_mu UFO): its class
    digest, factor, input order and exchange factor equal the eager
    comparison of every complete signature, including candidates tied on
    their first components (zeros, symmetric and antisymmetric kernels) and
    unequal input dimensions."""

    import random

    from symbolica import E, Expression

    from pyamplicol.models import compiler_symbolica
    from pyamplicol.models.compiler_entry import _kernel_equivalence_proof

    compiler_symbolica._ensure_symbolica()
    x, y, z, g = (E(name) for name in ("x", "y", "z", "g"))
    cases = [
        ({(0, 1): (x * y, E("0"), y), (1, 0): (y * x, E("0"), -y)}, g, (4, 4, 4)),
        ({(0, 1): (E("0"), x - y), (1, 0): (E("0"), y - x)}, g + 1, (2, 2, 1)),
        ({(0, 1): (x, y), (1, 0): (y, x)}, g, (4, 1, 4)),
        ({(0, 1): (x + y,), (1, 0): (y + x,)}, E("1"), (1, 1, 1)),
        ({(0, 1): (x - y, z), (1, 0): (y - x, -z)}, g * z, (3, 3, 3)),
    ]
    terms = [x, y, z, x * y, x**2, y * z, E("2") * x, E("-3/2") * z, Expression.I * y]
    rng = random.Random(7)
    for _ in range(40):
        count = rng.randint(1, 4)

        def component():
            if rng.random() < 0.2:
                return E("0")
            return sum(
                (rng.choice((1, -1)) * rng.choice(terms) for _ in range(3)), E("0")
            )

        direct = tuple(component() for _ in range(count))
        swapped = tuple(
            rng.choice((1, -1)) * c if rng.random() < 0.5 else component()
            for c in direct
        )
        dimensions = (rng.choice((1, 4)), rng.choice((1, 4)), rng.choice((1, 4)))
        coupling = rng.choice((E("1"), g, g + x, E("-1") * g))
        cases.append(({(0, 1): direct, (1, 0): swapped}, coupling, dimensions))
    for remapped, coupling, dimensions in cases:
        assert _kernel_equivalence_proof(
            remapped, coupling, dimensions
        ) == _eager_kernel_proof(remapped, coupling, dimensions), (
            remapped,
            coupling,
            dimensions,
        )


def _rule(rules, kind, particles, alternative):
    """The prepared CT rule of ``kind`` on the field multiset ``particles``
    with the loop alternative ``alternative`` (|PDG| codes)."""

    (rule,) = [
        rule
        for rule in rules.values()
        if rule.kind == kind
        and rule.entries
        and sorted(rule.entries[0].particle_names) == sorted(particles)
        and tuple(alternative) in rule.alternatives
    ]
    return rule


def test_counterterm_loops_are_loops_of_the_selected_corrections(
    gmu, loop_sm_compiled_model
):
    """The plain admission (design §6.2) with the model's perturbative species
    facts: a counterterm's loop alternative is admitted only as a loop of a
    selected correction, every particle of it perturbative in one (the
    families' eligible species). The UFO declares no correction type of a CT
    vertex, so this tells the QCD correction of the photon vertex (the R2 of
    u u~ a with a u-gluon loop, orders QCD^2 QED) from the QED one (its UV
    counterterm with a u-photon loop) when a mixed-order Born admits both by
    coupling orders (u u~ > d d~ with QED corrections, Q3). Without facts the
    rule is the old one, and loop_sm's QCD admission is unchanged rule by
    rule."""

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.v2.ct_recursion import admitted_alternatives
    from pyamplicol.models.loading import load_compiled_model

    prepared = prepare_loop_request_model(
        gmu.compiled, fixed_parameters={}, compatibility="gmu_particle_anti"
    )
    species = prepared.structure.perturbative_species_orders
    rules = prepared.counterterms.rules
    gluon_loop = _rule(rules, "R2", ("A", "u", "u~"), (2, 21))
    photon_loop = _rule(rules, "UVloop", ("A", "u", "u~"), (2, 22))
    admitted = {
        (name, corrections): admitted_alternatives(rule, (), corrections, species)
        for name, rule in (("gluon", gluon_loop), ("photon", photon_loop))
        for corrections in (("QCD",), ("QED",), ("QCD", "QED"))
    }
    assert admitted == {
        ("gluon", ("QCD",)): (0,),
        ("gluon", ("QED",)): (),
        ("gluon", ("QCD", "QED")): (0,),
        ("photon", ("QCD",)): (),
        ("photon", ("QED",)): (0,),
        ("photon", ("QCD", "QED")): (0,),
    }
    assert admitted_alternatives(gluon_loop, (), ("QED",)) == (0,)
    assert admitted_alternatives(gluon_loop, (), None, species) == (0,)
    loop_sm = prepare_loop_request_model(
        load_compiled_model(loop_sm_compiled_model), fixed_parameters={}
    )
    facts = loop_sm.structure.perturbative_species_orders
    assert all(
        admitted_alternatives(rule, (), ("QCD",), facts)
        == admitted_alternatives(rule, (), ("QCD",))
        for rule in loop_sm.counterterms.rules.values()
    )


# ------------------------------------------------------------ native inputs

# Native against Python inputs, as ``test_loop_v2_native_trees`` checks them:
# every component on the scale of its sector's complete virtual.
# (arithmetic, tolerance, points)
NATIVE_TOLERANCES = (("double", 1e-12, 3), ("double-double", 1e-26, 1))


def _native_deviation(native, python) -> float:
    from decimal import Decimal

    def real(value) -> Decimal:
        if isinstance(value, complex):
            assert value.imag == 0
            value = value.real
        return Decimal(value)

    worst = Decimal(0)
    for mine, theirs in zip(native, python, strict=True):
        assert mine.components.keys() == theirs.components.keys()
        for name, sectors in theirs.components.items():
            for sector, value in sectors.items():
                scale = max(
                    abs(real(x))
                    for x in theirs.components["virtual"][sector].coefficients
                )
                for x, y in zip(
                    mine.components[name][sector].coefficients,
                    value.coefficients,
                    strict=True,
                ):
                    worst = max(worst, abs(real(x) - real(y)) / scale)
    return float(worst)


@pytest.mark.loop_nightly
def test_counterterm_references_are_native_parameter_program_inputs(
    gmu, loop_native, loop_reference
):
    """The G_mu UFO's QCD CT scalars (G_UVb, G_UVt, tMass_UV) call its CMS
    reglog, so g g > t t~ with QCD corrections lowers them on reference
    parameters (EWI-3). The native parameter program binds them per point
    from the coefficient plan (``native_trees._CountertermReferences``): the
    plan exports native point inputs (no load-time regeneration, ADDENDUM G)
    and every component of the native virtual equals the Python inputs'
    (``CountertermModel.coefficient_values``) in binary64 at the fixture's
    three points and scales, and in double-double at one. Nightly: the cold
    G_mu CT lowering takes 83 s (pinned idle cores), whatever the QCD
    request (review minor 13: the default tier holds no EW CT lowering)."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    request = SliceRequest(
        "g g > t t~",
        {"QCD": 2, "QED": 0},
        {"QCD": 4, "QED": 0},
        corrections=("QCD",),
        fixed_parameters=_G_MU_FIXED,
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )
    slice_ = LoopSlice.generate(gmu.compiled, request, loop_native)
    model = slice_.counterterm_program.recursion.model
    assert {scalar for scalar, _power, _part in model.coefficients.values()} >= {
        "G_UVb",
        "G_UVt",
        "tMass_UV",
    }
    native = NativeExecutor(slice_, forward=False, inputs="native")
    record = native.exported.trees.program["parameters"]
    assert any(item[0] == "output" for item in record["counterterm_model"])
    python = NativeExecutor(slice_, forward=False, exported=native.exported)
    fixture = loop_reference("required/gg_ttx")
    points, scales = fixture["points"], [str(int(mu) ** 2) for mu in fixture["mu_r"]]
    for arithmetic, tolerance, count in NATIVE_TOLERANCES:
        options = {"parameters": {"aS": "0.118", "MT": "173"}}
        if arithmetic == "double-double":
            options["precision"] = Precision.double_double()
        mine = native.evaluate(points[:count], scales[:count], **options)
        theirs = python.evaluate(points[:count], scales[:count], **options)
        assert _native_deviation(mine, theirs) <= tolerance, arithmetic


# ------------------------------------------------------------------ nightly


@pytest.mark.loop_nightly
@pytest.mark.parametrize("carrier", ["double-double", "arbitrary"])
def test_counterterm_scalars_equal_mpmath(ew, carrier):
    """The plan's CT values of each electroweak UFO (G_mu, loop_qcd_qed_sm)
    in the native double-double carrier and at the working digits of a
    60-digit evaluation equal an mpmath transliteration of the UFO's source
    (every literal exact) at 64 and 140 digits: to 1e-28 and 1e-55
    relative; the plan's zeros are zero there. mpmath has no negative zero:
    its principal branch is the +0 side of the cut, MadLoop's and the plan's
    (``test_counterterm_scalars_equal_madloop``)."""

    import mpmath

    from pyamplicol.loops.v2.assemble import Precision

    if carrier == "double-double":
        options, digits, tolerance = {"arithmetic": "double-double"}, 64, "1e-28"
    else:
        working = Precision.arbitrary(60).working_digits
        options = {"arithmetic": "arbitrary", "precision": working}
        digits, tolerance = 2 * working, "1e-55"
    reference, _environment = _ufo_values(ew.ufo, _oracle_card(ew.model), digits=digits)
    values = _plan_values(ew.plan.plan, ew.model.card, **options)
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
        assert all(reference[key] == 0 for key in ew.plan.zero)


@pytest.mark.loop_nightly
def test_every_referenced_coupling_equals_the_ufo_coupling(gmu, gmu_ufo, gmu_plan):
    """``test_referenced_couplings_equal_the_ufo_couplings`` on every
    coupling of the model."""

    _check_referenced_couplings(gmu, gmu_ufo, gmu_plan, every=1)


# ------------------------------------------------------------------- ladder

# Blockers (PLAN wp32 §2): the exception and message each request stops at.
# B1 (the CT helpers) is lifted by EWI-3; B2 (loop species), B5 (a W on a
# loop quark line) and B6 (a W at the root) by EWI-1 and EWI-2 (ew-lines);
# B3 (unitary-gauge massive vectors in the loops) by EWI-4b (ew-gauge: every
# electroweak request takes 't Hooft-Feynman gauge throughout, its
# Goldstone-vector mixings are lowered as vertex insertions).
# B4: more than one colour-singlet fermion line, refused when the request is
# prepared, before any tree or counterterm is built
# (``ct_sites.require_one_singlet_line``, until EWI-5).
B4 = ("FermionSignError", r"colour-singlet fermion lines .*Koszul signs")
# B4 in the loop: a colour-singlet fermion line through the loop
# (``conventions.LeafWeights``, until EWI-5), reached after the families,
# the counterterm program and the kernel catalogue.
B4_LOOP = ("FermionSignError", r"colour-singlet fermion line through the loop")
LIFTED = None  # the rung generates (and runs its check)
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


# g t > t t t~ (binary64, on shell to rounding): a top line joined to a top
# pair by a Z, a photon or a G0, with the gluon at leg 1.
_GT_TTTX_POINT = [
    ["740.0236666666667", "0", "0", "740.0236666666667"],
    ["759.9763333333334", "0", "0", "-740.0236666666667"],
    [
        "542.1014268447362",
        "453.946802621722",
        "-234.14080644548642",
        "55.27513124065375",
    ],
    [
        "679.7890319233611",
        "-642.7918716317638",
        "125.73449804624319",
        "-56.51171284339019",
    ],
    [
        "278.1095412319028",
        "188.84506901004178",
        "108.40630839924326",
        "1.2365816027364396",
    ],
]


def _rung(name, model, process, born, loop, blocker, **options):
    """One ladder rung: a strict xfail raising its blocker's exception, or a
    plain test once its blockers are lifted (``LIFTED``)."""

    import pyamplicol.loops.v2.catalogue as catalogue
    import pyamplicol.loops.v2.conventions as conventions

    errors = {
        "LoopKernelCatalogueError": catalogue.LoopKernelCatalogueError,
        "FermionSignError": conventions.FermionSignError,
    }
    marks = ()
    if blocker is not LIFTED:
        marks = pytest.mark.xfail(
            strict=True, raises=errors[blocker[0]], reason=f"blocker: {blocker[1]}"
        )
    return pytest.param(
        model, process, born, loop, blocker, options, id=name, marks=marks
    )


LADDER = [
    # Photon-only QED corrections of a lepton process: two colour-singlet
    # fermion lines (B4, EWI-5; B2 was lifted by EWI-1).
    _rung(
        "qq_eemm_photon",
        "loop_qcd_qed_sm",
        "e- e+ > m- m+",
        {"QED": 2, "QCD": 0},
        {"QED": 4, "QCD": 0},
        B4,
        corrections=("QED",),
        particle_veto=_PHOTON_ONLY,
        fixed_parameters=_ZERO_WIDTHS,
    ),
    # Lifted by EWI-1 (B2 at the root: a lepton as leg 1); its values
    # against legacy: ``test_loop_v2_ew_lines`` (epem_uux).
    _rung(
        "sm_epem_uux",
        "loop_sm",
        "e+ e- > u u~",
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
        LIFTED,
    ),
    # Q3: photon-only QED corrections of u u~ > d d~ pass the CT lowering
    # (EWI-3). Their CT roots declare no loop species since the admission
    # takes only loops of the selected corrections (the QCD R2 of u u~ a
    # with a gluon loop declared the gluon and the quark before); the
    # catalogue takes the families' species since EWI-1. Validated: the bare
    # and counterterm UV poles cancel (UV renormalisation is complete). Not
    # validated: the finite part. loop_qcd_qed_sm writes the photon and Z
    # parts of the u u~ g R2 in one coupling (R2_GUU2Cp, (1 + lhv)/cw^2 (Qu^2
    # + I3u^2/sw^2 - 2 Qu I3u) / 16 + ...) with the loop particles [[u, Z]],
    # so the photon-only particle_veto drops its photon part too (r2 = 0):
    # a veto that splits such a coupling is for the [U2] guard (EWI-7).
    _rung(
        "uudd_qed_photon_uv",
        "loop_qcd_qed_sm",
        "u u~ > d d~",
        {"QCD": 2, "QED": 0},
        {"QCD": 2, "QED": 2},
        LIFTED,
        corrections=("QED",),
        particle_veto=_PHOTON_ONLY,
        point=_MASSLESS_POINT,
        poles="uv",
        fixed_parameters=_ZERO_WIDTHS,
    ),
    # Lifted by EWI-4b: the full QED corrections (Z, W, H, Goldstones and
    # ghosts in the loops, 't Hooft-Feynman gauge) generate and their bare
    # and counterterm UV poles cancel (the finite part is EWI-7's).
    _rung(
        "uudd_qed",
        "loop_qcd_qed_sm",
        "u u~ > d d~",
        {"QCD": 2, "QED": 0},
        {"QCD": 2, "QED": 2},
        LIFTED,
        corrections=("QED",),
        point=_MASSLESS_POINT,
        poles="uv",
        fixed_parameters=_ZERO_WIDTHS,
    ),
    _rung(
        "uudd_qcd_qed",
        "loop_qcd_qed_sm",
        "u u~ > d d~",
        {"QCD": 2, "QED": 0},
        ({"QCD": 4, "QED": 0}, {"QCD": 2, "QED": 2}),
        LIFTED,
        corrections=("QCD", "QED"),
        fixed_parameters=_ZERO_WIDTHS,
    ),
    # Lifted by EWI-2 (B5: the W on a loop quark line); its values against
    # legacy: ``test_loop_v2_ew_lines`` (gb_twm).
    _rung(
        "gb_twm",
        "loop_sm",
        "g b > t w-",
        {"QCD": 1, "QED": 1},
        {"QCD": 3, "QED": 1},
        LIFTED,
        fixed_parameters={"MB": "0", "WT": "0"},
        point=_GB_TW_POINT,
    ),
    # Lifted by EWI-2 (B6: the W at the root, read through the conjugate
    # rule); its values against legacy: ``test_loop_v2_ew_lines`` (udx_epve).
    _rung(
        "udx_epve",
        "loop_sm",
        "u d~ > e+ ve",
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
        LIFTED,
    ),
    # The G_mu ladder of PLAN wp32 §1 (real masses, widths zero). e- e+ >
    # m- m+ has two colour-singlet fermion lines: B4 (EWI-5) is refused
    # before its CT lowering, which M1 measured before that guard (EWI-1).
    _rung(
        "gmu_eemm",
        "loop_gmu",
        "e- e+ > m- m+",
        {"QED": 2, "QCD": 0},
        {"QED": 4, "QCD": 0},
        B4,
        corrections=("QED",),
        fixed_parameters=_G_MU_FIXED,
    ),
    # Lifted by EWI-4b; its values against MadLoop:
    # ``test_loop_v2_ew_gauge.test_gg_ttx_electroweak_equals_madloop_at_point_0``.
    _rung(
        "gmu_gg_ttx",
        "loop_gmu",
        "g g > t t~",
        {"QCD": 2, "QED": 0},
        {"QCD": 2, "QED": 2},
        LIFTED,
        corrections=("QCD", "QED"),
        fixed_parameters=_G_MU_FIXED,
    ),
    # The Goldstone-vector mixings (G0-Z, G0-A, G+-W-) as vertex insertions
    # (EWI-4b): its trees join the top line and the top pair through a Z or
    # photon on one side and a G0 on the other, with the gluon at leg 1; the
    # UV poles of the QED corrections cancel only with the mixings (they do
    # not with the mixings' momentum sign flipped, ew-gauge M-mix probe).
    _rung(
        "gmu_gt_tttx_mixing",
        "loop_gmu",
        "g t > t t t~",
        {"QCD": 1, "QED": 2},
        {"QCD": 1, "QED": 4},
        LIFTED,
        corrections=("QED",),
        point=_GT_TTTX_POINT,
        poles="uv",
        fixed_parameters=_G_MU_FIXED,
    ),
    # The lepton line of the Z/photon decay runs through the loops (B4, EWI-5);
    # g g > t t~ e- e+ lowers its G0-Z and G0-A mixings first (EWI-4b).
    _rung(
        "gmu_uux_emepg",
        "loop_gmu",
        "u u~ > e- e+ g",
        {"QCD": 1, "QED": 2},
        {"QCD": 1, "QED": 4},
        B4_LOOP,
        corrections=("QCD", "QED"),
        fixed_parameters=_G_MU_FIXED,
    ),
    _rung(
        "gmu_gg_ttxemep",
        "loop_gmu",
        "g g > t t~ e- e+",
        {"QCD": 2, "QED": 2},
        {"QCD": 2, "QED": 4},
        B4_LOOP,
        corrections=("QCD", "QED"),
        fixed_parameters=_G_MU_FIXED,
    ),
    _rung(
        "gmu_gg_ttxemepg",
        "loop_gmu",
        "g g > t t~ e- e+ g",
        {"QCD": 3, "QED": 2},
        {"QCD": 3, "QED": 4},
        B4_LOOP,
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


def _check_admitted_loops(slice_, request) -> None:
    """Every loop alternative the request's CT program admits is a loop of a
    selected correction (``ct_recursion.admitted_alternatives`` with the
    model's perturbative species facts)."""

    program = slice_.counterterm_program
    if program is None:
        return
    recursion = program.recursion
    species = recursion.model.species_orders
    assert species is not None
    corrections = frozenset(request.corrections)
    for name, indices in recursion.admission.items():
        for index in indices:
            alternative = recursion.rules[name].alternatives[index]
            assert all(
                species.get(abs(pdg), frozenset()) & corrections for pdg in alternative
            ), (name, alternative)


@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    ("model", "process", "born", "loop", "blocker", "options"), LADDER
)
def test_electroweak_ladder_reaches_its_blocker(
    ladder_models, loop_native, model, process, born, loop, blocker, options
):
    """Each rung stops at its named blocker, never in the CT lowering
    (WP3.2 EWI-3); a rung whose blockers are lifted generates, runs its
    check (the UV poles cancel) and is a plain test."""

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
        _check_admitted_loops(slice_, request)
        if point is not None:
            result = slice_.evaluate(
                point, "1000000", parameters={"aS": "0.118", "MT": "173"}, poles=poles
            )
    except Exception as error:
        if blocker is LIFTED:
            raise
        if type(error).__name__ == blocker[0] and re.search(blocker[1], str(error)):
            raise
        raise AssertionError(f"stopped before its blocker: {error!r}") from error
    if poles == "uv":
        # Bare UV poles + counterterm UV poles = 0 per sector and coefficient
        # (REQUIREMENTS_ADDENDUM I): UV renormalisation is complete (the
        # finite part is not checked here).
        for sector, bare in result.components["bare"].items():
            counterterms = result.components["uv"][sector]
            for index in (1, 2):
                total = float(bare.coefficients[index]) + float(
                    counterterms.coefficients[index]
                )
                assert abs(total) <= 1e-10 * max(
                    abs(float(bare.coefficients[index])), 1e-300
                ), (sector, index, bare, counterterms)
