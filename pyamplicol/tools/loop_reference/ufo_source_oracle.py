# SPDX-License-Identifier: 0BSD
"""A loop UFO's own Python source as an oracle (moved from
``tests/unit/test_loop_v2_ew.py``, whose tests, ``test_loop_v2_ew_gauge.py``,
the campaign harness ``tools/loop_benchmark/mqcd.py`` (the R2 rule's
coupling classes) and ``tools/loop_benchmark/msm_manifest.py`` (the photon
conversion values) read it).

``import_ufo`` imports a UFO's ``parameters``, ``CT_parameters``,
``function_library`` and ``CT_couplings`` privately; ``ufo_source_values``
evaluates its parameters and every Laurent coefficient of its CT parameters
at a card, in binary64 (``cmath``) or with mpmath with every number literal
exact; ``compiled`` compiles one UFO expression for that evaluation
(``cond(c, a, b)`` lazily, the branch not taken never evaluated).
"""

from __future__ import annotations

import ast
import cmath
import importlib
import importlib.util
import math
import sys
import warnings

# The UFO modules the oracle reads.
UFO_PARTS = ("parameters", "CT_parameters", "function_library", "CT_couplings")


def import_ufo(path, name) -> tuple:
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
                importlib.import_module(f"{prefix}{part}") for part in UFO_PARTS
            )
    finally:
        sys.dont_write_bytecode = previous
        if not relative:
            sys.path.remove(str(path))
        for key in [key for key in sys.modules if ours(key)]:
            del sys.modules[key]
        sys.modules.update(shadowed)


class Source(ast.NodeTransformer):
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


def compiled(source: str, exact: bool):
    tree = Source(source, exact).visit(ast.parse(source, mode="eval"))
    return compile(ast.fix_missing_locations(tree), "<ufo>", "eval")


class Library:
    """``cmath`` and ``math`` of the mpmath transliteration."""

    def __init__(self, mp):
        self._mp, self.pi = mp, mp.pi

    def __getattr__(self, name):
        return getattr(self._mp, name)


def ufo_values(modules, card, *, digits=None, wrap=None) -> dict:
    """{(CT parameter, epsilon power): value} and the parameters' values of
    the UFO's own source at ``card`` (decimal strings; other external
    parameters at the UFO's values): Python's ``cmath`` in binary64, or
    mpmath at ``digits``. ``wrap(name, function)`` may replace a helper."""

    if digits is not None:
        import mpmath

        with mpmath.workdps(digits):
            return ufo_source_values(modules, card, exact=True, wrap=wrap)
    return ufo_source_values(modules, card, exact=False, wrap=wrap)


def ufo_source_values(modules, card, *, exact, wrap) -> dict:
    parameters, counterterms, functions, _couplings = modules
    if exact:
        import mpmath

        mp = mpmath.mp
        environment = {
            "cmath": Library(mp),
            "math": Library(mp),
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
            compiled(f"lambda {','.join(arguments)}: {function.expr}", exact),
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
                compiled(parameter.value, exact), environment
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
                compiled(source, exact), environment
            )
    return values, environment


def coupling_ratio(path, numerator: str, denominator: str, card) -> float:
    """The ratio of two UFO coupling values at ``card``, from the UFO's own
    Python source at ``path`` (binary64; the CT parameters at their
    epsilon^0 coefficient), a real number."""

    modules = import_ufo(path, "_pyamplicol_ufo_coupling_ratio")
    values, environment = ufo_source_values(modules, card, exact=False, wrap=None)
    environment = {
        **environment,
        **{key: value for (key, power), value in values.items() if power == 0},
    }
    couplings = modules[3]

    def value(coupling: str) -> complex:
        # A coupling the UFOmodeldb issue-1 patch splits by loop content
        # (R2_G0tt_b = R2_G0tt_b_WG + R2_G0tt_b_G) is the sum of its pieces.
        parts = (
            (coupling,)
            if hasattr(couplings, coupling)
            else (f"{coupling}_WG", f"{coupling}_G")
        )
        return sum(
            complex(eval(compiled(getattr(couplings, part).value, False), environment))
            for part in parts
        )

    ratio = value(numerator) / value(denominator)
    if abs(ratio.imag) > 1e-15 * abs(ratio.real):
        raise ValueError(f"{numerator} / {denominator} is not real: {ratio}")
    return ratio.real
