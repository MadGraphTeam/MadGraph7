# SPDX-License-Identifier: 0BSD
"""Exact UFO scalar expressions and dependency-owned Laurent arithmetic.

This module never evaluates Python model expressions or converts input through
binary64. Decimal token normalization uses Python's exact Fraction constructor;
the existing UFO parser and Symbolica own expression and series algebra.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass, field
from fractions import Fraction
from graphlib import TopologicalSorter
from types import MappingProxyType
from typing import Any

from .._internal.physics.symbols import symbols


def parse_exact_ufo(expression: str) -> Any:
    """Parse the UFO expression language with exact numeric literal spelling.

    Arbitrary Python programs, conditional expressions and attribute evaluation
    are not supported. The UFO parser supplies complex, conjugate, cond, reglog,
    trigonometric and square-root semantics. Unexpanded model functions remain
    symbolic, with their raw definitions retained by LoopModelIR.
    """
    from ufo_model_loader.symbolica_processing import parse_python_expression_safe

    syntax = ast.parse(expression, mode="eval")
    supported = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Attribute,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.UAdd,
        ast.USub,
    )
    for node in ast.walk(syntax):
        if not isinstance(node, supported):
            raise NotImplementedError(
                f"unsupported UFO scalar syntax: {type(node).__name__}"
            )
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool)
            or not isinstance(node.value, (int, float, complex))
        ):
            raise NotImplementedError("UFO scalar constants must be numeric")
        if isinstance(node, ast.Attribute) and (
            not isinstance(node.value, ast.Name)
            or node.value.id not in {"cmath", "math"}
        ):
            raise NotImplementedError("only standard math attributes are supported")
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(expression).readline):
        value = token.string
        if token.type == tokenize.NUMBER:
            imaginary = value.lower().endswith("j")
            numeric = value[:-1] if imaginary else value
            # Fraction accepts decimal/scientific strings without float. UFO
            # integers may use Python base prefixes; handle these with int.
            if numeric.lower().startswith(("0x", "0b", "0o")):
                rational = Fraction(int(numeric, 0))
            else:
                rational = Fraction(numeric)
            value = f"({rational.numerator}/{rational.denominator})"
            if imaginary:
                value = f"complex(0,{value})"
        if token.type not in {
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.ENDMARKER,
            tokenize.COMMENT,
        }:
            tokens.append(value)
    # Validated arithmetic contains no significant inter-token whitespace.
    # Symbolica requires a function head immediately followed by its opening '('.
    normalized = "".join(tokens)
    # These qualified standard functions have the same native Symbolica heads.
    # No custom function evaluation or branch prescription is introduced here.
    for module in ("cmath", "math"):
        for function in ("log", "exp", "tan", "atan", "sinh", "cosh", "tanh"):
            normalized = normalized.replace(f"{module}.{function}", function)
    return parse_python_expression_safe(normalized)


def epsilon_symbol() -> Any:

    return symbols.qualified("pyamplicol_loop::epsilon")


def _value_expression(value: Any) -> Any:
    from symbolica import E

    if value["kind"] == "expression":
        return parse_exact_ufo(value["expression"])
    if value["kind"] == "laurent":
        epsilon = epsilon_symbol()
        return sum(
            (parse_exact_ufo(text) * epsilon**power for power, text in value["terms"]),
            E("0"),
        )
    raise ValueError("unknown raw UFO scalar value representation")


@dataclass(frozen=True, slots=True)
class LaurentExpression:
    """A full exact expression; truncation occurs only at an explicit request."""

    expression: Any = field(repr=False)

    def series(self, maximum_power: int = 0) -> Any:
        """Return native Series including maximum_power and its remainder order."""
        if isinstance(maximum_power, bool) or not isinstance(maximum_power, int):
            raise TypeError("maximum epsilon power must be an integer")
        return self.expression.series(
            epsilon_symbol(), 0, maximum_power, depth_is_absolute=True
        )

    def coefficient(self, power: int) -> Any:
        return self.series(power)[power]


@dataclass(frozen=True, slots=True)
class ScalarDefinition:
    collection: str
    id: int
    name: str
    source: Any = field(repr=False)
    value: LaurentExpression


def lower_scalar_definitions(raw: Any) -> Any:
    """Substitute all CT dependencies before requesting any epsilon coefficient.

    Ordinary parameters stay symbolic in couplings, preserving their exact
    definitions separately. Ordinary definitions containing CT parameters are
    substituted as well, so they cannot conceal a Laurent pole. Symbolic names
    must be unique across scalar declarations; vertex display names need not be.
    """

    definitions = {}
    by_symbol = {}
    ct_symbols = set()
    for collection in ("parameters", "counterterm_parameters", "couplings"):
        for record in raw.collections[collection]:
            symbol = symbols.qualified(f"UFO::{record['name']}")
            if symbol in by_symbol:
                raise ValueError(f"ambiguous UFO scalar name {record['name']!r}")
            expression = _value_expression(raw.exact_value(collection, record["id"]))
            if epsilon_symbol() in expression.get_all_symbols():
                # Only our Laurent dictionary construction may introduce eps.
                value = raw.exact_value(collection, record["id"])
                if value["kind"] != "laurent":
                    raise ValueError("reserved loop epsilon symbol in UFO expression")
            key = (collection, record["id"])
            definitions[key] = (record, expression)
            by_symbol[symbol] = key
            if (
                collection == "counterterm_parameters"
                or epsilon_symbol() in expression.get_all_symbols()
            ):
                ct_symbols.add(symbol)
    dependencies = {
        symbol: set(definitions[key][1].get_all_symbols()) & by_symbol.keys()
        for symbol, key in by_symbol.items()
    }
    # TopologicalSorter is the standard library implementation, not a private
    # graph algorithm. It also fails explicitly on recursive model definitions.
    resolved = {}
    ct_dependent = set(ct_symbols)
    for symbol in TopologicalSorter(dependencies).static_order():
        expression = definitions[by_symbol[symbol]][1]
        for dependency in dependencies[symbol]:
            if dependency in ct_dependent:
                expression = expression.replace(dependency, resolved[dependency])
                ct_dependent.add(symbol)
        resolved[symbol] = expression
    result = {}
    for symbol, key in by_symbol.items():
        record, source = definitions[key]
        result[key] = ScalarDefinition(
            key[0], key[1], record["name"], source, LaurentExpression(resolved[symbol])
        )
    return MappingProxyType(result)
