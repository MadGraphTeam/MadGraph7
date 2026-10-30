# SPDX-License-Identifier: 0BSD
"""Bounded exact source helpers, native conditionals and active-path diagnostics.

The frontend translates syntax; Symbolica owns all expression algebra and
numerical execution. A program always exports its value and its diagnostic.
Both must be checked for finiteness before accepting a numerical result.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import tokenize
from dataclasses import dataclass
from fractions import Fraction
from graphlib import TopologicalSorter
from types import MappingProxyType
from typing import Any

from .._internal.physics.symbols import symbols


@dataclass(frozen=True, slots=True)
class ScalarExpression:
    value: Any
    diagnostic: Any


def _producer_predicate(mode: str):
    """Only mark evaluator calls; native execution rebinds fixed typed callbacks.

    The binary64 implementation also defines correct ordinary producer semantics.
    Constant calls remain in the saved exact evaluator until carrier mapping.
    """

    def predicate(args):
        if len(args) == 1 and mode in {"real_part", "imaginary_part"}:
            return complex(args[0].real if mode == "real_part" else args[0].imag)
        count = 1 if mode in {"truth", "real_part", "imaginary_part"} else 2
        if len(args) != count or any(
            not math.isfinite(x.real) or not math.isfinite(x.imag) for x in args
        ):
            return complex(float("nan"))
        if mode in {"less", "less_equal"} and any(x.imag != 0 for x in args):
            return complex(float("nan"))
        if mode == "truth":
            result = args[0] != 0
        elif mode == "equal":
            result = args[0] == args[1]
        elif mode == "less":
            result = args[0].real < args[1].real
        else:
            result = args[0].real <= args[1].real
        return complex(result)

    return predicate


class ScalarFunctionLibrary:
    """A source-bound helper library; raw registry order never chooses a helper."""

    def __init__(self, raw: Any):
        from symbolica import E, FunctionDefinition

        self.raw = raw
        # Lowered sources without local bindings, by (source, closed): a pure
        # function of the source and this library (``lower``).
        self._lowered: dict[tuple[str, bool], ScalarExpression] = {}
        self.zero, self.one, self.imaginary = E("0"), E("1"), E("1i")
        self.choose = symbols.qualified("symbolica::if")
        self.predicates = {
            name: symbols.qualified(
                f"pyamplicol_scalar_v1::{name}",
                eval={"complex": _producer_predicate(name)},
            )
            for name in (
                "truth",
                "equal",
                "less",
                "less_equal",
                "real_part",
                "imaginary_part",
            )
        }
        self._choose_name = self.choose.get_name()
        self._predicate_heads = {
            symbol.get_name(): name for name, symbol in self.predicates.items()
        }
        records = raw.collections["functions"]
        bindings = dict(raw.metadata.get("function_bindings", {}))
        if records and raw.metadata.get("function_binding_scope") != "function_library":
            raise ValueError("source helper namespace bindings were not captured")
        selected = {}
        for name, record_id in bindings.items():
            if (
                not isinstance(name, str)
                or not name.isidentifier()
                or type(record_id) is not int
            ):
                raise ValueError("invalid source helper binding")
            if not 0 <= record_id < len(records):
                raise ValueError("invalid source helper record identity")
            if name in {"math", "cmath"}:
                raise ValueError("helper binding shadows a standard math module")
            record = records[record_id]
            fields = record["fields"]
            if not fields.get("expression_source", {}).get("exact", False):
                from pyamplicol.models.source_refusals import refuse_value_writes

                refuse_value_writes(raw.name, fields["expression_source"])
                raise ValueError(f"source helper {name!r} lacks a certified body")
            arguments = fields["arguments"]
            if isinstance(arguments, str):
                # Recognize historical UFO ('z'), not arbitrary string iteration.
                if len(arguments) != 1 or not arguments.isidentifier():
                    raise ValueError("ambiguous source helper argument string")
                arguments = (arguments,)
            elif isinstance(arguments, dict) or hasattr(arguments, "keys"):
                if arguments.get("kind") != "tuple":
                    raise ValueError("helper arguments must be a tuple")
                arguments = tuple(arguments["items"])
            else:
                raise ValueError("helper arguments must be a tuple")
            if len(set(arguments)) != len(arguments) or any(
                not isinstance(arg, str) or not arg.isidentifier() for arg in arguments
            ):
                raise ValueError("invalid source helper arguments")
            if set(arguments) & {"math", "cmath"}:
                raise ValueError("helper arguments shadow a standard math module")
            body = fields["expr"]
            if not isinstance(body, str):
                raise ValueError("source helper body must be a string")
            selected[name] = (record_id, arguments, body)
        identity = hashlib.sha256(
            json.dumps(
                {"model": raw.name, "helpers": selected}, sort_keys=True
            ).encode()
        ).hexdigest()
        self.namespace = f"pyamplicol_scalar_model_{identity}"
        self.helpers = {
            name: (
                symbols.qualified(f"{self.namespace}::value_{record_id}"),
                symbols.qualified(f"{self.namespace}::diagnostic_{record_id}"),
                arguments,
            )
            for name, (record_id, arguments, _body) in selected.items()
        }
        dependencies = {
            name: {
                node.func.id
                for node in ast.walk(ast.parse(body, mode="eval"))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in selected
            }
            for name, (_record_id, _args, body) in selected.items()
        }
        definitions, bodies, by_record = [], {}, {}
        # head -> (parameters, lowered body part): exact bodies for ``inline``.
        self._bodies: dict[Any, tuple[list[Any], Any]] = {}
        # An identically zero helper diagnostic needs no opaque call. Record
        # this exact proof in dependency order so callers can retain structural
        # zeros without evaluating any model parameter or predicate numerically.
        self._zero_diagnostic_heads = set()
        for name in TopologicalSorter(dependencies).static_order():
            record_id, arguments, body = selected[name]
            if record_id in by_record:
                bodies[name] = by_record[record_id]
                continue
            parameters = [
                symbols.qualified(f"{self.namespace}::arg_{record_id}_{i}")
                for i in range(len(arguments))
            ]
            lowered = self.lower(
                body, locals=dict(zip(arguments, parameters, strict=True)), closed=True
            )
            value, diagnostic, _ = self.helpers[name]
            self._bodies[value] = (parameters, lowered.value)
            self._bodies[diagnostic] = (parameters, lowered.diagnostic)
            if lowered.diagnostic == self.zero:
                self._zero_diagnostic_heads.add(diagnostic)
            definitions.extend(
                (
                    FunctionDefinition(
                        value, parameters, lowered.value, inlining="never"
                    ),
                    FunctionDefinition(
                        diagnostic, parameters, lowered.diagnostic, inlining="never"
                    ),
                )
            )
            bodies[name] = by_record[record_id] = lowered
        self._zero_diagnostic_heads = frozenset(self._zero_diagnostic_heads)
        self.function_definitions = tuple(definitions)
        self.helper_bodies = MappingProxyType(bodies)
        self.helpers = MappingProxyType(self.helpers)

    def decided_number(self, expression: Any) -> Fraction | None:
        """The exact real value of ``expression`` when it is decided where it
        is formed: an exact rational number, or this library's predicates
        (``truth``, ``equal``, ``less``, ``less_equal``) and conditionals
        (``symbolica::if``) of such values; else None (e.g. a model symbol,
        a complex or floating-point number)."""

        from symbolica import AtomType

        kind = expression.get_type()
        if kind == AtomType.Num:
            try:
                return Fraction(expression.to_canonical_string())
            except (ValueError, ZeroDivisionError):
                return None
        if kind != AtomType.Fn:
            return None
        name, arguments = expression.get_name(), list(expression)
        if name == self._choose_name and len(arguments) == 3:
            condition = self.decided_number(arguments[0])
            if condition is None:
                return None
            return self.decided_number(arguments[1 if condition != 0 else 2])
        head = self._predicate_heads.get(name)
        if head is None:
            return None
        numbers = [self.decided_number(argument) for argument in arguments]
        if any(number is None for number in numbers):
            return None
        if head == "truth" and len(numbers) == 1:
            return Fraction(int(numbers[0] != 0))
        if head not in ("equal", "less", "less_equal") or len(numbers) != 2:
            return None
        left, right = numbers
        decision = {"equal": left == right, "less": left < right}.get(
            head, left <= right
        )
        return Fraction(int(decision))

    def inline(self, expression: Any, where: Any) -> Any:
        """Replace every helper call (value or diagnostic) whose arguments
        satisfy ``where(arguments)`` by its exact lowered body, recursively.

        The bodies keep their native conditionals (``symbolica::if``), so a
        caller can, e.g., collect a symbol that only reaches a helper through
        its arguments (formal-N_c grading of source couplings).
        """
        from symbolica import Replacement

        original = expression
        while True:
            before = expression
            for head, (parameters, body) in self._bodies.items():
                wildcards = [
                    symbols.qualified(f"pyamplicol_inline::x{i}_")
                    for i in range(len(parameters))
                ]

                def rewrite(
                    match,
                    head=head,
                    parameters=parameters,
                    body=body,
                    wildcards=wildcards,
                ):
                    arguments = [match[w] for w in wildcards]
                    if not where(arguments):
                        return head(*arguments)
                    return body.replace_multiple(
                        [
                            Replacement(parameter, argument)
                            for parameter, argument in zip(
                                parameters, arguments, strict=True
                            )
                        ]
                    )

                expression = expression.replace(
                    head(*wildcards), rewrite, rhs_cache_size=0
                )
            if expression == before:
                # Unchanged expressions keep their identity (and term order).
                return original if expression == original else before

    def lower(
        self, source: str, *, locals: dict[str, Any] | None = None, closed: bool = False
    ) -> ScalarExpression:
        """Lower supported scalar syntax, retaining Python and/or return values.

        Without ``locals`` the result is a pure function of ``source``: it is
        formed once per library (e.g. the dependency scan of every source
        declaration by ``prepare_scalar_bindings`` and the counterterm
        lowering's reads of the same sources)."""

        if locals is not None:
            return self._lower(source, locals=locals, closed=closed)
        key = (source, closed)
        found = self._lowered.get(key)
        if found is None:
            found = self._lowered[key] = self._lower(source, closed=closed)
        return found

    def _lower(
        self, source: str, *, locals: dict[str, Any] | None = None, closed: bool = False
    ) -> ScalarExpression:
        from symbolica import E

        lines = source.splitlines()
        numeric_tokens = {
            (
                token.start[0],
                len(lines[token.start[0] - 1][: token.start[1]].encode()),
            ): token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.NUMBER
        }
        local = locals or {}
        zero, one, imaginary = self.zero, self.one, self.imaginary

        def merge(parts):
            return sum((part.diagnostic for part in parts), zero)

        def truth(part):
            condition = self.predicates["truth"](part.value)
            return condition, part.diagnostic + condition

        def lower(node):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, bool):
                    return ScalarExpression(one if node.value else zero, zero)
                if not isinstance(node.value, (int, float, complex)):
                    raise NotImplementedError("scalar constants must be numeric")
                token = numeric_tokens[(node.lineno, node.col_offset)]
                is_imaginary = token.lower().endswith("j")
                number = token[:-1] if is_imaginary else token
                rational = (
                    Fraction(int(number, 0))
                    if number.lower().startswith(("0x", "0o", "0b"))
                    else Fraction(number)
                )
                value = E(f"{rational.numerator}/{rational.denominator}")
                return ScalarExpression(
                    value * imaginary if is_imaginary else value, zero
                )
            if isinstance(node, ast.Name):
                if node.id in local:
                    return (
                        local[node.id]
                        if isinstance(local[node.id], ScalarExpression)
                        else ScalarExpression(local[node.id], zero)
                    )
                if closed:
                    raise ValueError(f"unsupported free helper symbol {node.id!r}")
                if node.id in {"math", "cmath"}:
                    raise NotImplementedError("math modules are not scalar values")
                return ScalarExpression(symbols.qualified(f"UFO::{node.id}"), zero)
            if isinstance(node, ast.UnaryOp):
                part = lower(node.operand)
                if isinstance(node.op, ast.Not):
                    condition, diagnostic = truth(part)
                    return ScalarExpression(
                        self.choose(condition, zero, one), diagnostic
                    )
                if isinstance(node.op, (ast.UAdd, ast.USub)):
                    return ScalarExpression(
                        part.value if isinstance(node.op, ast.UAdd) else -part.value,
                        part.diagnostic,
                    )
            if isinstance(node, ast.BinOp):
                left, right = lower(node.left), lower(node.right)
                operations = {
                    ast.Add: lambda: left.value + right.value,
                    ast.Sub: lambda: left.value - right.value,
                    ast.Mult: lambda: left.value * right.value,
                    ast.Div: lambda: left.value / right.value,
                    ast.Pow: lambda: left.value**right.value,
                }
                operation = operations.get(type(node.op))
                if operation is not None:
                    return ScalarExpression(
                        operation(), left.diagnostic + right.diagnostic
                    )
            if isinstance(node, ast.IfExp):
                condition, diagnostic = truth(lower(node.test))
                decided = self.decided_number(condition)
                if decided is not None:
                    # Python's conditional is lazy: a condition the bindings
                    # (fixed parameters) decide forms its taken branch only,
                    # so a singularity of the other branch at the fixed
                    # values (the 1/MB^2 of ``0 if MB == 0 else ...`` at
                    # MB = 0) never forms an infinite coefficient.
                    taken = lower(node.body if decided != 0 else node.orelse)
                    return ScalarExpression(taken.value, diagnostic + taken.diagnostic)
                yes, no = lower(node.body), lower(node.orelse)
                return ScalarExpression(
                    self.choose(condition, yes.value, no.value),
                    diagnostic + self.choose(condition, yes.diagnostic, no.diagnostic),
                )
            if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                result = lower(node.values[-1])
                for operand in reversed(node.values[:-1]):
                    part = lower(operand)
                    condition, diagnostic = truth(part)
                    yes, no = (
                        (result, part)
                        if isinstance(node.op, ast.And)
                        else (part, result)
                    )
                    branch_diag = (
                        self.choose(condition, result.diagnostic, zero)
                        if isinstance(node.op, ast.And)
                        else self.choose(condition, zero, result.diagnostic)
                    )
                    result = ScalarExpression(
                        self.choose(condition, yes.value, no.value),
                        diagnostic + branch_diag,
                    )
                return result
            if isinstance(node, ast.Compare):
                operands = [lower(node.left), *(lower(x) for x in node.comparators)]
                result = ScalarExpression(one, zero)
                for index in reversed(range(len(node.ops))):
                    left, right, operator = (
                        operands[index],
                        operands[index + 1],
                        node.ops[index],
                    )
                    if isinstance(operator, (ast.Eq, ast.NotEq)):
                        predicate = self.predicates["equal"](left.value, right.value)
                        value = (
                            self.choose(predicate, zero, one)
                            if isinstance(operator, ast.NotEq)
                            else predicate
                        )
                    elif isinstance(operator, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                        a, b = (
                            (right.value, left.value)
                            if isinstance(operator, (ast.Gt, ast.GtE))
                            else (left.value, right.value)
                        )
                        predicate = self.predicates[
                            "less_equal"
                            if isinstance(operator, (ast.LtE, ast.GtE))
                            else "less"
                        ](a, b)
                        value = predicate
                    else:
                        raise NotImplementedError("unsupported scalar comparison")
                    result = ScalarExpression(
                        self.choose(value, result.value, zero),
                        left.diagnostic
                        + right.diagnostic
                        + predicate
                        + self.choose(value, result.diagnostic, zero),
                    )
                return result
            if isinstance(node, ast.Attribute):
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id in {"math", "cmath"}
                    and node.attr == "pi"
                ):
                    return ScalarExpression(E("pi"), zero)
                part = lower(node.value)
                if node.attr == "real":
                    return ScalarExpression(
                        self.predicates["real_part"](part.value), part.diagnostic
                    )
                if node.attr == "imag":
                    return ScalarExpression(
                        self.predicates["imaginary_part"](part.value),
                        part.diagnostic,
                    )
            if isinstance(node, ast.Call) and not node.keywords:
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "conjugate"
                    and not node.args
                ):
                    part = lower(node.func.value)
                    return ScalarExpression(part.value.conj(), part.diagnostic)
                qualified = (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in {"math", "cmath"}
                )
                name = (
                    node.func.attr
                    if qualified
                    else node.func.id
                    if isinstance(node.func, ast.Name)
                    else None
                )
                if not qualified and name in local:
                    raise NotImplementedError(
                        "calling a local helper argument is unsupported"
                    )
                parts = [lower(arg) for arg in node.args]
                values = [part.value for part in parts]
                diagnostic = merge(parts)
                if not qualified and name in self.helpers:
                    value_head, diagnostic_head, arguments = self.helpers[name]
                    if len(parts) != len(arguments):
                        raise ValueError(f"helper {name!r} argument count differs")
                    return ScalarExpression(
                        value_head(*values),
                        diagnostic
                        + (
                            zero
                            if diagnostic_head in self._zero_diagnostic_heads
                            else diagnostic_head(*values)
                        ),
                    )
                if not qualified and name == "complex" and len(parts) in {1, 2}:
                    return ScalarExpression(
                        values[0]
                        + (imaginary * values[1] if len(parts) == 2 else zero),
                        diagnostic,
                    )
                standard = {
                    "sqrt",
                    "log",
                    "exp",
                    "sin",
                    "cos",
                    "tan",
                    "asin",
                    "acos",
                    "atan",
                    "sinh",
                    "cosh",
                    "tanh",
                    "asinh",
                    "acosh",
                    "atanh",
                }
                if len(parts) == 1 and (
                    (qualified and name in standard)
                    or (not qualified and name == "abs")
                ):
                    return ScalarExpression(getattr(values[0], name)(), diagnostic)
                raise NotImplementedError(f"unsupported scalar function {name!r}")
            raise NotImplementedError(
                f"unsupported UFO scalar syntax: {type(node).__name__}"
            )

        return lower(ast.parse(source, mode="eval").body)

    def lower_value(self, value, *, bindings=None) -> ScalarExpression:
        """Lower an exact raw scalar/Laurent declaration with scoped bindings."""
        from .model_expressions import epsilon_symbol

        if value["kind"] == "expression":
            return self.lower(value["expression"], locals=bindings)
        if value["kind"] != "laurent":
            raise ValueError("unknown raw UFO scalar value representation")
        terms = [
            (power, self.lower(text, locals=bindings)) for power, text in value["terms"]
        ]
        return ScalarExpression(
            sum(
                (part.value * epsilon_symbol() ** power for power, part in terms),
                self.zero,
            ),
            sum((part.diagnostic for _, part in terms), self.zero),
        )

    def evaluator(
        self, expressions: tuple[ScalarExpression, ...], parameters: tuple[Any, ...]
    ):
        """Native saved program outputs interleaved (value, diagnostic) pairs."""
        from symbolica import Expression

        outputs = [
            part
            for expression in expressions
            for part in (expression.value, expression.diagnostic)
        ]
        return Expression.evaluator_multiple(
            outputs,
            list(parameters),
            functions=list(self.function_definitions),
            jit_compile=False,
            n_cores=1,
        )


@dataclass(frozen=True, slots=True)
class SourceScalarDefinition:
    """Dependency-owned Laurent value with its mandatory active diagnostic."""

    collection: str
    id: int
    name: str
    value: Any
    diagnostic: Any

    def coefficient(self, power: int) -> ScalarExpression:
        return ScalarExpression(self.value.coefficient(power), self.diagnostic)


@dataclass(frozen=True, slots=True)
class ScalarProgram:
    library: ScalarFunctionLibrary
    definitions: Any


def require_laurent_scalar(expression: ScalarExpression) -> None:
    """Reject epsilon hidden in native opaque functions or active diagnostics."""
    from symbolica import AtomType

    from .model_expressions import epsilon_symbol

    epsilon = epsilon_symbol()
    pending = [expression.value, expression.diagnostic]
    while pending:
        atom = pending.pop()
        if atom.get_type() == AtomType.Fn and epsilon in atom.get_all_symbols():
            raise NotImplementedError(
                "epsilon-dependent scalar helper or function arguments "
                "require expansion"
            )
        if atom.get_type() not in {AtomType.Num, AtomType.Var}:
            pending.extend(atom)
    if epsilon in expression.diagnostic.get_all_symbols():
        raise NotImplementedError("epsilon-dependent predicate diagnostic")


def require_scalar_value(expression: ScalarExpression, context: str, program=None):
    """Value-only access cannot discard diagnostics or an owning helper library."""
    from symbolica import AtomType

    if expression.diagnostic != 0:
        raise NotImplementedError(
            f"{context} requires paired scalar value and diagnostic"
        )
    if program is not None:
        helpers = {
            head.get_name()
            for value, diagnostic, _ in program.library.helpers.values()
            for head in (value, diagnostic)
        }
        pending = [expression.value]
        while pending:
            atom = pending.pop()
            if atom.get_type() == AtomType.Fn and atom.get_name() in helpers:
                raise NotImplementedError(
                    f"{context} requires the owning scalar helper library"
                )
            if atom.get_type() not in {AtomType.Num, AtomType.Var}:
                pending.extend(atom)
    return expression.value


def lower_scalar_program(raw: Any) -> ScalarProgram:
    """Lower exact declarations and CT dependencies without discarding diagnostics.

    Epsilon-dependent helper/conditional arguments are rejected explicitly.
    Supporting them requires dependency-owned helper expansion before series.
    Ordinary parameter definitions remain separate for exact binding by a caller.
    This preparation function does not enable an EW process factory.
    """

    from .model_expressions import LaurentExpression, epsilon_symbol

    library = ScalarFunctionLibrary(raw)
    epsilon = epsilon_symbol()
    declarations, by_symbol, ct_symbols = {}, {}, set()

    for collection in ("parameters", "counterterm_parameters", "couplings"):
        for record in raw.collections[collection]:
            symbol = symbols.qualified(f"UFO::{record['name']}")
            if symbol in by_symbol:
                raise ValueError(f"ambiguous UFO scalar name {record['name']!r}")
            key = (collection, record["id"])
            source = raw.exact_value(*key)
            lowered = library.lower_value(source)
            declarations[key] = (record, source, lowered)
            by_symbol[symbol] = key
            if (
                collection == "counterterm_parameters"
                or epsilon in lowered.value.get_all_symbols()
            ):
                ct_symbols.add(symbol)
    dependencies = {
        symbol: (
            set(declarations[key][2].value.get_all_symbols())
            | set(declarations[key][2].diagnostic.get_all_symbols())
        )
        & by_symbol.keys()
        for symbol, key in by_symbol.items()
    }
    resolved, ct_dependent, definitions = {}, set(ct_symbols), {}
    for symbol in TopologicalSorter(dependencies).static_order():
        key = by_symbol[symbol]
        record, source, original = declarations[key]
        bindings = {
            declarations[by_symbol[dep]][0]["name"]: resolved[dep]
            for dep in dependencies[symbol]
            if dep in ct_dependent
        }
        expression = (
            library.lower_value(source, bindings=bindings) if bindings else original
        )
        if bindings:
            ct_dependent.add(symbol)
        require_laurent_scalar(expression)
        resolved[symbol] = expression
        definitions[key] = SourceScalarDefinition(
            key[0],
            key[1],
            record["name"],
            LaurentExpression(expression.value),
            expression.diagnostic,
        )
    return ScalarProgram(library, MappingProxyType(definitions))
