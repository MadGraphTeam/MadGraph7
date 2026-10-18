# SPDX-License-Identifier: 0BSD
"""Unlowered NLO UFO data captured before tree-model normalization.

This is an import boundary, not a loop evaluator or a completeness certificate.
In particular, a scalar expression may reference Laurent-valued CT parameters;
it must not be interpreted as an epsilon-zero coefficient before substitution.
Expressions remain UFO strings. No series algebra or numerical evaluation is
performed here.

Direct literal value arguments and compositions of certified source strings
retain their source lexemes through a scoped constructor observer. Numeric
computations or dynamically reassigned values remain uncertified. No general
Python evaluation or expression algebra is implemented
here, and source preservation does not certify arbitrary-precision physics.
"""

from __future__ import annotations

import ast
import importlib
import math
import sys
import tokenize
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from types import FrameType, MappingProxyType, ModuleType
from typing import Any

RAW_LOOP_MODEL_KIND = "pyamplicol-raw-loop-model"
RAW_LOOP_MODEL_SCHEMA_VERSION = 1

_COLLECTIONS = {
    "parameters": "all_parameters",
    "counterterm_parameters": "all_CTparameters",
    "particles": "all_particles",
    "couplings": "all_couplings",
    "vertices": "all_vertices",
    "counterterm_vertices": "all_CTvertices",
    "lorentz_structures": "all_lorentz",
    "coupling_orders": "all_orders",
    "functions": "all_functions",
    "propagators": "all_propagators",
    "form_factors": "all_form_factors",
}

_VALUE_COLLECTIONS = {"parameters", "counterterm_parameters", "couplings"}
_VALUE_TYPES = {"Parameter", "CTParameter", "Coupling", "Function"}


def _runtime_exact(value: object) -> bool:
    """Exact input types, without treating a stored binary64 as decimal input."""
    if isinstance(value, Mapping):
        return all(_runtime_exact(item) for item in value.values())
    return isinstance(value, (str, int, Decimal, Fraction)) and not isinstance(
        value, bool
    )


def _literal_expression(
    node: ast.expr,
    text: str,
    runtime: object,
    string_reference: Callable[[ast.Subscript], str] | None = None,
) -> dict[str, object]:
    """Retain literal spelling; literal_eval is only a runtime-binding check."""
    if isinstance(node, ast.Dict):
        if not isinstance(runtime, Mapping) or any(key is None for key in node.keys):
            raise ValueError("value is not a direct Laurent dictionary literal")
        terms = []
        seen = set()
        for key, coefficient in zip(node.keys, node.values, strict=True):
            power = ast.literal_eval(key)
            if isinstance(power, bool) or not isinstance(power, int) or power in seen:
                raise ValueError("Laurent powers must be distinct integer literals")
            seen.add(power)
            if power not in runtime:
                raise ValueError("Laurent dictionary changed during construction")
            item = _literal_expression(
                coefficient, text, runtime[power], string_reference
            )
            if item["kind"] != "expression":
                raise ValueError("nested Laurent values are unsupported")
            terms.append([power, item["expression"]])
        if seen != set(runtime):
            raise ValueError("Laurent dictionary changed during construction")
        return {"kind": "laurent", "terms": terms}
    if string_reference is not None and isinstance(node, (ast.BinOp, ast.Subscript)):

        def source_string(part: ast.expr) -> str:
            if isinstance(part, ast.Constant) and type(part.value) is str:
                return part.value
            if isinstance(part, ast.BinOp) and isinstance(part.op, ast.Add):
                return source_string(part.left) + source_string(part.right)
            if isinstance(part, ast.Subscript):
                return string_reference(part)
            raise ValueError("computed value is not a certified string composition")

        expression = source_string(node)
        if type(runtime) is not str or expression != runtime:
            raise ValueError("source string does not match the constructed value")
        return {"kind": "expression", "expression": expression}
    literal = node
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        literal = node.operand
    if not isinstance(literal, ast.Constant) or isinstance(literal.value, bool):
        raise ValueError("computed value argument is not a direct literal")
    evaluated = ast.literal_eval(node)
    if type(evaluated) is not type(runtime) or evaluated != runtime:
        raise ValueError("value argument does not match the constructed value")
    if isinstance(evaluated, str):
        expression = evaluated
    elif isinstance(evaluated, int):
        expression = str(evaluated)
    elif isinstance(evaluated, (float, complex)):
        expression = ast.get_source_segment(text, node)
        if expression is None:
            raise ValueError("value argument source is unavailable")
        # Python imaginary literals are UFO complex calls, with the original
        # imaginary decimal retained rather than its rounded Python component.
        if isinstance(evaluated, complex):
            expression = f"complex(0,{expression[:-1]})"
    else:
        raise ValueError("unsupported literal value argument")
    return {"kind": "expression", "expression": expression}


@dataclass(slots=True)
class _ValueObservation:
    instance: object
    namespace: Mapping[str, object]
    binding: str | None
    argument: object
    terms: tuple[tuple[object, object], ...] | None
    expression: dict[str, object] | None
    source: dict[str, object]
    error: str | None
    dependencies: tuple[object, ...]
    attribute: str


def _has_value_write(statement: ast.AST) -> bool:
    for node in ast.walk(statement):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in {"value", "expression", "expr"}
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            return True
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and any(
                isinstance(part, ast.Attribute)
                and part.attr in {"value", "expression", "expr", "__dict__"}
                for part in ast.walk(node)
            )
        ):
            return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {
                "exec",
                "eval",
                "setattr",
                "delattr",
            }:
                return True
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr in {"value", "expression", "expr"}
            ):
                return True
    return False


class _ValueSourceObserver:
    """Observe direct UFO constructors without replacing objects or their values."""

    def __init__(self, source: Path):
        self.source = source
        self.files: dict[
            str, tuple[list[str], dict[int, list[ast.Assign | ast.AnnAssign]]]
        ] = {}
        self.observations: dict[int, _ValueObservation] = {}
        self.dynamic_writes = False
        self.mutating_methods: set[tuple[str, int]] = set()
        for path in source.glob("*.py"):
            try:
                with tokenize.open(path) as stream:
                    text = stream.read()
                tree = ast.parse(text, filename=str(path))
            except (SyntaxError, UnicodeError, OSError):
                # Optional, unimported legacy modules must not break tree loading.
                continue
            declarations: dict[int, list[ast.Assign | ast.AnnAssign]] = {}
            for node in tree.body:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    for line in range(
                        node.lineno, (node.end_lineno or node.lineno) + 1
                    ):
                        declarations.setdefault(line, []).append(node)
            self.files[str(path)] = (text.splitlines(keepends=True), declarations)
            # A conservative boundary, not alias analysis or a Python interpreter.
            # Any potential value mutation outside class implementations prevents
            # source certification, even if a branch was not executed. Constructor
            # changes are checked separately against the observed argument identity.
            for statement in tree.body:
                if isinstance(statement, ast.ClassDef):
                    for method in ast.walk(statement):
                        if (
                            isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and method.name != "__init__"
                            and _has_value_write(method)
                        ):
                            first_line = min(
                                [
                                    method.lineno,
                                    *(
                                        decorator.lineno
                                        for decorator in method.decorator_list
                                    ),
                                ]
                            )
                            self.mutating_methods.add((str(path), first_line))
                    if any(
                        _has_value_write(body)
                        for body in statement.body
                        if not isinstance(
                            body, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                        )
                    ):
                        self.mutating_methods.add((str(path), statement.lineno))
                    continue
                if _has_value_write(statement):
                    self.dynamic_writes = True

    def observe(self, frame: FrameType, event: str, _argument: object) -> None:
        if event != "call":
            return
        if (
            frame.f_code.co_filename,
            frame.f_code.co_firstlineno,
        ) in self.mutating_methods:
            self.dynamic_writes = True
        if frame.f_code.co_name != "__init__":
            return
        instance = frame.f_locals.get("self")
        if type(instance).__name__ not in _VALUE_TYPES:
            return
        if getattr(type(instance).__init__, "__code__", None) is not frame.f_code:
            return
        caller = frame.f_back
        if caller is None or caller.f_code.co_filename not in self.files:
            return
        attribute = "expression" if type(instance).__name__ == "Function" else "value"
        value = frame.f_locals.get(attribute)
        lines, declarations = self.files[caller.f_code.co_filename]
        candidates = declarations.get(caller.f_lineno, [])
        binding = None
        expression = None
        source: dict[str, object] = {
            "file": Path(caller.f_code.co_filename).name,
            "line": caller.f_lineno,
        }
        error = None
        dependencies: list[object] = []
        try:
            if caller.f_code.co_name != "<module>" or len(candidates) != 1:
                raise ValueError("constructor is not a direct module declaration")
            declaration = candidates[0]
            targets = (
                declaration.targets
                if isinstance(declaration, ast.Assign)
                else [declaration.target]
            )
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                raise ValueError("constructor has no unique module binding")
            binding = targets[0].id
            call = declaration.value
            if (
                not isinstance(call, ast.Call)
                or not isinstance(call.func, ast.Name)
                or caller.f_globals.get(call.func.id) is not type(instance)
            ):
                raise ValueError("constructor identity is not statically bound")
            values = [
                keyword.value for keyword in call.keywords if keyword.arg == attribute
            ]
            if len(values) != 1 or any(
                keyword.arg is None for keyword in call.keywords
            ):
                raise ValueError("constructor requires one direct value keyword")
            node = values[0]
            # Restrict get_source_segment to the argument's lines. Splitting a
            # multi-megabyte coupling file for every coefficient is quadratic.
            location = ast.copy_location(ast.Constant(value=None), node)
            location.lineno = 1
            location.end_lineno = (node.end_lineno or node.lineno) - node.lineno + 1
            text = "".join(lines[node.lineno - 1 : node.end_lineno])
            argument = ast.get_source_segment(text, location)
            if argument is None:
                raise ValueError("value argument source is unavailable")
            source["argument"] = argument

            def string_reference(part: ast.Subscript) -> str:
                if not (
                    isinstance(part.value, ast.Attribute)
                    and part.value.attr == "value"
                    and isinstance(part.value.value, ast.Name)
                ):
                    raise ValueError("source string requires Name.value[integer]")
                power = ast.literal_eval(part.slice)
                if type(power) is not int:
                    raise ValueError("source string requires an integer Laurent power")
                referenced = caller.f_globals.get(part.value.value.id)
                exact, evidence = self.value(referenced)
                if exact is None or exact["kind"] != "laurent":
                    raise ValueError(
                        "referenced string declaration is not certified: "
                        + str(evidence.get("reason", "requires a Laurent dictionary"))
                    )
                terms = dict(exact["terms"])
                stored = getattr(referenced, "value", None)
                if (
                    power not in terms
                    or not isinstance(stored, Mapping)
                    or type(stored.get(power)) is not str
                    or stored[power] != terms[power]
                ):
                    raise ValueError(
                        "referenced Laurent coefficient is not a source string"
                    )
                dependencies.append(referenced)
                return terms[power]

            parsed = ast.parse(argument, mode="eval").body
            expression = _literal_expression(parsed, argument, value, string_reference)
            if any(
                isinstance(part, (ast.BinOp, ast.Subscript))
                for part in ast.walk(parsed)
            ):
                source["kind"] = "source-string-composition"
        except (TypeError, ValueError) as exception:
            error = str(exception)
        self.observations[id(instance)] = _ValueObservation(
            instance,
            caller.f_globals,
            binding,
            value,
            tuple(value.items()) if isinstance(value, Mapping) else None,
            expression,
            source,
            error,
            tuple(dependencies),
            "expr" if type(instance).__name__ == "Function" else attribute,
        )

    def value(
        self, instance: object
    ) -> tuple[dict[str, object] | None, dict[str, object]]:
        observation = self.observations.get(id(instance))
        if observation is None:
            return None, {
                "kind": "unsupported",
                "exact": False,
                "reason": "no observed direct constructor",
            }
        reason = observation.error
        runtime = getattr(instance, observation.attribute, None)
        if self.dynamic_writes:
            reason = "model contains potentially dynamic value writes"
        elif reason is None and (
            observation.binding is None
            or observation.namespace.get(observation.binding) is not instance
        ):
            reason = "declaration no longer binds the imported object"
        elif reason is None and (
            runtime is not observation.argument
            or (
                observation.terms is not None
                and (
                    not isinstance(runtime, Mapping)
                    or len(runtime) != len(observation.terms)
                    or any(
                        key not in runtime or runtime[key] is not item
                        for key, item in observation.terms
                    )
                )
            )
        ):
            reason = "constructed value was changed or overridden"
        if reason is None:
            for dependency in observation.dependencies:
                exact, evidence = self.value(dependency)
                if exact is None:
                    reason = "referenced declaration is no longer exact: " + str(
                        evidence.get("reason", "unknown source")
                    )
                    break
        if reason is not None:
            return None, {
                **observation.source,
                "kind": "unsupported",
                "exact": False,
                "reason": reason,
            }
        return observation.expression, {
            "kind": "source-literal",
            "exact": True,
            **observation.source,
        }


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RawLoopModel:
    """Immutable source records, independent of the restricted tree-model IR.

    Every record has ``id``, ``name``, ``ufo_type`` and ``fields``. Named UFO objects in
    fields become typed references. Maps with non-string keys (including CT
    coupling index tuples) are represented by ordered key/value entries. UFO
    expression strings and all explicit Laurent powers are retained verbatim.
    Value records include ``value_source`` status. Direct constructor literals
    use their original numeric spelling; unsupported values retain their runtime
    representation and must not be used as exact inputs. ``exact_value`` enforces
    that boundary for subsequent loop lowering.
    """

    name: str
    collections: Mapping[str, tuple[Mapping[str, object], ...]]
    metadata: Mapping[str, object]
    capabilities: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.name or set(self.collections) != set(_COLLECTIONS):
            raise ValueError(
                "raw loop model requires a name and all source collections"
            )
        for collection, records in self.collections.items():
            for index, record in enumerate(records):
                if set(record) != {"id", "name", "ufo_type", "fields"}:
                    raise ValueError(f"invalid raw UFO record in {collection}")
                name = record["name"]
                if not isinstance(name, str) or not name or record["id"] != index:
                    raise ValueError(f"invalid raw UFO name or ID in {collection}")
                if not isinstance(record["ufo_type"], str) or not isinstance(
                    record["fields"], Mapping
                ):
                    raise ValueError(f"invalid raw UFO fields for {collection}.{name}")
        object.__setattr__(self, "collections", _freeze(self.collections))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        object.__setattr__(self, "capabilities", _freeze(self.capabilities))

    def record(self, collection: str, name: str) -> Mapping[str, object]:
        matches = [
            record for record in self.collections[collection] if record["name"] == name
        ]
        if not matches:
            raise KeyError(f"unknown raw UFO record {collection}.{name}")
        if len(matches) != 1:
            raise ValueError(
                f"ambiguous raw UFO name {collection}.{name}; use its record ID"
            )
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": RAW_LOOP_MODEL_KIND,
            "schema_version": RAW_LOOP_MODEL_SCHEMA_VERSION,
            "name": self.name,
            "collections": _plain(self.collections),
            "metadata": _plain(self.metadata),
            "capabilities": _plain(self.capabilities),
        }

    def exact_value(self, collection: str, record_id: int) -> Mapping[str, object]:
        """Return an input expression only when its exact origin was established."""
        if collection not in _VALUE_COLLECTIONS:
            raise ValueError(f"{collection} does not contain UFO value expressions")
        if (
            isinstance(record_id, bool)
            or not isinstance(record_id, int)
            or not 0 <= record_id < len(self.collections[collection])
        ):
            raise ValueError(f"invalid raw UFO record ID: {record_id!r}")
        record = self.collections[collection][record_id]
        fields = record["fields"]
        source = fields.get("value_source", {})
        if not source.get("exact", False):
            raise ValueError(
                f"exact input unavailable for {collection}.{record['name']}: "
                f"{source.get('reason', 'missing value source')}"
            )
        return fields["value"]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RawLoopModel:
        if (
            payload.get("kind") != RAW_LOOP_MODEL_KIND
            or payload.get("schema_version") != RAW_LOOP_MODEL_SCHEMA_VERSION
        ):
            raise ValueError("unsupported raw loop model schema; regenerate the model")
        if set(payload) != {
            "kind",
            "schema_version",
            "name",
            "collections",
            "metadata",
            "capabilities",
        }:
            raise ValueError("invalid raw loop model fields")
        for field in ("collections", "metadata", "capabilities"):
            if not isinstance(payload[field], Mapping):
                raise ValueError(f"raw loop model {field} must be an object")
        if not isinstance(payload["name"], str):
            raise ValueError("raw loop model name must be a string")
        return cls(
            payload["name"],
            payload["collections"],
            payload["metadata"],
            payload["capabilities"],
        )


def _expression(value: object, context: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an expression or number, not a boolean")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    if (
        isinstance(value, complex)
        and math.isfinite(value.real)
        and math.isfinite(value.imag)
    ):
        return f"complex({value.real!r},{value.imag!r})"
    if isinstance(value, Decimal) and value.is_finite():
        return str(value)
    if isinstance(value, Fraction):
        return f"({value.numerator}/{value.denominator})"
    raise ValueError(
        f"unsupported UFO expression value in {context}: {type(value).__name__}"
    )


def _expression_value(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {"kind": "expression", "expression": _expression(value, context)}
    terms = []
    for power, coefficient in value.items():
        if isinstance(power, bool) or not isinstance(power, int):
            raise ValueError(f"{context} has a non-integer epsilon power: {power!r}")
        terms.append([power, _expression(coefficient, context)])
    return {"kind": "laurent", "terms": terms}


def _encode(
    value: object, references: Mapping[int, tuple[str, int, str]], context: str
) -> object:
    reference = references.get(id(value))
    if reference is not None:
        return {
            "kind": "reference",
            "collection": reference[0],
            "id": reference[1],
            "name": reference[2],
        }
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (complex, Decimal, Fraction)):
        return {"kind": "number", "expression": _expression(value, context)}
    if isinstance(value, type):
        return {"kind": "type", "name": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, Mapping):
        # Use entries even for string keys: a UFO mapping cannot then collide
        # with one of this schema's tagged references or expression records.
        return {
            "kind": "mapping",
            "entries": [
                [_encode(key, references, context), _encode(item, references, context)]
                for key, item in value.items()
            ],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_encode(item, references, context) for item in value],
        }
    if isinstance(value, list):
        return [_encode(item, references, context) for item in value]
    raise ValueError(f"unsupported raw UFO value in {context}: {type(value).__name__}")


def _objects(module: ModuleType, registry: str) -> list[Any]:
    values = getattr(module, registry, None)
    if values is None:
        values = getattr(getattr(module, "object_library", None), registry, ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"UFO registry {registry} must be a sequence")
    result: list[Any] = []
    seen: set[int] = set()
    for value in values:
        name = getattr(value, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError(f"UFO registry {registry} contains an unnamed object")
        if id(value) in seen:
            continue
        seen.add(id(value))
        result.append(value)
    return result


def capture_raw_loop_model(
    module: ModuleType,
    *,
    name: str | None = None,
    restriction: Mapping[str, object] | None = None,
    value_overrides: Mapping[tuple[str, int], object] | None = None,
) -> RawLoopModel:
    """Snapshot an imported UFO before any tree normalization or restriction.

    ``value_overrides`` targets ``(collection, record_id)`` and accepts exact
    strings, integers, Decimal and Fraction values, including Laurent maps.
    Overrides change only the raw sidecar, never the imported tree objects.
    String literals may be concatenated with previously certified
    Name.value[integer] strings. Other computed values remain uncertified unless
    the caller supplies an explicit exact override.
    """
    objects = {
        key: _objects(module, registry) for key, registry in _COLLECTIONS.items()
    }
    # Some UFOs register CTVertex objects in both registries, or only all_vertices.
    ct_vertices = objects["counterterm_vertices"]
    ct_ids = {id(vertex) for vertex in ct_vertices}
    for vertex in objects["vertices"]:
        if type(vertex).__name__ == "CTVertex" and id(vertex) not in ct_ids:
            ct_vertices.append(vertex)
            ct_ids.add(id(vertex))
    objects["vertices"] = [v for v in objects["vertices"] if id(v) not in ct_ids]
    ct_parameter_ids = {
        id(parameter) for parameter in objects["counterterm_parameters"]
    }
    objects["parameters"] = [
        p for p in objects["parameters"] if id(p) not in ct_parameter_ids
    ]
    # A UFO may attach a custom propagator without exporting all_propagators.
    propagator_ids = {id(prop) for prop in objects["propagators"]}
    for particle in objects["particles"]:
        prop = getattr(particle, "propagator", None)
        if (
            prop is not None
            and not isinstance(prop, str)
            and id(prop) not in propagator_ids
        ):
            objects["propagators"].append(prop)
            propagator_ids.add(id(prop))
    references = {
        id(value): (collection, index, str(value.name))
        for collection, values in objects.items()
        for index, value in enumerate(values)
    }
    overrides = dict(value_overrides or {})
    for key, override in overrides.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or key[0] not in _VALUE_COLLECTIONS
            or isinstance(key[1], bool)
            or not isinstance(key[1], int)
            or not 0 <= key[1] < len(objects[key[0]])
        ):
            raise ValueError(f"invalid raw UFO value override target: {key!r}")
        if not _runtime_exact(override):
            raise ValueError(
                f"value override {key!r} must use exact strings, integers, "
                "Decimal or Fraction"
            )
    observer = getattr(module, "_pyamplicol_value_sources", None)
    exact_inputs = True
    collections: dict[str, tuple[Mapping[str, object], ...]] = {}
    for collection, values in objects.items():
        records = []
        for index, value in enumerate(values):
            context = f"{collection}.{value.name}"
            fields = {
                key: _encode(item, references, f"{context}.{key}")
                for key, item in vars(value).items()
                if key != "value" or collection not in _VALUE_COLLECTIONS
            }
            if collection == "functions":
                if observer is not None:
                    _body, fields["expression_source"] = observer.value(value)
                else:
                    fields["expression_source"] = {
                        "kind": "unsupported",
                        "exact": False,
                        "reason": "helper source declaration was not observed",
                    }
            if collection in _VALUE_COLLECTIONS and hasattr(value, "value"):
                exact = None
                if (collection, index) in overrides:
                    exact = _expression_value(
                        overrides[(collection, index)], f"{context}.value override"
                    )
                    source = {"kind": "exact-override", "exact": True}
                elif observer is not None:
                    exact, source = observer.value(value)
                elif _runtime_exact(value.value):
                    source = {"kind": "runtime-exact", "exact": True}
                else:
                    source = {
                        "kind": "unsupported",
                        "exact": False,
                        "reason": "numeric source declaration was not observed",
                    }
                fields["value_source"] = source
                fields["value"] = (
                    exact
                    if exact is not None
                    else _expression_value(value.value, f"{context}.value")
                )
                exact_inputs = exact_inputs and bool(source["exact"])
            records.append(
                {
                    "id": index,
                    "name": value.name,
                    "ufo_type": type(value).__name__,
                    "fields": fields,
                }
            )
        collections[collection] = tuple(records)

    # Preserve explicitly declared conventions; never infer a scheme from a
    # model name, a parameter named lhv, or the presence of gamma-five tensors.
    metadata = {
        key: _encode(value, references, f"model.{key}")
        for key, value in vars(module).items()
        if not key.startswith("_")
        and not key.startswith("all_")
        and not isinstance(value, ModuleType)
        and not callable(value)
        and type(value).__module__ != "__future__"
    }
    function_namespace = getattr(module, "function_library", None)
    metadata["function_bindings"] = {
        name: references[id(value)][1]
        for name, value in (
            vars(function_namespace).items()
            if isinstance(function_namespace, ModuleType)
            else ()
        )
        if id(value) in references and references[id(value)][0] == "functions"
    }
    metadata["function_binding_scope"] = (
        "function_library" if isinstance(function_namespace, ModuleType) else None
    )
    metadata["restriction"] = dict(
        restriction or {"applied": False, "name": None, "card": None}
    )
    particles = objects["particles"]
    ct_types = sorted(
        {str(getattr(vertex, "type", "unknown")) for vertex in ct_vertices}
    )
    capabilities: dict[str, object] = {
        "stage": "raw-import",
        "counterterm_vertex_types": ct_types,
        "has_uv_vertices": any(kind.startswith("UV") for kind in ct_types),
        "has_r2_vertices": "R2" in ct_types,
        "has_laurent_parameters": any(
            isinstance(getattr(item, "value", None), Mapping)
            for key in ("parameters", "counterterm_parameters")
            for item in objects[key]
        ),
        "has_laurent_couplings": any(
            isinstance(getattr(item, "value", None), Mapping)
            for item in objects["couplings"]
        ),
        "ghost_particles": [
            str(p.name) for p in particles if getattr(p, "spin", None) == -1
        ],
        "goldstone_particles": [
            str(p.name)
            for p in particles
            if bool(getattr(p, "GoldstoneBoson", getattr(p, "goldstoneboson", False)))
        ],
        "perturbative_orders": [
            str(order.name)
            for order in objects["coupling_orders"]
            if getattr(order, "perturbative_expansion", 0) > 0
        ],
        "uv_complete": None,
        "model_r2_complete": None,
        "dimensional_kernels_validated": False,
        "complex_mass_kernels_validated": False,
        "loop_runtime_supported": False,
        "expression_strings_preserved": True,
        "numeric_literal_source_preserved": exact_inputs,
        "arbitrary_precision_inputs_validated": False,
    }
    return RawLoopModel(name or module.__name__, collections, metadata, capabilities)


@contextmanager
def imported_ufo_module(source: Path) -> Iterator[ModuleType]:
    """Use the loader's import convention while isolating same-name UFOs.

    The caller owns the existing model-load lock/environment scope. Keep this
    context open through tree normalization so both readers see one import.
    """
    source = source.resolve()
    if not (source / "__init__.py").is_file():
        raise ValueError(f"UFO model directory has no __init__.py: {source}")
    names = {source.name, *(path.stem for path in source.glob("*.py"))}

    def owned(name: str) -> bool:
        return name.split(".", 1)[0] in names

    saved = {name: module for name, module in sys.modules.items() if owned(name)}
    previous_path = sys.path[:]
    for name in saved:
        del sys.modules[name]
    sys.path[:0] = [str(source), str(source.parent)]
    try:
        observer = _ValueSourceObserver(source)
        previous_profile = sys.getprofile()

        def profile(frame: FrameType, event: str, argument: object) -> None:
            observer.observe(frame, event, argument)
            if previous_profile is not None:
                previous_profile(frame, event, argument)

        try:
            sys.setprofile(profile)
            module = importlib.import_module(source.name)
        finally:
            sys.setprofile(previous_profile)
        module._pyamplicol_value_sources = observer
        yield module
    finally:
        for name in tuple(sys.modules):
            if owned(name):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:] = previous_path


def load_raw_loop_model(
    source: str | Path,
    *,
    restriction: str = "default",
    value_overrides: Mapping[tuple[str, int], object] | None = None,
) -> RawLoopModel:
    """Read NLO UFO records without requiring tree compilation or loop kernels."""
    from .loading import _loader_restriction_name, _sanitized_model_environment

    path = Path(source).expanduser().resolve()
    selected = _loader_restriction_name(path, restriction)
    with _sanitized_model_environment(), imported_ufo_module(path) as module:
        return capture_raw_loop_model(
            module,
            name=path.name,
            restriction=raw_restriction(path, selected),
            value_overrides=value_overrides,
        )


def raw_restriction(source: Path, restriction: str | None) -> dict[str, object]:
    """Keep the restriction card for later loop lowering; do not apply it here."""
    card = (
        None
        if restriction in {"full", "none"}
        else source / f"restrict_{restriction or 'default'}.dat"
    )
    if card is not None and restriction is not None and not card.is_file():
        raise ValueError(f"UFO restriction card does not exist: {card}")
    return {
        "applied": False,
        "name": None
        if card is None or not card.is_file()
        else restriction or "default",
        "card": None
        if card is None or not card.is_file()
        else card.read_text(encoding="utf-8"),
    }
