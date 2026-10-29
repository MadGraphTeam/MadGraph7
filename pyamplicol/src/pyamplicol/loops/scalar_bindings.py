# SPDX-License-Identifier: 0BSD
"""Exact global source bindings and native per-point scale views.

These scalar-program consumers do not set a physics scheme, run a coupling or
certify an EW process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from graphlib import TopologicalSorter
from threading import Lock
from types import MappingProxyType
from typing import Any

from .._internal.physics.symbols import symbols as symbol_registry
from .model_expressions import LaurentExpression
from .scalar_frontend import ScalarExpression, ScalarProgram, lower_scalar_program


def _normalize_parameter_values(changes, names, types):
    if not isinstance(changes, Mapping):
        raise TypeError("external updates must be a parameter mapping")
    result = {}
    for name, value in changes.items():
        if name not in names:
            raise ValueError(f"unknown external parameter {name!r}")
        if isinstance(value, str):
            value = (value, "0")
        elif isinstance(value, (tuple, list)):
            value = tuple(value)
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or any(not isinstance(component, str) for component in value)
        ):
            raise TypeError("external values require decimal strings or complex pairs")
        try:
            exact = tuple(Decimal(component) for component in value)
        except InvalidOperation as error:
            raise ValueError("external values must be decimal numbers") from error
        if not all(component.is_finite() for component in exact):
            raise ValueError("external values must be finite")
        index = names.index(name)
        if types[index] == "real" and not exact[1].is_zero():
            raise ValueError(f"real external parameter {name!r} has an imaginary part")
        result[name] = value
    return result


@dataclass(frozen=True, slots=True)
class ScalarBindingPlan:
    """Carrier-independent selected scalar outputs and exact external defaults."""

    program: ScalarProgram
    external_names: tuple[str, ...]
    external_types: tuple[str, ...]
    input_symbols: tuple[Any, ...]
    defaults: tuple[ScalarExpression, ...]
    outputs: tuple[tuple[str, int], ...]
    expressions: tuple[ScalarExpression, ...]
    fixed_parameters: Mapping[str, tuple[str, str]]
    anonymous_definitions: tuple[tuple[str, Mapping[str, Any]], ...] = ()
    _prepared: Any = field(default=None, init=False, repr=False, compare=False)
    _prepare_lock: Any = field(
        default_factory=Lock, init=False, repr=False, compare=False
    )

    def evaluator_bytes(self):
        return self.program.library.evaluator(
            self.expressions, self.input_symbols
        ).save()

    @property
    def prepared(self):
        if self._prepared is None:
            with self._prepare_lock:
                if self._prepared is None:
                    object.__setattr__(
                        self, "_prepared", PreparedScalarBindings.from_plan(self)
                    )
        return self._prepared


@dataclass(frozen=True, slots=True)
class PreparedScalarBindings:
    """Exact scalar instructions and plain metadata, without a source producer.

    A record can be shared by independently mutable native owners. Values,
    defaults and real-input guards are all paired saved dependency programs;
    no numerical buffers, scalar library or raw UFO are retained here.
    """

    external_names: tuple[str, ...]
    external_types: tuple[str, ...]
    outputs: tuple[tuple[str, int], ...]
    fixed_parameters: tuple[tuple[str, tuple[str, str]], ...]
    values_state: bytes
    defaults_state: bytes | None
    guards_state: bytes | None

    @classmethod
    def from_plan(cls, plan):
        library = plan.program.library
        guards = tuple(
            library.lower("1 / (1 if z.imag == 0 else 0)", locals={"z": symbol})
            for kind, symbol in zip(
                plan.external_types, plan.input_symbols, strict=True
            )
            if kind == "real"
        )
        return cls(
            plan.external_names,
            plan.external_types,
            plan.outputs,
            tuple(plan.fixed_parameters.items()),
            plan.evaluator_bytes(),
            library.evaluator(plan.defaults, ()).save() if plan.defaults else None,
            library.evaluator(guards, plan.input_symbols).save() if guards else None,
        )

    def bind(
        self,
        *,
        arithmetic="double",
        precision=None,
        backend=None,
        parameters=None,
    ):
        return NativeScalarBindings(
            self,
            arithmetic=arithmetic,
            precision=precision,
            backend=backend,
            parameters=parameters,
        )


def prepare_scalar_bindings(
    raw: Any,
    outputs: Sequence[tuple[str, int]],
    *,
    program: ScalarProgram | None = None,
    fixed_parameters: Mapping[str, str | tuple[str, str]] | None = None,
    anonymous_definitions: Mapping[str, Mapping[str, Any]] | None = None,
    semantic_parameters: Mapping[str, Any] | None = None,
) -> ScalarBindingPlan:
    """Resolve required definitions using source syntax and dependency algebra.

    Re-lowering each source declaration with ScalarExpression bindings preserves
    diagnostics in the same active branches as their values. Adding dependency
    diagnostics unconditionally would incorrectly activate bypassed conditions.
    Ordinary external parameters are the only independently settable inputs.
    Explicit fixed parameters are exact generation constants, excluded from the
    runtime input map. The raw model and its source scalar program stay unchanged.

    Anonymous definitions are terminal exact source expressions, such as a
    particle's field factor. They may reference ordinary source declarations,
    but cannot override them or introduce another dependency graph. Their
    original syntax is lowered with the same paired bindings and helper owner.

    Explicit semantic parameter values are substituted before dependent source
    declarations are resolved. Their original active diagnostics are retained.
    These cold symbolic views do not certify the proposed physical identities.
    Adjustable external parameters cannot be replaced: fix them explicitly so
    their runtime updates cannot silently become ineffective.
    """

    from .exact import require_exact_numbers
    from .scalar_frontend import require_laurent_scalar

    program = lower_scalar_program(raw) if program is None else program
    library = program.library
    if library.raw is not raw:
        raise ValueError("scalar program belongs to a different raw source instance")
    records = {
        record["name"]: (collection, record)
        for collection in ("parameters", "counterterm_parameters", "couplings")
        for record in raw.collections[collection]
    }
    if len(records) != len(program.definitions):
        raise ValueError("scalar program and source declarations differ")
    if anonymous_definitions is not None and not isinstance(
        anonymous_definitions, Mapping
    ):
        raise TypeError("anonymous scalar definitions require a mapping")
    anonymous = {}
    for name, source in (anonymous_definitions or {}).items():
        if not isinstance(name, str) or not name.isidentifier() or name in records:
            raise ValueError(
                "anonymous scalar names must be distinct from source declarations"
            )
        if not isinstance(source, Mapping):
            raise TypeError("anonymous scalar definitions require exact source syntax")
        if source.get("kind") == "expression" and set(source) == {"kind", "expression"}:
            if not isinstance(source["expression"], str):
                raise TypeError("anonymous scalar expressions require exact strings")
            normalized = dict(source)
        elif source.get("kind") == "laurent" and set(source) == {"kind", "terms"}:
            terms = tuple(tuple(term) for term in source["terms"])
            if any(
                len(term) != 2
                or type(term[0]) is not int
                or not isinstance(term[1], str)
                for term in terms
            ) or len({term[0] for term in terms}) != len(terms):
                raise ValueError(
                    "anonymous Laurent terms require unique integer powers "
                    "and exact strings"
                )
            normalized = {"kind": "laurent", "terms": terms}
        else:
            raise ValueError("unsupported anonymous scalar source representation")
        anonymous[name] = MappingProxyType(normalized)
    sources = {
        name: raw.exact_value(collection, record["id"])
        for name, (collection, record) in records.items()
    }
    sources.update(anonymous)
    outputs = tuple(tuple(output) for output in outputs)
    if not outputs or len(set(outputs)) != len(outputs):
        raise ValueError("scalar outputs must be nonempty and distinct")
    for name, power in outputs:
        if name not in sources:
            raise ValueError(f"unknown scalar output {name!r}")
        if type(power) is not int:
            raise TypeError("scalar output epsilon powers must be integers")
    all_external = tuple(
        record
        for record in raw.collections["parameters"]
        if record["fields"].get("nature") == "external"
    )
    all_names = tuple(record["name"] for record in all_external)
    all_types = tuple(record["fields"].get("type") for record in all_external)
    if any(kind not in {"real", "complex"} for kind in all_types):
        raise ValueError("external parameter type must be real or complex")
    fixed = _normalize_parameter_values(
        {} if fixed_parameters is None else fixed_parameters, all_names, all_types
    )
    fixed_values = {
        name: library.lower(f"complex({Decimal(real)},{Decimal(imaginary)})")
        for name, (real, imaginary) in fixed.items()
    }
    if semantic_parameters is not None and not isinstance(semantic_parameters, Mapping):
        raise TypeError("semantic parameter identities require a mapping")
    semantic = dict(semantic_parameters or {})
    for name, expression in semantic.items():
        if name not in records or records[name][0] != "parameters":
            raise ValueError(
                f"semantic identity requires an ordinary parameter: {name!r}"
            )
        require_exact_numbers(expression)
        if name in all_names and name not in fixed:
            raise ValueError(
                f"semantic external parameter {name!r} must be fixed during generation"
            )
    external = tuple(record for record in all_external if record["name"] not in fixed)
    external_names = tuple(record["name"] for record in external)
    types = tuple(record["fields"].get("type") for record in external)
    symbols = tuple(
        symbol_registry.qualified(f"loop_model_input::{name}")
        for name in external_names
    )
    inputs = dict(zip(external_names, symbols, strict=True))
    by_symbol = {symbol_registry.qualified(f"UFO::{name}"): name for name in sources}
    anonymous_symbols = {
        symbol_registry.qualified(f"UFO::{name}") for name in anonymous
    }
    dependencies = {}
    for name, source in sources.items():
        if name in records:
            collection, record = records[name]
            definition = program.definitions[collection, record["id"]]
            if definition.name != name:
                raise ValueError("scalar program record identity differs")
        original = library.lower_value(source)
        used = set(original.value.get_all_symbols()) | set(
            original.diagnostic.get_all_symbols()
        )
        if used & anonymous_symbols:
            raise ValueError("anonymous scalar definitions must be terminal outputs")
        dependencies[name] = {by_symbol[symbol] for symbol in used & by_symbol.keys()}
        if name in all_names and dependencies[name]:
            raise ValueError(
                "external parameter defaults must be independent constants"
            )
    # TopologicalSorter validates the source's parameter dependency graph.
    ordered = tuple(TopologicalSorter(dependencies).static_order())
    needed = {name for name, _power in outputs}
    for name in reversed(ordered):
        if name in needed:
            needed.update(dependencies[name])
    resolved = {}
    for name in ordered:
        if name not in needed:
            continue
        if name in fixed_values:
            value = fixed_values[name]
        elif name in inputs:
            value = ScalarExpression(inputs[name], library.zero)
        else:
            value = library.lower_value(
                sources[name],
                bindings={
                    dependency: resolved[dependency]
                    for dependency in dependencies[name]
                },
            )
        resolved[name] = (
            ScalarExpression(semantic[name], value.diagnostic)
            if name in semantic
            else value
        )
        if name in anonymous:
            require_laurent_scalar(resolved[name])
    expressions = tuple(
        ScalarExpression(
            LaurentExpression(resolved[name].value).coefficient(power),
            resolved[name].diagnostic,
        )
        for name, power in outputs
    )
    defaults = tuple(
        library.lower_value(raw.exact_value("parameters", record["id"]))
        for record in external
    )
    return ScalarBindingPlan(
        program,
        external_names,
        types,
        symbols,
        defaults,
        outputs,
        expressions,
        MappingProxyType(fixed),
        tuple(anonymous.items()),
    )


class SourceBindingFailure(ArithmeticError):
    """A numerical source realization failed at a known native scalar stage."""

    def __init__(self, stage, detail):
        self.stage, self.detail = stage, detail
        super().__init__(f"{stage}: {detail}")


def _require_source_finite(values, stage):
    rows = values.nonfinite_rows()
    if rows:
        raise SourceBindingFailure(stage, f"nonfinite native rows {rows}")


@dataclass(frozen=True, slots=True)
class NativeScalarSnapshot:
    """Exact admitted global state with its checked buffers."""

    arguments: Any = field(repr=False)
    values: Any = field(repr=False)
    overrides: Mapping[str, tuple[str, str]]
    revision: int
    _owner: object = field(repr=False, compare=False)


class NativeScalarBindings:
    """Recompute a selected exact model state with atomic external updates.

    Inputs are original decimal strings or (real, imaginary) decimal pairs.
    Missing inputs use compiled exact source defaults in the selected carrier.
    The returned NativeBatch has one global-model row, not one row per event;
    a physical batch can broadcast it through native take_rows. No model values
    are rounded through Python complex or stored in a binary64 parameter map.
    """

    def __init__(
        self,
        plan: ScalarBindingPlan | PreparedScalarBindings,
        *,
        arithmetic="double",
        precision=None,
        backend=None,
        parameters=None,
    ):
        if backend is None:
            from .native import load_loop_native

            backend = load_loop_native()

        if not isinstance(plan, (ScalarBindingPlan, PreparedScalarBindings)):
            raise TypeError("scalar bindings require a scalar plan or prepared record")
        self.plan = plan if isinstance(plan, ScalarBindingPlan) else None
        self.prepared = plan.prepared if self.plan is not None else plan
        self.backend = backend
        self.options = MappingProxyType(
            {"arithmetic": arithmetic, "precision": precision}
        )
        self._update_lock = Lock()
        self._identity = object()
        self._evaluator = backend.CoefficientEvaluator(
            self.prepared.values_state, **self.options
        )
        self._defaults = (
            backend.CoefficientEvaluator(self.prepared.defaults_state, **self.options)
            if self.prepared.defaults_state is not None
            else None
        )
        self._guards = (
            backend.CoefficientEvaluator(self.prepared.guards_state, **self.options)
            if self.prepared.guards_state is not None
            else None
        )
        overrides = self._normalize({} if parameters is None else parameters)
        self._state = self._realize_snapshot(overrides, 0)

    def _realize_snapshot(self, overrides, revision):
        arguments, values = self._evaluate(overrides)
        return NativeScalarSnapshot(
            arguments,
            values,
            MappingProxyType(dict(overrides)),
            revision,
            self._identity,
        )

    def snapshot(self) -> NativeScalarSnapshot:
        """Read arguments, values and revision together without copying carriers."""
        return self._state

    def require_snapshot(self, snapshot):
        """Check ownership before a consumer drops or evaluates any point."""
        if (
            not isinstance(snapshot, NativeScalarSnapshot)
            or snapshot._owner is not self._identity
        ):
            raise ValueError("scale snapshot belongs to a different global state")
        return snapshot

    def reconstruct_snapshot(self, source, snapshot):
        """Re-evaluate captured exact overrides/defaults in this owner's carrier.

        This creates a detached snapshot and never publishes a global update.
        The identical cold plan is required; rounded source buffers are unused.
        """
        source.require_snapshot(snapshot)
        if self.prepared is not source.prepared:
            raise ValueError("snapshot reconstruction requires the identical plan")
        with self._update_lock:
            arguments, values = self._evaluate(snapshot.overrides)
        return NativeScalarSnapshot(
            arguments, values, snapshot.overrides, snapshot.revision, self._identity
        )

    @property
    def revision(self):
        return self._state.revision

    @property
    def overrides(self):
        return self._state.overrides

    @property
    def inputs(self):
        return self._state.arguments.take_columns(
            list(range(len(self.prepared.external_names)))
        )

    @property
    def values(self):
        return self._state.values.take_columns(list(range(len(self.prepared.outputs))))

    def _normalize(self, changes):
        if isinstance(changes, Mapping):
            fixed = changes.keys() & dict(self.prepared.fixed_parameters).keys()
            if fixed:
                raise ValueError(
                    "parameters fixed during generation cannot be updated: "
                    f"{sorted(fixed)}"
                )
        return _normalize_parameter_values(
            changes, self.prepared.external_names, self.prepared.external_types
        )

    def _evaluate(self, overrides):
        batch = self.backend.NativeBatch
        empty = batch.from_decimal([], 1, 0, **self.options)
        if self._defaults is None:
            arguments = empty
        else:
            missing = [
                index
                for index, name in enumerate(self.prepared.external_names)
                if name not in overrides
            ]
            default = self._defaults.evaluate_batch(empty) if missing else None
            if default is not None:
                _require_source_finite(
                    default.take_columns(
                        [
                            column
                            for index in missing
                            for column in (2 * index, 2 * index + 1)
                        ]
                    ),
                    "model external defaults and diagnostics",
                )
            columns = []
            for index, name in enumerate(self.prepared.external_names):
                columns.append(
                    batch.from_decimal([overrides[name]], 1, 1, **self.options)
                    if name in overrides
                    else default.take_columns([2 * index])
                )
            arguments = batch.concat_columns(columns)
        _require_source_finite(arguments, "model external inputs")
        if self._guards is not None:
            _require_source_finite(
                self._guards.evaluate_batch(arguments), "real model input type"
            )
        computed = self._evaluator.evaluate_batch(arguments)
        _require_source_finite(computed, "model scalar values and active diagnostics")
        return arguments, computed.take_columns(
            list(range(0, 2 * len(self.prepared.outputs), 2))
        )

    def update(self, changes: Mapping[str, str | tuple[str, str]]):
        """Atomically publish and return the admitted immutable source snapshot."""
        # Native evaluation releases the GIL. Serialize writers, then publish
        # all state fields with one assignment; readers retain either snapshot.
        with self._update_lock:
            previous = self._state
            candidate = {**previous.overrides, **self._normalize(changes)}
            snapshot = self._realize_snapshot(candidate, previous.revision + 1)
            self._state = snapshot
            return snapshot
