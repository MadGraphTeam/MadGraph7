# SPDX-License-Identifier: 0BSD
"""Exact source model bindings executed once per global parameter state.

This is a scalar-program consumer. It does not set a physics scheme, run a
coupling, bind per-event renormalization scales or certify an EW process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from graphlib import TopologicalSorter
from types import MappingProxyType
from typing import Any

from .kernels import require_finite_batch
from .model_expressions import LaurentExpression
from .scalar_frontend import ScalarExpression, ScalarProgram, lower_scalar_program


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

    def evaluator_bytes(self):
        return self.program.library.evaluator(
            self.expressions, self.input_symbols
        ).save()


def prepare_scalar_bindings(
    raw: Any,
    outputs: Sequence[tuple[str, int]],
    *,
    program: ScalarProgram | None = None,
) -> ScalarBindingPlan:
    """Resolve required definitions using source syntax and dependency algebra.

    Re-lowering each source declaration with ScalarExpression bindings preserves
    diagnostics in the same active branches as their values. Adding dependency
    diagnostics unconditionally would incorrectly activate bypassed conditions.
    Ordinary external parameters are the only independently settable inputs.
    """
    from symbolica import S

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
    outputs = tuple(tuple(output) for output in outputs)
    if not outputs or len(set(outputs)) != len(outputs):
        raise ValueError("scalar outputs must be nonempty and distinct")
    for name, power in outputs:
        if name not in records:
            raise ValueError(f"unknown scalar output {name!r}")
        if type(power) is not int:
            raise TypeError("scalar output epsilon powers must be integers")
    external = tuple(
        record
        for record in raw.collections["parameters"]
        if record["fields"].get("nature") == "external"
    )
    external_names = tuple(record["name"] for record in external)
    types = tuple(record["fields"].get("type") for record in external)
    if any(kind not in {"real", "complex"} for kind in types):
        raise ValueError("external parameter type must be real or complex")
    symbols = tuple(S(f"loop_model_input::{name}") for name in external_names)
    inputs = dict(zip(external_names, symbols, strict=True))
    by_symbol = {S(f"UFO::{name}"): name for name in records}
    dependencies = {}
    for name, (collection, record) in records.items():
        definition = program.definitions[collection, record["id"]]
        if definition.name != name:
            raise ValueError("scalar program record identity differs")
        original = library.lower_value(raw.exact_value(collection, record["id"]))
        used = set(original.value.get_all_symbols()) | set(
            original.diagnostic.get_all_symbols()
        )
        dependencies[name] = {by_symbol[symbol] for symbol in used & by_symbol.keys()}
        if name in inputs and dependencies[name]:
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
        if name in inputs:
            resolved[name] = ScalarExpression(inputs[name], library.zero)
            continue
        collection, record = records[name]
        resolved[name] = library.lower_value(
            raw.exact_value(collection, record["id"]),
            bindings={
                dependency: resolved[dependency] for dependency in dependencies[name]
            },
        )
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
        program, external_names, types, symbols, defaults, outputs, expressions
    )


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
        plan: ScalarBindingPlan,
        *,
        arithmetic="double",
        precision=None,
        backend=None,
        parameters=None,
    ):
        if backend is None:
            import _loop_reduction as backend

        self.plan, self.backend = plan, backend
        self.options = {"arithmetic": arithmetic, "precision": precision}
        self._evaluator = backend.CoefficientEvaluator(
            plan.evaluator_bytes(), **self.options
        )
        self._defaults = (
            backend.CoefficientEvaluator(
                plan.program.library.evaluator(plan.defaults, ()).save(), **self.options
            )
            if plan.defaults
            else None
        )
        library = plan.program.library
        # A real-typed input must have exactly zero imaginary part. Keep the
        # guard in the selected native carrier, including tiny AP components.
        guards = tuple(
            library.lower("1 / (1 if z.imag == 0 else 0)", locals={"z": symbol})
            for kind, symbol in zip(
                plan.external_types, plan.input_symbols, strict=True
            )
            if kind == "real"
        )
        self._guards = (
            backend.CoefficientEvaluator(
                library.evaluator(guards, plan.input_symbols).save(), **self.options
            )
            if guards
            else None
        )
        self._overrides = self._normalize({} if parameters is None else parameters)
        self._values = self._evaluate(self._overrides)
        self.revision = 0

    @property
    def overrides(self):
        return MappingProxyType(dict(self._overrides))

    @property
    def values(self):
        return self._values.take_columns(list(range(len(self.plan.outputs))))

    def _normalize(self, changes):
        if not isinstance(changes, Mapping):
            raise TypeError("external updates must be a parameter mapping")
        result = {}
        for name, value in changes.items():
            if name not in self.plan.external_names:
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
                raise TypeError(
                    "external values require decimal strings or complex pairs"
                )
            try:
                exact = tuple(Decimal(component) for component in value)
            except InvalidOperation as error:
                raise ValueError("external values must be decimal numbers") from error
            if not all(component.is_finite() for component in exact):
                raise ValueError("external values must be finite")
            index = self.plan.external_names.index(name)
            if self.plan.external_types[index] == "real" and not exact[1].is_zero():
                raise ValueError(
                    f"real external parameter {name!r} has an imaginary part"
                )
            result[name] = value
        return result

    def _evaluate(self, overrides):
        batch = self.backend.NativeBatch
        empty = batch.from_decimal([], 1, 0, **self.options)
        if self._defaults is None:
            arguments = empty
        else:
            missing = [
                index
                for index, name in enumerate(self.plan.external_names)
                if name not in overrides
            ]
            default = self._defaults.evaluate_batch(empty) if missing else None
            if default is not None:
                require_finite_batch(
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
            for index, name in enumerate(self.plan.external_names):
                columns.append(
                    batch.from_decimal([overrides[name]], 1, 1, **self.options)
                    if name in overrides
                    else default.take_columns([2 * index])
                )
            arguments = batch.concat_columns(columns)
        require_finite_batch(arguments, "model external inputs")
        if self._guards is not None:
            require_finite_batch(
                self._guards.evaluate_batch(arguments), "real model input type"
            )
        computed = self._evaluator.evaluate_batch(arguments)
        require_finite_batch(computed, "model scalar values and active diagnostics")
        return computed.take_columns(list(range(0, 2 * len(self.plan.outputs), 2)))

    def update(self, changes: Mapping[str, str | tuple[str, str]]):
        candidate = {**self._overrides, **self._normalize(changes)}
        values = self._evaluate(candidate)
        self._overrides, self._values = candidate, values
        self.revision += 1
        return self.values
