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

from .kernels import require_finite_batch
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

    def evaluator_bytes(self):
        return self.program.library.evaluator(
            self.expressions, self.input_symbols
        ).save()


def prepare_scalar_bindings(
    raw: Any,
    outputs: Sequence[tuple[str, int]],
    *,
    program: ScalarProgram | None = None,
    fixed_parameters: Mapping[str, str | tuple[str, str]] | None = None,
) -> ScalarBindingPlan:
    """Resolve required definitions using source syntax and dependency algebra.

    Re-lowering each source declaration with ScalarExpression bindings preserves
    diagnostics in the same active branches as their values. Adding dependency
    diagnostics unconditionally would incorrectly activate bypassed conditions.
    Ordinary external parameters are the only independently settable inputs.
    Explicit fixed parameters are exact generation constants, excluded from the
    runtime input map. The raw model and its source scalar program stay unchanged.
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
    external = tuple(record for record in all_external if record["name"] not in fixed)
    external_names = tuple(record["name"] for record in external)
    types = tuple(record["fields"].get("type") for record in external)
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
            resolved[name] = fixed_values[name]
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
        program,
        external_names,
        types,
        symbols,
        defaults,
        outputs,
        expressions,
        MappingProxyType(fixed),
    )


@dataclass(frozen=True, slots=True)
class NativeScalarSnapshot:
    """One checked global state; native buffers remain valid after updates."""

    arguments: Any
    values: Any
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
        self.options = MappingProxyType(
            {"arithmetic": arithmetic, "precision": precision}
        )
        self._update_lock = Lock()
        self._identity = object()
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
        overrides = self._normalize({} if parameters is None else parameters)
        arguments, values = self._evaluate(overrides)
        self._state = NativeScalarSnapshot(
            arguments, values, MappingProxyType(overrides), 0, self._identity
        )

    def snapshot(self) -> NativeScalarSnapshot:
        """Read arguments, values and revision together without copying carriers."""
        return self._state

    @property
    def revision(self):
        return self._state.revision

    @property
    def overrides(self):
        return self._state.overrides

    @property
    def inputs(self):
        return self._state.arguments.take_columns(
            list(range(len(self.plan.external_names)))
        )

    @property
    def values(self):
        return self._state.values.take_columns(list(range(len(self.plan.outputs))))

    def _normalize(self, changes):
        if isinstance(changes, Mapping):
            fixed = changes.keys() & self.plan.fixed_parameters.keys()
            if fixed:
                raise ValueError(
                    "parameters fixed during generation cannot be updated: "
                    f"{sorted(fixed)}"
                )
        return _normalize_parameter_values(
            changes, self.plan.external_names, self.plan.external_types
        )

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
        return arguments, computed.take_columns(
            list(range(0, 2 * len(self.plan.outputs), 2))
        )

    def update(self, changes: Mapping[str, str | tuple[str, str]]):
        # Native evaluation releases the GIL. Serialize writers, then publish
        # all state fields with one assignment; readers retain either snapshot.
        with self._update_lock:
            previous = self._state
            candidate = {**previous.overrides, **self._normalize(changes)}
            arguments, values = self._evaluate(candidate)
            self._state = NativeScalarSnapshot(
                arguments,
                values,
                MappingProxyType(candidate),
                previous.revision + 1,
                self._identity,
            )
            return values.take_columns(list(range(len(self.plan.outputs))))


class NativeScaleBindings:
    """Compiled scale-dependent outputs sharing one global parameter owner.

    Only the selected real external scale is replaced by sqrt(mu_r2). All other
    inputs come from a single owner snapshot per call. Selected output plans may
    differ but must share their ScalarProgram, external layout and fixed map.
    No independent update interface, running coupling or per-point model inputs
    are introduced. As with native evaluators, calls on one view are serialized.
    """

    def __init__(
        self,
        global_state: NativeScalarBindings,
        plan: ScalarBindingPlan | None = None,
        *,
        scale_parameter="MU_R",
    ):
        from symbolica import S

        if not isinstance(global_state, NativeScalarBindings):
            raise TypeError("scale bindings require a NativeScalarBindings owner")
        owner = global_state.plan
        plan = owner if plan is None else plan
        if plan.program is not owner.program:
            raise ValueError("scale plan must share the identical ScalarProgram owner")
        if (
            plan.external_names != owner.external_names
            or plan.external_types != owner.external_types
            or plan.input_symbols != owner.input_symbols
            or plan.fixed_parameters != owner.fixed_parameters
        ):
            raise ValueError("scale plan external layout or fixed parameters differ")
        if scale_parameter in owner.fixed_parameters:
            raise ValueError("scale parameter is fixed during generation")
        if scale_parameter not in owner.external_names:
            raise ValueError("scale parameter must be an adjustable external parameter")
        index = owner.external_names.index(scale_parameter)
        if owner.external_types[index] != "real":
            raise ValueError("scale parameter must be real")
        self.global_state, self.plan = global_state, plan
        self.scale_parameter, self._scale_index = scale_parameter, index
        self.backend = global_state.backend
        library = plan.program.library
        scale = S("loop_model_scale::mu_r2")
        scale_outputs = (
            library.lower(
                "1 / (1 if s.imag == 0 and s.real > 0 else 0)",
                locals={"s": scale},
            ),
            library.lower("cmath.sqrt(s)", locals={"s": scale}),
        )
        self._scale = self.backend.CoefficientEvaluator(
            library.evaluator(scale_outputs, (scale,)).save(), **global_state.options
        )
        self._evaluator = self.backend.CoefficientEvaluator(
            plan.evaluator_bytes(), **global_state.options
        )

    def evaluate_native(self, mu_r2, *, snapshot: NativeScalarSnapshot | None = None):
        """Return checked source values in scale-lane order, entirely natively."""
        snapshot = self.global_state.snapshot() if snapshot is None else snapshot
        if (
            not isinstance(snapshot, NativeScalarSnapshot)
            or snapshot._owner is not self.global_state._identity
        ):
            raise ValueError("scale snapshot belongs to a different global state")
        if not isinstance(mu_r2, self.backend.NativeBatch):
            raise TypeError("scales require a NativeBatch")
        if mu_r2.columns != 1:
            raise ValueError("scales require one mu_r2 column")
        require_finite_batch(mu_r2, "renormalization scales")
        # The native evaluator checks both carrier and precision, including
        # empty batches and arbitrary precision whose bits match double/DD.
        scales = self._scale.evaluate_batch(mu_r2)
        require_finite_batch(scales, "positive real renormalization scales")
        arguments = snapshot.arguments.take_rows(
            [0] * mu_r2.rows, row_ids=mu_r2.row_ids
        )
        columns = [
            scales.take_columns([2])
            if index == self._scale_index
            else arguments.take_columns([index])
            for index in range(len(self.plan.external_names))
        ]
        computed = self._evaluator.evaluate_batch(
            self.backend.NativeBatch.concat_columns(columns)
        )
        require_finite_batch(
            computed, "scale-dependent scalar values and active diagnostics"
        )
        return computed.take_columns(list(range(0, 2 * len(self.plan.outputs), 2)))
