# SPDX-License-Identifier: 0BSD
"""Explicit semantic colour views of paired, exact source scalar bindings.

This preparation boundary does not certify an SU(Nc) model continuation. In
particular, fixed-SU3 effective tensors need their own model-supplied extension.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from pyamplicol.models.loop_declaration import LoopSchemeDeclaration

from .model import LoopModelIR
from .physical_color import split_color_grades
from .scalar_bindings import ScalarBindingPlan, prepare_scalar_bindings
from .scalar_frontend import ScalarExpression


@dataclass(frozen=True, slots=True)
class SourceColorContinuation:
    """Explicit source-owned model data, not an inferred model certificate.

    Listed local source tensors use their declared SU(Nc) interpretation with
    the supplied exact scalar identities. Fixed-SU3 whole-rule tensors must be
    replaced by independently checked model input before listing their keys.
    Particle-field keys are (particle ID, correction, loop-group index).
    """

    model: LoopModelIR
    nc: Any
    parameter_identities: Mapping[str, Any]
    kernel_keys: frozenset[tuple[Any, ...]]
    field_keys: frozenset[tuple[int, str, int]]
    scheme: LoopSchemeDeclaration
    whole_rule_continuations: tuple[Any, ...] = ()
    scheme_parameter_values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        from symbolica import AtomType, Expression

        if not isinstance(self.model, LoopModelIR):
            raise TypeError("formal source colour requires an owned loop model")
        if not isinstance(self.scheme, LoopSchemeDeclaration):
            raise TypeError("formal source colour requires an explicit loop scheme")
        if not isinstance(self.nc, Expression) or self.nc.get_type() != AtomType.Var:
            raise TypeError("Nc must be an exact symbolic variable")
        if not isinstance(self.parameter_identities, Mapping):
            raise TypeError("semantic colour identities require a mapping")
        keys = frozenset(self.kernel_keys)
        if not keys <= {entry.key for entry in self.model.kernels}:
            raise ValueError("formal colour keys do not belong to the source model")
        fields = frozenset(self.field_keys)
        if any(
            len(key) != 3
            or type(key[0]) is not int
            or not 0 <= key[0] < len(self.model.raw.collections["particles"])
            or not isinstance(key[1], str)
            or not key[1]
            or type(key[2]) is not int
            or key[2] < 0
            for key in fields
        ):
            raise ValueError("invalid source particle-field colour key")
        object.__setattr__(self, "kernel_keys", keys)
        object.__setattr__(self, "field_keys", fields)
        object.__setattr__(
            self,
            "parameter_identities",
            MappingProxyType(dict(self.parameter_identities)),
        )
        from .qcd_r2_color import QCDR2Continuation

        continuations = tuple(self.whole_rule_continuations)
        if any(not isinstance(item, QCDR2Continuation) for item in continuations):
            raise TypeError(
                "whole-rule colour continuation requires checked model input"
            )
        if len({item.key for item in continuations}) != len(continuations):
            raise ValueError("duplicate whole-rule colour continuation")
        for item in continuations:
            if (
                item.nc != self.nc
                or item.scheme != self.scheme.dimensional_regularization
            ):
                raise ValueError(
                    "whole-rule colour continuation and declared Nc/scheme differ"
                )
            if any(
                not any(entry is owned for owned in self.model.kernels)
                for entry in item.source_entries
            ):
                raise ValueError(
                    "whole-rule colour continuation belongs to another source"
                )
        if not isinstance(self.scheme_parameter_values, Mapping):
            raise TypeError("scheme parameter ties require a mapping")
        object.__setattr__(self, "whole_rule_continuations", continuations)
        object.__setattr__(
            self,
            "scheme_parameter_values",
            MappingProxyType(dict(self.scheme_parameter_values)),
        )

    def require_whole_rules(self, rules, scalar_plan):
        """Check actual applied rules and structural scheme ties before projection."""
        for rule in rules:
            actual = rule.whole_rule_continuation
            expected = next(
                (
                    item
                    for item in self.whole_rule_continuations
                    if item.key
                    == (
                        rule.entries[0].collection,
                        rule.entries[0].vertex_id,
                        rule.entries[0].loop_group_index,
                        rule.order,
                    )
                ),
                None,
            )
            if actual is not expected:
                raise ValueError(
                    "prepared counterterm and declared whole-rule continuation differ"
                )
        if not self.scheme_parameter_values:
            return
        from .exact import require_exact_numbers

        original = prepare_scalar_bindings(
            self.model.raw,
            tuple((name, 0) for name in self.scheme_parameter_values),
            program=scalar_plan.program,
            fixed_parameters=scalar_plan.fixed_parameters,
        )
        parameters = {p["name"] for p in self.model.raw.collections["parameters"]}
        for (name, expected), expression in zip(
            self.scheme_parameter_values.items(), original.expressions, strict=True
        ):
            if name not in parameters:
                raise ValueError("scheme ties require ordinary source parameters")
            require_exact_numbers(expected)
            if expected.get_all_symbols(False):
                raise ValueError("scheme parameter ties must be exact constants")
            if (expression.value - expected).together().expand() != 0:
                raise ValueError(
                    f"scheme parameter {name!r} must remain fixed at {expected}; "
                    "use the shared source fixed_parameters for an external tie"
                )

    def require_kernels(self, keys):
        for key in keys:
            if key not in self.kernel_keys:
                raise NotImplementedError(
                    f"selected source kernel {key!r} has no declared "
                    "formal-colour continuation"
                )

    def require_fields(self, keys):
        for key in keys:
            if key not in self.field_keys:
                raise NotImplementedError(
                    f"selected source particle field {key!r} has no declared "
                    "formal-colour continuation"
                )

    def require_ordinary_kernels(self, rules, bound_names, *, scalar_plan=None):
        """Use the structural loader's unchanged source rule/coupling names.

        Its native graph boundary also checks homogeneous coupling orders. A
        future name or binding change must fail here, rather than silently
        omitting an ordinary interaction from continuation coverage.
        """
        rules, bound_names = frozenset(rules), frozenset(bound_names)
        selected = tuple(
            entry
            for entry in self.model.kernels
            if entry.kind == "tree" and (entry.vertex_name, entry.order) in rules
        )
        missing = rules - {(entry.vertex_name, entry.order) for entry in selected}
        if missing:
            raise ValueError(
                "selected ordinary rules have no exact source kernel mapping: "
                + repr(sorted(missing))
            )
        omitted = tuple(
            entry for entry in selected if entry.coupling.name not in bound_names
        )
        if omitted:
            # The structural loader may remove exact fixed-parameter zeros.
            # Prove every such omission in this same formal source view; one
            # surviving entry must not conceal an unbound nonzero tensor term.
            outputs = tuple(dict.fromkeys((e.coupling.name, 0) for e in omitted))
            if scalar_plan is None:
                raise ValueError(
                    f"selected ordinary source kernel {omitted[0].key!r} "
                    "has no bound coupling or exact-zero proof"
                )
            formal = self.scalars(
                outputs,
                program=scalar_plan.program,
                fixed_parameters=scalar_plan.fixed_parameters,
            ).formal
            values = dict(zip(formal.outputs, formal.expressions, strict=True))
            for entry in omitted:
                value = values[entry.coupling.name, 0]
                if value.value != 0 or value.diagnostic != 0:
                    raise ValueError(
                        f"selected ordinary source kernel {entry.key!r} "
                        "has no bound coupling or exact-zero proof"
                    )
        self.require_kernels(
            entry.key for entry in selected if entry.coupling.name in bound_names
        )

    def scalars(
        self,
        outputs,
        *,
        program,
        fixed_parameters=None,
        anonymous_definitions=None,
        exact_outputs=None,
    ):
        if program.library.raw is not self.model.raw:
            raise ValueError("formal colour and scalar source owners differ")
        return prepare_source_color_scalars(
            self.model.raw,
            outputs,
            nc=self.nc,
            parameter_identities=self.parameter_identities,
            program=program,
            fixed_parameters=fixed_parameters,
            anonymous_definitions=anonymous_definitions,
            exact_outputs=exact_outputs,
        )

    def require_ordinary_scalars(self, plan):
        """Current ordinary currents require Nc-independent couplings and masses.

        This is an exact formula check over the whole external parameter space,
        never a test of default numerical values. Original active diagnostics
        are preserved by the existing semantic preparation boundary.
        """
        formal = self.scalars(
            plan.outputs, program=plan.program, fixed_parameters=plan.fixed_parameters
        )
        for grade, columns in zip(formal.grades, formal.columns, strict=True):
            if grade == 0:
                continue
            for output, column in zip(plan.outputs, columns, strict=True):
                if formal.coefficients.expressions[column].value != 0:
                    raise NotImplementedError(
                        f"selected ordinary scalar {output!r} depends on Nc; "
                        "ordinary source-current colour grading is not yet supported"
                    )
        return formal


@dataclass(frozen=True, slots=True)
class SourceColorScalarPlan:
    """Nc-independent native outputs, indexed by grade and original output.

    Each grade carries the same original active diagnostics. Exact zero
    coefficients remain zero and must not determine a physical leading degree.
    A wholly zero value catalogue retains grade zero for diagnostic evaluation.
    """

    nc: Any
    formal: ScalarBindingPlan
    coefficients: ScalarBindingPlan
    grades: tuple[int, ...]
    columns: tuple[tuple[int, ...], ...]


def prepare_source_color_scalars(
    raw,
    outputs,
    *,
    nc,
    parameter_identities,
    program=None,
    fixed_parameters=None,
    anonymous_definitions=None,
    exact_outputs=None,
):
    """Retain explicit named invariants before recursive scalar substitution.

    At Nc=3 each declared value must reproduce the original source parameter
    identically, including any ordinary external inputs. Mutable external
    invariants are rejected by scalar preparation unless explicitly fixed.
    Diagnostics and helper ownership are preserved, never algebraically
    interpreted as colour powers or dropped when a value coefficient vanishes.
    """
    from symbolica import AtomType, E, Expression, Replacement, S

    if not isinstance(nc, Expression) or nc.get_type() != AtomType.Var:
        raise TypeError("Nc must be an exact symbolic variable")
    if not isinstance(parameter_identities, Mapping):
        raise TypeError("semantic colour identities require a mapping")
    identities = dict(parameter_identities)
    outputs = tuple(outputs)
    if exact_outputs is not None and not isinstance(exact_outputs, Mapping):
        raise TypeError("prepared scalar outputs require an exact-expression mapping")
    exact_outputs = dict(exact_outputs or {})
    source_symbols = {
        S(f"UFO::{record['name']}"): record["name"]
        for record in raw.collections["parameters"]
    }
    source_names = {
        record["name"]
        for collection in ("parameters", "counterterm_parameters", "couplings")
        for record in raw.collections[collection]
    }
    from .exact import require_exact_numbers

    dependencies = set()
    for output, expression in exact_outputs.items():
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not isinstance(output[0], str)
            or not output[0]
            or type(output[1]) is not int
            or output in outputs
            or output[0] in source_names
        ):
            raise ValueError(
                "prepared scalar outputs require distinct name/power pairs"
            )
        require_exact_numbers(expression)
        symbols = set(expression.get_all_symbols(False))
        unknown = symbols - {nc, S("symbolica::pi")} - source_symbols.keys()
        if unknown:
            raise ValueError(
                "prepared scalar coefficient has unbound symbols: " + repr(unknown)
            )
        dependencies.update(
            source_symbols[symbol] for symbol in symbols & source_symbols.keys()
        )
    outputs = tuple(
        dict.fromkeys((*outputs, *((name, 0) for name in sorted(dependencies))))
    )
    formal = prepare_scalar_bindings(
        raw,
        outputs,
        semantic_parameters=identities,
        anonymous_definitions=anonymous_definitions,
        program=program,
        fixed_parameters=fixed_parameters,
    )
    if identities:
        original = prepare_scalar_bindings(
            raw,
            tuple((name, 0) for name in identities),
            program=formal.program,
            fixed_parameters=fixed_parameters,
        )
        # This establishes compatibility at the physical colour value, not a
        # proof of the supplied continuation away from that value.
        for name, expression in zip(identities, original.expressions, strict=True):
            available = {
                nc,
                *formal.input_symbols,
                *expression.value.get_all_symbols(False),
            }
            unknown = set(identities[name].get_all_symbols(False)) - available
            if unknown:
                raise ValueError(
                    f"semantic colour identity for {name!r} has unbound symbols: "
                    + ", ".join(map(str, sorted(unknown, key=str)))
                )
            if (
                expression.value - identities[name].replace(nc, 3)
            ).together().expand() != 0:
                raise ValueError(
                    f"semantic colour identity for {name!r} differs from source at Nc=3"
                )
    if exact_outputs:
        resolved = dict(zip(formal.outputs, formal.expressions, strict=True))
        substitutions = [
            Replacement(S(f"UFO::{name}"), resolved[name, 0].value)
            for name in dependencies
        ]
        # Every referenced source output remains in the same evaluated plan,
        # carrying its original diagnostic independently of the new value.
        formal = replace(
            formal,
            outputs=(*formal.outputs, *exact_outputs),
            expressions=(
                *formal.expressions,
                *(
                    ScalarExpression(value.replace_multiple(substitutions), E("0"))
                    for value in exact_outputs.values()
                ),
            ),
        )
    for output, expression in zip(formal.outputs, formal.expressions, strict=True):
        if nc in expression.diagnostic.get_all_symbols():
            raise NotImplementedError(
                f"scalar {output!r} has an Nc-dependent active diagnostic"
            )
    try:
        graded = split_color_grades((part.value for part in formal.expressions), nc)
    except (ValueError, NotImplementedError) as error:
        raise NotImplementedError(
            f"source scalar outputs {formal.outputs!r} cannot be graded: {error}"
        ) from error
    if not graded:
        graded = ((0, (E("0"),) * len(formal.outputs)),)
    coefficients, names, columns = [], [], []
    for degree, values in graded:
        columns.append(tuple(range(len(coefficients), len(coefficients) + len(values))))
        for (name, power), original, value in zip(
            formal.outputs, formal.expressions, values, strict=True
        ):
            coefficients.append(ScalarExpression(value, original.diagnostic))
            names.append((f"{name}[Nc^{degree}]", power))
    return SourceColorScalarPlan(
        nc,
        formal,
        replace(formal, outputs=tuple(names), expressions=tuple(coefficients)),
        tuple(degree for degree, _ in graded),
        tuple(columns),
    )
