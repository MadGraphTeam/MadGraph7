# SPDX-License-Identifier: 0BSD
"""Explicit semantic colour views of paired, exact source scalar bindings.

This preparation boundary does not certify an SU(Nc) model continuation. In
particular, fixed-SU3 effective tensors need their own model-supplied extension.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
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
            if entry.kind == "tree"
            and (entry.vertex_name, entry.order) in rules
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
        self, outputs, *, program, fixed_parameters=None, anonymous_definitions=None
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
):
    """Retain explicit named invariants before recursive scalar substitution.

    At Nc=3 each declared value must reproduce the original source parameter
    identically, including any ordinary external inputs. Mutable external
    invariants are rejected by scalar preparation unless explicitly fixed.
    Diagnostics and helper ownership are preserved, never algebraically
    interpreted as colour powers or dropped when a value coefficient vanishes.
    """
    from symbolica import AtomType, E, Expression

    if not isinstance(nc, Expression) or nc.get_type() != AtomType.Var:
        raise TypeError("Nc must be an exact symbolic variable")
    if not isinstance(parameter_identities, Mapping):
        raise TypeError("semantic colour identities require a mapping")
    identities = dict(parameter_identities)
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
