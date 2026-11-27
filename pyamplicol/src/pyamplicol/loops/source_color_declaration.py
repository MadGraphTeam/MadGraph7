# SPDX-License-Identifier: 0BSD
"""Bind explicit model-bundle colour data to the actual owned source model."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from decimal import Decimal
from typing import TYPE_CHECKING, cast

from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
from pyamplicol.models.loop_declaration import LoopSchemeDeclaration

from .._internal.physics.symbols import symbols as symbol_registry
from .scalar_bindings import _normalize_parameter_values

if TYPE_CHECKING:
    from symbolica import Expression

    from pyamplicol.models.loop_import import RawLoopModel

    from .compiler import PreparedLoopRequestModel
    from .source_color import SourceColorContinuation


# The existing exact input normalizer is shared with the source owner. Give its
# current checked contract a type here rather than duplicate its implementation.
_normalize_values = cast(
    Callable[[object, Sequence[str], Sequence[str]], dict[str, tuple[str, str]]],
    _normalize_parameter_values,
)


def source_color_fixed_parameters(
    raw: RawLoopModel,
    declaration: LoopColorDeclaration,
    *,
    fixed_parameters: Mapping[str, str | tuple[str, str]] | None = None,
    parameters: Mapping[str, str | tuple[str, str]] | None = None,
) -> dict[str, tuple[str, str]]:
    """Merge required external scheme constants without overriding user inputs.

    Internal constants are proved from their source formulas during preparation.
    The returned external map feeds the ordinary shared source owner, which
    rejects later conflicting updates atomically. No second constraint engine
    or numerical evaluation of the scheme is introduced.
    """
    if not isinstance(declaration, LoopColorDeclaration):
        raise TypeError("expected explicit source colour model data")
    records = tuple(raw.collections["parameters"])
    names = tuple(record["name"] for record in records)
    types = tuple(record["fields"]["type"] for record in records)
    required = _normalize_values(declaration.scheme_parameter_values, names, types)
    external = tuple(
        record for record in records if record["fields"]["nature"] == "external"
    )
    external_names = tuple(record["name"] for record in external)
    external_types = tuple(record["fields"]["type"] for record in external)
    fixed = _normalize_values(fixed_parameters or {}, external_names, external_types)
    overrides = _normalize_values(parameters or {}, external_names, external_types)
    for name, value in required.items():
        if name not in external_names:
            continue
        for source in (fixed, overrides):
            if name in source and tuple(map(Decimal, source[name])) != tuple(
                map(Decimal, value)
            ):
                raise ValueError(
                    f"source colour scheme parameter {name!r} conflicts "
                    "with an explicit value"
                )
        fixed.setdefault(name, value)
    return fixed


def _expressions(
    prepared: PreparedLoopRequestModel,
    declaration: LoopColorDeclaration,
    nc: Expression,
) -> tuple[dict[str, Expression], dict[str, Expression]]:
    """Use the exact scalar frontend for source dependencies of formal identities."""
    from symbolica import E, Replacement

    from .model_expressions import parse_exact_ufo

    records = prepared.model.raw.collections["parameters"]
    names = {record["name"] for record in records}
    if declaration.formal_symbol in names:
        raise ValueError(
            "formal Nc placeholder collides with a source parameter; "
            "choose another name"
        )
    placeholder = symbol_registry.qualified(f"UFO::{declaration.formal_symbol}")
    source_symbols = {symbol_registry.qualified(f"UFO::{name}"): name for name in names}
    identities = {}
    for name, text in declaration.parameter_identities.items():
        expression = parse_exact_ufo(text).replace(placeholder, nc)
        symbols = set(expression.get_all_symbols(False))
        unknown = (
            symbols
            - {nc, symbol_registry.qualified("symbolica::pi")}
            - source_symbols.keys()
        )
        if unknown:
            raise ValueError(
                f"source colour identity {name!r} has unbound symbols: {unknown!r}"
            )
        dependencies = tuple(
            sorted(source_symbols[s] for s in symbols & source_symbols.keys())
        )
        if dependencies:
            plan = prepared.structure.scalar_bindings(
                tuple((item, 0) for item in dependencies)
            )
            # Referring to this parameter's original source value retains its
            # own active diagnostic through semantic_parameters. An unrelated
            # diagnostic-bearing dependency needs a separately defined paired
            # continuation; adding that guard unconditionally would be wrong.
            if any(
                item != name and value.diagnostic != 0
                for item, value in zip(dependencies, plan.expressions, strict=True)
            ):
                raise NotImplementedError(
                    f"source colour identity {name!r} introduces diagnostic-bearing "
                    "source dependencies; their paired continuation is not defined"
                )
            expression = expression.replace_multiple(
                [
                    Replacement(symbol_registry.qualified(f"UFO::{item}"), value.value)
                    for item, value in zip(dependencies, plan.expressions, strict=True)
                ]
            )
        identities[name] = expression
    normalized = _normalize_values(
        declaration.scheme_parameter_values,
        tuple(record["name"] for record in records),
        tuple(record["fields"]["type"] for record in records),
    )
    ties = {
        name: parse_exact_ufo(real) + E("1i") * parse_exact_ufo(imaginary)
        for name, (real, imaginary) in normalized.items()
    }
    return identities, ties


def prepare_declared_source_color(
    prepared: PreparedLoopRequestModel,
    declaration: LoopColorDeclaration,
    *,
    scheme: LoopSchemeDeclaration,
) -> tuple[PreparedLoopRequestModel, SourceColorContinuation]:
    """Resolve recipes against concrete source entries, then retain shared owners.

    No graph, amplitude or colour coefficient is inferred from a model name.
    Whole-rule recipes outside this request's admitted order scope remain unused;
    every applied recipe is checked against the complete original source group.
    """

    from .counterterms import prepare_counterterm_model
    from .model import CovariantKernelCatalogue
    from .model_expressions import parse_exact_ufo
    from .qcd_r2_color import prepare_qcd_four_gluon_r2
    from .source_color import SourceColorContinuation
    from .wavefunction import prepare_external_field_rules

    if not isinstance(declaration, LoopColorDeclaration):
        raise TypeError("expected explicit source colour model data")
    if declaration.scheme != scheme:
        raise ValueError("source colour continuation and effective loop scheme differ")
    nc = symbol_registry.qualified("source_color_declaration::Nc")
    identities, ties = _expressions(prepared, declaration, nc)
    actual_fields = {
        (rule.particle_id, rule.correction, rule.loop_group_index)
        for rule in prepare_external_field_rules(prepared.model)
    }
    if not set(declaration.field_keys) <= actual_fields:
        raise ValueError(
            "source colour field declaration does not match an actual field rule"
        )
    continuation = SourceColorContinuation(
        prepared.model,
        nc,
        identities,
        frozenset(declaration.kernel_keys),
        frozenset(declaration.field_keys),
        scheme,
        scheme_parameter_values=ties,
    )
    checked_names = tuple(dict.fromkeys((*ties, *identities)))
    scalar_plan = (
        prepared.structure.scalar_bindings(tuple((name, 0) for name in checked_names))
        if checked_names
        else None
    )
    if ties:
        assert scalar_plan is not None
        for (name, _), value in zip(
            scalar_plan.outputs, scalar_plan.expressions, strict=True
        ):
            if name in ties and value.diagnostic != 0:
                raise ValueError(
                    f"source colour scheme parameter {name!r} has an "
                    "unresolved active diagnostic"
                )
    continuation.require_whole_rules((), scalar_plan)  # type: ignore[no-untyped-call]
    # Establish all supplied Nc=3 identities now, even if this process does not
    # select a counterterm containing each invariant. The process still owns
    # the later ordinary-scalar and selected kernel/field coverage checks.
    if identities:
        assert scalar_plan is not None
        continuation.scalars(  # type: ignore[no-untyped-call]
            tuple((name, 0) for name in identities),
            program=scalar_plan.program,
            fixed_parameters=scalar_plan.fixed_parameters,
        )
    original = prepared.counterterms
    selected = {
        entry.key for rule in original.rules.values() for entry in rule.entries
    } | set(original.unavailable_entries)
    replacements = []
    keys = set(continuation.kernel_keys)
    catalogue = None
    for recipe in declaration.whole_rules:
        entries = tuple(
            entry
            for entry in prepared.model.kernels
            if (entry.collection, entry.vertex_id, entry.loop_group_index, entry.order)
            == recipe.key
        )
        if not entries:
            raise ValueError(
                f"whole-rule colour recipe {recipe.key!r} has no actual source group"
            )
        if not any(entry.key in selected for entry in entries):
            continue
        if any(entry.key not in selected for entry in entries):
            raise ValueError(
                "whole-rule colour recipe selects only part of a source group"
            )
        if catalogue is None:
            catalogue = CovariantKernelCatalogue(
                prepared.model, prepared.structure.native
            )
        replacement = prepare_qcd_four_gluon_r2(
            prepared.model,
            catalogue,
            entries,
            nc,
            gauge_coupling=parse_exact_ufo(recipe.gauge_coupling),
            contribution=recipe.contribution,
            scheme=scheme.dimensional_regularization,
        )
        replacements.append(replacement)
        keys.update(entry.key for entry in replacement.source_entries)
    if not replacements:
        return prepared, continuation
    if any(
        rule.whole_rule_continuation is not None
        or any(
            item is not None
            for item in (*rule.basis_correspondences, *rule.tensor_representatives)
        )
        for rule in original.rules.values()
    ):
        raise ValueError(
            "declared source colour recipes require the original "
            "raw counterterm preparation"
        )
    continuation = replace(
        continuation,
        kernel_keys=frozenset(keys),
        whole_rule_continuations=tuple(replacements),
    )
    rebuilt = prepare_counterterm_model(
        prepared.model,
        original.ordinary_model,
        entries=tuple(
            entry for entry in prepared.model.kernels if entry.key in selected
        ),
        primitive_validations=tuple(
            item
            for rule in original.rules.values()
            for item in rule.primitive_validations
            if item is not None
        ),
        whole_rule_continuations=continuation.whole_rule_continuations,
        vertex_spinor_frame=original.vertex_spinor_frame,
    )
    continuation.require_whole_rules(  # type: ignore[no-untyped-call]
        rebuilt.rules.values(), scalar_plan
    )
    return replace(prepared, counterterms=rebuilt), continuation
