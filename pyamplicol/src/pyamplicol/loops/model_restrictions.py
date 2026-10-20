# SPDX-License-Identifier: 0BSD
"""Exact restriction declarations using the installed UFO card parser."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any

from pyamplicol.models.loop_import import RawLoopModel


@dataclass(frozen=True, slots=True)
class RestrictedLoopDeclarations:
    original: RawLoopModel
    derived: RawLoopModel
    fixed_parameters: Any


def _exact_card_entries(text):
    """Let ParamCard own syntax/addresses; carry lexemes through its comments.

    Its numeric values are binary64 and are never consumed here. Annotation
    changes comments only. Every nonempty source line must survive in a native
    block/parameter record, so ignored syntax and duplicate rounded entries do
    not silently acquire exact-input status.
    """
    from ufo_model_loader.param_card import ParamCard

    lines, annotated = {}, []
    for index, line in enumerate(text.splitlines()):
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        lines[index] = body
        annotated.append(f"{body} # __pyamplicol_card_line_{index}\n")
    card = ParamCard()
    card.read(io.StringIO("".join(annotated)))
    seen, entries = set(), {}

    def source(comment):
        match = re.fullmatch(r"\s*__pyamplicol_card_line_(\d+)\s*", comment)
        if match is None or int(match[1]) not in lines:
            raise ValueError("restriction parser lost the exact source line")
        index = int(match[1])
        seen.add(index)
        return lines[index]

    for block_name, block in card.items():
        if block.scale is not None:
            raise NotImplementedError(
                "scale-dependent restriction blocks are unsupported"
            )
        if block_name != "decay":
            header = source(block.comment).split()
            if len(header) != 2 or [part.lower() for part in header] != [
                "block",
                block_name.lower(),
            ]:
                raise ValueError("unsupported restriction block header")
        if block_name.startswith("decay_table") or getattr(block, "decay_table", {}):
            raise NotImplementedError("restriction decay tables are unsupported")
        for code, parameter in block.param_dict.items():
            tokens = source(parameter.comment).split()
            if block_name == "decay":
                if tokens[0].lower() != "decay":
                    raise ValueError("restriction parser changed the decay address")
                tokens = tokens[1:]
            try:
                if any(
                    re.fullmatch(r"[+-]?[0-9]+", token) is None for token in tokens[:-1]
                ):
                    raise ValueError("non-integer address token")
                source_code = tuple(int(token) for token in tokens[:-1])
            except ValueError as error:
                raise ValueError("unsupported restriction parameter address") from error
            if source_code != tuple(code):
                raise ValueError("restriction parser changed the parameter address")
            lexeme = tokens[-1]
            if (
                re.fullmatch(
                    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[EeDd][+-]?[0-9]+)?",
                    lexeme,
                )
                is None
            ):
                raise ValueError("restriction values must be finite decimal literals")
            try:
                value = Decimal(lexeme.replace("D", "E").replace("d", "e"))
            except InvalidOperation as error:
                raise ValueError(
                    "restriction values must be finite decimal literals"
                ) from error
            if not value.is_finite():
                raise ValueError("restriction values must be finite decimal literals")
            entries[block_name.upper(), tuple(code)] = str(value)
    if seen != lines.keys() or card.not_parsed_entry:
        raise ValueError("restriction contains duplicate or unsupported card records")
    return entries


def prepare_loop_restrictions(raw: RawLoopModel, *, simplify=True):
    """Apply exact card defaults and freeze zeros under the loader convention.

    Nonzero values remain adjustable external defaults. With simplification,
    exactly zero external values (including omitted zero defaults) become fixed
    generation inputs, as in ``Model.apply_input_param_card``. Source records,
    the original card and the unrestricted input remain available unchanged.
    No parameter equality, flavour identification or numerical pruning is inferred.

    As in MadGraph's restriction reader, a card entry whose LHA address names no
    external parameter of this model (e.g. a decay width the model does not
    declare) is ignored; such entries are recorded under ``ignored_entries`` in
    the derived restriction metadata. An ``IM`` block entry only addresses the
    imaginary part of a complex parameter with the matching real address.
    """
    if not isinstance(simplify, bool):
        raise TypeError("restriction simplification must be a boolean")
    restriction = raw.metadata.get("restriction", {})
    text = restriction.get("card")
    if text is None:
        return RestrictedLoopDeclarations(raw, raw, MappingProxyType({}))
    if restriction.get("applied", False):
        raise ValueError("source restrictions have already been applied")
    entries = _exact_card_entries(text)
    payload = raw.to_dict()
    external = [
        record
        for record in payload["collections"]["parameters"]
        if record["fields"].get("nature") == "external"
    ]
    addresses = {}
    for record in external:
        fields = record["fields"]
        block, code = fields.get("lhablock"), fields.get("lhacode")
        if block is None or code is None:
            continue
        address = (str(block).upper(), tuple(code))
        if address in addresses:
            raise ValueError("ambiguous external parameter LHA address")
        addresses[address] = record
    values, ignored = {}, []
    for (block, code), literal in entries.items():
        address, component = (block, code), 0
        if address not in addresses and block.startswith("IM"):
            real = addresses.get((block[2:], code))
            if real is not None and real["fields"].get("type") == "complex":
                address, component = (block[2:], code), 1
        if address not in addresses:
            ignored.append([block, list(code), literal])
            continue
        record = addresses[address]
        pair = values.setdefault(record["id"], [None, None])
        pair[component] = literal
    for record in external:
        if record["id"] not in values:
            continue
        real, imaginary = values[record["id"]]
        if record["fields"].get("type") == "complex" and (
            real is None or imaginary is None
        ):
            raise NotImplementedError(
                "complex restriction entries require both real and imaginary blocks; "
                "partial component updates are unsupported"
            )
        if imaginary is None:
            imaginary = "0"
        record["fields"]["value"] = {
            "kind": "expression",
            "expression": f"complex({real},{imaginary})",
        }
        record["fields"]["value_source"] = {
            "kind": "exact-restriction-card",
            "exact": True,
            "declaration": [record["id"], record["name"]],
        }
    payload["metadata"]["restriction"].update(
        applied=True, simplify=simplify, ignored_entries=ignored
    )
    derived = RawLoopModel.from_dict(payload)
    fixed = {}
    if simplify:
        from .scalar_frontend import lower_scalar_program

        program = lower_scalar_program(derived)
        for record in external:
            definition = program.definitions["parameters", record["id"]]
            if definition.value.expression == 0 and definition.diagnostic == 0:
                fixed[record["name"]] = ("0", "0")
    payload["metadata"]["restriction"]["fixed_parameters"] = fixed
    derived = RawLoopModel.from_dict(payload)
    return RestrictedLoopDeclarations(raw, derived, MappingProxyType(fixed))


def restriction_fixed_parameters(raw, additional=None):
    """Merge explicit generation constants without undoing a card restriction."""
    from .scalar_bindings import _normalize_parameter_values

    external = [
        record
        for record in raw.collections["parameters"]
        if record["fields"].get("nature") == "external"
    ]
    names = tuple(record["name"] for record in external)
    types = tuple(record["fields"]["type"] for record in external)
    required = _normalize_parameter_values(
        raw.metadata.get("restriction", {}).get("fixed_parameters", {}), names, types
    )
    extra = _normalize_parameter_values(additional or {}, names, types)
    for name in required.keys() & extra.keys():
        if tuple(map(Decimal, required[name])) != tuple(map(Decimal, extra[name])):
            raise ValueError(
                f"fixed parameter {name!r} contradicts the source restriction"
            )
    return {**required, **extra}
