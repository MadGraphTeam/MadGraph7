# SPDX-License-Identifier: 0BSD
"""Cold complex-mass declarations for the source Gmu model.

This is a declaration transformation, not a CMS process certificate. The source
supplies its renormalization formulas and analytic continuations. Both scalar
and structural lowering consume the same derived immutable model.
"""

from __future__ import annotations

import io
import tokenize
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pyamplicol.models.loop_import import RawLoopModel


@dataclass(frozen=True, slots=True)
class CMSModelDeclarations:
    original: RawLoopModel
    derived: RawLoopModel
    mass_aliases: Any
    tied_yukawas: Any
    diagnostics: tuple[Any, ...]


def _rename_expression(expression, replacements):
    """Rename source identifiers without reparsing numeric literal values."""
    tokens = list(tokenize.generate_tokens(io.StringIO(expression).readline))
    previous = None
    changed = False
    rewritten = []
    for token in tokens:
        name = token.string
        if token.type == tokenize.NAME and previous != "." and name in replacements:
            name = replacements[name]
            changed = True
        rewritten.append((token.type, name))
        if token.type not in {tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT}:
            previous = token.string
    return tokenize.untokenize(rewritten) if changed else expression


def _replace_value(value, replacements):
    if value["kind"] == "expression":
        return {
            "kind": "expression",
            "expression": _rename_expression(value["expression"], replacements),
        }
    if value["kind"] == "laurent":
        return {
            "kind": "laurent",
            "terms": [
                [power, _rename_expression(expression, replacements)]
                for power, expression in value["terms"]
            ],
        }
    raise ValueError("unsupported exact source scalar value")


def _source_tokens(expression):
    return tuple(
        (token.type, token.string)
        for token in tokenize.generate_tokens(io.StringIO(expression).readline)
        if token.type
        not in {tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER, tokenize.COMMENT}
    )


def _set_value(record, value, operation):
    fields = record["fields"]
    fields["value"] = value
    fields["value_source"] = {
        "kind": "derived-complex-mass-declaration",
        "exact": True,
        "operation": operation,
        "declaration": [record["id"], record["name"]],
    }


def prepare_gmu_complex_mass_declarations(
    raw: RawLoopModel, *, reglog_convention: str = "source"
) -> CMSModelDeclarations:
    """Continue the external-mass Gmu source without changing its original data.

    Real external masses and widths remain independent inputs. Pole masses are
    shared by every species using the corresponding mass, including covariant
    multiplet partners. Yukawa inputs become internal mass aliases. The source's
    Gfbar and recms definitions are retained, with CMSParam enabled. The default
    preserves the bound source reglog too. Explicit ``ufo-standard`` selects the
    reserved principal-log convention used by UFO 2.0 and MadGraph's exporter,
    only for the recognized Gmu helper bodies. Its discrepancy with the original
    source is retained in diagnostics. No other logarithm or CT formula changes.
    """
    from .covariant_model import _repair

    if reglog_convention not in {"source", "ufo-standard"}:
        raise ValueError("reglog_convention must be source or ufo-standard")
    if "complex_mass_derivation" in raw.metadata:
        raise ValueError("complex-mass declarations are already derived")
    if tuple(raw.metadata.get("gauge", ())) != (1,) or raw.collections["propagators"]:
        raise NotImplementedError(
            "Gmu CMS declarations require the source Feynman gauge"
        )
    profile = {
        "CMSParam": "0.0",
        "sw2": "1 - MW**2/MZ**2",
        "aEW": "cmath.sqrt(2.)*Gfbar*MW**2*(1-MW**2/MZ**2)/cmath.pi",
    }
    for name, expression in profile.items():
        record = raw.record("parameters", name)
        value = raw.exact_value("parameters", record["id"])
        if (
            record["fields"]["nature"] != "internal"
            or value["kind"] != "expression"
            or _source_tokens(value["expression"]) != _source_tokens(expression)
        ):
            raise NotImplementedError(f"source {name} differs from the Gmu CMS profile")
    # Verify the source's actual bound helper, rather than registry ordering.
    for name, expression in {
        "recms": "(z if cms else z.real)",
        "crecms": "(z.conjugate() if cms else z.real)",
    }.items():
        record = raw.collections["functions"][raw.metadata["function_bindings"][name]]
        if _source_tokens(record["fields"]["expr"]) != _source_tokens(expression):
            raise NotImplementedError(f"source {name} differs from the Gmu CMS profile")

    payload = raw.to_dict()
    collections = payload["collections"]
    parameters = collections["parameters"]
    by_name = {record["name"]: record for record in parameters}
    if len(by_name) != len(parameters):
        raise ValueError("ambiguous source parameter names")
    mass_widths = {"MZ": "WZ", "MW": "WW", "MT": "WT", "MH": "WH"}
    aliases = {mass: f"CMASS_{mass}" for mass in mass_widths}
    for mass, width in mass_widths.items():
        for name in (mass, width):
            fields = by_name[name]["fields"]
            if fields["nature"] != "external" or fields["type"] != "real":
                raise NotImplementedError(
                    "Gmu CMS requires real external masses and widths"
                )
        if aliases[mass] in by_name:
            raise ValueError("complex pole-mass alias already exists")
    species = {
        record["fields"]["pdg_code"]: record for record in collections["particles"]
    }
    for pdg, mass in {23: "MZ", 24: "MW", 6: "MT", 25: "MH"}.items():
        fields = species[pdg]["fields"]
        if (
            fields["mass"]["name"] != mass
            or fields["width"]["name"] != mass_widths[mass]
        ):
            raise NotImplementedError(
                "source particle mass/width pair differs from Gmu"
            )

    # Reuse the existing exact source-profile repair before mass names change.
    # This leaves the original anti-constructor evidence available for its guard.
    repaired = [dict(record["fields"]) for record in collections["particles"]]
    diagnostics = _repair(raw, repaired, "gmu_particle_anti")
    for diagnostic in diagnostics:
        collections["particles"][diagnostic["particle_id"]]["fields"][
            diagnostic["field"]
        ] = diagnostic["derived"]

    if reglog_convention == "ufo-standard":
        helper_id = raw.metadata["function_bindings"]["reglog"]
        helper = collections["functions"][helper_id]
        fields = helper["fields"]
        principal = "(0.0 if z==0.0 else cmath.log(z))"
        continued = (
            "(0.0 if z.imag==0.0 and z.real==0.0 else "
            "(cmath.log(z)-2*math.pi*1j if "
            "(z.real<0.0 and z.imag>0.0) else cmath.log(z)))"
        )
        if (
            helper["name"] != "reglog"
            or fields.get("arguments") not in ("z", ["z"])
            or not fields.get("expression_source", {}).get("exact", False)
            or _source_tokens(fields["expr"])
            not in {_source_tokens(principal), _source_tokens(continued)}
        ):
            raise NotImplementedError(
                "source reglog differs from the recognized Gmu helper profile"
            )
        diagnostic = {
            "profile": "gmu-ufo-standard-reglog",
            "function_id": helper_id,
            "function": "reglog",
            "raw": fields["expr"],
            "derived": principal,
            "reason": (
                "explicit UFO 2.0 reserved principal-log convention; legacy "
                "source overwrites reglog with reglogm semantics, whereas "
                "MadGraph emits its reserved principal implementation"
            ),
        }
        fields["expr"] = principal
        fields["expression_source"] = {
            "kind": "derived-complex-mass-declaration",
            "exact": True,
            "operation": "select-ufo-standard-reglog",
            "declaration": [helper_id, "reglog"],
        }
        diagnostics = (*diagnostics, MappingProxyType(diagnostic))

    tied = {}
    for record in parameters:
        fields = record["fields"]
        if (fields.get("lhablock") or "").lower() != "yukawa":
            continue
        code = fields.get("lhacode")
        if (
            fields["nature"] != "external"
            or not isinstance(code, (list, tuple))
            or len(code) != 1
            or code[0] not in species
        ):
            raise NotImplementedError("unsupported source Yukawa declaration")
        mass = species[code[0]]["fields"]["mass"]["name"]
        tied[record["name"]] = aliases.get(mass, mass)
        fields.update(nature="internal", type="complex")
        fields.pop("lhablock", None)
        fields.pop("lhacode", None)
        _set_value(
            record, {"kind": "expression", "expression": mass}, "tie-yukawa-mass"
        )

    for collection in ("parameters", "counterterm_parameters", "couplings"):
        for record in collections[collection]:
            fields = record["fields"]
            if collection == "parameters" and fields["nature"] == "external":
                continue
            if record["name"] not in tied:
                raw.exact_value(collection, record["id"])
            if collection != "couplings":
                fields["type"] = "complex"
            value = _replace_value(fields["value"], aliases)
            if record["name"] == "CMSParam":
                value = {"kind": "expression", "expression": "1"}
            if value != fields["value"]:
                _set_value(
                    record,
                    value,
                    "enable-cms"
                    if record["name"] == "CMSParam"
                    else "continue-mass-dependencies",
                )

    alias_records = {}
    for mass, width in mass_widths.items():
        name = aliases[mass]
        record = {
            "id": len(parameters),
            "name": name,
            "ufo_type": "Parameter",
            "fields": {
                "name": name,
                "nature": "internal",
                "type": "complex",
                "texname": name,
            },
        }
        # Appended after the rewrite: the real input names must not self-replace.
        _set_value(
            record,
            {
                "kind": "expression",
                "expression": f"cmath.sqrt({mass}**2-complex(0,1)*{mass}*{width})",
            },
            "pole-mass-from-real-inputs",
        )
        parameters.append(record)
        alias_records[mass] = record
    for record in collections["particles"]:
        mass = record["fields"]["mass"]["name"]
        if mass in alias_records:
            alias = alias_records[mass]
            record["fields"]["mass"] = {
                "kind": "reference",
                "collection": "parameters",
                "id": alias["id"],
                "name": alias["name"],
            }

    metadata = {
        "profile": "gmu-external-mass-cms",
        "mass_aliases": aliases,
        "mass_widths": mass_widths,
        "tied_yukawas": tied,
        "reglog_convention": reglog_convention,
        "diagnostics": [dict(diagnostic) for diagnostic in diagnostics],
        "physical_certification": False,
    }
    payload["metadata"]["complex_mass_derivation"] = metadata
    payload["name"] += "_cms_declarations"
    return CMSModelDeclarations(
        raw,
        RawLoopModel.from_dict(payload),
        MappingProxyType(aliases),
        MappingProxyType(tied),
        diagnostics,
    )
