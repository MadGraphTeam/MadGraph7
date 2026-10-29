# SPDX-License-Identifier: 0BSD
"""The complex-mass scheme of a UFO model, on its exact source records.

``complex_mass_scheme`` ports MG5_aMC's own transform of a model to the
complex-mass scheme (``Model.change_mass_to_complex_scheme``, MG5_aMC 3.7.0
``madgraph/core/base_objects.py``), so MadLoop's transform is both the
specification and the oracle (m-sm PLAN §3.2 MS-CMS). The steps, as MG5:

1. the model's NLOCT convention parameter ``CMSParam`` (looked up by name, as
   MG5 does: a UFO convention, not a particle or process name) becomes 1;
2. every particle whose width parameter is first met (MG5 visits each width
   once, in source order) leaves its mass and width real; if that width is
   nonzero at the card (the restriction card with the request's fixed
   parameters, as MG5 reads the restriction card before it transforms):

   * an internal width takes ``re()``;
   * an external mass ``M`` gets the complex pole mass
     ``CMASS_M = cmath.sqrt(M**2 - complex(0,1) * M * W)`` (MG5's expression);
   * an internal mass is refused (``cms-internal-mass``): MG5's electroweak-
     scheme switch for an internal W mass (``change_electroweak_mode``, keyed
     by PDG 24 and the MW expression) is not ported (genericity; m-sm PLAN
     §1.7);
3. the YUKAWA-block inputs become the masses of their particles (the complex
   pole mass where the mass has one): under the scheme they are derived, and a
   request that sets one is refused (``cms-yukawa-input``);
4. if some mass is complex, every internal parameter other than the masses
   and widths of step 2 becomes complex, and every reference to a mass ``M``
   in parameters, counterterm parameters, couplings and the particles' field
   counterterms (MG5's ``UVWfct`` couplings) becomes ``CMASS_M``.

Two things MG5 does in its output rather than in its model are done here on
the records, so that one derived model serves the tree and the loop lowering:
every particle whose mass parameter has a complex pole mass propagates with it
and with width ZERO (the width is part of the pole mass; MG5's HELAS calls
take ``CMASS_M`` for a particle with a width, MadLoop's loop denominators
the loop group's ``CMASS_M``), ghosts and Goldstone bosons of that mass
included (m-sm PLAN §1.5); and the UFO-standard principal ``reglog``, which
MG5's model library writes whatever the UFO binds ([U3]), is the bound helper
(``reglog_convention = "ufo-standard"``; ``"source"`` keeps the model's own
binding as a compatibility view).
"""

from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any

from pyamplicol.models.loop_import import RawLoopModel

# MG5_aMC's lookup of the NLOCT convention parameter (``get_parameter``).
CMS_PARAMETER_NAMES = ("CMSParam", "mdl_CMSParam")
# The UFO-standard reglog that MG5_aMC's model library writes ([U3]).
UFO_STANDARD_REGLOG = "(0.0 if z==0.0 else cmath.log(z))"
CMASS_PREFIX = "CMASS_"
METADATA_KEY = "complex_mass_scheme"
RAW_REFUSALS = ("cms-internal-mass", "cms-yukawa-input")


class ComplexMassSchemeUnsupported(ValueError):
    """A model or request the complex-mass transform refuses; ``key`` names
    the rule (``cms-internal-mass``, ``cms-yukawa-input``)."""

    def __init__(self, key: str, message: str) -> None:
        if key not in RAW_REFUSALS:
            raise ValueError(f"unknown complex-mass refusal {key!r}")
        super().__init__(f"{key}: {message}")
        self.key = key


@dataclass(frozen=True, slots=True)
class ComplexMassScheme:
    """The complex-mass scheme of ``original`` (``complex_mass_scheme``).

    ``masses`` maps each mass parameter with a complex pole mass to it (MG5's
    ``to_change``) and ``widths`` to its width; ``real`` are the masses and
    widths MG5 leaves real (its ``mass_widths``); ``internal_widths`` took
    ``re()``; ``yukawas`` maps each former Yukawa input to its mass (the pole
    mass where there is one); ``cms_parameter`` is the enabled convention
    parameter, if the model has one."""

    original: RawLoopModel
    derived: RawLoopModel
    masses: Mapping[str, str]
    widths: Mapping[str, str]
    real: tuple[str, ...]
    internal_widths: tuple[str, ...]
    yukawas: Mapping[str, str]
    cms_parameter: str | None
    reglog_convention: str
    propagators: Mapping[int, str] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        return {
            "masses": dict(self.masses),
            "widths": dict(self.widths),
            "real": list(self.real),
            "internal_widths": list(self.internal_widths),
            "yukawas": dict(self.yukawas),
            "cms_parameter": self.cms_parameter,
            "reglog_convention": self.reglog_convention,
            "propagators": {str(pdg): mass for pdg, mass in self.propagators.items()},
        }


@dataclass(frozen=True, slots=True)
class CMSModelDeclarations:
    original: RawLoopModel
    derived: RawLoopModel
    mass_aliases: Any
    tied_yukawas: Any
    diagnostics: tuple[Any, ...]


def _rename_expression(expression, replacements):
    """Rename source identifiers (not attributes, e.g. ``cmath.pi``) in place,
    every other character of the source text kept as it is (numeric lexemes
    are never reparsed)."""
    lines = expression.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    previous = None
    edits = []
    for token in tokenize.generate_tokens(io.StringIO(expression).readline):
        if (
            token.type == tokenize.NAME
            and previous != "."
            and token.string in replacements
        ):
            (row, column), (end_row, end_column) = token.start, token.end
            edits.append(
                (
                    offsets[row - 1] + column,
                    offsets[end_row - 1] + end_column,
                    replacements[token.string],
                )
            )
        if token.type not in {tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT}:
            previous = token.string
    for begin, finish, name in reversed(edits):
        expression = expression[:begin] + name + expression[finish:]
    return expression


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


# ------------------------------------------------------------ the MG5_aMC port


def _exact_pair(expression: str) -> tuple[Decimal, Decimal] | None:
    """The exact (real, imaginary) value of a numeric literal source value
    (``2.4952``, ``-1e-3``, ``complex(2.49877,0)``), else None."""

    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return None

    def number(part: ast.expr) -> Decimal | None:
        sign = Decimal(1)
        while isinstance(part, ast.UnaryOp) and isinstance(
            part.op, (ast.UAdd, ast.USub)
        ):
            sign = -sign if isinstance(part.op, ast.USub) else sign
            part = part.operand
        if not isinstance(part, ast.Constant) or type(part.value) not in (int, float):
            return None
        segment = ast.get_source_segment(expression.strip(), part)
        try:
            return sign * Decimal(segment if segment is not None else repr(part.value))
        except InvalidOperation:
            return None

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "complex"
        and not node.keywords
        and 1 <= len(node.args) <= 2
    ):
        parts = [number(argument) for argument in node.args]
        if any(part is None for part in parts):
            return None
        return parts[0], parts[1] if len(parts) == 2 else Decimal(0)
    value = number(node)
    return None if value is None else (value, Decimal(0))


class _Card:
    """The values the transform decides on: the request's fixed parameters,
    else the restriction card (``model_restrictions``), else the source
    defaults; an internal parameter through the model's scalar program (only
    when one is needed: an internal width)."""

    def __init__(self, raw: RawLoopModel, fixed: Mapping[str, Any]):
        from .model_restrictions import prepare_loop_restrictions

        restriction = raw.metadata.get("restriction", {})
        self.card = (
            raw
            if restriction.get("card") is None or restriction.get("applied", False)
            else prepare_loop_restrictions(raw, simplify=False).derived
        )
        self.fixed = {}
        for name, value in fixed.items():
            pair = (value, "0") if isinstance(value, str) else tuple(value)
            self.fixed[name] = tuple(Decimal(str(part)) for part in pair)
        self._internal: dict[str, tuple[Decimal, Decimal]] = {}

    def value(self, record: Mapping[str, Any]) -> tuple[Decimal, Decimal]:
        name = record["name"]
        if name in self.fixed:
            return self.fixed[name]
        source = self.card.exact_value("parameters", record["id"])
        pair = (
            _exact_pair(source["expression"])
            if source.get("kind") == "expression"
            else None
        )
        if pair is not None:
            return pair
        if name not in self._internal:
            self._internal[name] = self._evaluate(name)
        return self._internal[name]

    def _evaluate(self, name: str) -> tuple[Decimal, Decimal]:
        from .scalar_bindings import prepare_scalar_bindings

        external = {
            record["name"]: self.value(record)
            for record in self.card.collections["parameters"]
            if record["fields"].get("nature") == "external"
        }
        plan = prepare_scalar_bindings(
            self.card,
            [(name, 0)],
            fixed_parameters={
                key: (str(real), str(imaginary))
                for key, (real, imaginary) in external.items()
            },
        )
        value = plan.expressions[0].value.evaluate({}, decimal_digit_precision=32)
        real, imaginary = (Decimal(part) for part in value.to_decimal_tuple())
        return real, imaginary


def _named(raw: RawLoopModel, collection: str) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for record in raw.collections[collection]:
        if record["name"] in records:
            raise ValueError(f"ambiguous source {collection} name {record['name']!r}")
        records[record["name"]] = record
    return records


def _reference(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "reference",
        "collection": "parameters",
        "id": record["id"],
        "name": record["name"],
    }


def _rename_counterterm(value: Any, replacements: Mapping[str, str]) -> Any:
    """A particle's field counterterm (``{key: {power: expression}}`` in the
    raw mapping encoding) with ``replacements`` applied to its expressions."""

    if isinstance(value, Mapping):
        if value.get("kind") == "mapping":
            return {
                "kind": "mapping",
                "entries": [
                    [key, _rename_counterterm(item, replacements)]
                    for key, item in value["entries"]
                ],
            }
        return {
            key: _rename_counterterm(item, replacements) for key, item in value.items()
        }
    if isinstance(value, str):
        return _rename_expression(value, replacements)
    return value


def complex_mass_scheme(
    raw: RawLoopModel,
    *,
    fixed_parameters: Mapping[str, Any] | None = None,
    reglog_convention: str = "ufo-standard",
) -> ComplexMassScheme:
    """MG5_aMC's ``change_mass_to_complex_scheme`` on the exact source records
    of ``raw`` (module docstring), deciding the widths at the card: the
    request's ``fixed_parameters`` (exact decimal strings or pairs), else
    ``raw``'s restriction card, else the source defaults. ``raw`` is left
    unchanged; the derived model records the transform under
    ``metadata["complex_mass_scheme"]``."""

    if reglog_convention not in {"ufo-standard", "source"}:
        raise ValueError("reglog_convention must be ufo-standard or source")
    if METADATA_KEY in raw.metadata or "complex_mass_derivation" in raw.metadata:
        raise ValueError("the complex-mass scheme is already applied to this model")
    fixed = dict(fixed_parameters or {})
    card = _Card(raw, fixed)
    payload = raw.to_dict()
    collections = payload["collections"]
    parameters = collections["parameters"]
    by_name = _named(raw, "parameters")

    # 1. The NLOCT convention parameter (MG5: CMSParam.expr = '1.0').
    cms_parameter = next(
        (name for name in CMS_PARAMETER_NAMES if name in by_name), None
    )
    if cms_parameter is not None:
        record = parameters[by_name[cms_parameter]["id"]]
        if record["fields"].get("nature") == "external":
            raise NotImplementedError(
                f"the model's {cms_parameter} is an external parameter"
            )
        _set_value(record, {"kind": "expression", "expression": "1.0"}, "enable-cms")

    # 2. Pole masses, per width parameter first met (MG5's particle loop).
    real: list[str] = []
    masses: dict[str, str] = {}
    widths: dict[str, str] = {}
    internal_widths: list[str] = []
    for particle in raw.collections["particles"]:
        fields = particle["fields"]
        width_name = fields["width"]["name"]
        if width_name in real:
            continue
        mass_name = fields["mass"]["name"]
        real += [width_name, mass_name]
        width = by_name.get(width_name)
        if width_name.lower() == "zero" or width is None:
            continue
        if card.value(width) == (0, 0):
            continue
        if width["fields"].get("nature") != "external":
            source = raw.exact_value("parameters", width["id"])
            if source.get("kind") != "expression":
                raise NotImplementedError(f"the internal width {width_name!r}")
            body = source["expression"]
            continued = (
                f"re({body})"
                if "re" in raw.metadata.get("function_bindings", {})
                else f"(({body})).real"
            )
            _set_value(
                parameters[width["id"]],
                {"kind": "expression", "expression": continued},
                "real-part-of-internal-width",
            )
            internal_widths.append(width_name)
        mass = by_name.get(mass_name)
        if (
            mass is None
            or mass["fields"].get("nature") != "external"
            or mass["fields"].get("type") != "real"
        ):
            raise ComplexMassSchemeUnsupported(
                "cms-internal-mass",
                f"{fields['name']!r} (PDG {fields['pdg_code']}) has the nonzero "
                f"width {width_name!r} at the card and its mass {mass_name!r} is "
                "not a real external parameter; its complex pole mass would need "
                "a change of the model's input scheme (MG5_aMC switches an "
                "internal W mass to (MZ, MW, alpha) inputs, which is not ported)",
            )
        alias = CMASS_PREFIX + mass_name
        if alias in by_name:
            raise ValueError(f"the pole-mass name {alias!r} is a model parameter")
        masses[mass_name] = alias
        widths[mass_name] = width_name

    # 3. The Yukawa inputs become their particles' (pole) masses.
    species = {
        int(record["fields"]["pdg_code"]): record
        for record in raw.collections["particles"]
    }
    yukawas: dict[str, str] = {}
    for record in parameters:
        fields = record["fields"]
        if fields.get("nature") != "external" or (
            str(fields.get("lhablock") or "").lower() != "yukawa"
        ):
            continue
        code = fields.get("lhacode")
        if not isinstance(code, (list, tuple)) or len(code) != 1:
            raise NotImplementedError(f"the Yukawa input {record['name']!r}")
        particle = species.get(int(code[0]))
        if particle is None:
            raise ValueError(
                f"the Yukawa input {record['name']!r} names no particle PDG {code[0]}"
            )
        if record["name"] in fixed:
            raise ComplexMassSchemeUnsupported(
                "cms-yukawa-input",
                f"the request fixes the Yukawa input {record['name']!r}; under the "
                "complex-mass scheme the Yukawa couplings follow the masses (MG5_aMC "
                "removes the YUKAWA block)",
            )
        mass_name = particle["fields"]["mass"]["name"]
        yukawas[record["name"]] = masses.get(mass_name, mass_name)
        fields.update(nature="internal", type="complex" if masses else "real")
        fields.pop("lhablock", None)
        fields.pop("lhacode", None)
        _set_value(
            record, {"kind": "expression", "expression": mass_name}, "yukawa-from-mass"
        )

    # 4. Complex internal parameters and the pole masses everywhere.
    propagators: dict[int, str] = {}
    if masses:
        kept = set(real)
        for collection in ("parameters", "counterterm_parameters", "couplings"):
            for record in collections[collection]:
                fields = record["fields"]
                if collection == "parameters" and (
                    fields.get("nature") == "external" or record["name"] in kept
                ):
                    continue
                if collection != "couplings":
                    fields["type"] = "complex"
                if record["name"] not in yukawas:
                    raw.exact_value(collection, record["id"])  # exact origin
                value = _replace_value(fields["value"], masses)
                if value != fields["value"]:
                    operation = fields.get("value_source", {}).get("operation")
                    _set_value(
                        record,
                        value,
                        operation
                        if operation in {"enable-cms", "yukawa-from-mass"}
                        else "complex-pole-masses",
                    )
        for record in collections["particles"]:
            counterterm = record["fields"].get("counterterm")
            if counterterm is not None:
                record["fields"]["counterterm"] = _rename_counterterm(
                    counterterm, masses
                )
        aliases = {}
        for mass_name, alias in masses.items():
            record = {
                "id": len(parameters),
                "name": alias,
                "ufo_type": "Parameter",
                "fields": {
                    "name": alias,
                    "nature": "internal",
                    "type": "complex",
                    "texname": alias,
                },
            }
            _set_value(
                record,
                {
                    "kind": "expression",
                    "expression": (
                        f"cmath.sqrt({mass_name}**2 - complex(0,1) * {mass_name} "
                        f"* {widths[mass_name]})"
                    ),
                },
                "complex-pole-mass",
            )
            parameters.append(record)
            aliases[mass_name] = record
        zero = by_name.get("ZERO")
        if zero is None:
            raise NotImplementedError("the model has no ZERO parameter")
        for record in collections["particles"]:
            fields = record["fields"]
            mass_name = fields["mass"]["name"]
            if mass_name in aliases:
                fields["mass"] = _reference(aliases[mass_name])
                fields["width"] = _reference(zero)
                propagators[int(fields["pdg_code"])] = aliases[mass_name]["name"]

    # [U3]: the UFO-standard principal reglog that MG5's model library writes.
    if reglog_convention == "ufo-standard":
        bound = raw.metadata.get("function_bindings", {}).get("reglog")
        if bound is not None:
            helper = collections["functions"][bound]
            if _source_tokens(helper["fields"]["expr"]) != _source_tokens(
                UFO_STANDARD_REGLOG
            ):
                helper["fields"]["expr"] = UFO_STANDARD_REGLOG
                helper["fields"]["expression_source"] = {
                    "kind": "derived-complex-mass-declaration",
                    "exact": True,
                    "operation": "select-ufo-standard-reglog",
                    "declaration": [bound, helper["name"]],
                }

    scheme = ComplexMassScheme(
        raw,
        raw,
        MappingProxyType(masses),
        MappingProxyType(widths),
        tuple(real),
        tuple(internal_widths),
        MappingProxyType(yukawas),
        cms_parameter,
        reglog_convention,
        MappingProxyType(propagators),
    )
    payload["metadata"][METADATA_KEY] = scheme.metadata()
    payload["name"] += "_cms"
    return ComplexMassScheme(
        raw,
        RawLoopModel.from_dict(payload),
        scheme.masses,
        scheme.widths,
        scheme.real,
        scheme.internal_widths,
        scheme.yukawas,
        cms_parameter,
        reglog_convention,
        scheme.propagators,
    )
