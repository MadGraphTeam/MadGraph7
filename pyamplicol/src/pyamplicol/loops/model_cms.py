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
   pole mass where the mass has one, written in the real mass and width, so
   that no coupling reads the pole-mass parameter whose sign the tree reading
   of a fermion loop line flips): under the scheme they are derived, and a
   request that sets one is refused (``cms-yukawa-input``);
4. if some mass is complex, every internal parameter other than the masses
   and widths of step 2 becomes complex, and every reference to a mass ``M``
   in parameters, counterterm parameters, couplings and the particles' field
   counterterms (MG5's ``UVWfct`` couplings) becomes ``CMASS_M``.

Not in MG5 (values unchanged): a mass that a Yukawa tie of step 3 reads and
that has no pole mass (a zero width, a nonzero mass at the card) gives its
particles a real line mass ``LMASS_M := M`` to propagate with (step 4b,
``ComplexMassScheme.lines``), so that, as with ``CMASS_M``, no coupling
reads the line's mass parameter.

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
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any

from pyamplicol.models.loop_import import RawLoopModel

from .._internal.physics.symbols import symbols as symbol_registry
from .v2.refusals import LoopRequestUnsupportedValue

# MG5_aMC's lookup of the NLOCT convention parameter (``get_parameter``).
CMS_PARAMETER_NAMES = ("CMSParam", "mdl_CMSParam")
# The UFO-standard reglog that MG5_aMC's model library writes ([U3]).
UFO_STANDARD_REGLOG = "(0.0 if z==0.0 else cmath.log(z))"
CMASS_PREFIX = "CMASS_"
# The line mass of a Yukawa-tied mass without a pole mass (step 4b).
LINE_MASS_PREFIX = "LMASS_"
METADATA_KEY = "complex_mass_scheme"
# The mass schemes of ``[loop.model.scheme] mass_scheme`` (the loop model
# declaration): real on-shell masses (every width zero in the trees of an
# electroweak request and in coloured trees, [U8]) or MG5_aMC's complex-mass
# scheme (``set complex_mass_scheme True``).
REAL_ON_SHELL = "real-on-shell"
COMPLEX_MASS = "complex-mass"
MASS_SCHEMES = (REAL_ON_SHELL, COMPLEX_MASS)
# The model parameters of the NLOCT UFOs' complex-mass conventions that a
# virtual's conventions record lists where the model declares them
# (m-sm PLAN §1.2: CMSParam, the G_mu UFO's LogSwitch and
# GfRedefinitionChoice); UFO conventions looked up by name as MG5_aMC does.
CONVENTION_PARAMETERS = (*CMS_PARAMETER_NAMES, "LogSwitch", "GfRedefinitionChoice")
# Whether the native point inputs of stored artifacts take complex pole masses
# (M-SM MS-CMS-N: complex plan masses through the native trees and executor,
# ``native_trees._parameter_program``'s ``["complex", o]`` plan masses). A build
# without them would refuse a public-route complex-mass request with a pole
# mass (``require_native_pole_masses``, ``complex-mass-scheme``).
NATIVE_POLE_MASSES = True
REFUSAL_KEYS = (
    "mass-scheme",
    "complex-mass-scheme",
    "cms-internal-mass",
    "cms-yukawa-input",
    "cms-external-width",
    "complex-fixed-parameter",
)


class ComplexMassSchemeUnsupported(LoopRequestUnsupportedValue):
    """A request or model the mass-scheme policy refuses
    (``refusals.LOOP_ADMISSION_REFUSALS``): ``key`` is one of
    ``REFUSAL_KEYS``; a ``ValueError`` too."""

    def __init__(self, key: str, message: str) -> None:
        if key not in REFUSAL_KEYS:
            raise ValueError(f"unknown complex-mass refusal {key!r}")
        super().__init__(key, message)


@dataclass(frozen=True, slots=True)
class ComplexMassScheme:
    """The complex-mass scheme of ``original`` (``complex_mass_scheme``).

    ``masses`` maps each mass parameter with a complex pole mass to it (MG5's
    ``to_change``) and ``widths`` to its width; ``real`` are the masses and
    widths MG5 leaves real (its ``mass_widths``); ``internal_widths`` took
    ``re()``; ``yukawas`` maps each former Yukawa input to its mass (the pole
    mass where there is one); ``cms_parameter`` is the enabled convention
    parameter, if the model has one; ``lines`` maps each Yukawa-tied mass
    without a pole mass to the line mass its particles propagate with."""

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
    ones: tuple[str, ...] = ()
    lines: Mapping[str, str] = field(default_factory=dict)

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
            "ones": list(self.ones),
            "lines": dict(self.lines),
        }


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
        # Whether a restriction card decides the values (MG5_aMC restricts a
        # model only with one).
        self.restricted = restriction.get("card") is not None
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


def pole_mass_expression(mass: str, width: str) -> str:
    """MG5_aMC's complex pole mass of a real external mass and its width
    (``change_mass_to_complex_scheme``'s ``CMASS_`` definition)."""

    return f"cmath.sqrt({mass}**2 - complex(0,1) * {mass} * {width})"


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
            raise ComplexMassSchemeUnsupported(
                "complex-mass-scheme",
                f"the model's {cms_parameter} is an external parameter",
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
                raise ComplexMassSchemeUnsupported(
                    "complex-mass-scheme",
                    f"the internal width {width_name!r} has no source expression",
                )
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
            raise ComplexMassSchemeUnsupported(
                "complex-mass-scheme",
                f"the pole-mass name {alias!r} is already a model parameter",
            )
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
            raise ComplexMassSchemeUnsupported(
                "complex-mass-scheme",
                f"the Yukawa input {record['name']!r} has the LHA code {code!r}, "
                "not one PDG code",
            )
        particle = species.get(int(code[0]))
        if particle is None:
            raise ComplexMassSchemeUnsupported(
                "complex-mass-scheme",
                f"the Yukawa input {record['name']!r} names no particle PDG {code[0]}",
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
        # MG5's ymX = CMASS_MX, written in the real inputs: no coupling then
        # reads the loop line's own mass parameter, whose sign the tree
        # reading of a fermion loop flips (``conventions``, module docstring).
        _set_value(
            record,
            {
                "kind": "expression",
                "expression": (
                    pole_mass_expression(mass_name, widths[mass_name])
                    if mass_name in masses
                    else mass_name
                ),
            },
            "yukawa-from-mass",
        )

    # 4. Complex internal parameters and the pole masses everywhere.
    propagators: dict[int, str] = {}
    if masses:
        kept = set(real)
        for collection in ("parameters", "counterterm_parameters", "couplings"):
            for record in collections[collection]:
                fields = record["fields"]
                if collection == "parameters" and (
                    fields.get("nature") == "external"
                    or record["name"] in kept
                    or record["name"] in yukawas
                ):
                    continue
                if collection != "couplings":
                    fields["type"] = "complex"
                raw.exact_value(collection, record["id"])  # its exact origin
                value = _replace_value(fields["value"], masses)
                if value != fields["value"]:
                    operation = fields.get("value_source", {}).get("operation")
                    _set_value(
                        record,
                        value,
                        "enable-cms"
                        if operation == "enable-cms"
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
                    "expression": pole_mass_expression(mass_name, widths[mass_name]),
                },
                "complex-pole-mass",
            )
            parameters.append(record)
            aliases[mass_name] = record
        zero = by_name.get("ZERO")
        if zero is None:
            raise ComplexMassSchemeUnsupported(
                "complex-mass-scheme",
                "the model has no ZERO parameter for the pole-mass particles' widths",
            )
        for record in collections["particles"]:
            fields = record["fields"]
            mass_name = fields["mass"]["name"]
            if mass_name in aliases:
                fields["mass"] = _reference(aliases[mass_name])
                fields["width"] = _reference(zero)
                propagators[int(fields["pdg_code"])] = aliases[mass_name]["name"]

    # 4b. A mass a Yukawa tie reads without a pole mass (a zero width at the
    # card) propagates as its own line mass LMASS_M := M: the couplings keep
    # reading M, as MG5's, and the tree reading of a fermion loop line, which
    # flips the sign of the line's mass parameter in the whole segment kernel
    # (``loops.v2.conventions``, module docstring), leaves them unchanged. A
    # pole mass splits them already (the tie in the real mass and width, step
    # 3; the line in CMASS_M).
    lines: dict[str, str] = {}
    created_lines: dict[str, dict[str, Any]] = {}
    for mass_name in dict.fromkeys(yukawas.values()):
        if mass_name in masses.values() or mass_name.lower() == "zero":
            continue
        mass = by_name.get(mass_name)
        if mass is None or card.value(mass) == (0, 0):
            continue  # a massless line: no mass sign to flip
        alias = LINE_MASS_PREFIX + mass_name
        if alias in by_name:
            raise ComplexMassSchemeUnsupported(
                "complex-mass-scheme",
                f"the line-mass name {alias!r} is already a model parameter",
            )
        record = {
            "id": len(parameters),
            "name": alias,
            "ufo_type": "Parameter",
            "fields": {
                "name": alias,
                "nature": "internal",
                "type": "real",
                "texname": alias,
            },
        }
        _set_value(record, {"kind": "expression", "expression": mass_name}, "line-mass")
        parameters.append(record)
        lines[mass_name] = alias
        created_lines[mass_name] = record
    for record in collections["particles"]:
        fields = record["fields"]
        line = created_lines.get(fields["mass"]["name"])
        if line is not None:
            fields["mass"] = _reference(line)

    # 5. MG5_aMC restricts the transformed model by its restriction card, if
    # any (``import_model``: CMS first, then ``restrict_model``), which fixes
    # every parameter of value 1 there (``detect_special_parameters``' one
    # parameters; zeros the loader's restriction already fixes): the external
    # switches at 1 (the
    # NLOCT TECHNICAL block of the G_mu UFO: epsUV, epsIR, LogSwitch,
    # GfRedefinitionChoice) become the constant 1, which removes their dead
    # branches (GfRedefinitionChoice's re() one: complex conjugates of the
    # pole masses that would otherwise blow up the expanded couplings by two
    # orders of magnitude). The masses and widths of step 2 stay inputs, and
    # so do the model's pole regulators (``poles.model_pole_regulators``:
    # epsUV and epsIR, read only by the counterterms' pole coefficients, so
    # they have no dead branch): at the card's 1 every value is MG5's, and
    # the "uv" and "ir" selections split the counterterms' poles by them
    # (DESIGN §12.4) under the scheme too.
    ones: list[str] = []
    if masses and card.restricted:
        from .v2.poles import model_pole_regulators

        kept = set(real)
        regulators = model_pole_regulators(raw)
        if regulators is not None:
            kept.update((regulators.uv, regulators.ir))
        for record in parameters:
            fields = record["fields"]
            if (
                fields.get("nature") != "external"
                or record["name"] in kept
                or card.value(record) != (1, 0)
            ):
                continue
            fields.update(nature="internal", type="real")
            fields.pop("lhablock", None)
            fields.pop("lhacode", None)
            _set_value(
                record, {"kind": "expression", "expression": "1.0"}, "restriction-one"
            )
            ones.append(record["name"])

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
        tuple(ones),
        MappingProxyType(lines),
    )
    payload["metadata"][METADATA_KEY] = scheme.metadata()
    payload["name"] += "_cms"
    return replace(scheme, derived=RawLoopModel.from_dict(payload))


# ------------------------------------------------------- the tree model view


def _ordered(parameters: list[Any], heads: Mapping[str, Any]) -> list[Any]:
    """``parameters`` in a dependency order (each after the parameters its
    expression reads), the source order kept where it already is one."""

    pending = list(parameters)
    placed: set[str] = set()
    ordered: list[Any] = []
    while pending:
        remaining = []
        for parameter in pending:
            expression = parameter.expression
            symbols = (
                frozenset()
                if expression is None
                else set(expression.get_all_symbols(False))
            )
            reads = {name for name, head in heads.items() if head in symbols} - {
                parameter.name
            }
            if reads <= placed:
                ordered.append(parameter)
                placed.add(parameter.name)
            else:
                remaining.append(parameter)
        if len(remaining) == len(pending):
            names = ", ".join(parameter.name for parameter in remaining)
            raise ValueError(f"cyclic complex-mass parameter definitions: {names}")
        pending = remaining
    return ordered


def tree_model_payload(payload: Mapping[str, Any], scheme: ComplexMassScheme):
    """The compiled model's normalized tree payload (``CompiledModel.model``,
    the loader's restricted view) under ``scheme``: the same steps on the
    loader's records (``complex_mass_scheme``), the internal parameter and
    coupling values evaluated again (``loading._canonical_loader_values``)."""

    import json

    from ufo_model_loader.common import JSONLook
    from ufo_model_loader.model import Model, Parameter, ParameterNature, ParameterType
    from ufo_model_loader.symbolica_processing import parse_python_expression_safe

    from pyamplicol.models.loading import _canonical_loader_values

    model = Model.from_json(json.dumps(dict(payload)))
    parameters = {parameter.name: parameter for parameter in model.parameters}
    if len(parameters) != len(model.parameters):
        raise ValueError("ambiguous tree model parameter names")
    raw = scheme.original
    heads = {
        name: symbol_registry.qualified(f"UFO::{name}")
        for name in (*parameters, *scheme.masses.values(), *scheme.lines.values())
    }
    renaming = [(heads[mass], heads[alias]) for mass, alias in scheme.masses.items()]

    def renamed(expression):
        for old, new in renaming:
            expression = expression.replace(old, new)
        return expression

    def source(name: str, derived: bool = False) -> str:
        model = scheme.derived if derived else raw
        record = model.record("parameters", name)
        return model.exact_value("parameters", record["id"])["expression"]

    if scheme.cms_parameter in parameters:
        parameter = parameters[scheme.cms_parameter]
        parameter.expression, parameter.value = None, complex(1.0)
    for name in scheme.internal_widths:
        parameter = parameters[name]
        parameter.expression = parse_python_expression_safe(f"re({source(name)})")
        parameter.value = None
    for name in scheme.yukawas:
        if name not in parameters:
            continue
        parameter = parameters[name]
        parameter.nature = ParameterNature.INTERNAL
        parameter.parameter_type = (
            ParameterType.COMPLEX if scheme.masses else ParameterType.REAL
        )
        parameter.lhablock = parameter.lhacode = None
        parameter.expression = parse_python_expression_safe(source(name, derived=True))
        parameter.value = None
    for name in scheme.ones:
        if name not in parameters:
            continue
        parameter = parameters[name]
        parameter.nature = ParameterNature.INTERNAL
        parameter.parameter_type = ParameterType.REAL
        parameter.lhablock = parameter.lhacode = None
        parameter.expression = parse_python_expression_safe("1.0")
        parameter.value = None
    if scheme.masses:
        kept = set(scheme.real) | set(scheme.yukawas) | set(scheme.ones)
        for parameter in model.parameters:
            if parameter.nature == ParameterNature.EXTERNAL or parameter.name in kept:
                continue
            parameter.parameter_type = ParameterType.COMPLEX
            if parameter.expression is not None:
                parameter.expression = renamed(parameter.expression)
        for mass, alias in scheme.masses.items():
            created = Parameter(
                None,
                None,
                alias,
                ParameterNature.INTERNAL,
                ParameterType.COMPLEX,
                None,
                pole_mass_expression(mass, scheme.widths[mass]),
            )
            model.parameters.append(created)
            parameters[alias] = created
        for coupling in model.couplings:
            if coupling.expression is not None:
                coupling.expression = renamed(coupling.expression)
        for propagator in model.propagators:
            propagator.numerator = renamed(propagator.numerator)
            propagator.denominator = renamed(propagator.denominator)
        for particle in model.particles:
            alias = scheme.masses.get(particle.mass.name)
            if alias is not None:
                particle.mass = parameters[alias]
                particle.width = parameters["ZERO"]
    for mass, alias in scheme.lines.items():
        created = Parameter(
            None, None, alias, ParameterNature.INTERNAL, ParameterType.REAL, None, mass
        )
        model.parameters.append(created)
        parameters[alias] = created
    for particle in model.particles:
        alias = scheme.lines.get(particle.mass.name)
        if alias is not None:
            particle.mass = parameters[alias]
    model.parameters = _ordered(model.parameters, heads)
    model.name_to_position["parameters"] = {
        parameter.name: index for index, parameter in enumerate(model.parameters)
    }
    _canonical_loader_values(model)
    result = json.loads(model.to_json(JSONLook.COMPACT))
    result.pop("raw_ufo", None)
    result.pop("raw_ufo_error", None)
    for key, value in payload.items():
        if key not in result:
            result[key] = value
    for old, new in zip(
        payload.get("propagators", ()), result["propagators"], strict=True
    ):
        for key, value in old.items():
            new.setdefault(key, value)
    return result


def complex_mass_model(
    compiled: Any,
    *,
    fixed_parameters: Mapping[str, Any] | None = None,
    reglog_convention: str = "ufo-standard",
    cache_dir: Any = None,
):
    """``compiled`` (a ``CompiledModel`` with its raw sidecar) in the
    complex-mass scheme: the raw sidecar transformed
    (``complex_mass_scheme``), the tree payload with the same decisions
    (``tree_model_payload``) compiled again (``compile_counterterm_model_ir``'s
    content-addressed cache), the external defaults without the former
    Yukawa inputs, the scheme recorded in the source options. Returns the
    model and its ``ComplexMassScheme``."""

    from dataclasses import replace

    from pyamplicol.models.loading import _counterterm_ir

    raw = compiled.raw_loop_model
    if raw is None:
        raise ValueError("the complex-mass scheme needs the model's raw UFO records")
    scheme = complex_mass_scheme(
        raw, fixed_parameters=fixed_parameters, reglog_convention=reglog_convention
    )
    payload = tree_model_payload(compiled.model, scheme)
    ir, _entry = _counterterm_ir(
        payload,
        tree_kernel_variant="transverse",
        u1_projected_vertices=frozenset(),
        cache_dir=cache_dir,
        gauge=compiled.ir.gauge,
    )
    external = {
        parameter["name"]
        for parameter in payload["parameters"]
        if parameter["nature"] == "external"
    }
    options = dict(compiled.source.get("options", {}))
    options[METADATA_KEY] = scheme.metadata()
    transformed = replace(
        compiled,
        source={**compiled.source, "options": options},
        model=payload,
        ir=ir,
        parameter_defaults={
            name: value
            for name, value in compiled.parameter_defaults.items()
            if name in external
        },
        raw_loop_model=scheme.derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    return transformed, scheme


# --------------------------------------------------------- the request policy


def require_mass_scheme(value: str) -> str:
    """``value``, a loop model declaration's ``mass_scheme``, if it names a
    scheme v2 serves (``MASS_SCHEMES``), else refused (``mass-scheme``): a
    misspelt complex-mass request must never be computed real on-shell."""

    if value not in MASS_SCHEMES:
        raise ComplexMassSchemeUnsupported(
            "mass-scheme",
            f"the loop model declares mass_scheme = {value!r}, which v2 does not "
            "implement",
        )
    return value


def scheme_metadata(compiled: Any) -> Mapping[str, Any] | None:
    """The complex-mass scheme of a model ``complex_mass_model`` derived
    (``ComplexMassScheme.metadata``, recorded in its source options), else
    None (a real on-shell model)."""

    source = getattr(compiled, "source", None) or {}
    return source.get("options", {}).get(METADATA_KEY)


def require_real_fixed_parameters(
    fixed: Mapping[str, tuple[Any, Any]],
) -> dict[str, str]:
    """The real parts of ``fixed`` (``[loop.preparation] fixed_parameters``:
    name -> (real, imaginary)); a fixed parameter with an imaginary part is
    refused (``complex-fixed-parameter``) in either mass scheme: external
    parameters are real (MG5_aMC's too), and the complex-mass scheme derives
    its complex pole masses from real masses and widths
    (``complex_mass_scheme``)."""

    complex_parameters = sorted(
        name for name, (_real, imaginary) in fixed.items() if Decimal(imaginary) != 0
    )
    if complex_parameters:
        raise ComplexMassSchemeUnsupported(
            "complex-fixed-parameter",
            "v2 virtuals take real fixed parameters; "
            + ", ".join(repr(name) for name in complex_parameters)
            + " complex",
        )
    return {name: str(real) for name, (real, _imaginary) in fixed.items()}


def apply_mass_scheme(
    compiled: Any,
    mass_scheme: str,
    *,
    fixed_parameters: Mapping[str, Any] | None = None,
    cache_dir: Any = None,
) -> Any:
    """The model a request of ``mass_scheme`` (``require_mass_scheme``) is
    generated on: ``compiled`` itself in the real on-shell scheme, MG5_aMC's
    complex-mass model of it (``complex_mass_model``, at the card of the
    request's real ``fixed_parameters``) in the complex-mass scheme. The
    transform's refusals are typed (``cms-internal-mass``,
    ``cms-yukawa-input``, ``complex-mass-scheme``)."""

    transformed = scheme_metadata(compiled) is not None
    if require_mass_scheme(mass_scheme) == REAL_ON_SHELL:
        if transformed:
            raise ComplexMassSchemeUnsupported(
                "mass-scheme",
                "the request declares the real on-shell scheme on a model in the "
                "complex-mass scheme",
            )
        return compiled
    if transformed:
        return compiled
    if compiled.raw_loop_model is None:
        raise ComplexMassSchemeUnsupported(
            "complex-mass-scheme",
            "the model carries no raw UFO records to transform",
        )
    model, _scheme = complex_mass_model(
        compiled, fixed_parameters=fixed_parameters, cache_dir=cache_dir
    )
    return model


def require_stable_external_states(compiled: Any, pdgs: Any, process: str) -> None:
    """Refuse (``cms-external-width``) a complex-mass request with an
    external particle of a complex pole mass (a nonzero width at the card,
    ``ComplexMassScheme.propagators``, particle or antiparticle): external
    states are stable (requirements EW-25); MG5_aMC computes them with the
    complex mass, which no on-shell external state has."""

    record = scheme_metadata(compiled)
    if not record:
        return
    propagators = {int(pdg): mass for pdg, mass in record["propagators"].items()}
    unstable = sorted(
        {int(pdg) for pdg in pdgs if int(pdg) in propagators},
        key=lambda pdg: (abs(pdg), pdg),
    )
    if unstable:
        named = ", ".join(f"PDG {pdg} ({propagators[pdg]})" for pdg in unstable)
        widths = sorted(
            {
                record["widths"][m.removeprefix(CMASS_PREFIX)]
                for m in (propagators[pdg] for pdg in unstable)
            }
        )
        raise ComplexMassSchemeUnsupported(
            "cms-external-width",
            f"{process}: external particles with a complex pole mass under the "
            f"complex-mass scheme: {named}; fix their widths to zero in "
            "[loop.preparation] fixed_parameters ("
            + ", ".join(f'{width} = "0"' for width in widths)
            + ")",
        )


def require_native_pole_masses(compiled: Any, process: str) -> None:
    """Refuse (``complex-mass-scheme``) a public-route request with complex
    pole masses while the native point inputs of stored artifacts take none
    (``NATIVE_POLE_MASSES``); ``LoopSlice.generate`` with Python inputs
    serves it meanwhile."""

    record = scheme_metadata(compiled)
    if record and record["masses"] and not NATIVE_POLE_MASSES:
        raise ComplexMassSchemeUnsupported(
            "complex-mass-scheme",
            f"{process}: the native point inputs of stored artifacts take no "
            "complex pole masses in this build (M-SM MS-CMS-N); "
            "loops.v2.assemble.LoopSlice.generate on model_cms.complex_mass_model "
            "evaluates the request with Python inputs",
        )


def convention_record(compiled: Any) -> dict[str, Any] | None:
    """The complex-mass conventions of a request's virtual (m-sm PLAN §1.2),
    None for a real on-shell model: the pole masses and the widths they
    take, the internal widths' real parts, the Yukawa ties, the reglog
    convention, and the values of the model's NLOCT convention parameters
    (``CONVENTION_PARAMETERS`` it declares: CMSParam, LogSwitch,
    GfRedefinitionChoice) as the derived model defines them."""

    record = scheme_metadata(compiled)
    if record is None:
        return None
    raw = compiled.raw_loop_model
    conventions: dict[str, str] = {}
    if raw is not None:
        named = _named(raw, "parameters")
        card = None
        for name in CONVENTION_PARAMETERS:
            parameter = named.get(name)
            if parameter is None:
                continue
            if parameter["fields"].get("nature") == "external":
                card = card or _Card(raw, {})
                real, imaginary = card.value(parameter)
                conventions[name] = (
                    str(real) if imaginary == 0 else f"({real},{imaginary})"
                )
            else:
                value = raw.exact_value("parameters", parameter["id"])
                conventions[name] = str(value.get("expression", value))
    return {
        "pole_masses": dict(record["masses"]),
        "widths": dict(record["widths"]),
        "internal_widths": list(record["internal_widths"]),
        "yukawas": dict(record["yukawas"]),
        "reglog_convention": record["reglog_convention"],
        "parameters": conventions,
    }
