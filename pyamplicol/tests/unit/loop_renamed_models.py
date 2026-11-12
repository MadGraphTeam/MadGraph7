# SPDX-License-Identifier: 0BSD
"""Renamed and reparametrised copies of NLO UFO models (test support).

The genericity harness of the 2026-10-25 audit
(``workspace/reports/engine-v2/genericity-audit/REPORT.md`` §3): a production
decision must not depend on how a model spells or numbers its content. Each
variant rewrites one aspect of a source UFO directory and returns the map
from the original spelling to the variant's, so a test can generate the same
request on both and compare decisions under the map and values to rounding:

* ``orders``: coupling orders renamed (``QCD`` -> ``QS``, ``QED`` -> ``Qe``)
  plus an unused mixed-case order ``NPprop``;
* ``lower``: coupling orders spelt in lower case (``QCD`` -> ``qcd``, ``QED``
  -> ``qed``), no other change;
* ``particles``: particles renamed (gluon -> ``gl``, Goldstones -> ``Gp``,
  ``Gm``, ``Gz``, ghosts), the W mass and width parameters renamed (``MW`` ->
  ``MWB``, ``WW`` -> ``WWB``) and BSM-range PDG codes renumbered (Z 23 ->
  9000023, G0 250 -> 9000250, G+ 251 -> 9000251, in the particles, their
  ``loop_particles`` and their MASS/DECAY codes);
* ``scale``: ``MU_R`` -> ``muR`` (same LOOP code) and ``aS`` moved off
  SMINPUTS 3 (to block ``GSINPUT`` 1);
* ``mass``: ``MW`` turned internal (an alias of a new external ``MWIN`` at
  ``WMASSIN`` 1, same value) where it is a card input, or external (MASS 24,
  the value it derives from the default card) where it is derived;
* ``quarks``: a quark flavour renamed and renumbered (``u`` -> ``uq``, PDG 2
  -> 9000002, in the particles and their ``loop_particles``), so identical-
  flavour lines are those of a quark no PDG range or name names;
* ``ckm``: the CKM elements and their Wolfenstein inputs renamed (``CKM13``
  -> ``VQ13``, ``lamWS`` -> ``wlam``, ...) and every vertex renamed
  (``V_101`` -> ``VX_101``), so vertices that a restriction removes are
  those of couplings no parameter or vertex name names.

The rewrite is textual and checked: every substitution must apply, so a
changed source fails loudly instead of yielding an unrenamed "variant". Run
as a script to write the variants of one model:
``python tests/unit/loop_renamed_models.py <ufo-dir> <output-dir>``.
"""

from __future__ import annotations

import cmath
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

VARIANTS = ("orders", "lower", "particles", "scale", "mass", "quarks", "ckm")


@dataclass(frozen=True)
class RenameMap:
    """Original -> variant spellings of one variant (identity when absent)."""

    variant: str
    orders: Mapping[str, str] = field(default_factory=dict)
    particles: Mapping[str, str] = field(default_factory=dict)
    pdgs: Mapping[int, int] = field(default_factory=dict)
    parameters: Mapping[str, str] = field(default_factory=dict)
    extra_orders: tuple[str, ...] = ()

    def order(self, name: str) -> str:
        return self.orders.get(name, name)

    def particle(self, name: str) -> str:
        return self.particles.get(name, name)

    def pdg(self, code: int) -> int:
        sign = -1 if code < 0 else 1
        return sign * self.pdgs.get(abs(code), abs(code))

    def parameter(self, name: str) -> str:
        return self.parameters.get(name, name)

    def process(self, expression: str) -> str:
        return " ".join(
            token if token == ">" else self.particle(token)
            for token in expression.split()
        )

    def orders_of(self, orders: Mapping[str, object]) -> dict[str, object]:
        return {self.order(name): value for name, value in orders.items()}


# ----------------------------------------------------------------- rewriting


def _sub(pattern: str, replacement, text: str, *, what: str, count: int = 0) -> str:
    result, made = re.subn(pattern, replacement, text, count=count)
    if not made:
        raise ValueError(f"renamed-model rewrite found no {what}")
    return result


def _bracket_span(text: str, start: int) -> int:
    """End (exclusive) of the bracketed expression opening at ``start``."""

    depth = 0
    for position in range(start, len(text)):
        if text[position] == "[":
            depth += 1
        elif text[position] == "]":
            depth -= 1
            if depth == 0:
                return position + 1
    raise ValueError("unbalanced loop_particles list")


def _renumber_loop_particles(text: str, pdgs: Mapping[int, int]) -> str:
    out, position = [], 0
    for match in re.finditer(r"loop_particles\s*=\s*(?=\[)", text):
        start = match.end()
        if start < position:
            continue
        end = _bracket_span(text, start)
        out.append(text[position:start])
        out.append(
            re.sub(
                r"-?\b\d+\b",
                lambda m: str(
                    (-1 if m.group(0).startswith("-") else 1)
                    * pdgs.get(abs(int(m.group(0))), abs(int(m.group(0))))
                ),
                text[start:end],
            )
        )
        position = end
    out.append(text[position:])
    return "".join(out)


def _parameter_span(text: str, name: str) -> tuple[int, int]:
    match = re.search(
        rf"^{re.escape(name)}\s*=\s*Parameter\(name\s*=\s*'{re.escape(name)}'",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"renamed-model rewrite found no parameter {name!r}")
    depth, position = 0, match.start()
    for position in range(text.index("(", match.start()), len(text)):
        if text[position] == "(":
            depth += 1
        elif text[position] == ")":
            depth -= 1
            if depth == 0:
                break
    return match.start(), position + 1


def _parameter_field(definition: str, key: str) -> str | None:
    match = re.search(rf"\b{key}\s*=\s*('[^']*'|\[[^\]]*\]|[^,)\s]+)", definition)
    return None if match is None else match.group(1)


class _Card:
    """A minimal SLHA restriction card editor (blocks and DECAY lines)."""

    def __init__(self, text: str):
        self.lines = text.splitlines()

    def _locate(self, block: str, code: int) -> int | None:
        current = None
        for index, line in enumerate(self.lines):
            words = line.split("#", 1)[0].split()
            if not words:
                continue
            head = words[0].upper()
            if head == "BLOCK":
                current = words[1].upper()
                continue
            if head == "DECAY":
                current = None
                if block.upper() == "DECAY" and int(words[1]) == code:
                    return index
                continue
            if current == block.upper() and int(words[0]) == code:
                return index
        return None

    def value(self, block: str, code: int) -> str | None:
        index = self._locate(block, code)
        if index is None:
            return None
        words = self.lines[index].split("#", 1)[0].split()
        return words[2] if block.upper() == "DECAY" else words[1]

    def remove(self, block: str, code: int) -> str | None:
        value = self.value(block, code)
        index = self._locate(block, code)
        if index is not None:
            del self.lines[index]
        return value

    def add(self, block: str, code: int, value: str, comment: str) -> None:
        if block.upper() == "DECAY":
            self.lines.append(f"DECAY {code} {value}")
            return
        for index, line in enumerate(self.lines):
            words = line.split()
            if (
                len(words) >= 2
                and words[0].upper() == "BLOCK"
                and words[1].upper() == block.upper()
            ):
                self.lines.insert(index + 1, f"  {code} {value} # {comment}")
                return
        self.lines.extend(["", f"Block {block}", f"  {code} {value} # {comment}"])

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _move_parameter(files, card, name, block, code) -> None:
    text = files["parameters.py"]
    start, end = _parameter_span(text, name)
    definition = text[start:end]
    old_block = _parameter_field(definition, "lhablock").strip("'")
    old_code = int(_parameter_field(definition, "lhacode").strip("[] "))
    definition = re.sub(r"lhablock\s*=\s*'[^']*'", f"lhablock = '{block}'", definition)
    definition = re.sub(
        r"lhacode\s*=\s*\[[^\]]*\]", f"lhacode = [ {code} ]", definition
    )
    files["parameters.py"] = text[:start] + definition + text[end:]
    if card is not None:
        value = card.remove(old_block, old_code)
        if value is not None:
            card.add(block, code, value, name)


def _numeric(text: str) -> float:
    return float(text.replace("d", "e").replace("D", "e"))


def _derived_mw(files, card) -> float:
    """loop_sm's MW from its default-card inputs (its own formula)."""

    def value(name):
        definition = files["parameters.py"][
            slice(*_parameter_span(files["parameters.py"], name))
        ]
        block = _parameter_field(definition, "lhablock").strip("'")
        code = int(_parameter_field(definition, "lhacode").strip("[] "))
        found = None if card is None else card.value(block, code)
        return _numeric(
            found if found is not None else _parameter_field(definition, "value")
        )

    mz, gf = value("MZ"), value("Gf")
    aew = 1.0 / value("aEWM1")
    mw = cmath.sqrt(
        mz**2 / 2.0
        + cmath.sqrt(mz**4 / 4.0 - (aew * cmath.pi * mz**2) / (gf * cmath.sqrt(2)))
    )
    return mw.real


def write_variant(source: Path, target: Path, variant: str) -> RenameMap:
    """Write ``variant`` of the UFO directory ``source`` to ``target``."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown renamed-model variant {variant!r}")
    source, target = Path(source), Path(target)
    target.mkdir(parents=True)  # a new directory: never overwrite
    files: dict[str, str] = {}
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix == ".py":
            # latin-1 round-trips every byte (some UFOs are not UTF-8).
            files[path.name] = path.read_text(encoding="latin-1")
    card_path = source / "restrict_default.dat"
    card = _Card(card_path.read_text()) if card_path.is_file() else None

    names = {
        m.group(1)
        for m in re.finditer(r"\b(?:anti)?name\s*=\s*'([^']*)'", files["particles.py"])
    }
    rename = RenameMap(variant)
    if variant in ("orders", "lower"):
        orders = (
            {"QCD": "QS", "QED": "Qe"}
            if variant == "orders"
            else {"QCD": "qcd", "QED": "qed"}
        )
        for path in [p for p in files if p != "coupling_orders.py"]:
            for old, new in orders.items():
                files[path] = re.sub(
                    rf"(['\"]){old}\1", rf"\g<1>{new}\g<1>", files[path]
                )
        text = files["coupling_orders.py"]
        for old, new in orders.items():
            text = _sub(rf"name\s*=\s*'{old}'", f"name = '{new}'", text, what=old)
        extra: tuple[str, ...] = ()
        if variant == "orders":
            extra = ("NPprop",)
            text += (
                "\nNPprop = CouplingOrder(name = 'NPprop',\n"
                "                       expansion_order = 99,\n"
                "                       hierarchy = 3)\n"
            )
            # UFO counterterm keys are (power of each declared order...,
            # loop group): the appended order takes a zero before the group
            # index.
            files["particles.py"] = _sub(
                r"\((\d+(?:\s*,\s*\d+)*)\s*,\s*(\d+)\)(\s*:)",
                r"(\g<1>,0,\g<2>)\g<3>",
                files["particles.py"],
                what="counterterm keys",
            )
        files["coupling_orders.py"] = text
        rename = RenameMap(variant, orders=orders, extra_orders=extra)
    elif variant == "particles":
        candidates = {
            "G": "gl",
            "g": "gl",
            "G+": "Gp",
            "G-": "Gm",
            "G0": "Gz",
            "gh": "ghost",
            "gh~": "ghost~",
            "ghG": "ghgl",
            "ghG~": "ghgl~",
        }
        particles = {old: new for old, new in candidates.items() if old in names}
        text = files["particles.py"]
        for old, new in particles.items():
            text = _sub(
                rf"\b(name|antiname)(\s*=\s*)'{re.escape(old)}'",
                rf"\g<1>\g<2>'{new}'",
                text,
                what=f"particle {old!r}",
            )
        pdgs = {23: 9000023, 250: 9000250, 251: 9000251}
        for old, new in pdgs.items():
            text = _sub(
                rf"pdg_code\s*=\s*{old}\b", f"pdg_code = {new}", text, what=str(old)
            )
        files["particles.py"] = text
        for path in files:
            files[path] = _renumber_loop_particles(files[path], pdgs)
        parameters = {"MW": "MWB", "WW": "WWB"}
        for path in files:
            for old, new in parameters.items():
                files[path] = re.sub(rf"\b{old}\b", new, files[path])
        text = files["parameters.py"]
        for name in ("MZ", "WZ"):
            start, end = _parameter_span(text, name)
            definition = text[start:end]
            block = _parameter_field(definition, "lhablock").strip("'")
            definition = _sub(
                r"lhacode\s*=\s*\[\s*23\s*\]",
                "lhacode = [ 9000023 ]",
                definition,
                what=f"{name} code",
            )
            text = text[:start] + definition + text[end:]
            if card is not None:
                value = card.remove(block, 23)
                if value is not None:
                    card.add(block, 9000023, value, name)
        files["parameters.py"] = text
        rename = RenameMap(
            variant, particles=particles, pdgs=pdgs, parameters=parameters
        )
    elif variant == "quarks":
        particles = {"u": "uq", "u~": "uq~"}
        text = files["particles.py"]
        for old, new in particles.items():
            text = _sub(
                rf"\b(name|antiname)(\s*=\s*)'{re.escape(old)}'",
                rf"\g<1>\g<2>'{new}'",
                text,
                what=f"particle {old!r}",
            )
        pdgs = {2: 9000002}
        for old, new in pdgs.items():
            text = _sub(
                rf"pdg_code\s*=\s*{old}\b", f"pdg_code = {new}", text, what=str(old)
            )
        files["particles.py"] = text
        for path in files:
            files[path] = _renumber_loop_particles(files[path], pdgs)
        rename = RenameMap(variant, particles=particles, pdgs=pdgs)
    elif variant == "ckm":
        candidates = {f"CKM{i}{j}": f"VQ{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)}
        candidates.update(lamWS="wlam", AWS="wA", rhoWS="wrho", etaWS="weta")
        parameters = {
            old: new
            for old, new in candidates.items()
            if re.search(rf"\bname\s*=\s*'{old}'", files["parameters.py"])
        }
        if not parameters:
            raise ValueError("renamed-model rewrite found no CKM parameter")
        for path in files:
            for old, new in parameters.items():
                files[path] = re.sub(rf"\b{old}\b", new, files[path])
        files["vertices.py"] = _sub(
            r"\bname(\s*=\s*)'V_(\d+)'",
            r"name\g<1>'VX_\g<2>'",
            files["vertices.py"],
            what="vertex names",
        )
        rename = RenameMap(variant, parameters=parameters)
    elif variant == "scale":
        parameters = {"MU_R": "muR"}
        for path in files:
            files[path] = re.sub(r"\bMU_R\b", "muR", files[path])
        _move_parameter(files, card, "aS", "GSINPUT", 1)
        rename = RenameMap(variant, parameters=parameters)
    else:  # mass
        text = files["parameters.py"]
        start, end = _parameter_span(text, "MW")
        definition = text[start:end]
        nature = _parameter_field(definition, "nature").strip("'")
        if nature == "external":
            block = _parameter_field(definition, "lhablock").strip("'")
            code = int(_parameter_field(definition, "lhacode").strip("[] "))
            value = None if card is None else card.remove(block, code)
            value = value or _parameter_field(definition, "value")
            replacement = (
                "MWIN = Parameter(name = 'MWIN',\n"
                "                 nature = 'external',\n"
                "                 type = 'real',\n"
                f"                 value = {_numeric(value)!r},\n"
                "                 texname = '\\\\text{MWIN}',\n"
                "                 lhablock = 'WMASSIN',\n"
                "                 lhacode = [ 1 ])\n\n"
                "MW = Parameter(name = 'MW',\n"
                "               nature = 'internal',\n"
                "               type = 'real',\n"
                "               value = 'MWIN',\n"
                "               texname = '\\\\text{MW}')"
            )
            if card is not None:
                card.add("WMASSIN", 1, value, "MWIN")
        else:
            mw = _derived_mw(files, card)
            replacement = (
                "MW = Parameter(name = 'MW',\n"
                "               nature = 'external',\n"
                "               type = 'real',\n"
                f"               value = {mw!r},\n"
                "               texname = 'M_W',\n"
                "               lhablock = 'MASS',\n"
                "               lhacode = [ 24 ])"
            )
            if card is not None:
                card.add("MASS", 24, repr(mw), "MW")
            # An external parameter precedes every internal one.
            text = text[:start] + text[end:]
            anchor = _parameter_span(text, "MZ")[1]
            text = text[:anchor] + "\n\n" + replacement + text[anchor:]
            files["parameters.py"] = text
            replacement = None
        if replacement is not None:
            files["parameters.py"] = text[:start] + replacement + text[end:]
    for name, text in files.items():
        (target / name).write_text(text, encoding="latin-1")
    if card is not None:
        (target / "restrict_default.dat").write_text(card.text())
    return rename


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    source, output = Path(argv[0]), Path(argv[1])
    for variant in VARIANTS:
        rename = write_variant(source, output / f"{source.name}_{variant}", variant)
        print(variant, rename)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
