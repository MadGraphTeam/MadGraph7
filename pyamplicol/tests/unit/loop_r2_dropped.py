# SPDX-License-Identifier: 0BSD
"""A test-only UFO defect shared by the loop-induced probe and ``pyamplicol
check`` tests: a copy of a UFO without one R2 counterterm vertex, selected
by the PDG codes of its particles from the model's own files
(``drop_r2_vertex``)."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path


def pdg_codes(ufo: Path) -> dict[str, int]:
    """The model's particle objects (``particles.py`` names) -> PDG code,
    antiparticles (``X.anti()``) negative."""

    out: dict[str, int] = {}
    for node in ast.parse((ufo / "particles.py").read_text()).body:
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        name = node.targets[0].id
        call = node.value
        if isinstance(call.func, ast.Attribute) and call.func.attr == "anti":
            out[name] = -out[call.func.value.id]
            continue
        for keyword in call.keywords:
            if keyword.arg == "pdg_code":
                out[name] = keyword.value.value
    return out


def drop_r2_vertex(ufo: Path, pdgs: list[int]) -> str:
    """Remove from the UFO copy ``ufo`` the one R2 counterterm vertex whose
    particles are the multiset ``pdgs`` (read from the model's own files);
    returns its name."""

    codes = pdg_codes(ufo)
    path = ufo / "CT_vertices.py"
    text = path.read_text()
    found = []
    for node in ast.parse(text).body:
        call = getattr(node, "value", None)
        if not (
            isinstance(node, ast.Assign)
            and isinstance(call, ast.Call)
            and getattr(call.func, "id", None) == "CTVertex"
        ):
            continue
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        kind = keywords.get("type")
        if not (isinstance(kind, ast.Constant) and kind.value == "R2"):
            continue
        particles = Counter(codes[e.attr] for e in keywords["particles"].elts)
        if particles == Counter(pdgs):
            found.append(node)
    (node,) = found
    lines = text.splitlines(keepends=True)
    path.write_text("".join(lines[: node.lineno - 1] + lines[node.end_lineno :]))
    return node.targets[0].id


def declaration(source: str) -> str:
    """A UFO by path is not the shipped loop_sm (its digest differs when
    edited): the card states its declaration, as a user does for a model of
    their own; here the user's claim that its R2 rules are complete."""

    if source == "loop_sm":
        return ""
    return """[loop.model]
loop_coupling_orders = ["QCD"]
uv_complete = true
model_r2_complete = true
[loop.model.scheme]
dimensional_regularization = "HV"
external_states = "physical-four-dimensional"
gamma5 = "source-UFO-four-dimensional-prescription"
renormalization = "source-UFO mass-dependent decoupling"
mass_scheme = "real-on-shell"
finite_restoration = "source-UFO-local-rules"
model_convention = "loop_sm lhv=1 supplied QCD rules"
"""
