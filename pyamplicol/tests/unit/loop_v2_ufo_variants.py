# SPDX-License-Identifier: 0BSD
"""UFO variants for the genericity checks of the loop-v2 tests: a model whose
external mass parameter is made internal with the same value (genericity
audit M2; ``test_loop_v2_onshell.py``, ``test_loop_v2_ew.py``), a model
whose strong coupling runs with the renormalisation scale (genericity audit
2 m3; ``test_loop_v2_sdk_parity.py``) and one whose colour-scheme constant is
an external parameter (genericity audit 2 M2; ``test_loop_v2_sdk_parity.py``).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


def running_coupling_ufo(source: Path, destination: Path) -> Path:
    """A copy of loop_sm whose strong coupling runs with the renormalisation
    scale (a toy running-coupling model, genericity audit 2 m3): ``G`` gains
    the factor ``1 + aS (MU_R - MZ) / (4 pi MZ)`` (algebraic, so the model's
    derived parameters need no helper function), so the Born couplings
    (``GC_10``, ``GC_11``, ...) depend on ``MU_R`` and the points of a batch
    of several scales no longer share them. Pickled model caches are not
    copied (they would shadow the edit)."""

    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns("*.pkl", "__pycache__")
    )
    path = destination / "parameters.py"
    text = path.read_text()
    old = "value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',"
    assert text.count(old) == 1, "loop_sm's G definition"
    path.write_text(
        text.replace(
            old,
            "value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)"
            "*(1 + aS*(MU_R - MZ)/(4*cmath.pi*MZ))',",
        )
    )
    return destination


def external_scheme_constant_ufo(source: Path, destination: Path) -> Path:
    """A copy of loop_sm whose colour-scheme constant ``lhv`` (internal,
    ``1.0``, the 't Hooft-Veltman flag its R2 and UV counterterms read) is an
    external parameter of the same value (block ``HVSCHEME``, code 1): a
    colour declaration fixes it at generation
    (``source_color_fixed_parameters``), so a runtime must refuse it at
    another value (genericity audit 2, M2). Pickled model caches are not
    copied (they would shadow the edit)."""

    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns("*.pkl", "__pycache__")
    )
    path = destination / "parameters.py"
    text = path.read_text()
    old = (
        "lhv = Parameter(name = 'lhv',\n"
        "                 nature = 'internal',\n"
        "                 type = 'real',\n"
        "                 value = '1.0',\n"
    )
    assert text.count(old) == 1, "loop_sm's lhv definition"
    path.write_text(
        text.replace(
            old,
            "lhv = Parameter(name = 'lhv',\n"
            "                 nature = 'external',\n"
            "                 type = 'real',\n"
            "                 value = 1.0,\n"
            "                 lhablock = 'HVSCHEME',\n"
            "                 lhacode = [ 1 ],\n",
        )
    )
    return destination


def internal_mass_ufo(source: Path, destination: Path, name: str) -> Path:
    """A copy of a UFO whose external mass parameter ``name`` is internal:
    ``name = {name}x``, a new external parameter in its place (same block,
    code and value), so every value is unchanged. Pickled model caches are
    not copied (they would shadow the edit)."""

    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns("*.pkl", "__pycache__")
    )
    path = destination / "parameters.py"
    text = path.read_text()
    match = re.search(
        rf"^{name} = Parameter\(name = '{name}',\n"
        r"(?:.*\n)*?.*lhacode = \[ *\d+ *\]\)\n",
        text,
        re.MULTILINE,
    )
    assert match is not None, name
    block = match.group(0)
    external = block.replace(
        f"{name} = Parameter(name = '{name}'", f"{name}x = Parameter(name = '{name}x'"
    )
    internal = (
        f"{name} = Parameter(name = '{name}',\n"
        "               nature = 'internal',\n"
        "               type = 'real',\n"
        f"               value = '{name}x',\n"
        f"               texname = '{name}')\n"
    )
    path.write_text(text.replace(block, external + "\n" + internal))
    return destination


# The leptoquark of ``leptoquark_ufo``: a colour-triplet scalar of charge 5/3
# and lepton number -1 (BSM-range PDG code), coupled by one Yukawa vertex to
# an up quark and an electron.
LEPTOQUARK_PDG = 9000005


def leptoquark_ufo(source: Path, destination: Path) -> Path:
    """A copy of a UFO with one scalar leptoquark ``lq`` (colour triplet,
    charge 5/3, lepton number -1, mass MLQ = 1000) and its Yukawa vertices
    u~ e- lq and e+ u lq~ (``V_LQ``, ``V_LQc``; coupling i yLQ of order QED):
    a fermion line that changes colour representation, from a quark to a
    lepton (genericity audit 2, G2-m1). Nothing else of the model changes;
    pickled model caches are not copied."""

    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns("*.pkl", "__pycache__")
    )

    def append(name: str, text: str) -> None:
        path = destination / name
        path.write_text(path.read_text() + "\n" + text)

    append(
        "parameters.py",
        "MLQ = Parameter(name = 'MLQ', nature = 'external', type = 'real', "
        "value = 1000., texname = 'MLQ', lhablock = 'MASS', "
        f"lhacode = [ {LEPTOQUARK_PDG} ])\n\n"
        "yLQ = Parameter(name = 'yLQ', nature = 'external', type = 'real', "
        "value = 0.5, texname = 'yLQ', lhablock = 'LQYUKAWA', lhacode = [ 1 ])\n",
    )
    append(
        "particles.py",
        f"lq = Particle(pdg_code = {LEPTOQUARK_PDG}, name = 'lq', antiname = 'lq~', "
        "spin = 1, color = 3, mass = Param.MLQ, width = Param.ZERO, "
        "texname = 'lq', antitexname = 'lq~', charge = 5/3, LeptonNumber = -1, "
        "GhostNumber = 0)\n\nlq__tilde__ = lq.anti()\n",
    )
    append(
        "couplings.py",
        "GC_LQ = Coupling(name = 'GC_LQ', value = 'complex(0,1)*yLQ', "
        "order = {'QED':1})\n",
    )
    append(
        "vertices.py",
        "V_LQ = Vertex(name = 'V_LQ', particles = [ P.u__tilde__, P.e__minus__, "
        "P.lq ], color = [ 'Identity(1,3)' ], lorentz = [ L.FFS1 ], "
        "couplings = {(0,0):C.GC_LQ})\n\n"
        "V_LQc = Vertex(name = 'V_LQc', particles = [ P.e__plus__, P.u, "
        "P.lq__tilde__ ], color = [ 'Identity(2,3)' ], lorentz = [ L.FFS1 ], "
        "couplings = {(0,0):C.GC_LQ})\n",
    )
    return destination
