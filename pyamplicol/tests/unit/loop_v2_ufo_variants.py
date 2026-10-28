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
