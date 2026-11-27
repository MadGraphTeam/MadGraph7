# SPDX-License-Identifier: 0BSD
"""tools/developer/fold_ufo_value_appends.py: module-level string appends to
UFO Laurent terms folded into their constructors (the shipped G_mu model's
charge counterterms, UFOmodeldb issue-1 patch)."""

from __future__ import annotations

import pytest

from tools.developer.fold_ufo_value_appends import fold_module

SOURCE = """A = CTParameter(name = 'A',
              value = {-1:'a', 0:'b'})

B = CTParameter(name = 'B',
              value = {0:'c' + '+' + A.value[0]})

# extend B
B.value[0] += '-0.5*(' + A.value[-1] + ')'
B.value[0] += ('+'
               + A.value[0])
"""


def _values(text: str) -> dict:
    class CTParameter:
        def __init__(self, name, value):
            self.name, self.value = name, value

    namespace = {"CTParameter": CTParameter}
    exec(compile(text, "<ufo>", "exec"), namespace)
    return {k: dict(v.value) for k, v in namespace.items() if k in ("A", "B")}


def test_appends_fold_into_the_constructor_with_the_same_strings():
    folded, log = fold_module(SOURCE)
    assert len(log) == 2
    assert _values(folded) == _values(SOURCE)
    assert "+=" not in "".join(
        line for line in folded.splitlines() if not line.startswith("#")
    )
    assert folded.count("# (folded) ") == 3  # the statements' lines, kept
    assert fold_module(folded) == (folded, [])


@pytest.mark.parametrize(
    "statement, match",
    [
        ("B.value[0] = 1.5\n", "not a string append"),
        ("B.value[0] += str(2)\n", "not a string composition"),
        (
            "D = CTParameter(name = 'D', value = {0:'d'})\nB.value[0] += D.value[0]\n",
            "not bound before",
        ),
        ("A.value[0] += 'x'\n", "itself extended"),
        ("C = 1\nB.value[0] += 'x'; C = 2\n", "shares its lines"),
    ],
)
def test_other_value_writes_are_refused(statement, match):
    with pytest.raises(ValueError, match=match):
        fold_module(SOURCE + statement)
