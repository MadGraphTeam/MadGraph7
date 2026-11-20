# SPDX-License-Identifier: 0BSD
"""A tool run as a script executes its ``if __name__ == "__main__":`` block
when the interpreter reaches it, so a function or class defined after that
block does not exist yet when ``main()`` runs: every command that calls it
fails with ``NameError`` while the tests, which import the module and call
``main()``, pass (``tools/loop_v2_generate.py``'s ``poles`` and ``validate
--inputs native`` did, M-FINAL MF-8G-1 fix round). Every tool therefore
defines its functions and classes before its last main guard (an earlier
guard, as the developer gates' source-only bootstrap, is fine)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    names = [test.left, *test.comparators]
    return any(isinstance(n, ast.Name) and n.id == "__name__" for n in names) and any(
        isinstance(n, ast.Constant) and n.value == "__main__" for n in names
    )


def _late_definitions(source: str) -> list[str]:
    body = ast.parse(source).body
    guards = [k for k, node in enumerate(body) if _is_main_guard(node)]
    if not guards:
        return []
    return [
        f"{node.name} (line {node.lineno})"
        for node in body[guards[-1] + 1 :]
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]


def test_the_check_sees_a_definition_after_the_main_guard():
    late = "\n".join(
        [
            "def main():",
            "    return f()",
            'if __name__ == "__main__":',
            "    main()",
            "def f():",
            "    return 0",
        ]
    )
    assert _late_definitions(late) == ["f (line 5)"]
    assert _late_definitions(late.replace("__main__", "x")) == []


def test_tools_define_everything_before_their_main_guard():
    tools = sorted((ROOT / "tools").rglob("*.py"))
    assert ROOT / "tools/loop_v2_generate.py" in tools
    late = {
        str(path.relative_to(ROOT)): found
        for path in tools
        if (found := _late_definitions(path.read_text(encoding="utf-8")))
    }
    assert late == {}
