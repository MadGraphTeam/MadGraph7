#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Fold module-level string appends to UFO Laurent terms into their constructors.

A UFO module may extend a declared value after its constructor, e.g. the
G_mu charge counterterm of the UFOmodeldb issue-1 patch::

    eCoup_a0_UV_EW = CTParameter(..., value = {-1: '<a>', 0: '<b>'}, ...)
    ...
    eCoup_a0_UV_EW.value[-1] += '-0.5*(' + dMB_AAWcft_UV_EW.value[-1] + ')'

ufo-model-loader certifies a value's exact source only through its constructor
(string compositions of literals and ``Name.value[k]`` references included); any
later write to a ``value`` marks every value of the model uncertified, and the
loop engine, which reads exact inputs only, cannot lower it. This tool rewrites
each such statement into the constructor term it extends::

    value = {-1: '<a>' + '-0.5*(' + dMB_AAWcft_UV_EW.value[-1] + ')', 0: ...}

and comments the statement out (``# (folded) ...``), leaving every other byte
unchanged. Only top-level
``Name.value[<int>] += <string composition>`` statements are folded; the folded
value must be a direct Laurent dictionary literal of a top-level constructor,
every referenced declaration must be bound before that constructor and must not
itself be extended, so the folded module builds the same strings. Anything else
is refused. ``--check`` imports the original and the folded model and compares
every value of every UFO object (the strings must be identical).

Usage: fold_ufo_value_appends.py <ufo-dir> [--check <original-ufo-dir>]
(rewrites the modules of <ufo-dir> in place).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


def _power(node: ast.expr) -> int:
    power = ast.literal_eval(node)
    if isinstance(power, bool) or not isinstance(power, int):
        raise ValueError("a Laurent power must be an integer literal")
    return power


def _target(node: ast.expr) -> tuple[str, int] | None:
    """(name, power) of ``name.value[power]``, else None."""
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "value"
        and isinstance(node.value.value, ast.Name)
    ):
        return node.value.value.id, _power(node.slice)
    return None


def _references(node: ast.expr) -> list[str]:
    """Names referenced by a string composition (literals, ``+``,
    ``Name.value[k]``); refuses anything else."""
    if isinstance(node, ast.Constant) and type(node.value) is str:
        return []
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _references(node.left) + _references(node.right)
    reference = _target(node)
    if reference is not None:
        return [reference[0]]
    raise ValueError(f"not a string composition: {ast.unparse(node)}")


def _writes_value(statement: ast.stmt) -> bool:
    for node in ast.walk(statement):
        stored = isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
            getattr(node, "ctx", None), (ast.Store, ast.Del)
        )
        if stored and any(
            isinstance(part, ast.Attribute) and part.attr == "value"
            for part in ast.walk(node)
        ):
            return True
    return False


def _offsets(text: str) -> list[int]:
    starts, total = [0], 0
    for line in text.splitlines(keepends=True):
        total += len(line)
        starts.append(total)
    return starts


def fold_module(text: str) -> tuple[str, list[str]]:
    """The folded source of one module and a log line per folded statement."""
    tree = ast.parse(text)
    constructors: dict[str, ast.Assign] = {}
    appends: list[tuple[ast.AugAssign, str, int]] = []
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            constructors[statement.targets[0].id] = statement
        if isinstance(statement, ast.AugAssign) and isinstance(statement.op, ast.Add):
            target = _target(statement.target)
            if target is not None:
                _references(statement.value)
                appends.append((statement, *target))
                continue
        if _writes_value(statement):
            raise ValueError(
                f"line {statement.lineno}: a value write that is not a string append"
            )
    if not appends:
        return text, []
    extended = {name for _, name, _ in appends}
    starts = _offsets(text)

    def span(node: ast.AST) -> tuple[int, int]:
        return (
            starts[node.lineno - 1] + node.col_offset,
            starts[node.end_lineno - 1] + node.end_col_offset,
        )

    edits: list[tuple[int, int, str]] = []
    folded: dict[tuple[str, int], list[str]] = {}
    log = []
    for statement, name, power in appends:
        constructor = constructors.get(name)
        if constructor is None or constructor.lineno > statement.lineno:
            raise ValueError(
                f"line {statement.lineno}: {name} has no prior constructor"
            )
        for reference in _references(statement.value):
            bound = constructors.get(reference)
            if bound is None or bound.lineno >= constructor.lineno:
                raise ValueError(
                    f"line {statement.lineno}: {reference} is not bound before {name}"
                )
            if reference in extended:
                raise ValueError(
                    f"line {statement.lineno}: {reference} is itself extended"
                )
        segment = ast.get_source_segment(text, statement.value)
        folded.setdefault((name, power), []).append(segment)
        start = starts[statement.lineno - 1]
        end = starts[statement.end_lineno]
        if (
            text[start : start + statement.col_offset].strip()
            or text[span(statement)[1] : end].strip()
        ):
            raise ValueError(f"line {statement.lineno}: the statement shares its lines")
        original = text[start:end]
        edits.append(
            (
                start,
                end,
                "".join(f"# (folded) {line}" for line in original.splitlines(True)),
            )
        )
        log.append(f"{name}.value[{power}] += {' '.join(segment.split())}")
    for (name, power), segments in folded.items():
        call = constructors[name].value
        values = [k.value for k in getattr(call, "keywords", []) if k.arg == "value"]
        if not isinstance(call, ast.Call) or len(values) != 1:
            raise ValueError(f"{name}: no direct value keyword")
        if not isinstance(values[0], ast.Dict):
            raise ValueError(f"{name}: the value is not a Laurent dictionary literal")
        terms = [
            v
            for k, v in zip(values[0].keys, values[0].values, strict=True)
            if k is not None and _power(k) == power
        ]
        if len(terms) != 1:
            raise ValueError(f"{name}: no Laurent term {power}")
        _, end = span(terms[0])
        edits.append((end, end, "".join(f"+({s})" for s in segments)))
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text, log


def _values(ufo: Path) -> dict[tuple[str, str], object]:
    """Every UFO object's ``value`` (parameters, CT parameters, couplings), by
    collection and name, from a plain import of the model."""
    import importlib

    sys.path[:0] = [str(ufo.parent)]
    try:
        module = importlib.import_module(ufo.name)
    finally:
        del sys.path[0]
    out = {}
    for collection in ("all_parameters", "all_CTparameters", "all_couplings"):
        for item in getattr(module, collection, ()):
            out[(collection, item.name)] = item.value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("ufo", type=Path)
    parser.add_argument("--check", type=Path, help="the original UFO (compare values)")
    args = parser.parse_args()
    for path in sorted(args.ufo.glob("*.py")):
        if not path.stem.isidentifier():  # e.g. AppleDouble ._*.py files
            continue
        text = path.read_text(encoding="utf-8")
        new, log = fold_module(text)
        if new != text:
            path.write_text(new, encoding="utf-8")
            print(f"{path.name}: folded {len(log)} statements")
            for line in log:
                print("  ", line)
    if args.check:
        import subprocess

        code = (
            "import sys, json; sys.path.insert(0, sys.argv[1]);"
            "from fold_ufo_value_appends import _values; from pathlib import Path;"
            "v = _values(Path(sys.argv[2]));"
            "print(json.dumps({'|'.join(k): repr(x) for k, x in v.items()}))"
        )
        here = str(Path(__file__).resolve().parent)
        found = [
            subprocess.run(
                [sys.executable, "-B", "-c", code, here, str(ufo.resolve())],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            for ufo in (args.check, args.ufo)
        ]
        if found[0] != found[1]:
            raise SystemExit("the folded model's values differ from the original's")
        print(f"check: {found[0].count('|')} values identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
