# SPDX-License-Identifier: 0BSD
"""Array, artifact-path, and progress helpers for evaluator adapters."""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .._internal.physics.types import NativeEvaluationError

ComplexOutput = np.ndarray | tuple[np.ndarray, ...]


# Python Symbolica's `Evaluator.save()` envelope (magic, then the producer's
# native-code flag) and, after Symbolica's state header magic (u32) and export
# format (u16), the header's whole-state flag.
_SAVED_EVALUATOR_MAGIC = b"\xffSYMEVAL\x01"
_WHOLE_STATE_FLAG = len(_SAVED_EVALUATOR_MAGIC) + 1 + 6


def saved_evaluator(evaluator: Any) -> bytes:
    """``evaluator.save()`` with only the Symbolica state the evaluator references.

    A Python Symbolica release without symbolica-dev#163 (PyPI 3.0.3) writes
    the producer's whole Symbolica state into every save: tens of MB per
    evaluator after a model compile, depending on every symbol the process
    created, and imported again by every load. The native re-encodes such a
    save in #163's state-minimal form with its payload unchanged
    (``state_minimal_saved_evaluator``). Any other save is returned as it is,
    without the native.
    """
    saved = bytes(evaluator.save())
    if not (
        saved.startswith(_SAVED_EVALUATOR_MAGIC)
        and saved[_WHOLE_STATE_FLAG : _WHOLE_STATE_FLAG + 1] == b"\x01"
    ):
        return saved
    from ..loops.native import load_loop_native

    return bytes(load_loop_native().state_minimal_saved_evaluator(saved))


def _symbolica_instruction_program(exported: Any) -> Any:
    """Extract the linear program from Symbolica's structured instruction export."""
    if isinstance(exported, tuple):
        return exported
    if exported.sub_evaluators:
        raise NativeEvaluationError(
            "native kernels require inlined Symbolica function bodies"
        )
    return (
        exported.instructions,
        exported.temporary_count,
        _resolved_constants(exported),
    )


def _resolved_constants(exported: Any) -> list[Any]:
    """The exported exact constants with every registered constant or
    fixed-argument function slot (pi, ...) holding its binary64 value.

    Symbolica's instruction export holds a placeholder 0 in such a slot and
    lists the slot in ``constant_functions`` (function, tags, exact
    arguments). The value is the one Symbolica's own binary64 evaluator reads,
    stored as the exact rational of that double so both consumers (SymJIT and
    the direct C++ export) read it back bitwise.
    """
    constants = list(exported.constants)
    for constant in exported.constant_functions:
        constants[constant.index] = _binary64_constant(constant)
    return constants


def _binary64_constant(constant: Any) -> Any:
    from fractions import Fraction

    from symbolica import E, Expression

    arguments = [*constant.tags, *constant.arguments]
    function = constant.function
    expression = function(*arguments) if arguments else function
    value = complex(
        np.asarray(
            expression.evaluator([]).evaluate_complex(np.zeros((1, 0))),
            dtype=np.complex128,
        ).ravel()[0]
    )
    if not (np.isfinite(value.real) and np.isfinite(value.imag)):
        raise NativeEvaluationError(
            f"the registered Symbolica constant {expression!r} is not finite "
            "in binary64"
        )
    real, imaginary = Fraction(value.real), Fraction(value.imag)
    exact = E(f"{real.numerator}/{real.denominator}")
    if imaginary:
        exact = (
            exact + E(f"{imaginary.numerator}/{imaginary.denominator}") * Expression.I
        )
    return exact


def _symbolica_instruction_program_repr(exported: Any) -> str:
    """Serialize Symbolica's structured instruction export for SymJIT.

    Symbolica omits coefficients of magnitude one before its imaginary-unit
    glyph, but SymJIT's instruction parser requires them. Keep the exact
    rational constants and the original evaluator state unchanged.

    Function bodies that Symbolica exports as sub-evaluators (functions
    defined with ``inlining="never"``, such as a model's helper functions) are
    inlined here, and function names that SymJIT's tokenizer cannot read bare
    (namespaced symbols) are quoted. SymJIT resolves such a name at compile and
    load time; Rusticol registers only pyAmpliCol's scalar primitives, so any
    other non-built-in function fails closed in the compiler. A program
    without sub-evaluators or namespaced functions serializes as before.

    Jumps and their targets (``if_else``, ``goto``, ``label``) are kept:
    SymJIT 2.28.3 carries the fix of its common-subexpression elimination
    across branches (siravan/symjit#31), for which they used to be dropped.
    """
    if isinstance(exported, tuple):
        instructions, temporary_count, constants = exported
    elif exported.sub_evaluators:
        instructions, temporary_count, constants = _InlinedProgram(exported).result()
    else:
        instructions = [_quote_function_name(row) for row in exported.instructions]
        temporary_count = exported.temporary_count
        constants = _resolved_constants(exported)
    serialized_constants = ", ".join(
        re.sub(r"(?<![0-9])\U0001d456", "1\U0001d456", repr(value))
        for value in constants
    )
    return f"({instructions!r}, {temporary_count}, [{serialized_constants}])"


_SYMJIT_BARE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_SYMJIT_QUOTED_NAME = re.compile(r"[^'\s]+")
_SLOT_KINDS = frozenset({"param", "out", "temp", "const"})
_WRITING_KINDS = frozenset(
    {"add", "mul", "pow", "powf", "fun", "external_fun", "assign", "join"}
)


class _QuotedName:
    """A function name that SymJIT reads as one quoted identifier."""

    def __init__(self, text: str):
        self.text = text

    def __repr__(self) -> str:
        return f"'{self.text}'"


def _quote_function_name(row: Any) -> Any:
    if row[0] not in {"fun", "external_fun"}:
        return row
    text = repr(row[2])
    if _SYMJIT_BARE_NAME.fullmatch(text):
        return row
    if not _SYMJIT_QUOTED_NAME.fullmatch(text):
        raise NativeEvaluationError(
            f"SymJIT cannot spell the Symbolica function name {text!r}"
        )
    return (*row[:2], _QuotedName(text), *row[3:])


def _is_slot(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and value[0] in _SLOT_KINDS
        and type(value[1]) is int
    )


def _instruction_labels(rows: Iterable[Any]) -> list[int]:
    labels = []
    for row in rows:
        if row[0] == "if_else":
            labels.append(row[2])
        elif row[0] in {"goto", "label"}:
            labels.append(row[1])
    return labels


class _InlinedProgram:
    """Symbolica's instruction export with every sub-evaluator call inlined.

    A call ``dst = f(args)`` of a sub-evaluator ``f`` becomes ``f``'s body
    with its parameters read from the call's argument slots, its temporaries
    and output in fresh temporaries, its constants appended to the program's
    and its labels renumbered past every label in use; the call's destination
    receives the body's output last, so a destination that is also an
    argument is read before it is written. Nested sub-evaluators inline in
    turn, on an explicit stack of bodies (no depth limit: the export is a
    finite tree of bodies). The evaluated arithmetic is unchanged (the Rust
    twin is ``rusticol-core`` ``symjit_plane::InlinedProgram``).
    """

    def __init__(self, exported: Any):
        self.rows: list[Any] = []
        self.constants = _resolved_constants(exported)
        self.temporary_count = exported.temporary_count
        self.next_label = 1 + max(_instruction_labels(exported.instructions), default=0)
        self._inline_all(exported)

    def result(self) -> tuple[list[Any], int, list[Any]]:
        return self.rows, self.temporary_count, self.constants

    def _inline_all(self, exported: Any) -> None:
        # A frame: (a body's sub-evaluators, its instructions iterator, its
        # slot map, its label offset, the call's (destination, output) or
        # None at the top level).
        stack = [self._frame(exported, lambda slot: slot, 0, None)]
        while stack:
            bodies, rows, slot_of, label_offset, call = stack[-1]
            row = next(rows, None)
            if row is None:
                stack.pop()
                if call is not None:
                    destination, output = call
                    self.rows.append(("assign", destination, ("temp", output)))
                continue
            kind = row[0]
            body = None
            if kind == "fun":
                body = bodies.get((repr(row[2]), tuple(map(repr, row[3]))))
            if body is not None:
                arguments = [slot_of(a) for a in row[4]]
                stack.append(self._enter(slot_of(row[1]), arguments, body))
            elif kind == "if_else":
                self.rows.append((kind, slot_of(row[1]), row[2] + label_offset))
            elif kind in {"goto", "label"}:
                self.rows.append((kind, row[1] + label_offset))
            elif kind in _WRITING_KINDS:
                if call is not None and row[1][0] == "param":
                    raise NativeEvaluationError(
                        "a Symbolica function body writes one of its parameters"
                    )
                self.rows.append(_quote_function_name(_map_slots(row, slot_of)))
            else:
                raise NativeEvaluationError(
                    f"cannot inline the Symbolica instruction kind {kind!r}"
                )

    @staticmethod
    def _frame(program: Any, slot_of: Any, label_offset: int, call: Any) -> tuple:
        bodies = {
            (repr(sub.function), tuple(map(repr, sub.tags))): sub.evaluator
            for sub in program.sub_evaluators
        }
        return (
            bodies,
            iter(program.instructions),
            slot_of,
            label_offset,
            call,
        )

    def _enter(self, destination: Any, arguments: list[Any], body: Any) -> tuple:
        if body.output_count != 1:
            raise NativeEvaluationError(
                "a Symbolica function body must have exactly one output"
            )
        if len(arguments) != body.input_count:
            raise NativeEvaluationError(
                "a Symbolica function call does not match its body's arity"
            )
        temporary_base = self.temporary_count
        output_base = temporary_base + body.temporary_count
        self.temporary_count = output_base + body.output_count
        constant_base = len(self.constants)
        self.constants.extend(_resolved_constants(body))
        label_offset = self.next_label
        self.next_label += 1 + max(_instruction_labels(body.instructions), default=0)

        def body_slot(slot: Any) -> Any:
            kind, index = slot
            if kind == "param":
                return arguments[index]
            if kind == "temp":
                return ("temp", temporary_base + index)
            if kind == "out":
                return ("temp", output_base + index)
            return ("const", constant_base + index)

        return self._frame(body, body_slot, label_offset, (destination, output_base))


def _map_slots(row: Any, slot_of: Any) -> tuple[Any, ...]:
    mapped: list[Any] = [row[0]]
    for value in row[1:]:
        if _is_slot(value):
            mapped.append(slot_of(value))
        elif isinstance(value, list):
            mapped.append([slot_of(v) if _is_slot(v) else v for v in value])
        else:
            mapped.append(value)
    return tuple(mapped)


def _complex128_parameter_rows(parameter_rows: Any) -> np.ndarray:
    if (
        isinstance(parameter_rows, np.ndarray)
        and parameter_rows.dtype == np.complex128
        and parameter_rows.flags.c_contiguous
    ):
        return parameter_rows
    return np.asarray(parameter_rows, dtype=np.complex128)


def _evaluate_prepared_complex(evaluator: Any, parameter_rows: np.ndarray) -> Any:
    evaluate_prepared = getattr(evaluator, "_evaluate_complex_prepared", None)
    if callable(evaluate_prepared):
        return evaluate_prepared(parameter_rows)
    return evaluator.evaluate_complex(parameter_rows)


def _evaluate_prepared_complex_profiled(
    evaluator: Any,
    parameter_rows: np.ndarray,
) -> tuple[Any, tuple[float, float, float]]:
    evaluate_profiled = getattr(evaluator, "_evaluate_complex_profiled_prepared", None)
    if not callable(evaluate_profiled):
        raise NativeEvaluationError(
            "evaluator does not expose native complex profiling"
        )
    return evaluate_profiled(parameter_rows)


def _evaluate_complex_outputs(evaluator: Any, parameter_rows: Any) -> ComplexOutput:
    evaluate_chunks = getattr(evaluator, "evaluate_complex_chunks", None)
    if callable(evaluate_chunks):
        return tuple(
            np.asarray(chunk, dtype=np.complex128)
            for chunk in evaluate_chunks(parameter_rows)
        )
    return np.asarray(evaluator.evaluate_complex(parameter_rows), dtype=np.complex128)


def _symbolica_evaluator_artifact_manifest(
    evaluator: Any,
    artifact_dir: Path,
) -> dict[str, Any]:
    manifest = getattr(evaluator, "artifact_manifest", None)
    if not callable(manifest):
        raise NativeEvaluationError(
            "saving evaluator artifacts is currently supported only for "
            "Symbolica evaluator adapters"
        )
    return manifest(artifact_dir)


def _artifact_path_for_manifest(path: Path, artifact_dir: Path) -> str:
    artifact_root = artifact_dir.resolve()
    source_path = path.expanduser()
    source_resolved = source_path.resolve()
    try:
        source_resolved.relative_to(artifact_root)
    except ValueError:
        if source_path.exists():
            target_dir = artifact_dir / "compiled" / "repackaged"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / source_path.name
            if source_resolved != target.resolve():
                shutil.copy2(source_path, target)
            source_path = target
        else:
            source_path = source_resolved
    try:
        return os.path.relpath(source_path, artifact_dir)
    except ValueError:
        return str(source_path)


def _artifact_subdirectory(artifact_dir: Path, name: str) -> Path:
    directory = artifact_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _artifact_path_from_manifest(path: str, artifact_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return artifact_dir / candidate


def _safe_symbol_name(value: str) -> str:
    result = "".join(character if character.isalnum() else "_" for character in value)
    if not result:
        return "eval"
    if result[0].isdigit():
        return f"eval_{result}"
    return result


def _progress_outputs(
    outputs: tuple[Any, ...],
    *,
    enabled: bool,
) -> Iterable[Any]:
    if not enabled:
        return outputs
    try:
        from colorama import Fore, Style  # type: ignore[import-untyped]
        from progressbar import (  # type: ignore[import-not-found]
            ETA,
            Bar,
            Percentage,
            ProgressBar,
        )
    except ImportError:
        return outputs
    widgets = [
        Fore.CYAN,
        " merging evaluators ",
        Percentage(),
        " ",
        Bar(),
        " ",
        ETA(),
        Style.RESET_ALL,
    ]
    return ProgressBar(max_value=len(outputs), widgets=widgets)(outputs)
