# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import contextlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from decimal import Decimal
from pathlib import Path
from typing import TextIO, cast

from pyamplicol.config import PyAmpliColError
from pyamplicol.licensing import detect_symbolica_license
from pyamplicol.reporting import (
    ProgressSink,
    TtyProgressSink,
    close_progress_sink,
    configure_cli_logging,
    get_logger,
    progress_sink,
    render_summary,
    reset_cli_logging,
)

from .handlers import CliServices, DefaultCliServices, dispatch
from .licensing import LicenseRequestInvocation
from .parser import parse_cli
from .recurrence_export import RecurrenceExportInvocation
from .utilities import UtilityInvocation, example_card, execute_utility

# The pre-encoded fragments of the document being converted (``RawJson``:
# placeholders until ``write_result`` splices them in).
_FRAGMENTS: list[object] = []


def _json_value(value: object) -> object:
    # The leaves first: a loop profile's per-sample records are millions of
    # them, each converted in a few isinstance checks.
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) is float:
        # JSON has no infinity: an estimate on an exact zero is "inf".
        return value if math.isfinite(value) else str(value)
    if type(value) in (list, tuple):
        return [_json_value(entry) for entry in cast("Sequence[object]", value)]
    if type(value) is dict:
        mapping = cast("dict[object, object]", value)
        return {str(key): _json_value(entry) for key, entry in mapping.items()}
    from pyamplicol.api.results import GenerationStatistics, TreeOrderEvaluation
    from pyamplicol.api.virtual_results import (
        ResolvedVirtualEvaluation,
        VirtualEvaluation,
        VirtualOrderEvaluation,
    )
    from pyamplicol.reporting.raw_json import RawJson, placeholder

    from .partial_evaluation import PartialEvaluation

    if isinstance(value, RawJson):
        _FRAGMENTS.append(value)
        return placeholder(len(_FRAGMENTS) - 1)
    cli_json = getattr(value, "cli_json", None)
    if callable(cli_json) and is_dataclass(value):
        # A result with its own (compact) JSON form.
        return _json_value(cli_json())
    if isinstance(value, PartialEvaluation):
        # An interrupted evaluate: the points done, plus how many of the
        # requested points they are (data, as the stability run's setup).
        inner = _json_value(value.result)
        partial = {
            "interrupted": True,
            "requested_points": value.requested,
            "measured_points": value.measured,
        }
        if isinstance(inner, dict):
            return {**inner, **partial}
        # A tree evaluation's list keeps its type (the JSON schema only
        # gains fields); ``write_result`` puts the notice on stderr.
        return inner
    if isinstance(value, VirtualOrderEvaluation):
        # The one evaluation, its order axes and the explicit sums (named:
        # orders are never summed silently).
        return {
            "evaluation": _json_value(value.evaluation),
            "contribution_kind": value.contribution,
            "orders": _json_value(value.orders),
            "born_orders": _json_value(value.born_orders),
            "sum_orders": _json_value(value.sum_orders()),
            "sum_born_orders": _json_value(value.sum_born_orders()),
        }
    if isinstance(value, GenerationStatistics):
        return value.to_json()
    if is_dataclass(value) and not isinstance(value, type):
        result = {
            item.name: _json_value(getattr(value, item.name)) for item in fields(value)
        }
        if isinstance(value, (VirtualEvaluation, ResolvedVirtualEvaluation)):
            result.update(
                # "virtual" or "loop-induced" (m-lit Q36: the result's kind).
                contribution_kind=getattr(value, "contribution", "virtual"),
                epsilon_powers=list(value.epsilon_powers),
                coefficient_names=["finite", "single_pole", "double_pole"],
                normalization="Cepsilon=(4*pi)^epsilon/Gamma(1-epsilon)",
                shape=list(value.shape),
                born_shape=list(value.born_shape),
                # Per-point status and the result-level warning metadata
                # (m-lit ux-besteffort: data for the caller, never printed).
                statuses=list(value.statuses),
                warnings=[item.to_json() for item in value.warnings],
            )
        if isinstance(value, TreeOrderEvaluation):
            result.update(
                contribution_kind="tree",
                orders=_json_value(value.orders),
                shape=list(value.shape),
            )
        return result
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(entry) for key, entry in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(entry) for entry in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        # JSON has no infinity: an estimate on an exact zero is "inf".
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize CLI result of type {type(value).__name__}")


def _terminal_columns(stream: TextIO) -> int | None:
    """A terminal's width (tables are fitted within it); None for a file or
    a pipe (tables keep their natural width)."""

    import os

    try:
        if not stream.isatty():
            return None
        columns = os.get_terminal_size(stream.fileno()).columns
    except (AttributeError, OSError, ValueError):
        return None
    return columns if columns >= 40 else None


def write_result(
    value: object,
    *,
    format: str,
    stream: TextIO,
    color: bool = False,
    fields_help: bool = False,
) -> None:
    if value is None:
        return
    if format == "json":
        notice = getattr(value, "json_notice", None)
        line = notice() if callable(notice) else None
        if line:
            sys.stderr.write(f"{line}\n")
            sys.stderr.flush()
        from pyamplicol.reporting.raw_json import RawJson, splice

        _FRAGMENTS.clear()
        try:
            plain = _json_value(value)
            text = json.dumps(plain, indent=2, sort_keys=True, allow_nan=False)
            fragments = [f for f in _FRAGMENTS if isinstance(f, RawJson)]
        finally:
            _FRAGMENTS.clear()
        # One write: ``json.dump`` streams thousands of small chunks, slow on
        # a profile's per-sample records; their rows are pre-encoded.
        document = splice(text, fragments)
        stream.write(document)
        stream.write("\n")
        if line:
            # The size written, after the estimate announced.
            sys.stderr.write(f"Wrote the JSON result ({len(document) / 1e6:.0f} MB)\n")
            sys.stderr.flush()
        from .partial_evaluation import PartialEvaluation

        if isinstance(value, PartialEvaluation) and not isinstance(plain, dict):
            sys.stderr.write(f"{value.notice()}\n")
    elif format == "human":
        from pyamplicol.reporting import palette

        from .partial_evaluation import PartialEvaluation

        notice = None
        if isinstance(value, PartialEvaluation):
            notice = palette.paint(
                value.notice(), palette.RECOVERED, color=color, bold=True
            )
            value = value.result
        from pyamplicol.reporting.summary import table_width

        with table_width(_terminal_columns(stream)):
            rendered = render_summary(value, color=color)
        if rendered is None:
            raise RuntimeError(
                "human table rendering is unavailable; use --json for "
                "machine-readable output"
            )
        stream.write(f"{rendered}\n")
        if notice is not None:
            stream.write(f"{notice}\n")
        if fields_help:
            from pyamplicol.reporting.descriptions import field_descriptions

            stream.write(f"\n{field_descriptions(value)}\n")
    else:
        raise ValueError(f"unsupported output format {format!r}")
    stream.flush()


_COMMAND_TASK = "pyamplicol-command"


def _close_display(sink: ProgressSink | None, *, success: bool) -> None:
    """End the command's placeholder task (a no-op once replaced) and close
    the display, on every way out of the command."""

    if sink is None:
        return
    from pyamplicol.reporting import ProgressEnd

    with contextlib.suppress(Exception):
        sink.emit(ProgressEnd(_COMMAND_TASK, success=success))
    close_progress_sink(sink)


def _color_enabled(mode: str, stream: TextIO) -> bool:
    """``--color``: "always", "never", or "auto": a capable terminal
    (``TERM`` not "dumb") without ``NO_COLOR`` set."""

    from pyamplicol.reporting.progress import color_wanted

    return mode == "always" or (mode == "auto" and color_wanted(stream))


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    services: CliServices | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    from pyamplicol import _clock

    # The command's clock (``pyamplicol._clock``) for its time budgets.
    _clock.begin_command()
    try:
        return _run_cli(
            argv, services=services, stdin=stdin, stdout=stdout, stderr=stderr
        )
    finally:
        _clock.end_command()


def _run_cli(
    argv: Sequence[str] | None,
    *,
    services: CliServices | None,
    stdin: TextIO | None,
    stdout: TextIO | None,
    stderr: TextIO | None,
) -> int:
    output_stream = sys.stdout if stdout is None else stdout
    diagnostic_stream = sys.stderr if stderr is None else stderr
    input_stream = sys.stdin if stdin is None else stdin
    logging_configured = False
    sink = None
    config = None
    try:
        invocation = parse_cli(argv)
        if isinstance(invocation, RecurrenceExportInvocation):
            # Progress on a terminal only (the command has no --progress: a
            # script's stderr stays quiet), the line display on a dumb one,
            # never with a JSON result.
            sink = progress_sink(
                "auto"
                if invocation.output_format != "json"
                and bool(getattr(diagnostic_stream, "isatty", lambda: False)())
                else "off",
                stream=diagnostic_stream,
                color=_color_enabled(invocation.output_color, diagnostic_stream),
            )
            result = invocation.run(progress=sink)
            close_progress_sink(sink)
            sink = None
            write_result(
                result,
                format=invocation.output_format,
                stream=output_stream,
                color=_color_enabled(invocation.output_color, output_stream),
            )
            return 0
        if isinstance(invocation, LicenseRequestInvocation):
            result = invocation.run(stdin=input_stream, stdout=output_stream)
            write_result(
                result,
                format=invocation.output_format,
                stream=output_stream,
                color=_color_enabled(invocation.output_color, output_stream),
            )
            return 0
        if isinstance(invocation, UtilityInvocation):
            if invocation.kind == "examples-run":
                assert invocation.name is not None
                arguments = [str(example_card(invocation.name))]
                for override in invocation.overrides:
                    arguments.extend(("--set", override))
                if invocation.output_format == "json":
                    arguments.append("--json")
                if invocation.fields_help:
                    arguments.append("--fields-description-help")
                arguments.extend(("--color", invocation.output_color))
                invocation = parse_cli(arguments)
                assert not isinstance(
                    invocation,
                    (
                        LicenseRequestInvocation,
                        UtilityInvocation,
                        RecurrenceExportInvocation,
                    ),
                )
            else:
                if (
                    invocation.kind == "self-test"
                    and invocation.output_format != "json"
                ):
                    sink = progress_sink(
                        "auto",
                        stream=diagnostic_stream,
                        color=_color_enabled(
                            invocation.output_color, diagnostic_stream
                        ),
                    )
                result = execute_utility(invocation, progress=sink)
                close_progress_sink(sink)
                sink = None
                write_result(
                    result,
                    format=invocation.output_format,
                    stream=output_stream,
                    color=_color_enabled(invocation.output_color, output_stream),
                    fields_help=invocation.fields_help,
                )
                if bool(getattr(result, "interrupted", False)):
                    return 130
                if invocation.kind in {"doctor", "self-test"}:
                    return 0 if bool(getattr(result, "ok", False)) else 1
                return 0
        resolution = invocation.resolve()
        if services is None and resolution.effective.action == "model-compile":
            initial = resolution.effective
            detect_symbolica_license(
                suggest=initial.symbolica.suggest_license,
                json_mode=initial.output.format == "json",
                stream=diagnostic_stream,
            )
        config = resolution.effective
        progress_color = _color_enabled(config.output.color, diagnostic_stream)
        sink = progress_sink(
            config.output.progress,
            stream=diagnostic_stream,
            logger=get_logger("progress"),
            color=progress_color,
        )
        logging_stream = (
            sink.message_stream
            if isinstance(sink, TtyProgressSink)
            else diagnostic_stream
        )
        configure_cli_logging(
            config.output.log_level,
            stream=logging_stream,
            color=progress_color,
        )
        logging_configured = True
        if bool(getattr(sink, "live", False)):
            # Something on the terminal at once, its clock running, before
            # the command's own phases begin (model loading, imports).
            from pyamplicol.reporting import ProgressStart

            # A placeholder: the command's first own task replaces it (the
            # sink drops it), its end closes it otherwise.
            sink.emit(
                ProgressStart(
                    _COMMAND_TASK,
                    f"pyamplicol {config.action}",
                    details={"step": "starting", "placeholder": True},
                )
            )
        selected_services = (
            DefaultCliServices(resolution=resolution, card=invocation.card)
            if services is None
            else services
        )
        if invocation.correlators is not None:
            from pyamplicol.correlators import CorrelatorConfig

            if invocation.dry_run:
                raise ValueError("--correlators requires generation, not --dry-run")
            if services is not None:
                raise ValueError(
                    "--correlators requires the built-in generation service"
                )
            declarations = CorrelatorConfig.from_json_dict(
                json.loads(invocation.correlators.read_text())
            )
            selected_services = DefaultCliServices(
                resolution=resolution, correlators=declarations, card=invocation.card
            )
        result = dispatch(
            config,
            selected_services,
            sink,
            dry_run=invocation.dry_run,
            full_physics=invocation.full_physics,
        )
        _close_display(sink, success=True)
        sink = None
        color = _color_enabled(config.output.color, output_stream)
        write_result(
            result,
            format=config.output.format,
            stream=output_stream,
            color=color,
            fields_help=invocation.fields_help,
        )
        if bool(getattr(result, "interrupted", False)):
            return 130
        from pyamplicol.loops.v2.checks import CheckReport

        # ``pyamplicol check`` carries its own exit status: 0 iff every
        # applicable check passed (``CheckReport.exit_code``).
        return result.exit_code if isinstance(result, CheckReport) else 0
    except (PyAmpliColError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _close_display(sink, success=False)
        sink = None
        if logging_configured:
            get_logger("cli").error("%s", exc)
        else:
            diagnostic_stream.write(f"error: {exc}\n")
            diagnostic_stream.flush()
        return 2
    except KeyboardInterrupt:
        from pyamplicol.reporting.progress import progress_whereabouts

        whereabouts = progress_whereabouts(sink)
        _close_display(sink, success=False)
        sink = None
        message = (
            "interrupted (Ctrl-C) "
            + (f"{whereabouts}, " if whereabouts else "")
            + "before a complete result was available"
        )
        if config is not None and str(config.action) == "generate":
            # The writer stages the artifact privately and publishes it only
            # when complete (``artifacts.writer``).
            message += (
                f"; nothing was written to {config.generation.output} (an "
                "artifact appears only when complete)"
            )
        diagnostic_stream.write(f"{message}\n")
        diagnostic_stream.flush()
        return 130
    finally:
        _close_display(sink, success=False)
        if logging_configured:
            reset_cli_logging()


def main(argv: Sequence[str] | None = None) -> int:
    from pyamplicol import _clock

    if argv is None:
        # The command line owns this process: its peak-memory measurements
        # may reset the kernel's high-water mark (never done as a library).
        from pyamplicol.reporting.resources import own_process_accounting

        own_process_accounting()
        # The command line: its clock starts with the package's first
        # import (the interpreter's start, portably).
        _clock.mark_entry(_clock.IMPORTED)
    code = run_cli(argv)
    if argv is None:
        # The command line (not an in-process call): the result is written.
        if code == 130:
            # After Ctrl-C, work threads that cannot be interrupted (a
            # native call of a process phase) would hold the exit until
            # they finish; nothing was published.
            _exit_without_lingering_threads(code)
        _exit_promptly(code)
    return code


def _exit_promptly(code: int) -> None:
    """End the command line once its result is out: the exit handlers run
    and the streams are flushed, but the interpreter's teardown (freeing
    every module and a generation's millions of objects: 1-7 s after the
    final table) is skipped. Threads still running are waited for."""

    import atexit
    import os
    import threading

    for thread in threading.enumerate():
        if thread is not threading.main_thread() and not thread.daemon:
            thread.join()
    atexit._run_exitfuncs()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(OSError, ValueError):
            stream.flush()
    os._exit(code)


def _exit_without_lingering_threads(code: int, *, grace: float = 2.0) -> None:
    import os
    import threading
    import time

    deadline = time.monotonic() + grace
    for thread in threading.enumerate():
        if thread is threading.main_thread() or thread.daemon:
            continue
        thread.join(timeout=max(deadline - time.monotonic(), 0.0))
    if any(
        thread.is_alive() and not thread.daemon
        for thread in threading.enumerate()
        if thread is not threading.main_thread()
    ):
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(OSError, ValueError):
                stream.flush()
        os._exit(code)


__all__ = ["main", "run_cli", "write_result"]
