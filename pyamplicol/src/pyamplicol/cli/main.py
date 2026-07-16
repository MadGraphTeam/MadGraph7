# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from decimal import Decimal
from pathlib import Path
from typing import TextIO

from pyamplicol.config import PyAmpliColError
from pyamplicol.licensing import detect_symbolica_license
from pyamplicol.reporting import (
    configure_cli_logging,
    get_logger,
    progress_sink,
    render_summary,
    reset_cli_logging,
)

from .handlers import CliServices, DefaultCliServices, dispatch
from .licensing import LicenseRequestInvocation
from .parser import parse_cli
from .utilities import UtilityInvocation, example_card, execute_utility


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_value(getattr(value, item.name)) for item in fields(value)
        }
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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize CLI result of type {type(value).__name__}")


def write_result(
    value: object,
    *,
    format: str,
    stream: TextIO,
    color: bool = False,
) -> None:
    if value is None:
        return
    plain = _json_value(value)
    if format == "json":
        json.dump(plain, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    elif isinstance(plain, str):
        stream.write(f"{plain}\n")
    else:
        rendered = render_summary(value, color=color)
        if rendered is not None:
            stream.write(f"{rendered}\n")
        else:
            json.dump(plain, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    stream.flush()


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    services: CliServices | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output_stream = sys.stdout if stdout is None else stdout
    diagnostic_stream = sys.stderr if stderr is None else stderr
    input_stream = sys.stdin if stdin is None else stdin
    logging_configured = False
    try:
        invocation = parse_cli(argv)
        if isinstance(invocation, LicenseRequestInvocation):
            invocation.run(stdin=input_stream, stdout=output_stream)
            return 0
        if isinstance(invocation, UtilityInvocation):
            if invocation.kind == "examples-run":
                assert invocation.name is not None
                arguments = [str(example_card(invocation.name))]
                for override in invocation.overrides:
                    arguments.extend(("--set", override))
                arguments.extend(("--format", invocation.output_format))
                invocation = parse_cli(arguments)
                assert not isinstance(
                    invocation, (LicenseRequestInvocation, UtilityInvocation)
                )
            else:
                result = execute_utility(invocation)
                write_result(
                    result,
                    format=invocation.output_format,
                    stream=output_stream,
                )
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
        configure_cli_logging(config.output.log_level, stream=diagnostic_stream)
        logging_configured = True
        sink = progress_sink(
            config.output.progress,
            stream=diagnostic_stream,
            logger=get_logger("progress"),
        )
        selected_services = (
            DefaultCliServices(resolution=resolution) if services is None else services
        )
        result = dispatch(config, selected_services, sink, dry_run=invocation.dry_run)
        color = config.output.color == "always" or (
            config.output.color == "auto"
            and bool(getattr(output_stream, "isatty", lambda: False)())
        )
        write_result(
            result,
            format=config.output.format,
            stream=output_stream,
            color=color,
        )
        return 0
    except (PyAmpliColError, OSError, RuntimeError, TypeError, ValueError) as exc:
        if logging_configured:
            get_logger("cli").error("%s", exc)
        else:
            diagnostic_stream.write(f"error: {exc}\n")
            diagnostic_stream.flush()
        return 2
    finally:
        if logging_configured:
            reset_cli_logging()


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(argv)


__all__ = ["main", "run_cli", "write_result"]
