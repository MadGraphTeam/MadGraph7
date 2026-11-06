#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run pytest one process per test file (or per test id) and count what ran.

Targets are test files, globs or node ids, run in the given order (globs
sorted). With ``--marker`` or ``--split`` the targets are collected once first:
a file holding no selected test is dropped, a file matching ``--split`` runs
one process per test id. ``--shard K/N`` keeps part K of a split into N
parts balanced by ``--durations`` (seconds per file from an earlier run,
unknown files the median; without it, by unit count). Without
``SYMBOLICA_LICENSE`` the units run one at a time (an unlicensed Symbolica
admits one process per user); with it ``--jobs`` run concurrently.

The run fails when a unit fails, with ``--require-each`` when a unit executes
no test (every selected test skipped: a vacuous unit), and when fewer than
``--min-executed`` tests ran in total; an empty selection fails unless
``--allow-empty`` gives the notice that makes it a pass. ``--record`` writes
the units' wall times as a ``--durations`` file.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


@dataclass
class Result:
    unit: str
    code: int
    seconds: float
    rss_gib: float
    tests: int = 0
    failed: int = 0
    skipped: int = 0
    log: Path | None = None

    @property
    def executed(self) -> int:
        return self.tests - self.skipped


def _targets(patterns: Sequence[str]) -> list[str]:
    targets: list[str] = []
    for pattern in patterns:
        path = pattern.split("::", 1)[0]
        if glob.has_magic(path):
            matches = sorted(glob.glob(path, recursive=True))
            if "::" in pattern:
                raise SystemExit(f"error: a node id cannot hold a glob: {pattern}")
            targets.extend(matches)
        elif Path(path).exists():
            targets.append(pattern)
        else:
            raise SystemExit(f"error: no such test target: {pattern}")
    return list(dict.fromkeys(targets))


def _collect(
    python: str, targets: list[str], pytest_args: list[str]
) -> tuple[list[str], str]:
    command = [python, "-m", "pytest", "--collect-only", "-q", *pytest_args]
    result = subprocess.run(
        [*command, *targets], capture_output=True, text=True, check=False
    )
    ids = [line for line in result.stdout.splitlines() if "::" in line]
    if result.returncode not in (0, 5):
        sys.stdout.write(result.stdout + result.stderr)
        raise SystemExit(f"error: collection failed (exit {result.returncode})")
    return ids, result.stdout.strip().splitlines()[-1] if result.stdout else ""


def _units(ids: list[str], split: list[str]) -> list[str]:
    units: list[str] = []
    for node in ids:
        path = node.split("::", 1)[0]
        split_here = any(Path(path).match(pattern) for pattern in split)
        unit = node if split_here else path
        if unit not in units:
            units.append(unit)
    return units


def _shard(units: list[str], shard: str, durations: dict[str, float]) -> list[str]:
    index, count = (int(part) for part in shard.split("/"))
    if not 1 <= index <= count:
        raise SystemExit(f"error: --shard {shard}: expected K/N with 1 <= K <= N")
    paths = [unit.split("::", 1)[0] for unit in units]
    known = [durations[path] for path in paths if path in durations]
    default = statistics.median(known) if known else 1.0

    def weight(unit: str) -> float:
        path = unit.split("::", 1)[0]
        # A file split into test ids shares its recorded time among them.
        return durations.get(path, default) / paths.count(path)

    # Longest first, each to the least loaded shard (ties: lowest shard).
    loads = [0.0] * count
    chosen: list[str] = []
    for unit in sorted(units, key=lambda unit: (-weight(unit), unit)):
        target = min(range(count), key=lambda shard: (loads[shard], shard))
        loads[target] += weight(unit)
        if target == index - 1:
            chosen.append(unit)
    order = {unit: position for position, unit in enumerate(units)}
    return sorted(chosen, key=order.__getitem__)


def _run_unit(
    python: str, unit: str, pytest_args: list[str], output: Path, number: int
) -> Result:
    report = output / f"unit-{number:04d}.xml"
    log = output / f"unit-{number:04d}.log"
    # Each unit has its own pytest base temporary directory, removed after it.
    basetemp = output / f"tmp-{number:04d}"
    command = [python, "-m", "pytest", "-q", f"--junitxml={report}", *pytest_args]
    start = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(
            [*command, f"--basetemp={basetemp}", unit],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        _, status, usage = os.wait4(process.pid, 0)
    process.returncode = os.waitstatus_to_exitcode(status)
    shutil.rmtree(basetemp, ignore_errors=True)
    result = Result(
        unit,
        process.returncode,
        time.monotonic() - start,
        usage.ru_maxrss / 1024**2,
        log=log,
    )
    if report.exists():
        for case in ElementTree.parse(report).getroot().iter("testcase"):
            result.tests += 1
            tags = {child.tag: child for child in case}
            if "failure" in tags or "error" in tags:
                result.failed += 1
            skipped = tags.get("skipped")
            if skipped is not None and skipped.get("type") != "pytest.xfail":
                result.skipped += 1
    return result


def _summary(results: list[Result], notes: list[str]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    rows = [
        f"| {r.unit} | {r.code} | {r.executed} | {r.skipped} | {r.failed} "
        f"| {r.seconds:.0f} | {r.rss_gib:.2f} |"
        for r in results
    ]
    with open(path, "a") as stream:
        stream.write("\n".join(notes) + "\n\n")
        stream.write("| unit | exit | executed | skipped | failed | s | GiB |\n")
        stream.write("|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", help="test files, globs or node ids")
    parser.add_argument("--marker", help="pytest -m expression (collects first)")
    parser.add_argument(
        "--split", action="append", default=[], help="glob: one process per test id"
    )
    parser.add_argument("--deselect", action="append", default=[])
    parser.add_argument("--shard", default="1/1", help="K/N")
    parser.add_argument("--durations", type=Path, help="JSON: seconds per file")
    parser.add_argument("--record", type=Path, help="write the units' wall times")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--min-executed", type=int, default=1)
    parser.add_argument(
        "--require-each",
        action="store_true",
        help="fail a unit that executes no test (all selected tests skipped)",
    )
    parser.add_argument("--allow-empty", metavar="NOTICE")
    parser.add_argument("--output", type=Path, help="logs and JUnit reports")
    parser.add_argument("--python", default=sys.executable)
    argv = list(sys.argv[1:] if argv is None else argv)
    cut = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:cut])
    pytest_args = argv[cut + 1 :]
    if args.marker:
        pytest_args[:0] = ["-m", args.marker]
    for node in args.deselect:
        pytest_args += ["--deselect", node]

    targets = _targets(args.targets)
    units = targets
    if args.marker or args.split:
        ids, line = _collect(args.python, targets, pytest_args)
        units = _units(ids, args.split)
        print(f"collected: {line} -> {len(units)} units from {len(targets)} targets")
    durations = json.loads(args.durations.read_text()) if args.durations else {}
    units = _shard(units, args.shard, durations)
    if not units:
        if args.allow_empty is not None:
            print(f"::notice::{args.allow_empty}")
            return 0
        print(f"::error::no test selected by {shlex.join(args.targets)}")
        return 1

    licensed = bool(os.environ.get("SYMBOLICA_LICENSE"))
    jobs = max(1, args.jobs) if licensed else 1
    output = args.output or Path(tempfile.mkdtemp(prefix="pytest-files-"))
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"shard {args.shard}: {len(units)} units, {jobs} at a time "
        f"({'licensed' if licensed else 'no SYMBOLICA_LICENSE: one at a time'})",
        flush=True,
    )

    def run(numbered: tuple[int, str]) -> Result:
        number, unit = numbered
        result = _run_unit(args.python, unit, pytest_args, output, number)
        print(
            f"[{number + 1}/{len(units)}] exit {result.code} "
            f"{result.seconds:7.1f} s {result.rss_gib:6.2f} GiB "
            f"executed {result.executed} skipped {result.skipped} "
            f"failed {result.failed}  {unit}",
            flush=True,
        )
        return result

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(run, enumerate(units)))

    problems: list[str] = []
    for result in results:
        if result.code != 0:
            problems.append(f"{result.unit} exited {result.code}")
        elif args.require_each and result.executed == 0:
            problems.append(f"{result.unit} executed no test (all skipped)")
    executed = sum(result.executed for result in results)
    if executed < args.min_executed:
        problems.append(f"{executed} tests executed, fewer than {args.min_executed}")
    for result in results:
        if result.code != 0 and result.log is not None:
            print(f"::group::{result.unit} (exit {result.code})")
            print(result.log.read_text(errors="replace"))
            print("::endgroup::")
    notes = [
        f"{len(results)} units, {executed} tests executed, "
        f"{sum(r.skipped for r in results)} skipped, "
        f"{sum(r.failed for r in results)} failed; "
        f"slowest {max(r.seconds for r in results):.0f} s, "
        f"peak {max(r.rss_gib for r in results):.2f} GiB"
    ]
    print(notes[0])
    _summary(results, notes)
    if args.record:
        recorded: dict[str, float] = {}
        for result in results:
            path = result.unit.split("::", 1)[0]
            recorded[path] = round(recorded.get(path, 0.0) + result.seconds, 1)
        args.record.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")
    for problem in problems:
        print(f"::error::{problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
