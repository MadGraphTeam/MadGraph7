# SPDX-License-Identifier: 0BSD
"""Command-line entry point for the three-mode performance report."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .catalog import REPORT_CATALOG
from .measurement import source_revision
from .models import Accuracy, ArtifactPolicy, ExecutionMode, ModelKey, Workload
from .prepared import ensure_report_prepared_model
from .runner import DEFAULT_TARGET_RUNTIME_SECONDS
from .scheduler import (
    CampaignScheduler,
    CampaignSettings,
    CellSelection,
    plan_campaign,
    select_cells,
)
from .service import ReportPaths, ReportService
from .worker import write_cell_result


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and populate the pyAmpliCol performance report.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_repo_root(),
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("validate", "validate canonical caches and rendered tables"),
        ("audit", "verify exact cache coverage and checked-in table rendering"),
        ("reset", "reset all report measurements to N/A"),
        ("render", "merge immutable results and render tables"),
        ("recover", "recover immutable worker results and render tables"),
    ):
        item = subparsers.add_parser(command, help=help_text)
        if command in {"reset", "render", "recover"}:
            item.add_argument(
                "--compile",
                action="store_true",
                help="compile pyAmpliCol.pdf after publishing tables",
            )

    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--cell-id", required=True)
    worker.add_argument("--attempt-root", type=Path, required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    worker.add_argument("--log-path", type=Path)
    worker.add_argument("--baseline-json", type=Path)
    worker.add_argument("--prepared-model", type=Path)
    worker.add_argument("--reused-measurement-json", type=Path)
    worker.add_argument(
        "--target-runtime",
        type=float,
        default=DEFAULT_TARGET_RUNTIME_SECONDS,
    )
    worker.add_argument("--batch-size", type=int, default=128)
    worker.add_argument("--cell-cores", type=int, default=1)

    prepare = subparsers.add_parser("_prepare", help=argparse.SUPPRESS)
    prepare.add_argument(
        "--model",
        required=True,
        choices=(ModelKey.BUILTIN_SM.value, ModelKey.UFO_SM.value),
    )
    prepare.add_argument("--result-json", type=Path, required=True)
    prepare.add_argument("--cell-cores", type=int, default=1)

    populate = subparsers.add_parser(
        "populate",
        help="run selected cells through isolated direct-API workers",
    )
    populate.add_argument("--dataset", action="append", default=[])
    populate.add_argument(
        "--mode",
        action="append",
        choices=tuple(mode.value for mode in ExecutionMode),
        default=[],
    )
    populate.add_argument(
        "--model",
        action="append",
        choices=tuple(model.value for model in ModelKey),
        default=[],
    )
    populate.add_argument(
        "--accuracy",
        action="append",
        choices=tuple(accuracy.value for accuracy in Accuracy),
        default=[],
    )
    populate.add_argument("--process-key", action="append", default=[])
    populate.add_argument("--process", action="append", default=[])
    populate.add_argument("--n-final", action="append", default=[])
    populate.add_argument("--variant", action="append", default=[])
    populate.add_argument(
        "--workload",
        choices=("selected-flow", "all-flow", "both", "contracted"),
    )
    populate.add_argument("--cell-id", action="append", default=[])
    populate.add_argument("--missing-only", action="store_true")
    populate.add_argument("--rerun", action="store_true")
    populate.add_argument(
        "--artifact-policy",
        choices=tuple(policy.value for policy in ArtifactPolicy),
        default=ArtifactPolicy.REGENERATE.value,
    )
    populate.add_argument("--workers", type=int, default=1)
    populate.add_argument("--cell-cores", type=int, default=1)
    populate.add_argument(
        "--target-runtime",
        type=float,
        default=DEFAULT_TARGET_RUNTIME_SECONDS,
    )
    populate.add_argument("--batch-size", type=int, default=128)
    populate.add_argument("--timeout-seconds", type=float)
    populate.add_argument("--max-ram-gib", type=float)
    populate.add_argument("--campaign-max-ram-gib", type=float)
    populate.add_argument(
        "--limit-gib",
        type=float,
        help="compatibility alias for --campaign-max-ram-gib",
    )
    populate.add_argument("--allow-symbolica-parallel", action="store_true")
    populate.add_argument("--dry-run", action="store_true")
    populate.add_argument(
        "--refresh-pdf",
        choices=("never", "end"),
        default="never",
    )
    return parser


def _multiplicities(values: Sequence[str]) -> frozenset[int]:
    selected: set[int] = set()
    for value in values:
        for item in value.split(","):
            text = item.strip()
            if not text:
                continue
            if ".." in text:
                start_text, stop_text = text.split("..", 1)
                start, stop = int(start_text), int(stop_text)
                if start > stop:
                    raise ValueError("n-final range start must not exceed its end")
                selected.update(range(start, stop + 1))
            else:
                selected.add(int(text))
    if any(value < 1 for value in selected):
        raise ValueError("n-final values must be positive")
    return frozenset(selected)


def _workloads(value: str | None) -> frozenset[Workload]:
    if value is None:
        return frozenset()
    if value == "both":
        return frozenset({Workload.SELECTED_FLOW, Workload.ALL_FLOW})
    return frozenset({Workload(value)})


def _selection(args: argparse.Namespace) -> CellSelection:
    return CellSelection(
        datasets=frozenset(args.dataset),
        modes=frozenset(ExecutionMode(value) for value in args.mode),
        models=frozenset(ModelKey(value) for value in args.model),
        accuracies=frozenset(Accuracy(value) for value in args.accuracy),
        process_keys=frozenset(args.process_key),
        processes=frozenset(args.process),
        multiplicities=_multiplicities(args.n_final),
        variants=frozenset(args.variant),
        workloads=_workloads(args.workload),
        cell_ids=frozenset(args.cell_id),
    )


def _gib_bytes(value: float | None) -> int | None:
    if value is None:
        return None
    if value <= 0.0:
        raise ValueError("RAM limits must be positive")
    return int(value * 1024**3)


def _compile_pdf(service: ReportService) -> Path:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise FileNotFoundError("latexmk is required for --compile")
    environment = os.environ.copy()
    environment.update({"LANG": "C", "LC_ALL": "C", "LC_CTYPE": "C"})
    build_root = service.paths.artifact_root / "pdf-builds"
    build_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="report-pdf-", dir=build_root))
    try:
        build_docs = staging / "docs"
        shutil.copytree(
            service.paths.docs_dir,
            build_docs,
            ignore=shutil.ignore_patterns(
                "*.aux",
                "*.fdb_latexmk",
                "*.fls",
                "*.log",
                "*.out",
                "*.toc",
                "pyAmpliCol.pdf",
                ".coordination",
            ),
        )
        completed = subprocess.run(
            (
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                "pyAmpliCol.tex",
            ),
            cwd=build_docs,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            tail = "\n".join(
                (*completed.stdout.splitlines(), *completed.stderr.splitlines())[-80:]
            )
            raise RuntimeError(
                f"latexmk failed with exit {completed.returncode}:\n{tail}"
            )
        built_pdf = build_docs / "pyAmpliCol.pdf"
        if not built_pdf.is_file() or built_pdf.stat().st_size == 0:
            raise RuntimeError("latexmk did not produce a non-empty PDF")
        output = service.paths.docs_dir / "pyAmpliCol.pdf"
        temporary = output.with_name(f".{output.name}.new")
        shutil.copy2(built_pdf, temporary)
        temporary.replace(output)
        return output
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve(strict=False)
    service = ReportService(ReportPaths.from_repo(repo_root))

    if args.command == "validate":
        print(json.dumps(service.validate(), sort_keys=True))
        return 0
    if args.command == "audit":
        print(json.dumps(service.audit(), sort_keys=True))
        return 0
    if args.command == "populate":
        try:
            requested = select_cells(_selection(args))
            if not requested:
                parser.error("cell filters select no report cells")
            campaign_limit = (
                args.campaign_max_ram_gib
                if args.campaign_max_ram_gib is not None
                else args.limit_gib
            )
            settings = CampaignSettings(
                workers=args.workers,
                cell_cores=args.cell_cores,
                target_runtime_seconds=args.target_runtime,
                batch_size=args.batch_size,
                timeout_seconds=args.timeout_seconds,
                max_rss_bytes=_gib_bytes(args.max_ram_gib),
                campaign_max_rss_bytes=_gib_bytes(campaign_limit),
                artifact_policy=ArtifactPolicy(args.artifact_policy),
                missing_only=args.missing_only,
                rerun=args.rerun,
                allow_symbolica_parallel=args.allow_symbolica_parallel,
            )
        except ValueError as error:
            parser.error(str(error))
        planned = plan_campaign(
            requested,
            store=service.store,
            settings=settings,
            expected_revision=source_revision(repo_root),
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "requested": len(requested),
                        "scheduled": len(planned),
                        "cells": [
                            {
                                "cell_id": item.cell.cell_id,
                                "dependency": item.dependency,
                                "baseline_cell_id": item.baseline_cell_id,
                                "rank": item.rank,
                            }
                            for item in planned
                        ],
                    },
                    sort_keys=True,
                )
            )
            return 0
        result = CampaignScheduler(service, settings=settings).run(planned)
        if args.refresh_pdf == "end":
            _compile_pdf(service)
        print(
            json.dumps(
                {
                    "planned": len(result.planned),
                    "completed": len(result.outcomes),
                    "failed": len(result.failed),
                    "outcomes": [
                        {
                            "cell_id": outcome.cell_id,
                            "status": outcome.status,
                            "detail": outcome.detail,
                        }
                        for outcome in result.outcomes
                    ],
                },
                sort_keys=True,
            )
        )
        return 1 if result.failed else 0
    if args.command in {"reset", "render", "recover"}:
        written = service.publish(
            reset=args.command == "reset",
            merge_artifacts=args.command != "reset",
        )
        if args.compile:
            written = (*written, _compile_pdf(service))
        for path in written:
            print(path.relative_to(repo_root))
        return 0
    if args.command == "_worker":
        if args.cell_id not in {
            cell.cell_id for cell in REPORT_CATALOG.measurement_cells()
        }:
            parser.error(f"unknown --cell-id {args.cell_id!r}")
        write_cell_result(
            args.cell_id,
            args.result_json,
            repo_root=repo_root,
            attempt_root=args.attempt_root,
            target_runtime_seconds=args.target_runtime,
            batch_size=args.batch_size,
            worker_cores=args.cell_cores,
            baseline_json=args.baseline_json,
            prepared_model_path=args.prepared_model,
            reused_measurement_json=args.reused_measurement_json,
            log_path=args.log_path,
        )
        return 0
    if args.command == "_prepare":
        try:
            path, reused = ensure_report_prepared_model(
                store=service.store,
                repo_root=repo_root,
                worker_cores=args.cell_cores,
                model=ModelKey(args.model),
            )
            payload = {"path": os.fspath(path), "reused": reused}
            returncode = 0
        except Exception as error:
            payload = {
                "error": {
                    "kind": type(error).__name__,
                    "message": str(error),
                }
            }
            returncode = 1
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="ascii",
        )
        return returncode
    parser.error(f"unsupported command {args.command!r}")
    return 2


__all__ = ["main"]
