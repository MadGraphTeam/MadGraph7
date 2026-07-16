#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Build or assemble exact artifacts and print, but never run, an upload."""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from _artifacts import (
    ArtifactError,
    audit_sdist,
    audit_wheel,
    collect_unique_artifacts,
    copy_artifacts,
    verify_manifest,
    verify_parity_evidence,
    write_manifest,
)
from _common import (
    CANDIDATE_ARTIFACTS,
    DIST,
    ROOT,
    ReleaseError,
    build_mode,
    check_dependency_gate,
    clean_environment,
    exactly_one,
    run,
)

LEGAL_GATE = ROOT / "tools" / "release" / "check_legal_inventory.py"


def _check_legal_gate(mode: str) -> None:
    """Run the normative legal checker before inspecting release artifacts."""

    if mode not in {"candidate", "release"}:
        raise ReleaseError(f"unsupported legal-gate mode: {mode}")
    if mode == "candidate":
        print(
            "NON-PUBLISHABLE CANDIDATE: candidate legal checks do not authorize "
            "release or upload."
        )
    run(
        [sys.executable, LEGAL_GATE, "--mode", mode],
        cwd=ROOT,
        env=clean_environment(mode=mode),
    )


def assemble_bundle(
    source: Path,
    destination: Path,
    *,
    mode: str,
    require_all_targets: bool,
    source_commit: str | None = None,
    source_tag: str | None = None,
) -> Path:
    """Consolidate recursive CI artifacts into one hash-exact flat bundle."""

    source = source.resolve()
    destination = destination.resolve()
    artifacts = collect_unique_artifacts(source)
    sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
    wheels = [path for path in artifacts if path.suffix == ".whl"]
    if mode == "candidate" and sdists:
        raise ArtifactError("candidate bundles are wheel-only and non-publishable")
    sdist = exactly_one(sdists, "retained release sdist") if mode == "release" else None
    if not wheels:
        raise ArtifactError("release bundle contains no wheels")
    if destination.exists() and any(destination.iterdir()):
        raise ArtifactError(f"bundle destination must be empty: {destination}")

    sdist_report = audit_sdist(sdist, mode=mode) if sdist is not None else None
    wheel_reports = []
    for wheel in wheels:
        report = audit_wheel(wheel, mode=mode, native_scan=False)
        if mode == "release":
            assert sdist is not None
            verify_parity_evidence(source, wheel, sdist)
            report = replace(report, native_scan=True)
        wheel_reports.append(report)
    copied = copy_artifacts(artifacts, destination)
    copied_by_name = {path.name: path for path in copied}
    copied_sdist_report = (
        replace(sdist_report, filename=copied_by_name[sdist.name].name)
        if sdist is not None and sdist_report is not None
        else None
    )
    copied_wheel_reports = [
        replace(report, filename=copied_by_name[report.filename].name)
        for report in wheel_reports
    ]
    write_manifest(
        destination,
        mode=mode,
        wheels=copied_wheel_reports,
        sdists=[copied_sdist_report] if copied_sdist_report is not None else [],
        parity=("verified" if mode == "release" else "candidate-not-release-validated"),
        retained_sdist=copied_sdist_report if mode == "release" else None,
        source_commit=source_commit,
        source_tag=source_tag,
    )
    verify_manifest(
        destination,
        require_release=mode == "release",
        require_all_targets=require_all_targets,
    )
    return destination


def _build_default_bundle(mode: str) -> Path:
    command: list[str | Path] = [
        sys.executable,
        ROOT / "tools" / "release" / "build_release_artifacts.py",
    ]
    if mode == "candidate":
        command.append("--candidate")
    run(command, cwd=ROOT, env=clean_environment(mode=mode))
    return CANDIDATE_ARTIFACTS if mode == "candidate" else DIST


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="existing flat or recursive artifact directory",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        help="empty destination used to consolidate recursive CI artifacts",
    )
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--require-all-targets", action="store_true")
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tag")
    parser.add_argument("--skip-twine-check", action="store_true")
    parser.add_argument("--repository", choices=("pypi", "testpypi"), default="pypi")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = build_mode(candidate=args.candidate)
    _check_legal_gate(mode)
    check_dependency_gate(mode, online=mode == "release")
    if args.artifact_dir is None:
        if args.no_build:
            source = CANDIDATE_ARTIFACTS if mode == "candidate" else DIST
        else:
            source = _build_default_bundle(mode)
    else:
        source = args.artifact_dir

    if args.bundle_dir is not None:
        bundle = assemble_bundle(
            source,
            args.bundle_dir,
            mode=mode,
            require_all_targets=args.require_all_targets,
            source_commit=args.source_commit,
            source_tag=args.source_tag,
        )
    else:
        bundle = source.resolve()
        verify_manifest(
            bundle,
            require_release=mode == "release",
            require_all_targets=args.require_all_targets,
        )
        artifacts = collect_unique_artifacts(bundle)
        if mode == "release":
            exactly_one(
                [path for path in artifacts if path.name.endswith(".tar.gz")],
                "retained release sdist",
            )
        for path in artifacts:
            if path.suffix == ".whl":
                audit_wheel(path, mode=mode)
            else:
                audit_sdist(path, mode=mode)

    upload_artifacts = sorted(
        [*bundle.glob("*.whl"), *bundle.glob("*.tar.gz")], key=lambda path: path.name
    )
    if not args.skip_twine_check:
        run(
            [sys.executable, "-m", "twine", "check", *upload_artifacts],
            cwd=bundle,
            env=clean_environment(),
        )
    manifest = verify_manifest(
        bundle,
        require_release=mode == "release",
        require_all_targets=args.require_all_targets,
    )
    print("Exact artifact hashes:")
    for entry in manifest["artifacts"]:
        print(f"  {entry['sha256']}  {entry['filename']}")
    if mode == "candidate":
        print("Candidate artifacts are non-publishable; no upload command is emitted.")
        return 0
    command = [
        sys.executable,
        "-m",
        "twine",
        "upload",
        "--repository",
        args.repository,
        "--",
        *(str(path.resolve()) for path in upload_artifacts),
    ]
    print("Dry run only; exact upload command (not executed):")
    print(shlex.join(command))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
