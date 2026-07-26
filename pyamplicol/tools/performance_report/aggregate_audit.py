# SPDX-License-Identifier: 0BSD
"""Authenticate a main commit assembled from audited architecture reports."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from .service import validate_profile_name


class AggregateAuditError(RuntimeError):
    """The aggregate report commit is not an allowed report-only descendant."""


_PROFILE_ROOT = PurePosixPath("docs/performance_reports")
_FIXED_OUTPUTS = frozenset(
    {
        "pyAmpliCol.pdf",
        "report_environment.json",
        "report_environment.tex",
        "result_validation_summary.tex",
    }
)


def _git(
    repo_root: Path,
    arguments: Sequence[str],
    *,
    text: bool = True,
) -> str | bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=text,
    )
    if completed.returncode != 0:
        detail = (
            completed.stderr.strip()
            if text
            else completed.stderr.decode("utf-8", errors="replace").strip()
        )
        raise AggregateAuditError(
            f"git {' '.join(arguments)} failed: {detail or completed.returncode}"
        )
    return completed.stdout


def _commit(repo_root: Path, revision: str) -> str:
    output = _git(repo_root, ("rev-parse", "--verify", f"{revision}^{{commit}}"))
    assert isinstance(output, str)
    return output.strip()


def _profile_for_output(path: str) -> str | None:
    candidate = PurePosixPath(path)
    try:
        relative = candidate.relative_to(_PROFILE_ROOT)
    except ValueError:
        return None
    if len(relative.parts) < 2:
        return None
    profile = relative.parts[0]
    try:
        validate_profile_name(profile)
    except ValueError:
        return None
    output = PurePosixPath(*relative.parts[1:])
    if len(output.parts) == 1:
        name = output.name
        if name in _FIXED_OUTPUTS:
            return profile
        if name.startswith("result_") and name.endswith("_table.tex"):
            return profile
        return None
    if (
        len(output.parts) == 2
        and output.parts[0] == "results"
        and output.name.endswith(".json")
    ):
        return profile
    return None


def _changed_paths(
    repo_root: Path,
    base_revision: str,
    revision: str,
) -> tuple[tuple[str, str], ...]:
    raw = _git(
        repo_root,
        (
            "diff",
            "--no-renames",
            "--name-status",
            "-z",
            f"{base_revision}..{revision}",
        ),
        text=False,
    )
    assert isinstance(raw, bytes)
    fields = raw.split(b"\0")
    if fields[-1:] == [b""]:
        fields.pop()
    if len(fields) % 2:
        raise AggregateAuditError("malformed NUL-delimited git diff output")
    changed: list[tuple[str, str]] = []
    for offset in range(0, len(fields), 2):
        status = fields[offset].decode("ascii", errors="strict")
        path = fields[offset + 1].decode("utf-8", errors="surrogateescape")
        changed.append((status, path))
    return tuple(changed)


def _tree_entry(repo_root: Path, revision: str, path: str) -> bytes:
    raw = _git(
        repo_root,
        ("ls-tree", "-z", revision, "--", path),
        text=False,
    )
    assert isinstance(raw, bytes)
    return raw


def _profile_tree(repo_root: Path, revision: str, profile: str) -> str:
    path = (_PROFILE_ROOT / profile).as_posix()
    output = _git(repo_root, ("rev-parse", f"{revision}:{path}"))
    assert isinstance(output, str)
    return output.strip()


def audit_aggregate_report(
    repo_root: Path,
    *,
    base_revision: str,
    revision: str,
    audited_profiles: Mapping[str, str],
) -> dict[str, object]:
    """Verify exact report-only paths and audited profile subtree identities."""

    root = repo_root.expanduser().resolve(strict=True)
    base = _commit(root, base_revision)
    aggregate = _commit(root, revision)
    _git(root, ("merge-base", "--is-ancestor", base, aggregate))

    audited = {
        validate_profile_name(profile): _commit(root, profile_revision)
        for profile, profile_revision in audited_profiles.items()
    }
    if not audited:
        raise AggregateAuditError("at least one audited profile is required")

    changed = _changed_paths(root, base, aggregate)
    changed_profiles: set[str] = set()
    for status, path in changed:
        if status not in {"A", "M"}:
            raise AggregateAuditError(
                f"aggregate path must be added or modified, not {status}: {path}"
            )
        profile = _profile_for_output(path)
        if profile is None:
            raise AggregateAuditError(
                f"aggregate commit changes a non-publication output: {path}"
            )
        entry = _tree_entry(root, aggregate, path)
        if not entry.startswith(b"100644 blob ") or not entry.endswith(b"\0"):
            raise AggregateAuditError(
                f"aggregate output is not one regular non-executable file: {path}"
            )
        changed_profiles.add(profile)

    if not changed_profiles:
        raise AggregateAuditError("aggregate commit contains no report outputs")
    missing = changed_profiles.difference(audited)
    if missing:
        raise AggregateAuditError(
            "aggregate changes profiles without audited revisions: "
            + ", ".join(sorted(missing))
        )

    for profile in sorted(changed_profiles):
        profile_revision = audited[profile]
        _git(root, ("merge-base", "--is-ancestor", base, profile_revision))
        aggregate_tree = _profile_tree(root, aggregate, profile)
        audited_tree = _profile_tree(root, profile_revision, profile)
        if aggregate_tree != audited_tree:
            raise AggregateAuditError(
                f"aggregate {profile!r} subtree differs from its audited revision"
            )

    return {
        "schema": "pyamplicol-report-aggregate-audit-v1",
        "base_revision": base,
        "aggregate_revision": aggregate,
        "changed_path_count": len(changed),
        "changed_profiles": sorted(changed_profiles),
        "audited_profile_revisions": {
            profile: audited[profile] for profile in sorted(changed_profiles)
        },
        "status": "ok",
    }


def _audited_profile(value: str) -> tuple[str, str]:
    profile, separator, revision = value.partition("=")
    if not separator or not revision:
        raise argparse.ArgumentTypeError(
            "audited profile must use PROFILE=REVISION"
        )
    try:
        validated = validate_profile_name(profile)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return validated, revision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticate a main commit assembled from audited reports."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument(
        "--audited-profile",
        action="append",
        default=[],
        type=_audited_profile,
        metavar="PROFILE=REVISION",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    audited = dict(arguments.audited_profile)
    if len(audited) != len(arguments.audited_profile):
        raise AggregateAuditError("each audited profile may appear only once")
    result = audit_aggregate_report(
        arguments.repo_root,
        base_revision=arguments.base_revision,
        revision=arguments.revision,
        audited_profiles=audited,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
