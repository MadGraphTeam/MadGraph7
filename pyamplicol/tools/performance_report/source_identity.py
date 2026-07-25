# SPDX-License-Identifier: 0BSD
"""Fail-closed source identity for reproducible report measurements."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SOURCE_IDENTITY_SCHEMA = "pyamplicol-report-source-v1"
PUBLICATION_SOURCE_IDENTITY_SCHEMA = "pyamplicol-report-publication-source-v1"


class ReportSourceIdentityError(RuntimeError):
    """The report source cannot be identified as an immutable checkout."""


def _git_output(repo_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repo_root,
            check=False,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ReportSourceIdentityError(
            f"cannot inspect report source at {repo_root}: {error}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReportSourceIdentityError(
            f"git {' '.join(arguments)} failed at {repo_root}: "
            f"{detail or f'exit {completed.returncode}'}"
        )
    return completed.stdout


def _git_commit(repo_root: Path, revision: str) -> str:
    commit = (
        _git_output(repo_root, "rev-parse", "--verify", f"{revision}^{{commit}}")
        .decode("ascii")
        .strip()
    )
    if not commit:
        raise ReportSourceIdentityError(
            f"git returned an empty commit identity for {revision!r}"
        )
    return commit


def _git_tree(repo_root: Path, revision: str) -> str:
    tree = (
        _git_output(repo_root, "rev-parse", "--verify", f"{revision}^{{tree}}")
        .decode("ascii")
        .strip()
    )
    if not tree:
        raise ReportSourceIdentityError(
            f"git returned an empty tree identity for {revision!r}"
        )
    return tree


def _generated_report_path(value: str) -> bool:
    """Return whether a dirty path is a generated report output.

    Benchmark caches, rendered tables, PDFs, local evaluator attempts, and the
    corresponding generated files inside one architecture-profile workspace are
    outputs of a report campaign. They do not alter the evaluator source being
    measured and therefore do not make the source checkout ineligible. Profile
    prose, manifests, entry points, nested files, and all other tracked or
    untracked changes do.
    """

    path = PurePosixPath(value)
    parts = path.parts
    if not parts:
        return False
    if parts[0] in {".artifacts", "tmp"}:
        return True
    if (
        len(parts) == 3
        and parts[:2] == ("docs", "results")
        and parts[2].endswith(".json")
    ):
        return True
    if len(parts) == 2 and parts[0] == "docs":
        return _generated_publication_member(parts[1], allow_auxiliary=True)
    if (
        len(parts) == 5
        and parts[:2] == ("docs", "performance_reports")
        and parts[3] == "results"
        and parts[4].endswith(".json")
    ):
        return True
    if len(parts) == 4 and parts[:2] == ("docs", "performance_reports"):
        return _generated_publication_member(parts[3], allow_auxiliary=True)
    return False


def _generated_publication_member(name: str, *, allow_auxiliary: bool) -> bool:
    if name == "pyAmpliCol.pdf":
        return True
    if name == "result_validation_summary.tex":
        return True
    if name.startswith("result_") and name.endswith("_table.tex"):
        return True
    return allow_auxiliary and name.endswith(
        (
            ".aux",
            ".bbl",
            ".bcf",
            ".blg",
            ".fdb_latexmk",
            ".fls",
            ".log",
            ".out",
            ".run.xml",
            ".synctex.gz",
            ".toc",
        )
    )


def _architecture_report_path(value: str, profile: str) -> bool:
    """Return whether ``value`` is a tracked publication member for ``profile``.

    This is deliberately narrower than a directory-prefix check.  In
    particular, executables, arbitrary nested files, evaluator tools, profile
    prose, and manifests are not report-only changes.  A measurement commit can
    therefore be published from a descendant commit only when every intervening
    path is raw result JSON, a generated table or validation summary, or the
    reviewed PDF.
    """

    path = PurePosixPath(value)
    root = ("docs", "performance_reports", profile)
    parts = path.parts
    if parts[:3] != root:
        return False
    relative = parts[3:]
    if len(relative) == 1:
        return _generated_publication_member(
            relative[0],
            allow_auxiliary=False,
        )
    if len(relative) == 2 and relative[0] == "results":
        return relative[1].endswith(".json")
    return False


def _diff_changes(
    repo_root: Path,
    measured: str,
    publication: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    tokens = iter(
        _nul_paths(
            _git_output(
                repo_root,
                "diff",
                "--name-status",
                "-z",
                "--find-renames",
                f"{measured}..{publication}",
                "--",
            )
        )
    )
    changes: list[tuple[str, tuple[str, ...]]] = []
    while True:
        try:
            status = next(tokens)
        except StopIteration:
            break
        try:
            if status.startswith(("R", "C")):
                paths = (next(tokens), next(tokens))
            else:
                paths = (next(tokens),)
        except StopIteration as error:
            raise ReportSourceIdentityError(
                "git returned a malformed publication diff"
            ) from error
        changes.append((status, paths))
    return tuple(changes)


def _git_path_mode(repo_root: Path, revision: str, path: str) -> str | None:
    raw = _git_output(repo_root, "ls-tree", "-z", revision, "--", path)
    if not raw:
        return None
    record = raw.split(b"\0", 1)[0]
    header, separator, recorded_path = record.partition(b"\t")
    if not separator or os.fsdecode(recorded_path) != path:
        raise ReportSourceIdentityError(
            f"git returned malformed tree metadata for {path!r}"
        )
    fields = header.split()
    if len(fields) != 3:
        raise ReportSourceIdentityError(
            f"git returned malformed tree mode for {path!r}"
        )
    return fields[0].decode("ascii")


def _nul_paths(payload: bytes) -> tuple[str, ...]:
    return tuple(
        os.fsdecode(raw)
        for raw in payload.split(b"\0")
        if raw
    )


@dataclass(frozen=True, slots=True)
class ReportSourceIdentity:
    """Committed source identity plus any disqualifying working-tree changes."""

    revision: str
    tree: str
    dirty_paths: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return not self.dirty_paths

    def provenance(self) -> dict[str, object]:
        return {
            "report_source_identity_schema": SOURCE_IDENTITY_SCHEMA,
            "report_source_revision": self.revision,
            "report_source_tree": self.tree,
            "report_measured_source_revision": self.revision,
            "report_measured_source_tree": self.tree,
            "report_source_clean": self.eligible,
        }


@dataclass(frozen=True, slots=True)
class ReportPublicationIdentity:
    """Authenticated relation between measured and published source commits."""

    profile: str
    measured_revision: str
    measured_tree: str
    publication_revision: str
    publication_tree: str
    changed_paths: tuple[str, ...]
    disallowed_paths: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return not self.disallowed_paths

    def provenance(self) -> dict[str, object]:
        """Return a non-self-referential audit record for external retention.

        The publication revision cannot be embedded in a file contained by
        that same commit without changing the commit hash.  Final-audit output
        or release metadata should retain this mapping instead.
        """

        return {
            "report_publication_source_identity_schema": (
                PUBLICATION_SOURCE_IDENTITY_SCHEMA
            ),
            "report_profile": self.profile,
            "report_measured_source_revision": self.measured_revision,
            "report_measured_source_tree": self.measured_tree,
            "report_publication_revision": self.publication_revision,
            "report_publication_tree": self.publication_tree,
            "report_publication_report_only": self.eligible,
            "report_publication_changed_paths": list(self.changed_paths),
        }


def inspect_report_source(repo_root: Path) -> ReportSourceIdentity:
    """Inspect the current Git commit and fail-relevant working-tree changes."""

    root = repo_root.expanduser().resolve(strict=False)
    revision = _git_commit(root, "HEAD")
    tree = _git_tree(root, revision)
    changed = {
        *_nul_paths(
            _git_output(
                root,
                "diff",
                "--name-only",
                "--no-renames",
                "-z",
                "HEAD",
                "--",
            )
        ),
        *_nul_paths(
            _git_output(
                root,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
            )
        ),
    }
    dirty = tuple(
        sorted(path for path in changed if not _generated_report_path(path))
    )
    return ReportSourceIdentity(revision, tree, dirty)


def inspect_report_publication(
    repo_root: Path,
    *,
    measured_revision: str,
    profile: str,
    publication_revision: str = "HEAD",
) -> ReportPublicationIdentity:
    """Inspect a publication commit descended from an exact measurement SHA."""

    if (
        not profile
        or "/" in profile
        or "\\" in profile
        or profile in {".", ".."}
        or ".." in profile
    ):
        raise ValueError("report profile must be one safe path component")
    root = repo_root.expanduser().resolve(strict=False)
    measured = _git_commit(root, measured_revision)
    publication = _git_commit(root, publication_revision)
    try:
        ancestry = subprocess.run(
            ("git", "merge-base", "--is-ancestor", measured, publication),
            cwd=root,
            check=False,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ReportSourceIdentityError(
            f"cannot authenticate report publication ancestry: {error}"
        ) from error
    if ancestry.returncode == 1:
        raise ReportSourceIdentityError(
            "publication commit is not a descendant of measured source "
            f"{measured}"
        )
    if ancestry.returncode != 0:
        detail = ancestry.stderr.decode("utf-8", errors="replace").strip()
        raise ReportSourceIdentityError(
            "cannot authenticate report publication ancestry: "
            f"{detail or f'exit {ancestry.returncode}'}"
        )
    changes = _diff_changes(root, measured, publication)
    changed = tuple(
        sorted({path for _status, paths in changes for path in paths})
    )
    disallowed: set[str] = set()
    for status, paths in changes:
        if status not in {"A", "M"}:
            disallowed.update(paths)
            continue
        path = paths[0]
        if not _architecture_report_path(path, profile):
            disallowed.add(path)
            continue
        old_mode = _git_path_mode(root, measured, path)
        new_mode = _git_path_mode(root, publication, path)
        if new_mode != "100644" or (
            status == "M" and old_mode != "100644"
        ):
            disallowed.add(path)
    return ReportPublicationIdentity(
        profile=profile,
        measured_revision=measured,
        measured_tree=_git_tree(root, measured),
        publication_revision=publication,
        publication_tree=_git_tree(root, publication),
        changed_paths=changed,
        disallowed_paths=tuple(sorted(disallowed)),
    )


def require_eligible_report_source(repo_root: Path) -> ReportSourceIdentity:
    """Return an exact source identity or reject a dirty measurement source."""

    identity = inspect_report_source(repo_root)
    if not identity.eligible:
        displayed = ", ".join(identity.dirty_paths[:8])
        if len(identity.dirty_paths) > 8:
            displayed += f", ... ({len(identity.dirty_paths)} paths total)"
        raise ReportSourceIdentityError(
            "report measurements require a clean evaluator source checkout; "
            f"dirty source paths: {displayed}"
        )
    return identity


def require_report_only_publication(
    repo_root: Path,
    *,
    measured_revision: str,
    profile: str,
    publication_revision: str = "HEAD",
) -> ReportPublicationIdentity:
    """Authenticate a descendant with only profile-publication changes."""

    identity = inspect_report_publication(
        repo_root,
        measured_revision=measured_revision,
        profile=profile,
        publication_revision=publication_revision,
    )
    if not identity.eligible:
        displayed = ", ".join(identity.disallowed_paths[:8])
        if len(identity.disallowed_paths) > 8:
            displayed += (
                f", ... ({len(identity.disallowed_paths)} paths total)"
            )
        raise ReportSourceIdentityError(
            "publication changed evaluator or non-profile source after "
            f"measurement; disallowed paths: {displayed}"
        )
    return identity


__all__ = [
    "PUBLICATION_SOURCE_IDENTITY_SCHEMA",
    "SOURCE_IDENTITY_SCHEMA",
    "ReportPublicationIdentity",
    "ReportSourceIdentity",
    "ReportSourceIdentityError",
    "inspect_report_publication",
    "inspect_report_source",
    "require_eligible_report_source",
    "require_report_only_publication",
]
