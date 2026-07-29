# SPDX-License-Identifier: 0BSD
"""Fail-closed publication projection and local-path resolution."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .service import ReportPaths

_ARTIFACT_ROOT = "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}"
_COORDINATION_ROOT = "${PYAMPLICOL_REPORT_COORDINATION_ROOT}"
_SOURCE_ROOT = "${PYAMPLICOL_SOURCE_ROOT}"
_REDACTED_ROOT = "${LOCAL_PATH_REDACTED}"
_PORTABLE_ROOTS = (
    _ARTIFACT_ROOT,
    _COORDINATION_ROOT,
    _SOURCE_ROOT,
    _REDACTED_ROOT,
)

# These are the only measurement fields whose machine-local value is not part
# of an authenticated identity or numerical result.  The patterns are relative
# to ``measurement``; ``*`` matches one sequence element.  Adding a locator to
# the report schema must therefore make an explicit, reviewable change here.
_LOCATOR_POINTERS = (
    ("artifact", "path"),
    ("artifact", "log_path"),
    # Both supported and scope-unavailable original-AmpliCol measurements
    # attach their authenticated structural proof at this artifact locator.
    ("artifact", "legacy_structural_proof"),
    ("provenance", "worker_log"),
    ("provenance", "repository"),
    ("provenance", "requested_config", "model", "source"),
    ("provenance", "requested_config", "model", "cache_dir"),
    ("provenance", "effective_config", "model", "source"),
    ("provenance", "effective_config", "model", "cache_dir"),
    ("provenance", "worker_environment", "LD_LIBRARY_PATH"),
    ("provenance", "worker_environment", "DYLD_LIBRARY_PATH"),
    ("provenance", "commands", "*", "cwd"),
    # Legacy command records are diagnostic, not authenticated evidence.  Their
    # executable, first operand, and the all-flow probe's process/momenta file
    # operands are the only schema positions that carry absolute launch paths.
    ("provenance", "commands", "*", "args", 0),
    ("provenance", "commands", "*", "args", 1),
    ("provenance", "commands", "*", "args", 5),
    ("provenance", "commands", "*", "args", 6),
    ("provenance", "commands", "*", "environment", "LD_LIBRARY_PATH"),
    ("provenance", "commands", "*", "environment", "DYLD_LIBRARY_PATH"),
    ("provenance", "runtime_profile", "measurement", "args", 0),
    ("provenance", "runtime_profile", "measurement", "args", 5),
    ("provenance", "runtime_profile", "measurement", "args", 6),
    ("provenance", "runtime_profile", "measurement", "cwd"),
    (
        "provenance",
        "runtime_profile",
        "measurement",
        "environment",
        "LD_LIBRARY_PATH",
    ),
    (
        "provenance",
        "runtime_profile",
        "measurement",
        "environment",
        "DYLD_LIBRARY_PATH",
    ),
    (
        "provenance",
        "runtime_profile",
        "measurement",
        "chunks",
        "*",
        "args",
        0,
    ),
    (
        "provenance",
        "runtime_profile",
        "measurement",
        "chunks",
        "*",
        "args",
        5,
    ),
    (
        "provenance",
        "runtime_profile",
        "measurement",
        "chunks",
        "*",
        "args",
        6,
    ),
    ("provenance", "runtime_profile", "measurement", "chunks", "*", "cwd"),
    (
        "provenance",
        "runtime_profile",
        "measurement",
        "chunks",
        "*",
        "environment",
        "LD_LIBRARY_PATH",
    ),
    (
        "provenance",
        "runtime_profile",
        "measurement",
        "chunks",
        "*",
        "environment",
        "DYLD_LIBRARY_PATH",
    ),
    ("provenance", "runtime_profile", "warmup", "args", 0),
    ("provenance", "runtime_profile", "warmup", "args", 5),
    ("provenance", "runtime_profile", "warmup", "args", 6),
    ("provenance", "runtime_profile", "warmup", "cwd"),
    (
        "provenance",
        "runtime_profile",
        "warmup",
        "environment",
        "LD_LIBRARY_PATH",
    ),
    (
        "provenance",
        "runtime_profile",
        "warmup",
        "environment",
        "DYLD_LIBRARY_PATH",
    ),
)

_SCHEME_URI_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s]+")
_FILE_URI_RE = re.compile(r"file:(?://)?/[^\r\n]*", flags=re.IGNORECASE)
_PATH_PREFIX = r"(?P<prefix>^|[\s:;,\"'=()\[\]{}@]|-[A-Za-z])"
_WINDOWS_PATH_RE = re.compile(
    _PATH_PREFIX + r"(?P<path>[A-Z]:[\\/][^\s:;,\"'=\]\[(){}]*)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_UNC_PATH_RE = re.compile(
    _PATH_PREFIX + r"(?P<path>(?:\\\\|//)[^\s:;,\"'=\]\[(){}]+)",
    flags=re.MULTILINE,
)
_POSIX_PATH_RE = re.compile(
    _PATH_PREFIX + r"(?P<path>/(?!/)[^\s:;,\"'=\]\[(){}]*)",
    flags=re.MULTILINE,
)
_KNOWN_ROOT_SUFFIX = r"(?=$|[/\\:;,\s\"'=)\]}\[])"


class PublicationPortabilityError(ValueError):
    """A publication value contains a non-whitelisted machine-local path."""


def _known_roots(paths: ReportPaths) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                (paths.artifact_root.as_posix(), _ARTIFACT_ROOT),
                (paths.coordination_root.as_posix(), _COORDINATION_ROOT),
                (paths.repo_root.as_posix(), _SOURCE_ROOT),
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
    )


def _measurement_pointer(
    pointer: tuple[str | int, ...],
) -> tuple[str | int, ...]:
    if (
        len(pointer) >= 3
        and pointer[0] == "entries"
        and isinstance(pointer[1], int)
        and pointer[2] == "measurement"
    ):
        return pointer[3:]
    if pointer and pointer[0] == "measurement":
        return pointer[1:]
    return pointer


def _pointer_matches(
    pointer: tuple[str | int, ...],
    pattern: tuple[str | int, ...],
) -> bool:
    return len(pointer) == len(pattern) and all(
        expected == "*" or observed == expected
        for observed, expected in zip(pointer, pattern, strict=True)
    )


def _locator_pointer(pointer: tuple[str | int, ...]) -> bool:
    relative = _measurement_pointer(pointer)
    return any(_pointer_matches(relative, pattern) for pattern in _LOCATOR_POINTERS)


def _canonical_bytes(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return b""
    return rendered.encode("ascii")


def _digest_matches(value: object, digest: object) -> bool:
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        return False
    payload = _canonical_bytes(value)
    return bool(payload) and hashlib.sha256(payload).hexdigest() == digest


def _digest_covered_keys(
    value: Mapping[object, object],
    pointer: tuple[str | int, ...],
) -> frozenset[str]:
    """Return authenticated sibling objects that publication must not rewrite."""

    rendered = {str(key): item for key, item in value.items()}
    protected = {
        key
        for key, item in rendered.items()
        if _digest_matches(item, rendered.get(f"{key}_sha256"))
    }
    relative = _measurement_pointer(pointer)
    postflight_key = "runtime_identity_postflight_loaded_module_origin_policy"
    if relative == ("provenance",):
        postflight = rendered.get(postflight_key)
        if isinstance(postflight, Mapping) and _digest_matches(
            postflight.get("observations"),
            postflight.get("observations_sha256"),
        ):
            protected.add(postflight_key)
    return frozenset(protected)


def _json_pointer(pointer: tuple[str | int, ...]) -> str:
    return "/" + "/".join(
        str(part).replace("~", "~0").replace("/", "~1") for part in pointer
    )


def _replace_known_root(value: str, local: str, portable: str) -> str:
    # The prefix prevents a root such as /repo from matching the suffix of
    # /other/repo.  The suffix prevents /repo from corrupting /repo-sibling.
    prefix = r"(?P<prefix>^|[\s:;,\"'=()\[\]{}@]|-[A-Za-z])"
    pattern = re.compile(
        prefix + rf"(?P<root>{re.escape(local)}){_KNOWN_ROOT_SUFFIX}",
        flags=re.MULTILINE,
    )
    return pattern.sub(
        lambda match: f"{match.group('prefix')}{portable}",
        value,
    )


def _portable_locator_string(value: str, paths: ReportPaths) -> str:
    result = value
    for local, portable in _known_roots(paths):
        result = _replace_known_root(result, local, portable)
        windows_local = local.replace("/", "\\")
        if windows_local != local:
            result = _replace_known_root(result, windows_local, portable)
    if _absolute_path_fragments(result):
        # Unknown local roots are deliberately irreversible.  Redact the whole
        # locator field instead of attempting a lossy partial path parser.
        return _REDACTED_ROOT
    return result


def _portable_publication_value(
    value: object,
    paths: ReportPaths,
    *,
    pointer: tuple[str | int, ...],
) -> object:
    if isinstance(value, str):
        if _locator_pointer(pointer):
            return _portable_locator_string(value, paths)
        fragments = _absolute_path_fragments(value)
        if fragments:
            raise PublicationPortabilityError(
                "absolute path outside publication locator allowlist at "
                f"{_json_pointer(pointer)}: {fragments[0]!r}"
            )
        return value
    if isinstance(value, Mapping):
        protected = _digest_covered_keys(value, pointer)
        result: dict[str, object] = {}
        for key, item in value.items():
            rendered_key = str(key)
            key_fragments = _absolute_path_fragments(rendered_key)
            if key_fragments:
                raise PublicationPortabilityError(
                    "absolute path in publication mapping key at "
                    f"{_json_pointer((*pointer, rendered_key))}: "
                    f"{key_fragments[0]!r}"
                )
            result[rendered_key] = (
                item
                if rendered_key in protected
                else _portable_publication_value(
                    item,
                    paths,
                    pointer=(*pointer, rendered_key),
                )
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            _portable_publication_value(
                item,
                paths,
                pointer=(*pointer, index),
            )
            for index, item in enumerate(value)
        ]
    return value


def portable_publication_value(value: object, paths: ReportPaths) -> object:
    """Project approved local locators and reject every unknown absolute path.

    Digest-verified sibling objects and the authenticated postflight
    loaded-origin policy are copied byte-identically.  Every absolute path
    outside those opaque evidence objects must occur at an explicit locator
    pointer above or publication fails closed.
    """

    return _portable_publication_value(value, paths, pointer=())


def resolve_publication_string(value: str, paths: ReportPaths) -> str:
    """Resolve reversible publication placeholders on the current machine."""

    replacements = (
        (_ARTIFACT_ROOT, paths.artifact_root.as_posix()),
        (_COORDINATION_ROOT, paths.coordination_root.as_posix()),
        (_SOURCE_ROOT, paths.repo_root.as_posix()),
    )
    result = value
    for portable, local in replacements:
        result = result.replace(portable, local)
    return result


def resolve_publication_path(value: str, paths: ReportPaths) -> Path:
    """Resolve one rooted portable path and enforce root containment."""

    if _REDACTED_ROOT in value:
        raise ValueError("redacted publication path cannot be resolved")
    roots = (
        (_ARTIFACT_ROOT, paths.artifact_root),
        (_COORDINATION_ROOT, paths.coordination_root),
        (_SOURCE_ROOT, paths.repo_root),
    )
    for marker, raw_root in roots:
        if value == marker:
            relative = ""
        elif value.startswith(f"{marker}/"):
            relative = value[len(marker) + 1 :]
        else:
            continue
        logical = PurePosixPath(relative)
        if relative and (
            logical.is_absolute()
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in logical.parts)
            or logical.as_posix() != relative
        ):
            raise ValueError(f"publication path is not canonical: {value!r}")
        root = raw_root.expanduser().resolve(strict=False)
        resolved = root.joinpath(*logical.parts).resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"publication path escapes its declared root: {value!r}"
            ) from error
        return resolved
    raise ValueError(
        f"publication path must begin with one recognized root placeholder: {value!r}"
    )


def publication_measurement_matches_current(
    measurement: Mapping[str, object],
    current: Mapping[str, object],
    paths: ReportPaths,
) -> bool:
    """Compare a portable cache with an already-authenticated raw result.

    The caller is responsible for authenticating ``current`` before invoking
    this projection.  Neither input is mutated.
    """

    portable = portable_publication_value(current, paths)
    return isinstance(portable, Mapping) and dict(measurement) == dict(portable)


def _absolute_path_fragments(value: str) -> tuple[str, ...]:
    sanitized = value
    for marker in _PORTABLE_ROOTS:
        sanitized = sanitized.replace(marker, "PORTABLE_ROOT")
    file_uris = tuple(_FILE_URI_RE.findall(sanitized))
    without_file_uris = _FILE_URI_RE.sub("", sanitized)
    without_urls = _SCHEME_URI_RE.sub("", without_file_uris)
    stripped = without_urls.strip()
    direct: tuple[str, ...] = ()
    if (
        stripped.startswith(("/", "\\\\"))
        or re.match(r"^[A-Za-z]:[\\/]", stripped) is not None
    ):
        direct = (stripped,)
        without_urls = ""
    fragments = (
        *file_uris,
        *direct,
        *(match.group("path") for match in _WINDOWS_PATH_RE.finditer(without_urls)),
        *(match.group("path") for match in _UNC_PATH_RE.finditer(without_urls)),
        *(match.group("path") for match in _POSIX_PATH_RE.finditer(without_urls)),
    )
    return tuple(dict.fromkeys(fragment for fragment in fragments if fragment))


def _publication_absolute_paths(
    value: object,
    *,
    pointer: tuple[str | int, ...],
) -> tuple[str, ...]:
    if isinstance(value, str):
        return _absolute_path_fragments(value)
    if isinstance(value, Mapping):
        protected = _digest_covered_keys(value, pointer)
        return tuple(
            fragment
            for key, item in value.items()
            if str(key) not in protected
            for fragment in (
                *_publication_absolute_paths(
                    str(key),
                    pointer=(*pointer, str(key), "<key>"),
                ),
                *_publication_absolute_paths(
                    item,
                    pointer=(*pointer, str(key)),
                ),
            )
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(
            fragment
            for index, item in enumerate(value)
            for fragment in _publication_absolute_paths(
                item,
                pointer=(*pointer, index),
            )
        )
    return ()


def publication_absolute_paths(value: object) -> tuple[str, ...]:
    """Return host paths outside authenticated digest-covered objects."""

    return _publication_absolute_paths(value, pointer=())


__all__ = [
    "PublicationPortabilityError",
    "portable_publication_value",
    "publication_absolute_paths",
    "publication_measurement_matches_current",
    "resolve_publication_path",
    "resolve_publication_string",
]
