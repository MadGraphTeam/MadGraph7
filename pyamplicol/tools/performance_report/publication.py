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
PORTABLE_CURRENT_REPRODUCTION_RECIPE_ABI = (
    "pyamplicol-portable-current-reproduction-recipe-v1"
)
_REPRODUCTION_RECIPE_POINTER = (
    "provenance",
    "manual_campaign",
    "public_cli_reproduction",
)
_REPRODUCTION_STAGES = ("prepare", "generate", "profile")

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
    (
        "provenance",
        "manual_campaign",
        "public_cli_reproduction",
        "prepare",
        "*",
    ),
    (
        "provenance",
        "manual_campaign",
        "public_cli_reproduction",
        "generate",
        "*",
    ),
    (
        "provenance",
        "manual_campaign",
        "public_cli_reproduction",
        "profile",
        "*",
    ),
)

# Generation retains its output directory in the requested/effective CLI
# configuration.  Unlike diagnostic locator fields, these values are expected
# to identify the published result's artifact and must never be silently
# redacted: an absolute path is accepted only when it is rooted in this
# profile's artifact store.
_STRICT_ARTIFACT_LOCATOR_POINTERS = (
    ("provenance", "requested_config", "generation", "output"),
    ("provenance", "effective_config", "generation", "output"),
)

# These exact-path fields are consumed when a current is reused.  Portable
# current records therefore require an explicit rooted locator here; a raw
# absolute path or a merely relative path must never be interpreted relative
# to the process working directory after a campaign has moved.
_CURRENT_ROOTED_PATH_POINTERS = (
    ("artifact", "path"),
    ("artifact", "log_path"),
    ("artifact", "legacy_structural_proof"),
    ("provenance", "worker_log"),
    *_STRICT_ARTIFACT_LOCATOR_POINTERS,
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


def _strict_artifact_locator_pointer(pointer: tuple[str | int, ...]) -> bool:
    relative = _measurement_pointer(pointer)
    return any(
        _pointer_matches(relative, pattern)
        for pattern in _STRICT_ARTIFACT_LOCATOR_POINTERS
    )


def _current_rooted_path_pointer(pointer: tuple[str | int, ...]) -> bool:
    relative = _measurement_pointer(pointer)
    return any(
        _pointer_matches(relative, pattern)
        for pattern in _CURRENT_ROOTED_PATH_POINTERS
    )


def _current_reproduction_recipe_pointer(pointer: tuple[str | int, ...]) -> bool:
    return _measurement_pointer(pointer) == _REPRODUCTION_RECIPE_POINTER


def _validated_reproduction_argv(
    value: Mapping[object, object],
) -> dict[str, tuple[str, ...] | None]:
    if value.get("abi") != PORTABLE_CURRENT_REPRODUCTION_RECIPE_ABI:
        raise PublicationPortabilityError(
            "portable current reproduction recipe has no supported structured ABI"
        )
    result: dict[str, tuple[str, ...] | None] = {}
    for stage in _REPRODUCTION_STAGES:
        raw_argv = value.get(stage)
        if raw_argv is None:
            result[stage] = None
            continue
        if (
            not isinstance(raw_argv, Sequence)
            or isinstance(raw_argv, (str, bytes, bytearray))
            or not raw_argv
            or any(
                not isinstance(argument, str) or not argument
                for argument in raw_argv
            )
        ):
            raise PublicationPortabilityError(
                f"portable current reproduction {stage} argv is malformed"
            )
        argv = tuple(raw_argv)
        result[stage] = argv
    return result


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


def _portable_artifact_locator_string(
    value: str,
    paths: ReportPaths,
    *,
    pointer: tuple[str | int, ...],
) -> str:
    """Project one exact artifact path, rejecting every external absolute root."""

    has_absolute_path = bool(_absolute_path_fragments(value))
    has_portable_root = any(marker in value for marker in _PORTABLE_ROOTS)
    if not has_absolute_path and not has_portable_root:
        return value
    portable = _portable_locator_string(value, paths)
    if not (portable == _ARTIFACT_ROOT or portable.startswith(f"{_ARTIFACT_ROOT}/")):
        raise PublicationPortabilityError(
            "artifact path outside publication artifact root at "
            f"{_json_pointer(pointer)}: {value!r}"
        )
    try:
        resolve_publication_path(portable, paths)
    except ValueError as error:
        raise PublicationPortabilityError(
            f"invalid publication artifact path at {_json_pointer(pointer)}: {value!r}"
        ) from error
    return portable


def _portable_publication_value(
    value: object,
    paths: ReportPaths,
    *,
    pointer: tuple[str | int, ...],
) -> object:
    if isinstance(value, str):
        if _strict_artifact_locator_pointer(pointer):
            return _portable_artifact_locator_string(
                value,
                paths,
                pointer=pointer,
            )
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


def _current_state_roots(paths: ReportPaths) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                (paths.artifact_root.as_posix(), _ARTIFACT_ROOT),
                (paths.coordination_root.as_posix(), _COORDINATION_ROOT),
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
    )


def _portable_current_locator_string(value: str, paths: ReportPaths) -> str:
    result = value
    for local, portable in _current_state_roots(paths):
        result = _replace_known_root(result, local, portable)
        windows_local = local.replace("/", "\\")
        if windows_local != local:
            result = _replace_known_root(result, windows_local, portable)
    return result


def _validate_current_state_tokens(value: str) -> None:
    forbidden = (_SOURCE_ROOT, _REDACTED_ROOT)
    if any(marker in value for marker in forbidden):
        raise PublicationPortabilityError(
            "portable current contains a non-state publication locator"
        )
    delimiters = frozenset(" \t\r\n:;,\"'=()[]{}@")
    for marker in (_ARTIFACT_ROOT, _COORDINATION_ROOT):
        offset = 0
        while True:
            start = value.find(marker, offset)
            if start < 0:
                break
            suffix_start = start + len(marker)
            if suffix_start >= len(value) or value[suffix_start] != "/":
                raise PublicationPortabilityError(
                    "portable current state locator has an empty or invalid suffix"
                )
            end = suffix_start + 1
            while end < len(value) and value[end] not in delimiters:
                end += 1
            suffix = value[suffix_start + 1 : end]
            logical = PurePosixPath(suffix)
            if (
                not suffix
                or "\\" in suffix
                or logical.is_absolute()
                or any(part in {"", ".", ".."} for part in logical.parts)
                or logical.as_posix() != suffix
            ):
                raise PublicationPortabilityError(
                    "portable current state locator suffix is not canonical"
                )
            offset = suffix_start


def _current_root_for_locator(value: str, paths: ReportPaths) -> tuple[str, Path]:
    for marker, root in (
        (_ARTIFACT_ROOT, paths.artifact_root),
        (_COORDINATION_ROOT, paths.coordination_root),
    ):
        if value.startswith(marker + "/"):
            return marker, root
    raise PublicationPortabilityError(
        "portable current rooted path has no state-root locator"
    )


def _resolve_current_rooted_path(value: str, paths: ReportPaths) -> str:
    _validate_current_state_tokens(value)
    marker, raw_root = _current_root_for_locator(value, paths)
    if value.count(_ARTIFACT_ROOT) + value.count(_COORDINATION_ROOT) != 1:
        raise PublicationPortabilityError(
            "portable current rooted path must contain exactly one locator"
        )
    try:
        resolved = resolve_publication_path(value, paths)
    except ValueError as error:
        raise PublicationPortabilityError(
            f"portable current contains an invalid rooted path: {value!r}"
        ) from error
    relative = PurePosixPath(value[len(marker) + 1 :])
    lexical_root = Path(raw_root).expanduser()
    lexical_root = Path(lexical_root.absolute())
    lexical = lexical_root.joinpath(*relative.parts)
    # ``resolve(strict=False)`` follows every existing symlink component.  A
    # difference from the lexical canonical path therefore rejects even an
    # in-root symlink, not merely an escape outside the state root.
    if lexical != resolved:
        raise PublicationPortabilityError(
            f"portable current rooted path traverses a symbolic link: {value!r}"
        )
    return resolved.as_posix()


def _portable_current_value(
    value: object,
    paths: ReportPaths,
    *,
    pointer: tuple[str | int, ...],
) -> object:
    if isinstance(value, str):
        if _current_rooted_path_pointer(pointer):
            portable = _portable_current_locator_string(value, paths)
            _resolve_current_rooted_path(portable, paths)
            return portable
        if _locator_pointer(pointer):
            portable = _portable_current_locator_string(value, paths)
            _validate_current_state_tokens(portable)
            return portable
        projected = _portable_current_locator_string(value, paths)
        _validate_current_state_tokens(projected)
        return projected
    if isinstance(value, Mapping):
        if _current_reproduction_recipe_pointer(pointer):
            _validated_reproduction_argv(value)
        protected = _digest_covered_keys(value, pointer)
        return {
            str(key): (
                item
                if str(key) in protected
                else _portable_current_value(
                    item,
                    paths,
                    pointer=(*pointer, str(key)),
                )
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            _portable_current_value(
                item,
                paths,
                pointer=(*pointer, index),
            )
            for index, item in enumerate(value)
        ]
    return value


def _materialize_current_value(
    value: object,
    paths: ReportPaths,
    *,
    pointer: tuple[str | int, ...],
) -> object:
    if isinstance(value, str):
        if _current_rooted_path_pointer(pointer):
            if _absolute_path_fragments(value):
                raise PublicationPortabilityError(
                    "portable current contains a raw rooted path at "
                    f"{_json_pointer(pointer)}: {value!r}"
                )
            return _resolve_current_rooted_path(value, paths)
        if _locator_pointer(pointer):
            _validate_current_state_tokens(value)
            return resolve_publication_string(value, paths)
        _validate_current_state_tokens(value)
        if any(marker in value for marker in (_ARTIFACT_ROOT, _COORDINATION_ROOT)):
            return resolve_publication_string(value, paths)
        return value
    if isinstance(value, Mapping):
        if _current_reproduction_recipe_pointer(pointer):
            _validated_reproduction_argv(value)
        protected = _digest_covered_keys(value, pointer)
        return {
            str(key): (
                item
                if str(key) in protected
                else _materialize_current_value(
                    item,
                    paths,
                    pointer=(*pointer, str(key)),
                )
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            _materialize_current_value(
                item,
                paths,
                pointer=(*pointer, index),
            )
            for index, item in enumerate(value)
        ]
    return value


def portable_current_value(value: object, paths: ReportPaths) -> object:
    """Project one newly sealed current into the portable locator ABI.

    The ordinary publication projection remains unchanged.  This additional
    boundary proves that every state path needed by current reuse is an exact,
    canonical artifact/coordination-root locator before immutable result bytes
    are written.
    """

    portable = _portable_current_value(value, paths, pointer=())
    _materialize_current_value(portable, paths, pointer=())
    return portable


def materialize_current_value(value: object, paths: ReportPaths) -> object:
    """Resolve one authenticated portable current against its present roots.

    This deliberately does not migrate historical absolute-path records.  A
    campaign current either uses the portable locator ABI in its stored bytes
    or fails closed.
    """

    return _materialize_current_value(value, paths, pointer=())


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
    "PORTABLE_CURRENT_REPRODUCTION_RECIPE_ABI",
    "PublicationPortabilityError",
    "materialize_current_value",
    "portable_current_value",
    "portable_publication_value",
    "publication_absolute_paths",
    "publication_measurement_matches_current",
    "resolve_publication_path",
    "resolve_publication_string",
]
