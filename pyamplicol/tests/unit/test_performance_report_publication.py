# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import shlex
from copy import deepcopy
from pathlib import Path

import pytest

from tools.performance_report.publication import (
    PORTABLE_CURRENT_REPRODUCTION_RECIPE_ABI,
    PublicationPortabilityError,
    materialize_current_value,
    portable_current_value,
    portable_publication_value,
    publication_absolute_paths,
    publication_measurement_matches_current,
    resolve_publication_path,
    resolve_publication_string,
)
from tools.performance_report.service import ReportPaths


def _paths(tmp_path: Path, profile: str = "macbook_M3") -> ReportPaths:
    repo = tmp_path / "source"
    return ReportPaths.from_repo(repo, profile=profile)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _identity_payload() -> tuple[dict[str, object], dict[str, object]]:
    observations: list[dict[str, object]] = [
        {
            "module": "pyamplicol",
            "kind": "package-member",
            "root_index": 0,
            "path": "__init__.py",
            "size": 12,
            "sha256": "a" * 64,
        },
        {
            "module": "pyamplicol._rusticol",
            "kind": "native-extension",
            "path": "_rusticol.cpython.so",
            "size": 34,
            "sha256": "b" * 64,
        },
    ]
    origin_policy: dict[str, object] = {
        "kind": "pyamplicol-loaded-module-origin-policy-v1",
        "all_loaded_origins_authenticated": True,
        "native_image_origin_bound": True,
        "loaded_bytecode_eligible": False,
        "observed_module_count": len(observations),
        "observations": observations,
        "observations_sha256": _digest(observations),
    }
    candidate = {
        "source_revision": "c" * 40,
        "source_checkout": "source-checkout",
        "native_build_inputs_sha256": "d" * 64,
    }
    identity: dict[str, object] = {
        "kind": "pyamplicol-report-runtime-identity-v1",
        "candidate_build_identity": candidate,
        "candidate_build_identity_sha256": _digest(candidate),
        "python_package_tree": {
            "kind": "pyamplicol-python-package-tree-v2",
            "root": "pyamplicol",
            "roots": ["pyamplicol"],
            "sha256": "e" * 64,
        },
        "native_extension": {
            "path": "_rusticol.cpython.so",
            "sha256": "f" * 64,
        },
        "loaded_module_origin_policy": origin_policy,
    }
    return identity, deepcopy(origin_policy)


def test_projection_changes_only_explicit_unhashed_locators_and_preserves_digests(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    runtime_identity, postflight_policy = _identity_payload()
    candidate = runtime_identity["candidate_build_identity"]
    assert isinstance(candidate, dict)
    candidate["source_checkout"] = str(paths.repo_root)
    runtime_identity["candidate_build_identity_sha256"] = _digest(candidate)
    package_tree = runtime_identity["python_package_tree"]
    assert isinstance(package_tree, dict)
    package_tree["root"] = str(paths.repo_root / "src/pyamplicol")
    package_tree["roots"] = [str(paths.repo_root / "src/pyamplicol")]
    native = runtime_identity["native_extension"]
    assert isinstance(native, dict)
    native["path"] = str(paths.repo_root / "native/_rusticol.so")
    initial_policy = runtime_identity["loaded_module_origin_policy"]
    assert isinstance(initial_policy, dict)
    initial_observations = initial_policy["observations"]
    assert isinstance(initial_observations, list)
    initial_observations[1]["path"] = str(  # type: ignore[index]
        paths.repo_root / "native/_rusticol.so"
    )
    initial_policy["observations_sha256"] = _digest(initial_observations)
    postflight_observations = postflight_policy["observations"]
    assert isinstance(postflight_observations, list)
    postflight_observations[1]["path"] = str(  # type: ignore[index]
        paths.repo_root / "native/_rusticol.so"
    )
    postflight_policy["observations_sha256"] = _digest(postflight_observations)
    runtime_before = _canonical_bytes(runtime_identity)
    postflight_before = _canonical_bytes(postflight_policy)
    payload = {
        "status": "ok",
        "artifact": {
            "path": str(paths.artifact_root / "cells/a/attempts/b/artifact"),
            "log_path": str(paths.artifact_root / "cells/a/attempts/b/legacy.log"),
        },
        "provenance": {
            "requested_config": {
                "model": {
                    "source": str(paths.repo_root / "models/sm.json"),
                    "cache_dir": str(paths.repo_root / ".cache/models"),
                }
            },
            "effective_config": {
                "model": {
                    "source": str(paths.repo_root / "models/sm.json"),
                    "cache_dir": str(paths.repo_root / ".cache/models"),
                }
            },
            "worker_log": str(paths.artifact_root / "cells/a/worker.log"),
            "runtime_identity": runtime_identity,
            "runtime_identity_sha256": _digest(runtime_identity),
            "runtime_identity_stable_sha256": _digest(runtime_identity),
            "runtime_identity_postflight_loaded_module_origin_policy": (
                postflight_policy
            ),
        },
    }

    portable = portable_publication_value(payload, paths)

    assert isinstance(portable, dict)
    assert publication_absolute_paths(portable) == ()
    assert str(portable["artifact"]["path"]).startswith(  # type: ignore[index]
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/"
    )
    provenance = portable["provenance"]
    assert isinstance(provenance, dict)
    assert str(provenance["worker_log"]).startswith(
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/"
    )
    assert _canonical_bytes(provenance["runtime_identity"]) == runtime_before
    assert (
        _canonical_bytes(
            provenance["runtime_identity_postflight_loaded_module_origin_policy"]
        )
        == postflight_before
    )
    assert (
        _digest(provenance["runtime_identity"]) == provenance["runtime_identity_sha256"]
    )
    retained_identity = provenance["runtime_identity"]
    assert isinstance(retained_identity, dict)
    candidate = retained_identity["candidate_build_identity"]
    assert isinstance(candidate, dict)
    assert _digest(candidate) == retained_identity["candidate_build_identity_sha256"]
    origin_policy = retained_identity["loaded_module_origin_policy"]
    assert isinstance(origin_policy, dict)
    assert (
        _digest(origin_policy["observations"]) == origin_policy["observations_sha256"]
    )


def test_projection_is_idempotent_for_a_complete_legacy_locator_shape(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    probe_args = [
        str(paths.repo_root / "dependencies/legacy/amplicol_color_probe"),
        "100000",
        "1",
        "1",
        "lc",
        "/private/tmp/pac-example/processes.txt",
        "/private/tmp/pac-example/momenta.dat",
        "-1",
    ]
    command = {
        "args": [
            str(paths.repo_root / ".venv/bin/python"),
            str(paths.repo_root / "dependencies/legacy/process_list.py"),
        ],
        "cwd": str(paths.artifact_root / "cells/example/artifact"),
        "environment": {
            "LD_LIBRARY_PATH": (f"{paths.artifact_root}/lib:/opt/local/lib with spaces")
        },
    }
    profile_command = {
        "args": probe_args,
        "cwd": str(paths.artifact_root / "cells/example/artifact"),
        "environment": dict(command["environment"]),
    }
    payload = {
        "artifact": {
            "path": str(paths.artifact_root / "cells/example/artifact"),
            "log_path": str(paths.artifact_root / "cells/example/legacy.log"),
        },
        "provenance": {
            "repository": str(paths.repo_root / "dependencies/legacy"),
            "commands": [command, profile_command],
            "runtime_profile": {
                "measurement": {
                    **profile_command,
                    "chunks": [profile_command],
                },
                "warmup": profile_command,
            },
            "worker_environment": {
                "LD_LIBRARY_PATH": "/opt/local/lib:/private/tmp/lib",
                "DYLD_LIBRARY_PATH": None,
            },
            "worker_log": str(paths.artifact_root / "cells/example/worker.log"),
        },
    }

    once = portable_publication_value(payload, paths)
    twice = portable_publication_value(once, paths)

    assert once == twice
    assert publication_absolute_paths(once) == ()
    assert once["provenance"]["worker_environment"]["LD_LIBRARY_PATH"] == (  # type: ignore[index]
        "${LOCAL_PATH_REDACTED}"
    )
    profile = once["provenance"]["runtime_profile"]  # type: ignore[index]
    assert profile["measurement"]["args"][5:] == [  # type: ignore[index]
        "${LOCAL_PATH_REDACTED}",
        "${LOCAL_PATH_REDACTED}",
        "-1",
    ]
    assert profile["measurement"]["chunks"][0]["args"][5:] == [  # type: ignore[index]
        "${LOCAL_PATH_REDACTED}",
        "${LOCAL_PATH_REDACTED}",
        "-1",
    ]


def test_snapshot_projection_relocates_legacy_structural_proof(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    artifact = paths.artifact_root / "cells/example/attempts/one/artifact"
    payload = {
        "entries": [
            {
                "measurement": {
                    "artifact": {
                        "path": str(artifact),
                        "legacy_structural_proof": str(
                            artifact / "legacy-structural-proof.json"
                        ),
                    }
                }
            }
        ]
    }

    portable = portable_publication_value(payload, paths)

    proof = portable["entries"][0]["measurement"]["artifact"][  # type: ignore[index]
        "legacy_structural_proof"
    ]
    assert proof == (
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/"
        "cells/example/attempts/one/artifact/legacy-structural-proof.json"
    )
    assert publication_absolute_paths(portable) == ()
    assert portable_publication_value(portable, paths) == portable


def test_unknown_legacy_argument_position_fails_closed(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    with pytest.raises(
        PublicationPortabilityError,
        match=r"/provenance/commands/0/args/2",
    ):
        portable_publication_value(
            {
                "provenance": {
                    "commands": [
                        {
                            "args": [
                                "probe",
                                "ordinary-operand",
                                "/private/tmp/not-a-schema-locator",
                            ]
                        }
                    ]
                }
            },
            paths,
        )


@pytest.mark.parametrize(
    "value",
    (
        "@/private/tmp/secret.bin",
        r"\\server\share\private\a.dll",
        "//server/share/private/a.dll",
        "file:///C:/Users/example/private/a.dll",
        "unknown=/Users/Alice Smith/private/file.so",
    ),
)
def test_unknown_absolute_path_outside_allowlist_fails_closed(
    tmp_path: Path,
    value: str,
) -> None:
    paths = _paths(tmp_path)

    with pytest.raises(
        PublicationPortabilityError,
        match=r"/diagnostic",
    ):
        portable_publication_value({"diagnostic": value}, paths)


def test_digest_covered_identity_with_absolute_locator_is_opaque_and_not_mutated(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    identity, postflight = _identity_payload()
    native = identity["native_extension"]
    assert isinstance(native, dict)
    native["path"] = "/private/runtime/_rusticol.so"
    payload = {
        "provenance": {
            "runtime_identity": identity,
            "runtime_identity_sha256": _digest(identity),
            "runtime_identity_postflight_loaded_module_origin_policy": postflight,
        }
    }
    before = deepcopy(payload)

    portable = portable_publication_value(payload, paths)

    assert payload == before
    assert portable == payload
    assert _canonical_bytes(portable["provenance"]["runtime_identity"]) == (  # type: ignore[index]
        _canonical_bytes(payload["provenance"]["runtime_identity"])  # type: ignore[index]
    )
    assert publication_absolute_paths(payload) == ()


def test_invalid_identity_digest_does_not_hide_an_absolute_path(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    identity, _postflight = _identity_payload()
    native = identity["native_extension"]
    assert isinstance(native, dict)
    native["path"] = "/private/runtime/_rusticol.so"
    payload = {
        "provenance": {
            "runtime_identity": identity,
            "runtime_identity_sha256": "0" * 64,
        }
    }

    with pytest.raises(
        PublicationPortabilityError,
        match=r"/provenance/runtime_identity/native_extension/path",
    ):
        portable_publication_value(payload, paths)

    assert publication_absolute_paths(payload) == ("/private/runtime/_rusticol.so",)


def test_any_valid_digest_covered_sibling_is_copied_byte_identically(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    evidence = {
        "path": "/private/authenticated/evidence.bin",
        "size": 123,
        "member_sha256": "a" * 64,
    }
    payload = {
        "authenticated_evidence": evidence,
        "authenticated_evidence_sha256": _digest(evidence),
    }
    before = _canonical_bytes(evidence)

    portable = portable_publication_value(payload, paths)

    assert _canonical_bytes(portable["authenticated_evidence"]) == before  # type: ignore[index]
    assert portable["authenticated_evidence"] is evidence  # type: ignore[index]
    assert publication_absolute_paths(portable) == ()


def test_known_root_replacement_observes_path_boundaries(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    sibling = f"{paths.repo_root}-sibling/models/sm.json"
    embedded_suffix = f"/unrelated{paths.repo_root}/models/sm.json"
    payload = {
        "provenance": {
            "effective_config": {
                "model": {
                    "source": sibling,
                    "cache_dir": embedded_suffix,
                }
            }
        }
    }

    portable = portable_publication_value(payload, paths)
    model = portable["provenance"]["effective_config"]["model"]  # type: ignore[index]

    assert model["source"] == "${LOCAL_PATH_REDACTED}"
    assert model["cache_dir"] == "${LOCAL_PATH_REDACTED}"
    assert "${PYAMPLICOL_SOURCE_ROOT}-sibling" not in repr(portable)


def test_resolver_relocates_known_root_and_rejects_escape_or_raw_path(
    tmp_path: Path,
) -> None:
    source_paths = _paths(tmp_path / "machine-a", "macbook_M3")
    target_paths = _paths(tmp_path / "machine-b", "cluster_EPYC")
    raw = source_paths.artifact_root / "cells/example/artifact"
    portable = portable_publication_value(
        {"artifact": {"path": str(raw)}},
        source_paths,
    )
    locator = portable["artifact"]["path"]  # type: ignore[index]
    assert isinstance(locator, str)

    assert resolve_publication_string(locator, target_paths) == str(
        target_paths.artifact_root / "cells/example/artifact"
    )
    assert resolve_publication_path(locator, target_paths) == (
        target_paths.artifact_root / "cells/example/artifact"
    )
    with pytest.raises(ValueError, match="not canonical"):
        resolve_publication_path(
            "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/../../outside",
            target_paths,
        )
    with pytest.raises(ValueError, match="recognized root"):
        resolve_publication_path("/tmp/untrusted/artifact", target_paths)
    with pytest.raises(ValueError, match="cannot be resolved"):
        resolve_publication_path(
            "${LOCAL_PATH_REDACTED}/libexample.dylib",
            target_paths,
        )


def test_raw_current_is_audited_before_projection_and_is_never_mutated(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    identity, postflight = _identity_payload()
    current = {
        "status": "ok",
        "artifact": {
            "path": str(paths.artifact_root / "cells/example/artifact"),
        },
        "provenance": {
            "runtime_identity": identity,
            "runtime_identity_sha256": _digest(identity),
            "runtime_identity_postflight_loaded_module_origin_policy": postflight,
        },
    }
    before = deepcopy(current)
    audit_order: list[str] = []

    def audit_raw(value: dict[str, object]) -> None:
        audit_order.append("raw")
        assert value["artifact"]["path"] == str(  # type: ignore[index]
            paths.artifact_root / "cells/example/artifact"
        )
        provenance = value["provenance"]
        assert isinstance(provenance, dict)
        assert (
            _digest(provenance["runtime_identity"])
            == provenance["runtime_identity_sha256"]
        )

    audit_raw(current)
    portable = portable_publication_value(current, paths)
    audit_order.append("projected")

    assert publication_measurement_matches_current(portable, current, paths)
    changed = dict(portable)
    changed["status"] = "failed"
    assert not publication_measurement_matches_current(changed, current, paths)
    assert audit_order == ["raw", "projected"]
    assert current == before


def test_portable_current_round_trip_rebases_only_approved_locators(
    tmp_path: Path,
) -> None:
    source = _paths(tmp_path / "before", "manual")
    target = _paths(tmp_path / "after", "renamed")
    identity = {"native_extension": {"path": "/authenticated/runtime.so"}}
    payload = {
        "artifact": {
            "path": str(source.artifact_root / "cells/a/attempts/id/artifact"),
            "log_path": str(source.artifact_root / "cells/a/attempts/id/worker.log"),
        },
        "provenance": {
            "worker_log": str(
                source.artifact_root / "cells/a/attempts/id/worker.log"
            ),
            "runtime_identity": identity,
            "runtime_identity_sha256": _digest(identity),
        },
    }
    identity_before = _canonical_bytes(identity)

    portable = portable_current_value(payload, source)
    materialized = materialize_current_value(portable, target)

    assert portable["artifact"]["path"] == (  # type: ignore[index]
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/cells/a/attempts/id/artifact"
    )
    assert materialized["artifact"]["path"] == str(  # type: ignore[index]
        target.artifact_root / "cells/a/attempts/id/artifact"
    )
    assert _canonical_bytes(portable["provenance"]["runtime_identity"]) == (  # type: ignore[index]
        identity_before
    )


@pytest.mark.parametrize(
    "locator",
    (
        "/tmp/old-campaign/artifact",
        "relative/artifact",
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}",
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/../escape",
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/artifact\\member",
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}//artifact",
    ),
)
def test_portable_current_rejects_legacy_or_noncanonical_artifact_locator(
    tmp_path: Path,
    locator: str,
) -> None:
    paths = _paths(tmp_path)

    with pytest.raises(PublicationPortabilityError):
        materialize_current_value({"artifact": {"path": locator}}, paths)


def test_portable_current_rejects_artifact_symlink_traversal(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.artifact_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (paths.artifact_root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(PublicationPortabilityError, match="invalid rooted path"):
        materialize_current_value(
            {
                "artifact": {
                    "path": "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/escape/payload"
                }
            },
            paths,
        )


def test_portable_current_rejects_in_root_artifact_symlink_traversal(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    target = paths.artifact_root / "real"
    target.mkdir(parents=True)
    (paths.artifact_root / "alias").symlink_to(target, target_is_directory=True)

    with pytest.raises(PublicationPortabilityError, match="symbolic link"):
        materialize_current_value(
            {
                "artifact": {
                    "path": "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/alias/payload"
                }
            },
            paths,
        )


def test_portable_current_preserves_external_installed_command_and_moves_state(
    tmp_path: Path,
) -> None:
    source = _paths(tmp_path / "before", "manual")
    target = _paths(tmp_path / "after path's", "renamed")
    external_python = "/opt/pyamplicol-venv/bin/python"
    profile_argv = (
        external_python,
        "-m",
        "pyamplicol",
        "profile",
        str(source.artifact_root / "manual-reproductions/cell/artifact"),
    )
    payload = {
        "artifact": {
            "path": str(source.artifact_root / "cells/a/attempts/id/artifact")
        },
        "provenance": {
            "manual_campaign": {
                "public_cli_reproduction": {
                    "abi": PORTABLE_CURRENT_REPRODUCTION_RECIPE_ABI,
                    "prepare": None,
                    "generate": None,
                    "profile": list(profile_argv),
                }
            }
        },
    }

    portable = portable_current_value(payload, source)
    stored_argv = portable["provenance"]["manual_campaign"][  # type: ignore[index]
        "public_cli_reproduction"
    ]["profile"]
    assert external_python in stored_argv
    assert stored_argv[-1].startswith(
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/manual-reproductions"
    )
    assert "${LOCAL_PATH_REDACTED}" not in stored_argv

    materialized = materialize_current_value(portable, target)
    moved_argv = materialized["provenance"]["manual_campaign"][  # type: ignore[index]
        "public_cli_reproduction"
    ]["profile"]
    assert external_python in moved_argv
    assert moved_argv[-1] == str(
        target.artifact_root / "manual-reproductions/cell/artifact"
    )
    assert shlex.split(shlex.join(moved_argv)) == moved_argv

    with pytest.raises(PublicationPortabilityError, match="structured ABI"):
        materialize_current_value(
            {
                "provenance": {
                    "manual_campaign": {
                        "public_cli_reproduction": {
                            "prepare": None,
                            "generate": None,
                            "profile": "old absolute command",
                        }
                    }
                }
            },
            target,
        )
