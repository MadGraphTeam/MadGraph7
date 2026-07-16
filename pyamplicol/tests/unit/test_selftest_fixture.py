# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/release/prepare_selftest_fixture.py"
FIXTURE = ROOT / "src/pyamplicol/assets/selftest/portable-64le"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_selftest_fixture", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolved_reduction_accepts_floating_point_roundoff() -> None:
    module = _module()

    assert module._complex_sequences_close(
        [0.3298449642209169 + 0j],
        [0.32984496422091686 + 0j],
    )
    assert not module._complex_sequences_close([1.0 + 0j], [1.001 + 0j])
    assert not module._complex_sequences_close([1.0 + 0j], [])


def test_generation_output_sanitizer_accepts_programmatic_config() -> None:
    module = _module()

    programmatic = '[generation]\nmode = "error"\n'
    assert module._sanitize_generation_output(programmatic, "x") == programmatic
    assert (
        module._sanitize_generation_output(
            '[generation]\noutput = "/private/tmp/build"\n', "x"
        )
        == '[generation]\noutput = "."\n'
    )


def test_generation_output_sanitizer_rejects_ambiguous_config() -> None:
    module = _module()

    text = 'output = "/first"\noutput = "/second"\n'
    with pytest.raises(RuntimeError, match="multiple generation outputs"):
        module._sanitize_generation_output(text, "configuration-effective")


def test_portable_manifest_retargeting_rewrites_every_target_tag() -> None:
    module = _module()
    manifest = {
        "producer": {
            "target": {"triple": "aarch64-apple-darwin", "cpu_features": ["x"]}
        },
        "payloads": [
            {"path": "metadata.json"},
            {
                "path": "stage.symjit",
                "target": {
                    "triple": "aarch64-apple-darwin",
                    "cpu_features": ["neon"],
                },
            },
        ],
    }

    module._retarget_portable_manifest(
        manifest,
        source_target="aarch64-apple-darwin",
    )

    expected_target = {"triple": "portable-64le", "cpu_features": []}
    assert manifest["producer"]["target"] == expected_target
    assert manifest["payloads"][1]["target"] == expected_target


def test_source_selftest_fixture_is_one_portable_64bit_template() -> None:
    module = _module()
    expected = json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (FIXTURE / "artifact/artifact.json").read_text(encoding="utf-8")
    )

    assert expected["target"] == module.PORTABLE_TEMPLATE
    assert expected["compatible_targets"] == list(module.COMPATIBLE_TARGETS)
    assert expected["serialization"] == {
        "endianness": "little",
        "kind": "symjit-application-mir-v3",
        "load_behavior": "recompile-mir-for-loading-host",
        "word_size_bits": 64,
    }
    assert manifest["producer"]["target"] == {
        "cpu_features": [],
        "triple": module.PORTABLE_TEMPLATE,
    }
    assert {
        payload["target"]["triple"]
        for payload in manifest["payloads"]
        if "target" in payload
    } == {module.PORTABLE_TEMPLATE}
    content = dict(manifest)
    claimed = content.pop("artifact_id")
    assert claimed == hashlib.sha256(module._canonical_json(content)).hexdigest()
