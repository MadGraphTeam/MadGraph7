# SPDX-License-Identifier: 0BSD
"""Virtual exports reuse ordinary payload transport without tree ME templates."""

from pyamplicol.artifacts.api_bundle import api_bundle_payloads
from pyamplicol.generation.artifact_writer import _default_api_bundle_hook


def test_virtual_bundle_carries_every_language_and_explicit_physics_contract():
    payloads = api_bundle_payloads(
        {"p": [["5e2", "0", "0", "5e2"]]}, contribution_kind="virtual"
    )
    contents = {p.path: p.content.decode() for p in payloads}
    assert len(contents) == len(payloads)
    for language, extension in (
        ("python", "py"),
        ("c", "c"),
        ("cpp", "cpp"),
        ("fortran", "f90"),
        ("rust", "rs"),
        ("umami", "c"),
    ):
        assert f"API/{language}/check_standalone.{extension}" in contents
    assert "API/common/input.h" in contents
    assert "5e2" in contents["API/validation_points.dat"]
    assert "MU_R2" in contents["API/README.md"]
    assert "not probabilities" in contents["API/README.md"]
    assert "UMAMI_IN_MOMENTA" in contents["API/umami/check_standalone.c"]
    assert "UMAMI_OUT_MATRIX_ELEMENT" not in contents["API/umami/check_standalone.c"]
    assert "rusticol_runtime_evaluate_f64" not in contents["API/c/check_standalone.c"]


def test_virtual_default_hook_uses_existing_builder_transport():
    class Builder:
        def __init__(self):
            self.paths = []

        def add_bytes(self, path, content, **metadata):
            self.paths.append(path)
            assert content
            return {"path": path, **metadata}

    builder = Builder()
    hook = _default_api_bundle_hook(contribution_kind="virtual")
    assert hook is not None
    result = hook(builder, {"p": [[1, 0, 0, 1]]})
    assert [x["path"] for x in result] == builder.paths
    assert "API/common/input.h" in builder.paths
