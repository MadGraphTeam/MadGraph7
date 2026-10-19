# SPDX-License-Identifier: 0BSD
"""Production uses one identity-checked installed extension, without a fallback."""

from types import SimpleNamespace

import pytest

from pyamplicol.api.errors import DependencyError
from pyamplicol.loops import native


def test_loop_module_uses_existing_native_identity(monkeypatch):
    loop = object()
    extension = SimpleNamespace(_loop=loop)
    requested, checked = [], []

    def imported(name):
        requested.append(name)
        return extension

    monkeypatch.setattr(native.importlib, "import_module", imported)
    monkeypatch.setattr(native, "verify_native_module", checked.append)
    assert native.load_loop_native() is loop
    assert requested == ["pyamplicol._rusticol"]
    assert checked == [extension]


def test_missing_embedded_loop_never_uses_external_companion(monkeypatch):
    requested = []

    def imported(name):
        requested.append(name)
        return SimpleNamespace()

    monkeypatch.setattr(native.importlib, "import_module", imported)
    monkeypatch.setattr(native, "verify_native_module", lambda value: None)
    with pytest.raises(DependencyError, match="lacks loop support"):
        native.load_loop_native()
    assert requested == ["pyamplicol._rusticol"]


def test_native_identity_error_is_not_suppressed(monkeypatch):
    class IdentityError(ValueError):
        pass

    def reject(value):
        raise IdentityError("wrong native build")

    monkeypatch.setattr(
        native.importlib, "import_module", lambda name: SimpleNamespace(_loop=object())
    )
    monkeypatch.setattr(native, "verify_native_module", reject)
    with pytest.raises(IdentityError, match="wrong native build"):
        native.load_loop_native()
