# SPDX-License-Identifier: 0BSD
"""Lazy access to the loop core shipped in the ordinary native extension."""

from __future__ import annotations

import importlib
from typing import Any

from pyamplicol._internal.versions import verify_native_module
from pyamplicol.api.errors import DependencyError


def load_loop_native() -> Any:
    """Use the existing native build identity and its registered loop module."""
    try:
        native = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(native)
    except ImportError as error:
        raise DependencyError(
            "the pyamplicol._rusticol native runtime extension is unavailable"
        ) from error
    module = getattr(native, "_loop", None)
    if module is None:
        raise DependencyError(
            "the pyamplicol._rusticol extension lacks loop support; "
            "rebuild the current package"
        )
    return module
