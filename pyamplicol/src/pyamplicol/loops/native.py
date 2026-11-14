# SPDX-License-Identifier: 0BSD
"""Lazy access to the loop core shipped in the ordinary native extension."""

from __future__ import annotations

import importlib
from typing import Any

from pyamplicol._internal.versions import verify_native_module
from pyamplicol.api.errors import DependencyError

# The extension modules whose identity was verified: a source checkout's
# verification hashes the ~100 MB extension (about 0.5 s), and generation
# loads the loop module from every native stage; an imported module cannot
# change, so it is verified once per interpreter.
_VERIFIED: list[Any] = []


def load_loop_native() -> Any:
    """Use the existing native build identity and its registered loop module."""
    try:
        native = importlib.import_module("pyamplicol._rusticol")
        if not any(native is verified for verified in _VERIFIED):
            verify_native_module(native)
            _VERIFIED.append(native)
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
