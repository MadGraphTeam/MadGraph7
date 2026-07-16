# SPDX-License-Identifier: 0BSD
"""Process-wide lazy Symbolica namespace for model compilation."""

from __future__ import annotations

from threading import RLock
from typing import Any

E: Any = None
S: Any = None
Expression: Any = None
Replacement: Any = None
LibraryTensor: Any = None
Representation: Any = None
TensorLibrary: Any = None
TensorName: Any = None
TensorNetwork: Any = None
_SYMBOLICA_READY = False
_SYMBOLICA_LOCK = RLock()


def _ensure_symbolica() -> None:
    global E, S, Expression, Replacement, LibraryTensor, Representation
    global TensorLibrary, TensorName, TensorNetwork, _SYMBOLICA_READY

    if _SYMBOLICA_READY:
        return
    with _SYMBOLICA_LOCK:
        if _SYMBOLICA_READY:
            return
        from symbolica import E as expression_parser
        from symbolica import Expression as expression_type
        from symbolica import Replacement as replacement_type
        from symbolica import S as symbol
        from symbolica.community.spenso import LibraryTensor as library_tensor_type
        from symbolica.community.spenso import Representation as representation_type
        from symbolica.community.spenso import TensorLibrary as tensor_library_type
        from symbolica.community.spenso import TensorName as tensor_name_type
        from symbolica.community.spenso import TensorNetwork as tensor_network_type

        E = expression_parser
        S = symbol
        Expression = expression_type
        Replacement = replacement_type
        LibraryTensor = library_tensor_type
        Representation = representation_type
        TensorLibrary = tensor_library_type
        TensorName = tensor_name_type
        TensorNetwork = tensor_network_type
        _SYMBOLICA_READY = True
