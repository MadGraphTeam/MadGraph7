# SPDX-License-Identifier: 0BSD
"""Translate exact request-file objects to the existing grouped API records."""

from __future__ import annotations

import re
from collections.abc import Mapping

from pyamplicol.config import ConfigurationError
from pyamplicol.correlators import CorrelatedRequest


def correlated_requests_from_json(value: object) -> dict[str, CorrelatedRequest]:
    if not isinstance(value, Mapping):
        raise ConfigurationError("correlated requests JSON must be an object")
    requests = {}
    for label, request in value.items():
        if not isinstance(label, str) or not label:
            raise ConfigurationError(
                "correlated request labels must be nonempty strings"
            )
        context = f"correlated request {label!r}"
        if not isinstance(request, Mapping):
            raise ConfigurationError(f"{context} must be an object")
        unknown = set(request) - {"color_correlation", "spin_vectors"}
        if unknown:
            raise ConfigurationError(
                f"unknown {context} fields: {sorted(unknown, key=str)}"
            )
        identifier = request.get("color_correlation", "born")
        if not isinstance(identifier, str) or not identifier:
            raise ConfigurationError(
                f"{context}.color_correlation must be a nonempty string"
            )
        raw_vectors = request.get("spin_vectors")
        vectors = None
        if raw_vectors is not None:
            if not isinstance(raw_vectors, Mapping):
                raise ConfigurationError(
                    f"{context}.spin_vectors must be an object or null"
                )
            vectors = {}
            for key, source in raw_vectors.items():
                if (
                    not isinstance(key, str)
                    or re.fullmatch(r"[+-]?[0-9]+", key) is None
                ):
                    raise ConfigurationError(
                        f"{context} has noninteger leg label {key!r}"
                    )
                leg = int(key)
                if leg in vectors:
                    raise ConfigurationError(
                        f"{context} repeats leg {leg} through label {key!r}"
                    )
                vectors[leg] = source
        requests[label] = CorrelatedRequest(identifier, vectors)
    return requests
