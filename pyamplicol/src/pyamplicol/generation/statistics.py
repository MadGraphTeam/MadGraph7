# SPDX-License-Identifier: 0BSD
"""Wall time and peak memory of a generation call, and their artifact record.

``measure_generation`` wraps one ``Generator.generate`` call: the monotonic
wall clock, the disjoint top-level phases the generation reports
(``progress.collect_phase_times``) and the peak resident memory of the
process tree (``reporting.resources.process_tree_peak_memory``).
``record_generation_statistics`` then stores them in the written artifact's
``artifact.json`` (``extensions.generation.statistics``): extension metadata,
outside the runtime-payload identity, so ``artifact_id`` is unchanged.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import AbstractContextManager
from pathlib import Path

from pyamplicol.api.errors import ArtifactError
from pyamplicol.api.results import GenerationStatistics
from pyamplicol.reporting.resources import process_tree_peak_memory

from .progress import PhaseTimes, collect_phase_times

_LOGGER = logging.getLogger(__name__)
STATISTICS_KEY = "statistics"


class measure_generation:
    """Measure the ``with`` block as one generation; ``statistics`` is set
    when it exits (also after an error, then unused). The measurement adds
    a daemon thread that reads the process tree's resident sizes every
    ``interval_seconds`` and a few file reads at the start and the end."""

    def __init__(self, *, interval_seconds: float = 1.0) -> None:
        self.statistics: GenerationStatistics | None = None
        self._interval = interval_seconds
        self._started = 0.0
        self._memory = process_tree_peak_memory(interval_seconds=interval_seconds)
        self._phases: AbstractContextManager[PhaseTimes] = collect_phase_times()
        self._times: PhaseTimes | None = None

    def __enter__(self) -> measure_generation:
        self._memory.__enter__()
        self._times = self._phases.__enter__()
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        wall = time.perf_counter() - self._started
        self._phases.__exit__(None, None, None)
        self._memory.__exit__(None, None, None)
        phases = () if self._times is None else tuple(self._times.seconds.items())
        memory = self._memory.result
        process = None if memory is None else memory.process
        self.statistics = GenerationStatistics(
            wall_seconds=wall,
            phases=phases,
            unattributed_seconds=max(wall - sum(s for _, s in phases), 0.0),
            peak_rss_bytes=None if memory is None else memory.peak_rss_bytes,
            peak_rss_bound="unknown" if memory is None else memory.bound,
            peak_rss_method="unavailable" if memory is None else memory.method,
            process_peak_rss_bytes=None if process is None else process.peak_rss_bytes,
            child_processes=None if memory is None else memory.child_processes,
            child_peak_rss_bytes=(
                None if memory is None else memory.child_peak_rss_bytes
            ),
        )


def record_generation_statistics(
    artifact: str | Path, statistics: GenerationStatistics
) -> bool:
    """Store ``statistics`` in the artifact's manifest, under its generation
    lock, atomically; False (with a warning) when the artifact cannot be
    updated: the generation itself has succeeded."""

    from pyamplicol.artifacts.manifest import (
        MANIFEST_NAME,
        ArtifactManifest,
        canonical_manifest_bytes,
    )
    from pyamplicol.artifacts.process_groups import _artifact_lock, _replace_file
    from pyamplicol.artifacts.security import confined_path, fsync_directory

    try:
        root = Path(artifact).expanduser().resolve(strict=True)
        with _artifact_lock(root):
            manifest_path = confined_path(root, MANIFEST_NAME)
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            extensions = raw["extensions"]
            generation = extensions.get("generation")
            if not isinstance(generation, dict):
                generation = extensions["generation"] = {}
            generation[STATISTICS_KEY] = statistics.to_json()
            ArtifactManifest.from_mapping(root, raw)
            _replace_file(manifest_path, canonical_manifest_bytes(raw))
            fsync_directory(root)
    except (ArtifactError, KeyError, OSError, TypeError, ValueError) as exc:
        _LOGGER.warning("generation statistics not recorded in %s: %s", artifact, exc)
        return False
    return True


def stored_generation_statistics(
    extensions: object,
) -> GenerationStatistics | None:
    """The statistics in a manifest's ``extensions``; None for an artifact
    written before they were recorded."""

    from collections.abc import Mapping

    if not isinstance(extensions, Mapping):
        return None
    generation = extensions.get("generation")
    if not isinstance(generation, Mapping):
        return None
    return GenerationStatistics.from_json(generation.get(STATISTICS_KEY))


__all__ = [
    "STATISTICS_KEY",
    "measure_generation",
    "record_generation_statistics",
    "stored_generation_statistics",
]
