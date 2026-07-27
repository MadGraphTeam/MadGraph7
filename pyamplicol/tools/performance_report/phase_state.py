# SPDX-License-Identifier: 0BSD
"""Authenticated atomic phase state for one performance-report worker."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

WORKER_PHASE_STATE_ABI = "pyamplicol-report-worker-phase-state-v1"
WORKER_PHASE_AUTHENTICATION = "hmac-sha256"

WorkerPhase = Literal["pre-generation", "generation", "post-generation"]

_PHASE_FIELDS = frozenset(
    {
        "abi",
        "run_id",
        "worker_pid",
        "sequence",
        "phase",
        "transition_monotonic_ns",
        "generation_started_monotonic_ns",
        "generation_finished_monotonic_ns",
    }
)
_STATE_FIELDS = _PHASE_FIELDS | {"authentication"}


class WorkerPhaseStateError(RuntimeError):
    """Raised when worker phase evidence is absent, malformed, or unauthenticated."""


@dataclass(frozen=True, slots=True)
class WorkerPhaseChannel:
    """Parent-issued credentials for one worker phase-state file."""

    path: Path
    run_id: str
    authentication_key: str

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ValueError("worker phase-state path must be absolute")
        if len(self.run_id) != 32 or any(
            character not in "0123456789abcdef" for character in self.run_id
        ):
            raise ValueError("worker phase-state run_id must be 128-bit lowercase hex")
        if len(self.authentication_key) != 64 or any(
            character not in "0123456789abcdef" for character in self.authentication_key
        ):
            raise ValueError(
                "worker phase-state authentication key must be 256-bit lowercase hex"
            )

    @classmethod
    def create(cls, path: Path) -> WorkerPhaseChannel:
        """Create fresh credentials bound to ``path``."""

        return cls(
            path=path.expanduser().resolve(strict=False),
            run_id=secrets.token_hex(16),
            authentication_key=secrets.token_hex(32),
        )


@dataclass(frozen=True, slots=True)
class WorkerPhaseState:
    """One authenticated phase transition written by the worker."""

    run_id: str
    worker_pid: int
    sequence: int
    phase: WorkerPhase
    transition_monotonic_ns: int
    generation_started_monotonic_ns: int | None
    generation_finished_monotonic_ns: int | None
    sha256: str

    def generation_elapsed_seconds(self, *, now_seconds: float) -> float | None:
        """Return generation duration through the authenticated boundary."""

        if self.generation_started_monotonic_ns is None:
            return None
        stop_ns = self.generation_finished_monotonic_ns
        if stop_ns is None:
            stop_ns = max(
                int(now_seconds * 1_000_000_000),
                self.generation_started_monotonic_ns,
            )
        return (stop_ns - self.generation_started_monotonic_ns) / 1_000_000_000


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _authentication_digest(
    payload: Mapping[str, object],
    authentication_key: str,
) -> str:
    return hmac.new(
        bytes.fromhex(authentication_key),
        _canonical_json(payload),
        hashlib.sha256,
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _unsigned_state(
    *,
    channel: WorkerPhaseChannel,
    worker_pid: int,
    sequence: int,
    phase: WorkerPhase,
    transition_monotonic_ns: int,
    generation_started_monotonic_ns: int | None,
    generation_finished_monotonic_ns: int | None,
) -> dict[str, object]:
    return {
        "abi": WORKER_PHASE_STATE_ABI,
        "run_id": channel.run_id,
        "worker_pid": worker_pid,
        "sequence": sequence,
        "phase": phase,
        "transition_monotonic_ns": transition_monotonic_ns,
        "generation_started_monotonic_ns": generation_started_monotonic_ns,
        "generation_finished_monotonic_ns": generation_finished_monotonic_ns,
    }


class WorkerPhaseReporter:
    """Publish the worker's one allowed generation interval."""

    def __init__(
        self,
        channel: WorkerPhaseChannel,
        *,
        worker_pid: int | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.channel = channel
        self.worker_pid = os.getpid() if worker_pid is None else worker_pid
        if self.worker_pid <= 0:
            raise ValueError("worker phase-state PID must be positive")
        self._clock_ns = clock_ns
        self._phase: WorkerPhase = "pre-generation"
        self._sequence = 0
        self._generation_started_ns: int | None = None
        self._write(self._clock())

    @property
    def phase(self) -> WorkerPhase:
        return self._phase

    def _clock(self) -> int:
        value = self._clock_ns()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WorkerPhaseStateError(
                "worker phase-state monotonic clock must return non-negative int"
            )
        return value

    def _write(self, transition_ns: int) -> None:
        finished_ns = transition_ns if self._phase == "post-generation" else None
        unsigned = _unsigned_state(
            channel=self.channel,
            worker_pid=self.worker_pid,
            sequence=self._sequence,
            phase=self._phase,
            transition_monotonic_ns=transition_ns,
            generation_started_monotonic_ns=self._generation_started_ns,
            generation_finished_monotonic_ns=finished_ns,
        )
        _atomic_json(
            self.channel.path,
            {
                **unsigned,
                "authentication": {
                    "kind": WORKER_PHASE_AUTHENTICATION,
                    "digest": _authentication_digest(
                        unsigned,
                        self.channel.authentication_key,
                    ),
                },
            },
        )

    @contextmanager
    def generation(self) -> Iterator[None]:
        """Mark one local generation-work interval.

        The interval may be empty when the worker authenticates reuse of an
        artifact generated by an earlier measurement.
        """

        if self._phase != "pre-generation":
            raise WorkerPhaseStateError(
                "worker phase reporter permits exactly one generation interval"
            )
        started_ns = self._clock()
        self._phase = "generation"
        self._sequence = 1
        self._generation_started_ns = started_ns
        self._write(started_ns)
        try:
            yield
        finally:
            finished_ns = max(self._clock(), started_ns)
            self._phase = "post-generation"
            self._sequence = 2
            self._write(finished_ns)


def _required_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkerPhaseStateError(f"{name} must be a non-negative integer")
    return value


def _validated_timing(
    payload: Mapping[str, object],
) -> tuple[WorkerPhase, int, int | None, int | None]:
    raw_phase = payload.get("phase")
    if raw_phase not in {"pre-generation", "generation", "post-generation"}:
        raise WorkerPhaseStateError("worker phase-state phase is unsupported")
    phase = cast(WorkerPhase, raw_phase)
    sequence = _required_int(payload.get("sequence"), "sequence")
    transition_ns = _required_int(
        payload.get("transition_monotonic_ns"),
        "transition_monotonic_ns",
    )
    started_raw = payload.get("generation_started_monotonic_ns")
    finished_raw = payload.get("generation_finished_monotonic_ns")
    started_ns = (
        None
        if started_raw is None
        else _required_int(started_raw, "generation_started_monotonic_ns")
    )
    finished_ns = (
        None
        if finished_raw is None
        else _required_int(finished_raw, "generation_finished_monotonic_ns")
    )
    expected_sequence = {
        "pre-generation": 0,
        "generation": 1,
        "post-generation": 2,
    }[phase]
    if sequence != expected_sequence:
        raise WorkerPhaseStateError(
            "worker phase-state phase and sequence are inconsistent"
        )
    if phase == "pre-generation":
        if started_ns is not None or finished_ns is not None:
            raise WorkerPhaseStateError(
                "pre-generation state cannot contain generation timestamps"
            )
    elif phase == "generation":
        if started_ns != transition_ns or finished_ns is not None:
            raise WorkerPhaseStateError(
                "generation state has inconsistent generation timestamps"
            )
    elif started_ns is None or finished_ns != transition_ns or finished_ns < started_ns:
        raise WorkerPhaseStateError(
            "post-generation state has inconsistent generation timestamps"
        )
    return phase, transition_ns, started_ns, finished_ns


def read_worker_phase_state(
    channel: WorkerPhaseChannel,
    *,
    expected_pid: int,
) -> WorkerPhaseState:
    """Read and authenticate the current atomic worker phase state."""

    if expected_pid <= 0:
        raise ValueError("expected worker PID must be positive")
    try:
        raw = channel.path.read_bytes()
    except FileNotFoundError:
        raise
    except OSError as error:
        raise WorkerPhaseStateError(
            f"cannot read worker phase-state file: {error}"
        ) from error
    if len(raw) > 64 * 1024:
        raise WorkerPhaseStateError("worker phase-state file is unreasonably large")
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerPhaseStateError(
            "worker phase-state file is not valid JSON"
        ) from error
    if not isinstance(decoded, Mapping) or set(decoded) != _STATE_FIELDS:
        raise WorkerPhaseStateError(
            "worker phase-state fields do not match the authenticated ABI"
        )
    unsigned = {field: decoded[field] for field in _PHASE_FIELDS}
    authentication = decoded.get("authentication")
    if (
        not isinstance(authentication, Mapping)
        or set(authentication) != {"kind", "digest"}
        or authentication.get("kind") != WORKER_PHASE_AUTHENTICATION
        or not isinstance(authentication.get("digest"), str)
        or not hmac.compare_digest(
            str(authentication["digest"]),
            _authentication_digest(unsigned, channel.authentication_key),
        )
    ):
        raise WorkerPhaseStateError("worker phase-state authentication is invalid")
    if decoded.get("abi") != WORKER_PHASE_STATE_ABI:
        raise WorkerPhaseStateError("worker phase-state ABI is unsupported")
    if decoded.get("run_id") != channel.run_id:
        raise WorkerPhaseStateError("worker phase-state run_id does not match")
    worker_pid = _required_int(decoded.get("worker_pid"), "worker_pid")
    if worker_pid != expected_pid:
        raise WorkerPhaseStateError(
            "worker phase-state PID does not match supervised worker"
        )
    phase, transition_ns, started_ns, finished_ns = _validated_timing(decoded)
    return WorkerPhaseState(
        run_id=channel.run_id,
        worker_pid=worker_pid,
        sequence=int(decoded["sequence"]),
        phase=phase,
        transition_monotonic_ns=transition_ns,
        generation_started_monotonic_ns=started_ns,
        generation_finished_monotonic_ns=finished_ns,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "WORKER_PHASE_AUTHENTICATION",
    "WORKER_PHASE_STATE_ABI",
    "WorkerPhaseChannel",
    "WorkerPhaseReporter",
    "WorkerPhaseState",
    "WorkerPhaseStateError",
    "read_worker_phase_state",
]
