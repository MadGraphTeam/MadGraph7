#!/usr/bin/env python3
"""Campaign-independent controller for the persistent MadLoop benchmark driver.

The caller prepares an isolated subprocess directory, cards, executable, point
stream and CPU affinity. This module never changes numerical settings or chooses
a CPU. Every command retains its raw files, including rejected measurements.
"""
import json
import math
import os
from pathlib import Path
import selectors
import signal
import statistics
import subprocess
import tempfile
import threading
import time


ROUTE_STATISTICS_OUTPUT = "tirust_routes.dat"
OUTPUTS = ("timing.dat", "values.dat", "timed_status.dat", ROUTE_STATISTICS_OUTPUT)
RAW_OUTPUTS = (*OUTPUTS, "dimensions.dat", "warmup.dat")
SCHEME = "tirust-persistent-worker-v1"
# (H, U) -> (MLReductionLib, precision) for the qualified backends 1, 6, 7, 8, 9.
# Return-code format 2 reports NinjaRS as U=2 (212 DP, 322 QP) and TIRust as
# U=3 (213 DP, 323 QP; IREGI shares U=3 but is never benchmarked). Format 1,
# used by campaigns before the madloop_ninjars merge, aliased their DP results
# to 218 and 219. Format 2 never emits those (U=8 and U=9 are QP-only), so
# both formats decode unambiguously.
BACKEND_CODES = {
    (2, 1): (1, "DP"), (3, 9): (1, "QP"),
    (2, 6): (6, "DP"), (3, 8): (6, "QP"),
    (2, 7): (7, "DP"),
    (2, 2): (8, "DP"), (2, 8): (8, "DP"), (3, 2): (8, "QP"),
    (2, 3): (9, "DP"), (2, 9): (9, "DP"), (3, 3): (9, "QP"),
}


def route_statistics_hooks(prefix, enabled=False):
    """Optional fixed-form hooks, entirely outside the measured interval.

    The generated Fortran getter owns MTIRUST capability handling.  No USE of
    a new module symbol is introduced into a legacy benchmark driver.
    """
    if not isinstance(prefix, str) or type(enabled) is not bool:
        raise ValueError("Invalid route-statistics render parameters")
    empty = dict.fromkeys(("DECLARATIONS", "BEFORE", "AFTER", "WRITE"), "")
    if not enabled:
        return empty
    return {
        "DECLARATIONS": ("      INTEGER*8 TIRUST_ROUTE_BEFORE(20),TIRUST_ROUTE_AFTER(20)\n"
                         "      INTEGER TIRUST_ROUTE_STATUS_BEFORE,TIRUST_ROUTE_STATUS_AFTER"),
        "BEFORE": (f"      CALL {prefix}GET_TIRUST_DP_ROUTE_STATISTICS(\n"
                   "     $ TIRUST_ROUTE_BEFORE,TIRUST_ROUTE_STATUS_BEFORE)"),
        "AFTER": (f"      CALL {prefix}GET_TIRUST_DP_ROUTE_STATISTICS(\n"
                  "     $ TIRUST_ROUTE_AFTER,TIRUST_ROUTE_STATUS_AFTER)"),
        "WRITE": ("      OPEN(73,FILE='tirust_routes.dat',STATUS='REPLACE')\n"
                  "      WRITE(73,*) 1,SEQUENCE,NLOOPS,NPOINTS,\n"
                  "     $ TIRUST_ROUTE_STATUS_BEFORE,TIRUST_ROUTE_STATUS_AFTER,\n"
                  "     $ TIRUST_ROUTE_BEFORE,TIRUST_ROUTE_AFTER,\n"
                  "     $ TIRUST_ROUTE_AFTER-TIRUST_ROUTE_BEFORE\n"
                  "      CLOSE(73)"),
    }


"""Pure interpretation of optional TIRust route evidence.

MadLoop return codes describe accepted input/output precision.  They do not
identify every internal arithmetic tier.  This module does not change legacy
observations and cannot certify arithmetic hidden by Fortran coefficient caches.
"""
import re

MAX_I64 = (1 << 63) - 1
COUNTERS = ("direct", "method_choice", "tier106", "tier256", "tier512",
            "refused", "unreported", "attempt106", "attempt256", "attempt512")
BITS = (106, 256, 512)
SCOPE = "bridge computations only; Fortran coefficient-cache hits are not observed"


def _positive_int(value, name):
    if type(value) is not int or not 1 <= value <= MAX_I64:
        raise ValueError(name + " must be a positive signed-64-bit integer")


def parse_route_statistics(text, points, loops, sequence=None):
    """Parse schema1: six header integers, then before20/after20/delta20.

    Nonzero snapshot statuses make the counters unavailable.  A syntactically
    valid but unavailable snapshot never supplies evidence of native DP.
    """
    _positive_int(points, "points")
    _positive_int(loops, "loops")
    if points * loops > MAX_I64:
        raise ValueError("timed amplitude count exceeds signed-64-bit range")
    if sequence is not None:
        _positive_int(sequence, "sequence")
    if not isinstance(text, str) or not text.isascii():
        raise ValueError("route statistics must be ASCII text")
    words = text.split()
    if (len(words) != 66
            or any(re.fullmatch(r"\+?[0-9]+", w) is None for w in words[:46])
            or any(re.fullmatch(r"[+-]?[0-9]+", w) is None for w in words[46:])):
        raise ValueError("route statistics require exactly66 decimal integers")
    values = [int(w) for w in words]
    if any(v > MAX_I64 or v < -MAX_I64 - 1 for v in values):
        raise ValueError("route statistic exceeds signed-64-bit range")
    schema, seq, actual_loops, actual_points, status_before, status_after = values[:6]
    if schema != 1 or seq < 1 or actual_loops != loops or actual_points != points:
        raise ValueError("route header differs from requested observation")
    if sequence is not None and seq != sequence:
        raise ValueError("route sequence differs from requested observation")
    before, after, delta = values[6:26], values[26:46], values[46:66]
    available = status_before == status_after == 0
    if any(a - b != d or (available and a < b) for b, a, d in zip(before, after, delta)):
        raise ValueError("route counters reset or do not match exact delta")
    return {"schema": schema, "sequence": seq, "loops": loops, "points": points,
            "status_before": status_before, "status_after": status_after,
            "available": available,
            "scope": SCOPE, "before": before, "after": after, "delta": delta,
            "primary": dict(zip(COUNTERS, delta[:10])),
            "check": dict(zip(COUNTERS, delta[10:]))}


def aggregate_native_dp_only(flags):
    """Three-valued aggregation: False dominates, otherwise unknown dominates."""
    flags = list(flags)
    if any(v is not None and type(v) is not bool for v in flags):
        raise ValueError("native-DP flags must be bool or None")
    if any(v is False for v in flags):
        return False
    return True if flags and all(v is True for v in flags) else None


def precision_usage(observation, selected=None, use_qp_integrand_for_tirust=None):
    """Conservative arithmetic classification, separate from legacy status fields.

    ``selected`` is the requested backend/chain known to the parser; it need
    not be inserted into the stored legacy observation.  Pass parsed schema1
    evidence as ``observation['route_statistics']``.  A complete binary64-only
    bridge interval can establish ``bridge_native_dp_only=True``.  It cannot
    establish ``native_dp_only=True`` for the complete amplitude because this
    sidecar does not observe Fortran coefficient-cache reuse.
    """
    if not isinstance(observation, dict):
        raise ValueError("observation must be a mapping")
    if use_qp_integrand_for_tirust is not None and type(use_qp_integrand_for_tirust) is not bool:
        raise ValueError("UseQPIntegrandForTIRust evidence must be bool or None")
    counts = observation.get("backend_precision_counts", {})
    if not isinstance(counts, dict):
        raise ValueError("backend precision counts must be a mapping")
    decoded = []
    for key, count in counts.items():
        if (not isinstance(key, str) or re.fullmatch(r"(?:1|6|7|8|9):(?:DP|QP)", key) is None
                or type(count) is not int or not 1 <= count <= MAX_I64 or key == "7:QP"):
            raise ValueError("invalid accepted backend/transport count")
        backend, precision = key.split(":")
        decoded.append((int(backend), precision))
    if selected is None:
        selected = observation.get("selected_chain")
    if type(selected) is int:
        selected = (selected,)
    if selected is not None and (not isinstance(selected, (tuple, list)) or not selected
            or any(type(x) is not int or x not in (1, 6, 7, 8, 9) for x in selected)):
        raise ValueError("selected backend/chain is invalid")
    first = selected[0] if selected is not None else None
    qp = any(precision == "QP" for _, precision in decoded)
    backends = {backend for backend, _ in decoded}
    fallback = (None if not backends else
                any(backend != first for backend in backends) if first is not None
                else True if len(backends) > 1 else None)
    tirust_qp = (9, "QP") in decoded
    route = observation.get("route_statistics")
    bridge_native = higher = success = None
    maximum = None
    successes = attempts = None
    state, bounds, counter_delta = "absent", "unknown", None
    if route is not None:
        # Reconstruct and revalidate instead of trusting caller-added flags/maps.
        if not isinstance(route, dict):
            raise ValueError("route statistics must be a parsed mapping")
        try:
            if any(not isinstance(route[k], (list, tuple)) or len(route[k]) != 20
                   for k in ("before", "after", "delta")):
                raise ValueError("route counter blocks must each contain20 integers")
            header = [route[k] for k in ("schema", "sequence", "loops", "points",
                                         "status_before", "status_after")]
            raw = header + list(route["before"]) + list(route["after"]) + list(route["delta"])
            if any(type(x) is not int for x in raw):
                raise ValueError("route statistics contain noninteger fields")
            checked = parse_route_statistics(" ".join(map(str, raw)), route["points"],
                                             route["loops"], route["sequence"])
        except (KeyError, TypeError) as error:
            raise ValueError("incomplete parsed route statistics") from error
        state = "snapshot-unavailable"
        if checked["available"]:
            counter_delta = checked["delta"]
            blocks = [checked["primary"], checked["check"]]
            successes = {name: {str(bit): block["tier" + str(bit)] for bit in BITS}
                         for name, block in zip(("primary", "check"), blocks)}
            attempts = {name: {str(bit): block["attempt" + str(bit)] for bit in BITS}
                        for name, block in zip(("primary", "check"), blocks)}
            observed = [bit for bit in BITS if any(block["tier" + str(bit)]
                        or block["attempt" + str(bit)] for block in blocks)]
            success = any(block["tier" + str(bit)] for block in blocks for bit in BITS)
            # Old warmup errors do not censor an otherwise complete new interval.
            censored = any(checked["delta"][offset + 6] for offset in (0, 10))
            refused = any(block["refused"] for block in blocks)
            complete_dp = all(block["direct"] + block["method_choice"] > 0 for block in blocks)
            bounds = "lower bounds" if censored else "reported counts"
            if observed:
                higher, maximum, bridge_native = True, max(observed), False
                state = "censored" if censored else "higher-precision-observed"
            elif censored or refused:
                state = "censored" if censored else "refused-route-observed"
            elif complete_dp:
                higher, bridge_native, state = False, True, "complete-binary64-bridge-interval"
            else:
                state = "no-bridge-computations" if not any(checked["delta"]) else "partial-route-coverage"
    if qp:
        bridge_native = False
    full_native = False if qp or fallback is True or higher is True else None
    label = ("genuine Fortran QP transport present" if qp else
             "fallback backend accepted" if fallback is True else
             "internal higher-precision attempted" if higher is True else
             "internal arithmetic unqualified")
    return {"accepted_backend_precision_counts": dict(counts),
            "accepted_precision_scope": "MadLoop input/output transport; not highest internal arithmetic",
            "requested_first_backend": first, "accepted_qp_transport": qp if decoded else None,
            "tirust_qp_transport_observed": tirust_qp if decoded else None,
            "use_qp_integrand_for_tirust": use_qp_integrand_for_tirust,
            "tirust_qp_numerator_enabled": (use_qp_integrand_for_tirust if tirust_qp else None),
            "tirust_qp_numerator_scope": ("QP numerator coefficients enabled by verified effective card"
                if tirust_qp and use_qp_integrand_for_tirust is True else
                "copied DP numerator coefficients selected by verified effective card"
                if tirust_qp and use_qp_integrand_for_tirust is False else
                "unknown; accepted transport alone does not establish numerator precision"),
            "fallback_backend_observed": fallback, "route_evidence_status": state,
            "route_counter_delta": counter_delta,
            "route_scope": SCOPE, "attempt_count_scope": bounds,
            "higher_precision_success_counts": successes,
            "higher_precision_attempt_counts": attempts,
            "internal_higher_precision_attempted": higher,
            "internal_higher_precision_success_observed": success,
            "highest_observed_internal_bits": maximum,
            "bridge_native_dp_only": bridge_native, "native_dp_only": full_native,
            "precision_classification": label}


def aggregate_precision_usage(observations, selected=None, use_qp_integrand_for_tirust=None):
    """Sum reported intervals, retaining unknown coverage and separate attempt counts."""
    uses = [precision_usage(o, selected, use_qp_integrand_for_tirust) for o in observations]
    deltas = [u["route_counter_delta"] for u in uses if u["route_counter_delta"] is not None]
    total = [sum(d[i] for d in deltas) for i in range(20)] if deltas else None
    complete = bool(uses) and len(deltas) == len(uses) and all(
        u["attempt_count_scope"] == "reported counts" for u in uses)
    counts = {key: sum(u["accepted_backend_precision_counts"].get(key, 0) for u in uses)
              for key in {k for u in uses for k in u["accepted_backend_precision_counts"]}}
    if any(x > MAX_I64 for x in (total or []) + list(counts.values())):
        raise ValueError("aggregate count exceeds signed-64-bit range")
    reported = {key: ({side: {str(bit): sum(u[key][side][str(bit)] for u in uses if u[key] is not None)
                              for bit in BITS} for side in ("primary", "check")}
                     if deltas else None)
                for key in ("higher_precision_success_counts", "higher_precision_attempt_counts")}
    higher = [u["internal_higher_precision_attempted"] for u in uses]
    qp_numerators = [u["tirust_qp_numerator_enabled"] for u in uses if u["tirust_qp_transport_observed"]]
    return {"observations": len(uses), "reported_route_intervals": len(deltas),
            "route_interval_coverage": "complete" if complete else "partial" if uses else "unknown",
            "attempt_count_scope": "reported counts" if complete else "lower bounds" if deltas else "unknown",
            "route_counter_delta": total, **reported, "accepted_backend_precision_counts": counts,
            "accepted_precision_scope": "MadLoop input/output transport; not highest internal arithmetic",
            "native_dp_only": aggregate_native_dp_only(u["native_dp_only"] for u in uses),
            "bridge_native_dp_only": aggregate_native_dp_only(u["bridge_native_dp_only"] for u in uses),
            "internal_higher_precision_attempted": (True if any(v is True for v in higher) else
                False if higher and all(v is False for v in higher) else None),
            "highest_observed_internal_bits": max((u["highest_observed_internal_bits"] for u in uses
                if u["highest_observed_internal_bits"] is not None), default=None),
            "precision_classifications": sorted({u["precision_classification"] for u in uses}),
            "route_evidence_statuses": sorted({u["route_evidence_status"] for u in uses}),
            "tirust_qp_numerator_enabled": aggregate_native_dp_only(qp_numerators),
            "use_qp_integrand_for_tirust": use_qp_integrand_for_tirust,
            "route_scope": SCOPE,
            "whole_amplitude_qp_scope": "transport and optional numerator-card evidence only; no whole-amplitude precision claim"}


class ProtocolError(RuntimeError):
    pass


class MeasurementError(RuntimeError):
    """A rejected result with the unchanged raw numerical evidence attached."""
    def __init__(self, message, raw_outputs):
        super().__init__(message)
        self.raw_outputs = raw_outputs
        self.raw_directory = None


def raw_outputs(directory):
    directory = Path(directory)
    return {name: (directory / name).read_text(errors="replace")
            for name in RAW_OUTPUTS if (directory / name).is_file()}


def validate_status(code, backend):
    """Decode H/T/U, requiring a warmed stable result from the selected chain.

    T=1/2 denotes steady DP/QP execution; initialization uses T=3/4. H,
    together with U, distinguishes the format-1 aliases (CutTools QP 329
    versus TIRust DP 219); see BACKEND_CODES.
    """
    selected = {backend} if isinstance(backend, int) else set(backend)
    if not selected or not selected.issubset({1, 6, 7, 8, 9}):
        raise ValueError("Expected backend or nonempty chain drawn from 1,6,7,8,9")
    if isinstance(code, bool) or not isinstance(code, int) or code < 0:
        raise ValueError("Status must be a nonnegative integer")
    h, t, u = code // 100, code // 10 % 10, code % 10
    actual = BACKEND_CODES.get((h, u))
    if t not in (1, 2) or actual is None or actual[0] not in selected:
        raise ValueError(f"Unexpected status {code} for backends {sorted(selected)}; "
                         "initialization, failure, or another backend was used")
    return {"backend": actual[0], "precision": actual[1], "stable": True,
            "hundreds": h, "tens": t, "units": u}


class Worker:
    """One persistent child with protocol FIFO and continuously drained stdout."""
    def __init__(self, command, directory, log, env=None, timeout=3600.):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.directory, self.log = Path(directory).resolve(), Path(log).resolve()
        self.timeout, self.sequence = timeout, 0
        self.pipe = self.directory / "worker-protocol.fifo"
        self.process = self.thread = self.stream = self.fd = self.selector = None
        self.closed = False
        self.buffer = bytearray()
        self.last_raw_directory = None
        self.raw_directory = None
        # Never unlink another live worker's FIFO.
        os.mkfifo(self.pipe, mode=0o600)
        try:
            self.fd = os.open(self.pipe, os.O_RDWR | os.O_NONBLOCK)
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.fd, selectors.EVENT_READ)
            self.log.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.log.open("wb")
            archive_root = self.directory / "worker-raw"
            archive_root.mkdir(exist_ok=True)
            self.raw_directory = Path(tempfile.mkdtemp(prefix="session-", dir=archive_root))
            # A previous process may have left files in the shared subprocess
            # directory. Preserve them separately, then avoid attributing them
            # to a child which fails before producing its own evidence.
            previous = raw_outputs(self.directory)
            if previous:
                retained = self.raw_directory / "previous-outputs"
                retained.mkdir()
                for name, value in previous.items():
                    (retained / name).write_text(value)
                    (self.directory / name).unlink()
            self.process = subprocess.Popen(
                [*map(str, command), str(self.pipe)], cwd=self.directory, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, start_new_session=True)
            self.thread = threading.Thread(target=self._drain, daemon=True)
            self.thread.start()
            fields = self._message()
            if len(fields) == 4 and fields[:2] in (
                    ["ERROR", "warmup-status"], ["ERROR", "warmup-limit"]):
                raise MeasurementError(
                    f"Numerical warmup failure ({fields[1]}) at point {fields[2]}, "
                    f"status {fields[3]}; see {self.log}", raw_outputs(self.directory))
            if len(fields) != 7 or fields[:2] != ["READY", "1"]:
                raise ProtocolError(f"Invalid READY message: {fields}")
            self.points, self.ndim, self.nso, self.nborn = map(int, fields[2:6])
            self.cold_seconds = float(fields[6])
            if (self.points <= 0 or min(self.nso, self.nborn) < 0 or
                    self.ndim != max(self.nso, self.nborn) or
                    not math.isfinite(self.cold_seconds) or self.cold_seconds <= 0):
                raise ProtocolError(f"Invalid READY dimensions/timing: {fields}")
            self._archive_startup(None)
        except BaseException as error:
            self.close(graceful=False)
            self._archive_startup(error)
            error.raw_outputs = raw_outputs(self.directory)
            error.raw_directory = self.last_raw_directory
            raise

    def _archive_startup(self, error):
        if self.raw_directory is None:
            return
        archive = self.raw_directory / "startup"
        archive.mkdir(exist_ok=True)
        self.last_raw_directory = archive
        for name, value in raw_outputs(self.directory).items():
            (archive / name).write_text(value)
        (archive / "startup.json").write_text(json.dumps({
            "error": {"type": type(error).__name__, "message": str(error)}
                     if error is not None else None,
            "log": str(self.log)}, indent=2) + "\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(graceful=exc_type is None)

    def _drain(self):
        while True:
            chunk = self.process.stdout.read1(65536)
            if not chunk:
                return
            self.stream.write(chunk)
            self.stream.flush()

    def _message(self):
        deadline = time.monotonic() + self.timeout
        while True:
            if b"\n" in self.buffer:
                line, _, remainder = self.buffer.partition(b"\n")
                self.buffer = bytearray(remainder)
                try:
                    return line.decode("ascii").split()
                except UnicodeDecodeError as error:
                    raise ProtocolError("Non-ASCII worker protocol") from error
            if time.monotonic() >= deadline:
                raise ProtocolError(f"Worker protocol timeout; see {self.log}")
            if self.selector.select(min(.2, max(0., deadline-time.monotonic()))):
                self.buffer.extend(os.read(self.fd, 65536))
                if len(self.buffer) > 4096:
                    raise ProtocolError("Oversized worker protocol message")
                continue
            status = self.process.poll()
            if status is not None:
                raise ProtocolError(f"Worker exited ({status}) before reply; see {self.log}")

    def _send(self, line):
        try:
            self.process.stdin.write(line.encode("ascii") + b"\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise ProtocolError(f"Worker command pipe closed; see {self.log}") from error

    def run(self, loops):
        """Run warmed traversals; return controller elapsed time, not the timer.

        Use parse_outputs()['seconds'] for the worker's wall-clock measurement.
        Raw files are archived outside the timed interval on every command.
        """
        if isinstance(loops, bool) or not isinstance(loops, int) or loops <= 0:
            raise ValueError("loops must be a positive integer")
        if self.closed:
            raise ProtocolError("Worker is closed")
        self.sequence += 1
        archive = self.raw_directory / f"{self.sequence:06d}"
        # Fail before sending anything if a previous controller owns this slot.
        archive.mkdir(parents=True, exist_ok=False)
        self.last_raw_directory = archive
        for name in OUTPUTS:
            (self.directory / name).unlink(missing_ok=True)
        start, failure = time.perf_counter(), None
        try:
            self._send(f"RUN {self.sequence} {loops}")
            message = self._message()
            if message != ["DONE", str(self.sequence), str(loops)]:
                raise ProtocolError(f"Unexpected worker completion: {message}")
            route_file = self.directory / ROUTE_STATISTICS_OUTPUT
            if route_file.is_file():
                try:
                    parse_route_statistics(route_file.read_text(), self.points, loops,
                                           sequence=self.sequence)
                except (OSError, ValueError) as error:
                    raise MeasurementError(str(error), raw_outputs(self.directory)) from error
            return time.perf_counter() - start
        except BaseException as error:
            failure = {"type": type(error).__name__, "message": str(error)}
            self.close(graceful=False)
            raise
        finally:
            for name, value in raw_outputs(self.directory).items():
                (archive / name).write_text(value)
            (archive / "command.json").write_text(json.dumps({
                "sequence": self.sequence, "loops": loops, "error": failure,
                "log": str(self.log)}, indent=2) + "\n")

    def close(self, graceful=True):
        if self.closed:
            return
        self.closed = True
        shutdown_error = None

        def terminate_group(sig):
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass

        try:
            if graceful and self.process is not None and self.process.poll() is not None:
                shutdown_error = ProtocolError(
                    f"Worker exited ({self.process.returncode}) before STOP/BYE")
            if self.process is not None and self.process.poll() is None:
                if graceful:
                    try:
                        self._send("STOP 0 0")
                        old_timeout, self.timeout = self.timeout, min(self.timeout, 10.)
                        try:
                            if self._message() != ["BYE"]:
                                raise ProtocolError("Missing graceful worker shutdown")
                        finally:
                            self.timeout = old_timeout
                        self.process.wait(timeout=10)
                        if self.process.returncode:
                            raise ProtocolError(f"Worker exited ({self.process.returncode}) after BYE")
                    except (ProtocolError, subprocess.TimeoutExpired) as error:
                        shutdown_error = error
                        terminate_group(signal.SIGTERM)
                else:
                    terminate_group(signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    terminate_group(signal.SIGKILL)
                    self.process.wait()
        finally:
            if self.process is not None:
                terminate_group(signal.SIGTERM)
                if self.process.stdin:
                    self.process.stdin.close()
                if self.thread:
                    self.thread.join(timeout=10)
                if self.process.stdout:
                    self.process.stdout.close()
            if self.stream:
                self.stream.close()
            if self.selector:
                self.selector.close()
            if self.fd is not None:
                os.close(self.fd)
            self.pipe.unlink(missing_ok=True)
        if shutdown_error is not None:
            raise ProtocolError(f"Worker shutdown failed: {shutdown_error}") from shutdown_error


def parse_outputs(directory, backend, points, loops):
    """Validate every retained column and timed status, preserving rejected raw data.

    Coefficient order is [Born, finite, single pole, double pole]. For a Born-only
    split-order column (order > loop_orders), accuracy is the total loop estimate.
    """
    raw = raw_outputs(directory)
    try:
        selected = (backend,) if isinstance(backend, int) else tuple(backend)
        if not selected:
            raise ValueError("Backend chain must not be empty")
        if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
               for x in (points, loops)):
            raise ValueError("points and loops must be positive integers")
        timing = list(map(float, raw["timing.dat"].split()))
        if len(timing) != 2 or not all(math.isfinite(x) and x > 0 for x in timing):
            raise ValueError("Invalid worker timing")
        dimensions = list(map(int, raw["dimensions.dat"].split()))
        if len(dimensions) != 3:
            raise ValueError("Invalid worker dimensions")
        ndim, nso, nborn = dimensions
        if min(nso, nborn) < 0 or ndim != max(nso, nborn):
            raise ValueError("Worker answer dimension omits physical columns")
        rows, counts = [], {}
        for line in raw["values.dat"].splitlines():
            fields = line.split()
            if len(fields) != 8:
                raise ValueError("Malformed worker numerical row")
            code = int(fields[2])
            actual = validate_status(code, selected)
            row = {"point": int(fields[0]), "order": int(fields[1]), "status": code,
                   "coefficients": list(map(float, fields[3:7])), "accuracy": float(fields[7]),
                   "actual_backend": actual["backend"], "precision": actual["precision"]}
            if not all(math.isfinite(x) for x in row["coefficients"] + [row["accuracy"]]):
                raise ValueError("Nonfinite worker numerical row")
            if row["accuracy"] < 0:
                raise ValueError("Missing worker stability estimate")
            if row["order"] == 0:
                counts[str(code)] = counts.get(str(code), 0) + 1
            rows.append(row)
        if [(r["point"], r["order"]) for r in rows] != [
                (p, order) for p in range(1, points+1) for order in range(ndim+1)]:
            raise ValueError("Missing, duplicated or unordered worker numerical rows")
        timed, usage = {}, {}
        for line in raw["timed_status.dat"].splitlines():
            code, count = map(int, line.split())
            actual = validate_status(code, selected)
            if count <= 0 or str(code) in timed:
                raise ValueError("Invalid worker status histogram")
            timed[str(code)] = count
            key = f"{actual['backend']}:{actual['precision']}"
            usage[key] = usage.get(key, 0) + count
        if sum(timed.values()) != loops*points:
            raise ValueError("Worker evaluation count differs from command")
        for code, count in counts.items():
            if timed.get(code, 0) < count:
                raise ValueError("Final traversal status absent from timed histogram")
        # One amplitude evaluation returns the same status for every order.
        for start in range(0, len(rows), ndim+1):
            if len({r["status"] for r in rows[start:start+ndim+1]}) != 1:
                raise ValueError("Split orders have inconsistent evaluation statuses")
        result = {"cold_seconds": timing[0], "seconds": timing[1],
                "seconds_per_point": timing[1]/(loops*points), "rows": rows,
                "status_counts": counts, "timed_status_counts": timed,
                "backend_precision_counts": usage,
                "native_dp_only": set(usage) == {f"{selected[0]}:DP"},
                "answer_dimensions": {"allocated": ndim, "loop_orders": nso,
                                      "born_orders": nborn}}
        # Keep historical observations byte-compatible when this opt-in schema
        # is absent. Their legacy native_dp_only flag is dispatch-scoped only;
        # current reports must use precision_usage() instead.
        if ROUTE_STATISTICS_OUTPUT in raw:
            result["route_statistics"] = parse_route_statistics(
                raw[ROUTE_STATISTICS_OUTPUT], points, loops)
            result["precision_usage"] = precision_usage(result, selected=selected)
            result["native_dp_only"] = result["precision_usage"]["native_dp_only"]
        return result
    except (ValueError, KeyError, TypeError) as error:
        raise MeasurementError(str(error), raw) from error


# Two-sided 95% Student-t critical values, rounded upward. For >30 degrees of
# freedom retain df=30, making the interval conservative without scipy.
_T975 = (0., 12.706205, 4.302653, 3.182447, 2.776446, 2.570582,
         2.446912, 2.364625, 2.306005, 2.262158, 2.228139, 2.200986,
         2.178813, 2.160369, 2.144787, 2.131450, 2.119906, 2.109816,
         2.100923, 2.093025, 2.085964, 2.079614, 2.073874, 2.068658,
         2.063899, 2.059539, 2.055530, 2.051831, 2.048408, 2.045230,
         2.042273)


def paired_statistics(baseline, candidate, threshold=1.05, minimum_pairs=7):
    """Confidence interval on paired log(candidate/baseline) warmed timings.

    Pairs must be adjacent, alternating measurements supplied by the caller.
    The upper two-sided 95% Student-t endpoint is a conservative one-sided 95%
    acceptance bound. This quantifies within-process warmed timing uncertainty,
    not process initialization or machine-to-machine variation.
    """
    baseline, candidate = list(baseline), list(candidate)
    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError("At least two matched timing pairs are required")
    if any(not math.isfinite(x) or x <= 0 for x in baseline+candidate):
        raise ValueError("Timings must be finite and positive")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if isinstance(minimum_pairs, bool) or not isinstance(minimum_pairs, int) or minimum_pairs < 2:
        raise ValueError("minimum_pairs must be an integer of at least two")
    logs = [math.log(c) - math.log(b) for b, c in zip(baseline, candidate)]
    n = len(logs)
    mean = statistics.mean(logs)
    stderr = statistics.stdev(logs)/math.sqrt(n)
    critical = _T975[min(n-1, 30)]
    upper, lower = math.exp(mean + critical*stderr), math.exp(mean - critical*stderr)
    enough = n >= minimum_pairs
    return {"pairs": n, "geometric_mean_ratio": math.exp(mean),
            "lower_95": lower, "upper_95": upper,
            "log_ratio_standard_error": stderr, "student_t_critical": critical,
            "confidence_method": "two-sided 95% Student-t on paired log ratios; df capped at 30",
            "threshold": threshold, "minimum_pairs": minimum_pairs,
            "acceptance_eligible": enough, "passed": enough and upper <= threshold}
