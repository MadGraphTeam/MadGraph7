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


OUTPUTS = ("timing.dat", "values.dat", "timed_status.dat")
RAW_OUTPUTS = (*OUTPUTS, "dimensions.dat", "warmup.dat")
SCHEME = "tirust-persistent-worker-v1"
BACKEND_CODES = {
    (2, 1): (1, "DP"), (3, 9): (1, "QP"),
    (2, 6): (6, "DP"), (3, 8): (6, "QP"),
    (2, 7): (7, "DP"),
    (2, 8): (8, "DP"), (3, 2): (8, "QP"),
    (2, 9): (9, "DP"), (3, 3): (9, "QP"),
}


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
    together with U, distinguishes aliases such as CutTools QP and TIRust DP.
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
            if isinstance(error, MeasurementError):
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
        return {"cold_seconds": timing[0], "seconds": timing[1],
                "seconds_per_point": timing[1]/(loops*points), "rows": rows,
                "status_counts": counts, "timed_status_counts": timed,
                "backend_precision_counts": usage,
                "native_dp_only": set(usage) == {f"{selected[0]}:DP"},
                "answer_dimensions": {"allocated": ndim, "loop_orders": nso,
                                      "born_orders": nborn}}
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
