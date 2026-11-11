#!/usr/bin/env python3
"""Strict offline analysis of the four frozen subprocess-cache workers.

Usage: python3 analyze.py RUN_ROOT --output analysis.json
RUN_ROOT contains plain-cache0, plain-cache-all, observed-cache0, and
observed-cache-all; each must contain records.txt and trace.txt. No program is
executed and no numerical library is imported. Output schema version 1 records
input hashes, every collected qualification failure, coverage, exact parity,
and native cache/lifecycle witnesses. Exit 0 means the retained-text gates
passed; binary/source/runtime provenance and worker exit statuses require the
separate execution binder. An incomplete or malformed stream fails closed.

Python API: parse_records(path), parse_trace(path), analyze(root). Parsers raise
EvidenceError on structural failures; analyze returns a JSON-serializable
report including failures instead of discarding another worker's evidence.
The CLI writes only its explicitly selected JSON output (default stdout).
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys


LABELS = ("plain-cache0", "plain-cache-all", "observed-cache0", "observed-cache-all")
CHECKED = {219, 229, 313, 323}
PHYSICAL = tuple(range(301, 320))
KEYS = tuple(range(1101, 1115)) + tuple(range(1201, 1215))
NA = -(1 << 31)
MAX_TEXT_BYTES = 20 * 1024 * 1024
HEX = re.compile(r"[0-9A-F]{16}\Z")
INTEGER = re.compile(r"-?[0-9]+\Z")
TRACE_NAMES = (
    "sequence", "action", "process", "kind", "token_before", "token_after",
    "status", "n", "rank", "poles", "finite", "check", "cache", "version",
    "stats_pre_aggregate", "stats_pre_qp", "stats_post_aggregate", "stats_post_qp",
    "f0", "f1", "f2", "f3", "r0", "r1", "r2", "r3",
)


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(text):
    require(INTEGER.fullmatch(text) is not None, f"invalid decimal integer {text!r}")
    return int(text)


def bits(text):
    require(HEX.fullmatch(text) is not None, f"invalid binary64 word {text!r}")
    return text


def binary64(word):
    return struct.unpack(">d", bytes.fromhex(bits(word)))[0]


def finite64(word):
    return ((int(bits(word), 16) >> 52) & 0x7ff) != 0x7ff


def finite128(words):
    # Frozen GNU/Linux little-endian INTEGER*8 TRANSFER representation.
    require(len(words) == 2, "binary128 component must contain two words")
    low, high = map(lambda value: int(bits(value), 16), words)
    return ((high >> 48) & 0x7fff) != 0x7fff


def promoted128(word):
    """Exact binary64-to-binary128 promotion, expressed as two little-endian words."""
    raw = int(bits(word), 16)
    sign, exponent, fraction = raw >> 63, (raw >> 52) & 0x7ff, raw & ((1 << 52) - 1)
    require(exponent != 0x7ff, "cannot promote a nonfinite diagnostic momentum")
    mantissa = fraction if exponent == 0 else fraction + (1 << 52)
    if mantissa == 0:
        result = sign << 127
    else:
        power = -1074 if exponent == 0 else exponent - 1023 - 52
        length = mantissa.bit_length()
        significand = (mantissa << (113 - length)) - (1 << 112)
        result = (sign << 127) | ((power + length - 1 + 16383) << 112) | significand
    return [f"{result & ((1 << 64) - 1):016X}", f"{result >> 64:016X}"]


def coefficient_signature(frame):
    """Canonicalize signed zero only, so a changed sign of zero is no witness."""
    result = []
    for words in frame["values"]:
        width = 1 if frame["precision"] == 1 else 2
        for offset in range(0, len(words), width):
            value = int(words[offset], 16)
            if width == 2:
                value |= int(words[offset + 1], 16) << 64
            magnitude = value & ((1 << (64 * width - 1)) - 1)
            result.append(0 if magnitude == 0 else value)
    return result


class Reader:
    def __init__(self, path):
        self.path = Path(path)
        require(self.path.stat().st_size <= MAX_TEXT_BYTES, f"{self.path}: oversized evidence")
        raw = self.path.read_text(encoding="ascii")
        require(raw.endswith("\n"), f"{self.path}: missing terminal newline")
        self.lines = raw.splitlines()
        require(all(line.strip() for line in self.lines), f"{self.path}: empty record")
        self.index = 0

    def peek(self):
        return self.lines[self.index].split()[0] if self.index < len(self.lines) else None

    def take(self, tag, count):
        require(self.index < len(self.lines), f"{self.path}: missing {tag}")
        fields = self.lines[self.index].split()
        self.index += 1
        require(len(fields) == count and fields[0] == tag,
                f"{self.path}:{self.index}: expected {tag} with {count - 1} fields")
        return fields[1:]

    def done(self):
        require(self.index == len(self.lines), f"{self.path}: unexpected trailing records")


def parse_amp(reader, dimensions):
    data = list(map(integer, reader.take("AMP", 9)))
    action, process, phase, code, nd, ns, generation_before, generation_after = data
    require(process in (1, 2), "AMP process must be 1 or 2")
    require((nd, ns) == (dimensions[process - 1], dimensions[process + 1]),
            f"AMP {action}: dimensions differ from HEADER")
    model_before = reader.take("MODELI", 4)
    require(integer(model_before[0]) == action, "MODELI action mismatch")
    model_before = list(map(bits, model_before[1:]))
    model = reader.take("MODEL", 4)
    require(integer(model[0]) == action, "MODEL action mismatch")
    model = list(map(bits, model[1:]))
    latches = {}
    for tag in ("LATCHI", "LATCH"):
        row = reader.take(tag, 6)
        require(integer(row[0]) == action and set(row[1:]) <= {"T", "F"},
                f"AMP {action}: malformed software latches")
        latches[tag] = row[1:]
    flags = reader.take("FLAGS", 12)
    require(integer(flags[0]) == action and set(flags[1:]) <= {"T", "F"},
            f"AMP {action}: malformed IEEE flags")
    momenta, momenta_before = [], []
    for leg in range(1, 4):
        row = reader.take("PI", 7)
        require(list(map(integer, row[:2])) == [action, leg], "PI ordering/action mismatch")
        momenta_before.append(list(map(bits, row[2:])))
        row = reader.take("P", 7)
        require(list(map(integer, row[:2])) == [action, leg], "P ordering/action mismatch")
        momenta.append(list(map(bits, row[2:])))
    values = []
    for column in range(nd + 1):
        row = reader.take("V", 8)
        require(list(map(integer, row[:2])) == [action, column], "V ordering/action mismatch")
        values.append(list(map(bits, row[2:])))
    return dict(action=action, process=process, phase=phase, status=code, nd=nd, ns=ns,
                generation_before=generation_before, generation_after=generation_after,
                model=model, model_before=model_before, flags=flags[1:], momenta=momenta,
                momenta_before=momenta_before, values=values,
                latches_before=latches["LATCHI"], latches_after=latches["LATCH"])


def parse_key(reader, capacity):
    fields = reader.take("KEY", 9)
    data = list(map(integer, fields[:7]))
    action, process, precision, key, n, rank, count = data
    require(fields[7] in ("T", "F"), "KEY stable flag malformed")
    require(precision in (1, 2) and 1 <= key <= 14, "KEY precision/index outside protocol")
    require(action == 1000 + 100 * precision + key and process == 1 + key % 2,
            "KEY identifier/process differs from prescribed sequence")
    require((n, rank) == ((1, 0) if key < 8 or key >= 13 else (3, 2)), "KEY topology/rank changed")
    require(count == math.comb(rank + 4, 4) and count <= capacity, "KEY coefficient capacity mismatch")
    counters = {}
    for name in ("BEFORE", "AFTER"):
        row = list(map(integer, reader.take(name, 17)))
        require(row[0] == action and all(0 <= x < 1 << 64 for x in row[1:]),
                f"KEY {action}: invalid {name} counters")
        counters[name.lower()] = row[1:]
    scale_before = reader.take("SCALEI", 3)
    require(integer(scale_before[0]) == action, "KEY SCALEI action mismatch")
    scale = reader.take("SCALE", 3)
    require(integer(scale[0]) == action, "KEY SCALE action mismatch")
    momenta, masses, momenta_before, masses_before = [], [], [], []
    qp_momenta, qp_momenta_before = [], []
    for leg in range(1, n + 1):
        row = reader.take("KPI", 7)
        require(list(map(integer, row[:2])) == [action, leg], "KPI ordering/action mismatch")
        momenta_before.append(list(map(bits, row[2:])))
        row = reader.take("KMI", 5)
        require(list(map(integer, row[:2])) == [action, leg], "KMI ordering/action mismatch")
        masses_before.append(list(map(bits, row[2:])))
        if precision == 2:
            row = reader.take("KQPI", 11)
            require(list(map(integer, row[:2])) == [action, leg], "KQPI ordering/action mismatch")
            qp_momenta_before.append(list(map(bits, row[2:])))
            row = reader.take("KQP", 11)
            require(list(map(integer, row[:2])) == [action, leg], "KQP ordering/action mismatch")
            qp_momenta.append(list(map(bits, row[2:])))
        row = reader.take("KP", 7)
        require(list(map(integer, row[:2])) == [action, leg], "KP ordering/action mismatch")
        momenta.append(list(map(bits, row[2:])))
        row = reader.take("KM", 5)
        require(list(map(integer, row[:2])) == [action, leg], "KM ordering/action mismatch")
        masses.append(list(map(bits, row[2:])))
    values = []
    for slot in range(1, 4):
        for coefficient in range(count):
            row = reader.take("D" if precision == 1 else "Q", 6 if precision == 1 else 8)
            require(list(map(integer, row[:3])) == [action, slot, coefficient],
                    "KEY coefficient slot/index ordering mismatch")
            values.append(list(map(bits, row[3:])))
    return dict(action=action, process=process, precision=precision, key=key, n=n, rank=rank,
                count=count, stable=fields[7] == "T", scale=bits(scale[1]), scale_before=bits(scale_before[1]),
                momenta=momenta, masses=masses, momenta_before=momenta_before, masses_before=masses_before,
                qp_momenta=qp_momenta, qp_momenta_before=qp_momenta_before, values=values, **counters)


def parse_records(path):
    reader = Reader(path)
    header = list(map(integer, reader.take("HEADER", 6)))
    cache, *dimensions = header
    require(cache in (0, -1), "HEADER cache must be 0 or -1")
    require(all(1 <= value <= 64 for value in dimensions), "invalid HEADER answer dimensions")
    require(dimensions[0] >= dimensions[2] and dimensions[1] >= dimensions[3],
            "HEADER answer dimension smaller than loop split-order count")
    capacity = integer(reader.take("CAP", 2)[0])
    require(15 <= capacity <= math.comb(10, 4), "CAP outside frozen rank-two-to-six domain")
    amplitudes, order = {}, []
    while reader.peek() == "AMP":
        frame = parse_amp(reader, dimensions)
        action = frame["action"]
        require(action not in amplitudes, "duplicate warm AMP")
        amplitudes[action] = frame
        order.append(action)
        require(len(order) <= 64, "warmup exceeds 64 total calls")
    warm = list(map(integer, reader.take("WARM", 3)))
    require(all(1 <= value <= 32 for value in warm), "WARM limit outside 1..32")
    expected_warm = list(range(101, 101 + warm[0])) + list(range(201, 201 + warm[1]))
    require(order == expected_warm, "warm action order/count does not match WARM")
    for action in range(301, 316):
        frame = parse_amp(reader, dimensions)
        require(frame["action"] == action, "physical action missing/out of order")
        amplitudes[action] = frame
        order.append(action)
    seed_status = integer(reader.take("SEED", 2)[0])
    seed = []
    for slot in range(1, 5):
        row = reader.take("SEEDVALUE", 4)
        require(integer(row[0]) == slot, "SEEDVALUE missing/out of order")
        seed.append(list(map(bits, row[1:])))
    order.append(350)
    frame = parse_amp(reader, dimensions)
    require(frame["action"] == 316, "missing seeded physical action316")
    amplitudes[316] = frame
    order.append(316)
    finalization = list(map(integer, reader.take("FINALIZE", 3)))
    order.append(360)
    for action in range(317, 320):
        frame = parse_amp(reader, dimensions)
        require(frame["action"] == action, "post-finalization action missing/out of order")
        amplitudes[action] = frame
        order.append(action)
    order.append(900)
    keys = {}
    for action in KEYS:
        frame = parse_key(reader, capacity)
        require(frame["action"] == action, "key action missing/out of order")
        keys[action] = frame
        order.append(action)
    finalization_end = list(map(integer, reader.take("FINALIZE_END", 3)))
    order.append(1900)
    done = list(map(integer, reader.take("DONE", 6)))
    closed = integer(reader.take("CLOSED", 2)[0])
    reader.done()
    require(done == warm + [19, 28, 0], "DONE count/health mismatch")
    require(closed == 0, "observer unhealthy after close")
    return dict(cache=cache, dimensions=dimensions, capacity=capacity, warm=warm,
                amplitudes=amplitudes, seed_status=seed_status, seed=seed,
                finalization=finalization, finalization_end=finalization_end,
                keys=keys, order=order)


def parse_trace(path):
    reader = Reader(path)
    version, observed, capacity, record_size = map(integer, reader.take("H", 5))
    require(version == 1 and observed in (0, 1) and capacity == 32768,
            "trace header differs from observer schema")
    require(0 < record_size * capacity < 16 * 1024 * 1024, "trace journal exceeds resource bound")
    records, flushes, pending = [], [], []
    while reader.peek() is not None:
        if reader.peek() == "R":
            values = list(map(integer, reader.take("R", 57)))
            record = dict(zip(TRACE_NAMES, values[:26]))
            record["before"], record["after"] = values[26:41], values[41:56]
            require(record["sequence"] == len(records) + 1, "trace sequence gap/duplicate")
            require(record["kind"] in range(1, 8), "unknown trace kind")
            require(all(0 <= x < 1 << 64 for x in record["before"] + record["after"]),
                    "trace counter outside uint64")
            require(all(record[f"f{x}"] >= 0 and record[f"r{x}"] >= 0 for x in range(4)),
                    "missing trace floating-environment sample")
            records.append(record)
            pending.append(record)
            require(len(pending) <= capacity, "trace exceeds per-flush capacity")
        elif reader.peek() == "F":
            serial, count, attempts, health, token = map(integer, reader.take("F", 6))
            require(serial == len(flushes) + 1 and count == len(pending) and attempts == len(records),
                    "trace flush count/serial/attempt mismatch")
            require(health == 0, f"observer health failure {health}")
            flushes.append(dict(serial=serial, token=token, records=pending))
            pending = []
        else:
            raise EvidenceError("unexpected trace record")
    require(not pending and flushes, "unflushed/incomplete trace")
    require(not flushes[-1]["records"], "missing final empty close flush")
    require(all(flush["records"] for flush in flushes[:-1]), "unexpected empty intermediate flush")
    groups = {}
    order = []
    for flush in flushes[:-1]:
        rows = flush["records"]
        begin = rows[0]
        action = begin["action"]
        require(begin["kind"] == 1 and all(row["kind"] != 1 for row in rows[1:]),
                "each flush must contain one leading BEGIN")
        require(action not in groups, "duplicate action flush")
        require(all(row["action"] == action and row["process"] == begin["process"] for row in rows),
                "mixed actions/processes inside a flush")
        groups[action] = rows
        order.append(action)
    return dict(observed=bool(observed), record_size=record_size, records=records,
                flushes=flushes, groups=groups, order=order)


class Checks:
    def __init__(self):
        self.failures = []

    def add(self, condition, message):
        if not condition:
            self.failures.append(message)


def key_inputs(frame):
    return {name: frame[name] for name in ("n", "rank", "scale", "momenta", "qp_momenta", "masses")}


def check_failure_latches(frame, checks, prefix):
    action, process = frame["action"], frame["process"]
    own = slice(0, 2) if process == 1 else slice(2, 4)
    other = slice(2, 4) if process == 1 else slice(0, 2)
    checks.add(frame["latches_after"][own] == ["F", "F"],
               prefix + f"AMP {action} retained active-prefix failure latches")
    checks.add(frame["latches_after"][other] == frame["latches_before"][other],
               prefix + f"AMP {action} changed another prefix failure latches")
    if action > 301:
        checks.add(frame["latches_before"] == ["T"] * 4,
                   prefix + f"AMP {action} lacks deliberate software-latch poison")
    elif action == 301:
        checks.add(frame["latches_before"] == ["F"] * 4,
                   prefix + "initial clean reference has preexisting software failure")


def check_records(data, checks, label):
    prefix = f"{label}: "
    for action, frame in data["amplitudes"].items():
        checked = action in PHYSICAL
        process = frame["process"]
        check_failure_latches(frame, checks, prefix)
        if checked:
            expected_process = 2 if action in (302, 305, 308, 311, 314, 317, 319) else 1
            phase = (action - 301) // 3 + 1 if action <= 315 else (6 if action == 316 else 7)
            checks.add(process == expected_process and frame["phase"] == phase,
                       prefix + f"AMP {action} process/phase changed")
            checks.add(frame["status"] in CHECKED, prefix + f"AMP {action} is not checked TIRust")
            checks.add(all(finite64(row[4]) and 0 <= binary64(row[4]) <= 1e-6
                           for row in frame["values"][:frame["ns"] + 1]),
                       prefix + f"AMP {action} loop accuracy outside [0,1e-6]")
        else:
            checks.add(action // 100 == process and frame["phase"] == 0,
                       prefix + f"warm AMP {action} metadata changed")
            checks.add(0 <= frame["status"] < 400, prefix + f"warm AMP {action} failed status")
        words = frame["model"] + [x for row in frame["momenta"] + frame["values"] for x in row]
        checks.add(all(finite64(word) for word in words), prefix + f"AMP {action} has nonfinite data")
        checks.add(frame["model"] == frame["model_before"] and frame["momenta"] == frame["momenta_before"],
                   prefix + f"AMP {action} mutated its physical/model inputs")
        checks.add(frame["generation_after"] > 0, prefix + f"AMP {action} lacks live context")
        checks.add(frame["generation_before"] in (-1, frame["generation_after"]),
                   prefix + f"AMP {action} changes an existing generation unexpectedly")
    for process, count in enumerate(data["warm"], 1):
        frames = [data["amplitudes"][100 * process + index] for index in range(1, count + 1)]
        checks.add(frames[-1]["status"] in CHECKED and
                   all(frame["status"] not in CHECKED for frame in frames[:-1]),
                   prefix + f"prefix {process} warmup is not first checked stopping point")
    nominal = data["amplitudes"][301]
    for phase in range(1, 6):
        actions = list(range(301 + 3 * (phase - 1), 304 + 3 * (phase - 1)))
        first, middle, last = [data["amplitudes"][action] for action in actions]
        checks.add(all(frame["model"] == first["model"] and frame["momenta"] == first["momenta"]
                       for frame in (middle, last)),
                   prefix + f"phase {phase} A/B/A physical inputs differ")
        checks.add(first["status"] == last["status"] and first["values"] == last["values"],
                   prefix + f"phase {phase} return-to-A amplitude differs")
    for action, reference in ((313, 301), (314, 302), (315, 301), (316, 301),
                              (317, 302), (318, 301), (319, 302)):
        actual, expected = data["amplitudes"][action], data["amplitudes"][reference]
        checks.add(actual["model"] == expected["model"] and actual["momenta"] == expected["momenta"] and
                   actual["status"] == expected["status"] and actual["values"] == expected["values"],
                   prefix + f"restored/recreated physical {action} differs from {reference}")
    for action in (303, 304, 306, 307, 309, 310, 312, 313, 315, 316, 318):
        frame = data["amplitudes"][action]
        if action in (304, 306):
            checks.add(frame["momenta"] == nominal["momenta"] and frame["model"][0] == nominal["model"][0]
                       and binary64(frame["model"][1]) == 2 * binary64(nominal["model"][1]),
                       prefix + "doubled-scale physical inputs incorrect")
        elif action in (307, 309):
            checks.add(frame["momenta"] != nominal["momenta"] and frame["model"] == nominal["model"],
                       prefix + "boost physical input unchanged or model changed")
        elif action in (310, 312):
            checks.add(binary64(frame["model"][0]) == 92.0 and frame["momenta"] != nominal["momenta"],
                       prefix + "alternate physical mass input incorrect")
        else:
            checks.add(frame["model"] == nominal["model"] and frame["momenta"] == nominal["momenta"],
                       prefix + f"nominal physical input not restored at {action}")
    checks.add(data["seed_status"] == 0 and all(finite64(x) for row in data["seed"] for x in row),
               prefix + "real seed failed or nonfinite")
    old, absent = data["finalization"]
    new, final_absent = data["finalization_end"]
    checks.add(old > 0 and absent == -1 and new > old and final_absent == -1,
               prefix + "finalization/recreation generation evidence invalid")
    checks.add(data["amplitudes"][317]["generation_before"] == -1 and
               all(data["amplitudes"][a]["generation_after"] == new for a in (317, 318, 319)),
               prefix + "B/A/B did not recreate/use the new generation")
    witnesses = []
    for precision in (1, 2):
        frames = {key: data["keys"][1000 + 100 * precision + key] for key in range(1, 15)}
        for key, frame in frames.items():
            checks.add(frame["stable"], prefix + f"KEY {frame['action']} unsuccessful")
            input_words = [frame["scale"]] + [x for rows in (frame["momenta"], frame["masses"]) for row in rows for x in row]
            checks.add(all(finite64(word) for word in input_words) and binary64(frame["scale"]) > 0,
                       prefix + f"KEY {frame['action']} nonfinite/invalid inputs")
            checks.add(frame["scale"] == frame["scale_before"] and frame["momenta"] == frame["momenta_before"] and
                       frame["masses"] == frame["masses_before"] and frame["qp_momenta"] == frame["qp_momenta_before"],
                       prefix + f"KEY {frame['action']} mutated its input buffers")
            if precision == 2 and all(finite64(word) for row in frame["momenta_before"] for word in row):
                promoted = [[part for word in row for part in promoted128(word)] for row in frame["momenta_before"]]
                checks.add(frame["qp_momenta_before"] == promoted,
                           prefix + f"KEY {frame['action']} QP input is not the exact declared DP promotion")
            if precision == 1:
                finite = all(finite64(word) for row in frame["values"] for word in row)
            else:
                finite = all(finite128(row[offset:offset + 2]) for row in frame["values"] for offset in (0, 2))
            checks.add(finite, prefix + f"KEY {frame['action']} nonfinite coefficients")
        for group in ((1, 2, 7, 13, 14), (3, 4), (5, 6), (8, 9, 12), (10, 11)):
            reference = frames[group[0]]
            for key in group[1:]:
                checks.add(key_inputs(frames[key]) == key_inputs(reference) and
                           frames[key]["values"] == reference["values"],
                           prefix + f"precision {precision} repeated key {key} differs")
        for first, second, field in ((1, 3, "masses"), (1, 5, "scale"), (8, 10, "momenta")):
            left, right = key_inputs(frames[first]), key_inputs(frames[second])
            allowed = {field, "qp_momenta"} if field == "momenta" else {field}
            checks.add(left[field] != right[field] and all(left[k] == right[k] for k in left if k not in allowed),
                       prefix + f"precision {precision} {field} variation is not isolated")
            changed = coefficient_signature(frames[first]) != coefficient_signature(frames[second])
            checks.add(changed, prefix + f"precision {precision} {field} variation has no coefficient witness")
            witnesses.append(dict(precision=precision, field=field, keys=[first, second], changed=changed))
    return witnesses


def check_key_caches(data, checks, label):
    witnesses = []
    cold = {1, 3, 5, 8, 10}
    previous = None
    for action in KEYS:
        frame = data["keys"][action]
        before, after = frame["before"], frame["after"]
        prefix = f"{label}: KEY {action}: "
        if previous is None:
            checks.add(not any(before), prefix + "initial key cache not empty")
        else:
            checks.add(before == previous, prefix + "cache changed between key actions")
        previous = after
        delta = [a - b for a, b in zip(after, before)]
        checks.add(all(value >= 0 for value in delta), prefix + "cache counters decreased without clear")
        checks.add(after[7] <= 268435456 and after[12] <= 134217728,
                   prefix + "numerical cache budget exceeded")
        qp_delta = delta[9:]
        dp_delta = delta[:9]
        for aggregate, qp in ((4, 0), (5, 1), (6, 2), (7, 3), (8, 4)):
            dp_delta[aggregate] -= qp_delta[qp]
        if frame["precision"] == 1:
            checks.add(not any(qp_delta), prefix + "DP changed QP cache")
        else:
            checks.add(not any(dp_delta), prefix + "QP changed DP caches")
        if frame["key"] == 13:
            expected_qp = [0, 0, 0, 0, 0, int(frame["precision"] == 2)]
            checks.add(not any(delta[:9]) and qp_delta == expected_qp,
                       prefix + "independent stability check reused or inserted primary cache")
        elif data["cache"] == 0:
            expected_qp = [0, 0, 0, 0, 0, int(frame["precision"] == 2)]
            checks.add(not any(delta[:9]) and qp_delta == expected_qp,
                       prefix + "cache-disabled request changed retained stores")
            checks.add(after[6] == after[7] == after[11] == after[12] == 0,
                       prefix + "cache-disabled request retained entries")
        elif frame["precision"] == 2:
            if frame["key"] in cold:
                checks.add(qp_delta[:3] == [0, 1, 1] and qp_delta[3] > 0 and qp_delta[4:] == [0, 0],
                           prefix + "new QP key did not create one distinct entry")
            else:
                checks.add(qp_delta == [1, 0, 0, 0, 0, 0], prefix + "repeat QP key did not hit")
        elif frame["key"] in cold:
            checks.add(dp_delta[6] > 0 and dp_delta[7] > 0 and sum(dp_delta[k] for k in (1, 3, 5)) > 0,
                       prefix + "new DP key did not create distinct entries")
        else:
            checks.add(dp_delta[6] == dp_delta[7] == 0 and sum(dp_delta[k] for k in (0, 2, 4)) > 0,
                       prefix + "repeat DP key did not hit without insertion")
        if data["cache"] == -1:
            checks.add(delta[8] == 0 and delta[13] == 0, prefix + "unexpected insertion refusal")
        witnesses.append(dict(action=action, precision=frame["precision"], cold=frame["key"] in cold,
                              before=before, after=after, delta=delta))
    return witnesses


def check_trace(data, trace, checks, label):
    prefix = label + ": "
    observed = label.startswith("observed-")
    checks.add(trace["observed"] == observed, prefix + "trace observed/plain marker incorrect")
    checks.add(trace["order"] == data["order"], prefix + "trace actions do not match complete driver order")
    for action, rows in trace["groups"].items():
        if action in data["amplitudes"]:
            expected_process = data["amplitudes"][action]["process"]
        elif action in data["keys"]:
            expected_process = data["keys"][action]["process"]
        else:
            expected_process = 0 if action == 350 else 1
        checks.add(rows[0]["process"] == expected_process, prefix + f"trace action {action} has wrong process tag")
    for flush in trace["flushes"]:
        if flush["records"]:
            checks.add(flush["token"] == flush["records"][-1]["token_after"],
                       prefix + f"flush {flush['serial']} token differs from last record")
        else:
            checks.add(flush["token"] == -1, prefix + "close flush retained active context")
    active, serial = -1, 0
    registration = {}
    seeded_rows = trace["groups"].get(316, [])
    first_seeded_evaluation = next((i for i, row in enumerate(seeded_rows)
                                    if row["kind"] in (6, 7)), None)
    seeded_entry_sequence = (next((row["sequence"] for row in seeded_rows[:first_seeded_evaluation]
                                   if row["kind"] == 4), None)
                             if first_seeded_evaluation is not None else None)
    for row in trace["records"]:
        action, kind = row["action"], row["kind"]
        checks.add(row["f0"] == row["f1"] and row["f2"] == row["f3"] and
                   row["r0"] == row["r1"] and row["r2"] == row["r3"],
                   prefix + f"observer altered IEEE state at sequence {row['sequence']}")
        checks.add(observed or kind == 1, prefix + "plain trace unexpectedly contains a native wrapper")
        if kind not in (5, 6, 7):
            checks.add(row["n"] == NA and row["rank"] == NA, prefix + "unexpected topology metadata on non-tensor call")
        if kind not in (6, 7):
            checks.add(all(row[key] == NA for key in ("poles", "finite", "check", "cache")),
                       prefix + "unexpected evaluation flags on non-evaluation call")
        if kind != 2:
            checks.add(row["version"] == 0, prefix + "unexpected ABI version metadata")
        if kind == 2:
            serial += 1
            checks.add(row["status"] == 0 and row["version"] == 1 and active == -1 and
                       row["token_before"] == -1 and row["token_after"] == serial,
                       prefix + "context creation token/status mismatch")
            active = row["token_after"]
        elif kind == 3:
            checks.add(action in (360, 1900) and row["status"] == 0 and active > 0 and
                       row["token_before"] == active and row["token_after"] == -1,
                       prefix + "unexpected/failed context destruction")
            active = -1
        else:
            checks.add(row["token_before"] == active and row["token_after"] == active,
                       prefix + f"context-token mismatch at sequence {row['sequence']}")
            if kind != 1:
                checks.add(active > 0, prefix + "native call on absent context")
        if kind in (1, 4, 5):
            checks.add(row["status"] == 0, prefix + f"non-evaluation call failed at {row['sequence']}")
        if kind == 4:
            checks.add(all(row[k] == 0 for k in TRACE_NAMES[14:18]), prefix + "clear statistics failed/unavailable")
            checks.add(not any(row["after"]), prefix + "clear did not reset all 15 counters")
            # Only the first entry clear may observe the explicitly cached seed;
            # later clears in action316 must obey the disabled-cache invariant.
            seeded_entry = action == 316 and row["sequence"] == seeded_entry_sequence
            if data["cache"] == 0 and not seeded_entry:
                checks.add(row["before"][6] == row["before"][7] == row["before"][11] == row["before"][12] == 0,
                           prefix + f"disabled event retained entries at {action}")
        else:
            checks.add(all(row[k] == NA for k in TRACE_NAMES[14:18]) and
                       not any(row["before"] + row["after"]),
                       prefix + "statistics queried outside clear wrapper")
        if kind == 5:
            registration.setdefault((active, row["process"]), []).append((row["n"], row["rank"]))
        if kind in (6, 7):
            expected_cache = 1 if action == 350 or data["cache"] == -1 else 0
            checks.add(1 <= row["n"] <= 6 and 0 <= row["rank"] <= 6 and row["poles"] == 0 and
                       row["finite"] == 1 and row["check"] in (0, 1) and row["cache"] == expected_cache,
                       prefix + f"native evaluation request metadata changed at {action}")
            checks.add(row["status"] in (0, 1, 2, 3), prefix + "invalid native evaluation status")
    if not observed:
        checks.add(serial == 0 and active == -1 and all(f["token"] == -1 for f in trace["flushes"]),
                   prefix + "plain observer token is not absent")
        return {"scope": "plain driver action journal; no native wrapper observations"}
    checks.add(serial == 2 and active == -1, prefix + "expected exactly two context generations and final absence")
    for process in (1, 2):
        old, new = registration.get((1, process), []), registration.get((2, process), [])
        checks.add(bool(old) and new == old, prefix + f"prefix {process} did not repeat its full plan registration")
    boundary = []
    for action in PHYSICAL:
        rows = trace["groups"].get(action, [])
        evaluations = [i for i, row in enumerate(rows) if row["kind"] in (6, 7)]
        clears = [i for i, row in enumerate(rows) if row["kind"] == 4]
        creates = [i for i, row in enumerate(rows) if row["kind"] == 2]
        checks.add(bool(evaluations), prefix + f"physical {action} has no native reductions")
        if not evaluations:
            continue
        first, last = evaluations[0], evaluations[-1]
        entry = [i for i in clears if i < first]
        final = [i for i in clears if i > last]
        # Successor avoids constructing an absent context merely to clear it.
        # Action317 recreates at first plan registration, before reduction.
        fresh_entry = action == 317 and len(creates) == 1 and not entry and creates[0] < first
        checks.add(not entry if action == 317 else bool(entry),
                   prefix + f"physical {action} incorrect automatic entry clear")
        checks.add(bool(final), prefix + f"physical {action} lacks final automatic native clear")
        checks.add(not creates or fresh_entry, prefix + f"physical {action} unexpectedly recreates context")
        if action == 317:
            checks.add(fresh_entry, prefix + "action317 created/cleared an absent context at entry")
        native = [rows[i] for i in evaluations]
        checks.add(len(native) == 2 and [r["kind"] for r in native] == [6, 6] and
                   [r["check"] for r in native] == [0, 1] and
                   [r["status"] for r in native] == [0, 0],
                   prefix + f"physical {action} lacks successful primary/independent DP passes")
        frame = data["amplitudes"][action]
        if rows:
            checks.add(rows[0]["token_before"] == frame["generation_before"] and
                       rows[-1]["token_after"] == frame["generation_after"],
                       prefix + f"physical {action} observer/module generation disagreement")
        nonempty = bool(final) and rows[final[-1]]["before"][6] > 0
        if action == 316:
            checks.add(bool(entry) and rows[entry[0]]["before"][6] > 0,
                       prefix + "seeded incoming cache was not observed and cleared")
        boundary.append(dict(action=action, native_evaluations=len(evaluations), native_clears=len(clears),
                             fresh_context_entry=fresh_entry, nonempty_before_final_clear=nonempty))
    if data["cache"] == -1:
        checks.add(any(row["nonempty_before_final_clear"] for row in boundary),
                   prefix + "enabled physical calls never retained native entries")
    for action, process in ((317, 2), (318, 1)):
        rows = trace["groups"].get(action, [])
        registers = [i for i, row in enumerate(rows) if row["kind"] == 5 and row["process"] == process]
        evaluations = [i for i, row in enumerate(rows) if row["kind"] in (6, 7)]
        checks.add(bool(registers) and bool(evaluations) and max(registers) < min(evaluations),
                   prefix + f"prefix {process} did not re-register before evaluation after finalize")
    for action in KEYS:
        rows = trace["groups"].get(action, [])
        frame = data["keys"][action]
        evaluations = [row for row in rows if row["kind"] in (6, 7)]
        checks.add(all(row["kind"] in (1, 5, 6, 7) for row in rows),
                   prefix + f"key {action} has a hidden clear/context lifecycle operation")
        checks.add(len(evaluations) == 1 and evaluations[0]["status"] == 0 and
                   evaluations[0]["kind"] == frame["precision"] + 5 and
                   evaluations[0]["n"] == frame["n"] and evaluations[0]["rank"] == frame["rank"] and
                   evaluations[0]["check"] == int(frame["key"] == 13),
                   prefix + f"key {action} lacks one successful native request of the correct precision")
    seed_rows = trace["groups"].get(350, [])
    checks.add([row["kind"] for row in seed_rows] == [1, 6] and
               seed_rows[-1]["status"] == 0 and seed_rows[-1]["n"] == 1 and seed_rows[-1]["rank"] == 0,
               prefix + "real seed trace differs from one DP tadpole request")
    for action in (360, 1900):
        checks.add([row["kind"] for row in trace["groups"].get(action, [])] == [1, 3],
                   prefix + f"finalization action {action} does not contain exactly one real destroy")
    checks.add([row["kind"] for row in trace["groups"].get(900, [])] == [1, 4],
               prefix + "key preparation does not contain exactly one real cache clear")
    return dict(boundaries=boundary, registrations={f"generation{k[0]}-process{k[1]}": v for k, v in registration.items()},
                generations=serial, native_records=sum(row["kind"] != 1 for row in trace["records"]))


def parity(left, right, checks, name, include_flags):
    checks.add(left["warm"] == right["warm"] and left["dimensions"] == right["dimensions"] and
               left["capacity"] == right["capacity"] and left["order"] == right["order"],
               name + ": warmup/shape/action coverage differs")
    for action in sorted(set(left["amplitudes"]) | set(right["amplitudes"])):
        a, b = left["amplitudes"].get(action), right["amplitudes"].get(action)
        if a is None or b is None:
            checks.add(False, name + f": AMP {action} absent in one run")
            continue
        fields = ("process", "phase", "status", "nd", "ns", "generation_before", "generation_after",
                  "model", "model_before", "momenta", "momenta_before", "values",
                  "latches_before", "latches_after")
        for field in fields + (("flags",) if include_flags else ()):
            checks.add(a[field] == b[field], name + f": AMP {action} {field} differs bit-for-bit")
    checks.add(left["seed_status"] == right["seed_status"] and left["seed"] == right["seed"],
               name + ": seed status/output differs")
    checks.add(left["finalization"] == right["finalization"] and left["finalization_end"] == right["finalization_end"],
               name + ": finalization/recreation differs")
    for action in KEYS:
        a, b = left["keys"][action], right["keys"][action]
        checks.add(key_inputs(a) == key_inputs(b) and a["values"] == b["values"] and a["stable"] == b["stable"],
                   name + f": KEY {action} inputs/status/coefficients differ bit-for-bit")
        if left["cache"] == right["cache"]:
            checks.add(a["before"] == b["before"] and a["after"] == b["after"],
                       name + f": KEY {action} observer changed counters")


def analyze(root):
    root = Path(root)
    checks = Checks()
    parsed, runs = {}, {}
    for label in LABELS:
        directory = root / label
        result = {"inputs": {}}
        runs[label] = result
        start_failures = len(checks.failures)
        try:
            for name in ("records.txt", "trace.txt"):
                path = directory / name
                result["inputs"][name] = {"sha256": digest(path), "bytes": path.stat().st_size}
            data = parse_records(directory / "records.txt")
            trace = parse_trace(directory / "trace.txt")
            checks.add(data["cache"] == (0 if label.endswith("cache0") else -1), label + ": incorrect cache configuration")
            result["coefficient_witnesses"] = check_records(data, checks, label)
            result["key_cache_witnesses"] = check_key_caches(data, checks, label)
            result["native_lifecycle"] = check_trace(data, trace, checks, label)
            result["coverage"] = dict(warmup=data["warm"], physical=19, key_actions=28,
                                      trace_records=len(trace["records"]), trace_flushes=len(trace["flushes"]))
            parsed[label] = data
        except (EvidenceError, OSError, UnicodeError, ValueError, KeyError, IndexError) as error:
            checks.add(False, f"{label}: incomplete/malformed evidence: {error}")
        result["failures"] = checks.failures[start_failures:]
        result["passed"] = not result["failures"]
    parity_start = len(checks.failures)
    comparisons = (
        ("plain-cache0", "observed-cache0"),
        ("plain-cache-all", "observed-cache-all"),
        ("plain-cache0", "plain-cache-all"),
        ("observed-cache0", "observed-cache-all"),
    )
    for left, right in comparisons:
        if left in parsed and right in parsed:
            parity(parsed[left], parsed[right], checks, f"{left} vs {right}", include_flags=True)
        else:
            checks.add(False, f"{left} vs {right}: parity unavailable because a stream is incomplete")
    return dict(schema_version=1, kind="subprocess-cache-lifecycle-retained-text-analysis",
                passed=not checks.failures, failures=checks.failures, runs=runs,
                parity={"exact_ieee_required_across_all_four": True,
                        "comparisons": [list(pair) for pair in comparisons],
                        "failures": checks.failures[parity_start:]},
                scope="Generated physical A/B/A lifecycle and separate generated tensor-key diagnostics; no timing acceptance",
                limitations=["Execution exit status, source/object/library hashes, and runtime provenance are checked by the separate binder.",
                             "Action317 skips absent-context entry clear, then creates/registers before the first native evaluation.",
                             "Physical MZ variation changes external mass/invariants. Internal complex mass-key variation belongs only to the separate tensor subsection.",
                             "Rank-six generated tensor qualification is outside this bounded protocol."],
                analyzer_sha256=digest(Path(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing the four named retained worker directories")
    parser.add_argument("--output", type=Path, help="write JSON here; otherwise print JSON to stdout")
    arguments = parser.parse_args()
    report = analyze(arguments.root)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
