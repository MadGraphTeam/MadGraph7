#!/usr/bin/env python3
"""Validate retained private-observer text; never launch or modify an artifact.

Public parity is necessary, but is not runtime qualification or cause evidence.
This source-only validator has not been executed or tested during preparation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import protocol


PUBLIC_HEADER = (
    "DCO_PUBLIC 1 flags:invalid,divzero,overflow,underflow,inexact;"
    "traps:same;rounding:nearest0,zero1,up2,down3"
)
HEX_WORD = re.compile(r"[0-9A-Fa-f]{16}")
INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)")
MAX_PUBLIC_BYTES = 2 * 1024 * 1024
MAX_NATIVE_BYTES = 512 * 1024 * 1024
MAX_NATIVE_LINE = 1024 * 1024
EXPECTED_CONTEXTS = [(0, 1, 0, 0)] + [(i, i, 1, 0) for i in range(1, 101)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integers(words, expected, label):
    require(len(words) == expected and all(INTEGER.fullmatch(word) for word in words),
            "Invalid integer record: " + label)
    values = tuple(map(int, words))
    require(all(-(1 << 63) <= value < (1 << 63) for value in values),
            "Integer exceeds signed 64-bit range: " + label)
    return values


def hexadecimal(words, expected, label):
    require(len(words) == expected and all(isinstance(word, str) and HEX_WORD.fullmatch(word) for word in words),
            "Invalid binary64 word record: " + label)
    # Integer bit patterns preserve signed zero, NaN payloads, and every bit.
    return tuple(int(word, 16) for word in words)


def split_record(line, tag):
    words = line.split()
    require(words and words[0] == tag, "Expected " + tag + " record")
    return words[1:]


def parse_public(data):
    """Strict v1 public grammar; all 101 calls and a clean CLOSE are required."""
    require(isinstance(data, bytes) and 0 < len(data) <= MAX_PUBLIC_BYTES,
            "Public log is empty or exceeds the declared bound")
    require(data.endswith(b"\n") and b"\r" not in data and b"\x00" not in data,
            "Public log is truncated or has invalid framing")
    lines = data.decode("ascii").splitlines()
    require(lines[0] == PUBLIC_HEADER, "Wrong public schema/header")
    require(len(lines) == 1 + 101 * 6 + 1, "Missing or extra public records")
    calls = []
    for index, context in enumerate(EXPECTED_CONTEXTS):
        offset = 1 + index * 6
        header = integers(split_record(lines[offset], "PUBLIC"), 8, "PUBLIC")
        require(header[:4] == context, "Public call/point/phase/loop ordering changed")
        seq, point, phase, loop, code, ninput, nans, nacc = header
        require((ninput, nans, nacc) == (21, 8, 2), "Public dimensions changed")
        require(0 <= code <= 499, "Out-of-range host status")
        input_words = hexadecimal(split_record(lines[offset + 1], "INPUT"), ninput, "INPUT")
        answers = hexadecimal(split_record(lines[offset + 2], "ANS"), nans, "ANS")
        accuracies = hexadecimal(split_record(lines[offset + 3], "ACC"), nacc, "ACC")
        state = integers(split_record(lines[offset + 4], "STATE"), 22, "STATE")
        require(all(value in (0, 1) for part in (state[:10], state[11:21]) for value in part),
                "Invalid IEEE flag or halting-mode value")
        require(state[10] in range(4) and state[21] in range(4), "Unsupported rounding mode")
        health = integers(split_record(lines[offset + 5], "HEALTH"), 3, "HEALTH")
        require(health == (0, 0, 0), "Public observer health failed")
        calls.append({"sequence": seq, "point": point, "phase": phase, "loop": loop,
                      "status": code, "input_words": input_words, "answer_words": answers,
                      "accuracy_words": accuracies, "entry_state": state[:11],
                      "exit_state": state[11:], "health": health})
    trailer = integers(split_record(lines[-1], "CLOSE"), 4, "CLOSE")
    require(trailer == (100, 100, 0, 0), "Missing or unhealthy public CLOSE")
    return calls


def retained_bytes(path, limit):
    """Read only a bounded existing file under the repository tree."""
    path = Path(path).resolve(strict=True)
    require(protocol.within(path, protocol.ROOT), "Artifact escaped repository root")
    require(path.is_file() and path.stat().st_size <= limit, "Artifact absent or exceeds bound")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, "Artifact grew beyond bound while reading")
    return data


def read_arm(directory):
    directory = Path(directory).resolve(strict=True)
    public_bytes = retained_bytes(directory / "cause-public.txt", MAX_PUBLIC_BYTES)
    warmup_bytes = retained_bytes(directory / "warmup.dat", MAX_PUBLIC_BYTES)
    return {"directory": str(directory), "calls": parse_public(public_bytes),
            "warmup_bytes": warmup_bytes,
            "hashes": {"cause-public.txt": hashlib.sha256(public_bytes).hexdigest(),
                       "warmup.dat": hashlib.sha256(warmup_bytes).hexdigest()}}


def exact_keys(record, names):
    require(isinstance(record, dict) and set(record) == set(names.split()),
            "Unexpected fields in native " + str(record.get("kind") if isinstance(record, dict) else "record"))


def integer_value(value, low=0, high=(1 << 64) - 1):
    require(type(value) is int and low <= value <= high, "Invalid native integer")
    return value


def signed_word(value):
    return value if value < (1 << 63) else value - (1 << 64)


def json_records(path):
    """Stream a bounded native journal, rejecting floats and duplicate fields."""
    path = Path(path).resolve(strict=True)
    require(protocol.within(path, protocol.ROOT) and path.is_file(), "Invalid native journal path")
    require(path.stat().st_size <= MAX_NATIVE_BYTES, "Native journal exceeds the declared bound")
    def object_pairs(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "Duplicate native JSON key")
            output[key] = value
        return output
    def no_float(token):
        raise ValueError("Floating numeric literal in integer-only journal: " + token)
    total = 0
    with path.open("rb") as stream:
        while True:
            line = stream.readline(MAX_NATIVE_LINE + 1)
            if not line:
                return
            total += len(line)
            require(total <= MAX_NATIVE_BYTES and len(line) <= MAX_NATIVE_LINE and line.endswith(b"\n"),
                    "Native journal exceeded bounds or is truncated")
            require(b"\r" not in line and b"\x00" not in line, "Invalid native line framing")
            record = json.loads(line, object_pairs_hook=object_pairs,
                                parse_float=no_float, parse_constant=no_float)
            require(isinstance(record, dict) and isinstance(record.get("kind"), str),
                    "Native record lacks its kind")
            yield record


NATIVE_PAYLOAD_KEYS = {
    "point": "kind",
    "host": "kind tag words",
    "native_begin": "kind context input_bits configured_work_bits n rank poles finite check cache",
    "native_end": "kind status error",
    "qp_cache_hit": "kind new_numerical_evaluation",
    "options": "kind method_choice partial_rescue precision_tiers gate_bits choice_gate_bits master_policy partial_work_observable",
    "upstream_completion": "kind result_ok workspace_bits",
    "lane_route": "kind lane route accepted_tier_bits direct_is_pure_dp_certificate",
    "diagnostic": "kind lane index text",
    "execution": "kind input_rows master_rows_all_precisions direct_successes fallback_rows fallback_successes symbolic_rows rotation_rows rotation_failures rotation_unchecked checked_rows checked_successes expansion_rows expansion_successes",
    "tier_attempt": "kind attempt bits rows successes",
    "fallback_preparation_error": "kind error",
}
HOST_LENGTHS = {10: 4, 11: 21, 12: 3, 13: 8, 14: 2, 20: 5, 21: 2, 22: 2,
                30: 4, 31: 1, 32: 1, 33: 1, 34: 2, 40: 4, 41: 2,
                43: 2, 44: 2, 45: 3, 46: 4, 47: 2, 49: 1}


def validate_payload(data):
    require(isinstance(data, dict) and data.get("kind") in NATIVE_PAYLOAD_KEYS,
            "Unknown native payload kind")
    kind = data["kind"]
    exact_keys(data, NATIVE_PAYLOAD_KEYS[kind])
    if kind == "host":
        tag = integer_value(data["tag"], 0, (1 << 31) - 1)
        require(tag in HOST_LENGTHS or tag == 42, "Unknown host event tag")
        require(isinstance(data["words"], list) and len(data["words"]) <= 4096,
                "Invalid host payload length")
        words = hexadecimal(data["words"], len(data["words"]), "native host words")
        if tag in HOST_LENGTHS:
            require(len(words) == HOST_LENGTHS[tag], "Wrong payload width for host tag " + str(tag))
        return {"kind": kind, "tag": tag, "words": words}
    if kind == "native_begin":
        for name in ("context", "n", "rank", "poles"):
            integer_value(data[name])
        require((data["input_bits"], data["configured_work_bits"]) in ((53, 53), (113, 256)),
                "Unexpected native input/work precision configuration")
        require(all(type(data[name]) is int and data[name] in (0, 1) for name in ("finite", "check", "cache")),
                "Invalid native call controls")
    elif kind == "native_end":
        integer_value(data["status"], -(1 << 31), (1 << 31) - 1)
        require(isinstance(data["error"], str) and len(data["error"].encode()) <= 65536,
                "Invalid native error text")
    elif kind == "qp_cache_hit":
        require(data["new_numerical_evaluation"] is False, "Cache hit presented as new evaluation")
    elif kind == "options":
        require(data["method_choice"] is True and data["partial_rescue"] is False
                and data["precision_tiers"] is False and data["gate_bits"] is None
                and data["choice_gate_bits"] is None and data["master_policy"] == "Default"
                and data["partial_work_observable"] is False, "Unqualified upstream numerical policy")
    elif kind == "upstream_completion":
        require(type(data["result_ok"]) is bool and type(data["workspace_bits"]) is int
                and data["workspace_bits"] == 53, "Unexpected upstream workspace/completion")
    elif kind == "lane_route":
        integer_value(data["lane"])
        require(data["route"] in ("direct", "method_choice", "refused")
                and type(data["accepted_tier_bits"]) is int and data["accepted_tier_bits"] == 0
                and data["direct_is_pure_dp_certificate"] is False, "Unqualified lane route")
    elif kind == "diagnostic":
        integer_value(data["lane"]); integer_value(data["index"])
        require(isinstance(data["text"], str) and len(data["text"].encode()) <= 65536,
                "Invalid upstream diagnostic text")
    elif kind == "execution":
        for name, value in data.items():
            if name != "kind":
                integer_value(value)
        require(data["symbolic_rows"] == 0, "Unqualified symbolic work")
    elif kind == "tier_attempt":
        raise ValueError("Unexpected precision-tier work under the default-only protocol")
    elif kind == "fallback_preparation_error":
        require(isinstance(data["error"], str) and len(data["error"].encode()) <= 65536,
                "Invalid fallback error text")
    return data


def check_public_join(host, call):
    tags = [event["tag"] for event in host]
    require(tags[:2] == [10, 11] and tags[-3:] == [12, 13, 14], "Public/native host framing changed")
    require(all(tags.count(tag) == 1 for tag in (10, 11, 12, 13, 14)), "Duplicate public host event")
    require(tuple(map(signed_word, host[0]["words"]))
            == tuple(call[key] for key in ("sequence", "point", "phase", "loop")), "Host tag10 context mismatch")
    require(host[1]["words"] == call["input_words"], "Host tag11 input words mismatch")
    require(tuple(map(signed_word, host[-3]["words"])) == (call["status"], 8, 2), "Host tag12 status mismatch")
    require(host[-2]["words"] == call["answer_words"] and host[-1]["words"] == call["accuracy_words"],
            "Host tag13/14 result words mismatch")


def parse_native(path, public_calls, enabled):
    """Validate envelopes, sequence/native nesting, policy, and public joins."""
    require(type(enabled) is bool and len(public_calls) == 101, "Invalid native parser configuration")
    records = iter(json_records(path))
    sequence = native_count = 0
    summaries = []
    for flush_index in range(1, 103):
        begin = next(records, None)
        exact_keys(begin, "kind schema flush enabled events health dropped sequence native_calls partial_work_observable")
        require(begin["kind"] == "flush_begin" and begin["schema"] == 1 and begin["flush"] == flush_index
                and begin["enabled"] is enabled and begin["partial_work_observable"] is False,
                "Invalid native flush header")
        for key in ("schema", "flush", "events", "health", "dropped", "sequence", "native_calls"):
            integer_value(begin[key])
        require(begin["health"] == begin["dropped"] == 0 and begin["events"] <= 200000,
                "Native health/drop/capacity failure")
        require(enabled or (begin["events"] == begin["sequence"] == begin["native_calls"] == 0),
                "Disabled journal contains numerical observation")
        call = public_calls[flush_index - 1] if flush_index <= 101 else None
        require(call is not None or begin["events"] == 0, "Final close flush is not empty")
        active = None
        diagnostic = None
        diagnostic_index = 0
        host, native, event_kinds = [], [], []
        for event_index in range(begin["events"]):
            event = next(records, None)
            exact_keys(event, "kind seq point phase native fp_before fp_after data")
            require(event["kind"] == "event" and event["seq"] == sequence + 1, "Native event sequence gap")
            for key in ("seq", "native"):
                integer_value(event[key])
            integer_value(event["point"], -(1 << 63), (1 << 63) - 1)
            integer_value(event["phase"], -(1 << 31), (1 << 31) - 1)
            sequence += 1
            require(call is not None and (event["point"], event["phase"]) == (call["point"], call["phase"]),
                    "Native/public point context mismatch")
            require(isinstance(event["fp_before"], list) and len(event["fp_before"]) == 4
                    and isinstance(event["fp_after"], list) and len(event["fp_after"]) == 4,
                    "Invalid native FP-state dimensions")
            for value in event["fp_before"] + event["fp_after"]:
                integer_value(value)
            require(event["fp_before"] == event["fp_after"], "Observer changed FP/MPFR state")
            payload = validate_payload(event["data"])
            kind = payload["kind"]
            event_kinds.append(kind)
            if kind == "native_begin":
                require(active is None, "Nested native call")
                require(diagnostic is not None, "Native call lacks preceding Fortran diagnostic context")
                require((payload["input_bits"], payload["n"], payload["rank"])
                        == (diagnostic["precision"], diagnostic["n"], diagnostic["rank"]),
                        "Native/Fortran diagnostic metadata mismatch")
                require(payload["check"] == int(diagnostic["ctmode"] in (2, 5)),
                        "Native/Fortran primary/check role mismatch")
                native_count += 1
                active = {"id": native_count, "begin": payload, "details": [],
                          "host_diagnostic": dict(diagnostic)}
            require(event["native"] == (active["id"] if active else 0), "Wrong active native-call identity")
            if kind == "point":
                require(event_index == 0 and active is None, "Misplaced point event")
            elif kind == "host":
                require(active is None, "Host event inside native call")
                host.append(payload)
                if payload["tag"] == 30:
                    precision, ctmode, n, rank = map(signed_word, payload["words"])
                    require((precision == 53 and ctmode in (1, 2))
                            or (precision == 113 and ctmode in (4, 5)), "Unexpected Fortran diagnostic mode")
                    diagnostic_index += 1
                    diagnostic = {"ordinal": diagnostic_index, "precision": precision,
                                  "ctmode": ctmode, "n": n, "rank": rank}
            elif kind == "native_end":
                require(active is not None, "Unmatched native completion")
                active["end"] = payload
                completion = [detail for detail in active["details"] if detail["kind"] == "upstream_completion"]
                if completion:
                    require(len(completion) == 1 and active["begin"]["input_bits"] == 53,
                            "Invalid upstream completion scope")
                    if completion[0]["result_ok"] is False:
                        require(all(detail["kind"] in ("options", "upstream_completion") for detail in active["details"]),
                                "Stale routes/counters after upstream error")
                native.append(active); active = None
            elif kind != "native_begin":
                require(active is not None, "Unscoped native detail")
                if kind == "qp_cache_hit":
                    require(active["begin"]["input_bits"] == 113, "QP cache hit in DP call")
                else:
                    require(active["begin"]["input_bits"] == 53, "DP detail in non-DP call")
                active["details"].append(payload)
        require(active is None, "Native call truncated at flush")
        require(begin["sequence"] == sequence and begin["native_calls"] == native_count,
                "Cumulative native counters disagree")
        end = next(records, None)
        exact_keys(end, "kind flush events health dropped")
        for key in ("flush", "events", "health", "dropped"):
            integer_value(end[key])
        require(end == {"kind": "flush_end", "flush": flush_index, "events": begin["events"],
                        "health": 0, "dropped": 0}, "Native flush trailer mismatch")
        if enabled and call is not None:
            require(event_kinds and event_kinds[0] == "point" and event_kinds.count("point") == 1,
                    "Missing or duplicate native point boundary")
            check_public_join(host, call)
            summaries.append(classify_point(call, host, native))
    close = next(records, None)
    exact_keys(close, "kind schema enabled health dropped sequence native_calls flushes")
    for key in ("schema", "health", "dropped", "sequence", "native_calls", "flushes"):
        integer_value(close[key])
    require(close["enabled"] is enabled, "Native close enabled state mismatch")
    require(close == {"kind": "close", "schema": 1, "enabled": enabled, "health": 0, "dropped": 0,
                      "sequence": sequence, "native_calls": native_count, "flushes": 102},
            "Native close trailer missing or inconsistent")
    require(next(records, None) is None, "Extra native records after close")
    return {"validated": True, "enabled": enabled, "event_count": sequence,
            "native_call_count": native_count, "flush_count": 102, "points": summaries,
            "sha256": protocol.sha256(path)}


def word_value(word):
    """Offline interpretation only; never called by the numerical prototype."""
    return struct.unpack(">d", word.to_bytes(8, "big"))[0]


def hex_words(words):
    return [f"{word:016x}" for word in words]


def classify_point(call, host, native):
    """Report concurrent observations, without forcing a single-cause label."""
    diagnostics, dp_checks, branches, dp_groups = [], [], [], []
    active_diagnostic = None
    active_dp_group = None
    for index, event in enumerate(host):
        tag, words = event["tag"], event["words"]
        signed = tuple(map(signed_word, words))
        if active_dp_group is not None and active_dp_group["diagnostic_stable"] is not None:
            require(tag == 21, "DP contraction completion must follow diagnostic return")
        if tag == 20:
            require(active_dp_group is None, "Nested DP loop group")
            ctmode, sqso, group_id, n, rank = signed
            require(ctmode in (1, 2) and min(sqso, group_id, n, rank) >= 0,
                    "Invalid DP loop-group metadata")
            active_dp_group = {"ordinal": len(dp_groups) + 1, "ctmode": ctmode,
                               "sqso": sqso, "group_id": group_id, "n": n, "rank": rank,
                               "diagnostic_stable": None, "latch_after_diagnostic": None,
                               "final_stable": None, "latch_after_group": None}
            dp_groups.append(active_dp_group)
        elif tag == 22:
            require(active_dp_group is not None and active_dp_group["diagnostic_stable"] is None,
                    "Unmatched or repeated DP diagnostic return")
            require(active_diagnostic is not None
                    and active_diagnostic.get("dp_group_ordinal") == active_dp_group["ordinal"],
                    "DP group lacks its tensor diagnostic")
            require(all(value in (0, 1) for value in signed), "Invalid DP diagnostic stable/latch flags")
            active_dp_group["diagnostic_stable"], active_dp_group["latch_after_diagnostic"] = signed
        elif tag == 21:
            require(active_dp_group is not None and active_dp_group["diagnostic_stable"] is not None,
                    "Unmatched DP loop-group completion")
            require(all(value in (0, 1) for value in signed), "Invalid DP final stable/latch flags")
            active_dp_group["final_stable"], active_dp_group["latch_after_group"] = signed
            active_dp_group["contraction_rejected"] = (
                active_dp_group["diagnostic_stable"] == 1 and active_dp_group["final_stable"] == 0)
            active_dp_group = None
        elif tag == 30:
            active_diagnostic = {"precision": signed[0], "ctmode": signed[1],
                                 "n": signed[2], "rank": signed[3],
                                 "last_stage": "rank_guard", "stable_reached": False,
                                 "native_status": None, "bridge_status": None}
            if signed[0] == 53:
                require(active_dp_group is not None
                        and signed[1:] == tuple(active_dp_group[key] for key in ("ctmode", "n", "rank")),
                        "DP tensor diagnostic does not match its loop group")
                require(not diagnostics or diagnostics[-1].get("dp_group_ordinal") != active_dp_group["ordinal"],
                        "Duplicate tensor diagnostic in DP group")
                active_diagnostic["dp_group_ordinal"] = active_dp_group["ordinal"]
            else:
                require(active_dp_group is None, "QP diagnostic nested in DP loop group")
            diagnostics.append(active_diagnostic)
        elif tag in (31, 32, 33, 34):
            require(active_diagnostic is not None, "Fortran progress lacks diagnostic start")
            if tag == 31:
                stages = {2: "support_guard", 3: "initialization_call", 6: "momentum_conservation_guard",
                          7: "input_construction", 8: "native_evaluation", 9: "output_finiteness_guard"}
                require(signed[0] in stages, "Unknown Fortran diagnostic stage")
                active_diagnostic["last_stage"] = stages[signed[0]]
            elif tag == 34:
                require(signed[0] in (4, 5), "Unknown bridge guard")
                active_diagnostic["last_stage"] = "initialization_status" if signed[0] == 4 else "cache_budget_status"
                active_diagnostic["bridge_status"] = signed[1]
            elif tag == 32:
                active_diagnostic["native_status"] = signed[0]
                active_diagnostic["last_stage"] = "native_status"
            else:
                require(signed[0] == active_diagnostic["precision"], "Wrong Fortran stable precision")
                active_diagnostic["stable_reached"] = True
        elif tag == 40:
            require(active_dp_group is None, "Unfinished DP group before aggregate stability check")
            require(index + 5 < len(host) and [row["tag"] for row in host[index:index + 6]]
                    == [40, 41, 42, 43, 44, 49], "Incomplete host DP stability block")
            backend, nevals, nso, before_latch = signed
            require(backend == 9 and 1 <= nevals <= 682 and nso == 1 and before_latch in (0, 1),
                    "Unexpected host DP stability dimensions/backend")
            pre = host[index + 1]["words"]
            results = host[index + 2]["words"]
            post_backend, after_latch = map(signed_word, host[index + 3]["words"])
            post = host[index + 4]["words"]
            threshold_word = host[index + 5]["words"][0]
            require(len(results) == 3 * (nso + 1) * nevals and post_backend == backend
                    and after_latch in (0, 1), "Host DP stability payload mismatch")
            threshold = word_value(threshold_word)
            require(math.isfinite(threshold) and threshold > 0, "Invalid recorded host stability threshold")
            pre_values, post_values = list(map(word_value, pre)), list(map(word_value, post))
            dp_checks.append({"backend": backend, "evaluations": nevals,
                              "latch_before_override": before_latch, "latch_after_override": after_latch,
                              "pre_accuracy_words": hex_words(pre), "post_accuracy_words": hex_words(post),
                              "dp_result_words": hex_words(results), "threshold_word": f"{threshold_word:016x}",
                              "pre_accuracy_nonfinite": any(not math.isfinite(value) for value in pre_values),
                              "finite_pre_accuracy_exceeds_threshold": any(math.isfinite(value) and value >= threshold for value in pre_values),
                              "finite_post_accuracy_exceeds_threshold": any(math.isfinite(value) and value >= threshold for value in post_values)})
        elif tag == 45:
            advanced, rejected, next_slot = signed
            require(2 <= advanced <= 10 and rejected == 9 and 0 <= next_slot <= 9,
                    "Unexpected host QP escalation branch metadata")
            branches.append({"advanced_slot": advanced, "rejected_backend": rejected, "next_slot_backend": next_slot})
        elif tag == 46:
            require(active_dp_group is None, "Unfinished DP group before QP stability check")
            require(index + 1 < len(host) and host[index + 1]["tag"] == 47,
                    "Incomplete host QP accuracy block")
    require(active_dp_group is None, "DP loop group truncated at public return")
    for index, event in enumerate(host):
        if event["tag"] in (41, 42, 43, 44, 49):
            predecessor = {41: 40, 42: 41, 43: 42, 44: 43, 49: 44}[event["tag"]]
            require(index > 0 and host[index - 1]["tag"] == predecessor, "Orphan host DP stability event")
        if event["tag"] == 47:
            require(index > 0 and host[index - 1]["tag"] == 46, "Orphan QP accuracy event")
    failures = [{"native_call": item["id"], "precision": item["begin"]["input_bits"],
                 "check": item["begin"]["check"], "n": item["begin"]["n"], "rank": item["begin"]["rank"],
                 "status": item["end"]["status"], "error": item["end"]["error"],
                 "host_diagnostic_ordinal": item["host_diagnostic"]["ordinal"]}
                for item in native if item["end"]["status"] != 0]
    guarded_returns = [item for item in diagnostics if not item["stable_reached"]]
    observations = []
    if any(item["precision"] == 53 for item in failures):
        observations.append("native_dp_status_failure")
    if any(item["precision"] == 53 for item in guarded_returns):
        observations.append("fortran_dp_guard_return")
    if any(item["final_stable"] == 0 for item in dp_groups):
        observations.append("host_dp_group_final_unstable")
    if any(item["contraction_rejected"] for item in dp_groups):
        observations.append("host_dp_contraction_rejection_after_stable_diagnostic")
    if any(item["latch_after_override"] for item in dp_checks):
        observations.append("host_dp_reduction_rejection_latch_set")
    if any(item["pre_accuracy_nonfinite"] for item in dp_checks):
        observations.append("host_dp_pre_override_accuracy_nonfinite")
    if any(item["finite_pre_accuracy_exceeds_threshold"] for item in dp_checks):
        observations.append("host_dp_pre_override_accuracy_exceeds_threshold")
    if any(item["finite_post_accuracy_exceeds_threshold"] for item in dp_checks):
        observations.append("host_dp_post_override_accuracy_exceeds_threshold")
    if branches:
        observations.append("host_qp_escalation_branch_taken")
    counts = Counter((item["begin"]["input_bits"], item["begin"]["check"], item["end"]["status"]) for item in native)
    route_counts = Counter(detail["route"] for item in native for detail in item["details"] if detail["kind"] == "lane_route")
    return {"point": call["point"], "phase": call["phase"], "status": call["status"],
            "observations": observations, "native_failures": failures, "fortran_guard_returns": guarded_returns,
            "host_dp_checks": dp_checks, "host_dp_groups": dp_groups, "qp_escalation_branches": branches,
            "native_counts": [{"input_bits": key[0], "check": key[1], "status": key[2], "count": count}
                              for key, count in sorted(counts.items())],
            "native_lane_route_counts": dict(sorted(route_counts.items())),
            "qp_cache_hits": sum(detail["kind"] == "qp_cache_hit" for item in native for detail in item["details"]),
            "native_precision_purity_proven": False,
            "diagnostic_note": "Rejection latch is a host software flag; its name alone does not prove an IEEE exception."}


def validate_public_triplet(case_id, original, disabled, enabled):
    require(case_id in protocol.CASES, "Unknown case")
    arms = {"original": original, "disabled": disabled, "enabled": enabled}
    reference_path = (protocol.CAMPAIGN / "experiments/numerical"
                      / f"full100-warmup-{case_id}-1/candidate-9/raw")
    observation_path = reference_path / "observation.json"
    protocol.file_pin(observation_path, protocol.CASES[case_id]["observation_sha256"])
    observation = protocol.read_json(observation_path)
    historical_warmup = retained_bytes(reference_path / "warmup.dat", MAX_PUBLIC_BYTES)
    require(historical_warmup.decode("ascii") == observation["raw_outputs"]["warmup.dat"],
            "Historical raw warmup no longer matches its pinned observation")
    historical_rows = protocol.warmup_records(historical_warmup.decode("ascii"))
    require(len(historical_rows) == 202, "Historical bootstrap/full100 stream changed")
    historical_codes = tuple(row["status"] for row in historical_rows[::2])
    for name, arm in arms.items():
        require(arm["warmup_bytes"] == historical_warmup,
                name + " warmup bytes differ from the historical full stream")
        require(tuple(call["status"] for call in arm["calls"]) == historical_codes,
                name + " raw-word log statuses differ from historical warmup")
    reference = original["calls"]
    # All fields, including IEEE flags on entry and after each complete call,
    # participate. Equal statuses or printed coefficients alone do not pass.
    for name, arm in (("disabled", disabled), ("enabled", enabled)):
        for index, (before, after) in enumerate(zip(reference, arm["calls"])):
            differences = [field for field in before if before[field] != after[field]]
            require(not differences,
                    f"{name} public parity failed at call {index}: {','.join(differences)}")
    rescues = [call["point"] for call in reference[1:] if call["status"] == 323]
    require(tuple(rescues) == protocol.CASES[case_id]["rescued_indices"], "Rescue membership drift")
    return {"schema": "dp-rescue-public-parity-v1", "case": case_id,
            "public_parity_passed": True, "full101_call_order_passed": True,
            "historical_warmup_bytes_exact": True, "all_input_answer_accuracy_words_exact": True,
            "all_entry_exit_ieee_states_exact": True,
            "status_counts": dict(sorted(Counter(str(call["status"]) for call in reference[1:]).items())),
            "rescued_indices": rescues, "rescued_ids": [f"ordinary-{i-1:04d}" for i in rescues],
            "artifacts": {name: {"directory": arm["directory"], "sha256": arm["hashes"]}
                          for name, arm in arms.items()},
            "native_journal_validated": False, "runtime_provenance_validated": False,
            "cause_attribution_qualified": False, "performance_eligible": False,
            "limitations": ["Public parity alone does not prove the loaded native runtime.",
                            "Native event parsing and cause attribution require the frozen journal schema.",
                            "No native precision purity claim follows from the public host status."]}


def analyze_triplet(case_id, original, disabled, enabled):
    result = validate_public_triplet(case_id, original, disabled, enabled)
    disabled_native = parse_native(Path(disabled["directory"]) / "cause-native.jsonl", disabled["calls"], False)
    enabled_native = parse_native(Path(enabled["directory"]) / "cause-native.jsonl", enabled["calls"], True)
    result["schema"] = "dp-rescue-cause-retained-text-v1"
    result["native_journal_validated"] = True
    result["native_journals"] = {
        name: {key: journal[key] for key in ("sha256", "enabled", "event_count", "native_call_count", "flush_count")}
        for name, journal in (("disabled", disabled_native), ("enabled", enabled_native))}
    result["multi_cause_observations"] = enabled_native["points"]
    result["native_precision_purity_proven"] = False
    result["limitations"] = [
        "Loaded runtimes, executable identities, launch/exit evidence, and protected-file preservation remain unvalidated.",
        "Cause observations are provisional until the external runtime/provenance controller passes.",
        "Each point can have several contributing observations; no single-cause label is forced.",
        "The host rejection latch alone does not prove an IEEE exception.",
        "Native master-row counters aggregate precisions; a Direct lane is not a pure-DP certificate.",
        "Threshold comparisons decode retained binary64 words offline and never affect the numerical prototype.",
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=tuple(protocol.CASES))
    for arm in ("original", "disabled", "enabled"):
        parser.add_argument("--" + arm, required=True, type=Path, help="Existing private run directory")
    args = parser.parse_args()
    result = analyze_triplet(args.case, read_arm(args.original),
                             read_arm(args.disabled), read_arm(args.enabled))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
