"""Pure reconstruction of the two unchanged EW rank-three replay inputs.

The six callbacks provide retained packet bytes/documents. This module does
not read the filesystem, execute retained helpers/controllers, load native
code, or inspect tensor coefficient buffers. The enclosing verifier checks
the sibling packet seals and the replay's execution, status and map records.
"""

import hashlib
import json
import re


SOURCE_SEAL = "9384471349f462e7f028161c5b9a1673027698d930e82ac7da0aa6910c08ad61"
RUN_SEAL = "ec0dcaf70e5921fca8b32041bfde688430cf07c347b5ed31275ecb5845f1ce04"
DERIVED_SEAL = "25c2114756ed3eb968c311945a27d175c5c5a0af54d4ae21158abc8ed42a4184"
OLD_EXECUTOR = ".tirust-build/kinematic-provenance-current-executor-1/"
PROPOSAL = ".tirust-build/ew-refusal-replay-proposal-1/"
HEADER = ("point", "serial", "id", "mode", "rank", "n", "i", "j", "branch", "split_order")
IDENTITY = ("section", "point", "serial", "id", "mode", "rank", "n", "split_order")
WORD = re.compile(r"[0-9A-F]{16}\Z")
PLANS = [[1, 0], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2],
         [3, 3], [4, 1], [4, 2], [4, 3], [5, 2], [5, 3]]


def require(condition, message):
    if not condition:
        raise ValueError("EW replay inputs: " + message)


def pin(raw):
    require(isinstance(raw, bytes), "payload callback did not return bytes")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def checked(data, name, expected):
    raw = data(name)
    actual = pin(raw)
    if isinstance(expected, str):
        require(actual["sha256"] == expected, "hash differs: " + name)
    else:
        require(actual == {key: expected[key] for key in ("bytes", "sha256")},
                "raw identity differs: " + name)
    return raw


def checked_document(data, document, name, expected=None):
    raw = data(name) if expected is None else checked(data, name, expected)
    parsed = json.loads(raw)
    require(parsed == document(name), "document/byte callback mismatch: " + name)
    return parsed


def relative(name, prefix):
    require(name.startswith(prefix), "path is outside the declared source: " + name)
    value = name[len(prefix):]
    require(value and all(part not in ("", ".", "..") for part in value.split("/")),
            "unsafe retained path")
    return value


def finite_words(words, width, count):
    """Inspect binary64/128 exponents using low-word-first integer arithmetic."""
    require(width in (1, 2) and len(words) == width * count
            and all(WORD.fullmatch(word) for word in words), "input word shape differs")
    fraction_bits, exponent_bits = (52, 11) if width == 1 else (112, 15)
    for start in range(0, len(words), width):
        bits = sum(int(words[start + part], 16) << (64 * part) for part in range(width))
        exponent = (bits >> fraction_bits) & ((1 << exponent_bits) - 1)
        require(exponent != (1 << exponent_bits) - 1, "nonfinite captured native input")


def trace_rows(raw, mode):
    rows, section = [], -1
    for number, line in enumerate(raw.decode("ascii").splitlines(), 1):
        fields = line.split()
        require(bool(fields), "empty trace line")
        if fields[0] == "KT_SCHEMA_V2":
            require(len(fields) == 6 and list(map(int, fields[1:])) == [2, 64, 128, 64, 32],
                    "trace schema/point differs")
            section += 1
            continue
        require(section >= 0 and fields[0].startswith("KT_") and len(fields) >= 11,
                "invalid trace record header")
        row = dict(zip(HEADER, map(int, fields[1:11])))
        row.update(tag=fields[0], line=number, section=section, words=fields[11:])
        if row["id"] == 5:
            rows.append(row)
    keys = {tuple(row[key] for key in IDENTITY) for row in rows}
    require(rows and len(keys) == 1, "group 5 input call is not unique")
    require(all(row["point"] == 2 and row["mode"] == mode and row["n"] == 4
                and row["rank"] == 3 and row["serial"] > 0 and row["split_order"] > 0
                for row in rows), "captured EW request scope differs")
    return rows, dict(zip(IDENTITY, next(iter(keys))))


def one(rows, tag):
    selected = [row for row in rows if row["tag"] == tag]
    require(len(selected) == 1, "input boundary is absent/ambiguous: " + tag)
    return selected[0]


def verify_inputs(data, document, runtime_data, runtime_document, build_data, build_document):
    """Reconstruct both word files and the full certificate from sealed sources.

    Main logical paths start with ``proposal/`` or ``results/``. Sibling
    callbacks use their existing payload names. Only the source/word identity
    of the closed replay is joined here; its runtime protocol and outcomes
    remain the enclosing verifier's responsibility.
    """
    source = checked_document(data, document, "proposal/index.json", SOURCE_SEAL)
    require(source["file_count"] == len(source["files"]) == 17,
            "proposal file inventory differs")
    total = 0
    for name, expected in source["files"].items():
        total += len(checked(data, "proposal/" + relative(PROPOSAL + name, PROPOSAL), expected))
    require(total == source["total_bytes"] == 101649, "proposal byte count differs")
    bindings = checked_document(data, document, "proposal/bindings.json")
    certificate = checked_document(data, document, "proposal/input-certificate.json")
    closure = checked_document(data, document, "results/run-terminal-closure.json", RUN_SEAL)
    require(closure["source_index_sha256"] == SOURCE_SEAL, "closed replay binds another source")

    derived = checked_document(build_data, build_document, "results/derived/index.json", DERIVED_SEAL)
    require(bindings["indices"][OLD_EXECUTOR + "results/derived/index.json"] == DERIVED_SEAL,
            "proposal binds another derived source")
    source_pins = {}
    for name in ("CT_interface.f", "TIRUST_interface.f", "derivation.json", "routing.json"):
        key = "ew/" + name
        source_pins[key] = pin(checked(build_data, "results/derived/" + key, derived["files"][key]))
    derivation = checked_document(build_data, build_document, "results/derived/ew/derivation.json")
    identities = [item for item in derivation["target_identities"] if item["reduction_identifier"] == 5]
    require(len(identities) == 1, "derived group 5 source is absent/ambiguous")
    identity = identities[0]
    require(identity["rank"] == 3 and identity["identity_kind"] == "group",
            "derived source is not the rank-three group")
    tir_name = "results/derived/ew/TIRUST_interface.f"
    tir = build_data(tir_name)
    plans = [list(map(int, pair)) for pair in
             re.findall(r"CALL TIRUST_REGISTER_PLAN\((\d+),(\d+),STATUS\)", tir.decode("ascii"))]
    require(plans == PLANS == bindings["scope"]["plan_registration_sequence"],
            "thirteen generated plan registrations/order differ")
    require(bindings["scope"]["n"] == 4 and bindings["scope"]["rank"] == 3
            and bindings["scope"]["tensor_component_capacity"] == (4 * 5 * 6 * 7) // 24 == 35,
            "rank-three cumulative component capacity differs")
    driver = data("proposal/replay.f90").decode("ascii")
    require(re.search(r"integer,\s*parameter\s*::\s*n=4,\s*rank=3,\s*count=35", driver),
            "Fortran replay capacity/header constants differ")
    plan_arrays = []
    for name in ("plan_n", "plan_rank"):
        matches = re.findall(name + r"\(nplans\)=\[([0-9,]+)\]", driver)
        require(len(matches) == 1, "Fortran plan array is absent/ambiguous")
        plan_arrays.append(list(map(int, matches[0].split(","))))
    require([list(pair) for pair in zip(*plan_arrays)] == PLANS
            and all(len(array) == 13 for array in plan_arrays), "Fortran plan order differs")

    old_closures = {}
    for kind, bound in bindings["closures"].items():
        old_closures[kind] = checked_document(runtime_data, runtime_document,
                                             relative(bound["path"], OLD_EXECUTOR), bound)
    require(set(old_closures) == {"ordinary", "stress"}, "bound prior closure scope differs")
    require(list(bindings["traces"]) == ["ew-point2-mode1-enabled", "ew-point2-mode4-enabled"],
            "exact two EW trace identities/order differ")
    requests, certificates, proof = [], [], []
    for label, bound in bindings["traces"].items():
        mode = bound["mode"]
        require(bound["point"] == 2 and mode in (1, 4) and bound["closure"] == "stress",
                "original trace is outside the requested EW case")
        require(old_closures["stress"]["workers"][label]["trace_pin"] ==
                {key: bound[key] for key in ("bytes", "sha256")}, "trace is not in its closed worker")
        raw = checked(runtime_data, relative(bound["path"], OLD_EXECUTOR), bound)
        rows, call_identity = trace_rows(raw, mode)
        suffix, width = ("DP", 1) if mode == 1 else ("QP", 2)
        native = one(rows, "KT_TIRUST_" + suffix)
        matrix = one(rows, "KT_MATRIX_" + suffix)
        status = one(rows, "KT_STATUS_" + suffix)
        words = native["words"]
        finite_words(words, width, 41)
        finite_words(matrix["words"], width, 24)
        require(status["words"] == ["1"], "original captured refusal status differs")
        for corner in range(4):
            for component in range(4):
                a, b = (4 * corner + component) * width, (4 * component + corner) * width
                require(words[a:a + width] == matrix["words"][b:b + width],
                        "component/corner transpose changes shift words")
        name = label.replace("-enabled", "-group5-stock")
        input_name = "inputs/" + name + ".words"
        serialized = ("KPROJ_WORDS_V1\n" + f"2 {mode} 4 3 {len(words)}\n"
                      + "\n".join(words) + "\n").encode("ascii")
        require(data("proposal/" + input_name) == serialized,
                "word file differs from the unchanged native trace: " + name)
        request = {"id": name, "variant": "stock", "source_trace": label, "point": 2,
                   "mode": mode, "input": {"path": PROPOSAL + input_name, **pin(serialized)}}
        requests.append(request)
        certificates.append({"trace": bound, "identity": call_identity,
                             "source_lines": {"native": native["line"], "matrix": matrix["line"],
                                              "status": status["line"]},
                             "original_native_status": 1, "native_words": words, "intervention": False})
        worker = closure["workers"][name]
        require(worker["input_pin"] == pin(serialized), "closed worker used another word input")
        require(all(worker["response"][key] == request[key] for key in ("id", "point", "mode", "variant")),
                "closed response/input identity differs")
        proof.append({"id": name, "rank": 3, "word_count": len(words),
                      "original_native_status": 1, "input_sha256": pin(serialized)["sha256"]})
    require(len(requests) == 2 and [item["mode"] for item in requests] == [1, 4]
            and set(closure["workers"]) == {item["id"] for item in requests},
            "closed replay/source input inventories differ")
    rebuilt = {"schema": "private-exact-ew-original-refusal-certificate-v1", "source_only": True,
               "native_executions": 0, "failed_tensor_outputs_read": False, "input_trace_only": True,
               "source_pins": source_pins, "source_identity": identity,
               "plan_registration_source": {"path": OLD_EXECUTOR + tir_name, **pin(tir)},
               "plan_registration_sequence": plans, "tensor_component_count": 35,
               "requests": requests, "certificates": certificates, "input_interventions": False,
               "production_fixture_or_policy_change": False, "success_status_is_not_accuracy_proof": True}
    require(certificate == rebuilt, "complete reconstructed EW certificate differs")
    require(data("proposal/input-certificate.json") ==
            (json.dumps(rebuilt, sort_keys=True, indent=2) + "\n").encode("ascii"),
            "canonical certificate bytes differ")
    return {"source_files_rehashed": 17, "trace_inputs_reconstructed": 2,
            "complete_certificates_reconstructed": 2, "word_inputs_reconstructed": 2,
            "plans_verified_in_generated_order": 13, "tensor_component_capacity": 35,
            "inputs": proof, "input_interventions": False, "tensor_outputs_read": False,
            "tensor_accuracy_qualified": False, "new_native_calls": 0}
