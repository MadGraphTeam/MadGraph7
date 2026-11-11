"""Pure, exact checks of the ten closed kinematic-projection replay inputs.

Only the six supplied byte/document callbacks access retained packet payloads.
There is no filesystem access, retained-controller execution, native loading,
floating-point input conversion, or tensor-output inspection in this module.
The caller verifies the declared sibling packet seals before invoking this
helper. Historical TIRust source provenance is checked as recorded metadata;
its external reference file is deliberately not reopened.
"""

from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import re


SOURCE_SEAL = "ba130c540e466550ad2e4fa560b0b8df75a93912c4d1b271ef4d60c5fd4571a9"
RUN_SEAL = "3a8e13486d4cd26425efefaceb7d6f2099be0a867a80e22fdb00fc3ff1e678d8"
DERIVED_SEAL = "25c2114756ed3eb968c311945a27d175c5c5a0af54d4ae21158abc8ed42a4184"
WHOLE_SEAL = "df69bf17225f942b8939558843da57b7156d652914890fa931b48af24f2222d3"
SUBSET_SEAL = "a18d54cc7b40c0a263bcf1a4ea61c88ce6f94cf3caf818203c6f5d0e382c40f8"
OLD_EXECUTOR = ".tirust-build/kinematic-provenance-current-executor-1/"
PROPOSAL = ".tirust-build/kinematic-projection-replay-proposal-1/"
HEADER = ("point", "serial", "id", "mode", "rank", "n", "i", "j", "branch", "split_order")
IDENTITY = ("section", "point", "serial", "id", "mode", "rank", "n", "split_order")
WORD = re.compile(r"[0-9A-F]{16}\Z")


def require(condition, message):
    if not condition:
        raise ValueError("Replay math: " + message)


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


def power2(exponent):
    return Fraction(2 ** exponent) if exponent >= 0 else Fraction(1, 2 ** (-exponent))


def ieee(words):
    """Decode GNU x86_64 binary64/binary128 words, with low 64-bit word first."""
    require(len(words) in (1, 2) and all(WORD.fullmatch(word) for word in words),
            "invalid IEEE word group")
    bits = 64 * len(words)
    fraction_bits, exponent_bits, bias = (52, 11, 1023) if bits == 64 else (112, 15, 16383)
    integer = sum(int(word, 16) << (64 * index) for index, word in enumerate(words))
    sign = integer >> (bits - 1)
    exponent = (integer >> fraction_bits) & ((1 << exponent_bits) - 1)
    fraction = integer & ((1 << fraction_bits) - 1)
    require(exponent != (1 << exponent_bits) - 1, "nonfinite replay input")
    significand = fraction + ((1 << fraction_bits) if exponent else 0)
    exponent2 = (exponent or 1) - bias - fraction_bits
    value = (-1 if sign else 1) * significand * power2(exponent2)
    return {"words": list(words), "bits": bits, "sign": sign,
            "kind": "finite" if significand else "zero",
            "significand": str(significand), "exponent2": exponent2, "value": value}


def wire(value):
    if isinstance(value, Fraction):
        with localcontext() as context:
            context.prec = 36
            approximation = str(Decimal(value.numerator) / Decimal(value.denominator))
        return {"numerator_hex": hex(value.numerator),
                "denominator_hex": hex(value.denominator),
                "decimal_approximation_36_digits": approximation}
    if isinstance(value, dict):
        return {key: wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(item) for item in value]
    return value


def negative_bounds(value):
    require(value < 0, "RAW/shift square is not strictly spacelike")
    magnitude = -value
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < power2(exponent):
        exponent -= 1
    lower, upper = -power2(exponent + 1), -power2(exponent)
    require(lower < value <= upper < 0, "invalid exact negative enclosure")
    return {"strict_lower": lower, "inclusive_upper": upper, "base2_exponent": exponent}


def trace_rows(raw, bound):
    """Parse headers independently; decode only the four required input/status tags."""
    rows, section = [], -1
    for line_number, line in enumerate(raw.decode("ascii").splitlines(), 1):
        fields = line.split()
        require(bool(fields), "empty trace line")
        if fields[0] == "KT_SCHEMA_V2":
            require(len(fields) == 6 and list(map(int, fields[1:])) ==
                    [bound["point"], 64, 128, 64, 32], "trace schema/point differs")
            section += 1
            continue
        require(section >= 0 and fields[0].startswith("KT_") and len(fields) >= 11,
                "invalid trace header")
        row = dict(zip(HEADER, map(int, fields[1:11])))
        row.update(tag=fields[0], line=line_number, section=section, payload=fields[11:])
        if row["id"] == 3:
            rows.append(row)
    identities = {tuple(row[key] for key in IDENTITY) for row in rows}
    require(rows and len(identities) == 1, "group 3 input call is not unique")
    require(all(row["point"] == bound["point"] and row["mode"] == bound["mode"]
                and row["rank"] == 2 and row["n"] == 4 and row["serial"] > 0
                and row["split_order"] > 0 for row in rows), "group 3 scope differs")
    return rows, dict(zip(IDENTITY, next(iter(identities))))


def one(rows, tag, pair=None):
    matches = [row for row in rows if row["tag"] == tag
               and (pair is None or (row["i"], row["j"]) == pair)]
    require(len(matches) == 1, "required input boundary is absent/ambiguous: " + tag)
    return matches[0]


def payload(row, width, count):
    words = row["payload"]
    require(len(words) == width * count, "input payload shape differs: " + row["tag"])
    return [ieee(words[index:index + width]) for index in range(0, len(words), width)]


def determinant3(matrix):
    a, b, c = matrix
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def verify_math(data, document, runtime_data, runtime_document, build_data, build_document):
    """Rebuild source-word certificates and both exact Gram analyses from bytes.

    Paths passed to the main callbacks start with ``proposal/`` or ``results/``;
    sibling callbacks use those capsules' existing logical payload names.
    This function checks retained historical evidence, not external artifacts or
    tensor accuracy. The enclosing verifier owns runtime protocol/map checks.
    """
    source = checked_document(data, document, "proposal/index.json", SOURCE_SEAL)
    require(source["file_count"] == len(source["files"]) == 26,
            "proposal file inventory differs")
    total = 0
    for name, expected in source["files"].items():
        safe_name = relative(PROPOSAL + name, PROPOSAL)
        total += len(checked(data, "proposal/" + safe_name, expected))
    require(total == source["total_bytes"] == 140471, "proposal byte count differs")
    # These exact source hashes bind the original Fortran/worker protocol too;
    # no controller or driver is imported, interpreted, compiled, or launched.
    bindings = checked_document(data, document, "proposal/bindings.json")
    certificate = checked_document(data, document, "proposal/input-certificate.json")
    closure = checked_document(data, document, "results/run-terminal-closure.json", RUN_SEAL)
    require(closure["source_index_sha256"] == SOURCE_SEAL,
            "closed native run has another proposal")

    derived = checked_document(build_data, build_document, "results/derived/index.json", DERIVED_SEAL)
    require(bindings["indices"][OLD_EXECUTOR + "results/derived/index.json"] == DERIVED_SEAL,
            "proposal binds another derived source")
    source_pins = {}
    for name in ("CT_interface.f", "TIRUST_interface.f", "derivation.json", "routing.json"):
        key = "cms/" + name
        source_pins[key] = pin(checked(build_data, "results/derived/" + key, derived["files"][key]))
    derivation = checked_document(build_data, build_document, "results/derived/cms/derivation.json")
    identities = [item for item in derivation["target_identities"] if item["reduction_identifier"] == 3]
    require(len(identities) == 1, "derived group 3 identity is absent/ambiguous")
    source_identity = identities[0]
    require(source_identity["identity_kind"] == "group" and source_identity["rank"] == 2,
            "derived target is not the recorded rank-2 group")

    old_closures = {}
    for kind, bound in bindings["closures"].items():
        old_closures[kind] = checked_document(runtime_data, runtime_document,
                                             relative(bound["path"], OLD_EXECUTOR), bound)
    require(set(old_closures) == {"ordinary", "stress"}, "prior closure scope differs")
    require([(bound["point"], bound["mode"]) for bound in bindings["traces"].values()] ==
            [(1, 1), (1, 4), (20, 1), (20, 4), (21, 1), (21, 4)], "trace scope/order differs")
    requests, certificates, native_inputs = [], [], {}

    def save_request(label, bound, words, variant):
        identifier = label.replace("-enabled", "-group3-" + variant)
        name = "inputs/" + identifier + ".words"
        raw = ("KPROJ_WORDS_V1\n" + f"{bound['point']} {bound['mode']} 4 2 {len(words)}\n"
               + "\n".join(words) + "\n").encode("ascii")
        require(data("proposal/" + name) == raw, "retained word file differs from trace: " + identifier)
        request = {"id": identifier, "variant": variant, "source_trace": label,
                   "point": bound["point"], "mode": bound["mode"],
                   "input": {"path": PROPOSAL + name, **pin(raw)}}
        require(identifier not in native_inputs, "duplicate input id")
        requests.append(request)
        native_inputs[identifier] = (list(words), request)

    for label, bound in bindings["traces"].items():
        require(old_closures[bound["closure"]]["workers"][label]["trace_pin"] ==
                {key: bound[key] for key in ("bytes", "sha256")}, "trace is not in its closed worker")
        trace = checked(runtime_data, relative(bound["path"], OLD_EXECUTOR), bound)
        rows, identity = trace_rows(trace, bound)
        width = {1: 1, 4: 2}[bound["mode"]]
        suffix = "DP" if bound["mode"] == 1 else "QP"
        native = one(rows, "KT_TIRUST_" + suffix)
        raw_pair = one(rows, "KT_RAW_" + suffix, (2, 4))
        matrix = one(rows, "KT_MATRIX_" + suffix)
        status = one(rows, "KT_STATUS_" + suffix)
        values = payload(native, width, 41)
        raw_values = payload(raw_pair, width, 4)
        matrix_values = payload(matrix, width, 24)
        real, imaginary = raw_values[:2]
        require(imaginary["value"] == 0, "captured RAW is complex")
        for corner in range(4):
            for component in range(4):
                require(values[4 * corner + component]["words"] ==
                        matrix_values[4 * component + corner]["words"],
                        "native/matrix shift transpose changes bits")
        differences = [values[4 + component]["value"] - values[12 + component]["value"]
                       for component in range(4)]
        square = differences[0] ** 2 - sum(value ** 2 for value in differences[1:])
        require(values[37]["words"] == values[31]["words"], "critical invariant pair is asymmetric")
        shifts = source_identity["modes"][str(bound["mode"])]["shifts"]
        conservation = source_identity["conservation_coefficients"]
        require(shifts[1] == [1, 0, 0, 0] and shifts[3] == [1, 1, 0, -1]
                and shifts[0] == conservation == [1, 1, -1, -1], "signed source routing differs")
        require([left - right for left, right in zip(shifts[1], shifts[3])] ==
                [coefficient - residual for coefficient, residual in zip([1, 0, -1, 0], conservation)],
                "p1-p3-r source identity is not exact")
        words = list(native["payload"])
        expected_status = 0 if bound["point"] == 1 else 1
        require(status["payload"] == [str(expected_status)], "original native status differs")
        save_request(label, bound, words, "stock")
        item = {"trace": bound, "identity": identity,
                "source_lines": {"native": native["line"], "raw_2_4": raw_pair["line"],
                                 "matrix": matrix["line"], "status": status["line"]},
                "original_native_status": expected_status, "raw_real": real,
                "raw_imaginary": imaginary, "negative_raw_bounds": negative_bounds(real["value"]),
                "critical_native_words": values[37]["words"],
                "exact_captured_shift_pair_square": square,
                "negative_shift_square_bounds": negative_bounds(square),
                "raw_minus_exact_shift_square": real["value"] - square,
                "symbolic_pair": "p1-p3-r",
                "first_shift_preserved_words": [value["words"] for value in values[:4]],
                "source_conservation_coefficients": conservation,
                "stock_word_sha256": hashlib.sha256(" ".join(words).encode("ascii")).hexdigest(),
                "intervention": None}
        if bound["point"] != 1:
            require(values[37]["value"] == values[31]["value"] == 0,
                    "stress stock critical pair is not zero")
            changed = list(words)
            offsets = (31 * width, 37 * width)
            for offset in offsets:
                changed[offset:offset + width] = real["words"]
                require(changed[offset:offset + width] != words[offset:offset + width],
                        "intervention must change both selected real entries")
            allowed = {offset + part for offset in offsets for part in range(width)}
            actual_changes = [index for index, pair in enumerate(zip(words, changed)) if pair[0] != pair[1]]
            require(set(actual_changes) <= allowed, "intervention changed another input word")
            item["intervention"] = {"matrix_pair": [2, 4], "transpose_pair": [4, 2],
                                    "allowed_word_positions_one_based": sorted(index + 1 for index in allowed),
                                    "changed_word_positions_one_based": [index + 1 for index in actual_changes]}
            save_request(label, bound, changed, "raw24")
        certificates.append(item)

    require(len(native_inputs) == 10 and len(certificates) == 6, "input reconstruction count differs")
    rebuilt_certificate = {
        "schema": "private-exact-kinematic-projection-replay-certificate-v1", "source_only": True,
        "native_executions": 0, "failed_tensor_outputs_read": False, "input_trace_only": True,
        "source_pins": source_pins, "source_identity": source_identity, "requests": requests,
        "certificates": certificates, "intervention_is_diagnostic_only": True,
        "production_fixture_or_policy_change": False, "success_status_is_not_accuracy_proof": True}
    require(certificate == wire(rebuilt_certificate), "complete reconstructed certificate differs")
    require(data("proposal/input-certificate.json") ==
            (json.dumps(wire(rebuilt_certificate), sort_keys=True, indent=2) + "\n").encode("ascii"),
            "certificate canonical bytes differ")

    whole = checked_document(data, document, "results/root-exact-gram-analysis.json", WHOLE_SEAL)
    subset = checked_document(data, document, "results/root-error-subset-gram-analysis.json", SUBSET_SEAL)
    require(whole["schema"] == "root-exact-causal-input-gram-analysis-v1"
            and whole["input_source_index_sha256"] == SOURCE_SEAL
            and whole["native_run_closure_sha256"] == RUN_SEAL
            and whole["determinant_units"] == "GeV^6", "whole Gram analysis source/convention differs")
    require(subset["schema"] == "root-exact-error-subset-gram-analysis-v1"
            and subset["source_index_sha256"] == SOURCE_SEAL
            and subset["native_closure_sha256"] == RUN_SEAL
            and subset["determinant_units"] == "GeV^4", "subset analysis source/convention differs")
    require(subset["reference_revision"] == "854b79dcb7fffcb35e92b4a8a3dd443369b798c4"
            and subset["reference_file"] == "crates/iregi-rust/src/direct/exec.rs"
            and subset["reference_file_sha256"] == "4053f5d3a24002829328f75bac5aae3d26f04764f841a8edbaaa0ef938836c81"
            and subset["source_regions"] == [
                "supplied scalar_products loaded directly as distances",
                "geometry SubsetKind::Pv origin=members[0]; matrix=d(origin,mi)+d(origin,mj)-d(mi,mj)"],
            "recorded source provenance/formula differs")
    require(whole["independent_tensor_accuracy_qualified"] is False and whole["tensor_outputs_read"] is False
            and whole["new_native_calls"] == 0 and subset["tensor_accuracy_qualified"] is False
            and subset["new_native_calls"] == 0, "input analysis overstates its evidence")
    require(set(closure["workers"]) == set(native_inputs), "closed worker/input inventories differ")

    whole_rows, subset_rows = [], []
    singular, whole_singular, first_diagonal_zero = [], [], []
    for identifier, (words, request) in native_inputs.items():
        width = {1: 1, 4: 2}[request["mode"]]
        values = [ieee(words[index:index + width])["value"] for index in range(0, len(words), width)]
        invariants = values[24:40]
        s = lambda i, j: invariants[i + 4 * j]
        require(all(s(i, j) == s(j, i) for i in range(4) for j in range(4)),
                "input invariant matrix is asymmetric")
        gram = [[(s(i, 0) + s(j, 0) - s(i, j)) / 2 for j in (1, 2, 3)] for i in (1, 2, 3)]
        gram_det = determinant3(gram)
        # This is the reducer's error-named subset, not the half-scaled whole Gram.
        h = [[s(1, i) + s(1, j) - s(i, j) for j in (2, 3)] for i in (2, 3)]
        h_det = h[0][0] * h[1][1] - h[0][1] * h[1][0]
        require(h == [[Fraction(0), s(1, 3)], [s(1, 3), 2 * s(1, 3)]]
                and h_det == -(s(1, 3) ** 2), "captured subset simplification differs")
        worker = closure["workers"][identifier]
        response = worker["response"]
        require(worker["input_pin"] == {key: request["input"][key] for key in ("bytes", "sha256")},
                "native worker used another input")
        for key in ("id", "point", "mode", "variant"):
            require(response[key] == request[key], "response/input identity differs")
        expected_failure = request["point"] != 1 and request["variant"] == "stock"
        require(response["fortran_status"] == int(expected_failure), "closed replay status differs")
        require((h_det == 0) == expected_failure, "subset singularity differs from captured cases")
        if expected_failure:
            require(all(value == 0 for row in h for value in row), "stock failed subset is not exactly zero")
        if h_det == 0:
            singular.append(identifier)
        if gram_det == 0:
            whole_singular.append(identifier)
        if h[0][0] == 0:
            first_diagonal_zero.append(identifier)
        whole_rows.append({"id": identifier, "point": request["point"], "mode": request["mode"],
                           "variant": request["variant"], "input_sha256": request["input"]["sha256"],
                           "s24": s(1, 3), "gram_determinant": gram_det,
                           "native_status": response["fortran_status"]})
        subset_rows.append({"id": identifier, "input_sha256": request["input"]["sha256"],
                            "subset_members_zero_based": [1, 2, 3], "origin_zero_based": 1,
                            "matrix": h, "determinant": h_det, "exactly_singular": h_det == 0})
    require(whole["rows"] == wire(whole_rows), "recomputed whole Gram rows differ")
    require(subset["rows"] == wire(subset_rows), "recomputed subset Gram rows differ")
    require(len(first_diagonal_zero) == 10 and len(singular) == 4,
            "first-diagonal/singularity case counts differ")
    return {"source_files_rehashed": 26, "retained_trace_inputs_reconstructed": 6,
            "complete_certificates_reconstructed": 6, "word_inputs_reconstructed": 10,
            "two_scalar_entry_interventions": 4, "whole_gram_rows_recomputed": 10,
            "error_subset_gram_rows_recomputed": 10, "error_subset_singular_inputs": singular,
            "whole_gram_singular_inputs": whole_singular,
            "error_subset_first_diagonal_zero_inputs": first_diagonal_zero,
            "first_zero_diagonal_is_not_singularity_proof": True,
            "reference_source_hash_is_recorded_provenance_only": True,
            "tensor_outputs_read": False, "tensor_accuracy_qualified": False,
            "production_fix_qualified": False, "new_native_calls": 0}
