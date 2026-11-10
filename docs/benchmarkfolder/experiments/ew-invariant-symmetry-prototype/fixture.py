#!/usr/bin/env python3
"""Pure binary64 arithmetic witness; no Fortran, reducer, or project imports.

Default verifies fixture.json without writes. --write creates it exclusively.
This is a source-derived synthetic input, not a captured amplitude input.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CAUSE = "docs/benchmarkfolder/experiments/dp-rescue-cause-observer/"
CARD_SHA = "891e4b89d522804e1f21c3f8b19b8cc2209b29cbfb7809189a5e7062c1363ea8"
PARAM_SHA = "5300bd6d7f3c4553c3402ab7a044e8fbca3c646f451a5f586215933bfa113b85"
SOURCES = {
    "madgraph/iolibs/template_files/loop_optimized/CT_interface.inc":
        "dfd7cf893599d2976041c1d603c08d888640bc74938d3605e14a707168e6cf16",
    "madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc":
        "ae18cc526313dac260be88c40d3ac92210687a576f94f32f1bf42a5311afa9ec",
    "vendor/tirust/src/lib.rs":
        "455529b929b725e8fe2d7337ab4a41fa3de9dee1b3a953afe6dc032cf99e5a4c",
    CAUSE + "prepared/ew_n3_epem_ttxa/TIRUST_interface.f.original":
        "e60570ec599263807165f2f289add2076a4c5b71b9e09fe0de83306e9ab2f7ff",
    CAUSE + "preparation/source-1/snapshot.json":
        "2aa5e87ead00c163d0f987dbef257654a8a22ec748f7d3f71be68aeaa64e7759",
}
GZIP_SHA = {
    CARD_SHA: "dfae8ac50aaec87b4e8e25f0db567aa2db963d45395b73669022217a2a893439",
    PARAM_SHA: "01964d08ef58f251a8025ef30041a954aba0e119c56a5bc25a9bcb573acb95e6",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def number(value):
    """Retain the exact binary64 value without decimal JSON-number ambiguity."""
    assert math.isfinite(value)
    num, den = value.as_integer_ratio()
    return {"decimal": repr(value), "hex_float": value.hex(),
            "binary64_word_be": struct.pack(">d", value).hex(),
            "exact_numerator": str(num), "exact_denominator": str(den)}


def source_bindings():
    bindings = []
    for path, expected in SOURCES.items():
        data = (ROOT / path).read_bytes()
        assert sha(data) == expected, path
        bindings.append({"path": path, "sha256": expected, "size": len(data)})
    payloads = {}
    for raw_sha, compressed_sha in GZIP_SHA.items():
        path = CAUSE + "numerical-qualification-1/payloads/" + raw_sha + ".gz"
        compressed = (ROOT / path).read_bytes()
        assert sha(compressed) == compressed_sha, path
        data = gzip.decompress(compressed)
        assert sha(data) == raw_sha, path
        payloads[raw_sha] = data.decode("utf-8")
        bindings.append({"path": path, "sha256": compressed_sha,
                         "size": len(compressed), "raw_sha256": raw_sha,
                         "raw_size": len(data)})
    snapshot = json.loads((ROOT / (CAUSE + "preparation/source-1/snapshot.json")).read_text())
    sources = snapshot["cases"]["ew_n3_epem_ttxa"]["all_sources_and_includes"]
    generated_ct = sources["SubProcesses/P0_epem_ttxa/CT_interface.f"]
    assert generated_ct["sha256"] == "ac240d6ffc2f7e1a15a72e61a325a53d0acf776f65b9329e4e9518cd85f3da0a"
    return bindings, payloads, generated_ct["sha256"]


def derive():
    assert sys.float_info.radix == 2 and sys.float_info.mant_dig == 53
    bindings, payloads, generated_ct_sha = source_bindings()
    card = payloads[CARD_SHA].splitlines()
    tokens = {}
    mass_block = False
    for lineno, line in enumerate(card, 1):
        fields = line.split("#", 1)[0].split()
        if fields and fields[0].lower() == "block":
            mass_block = fields[1].lower() == "mass"
        elif mass_block and len(fields) == 2 and fields[0] in ("6", "24"):
            tokens[fields[0]] = {"line": lineno, "token": fields[1], "text": line}
    assert {key: value["token"] for key, value in tokens.items()} == {
        "6": "1.733000e+02", "24": "8.041900e+01"}
    params = payloads[PARAM_SHA].splitlines()
    threshold_line = params.index("#OSThres") + 1
    threshold_token = params[threshold_line].strip()
    assert threshold_token == "1.000000d-08"
    threshold = float(threshold_token.replace("d", "e"))
    steps = []

    def op(label, operator, left, right):
        a, b = Fraction.from_float(left), Fraction.from_float(right)
        if operator == "+":
            value, exact = left + right, a + b
        elif operator == "-":
            value, exact = left - right, a - b
        elif operator == "*":
            value, exact = left * right, a * b
        elif operator == "/":
            value, exact = left / right, a / b
        else:
            raise AssertionError(operator)
        # Independent exact-rational rounding check for every scalar operation.
        assert value == float(exact), label
        steps.append({"label": label, "operator": operator,
                      "left": number(left), "right": number(right),
                      "exact_unrounded": str(exact), "result": number(value)})
        return value

    mass = [float(tokens[pdg]["token"]) for pdg in ("6", "24")]
    assert all(m == float(Fraction(tokens[pdg]["token"]))
               for m, pdg in zip(mass, ("6", "24")))
    squared = [op(f"mass_squared_{i+1}", "*", m, m) for i, m in enumerate(mass)]
    shifts = [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    s_matrix = [[None, None], [None, None]]
    matrix = [[None, None], [None, None]]
    pairs = []
    for i in range(2):
        for j in range(2):
            label = f"I{i+1}J{j+1}"
            if i == j:
                s = -op(label + "_mass_sum", "+", squared[i], squared[j])
            else:
                d0 = op(label + "_time_difference", "-", shifts[i][0], shifts[j][0])
                diff = op(label + "_time_square", "*", d0, d0)
                for k in range(1, 4):
                    dk = op(label + f"_space{k}_difference", "-", shifts[i][k], shifts[j][k])
                    dk2 = op(label + f"_space{k}_square", "*", dk, dk)
                    diff = op(label + f"_subtract_space{k}", "-", diff, dk2)
                s1 = op(label + "_subtract_mass_i", "-", diff, squared[i])
                s = op(label + "_subtract_mass_j", "-", s1, squared[j])
                guards = []
                for index in (i, j):
                    delta = op(label + f"_shell_delta_{index+1}", "-", diff, squared[index])
                    ratio = op(label + f"_shell_ratio_{index+1}", "/", delta, squared[index])
                    guards.append(abs(ratio) < threshold)
                ref = op(label + "_ref_sum_i", "+", 0.0, abs(shifts[i][0]))
                ref = op(label + "_ref_sum_j", "+", ref, abs(shifts[j][0]))
                ref = op(label + "_ref_half", "/", ref, 2.0)
                ref = op(label + "_ref_square", "*", ref, ref)
                ratio = op(label + "_zero_shell_ratio", "/", diff, ref)
                guards.append(abs(ratio) < threshold)
                assert guards == [False, False, False]
                pairs.append({"i": i + 1, "j": j + 1, "diffsq": number(diff),
                              "s_mat": number(s), "mass_i_override": guards[0],
                              "mass_j_override": guards[1], "zero_override": guards[2]})
            s_matrix[i][j] = s
    # The shared routine completes its I-major matrix before the adapter's
    # J-major reconstruction. Preserve that global ordering in the trace too.
    for j in range(2):
        for i in range(2):
            label = f"I{i+1}J{j+1}"
            r1 = op(label + "_restore_mass_i", "+", s_matrix[i][j], squared[i])
            matrix[i][j] = op(label + "_restore_mass_j", "+", r1, squared[j])
    upper, lower = matrix[0][1], matrix[1][0]
    assert upper.hex() == "0x1.fffffffffc000p-1"
    assert lower.hex() == "0x1.fffffffff8000p-1"
    assert upper - lower == 2.0**-39
    assert all(math.isfinite(x) for row in matrix for x in row)
    repaired = [row[:] for row in matrix]
    if all(math.isfinite(x) for row in repaired for x in row):
        if repaired[1][0] != repaired[0][1]:
            repaired[1][0] = repaired[0][1]
    assert repaired[0] == matrix[0] and repaired[1][1] == matrix[1][1]
    assert repaired[0][1] == repaired[1][0]
    return {
        "schema": "ew-invariant-symmetry-synthetic-binary64-v1",
        "scope": {"synthetic_source_derived": True, "captured_physical_input": False,
                  "compiled_fortran": False, "reducer_evaluation": False,
                  "ieee_state_measured": False, "binary128_witness": False,
                  "physical_or_performance_qualification": False},
        "arithmetic": "separate Python binary64 operations, each checked against exact Fraction rounded to binary64; finite zero-width real components only",
        "source_bindings": bindings,
        "generated_ct_source_sha256_via_snapshot": generated_ct_sha,
        "fixture_script_sha256": sha(Path(__file__).read_bytes()),
        "mass_source": {"model": "loop_qcd_qed_sm_Gmu", "case": "ew_n3_epem_ttxa",
                        "card_sha256": CARD_SHA, "tokens": tokens},
        "threshold": {"token": threshold_token, "line": threshold_line + 1,
                      "value": number(threshold)},
        "mass_values": [number(x) for x in mass],
        "mass_squared": [number(x) for x in squared],
        "shifts": [[number(x) for x in row] for row in shifts],
        "pairs": pairs, "operation_trace": steps,
        "matrix_before": [[number(x) for x in row] for row in matrix],
        "matrix_after_conditional_upper_to_lower": [[number(x) for x in row] for row in repaired],
        "upper_minus_lower": number(upper - lower),
        "strict_symmetry_before": upper == lower, "strict_symmetry_after": repaired[0][1] == repaired[1][0],
        "upper_triangle_and_diagonal_bits_unchanged": True,
        "qualification_limit": "The scalar rounding mechanism is reproducible; it does not identify the actual invariant words behind any of the 531 observed EW errors or prove native/compiled behavior."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="exclusively create fixture.json")
    args = parser.parse_args()
    result = derive()
    path = HERE / "fixture.json"
    if args.write:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
            stream.write("\n")
    else:
        assert json.loads(path.read_text()) == result, "fixture.json differs"
    print(json.dumps({"passed": True, "fixture_sha256": sha(path.read_bytes()),
                      "strict_symmetry_before": False, "strict_symmetry_after": True,
                      "native_execution": False}, sort_keys=True))


if __name__ == "__main__":
    main()
