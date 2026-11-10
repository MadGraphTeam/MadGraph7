#!/usr/bin/env python3
"""Closed CMS classification audit: integer/Fraction arithmetic, no native calls.

This records a nearest-even, operation-by-operation model, not an observation of
compiler instructions, runtime rounding mode, or unprinted native values.
Only the sibling summary.json is written, and only with --write.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CMS = "docs/benchmarkfolder/experiments/complex-mass/"
POINTS = CMS + "points/cms_ew_n2_epem_mupmum.dat"
FIXED = CMS + "inputs/baseline/cms_ew_n2_epem_mupmum/cms_onshell_probe"
MP = CMS + "direct-qp/cms_mp_shell_probe"
TRACE = CMS + "direct-qp/actual-matrix-trace/"
HELPER = "madgraph/iolibs/template_files/loop_optimized/CT_interface.inc"
INPUTS = [POINTS, POINTS[:-4] + ".json", FIXED + ".f", FIXED + ".dat",
          FIXED + ".json", MP + ".f", MP + ".dat",
          CMS + "direct-qp/cms_mp_shell_audit.json", HELPER,
          TRACE + "cms_actual_mp_matrices.dat", TRACE + "specification.json",
          TRACE + "summary.json", TRACE + "trace-insertion.f"]


def p2(e: int) -> F:
    return F(1 << e) if e >= 0 else F(1, 1 << -e)


def log2floor(x: F) -> int:
    assert x > 0
    e = x.numerator.bit_length() - x.denominator.bit_length()
    return e - (x < p2(e))


def rn(x: F, bits: int) -> F:
    """Nearest-even binary rounding, restricted to finite normal nonzero values."""
    if not x:
        return F(0)
    sign = -1 if x < 0 else 1
    x = abs(x)
    e = log2floor(x)
    assert (-1022 if bits == 53 else -16382) <= e <= (1023 if bits == 53 else 16383)
    step = p2(e - bits + 1)
    scaled = x / step
    m, rem = divmod(scaled.numerator, scaled.denominator)
    if 2 * rem > scaled.denominator or (2 * rem == scaled.denominator and m % 2):
        m += 1
    return sign * m * step


def dec(token: str) -> F:
    return F(Decimal(token.replace("D", "E").replace("d", "E")))


def scientific(x: F) -> str:
    if not x:
        return "0"
    with localcontext() as ctx:
        ctx.prec = 65
        return format(Decimal(x.numerator) / Decimal(x.denominator), ".44E")


def value(x: F) -> dict:
    return {"fraction": str(x), "decimal_45_significant_digits": scientific(x)}


def word(x: F, bits: int, negative_zero: bool = False) -> str:
    total, ebits, bias = (64, 11, 1023) if bits == 53 else (128, 15, 16383)
    if not x:
        n = int(negative_zero) << (total - 1)
    else:
        e = log2floor(abs(x))
        m = abs(x) / p2(e - bits + 1)
        assert m.denominator == 1 and 1 << (bits - 1) <= m < 1 << bits
        n = (int(x < 0) << (total - 1)) | ((e + bias) << (bits - 1)) | (int(m) - (1 << (bits - 1)))
    return f"{n:0{total // 4}x}"


def printed_bin(token: str, bits: int = 113) -> tuple[F, dict]:
    """Enumerate ALL binary normals in the closed half-decimal-quantum bin.

    Closed endpoints conservatively include either decimal tie convention. A
    unique candidate proves reconstruction without assuming decimal tie rules.
    Scientific ES zero is treated separately: a finite nonzero normalized ES
    value never prints an all-zero significand with its four-digit exponent.
    """
    d = Decimal(token.replace("D", "E"))
    center = F(d)
    negzero = d.is_signed() and not d
    if not center:
        return F(0), {"token": token, "candidate_count": 1,
                      "method": "exact signed zero from normalized ES format, not a fixed-decimal interval",
                      "negative_zero": negzero, "ieee_hex": word(F(0), bits, negzero)}
    quantum = F(10) ** d.as_tuple().exponent
    lo, hi = abs(center) - quantum / 2, abs(center) + quantum / 2
    assert lo > 0
    candidates = []
    for e in range(log2floor(lo), log2floor(hi) + 1):
        step = p2(e - bits + 1)
        a, b = lo / step, hi / step
        start = max(-(-a.numerator // a.denominator), 1 << (bits - 1))
        stop = min(b.numerator // b.denominator, (1 << bits) - 1)
        assert stop - start < 10, "unexpected broad print interval"
        candidates.extend(m * step for m in range(start, stop + 1))
    assert len(candidates) == 1, (token, bits, len(candidates))
    x = candidates[0] * (-1 if center < 0 else 1)
    return x, {"token": token, "candidate_count": len(candidates),
               "method": "complete normal-binade enumeration in closed half-quantum interval",
               "absolute_lower_bound": str(lo), "absolute_upper_bound": str(hi),
               "ieee_hex": word(x, bits)}


def in_print_bin(x: F, token: str) -> bool:
    d = Decimal(token.replace("D", "E"))
    if not d:
        return x == 0
    return abs(x - F(d)) <= F(10) ** d.as_tuple().exponent / 2


def differences(qi: list[F], qj: list[F], bits: int) -> list[F]:
    return [rn(a - b, bits) for a, b in zip(qi, qj)]


def exact_minkowski(q: list[F]) -> F:
    return q[0] ** 2 - sum((x * x for x in q[1:]), F(0))


def sequential(qi: list[F], qj: list[F], bits: int) -> F:
    squares = [rn(x * x, bits) for x in differences(qi, qj, bits)]
    result = squares[0]
    for term in squares[1:]:
        result = rn(result - term, bits)
    return result


def direct_sum(qi: list[F], qj: list[F], bits: int) -> F:
    squares = [rn(x * x, bits) for x in differences(qi, qj, bits)]
    total = F(0)
    for term in squares[1:]:
        total = rn(total + term, bits)
    return rn(squares[0] - total, bits)


def metrics(qi: list[F], qj: list[F], bits: int) -> dict:
    exact_delta = [a - b for a, b in zip(qi, qj)]
    delta = differences(qi, qj, bits)
    diffsq = sequential(qi, qj, bits)
    # The helper's additions are left associative from zero; only k=0 executes.
    energy_sum = rn(rn(F(0) + abs(qi[0]), bits) + abs(qj[0]), bits)
    ref = rn(rn(energy_sum / 2, bits) ** 2, bits)
    l2sq = sum((x * x for x in delta), F(0))
    linfsq = max(map(abs, delta)) ** 2
    l1sq = sum(map(abs, delta), F(0)) ** 2
    ratio = abs(rn(diffsq / ref, bits)) if ref else None
    return {
        "precision_bits": bits,
        "qi_ieee_hex": [word(x, bits) for x in qi],
        "qj_ieee_hex": [word(x, bits) for x in qj],
        "exact_binary_input_difference": [str(x) for x in exact_delta],
        "rounded_component_differences": [str(x) for x in delta],
        "exact_binary_input_minkowski_gev2": value(exact_minkowski(exact_delta)),
        "helper_sequential_diffsq_gev2": value(diffsq),
        "direct_sum_left_to_right_gev2": value(direct_sum(qi, qj, bits)),
        "helper_energy_denominator_gev2": value(ref),
        "exact_energy_ratio_of_rounded_values": value(abs(diffsq / ref)) if ref else None,
        "rounded_helper_energy_ratio": value(ratio) if ratio is not None else None,
        "difference_euclidean_norm_squared_gev2": value(l2sq),
        "difference_max_component_squared_gev2": value(linfsq),
        "difference_l1_norm_squared_gev2": value(l1sq),
        "abs_diffsq_over_difference_euclidean_norm_squared": value(abs(diffsq) / l2sq) if l2sq else None,
        "massless_classification_by_saved_thresholds": {
            token: bool(ref and ratio < rn(dec(token), 53))
            for token in ("1e-8", "1e-14", "1e-24")},
    }


def read_points() -> list[list[list[F]]]:
    lines = (ROOT / POINTS).read_text().splitlines()
    count = int(lines[0])
    vectors = [[rn(dec(x), 53) for x in line.split()] for line in lines[1:] if line.strip()]
    assert len(vectors) == 4 * count == 84
    return [vectors[4 * i:4 * i + 4] for i in range(count)]


def read_fixed_source() -> list[list[list[F]]]:
    """Bind fixed probe literal assignments, not only its printed outputs."""
    vectors, blocks = {}, []
    for line in (ROOT / (FIXED + ".f")).read_text().splitlines():
        if "SUBROUTINE" in line:
            break
        match = re.fullmatch(r"\s*P\(([12]),([0-3])\)=([0-9.D+\-]+)\s*", line)
        if match:
            i, a, literal = match.groups()
            vectors[int(i), int(a)] = rn(dec(literal), 53)
        if line.strip() == "QP=REAL(P,KIND=16)":
            assert len(vectors) == 8
            blocks.append([[vectors[i, a] for a in range(4)] for i in (1, 2)])
            vectors = {}
    assert len(blocks) == 3
    return blocks


def read_probe():
    coords, shells, km, proofs = {}, {}, {}, []
    for lineno, line in enumerate((ROOT / (MP + ".dat")).read_text().splitlines(), 1):
        fields = line.split()
        if fields[0] in ("RAW", "IMPROVED") and int(fields[1]) in (20, 21):
            key = (fields[0], int(fields[1]), int(fields[2]))
            pairs = [printed_bin(token) for token in fields[3:]]
            coords[key] = [pair[0] for pair in pairs]
            proofs.append({"line": lineno, "kind": key[0], "point": key[1], "leg": key[2],
                           "coordinates": [pair[1] for pair in pairs]})
        elif fields[0] == "SHELL" and int(fields[1]) in (20, 21):
            shells[int(fields[1]), int(fields[2])] = fields[3]
        elif fields[0] == "KM" and int(fields[1]) in (20, 21):
            km[tuple(map(int, fields[1:5]))] = fields[5:]
    assert len(coords) == 16 and len(shells) == 8 and len(km) == 96
    return coords, shells, km, proofs


def trace_audit(repaired: list[list[F]]) -> dict:
    matrices = []
    token_count = 0
    for lineno, line in enumerate((ROOT / (TRACE + "cms_actual_mp_matrices.dat")).read_text().splitlines(), 1):
        fields = line.split()
        if fields[0] == "BEGIN":
            current = {"n": int(fields[1]), "q": {}, "s": {}, "begin_line": lineno}
            current["threshold"], _ = printed_bin(fields[2], 53)
        elif fields[0] == "Q":
            pairs = [printed_bin(x) for x in fields[2:]]
            current["q"][int(fields[1])] = [x[0] for x in pairs]
            token_count += len(pairs)
        elif fields[0] == "S":
            pairs = [printed_bin(x) for x in fields[3:]]
            current["s"][int(fields[1]), int(fields[2])] = [x[0] for x in pairs]
            token_count += len(pairs)
        elif fields[0] == "END":
            assert len(current["q"]) == current["n"] and len(current["s"]) == current["n"] ** 2
            matrices.append(current)
    target = differences(repaired[0], repaired[2], 113)
    exact_matches, close_only, classes = [], [], {}
    for number, matrix in enumerate(matrices, 1):
        for (i, j), s in matrix["s"].items():
            qi, qj = matrix["q"][i], matrix["q"][j]
            delta = differences(qi[:4], qj[:4], 113)
            exact = delta == target or delta == [-x for x in target]
            gap = min(max(abs(a - sign * b) for a, b in zip(delta, target)) for sign in (-1, 1))
            if not exact:
                if gap < F(1, 10 ** 28):
                    close_only.append({"matrix": number, "i": i, "j": j,
                                       "maximum_component_gap_gev": str(gap)})
                continue
            entry = {"matrix": number, "matrix_begin_line": matrix["begin_line"], "i": i, "j": j}
            exact_matches.append(entry)
            key = tuple(qi + qj + s)
            if key not in classes:
                recovered = [s[k] + qi[4 + k] + qj[4 + k] for k in range(2)]
                classes[key] = {"occurrences": [], "metrics": metrics(qi[:4], qj[:4], 113),
                                "mass_i_squared_re_im": list(map(str, qi[4:])),
                                "mass_j_squared_re_im": list(map(str, qj[4:])),
                                "exact_recovered_s_plus_masses_re_im": list(map(str, recovered)),
                                "saved_threshold": str(matrix["threshold"])}
            classes[key]["occurrences"].append(entry)
    old = json.loads((ROOT / (TRACE + "summary.json")).read_text())
    return {"point": 20, "matrices": len(matrices), "binary128_trace_token_occurrences_reconstructed": token_count,
            "all_trace_q_and_s_values_reconstructed_with_unique_nonzero_bins_and_zero_format_semantics": True,
            "exact_rounded_delta_matches_to_repaired_p1_minus_p3": len(exact_matches),
            "close_only_matches_at_historical_1e_minus28_coordinate_tolerance": close_only,
            "historical_t_entry_count": old["counts"]["physical_t_entries"],
            "distinct_channel_input_output_classes": list(classes.values()),
            "scope": "Saved point20 alternate OSThres=1e-24 four-call trace; no primary-policy execution, no point21 trace, no loop-group labels."}


def compute() -> dict:
    points = read_points()
    fixed_source = read_fixed_source()
    coords, shells, km, proofs = read_probe()
    cases = []
    validation = {"raw_promotions_equal_exact_saved_binary64": True,
                  "fixed_probe_binary64_literal_inputs_equal_saved_point_p1_and_p3": True,
                  "all_repaired_shell_prints_contain_direct_sum_model": True,
                  "all_96_saved_km_direct_raw_fields_match_left_to_right_sum_model": True,
                  "all_96_saved_km_helper_fields_match_sequential_model_and_threshold": True}
    fixed_rows = {row.split()[0]: row.split()[1:] for row in (ROOT / (FIXED + ".dat")).read_text().splitlines()}
    for point in (20, 21):
        raw = points[point - 1]
        assert fixed_source[point - 19] == [raw[0], raw[2]]
        repaired = [coords["IMPROVED", point, leg] for leg in range(1, 5)]
        assert [coords["RAW", point, leg] for leg in range(1, 5)] == raw
        variants = {"raw_binary64": (raw, 53), "raw_binary64_promoted_exactly_to_binary128": (raw, 113),
                    "repaired_binary128_from_saved_unique_print_bins": (repaired, 113)}
        row = {"point": point, "id": "forward-1e-06" if point == 20 else "forward-1e-08", "variants": {}}
        for name, (vectors, bits) in variants.items():
            row["variants"][name] = {"physical_t_p1_minus_p3": metrics(vectors[0], vectors[2], bits),
                                      "outgoing_leg3_shell": metrics(vectors[2], [F(0)] * 4, bits)}
        fixed = fixed_rows[row["id"]]
        fixed_checks = [in_print_bin(direct_sum(raw[0], raw[2], b), fixed[k]) for b, k in ((53, 0), (113, 2))]
        assert all(fixed_checks) and not dec(fixed[1]) and not dec(fixed[3])
        row["fixed_probe_print_contains_direct_sum_model"] = fixed_checks
        for leg, vector in enumerate(repaired, 1):
            assert in_print_bin(direct_sum(vector, [F(0)] * 4, 113), shells[point, leg])
        shifts = [[F(0)] * 4, repaired[0], [rn(a + b, 113) for a, b in zip(repaired[0], repaired[1])], repaired[2]]
        for h in (1, 2, 3):
            for i in range(1, 5):
                for j in range(1, 5):
                    threshold_token, direct_token, saved_token = km[point, h, i, j]
                    threshold = rn(dec(threshold_token), 53)
                    assert in_print_bin(direct_sum(shifts[i - 1], shifts[j - 1], 113), direct_token)
                    m = metrics(shifts[i - 1], shifts[j - 1], 113)
                    ds = sequential(shifts[i - 1], shifts[j - 1], 113)
                    ratio_record = m["rounded_helper_energy_ratio"]
                    predicted = F(0) if i == j or (ratio_record is not None and F(ratio_record["fraction"]) < threshold) else ds
                    assert in_print_bin(predicted, saved_token), (point, h, i, j, predicted, saved_token)
        row["retained_80digit_physical_expectations"] = next(r for r in json.loads((ROOT / (CMS + "direct-qp/cms_mp_shell_audit.json")).read_text())["rows"] if r["point_index"] == point)
        cases.append(row)
    blobs = {name: (ROOT / name).read_bytes() for name in INPUTS}
    return {
        "schema": "closed-cms-invariant-classification-v1",
        "scope": "Pure exact-fraction audit of points20/21 and saved probes; no new physics or numerical-library evaluation.",
        "rounding_model": "Round-to-nearest ties-to-even after each real subtraction, square, sequential subtraction, normalization and division; finite normal53/113-bit values only. Complex helper inputs here have zero imaginary momentum components. Compiler transformations, hardware flags and actual rounding mode are not observed.",
        "print_reconstruction": "All retained RAW/IMPROVED coordinates and all actual-trace Q/S tokens are reconstructed by full closed-bin enumeration; no blanket35-significant-digit round-trip assumption. Zero tokens use normalized scientific-format semantics and retain sign in reconstruction proofs. Probe raw DIFFSQ prints have only33significant digits and are interval-checked, never treated as exact113-bit tokens.",
        "operation_order": {"production_helper": HELPER + ":198-225,270-297; sequential spatial subtraction",
                            "fixed_probe": FIXED + ".f:21-23,37-39,53-55; spatial SUM",
                            "repaired_probe": MP + ".f:28,40-43; spatial SUM; supplied helper executes sequentially",
                            "sum_model_limit": "SUM modeled left-to-right; agreement with saved prints is checked, not assumed to establish compiler reduction order."},
        "saved_threshold_binary64": {s: value(rn(dec(s), 53)) for s in ("1e-8", "1e-14", "1e-24")},
        "validation": validation,
        "input_inventory": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in blobs.items()},
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coordinate_reconstruction_proofs": proofs,
        "repaired_coordinate_reconstruction_counts": {
            "nonzero_unique_closed_bin_coordinates": sum(bool(dec(p["token"])) for row in proofs if row["kind"] == "IMPROVED" for p in row["coordinates"]),
            "zero_coordinates_from_normalized_ES_format": sum(not dec(p["token"]) for row in proofs if row["kind"] == "IMPROVED" for p in row["coordinates"])},
        "concrete_findings": [
            "At both points, the p1-p3 transfer is nonzero and spacelike for raw53, exact-promoted113 and repaired113 inputs. Its energy denominator is exactly250000GeV² and the rounded ratios are approximately1e-12/1e-16. OSThres1e-8 therefore snaps both transfers to zero in the massless probe.",
            "The rounded physical-channel time difference is zero; all nonzero squared components have the same spatial sign. The difference-vector Euclidean-square scale equals the absolute exact Minkowski product. This local calculation has no cancellation between timelike and spacelike squares, despite its small energy-normalized ratio.",
            "Repaired outgoing-leg3 sequential113 shells are +2^-95 and -2^-95GeV² for points20/21; their exact rational Minkowski products are separately retained and are different. The shell energy denominator for the zero-to-p3 probe pair is62500GeV², giving ratios approximately4.04e-34.",
            "The point20 alternate-threshold actual trace contains40 exact rounded-delta matches to repaired p1-p3 across104 matrices, in two ordered input/output classes. Both masses are exactly zero in those classes; saved S equals the sequential113 transfer. There are no additional tolerance-only matches under the historical1e-28coordinate test.",
            "No conclusion here reconciles EW threshold mass-square conventions, changes a classification guard, or establishes full CMS numerical acceptance."],
        "cases": cases,
        "actual_point20_trace": trace_audit([coords["IMPROVED", 20, leg] for leg in range(1, 5)]),
        "limitations": ["No amplitude accuracy or stability qualification; no finite-term oracle.",
                        "The actual-matrix trace used alternate OSThres=1e-24. Predicted primary-threshold classification is arithmetic analysis, not a new primary-policy run.",
                        "Massless probe invariants omit complex internal masses; the full CMS model is established by the separately retained campaign probes.",
                        "Difference-vector norm scales are descriptive diagnostics, not proposed guard replacements or certified error bounds.",
                        "Fractions do not encode zero signs; signed zero is retained separately for printed-coordinate proofs."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write only this new audit's summary.json")
    parser.add_argument("--check", action="store_true", help="compare a pure recomputation with the existing summary bytes")
    args = parser.parse_args()
    if args.write and args.check:
        parser.error("choose --write or --check")
    data = json.dumps(compute(), indent=2, sort_keys=True) + "\n"
    if args.write:
        (HERE / "summary.json").write_text(data)
    if args.check:
        assert (HERE / "summary.json").read_bytes() == data.encode(), "summary differs from pure recomputation"
    print(json.dumps({"summary_sha256": hashlib.sha256(data.encode()).hexdigest(),
                      "bytes": len(data.encode()), "written": args.write, "checked": args.check}))


if __name__ == "__main__":
    main()
