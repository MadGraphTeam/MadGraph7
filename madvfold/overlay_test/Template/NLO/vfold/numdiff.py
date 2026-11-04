#!/usr/bin/env python3
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Aug 2026) for project MadVFold (arXiv:2609.23500).

"""
numdiff.py  Tolerant numerical comparison of text result files.

Usage:
    python numdiff.py file1 file2 [options]

Options:
    --rtol FLOAT    Relative tolerance (default: 1e-6)
    --atol FLOAT    Absolute tolerance (default: 0.0)
    --atol-lines L  Comma-separated labels (first token of a line, e.g. P,PB): use the
                    absolute tolerance only for these lines, and 0 for all others
    --verbose       Print all compared number pairs, not just failures
    --summary       Print a summary line at the end

A number pair (a, b) passes if:
    |a - b| <= atol + rtol * max(|a|, |b|)
"""

import re
import sys
import argparse

# Matches standard floats and scientific notation, e.g. 6.33271335e-02, 3.14, -1.5e+10
NUMBER_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def extract_numbers(line):
    """Return list of (start, end, float_value) for all numbers in the line."""
    results = []
    for m in NUMBER_RE.finditer(line):
        try:
            results.append((m.start(), m.end(), float(m.group())))
        except ValueError:
            pass
    return results


def close_enough(a, b, rtol, atol):
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def compare_files(path1, path2, rtol, atol, verbose, summary, atol_lines=None):
    with open(path1) as f1, open(path2) as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    n_lines = max(len(lines1), len(lines2))
    n_compared = 0
    n_failed = 0
    line_failures = []

    for i in range(n_lines):
        l1 = lines1[i] if i < len(lines1) else ""
        l2 = lines2[i] if i < len(lines2) else ""

        nums1 = extract_numbers(l1)
        nums2 = extract_numbers(l2)

        # Non-numeric mismatch (structural difference)
        if len(nums1) != len(nums2):
            print(f"Line {i+1}: token count mismatch ({len(nums1)} vs {len(nums2)})")
            print(f"  < {l1.rstrip()}")
            print(f"  > {l2.rstrip()}")
            n_failed += 1
            continue

        atol_used = atol
        if atol_lines is not None:
            tokens = l1.split()
            atol_used = atol if tokens and tokens[0] in atol_lines else 0.0
        line_ok = True
        for j, ((s1, e1, v1), (s2, e2, v2)) in enumerate(zip(nums1, nums2)):
            n_compared += 1
            ok = close_enough(v1, v2, rtol, atol_used)
            if not ok:
                n_failed += 1
                line_ok = False
                abs_diff = abs(v1 - v2)
                rel_diff = abs_diff / max(abs(v1), abs(v2)) if max(abs(v1), abs(v2)) != 0 else 0.0
                print(f"Line {i+1}, number {j+1}: FAIL  {v1!r} vs {v2!r}"
                      f"  |diff|={abs_diff:.3e}  rel={rel_diff:.3e}"
                      f"  (rtol={rtol:.1e}, atol={atol_used:.1e})")
                print(f"  < {l1.rstrip()}")
                print(f"  > {l2.rstrip()}")
            elif verbose:
                print(f"Line {i+1}, number {j+1}: ok    {v1!r} vs {v2!r}")

    if summary:
        status = "PASS" if n_failed == 0 else "FAIL"
        print(f"[{status}] {n_compared} numbers compared, {n_failed} outside tolerance "
              f"(rtol={rtol:.1e}, atol={atol:.1e})")

    return n_failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Compare two numerical result files within a tolerance."
    )
    parser.add_argument("file1", help="First file (reference)")
    parser.add_argument("file2", help="Second file (to compare)")
    parser.add_argument("--rtol", type=float, default=1e-6,
                        help="Relative tolerance (default: 1e-6)")
    parser.add_argument("--atol", type=float, default=0.0,
                        help="Absolute tolerance (default: 0.0)")
    parser.add_argument("--atol-lines", default=None,
                        help="Use atol only for lines with these comma-separated labels (default: all lines)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print all compared pairs, not just failures")
    parser.add_argument("--summary", action="store_true", default=True,
                        help="Print summary at end (default: on)")
    parser.add_argument("--no-summary", dest="summary", action="store_false")
    args = parser.parse_args()

    ok = compare_files(args.file1, args.file2,
                       rtol=args.rtol, atol=args.atol,
                       verbose=args.verbose, summary=args.summary,
                       atol_lines=args.atol_lines.split(",") if args.atol_lines else None)
    sys.exit(0 if ok else 1)


#--------------------------------------------------------------------------------------------------
# EXAMPLES:
# Default: relative tolerance 1e-6 (run_01 vs run_08 comparison succeeds)
#   python3 numdiff.py Events/run_01/res_2.txt Events/run_08/res_2.txt
# Stricter: relative tolerance 1e-9 (run_01 vs run_08 comparison fails)
#   python3 numdiff.py Events/run_01/res_2.txt Events/run_08/res_2.txt --rtol 1e-9
# Combined relative 1e-9 + absolute tolerance 1e-10 (run_01 vs run_08 comparison succeeds)
#   python3 numdiff.py Events/run_01/res_2.txt Events/run_08/res_2.txt --rtol 1e-9 --atol 1e-10
# Combined relative 1e-9 + absolute tolerance 1e-11 (run_01 vs run_08 comparison fails)
#   python3 numdiff.py Events/run_01/res_2.txt Events/run_08/res_2.txt --rtol 1e-9 --atol 1e-11
# Show every compared pair (not just failures)
#   python3 numdiff.py Events/run_01/res_2.txt Events/run_08/res_2.txt --verbose
#--------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
