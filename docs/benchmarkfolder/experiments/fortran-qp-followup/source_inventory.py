#!/usr/bin/env python3
"""Count actual QCD3 polynomial DAG reuse without evaluating an amplitude."""
from collections import Counter, defaultdict
import hashlib
import json
from math import comb
from pathlib import Path
import re
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prototype import logical_lines


def main():
    directory = next((ROOT / ".benchmark-work/policy-v2/generation/outputs/candidate/qcd-3").glob(
        "SubProcesses/P*_*"))
    sources = sorted(directory.glob("mp_coef_construction_*.f"))
    text = "\n".join(logical_lines(p.read_text()) for p in sources)
    calls = [tuple(map(int, match.groups())) for match in re.finditer(
        r"CALL MP_\w+UPDATE_WL_(\d+)_(\d+)\(WL\(1,0,1,(-?\d+)\),"
        r"(\d+),COEFS,(\d+),(\d+),WL\(1,0,1,(-?\d+)\)\)", text)]
    if not calls:
        raise RuntimeError("No generated updater calls recognized")
    roots = list(map(int, re.findall(
        r"CALL MP_\w+CREATE_LOOP_COEFS\(WL\(1,0,1,(-?\d+)\)", text)))
    producers, consumers, scans = {}, Counter(), defaultdict(list)
    for left, right, origin, cut, incoming, outgoing, destination in calls:
        if destination in producers:
            raise RuntimeError("Wavefunction number is reused within the batch")
        producers[destination] = origin
        consumers[origin] += 1
        scans[origin].append(comb(left+4, 4)*cut*incoming)
    needed = set(roots)
    stack = list(roots)
    while stack:
        node = stack.pop()
        predecessor = producers.get(node)
        if predecessor is not None and predecessor not in needed:
            needed.add(predecessor)
            stack.append(predecessor)
    terminal_roots = set(roots) - set(consumers)
    def dense_products(row):
        left, right, _, cut, incoming, outgoing, _ = row
        return comb(left+4, 4)*comb(right+4, 4)*cut*incoming*outgoing
    full_products = sum(map(dense_products, calls))
    terminal_products = sum(dense_products(row) for row in calls
                            if row[-1] in terminal_roots)
    omitted_offdiagonal = sum(dense_products(row)*(1-1/row[-2]) for row in calls
                             if row[-1] in terminal_roots)
    result = {
        "scope": "Static calls in one final-QCD3 helicity coefficient batch. "
                 "Scan estimates count active complex A entries, not cycles or an estimated speedup.",
        "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sources},
        "updater_calls": len(calls),
        "calls_by_rank": dict(Counter(f"{a}/{b}" for a,b,*_ in calls)),
        "merge_roots": len(roots),
        "unique_produced_wavefunctions": len(producers),
        "all_produced_wavefunctions_needed": set(producers).issubset(needed),
        "unique_updater_inputs": len(consumers),
        "repeated_input_calls": sum(max(0,n-1) for n in consumers.values()),
        "reused_inputs_by_count": consumers.most_common(),
        "a_complex_entries_checked_per_batch": sum(sum(v) for v in scans.values()),
        "optimistic_a_entries_if_once_per_wavefunction": sum(max(v) for v in scans.values()),
        "rank_two_or_higher_repeated_input_calls": sum(
            max(0, consumers[n]-1) for n in consumers
            if any(a >= 2 and origin == n for a,b,origin,*_ in calls)),
        "terminal_merge_roots": len(terminal_roots),
        "dense_complex_products_per_batch": full_products,
        "terminal_dense_complex_products_per_batch": terminal_products,
        "unconsumed_terminal_offdiagonal_product_fraction": omitted_offdiagonal/full_products,
        "terminal_structure_caveat": "MERGE_WL consumes only the diagonal, but omitting "
                                    "off-diagonal arithmetic can change IEEE flags and "
                                    "unconsumed internal wavefunction values. No pruning "
                                    "is implemented or authorized by this inventory. "
                                    "Counts precede all existing zero skips.",
        "limitations": "A persistent mask requires explicit per-helicity/precision invalidation "
                       "and finite/rounding semantics. The count does not establish an optimization. "
                       "Every generated root must remain available to reduction.",
    }
    (HERE / "source-inventory.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in
                      ("sources","reused_inputs_by_count","limitations")},indent=2))


if __name__ == "__main__":
    main()
