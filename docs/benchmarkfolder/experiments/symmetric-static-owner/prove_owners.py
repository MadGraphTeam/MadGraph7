#!/usr/bin/env python3
"""Check owner dominance for every actual QCD3/QCD4 source-derived group."""
import hashlib
import json
from pathlib import Path

import static_owner
from static_owner import shared

HERE = Path(__file__).resolve().parent


def main():
    inventory = static_owner.PREVIOUS / "final-inventory.json"
    data = json.loads(inventory.read_text())
    results = []
    for case in ("qcd-3", "qcd-4"):
        for vertex, routine in data[case]["routines"].items():
            if not any(s["in_size"] == s["out_size"] == 4
                       for s in routine["shapes"]):
                continue
            for terminal in (False, True):
                groups = shared.groups(vertex, terminal, case=case,
                                       inventory_path=inventory)
                if not groups:
                    continue
                ids = static_owner.owner_map(groups, terminal)
                for rank in (2, 3, 4):
                    source = shared.gather.Polynomial(rank)
                    output = shared.gather.Polynomial(rank + 1)
                    ncoef = shared.gather.get_number_of_coefs_for_rank(rank)
                    outcoef = shared.gather.get_number_of_coefs_for_rank(rank + 1)
                    writes = reads = predicates = 0
                    for lc in range(5):
                        # Each A(K,L,I) may independently vanish. Testing both
                        # predicate values per cell proves the implication for
                        # arbitrary simultaneous sparsity without enumerating
                        # 2^(4*ncoef*LCUT_SIZE) complete tensors.
                        for b_active in (False, True):
                            for a_active in (False, True):
                                assigned = set()
                                for j in range(1, 5):
                                    for m in range(5):
                                        for k in range(1, 5):
                                            group = ids.get((j, m, k), 0)
                                            if group == 0:
                                                continue
                                            indices = ([j] if j <= lc else []) if terminal else range(1, lc + 1)
                                            for i in indices:
                                                for l in range(ncoef):
                                                    n = output.get_coef_position(
                                                        source.get_coef_at_position(l)
                                                        + ([] if m == 0 else [m - 1]))
                                                    assert 0 <= n < outcoef
                                                    predicates += 1
                                                    if not b_active or not a_active:
                                                        continue
                                                    key = (abs(group), l, i)
                                                    if group > 0:
                                                        assert key not in assigned
                                                        assigned.add(key)
                                                        writes += 1
                                                    else:
                                                        assert key in assigned
                                                        reads += 1
                    results.append(dict(case=case, vertex=vertex,
                        terminal=terminal, rank=rank, groups=groups,
                        owner_map={",".join(map(str, k)): v
                                   for k, v in ids.items()},
                        writes=writes, reads=reads, predicate_checks=predicates))
    record = dict(complete=True, acceptance_eligible=False, results=results,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(
            (HERE / "static_owner.py").read_bytes()).hexdigest(),
        inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
        scope="Static owner precedes all consumers in J/M/K order. Group members "
              "have identical immutable B bits and the same K/L/I A operand; "
              "terminal groups also fix J. For every LCUT_SIZE 0..4 and "
              "coefficient index, both zero/nonzero guard outcomes are checked. "
              "Runtime bounds/shape guards and distinct caller arrays are "
              "separate obligations, covered by generated-source review and "
              "compiled fixtures. This proves no numerical performance claim.")
    path = HERE / "owner-proof.json"
    assert not path.exists()
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(len(results), sum(x["predicate_checks"] for x in results))


if __name__ == "__main__":
    main()
