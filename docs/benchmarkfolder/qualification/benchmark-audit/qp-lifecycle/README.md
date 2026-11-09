# QP lifecycle audit evidence

This package records source inspection and existing warmup-trace analysis.
It contains no native execution, compilation, optimization, or performance
qualification. Read [proposal.md](proposal.md) for priorities and
[scalar-cache-feasibility.md](scalar-cache-feasibility.md) for the exact API,
key, lifetime, and proposed shadow-profile boundary.

- [evidence.json](evidence.json) binds source hashes, function equivalence to
  the historical bridge, exact dependency revisions, observed policies,
  static generated dimensions, and retained trace hashes/counts.
- [qcd4-active-warmup-snapshot.dat](qcd4-active-warmup-snapshot.dat) preserves
  the earlier 21-call observation. The final completed-attempt entry in
  evidence.json points to the separately retained 28-call QCD4 trace.
- [SHA256SUMS](SHA256SUMS) covers every other file in this directory.

The source paths in evidence.json are relative to the repository root.
Upstream line references in the two proposals use the explicitly hashed
installation-copy source and pinned dependencies, not modified references.
Current and historical QP functions were compared separately from the DP
workspace change. Source inspection does not establish scalar hit rates or
runtime savings. The two explicit-backend initialization timeouts do not
measure the full rescue chain.

From the repository root, verify this package with:

```sh
python - <<'PY'
from pathlib import Path
import hashlib
root = Path('docs/benchmarkfolder/qualification/benchmark-audit/qp-lifecycle')
for row in (root / 'SHA256SUMS').read_text().splitlines():
    expected, relative = row.split('  ', 1)
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
print('QP lifecycle evidence hashes verified')
PY
```
