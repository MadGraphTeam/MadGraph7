The exact exporter bytes used by the original CMS source audit have been
recovered and preserved. This closes the distinction between its repository
HEAD and independently recorded working-tree source hash without replacing or
rerunning the original audit.

The original `audit.json` recorded repository HEAD
`5b88a0b3f569d4318c8c7cf55e9db1fb4f1690c3`. At that commit,
`madgraph/loop/loop_exporters.py` has Git blob
`76294168e1e8619841acc187edfbe8317d48f018` and SHA-256 `0541caac…`.
The separately recorded audited source hash is `0020f509…`: it identifies a
dirty working-tree file containing the then-uncommitted DP-terminal exporter
proposal. Repository HEAD alone therefore does not reproduce that audited file.

The retained `dp-terminal-production/initial-proof-hardening` source identity
`c1d6ea3d…json` records the same HEAD, dirty state and exact source hash. Its
adjacent patch has SHA-256
`4763974aac59a2c012d9574646971c79ef1f2441b1fc08f8278f69ea5c62a2a1`.
`capture.py` extracts only the exporter diff, applies it to the recorded Git
blob entirely in memory, checks every hunk/context byte, and requires the
result to match the original audited SHA-256 exactly. This check passed.

| Preserved file | SHA-256 |
|---|---|
| `recorded-head-loop_exporters.py` | `0541caac8650b6ff0225f998bbbe10a3aecebf21c08150d174c441251364d426` |
| `audited-working-tree.patch` | `eb0038582fefd96a33e92f67d310ad9463b685002fe221a81c2165d00c72fbbc` |
| `audited-loop_exporters.py` | `0020f509bdadf5ec3a50de08148e2ac325922f58a92045bc8c50b2374aac4a9b` |

`manifest.json` binds the retained identity and complete original patch, the
Git base, extracted diff, recovered snapshot, recovery script and all four
unchanged original CMS audit files. It also records the distinct current
exporter hash observed during recovery (`d70191ac…`) and its repository HEAD.
That current source was not substituted for the audited bytes. The rejected
initial DP-proof attempt supplies provenance, not numerical qualification of
that proposal; its original status remains unchanged.

All writes are confined to this supplement. Production source, reference
repositories, the frozen CMS data and the original audit remain unchanged.
No compiler or reducer was invoked, and no new CMS numerical or performance
claim is made.

Reproduce or verify the recovery from the repository root:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/cms-cuttools-fix-audit/provenance-supplement/capture.py
```

Existing snapshots and the manifest are checked instead of overwritten. A
later repository HEAD does not alter the preserved recovery-time observation.
