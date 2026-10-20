# Corrected ABI 3 representative validation

All four prepared candidate runs completed using the clean `b44d20e7` CLI
exports and the installed ABI 3 archive. Eight targets cover all four backends,
loop-induced amplitudes, mixed split orders and actual CutTools/Ninja/NinjaRS
higher-precision rescue. Each target retains fresh-process parity and two
persistent commands over its complete ten-point stream.

The representative proof combines these eight refreshed candidate targets with
the previously validated legacy bindings. Its validator recomputes source
roles, complete runtime signatures, raw numerical parity and DP/QP coverage.
The four candidate runs preserve the following measured status counts:

| Stream | Backend | Status counts per traversal |
|---|---|---|
| ggH | CutTools | 9 × 211, 1 × 329 |
| ggH | Ninja | 10 × 216 |
| ggH | Collier | 10 × 217 |
| ggH | NinjaRS | 10 × 218 |
| Mixed orders | NinjaRS | 10 × 218 |
| Forced QP | CutTools | 10 × 329 |
| Forced QP | NinjaRS | 10 × 322 |
| Near-singular | Ninja | 10 × 328 |

This is representative validation, not the complete eight-process correctness
or performance gate. Ordinary strict forced-Ninja failures remain preserved in
their earlier evidence; the near-singular stream supplies separate successful
Ninja QP coverage. No rows or failed statuses are excluded from these runs.

Run the read-only verifier from the repository root:

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/policy-v2/representative-retention-abi3-terminal-fixed/verify.py
```

The retention manifest covers the four run directories, their runtime bindings,
the new proof, this documentation and verifier. Already committed legacy
dependencies remain independently bound by the proof.
