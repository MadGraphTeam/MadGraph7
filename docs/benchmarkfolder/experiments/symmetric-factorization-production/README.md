# QP factor helper production port

`FortranPolynomialRoutines.write_factorized_wl_updater(rank, vertex, terminal=False)`
emits private process-prefixed helpers for QP ranks2–4 and the FFV1L1_2/FFV1L2_1
orientations. The exporter owns conservative call selection and the preexisting
proof that a terminal result is consumed only on its diagonal. Public updater
and DP routines are unchanged. Unknown domains or excessively long private
names are rejected by the generator, allowing the exporter to retain its old
path.

The generated helpers preserve the isolated prototype's exact source, guards,
fallbacks and changed-association arithmetic. They check the actual stored
derivative-row relations, four-component shapes, binary128 layout/range,
nearest rounding, disabled supported traps and the caller-captured sticky
INEXACT flag. Numerical row equality deliberately ignores signed-zero component
differences; this remains a finite-accuracy contract rather than a universal
internal-bit claim.

`exact-replay-v1` binds the current generator snapshot to all20 independently
validated FFV oracle streams:200 fixtures and250,880 complex outputs across
the two orientations, ranks2–4, full/terminal helpers and actual standalone/NLO
compiler flags. Each regenerated original/helper/constants file equals its
sealed source after normalizing only the private helper declaration for the
unchanged driver. All freshly compiled raw binary128 input/output/flag streams
are byte-identical to their independently exact-Fraction-validated streams.
The source snapshots, flags, logs, data, executable hashes and original oracle
result hashes are retained.

The new focused regression covers408 compiled per-helper cases across the
standalone and `-fno-automatic` NLO builds. It checks accepted dense, sparse and
signed-zero inputs; exact fallback output/input bits and cumulative flags for
false entry, broken row relations, out-of-range values, signaling NaNs,
nonnearest rounding, empty and changed shapes; and fallback under each enabled
IEEE trap. Its separate API test rejects unsupported precision, ranks, vertices
and excessive symbol lengths.

The existing45 polynomial unit and compiled regression tests also pass,
including compact/expanded legacy results, binary128 sparse guards, unsupported
layout fallback, NLO static locals and consumed terminal bits/IEEE flags.
Together with the two new tests,47 unittest methods passed. The independent
source review found no helper-level blocker; the actual rendered source equality
and compiled replay provide the corresponding numerical evidence.

The runtime motivation remains the separate fixed21-pair QCD3 confirmation:
ratio0.96809491 with95% interval[0.95849113,0.98213763], all16,800 sampled printed
coefficients and accuracy/status results unchanged. These helper tests alone
do not validate the exporter's generated call selection or establish final
benchmark acceptance; those require fresh CLI/process controls.

The retained commands were:

```sh
python docs/benchmarkfolder/experiments/symmetric-factorization-production/replay_oracle.py --run-id exact-replay-v1 --cpu 2
taskset -c 2 python -m unittest tests.unit_tests.various.test_q_polynomial_factorized -v
taskset -c 2 python -m unittest tests.unit_tests.various.test_q_polynomial tests.unit_tests.various.test_q_polynomial_compiled -v
```

`verify.py` rehashes the explicitly bound helper package and checks every source
and raw stream against its prior exact-oracle result without compiling or
evaluating amplitudes. `--current-source` additionally requires the working
generator and focused test to equal the retained source snapshots.
