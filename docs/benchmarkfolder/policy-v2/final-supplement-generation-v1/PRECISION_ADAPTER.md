# Final installed-archive precision adapter

`tools/ninjars_precision_v2.py` and its small C driver evaluate the final installed
public C ABI directly. The driver implements no tensor/reduction mathematics.
The collector is prepared only after final source/install bindings and launches
only following the root's serialized engine handoff.

Input specification fields:

- `report`, `work`: fresh paths inside this repository;
- `installation`: bound final actual-CLI `installation.json` with the existing
  installer proof schema, final source hashes, archive/interface hashes and
  exact CLI card/log hashes;
- `unit_proof`: bound JSON containing `exit_code: 0`, bound `stdout`, `stderr`,
  `build_record`, `executable`, and `bridge_source_sha256`;
- `candidate_runtime_identity_sha256` and full `candidate_runtime_signature`:
  the final measured candidate8 `policy.reducer_signature` descriptor.

The unit build record must contain the same exact `bridge_source_sha256` map
and `executable` binding as the unit proof. That source map must equal the
installation's `archived_bridge_source_sha256`. A complete final bridge unit
suite must pass, including mandatory transport, reset, invalid-domain,
transactionality, buffer, lifecycle and convergence tests. The raw stdout
must contain exactly one `NINJARS_CONVERGENCE` record with exact binary128 input
records and final-source 256/384-bit results. A changed final source, feature
selection or installed archive requires a new proof; earlier private studies
are not silently promoted.

The installed C driver receives all seven retained actual-MadLoop tensor cases,
all twenty-one lanes and native DP / compensated DP / genuine 256-bit modes:
63 calls. One further public MP call receives the raw final-source complex-mass
pentagon input, including numerator information beyond binary64. Its output
must exactly reproduce the same final-source 256-bit ASCII fields and agree
with the independent 384-bit result. Every raw request, status, output, failure
and observed rational split remains retained.

Full finite/single/double Laurent coefficients are the cross-basis oracle, with
the retained 1e-9 scaled fixture tolerance. Generic MP R1 is reported separately
from the original fast-basis split. The independent 1e-31 convergence scale is
the largest of the first six complete Laurent components; a large cancelling
R1 can never loosen physical convergence. The same-basis R1256/384 difference
is reported separately under that stricter physical scale.

The reader recomputes the full measured runtime descriptor identity and compares
the actual installed archive plus module/config/native-link metadata to it;
a stale archive cannot be relabelled with a final identity string. The driver
reports actual loaded shared-library paths with `dl_iterate_phdr`; their hashes
are retained and checked with the raw driver report. Source and build hashes
remain available after temporary ELF/object cleanup.

Preparation uses `prepare SPEC` and compiles the driver without executing it.
Execution uses `execute PREPARED --cpu CPU --execute-numerical`. The final
`result.json` is consumed through `validate`, which recomputes all numerical
conclusions rather than trusting a supplied pass flag. These are correctness
controls, not benchmark timing samples.
