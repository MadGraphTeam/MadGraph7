# Final focused ABI3 control audit

This is a source/evidence audit and command plan, not a new test result.
The environment setup in the adjacent README applies to every command below.
Run commands from the repository root.

| Area | Existing evidence | Remaining final-source check |
| --- | --- | --- |
| DP/MP transport and buffer contract | ABI3 production port passed all 16 Rust tests, including genuine binary128 input, 256/384-bit convergence, malformed buffers, lazy compensated state and parameter rebinding; compiled recording Fortran checks DP/QP normalization and extra binary128 bits. | The NLO generator fix does not change bridge source. Bind the bridge hashes to that prior result; the command below can rerun the suite if those files change. The new installed smoke and relocated QP call remain mandatory. |
| Return codes and retry counters | Actual production-template mock flow passed all 24 ordered scenarios, rotations 0/2 and backend 8/8→6. The code-mapping test covers every legacy backend and NinjaRS. | Rerun the three-module focused Python suite, including the previously separate statistics tests for legacy/mixed XML, aggregation and NLO log parsing. Root's final numerical cohort supplies actual rescue statuses. |
| Absent NinjaRS | Compiled cleanup test passes with and without the module; installer tests reject missing modules and old ABI metadata. | Perform a full final-source CLI export with `set ninjars None`, then compile its legacy-Ninja executable without any `mninjars` module or NinjaRS library in that output. The prepared card is `absent-install.mg5`. |
| Internal dependency relocation | ABI3 private pinned install and relocated standalone DP 110/QP 120/stable 218 passed; all copied artifacts were regular files matching the install. | Repeat in this fresh namespace after the committed fix, including the previously failed full NLO compilation and actual pole/soft-collinear checks. |
| Local source overrides | Actual dual-source CLI compile/install/smoke exists for ABI2. ABI3 installer unit tests verify copied source isolation, but are not actual builds. | Run `local-source-install.mg5` through the archived fixed CLI with the shared Symbolica slot reserved. It copies both read-only references into its build stage and publishes only under this namespace. Retain before/after reference manifests, installed ABI3 metadata, lockfile, local revision provenance and actual native/compensated/native/MP smoke log. |

The prior ABI3 production-port proof is sealed under
`docs/benchmarkfolder/experiments/t3-dp-retry-port`; the earlier actual local
override result is in `docs/benchmarkfolder/data/installer-validation.json`.
Prototype actual rotations, loop-induced and mixed controls remain distinct
from final-source proof. Root owns final regenerated process coverage.

The short compiled transport, return-code/statistics and installer checks:

```bash
python -m unittest \
  tests.unit_tests.loop.test_ninjars_fortran \
  tests.unit_tests.various.test_ninjars_install \
  tests.unit_tests.various.test_ninjars_statistics
```

If fresh bridge execution is needed, reserve the shared Symbolica slot first:

```bash
cargo test --manifest-path vendor/ninjars/Cargo.toml --locked --release \
  --features fenv-test-support -- --test-threads=1
```

After the archived pinned installation exists, optional-library absence can be
tested without a Symbolica process:

```bash
python .benchmark-work/policy-v2/install-export-abi3-terminal-fixed/source/bin/madgraph \
  --plain docs/benchmarkfolder/policy-v2/install-export-abi3-terminal-fixed/absent-install.mg5
make -C .benchmark-work/policy-v2/install-export-abi3-terminal-fixed/absent-ninjars/SubProcesses/P0_uux_z \
  -j2 "SHELL=$(command -v bash)" check
```

Before declaring that check passed, require no copied `mninjars.*` or
`libninjars.a` in the output, no `-lninjars` in its makefile definitions, and no
`USE mninjars` or NinjaRS calls in the generated executable's source. This checks
absence in the exported build; it does not rename or alter the main install.

The dual-source override command invokes actual compilation and MP smoke and
therefore requires the shared Symbolica slot:

```bash
python .benchmark-work/policy-v2/install-export-abi3-terminal-fixed/source/bin/madgraph \
  --plain docs/benchmarkfolder/policy-v2/install-export-abi3-terminal-fixed/local-source-install.mg5
```

Its prefix is
`.benchmark-work/policy-v2/install-export-abi3-terminal-fixed/local-source-HEPTools`.
The card explicitly selects the sibling NinjaRS and oneloopmaster clones as
read-only inputs; the installer copies them before Cargo runs. This command
does not promote or replace the main installation.
