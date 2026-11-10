# Conditional V3 continuation: source proposal only

This proposes one build-only attempt using the already retained V2 dependency
artifacts. It creates no new numerical implementation. No build, DSO loading
or tensor evaluation has occurred under this proposal. Root must review and
admit it separately before `continue_build.py build` may run.

The sealed V2 source index is
`9c962ee3f7117eba4399d5db3253533ecf92dbcf138964b18b446c363f4ca1d4`;
its closed timeout index is
`64d35be39e7d7467bd4e2d7a1477abfee2bc6af429a65b5f3616c45810fc2003`.
Of 168 Cargo fingerprint groups, 167 have completed metadata. Only the private
bridge is unfinished. There is no private object/library to link directly:
its complete Rust compilation and ThinLTO/link remain, followed by C compilation
and the Fortran driver link. Completion within a fresh 900 seconds is unproven.

`bindings.json` references the published 1161-file partial-output inventory
instead of duplicating it. Before copying, every content hash is checked.
Copies go only to new `.tirust-build/tirust-private-diagnostics-3/target` and
`cargo-build` directories. Modes and mtimes are preserved; every copy must have
a distinct inode from the original. Existing destinations, symlinks, hardlinks
and unsafe input paths are rejected. No old fingerprint is edited or forged.

The original V2 bridge manifest/config/source paths remain Cargo inputs so
their path-package identities stay unchanged. Cached build scripts contain
absolute V2 paths, notably the GMP archive/header directory. Those exact bytes
are bound by the imported inventory and the complete original-tree guard.
The original tree is input-only: all new outputs are directed to V3. Before
and after every outcome, the controller compares the complete V2 tree's
content, modes, ownership, inode/link identity and timestamps excluding atime.
Any drift is a retained failure, never silently repaired. Source and external
tool/native pins are also checked before/after. These checks detect changes;
they are not a filesystem mount or a race-proof write barrier.

V2 source helpers, the exact tested Rust/C sources, resolved lock and original
link recipe are reused through verified references. No dependency revision,
guard, precision, cache numerical policy, optimization level or scientific
input changes. Jobs remain 2, CPUs 16–17 nonexclusive, each child has a 16 GiB
address-space limit, and the build has a fresh 900-second bound with cleanup
reserve. Existing compiler-wrapper/flags restrictions remain in force.

Cargo uses the same locked/offline/release/features selection, adding only
verbose uncolored logging. If it announces a Dirty/Compiling dependency or a
child command other than the private bridge's rustc invocation, the controller
stops and cleans its owned process group. This is observation followed by
termination; it cannot prevent all instructions of an unexpected Cargo child
from running. Actual Cargo freshness decisions remain authoritative: relocation
of the output directories could still invalidate cached units. Unexpected
rebuilds are retained failures, not permission to expand the attempt.

Imported Rust/native artifacts carry observed byte identities, not a complete
retrospective compiler-environment provenance. V2 reused the system GMP native
cache; the exact resulting archives and headers are now pinned as inputs.
Additional wrapper/support/real-driver and Rust runtime-library pins improve
the tool binding but do not claim a complete observed compiler runtime closure.
The experiment must not be described as rebuilding all native dependencies or
qualifying numerical accuracy or performance.

Successful output would be an unloaded uniquely named private diagnostic DSO,
C object, driver and link map with actual post-link hashes. Static ELF/runtime
review, loader checks and any twelve-call replay require separate admission.
This controller never loads the private DSO or starts a tensor calculation.
Every success/failure retains the transcript, actual child outcomes, imported
manifest and before/after original-tree guards. No automatic retry is allowed.

From the required checkout, pure checks are:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/diagnostics-bridge-continuation-proposal-3/verify.py
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/diagnostics-bridge-continuation-proposal-3/test_continuation.py
```

The new tests use temporary files only and deny child processes/native loading.
The original 18 V2 source/driver/parser tests remain unchanged and can be run
using `../diagnostics-bridge-proposal-2/run_pure.py`. The proposed build entry
point, requiring explicit root admission, is `continue_build.py build` here.
