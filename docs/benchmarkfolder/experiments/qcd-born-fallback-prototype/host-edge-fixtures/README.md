# Born host edge fixtures

The revision-2 host policy passes the four compiled scenarios at both `-O0`
and `-O3`. The immutable revision-1 host fails both intended counterexamples
at both compiler levels: repeated failed Born evaluations advance startup and
publish filters, and a selected filtered-zero helicity inherits an accepted H3
status from a previous call. The final wrapper rounding corpus passes 128
cases at each compiler level; deliberately dropping the new QP flag merge or
forcing nearest rounding before conversion is detected.

These are **source-derived policy fixtures with explicit stubs**, not physical
amplitude or native reducer evaluations. The actual Born/native qualification
is retained separately by the prototype owner. There is no performance claim.
That owner's full backend-8/9 replay reports ordinary bit/status/flag identity
but rejects both stress points with code 420: repairing the Born does not
repair the failed QP loop reduction. The prototype therefore remains private;
passing these policy fixtures does not establish successful stress amplitudes.
The short compiler and stub processes use CPU11 and may overlap other
correctness work; no timing values are used as acceptance evidence.

## Host policy

`run_policy.py` pins revision 2 to the original `preparation.json` and host
SHA256, extracts exact source blocks, and verifies the sources remain unchanged
after execution. `qualification-1/results.json` records original line ranges,
block hashes, generated fixture hashes, compiler arguments/version, executable
hashes and individual exits. Generated `revision-2.f` and the revision-1
counterexample are retained beside their raw logs. No production or reviewed
prototype source was edited.

The retained host statements cover rotation backup, initial filter reads, the
Born call, failed-Born return, startup counter, filter publication, bypass
handling, the selected-helicity shortcut, the current-call QP certificate and
final cleanup. Native Born/cache calls are replaced by recording stubs.
`simulated_loop_completion` supplies controlled finite/nonfinite outputs and
accuracy/FPE outcomes before the exact certificate statements; it does not
implement loop arithmetic, stability retries or any reducer. The source's
generated dimensions (5 external particles, 48 helicities, 116 loop groups,
one Born/loop split order) are retained. A synthetic spin-4 setting exercises
the existing rotation-disable/restore branch; it is a policy input, not a
claim about the physical QCD process's spins.

The four positive scenarios are:

- Eight failed Born rescues, exceeding the three-call check cycle, leave NTRY
  at zero, publish no filter files, do not enter the loop hook or publish REF,
  return NaN/H4/T2/U0/accuracy 1 and restore rotation/cache state. The next valid
  Born still enters initialization and filter learning. Momentum, helicity and
  color inputs reach the Born stub unchanged.
- A previous H3 status plus a selected filtered-zero helicity returns H4/U0/
  accuracy 1 after a rescued Born, with no current-call QP certificate or loop
  hook invocation. Its finite Born and zero loop coefficients are retained.
- A simulated finite checked QP result passes; DP-only, nonfinite accuracy,
  nonfinite coefficients, QP FPE and excessive error cannot certify a rescued
  Born. Ordinary DP and intentionally unchecked H1 controls retain their
  respective classifications.
- Successful QP certification restores temporary threshold, rotation and
  bypass-filter state.

The two revision-1 negative controls are expected exits 71 and 72. Their logs
must contain the corresponding named assertion, so an unrelated crash or
compile error does not qualify as detection. All compilations enable bounds
checking. These tests establish only the extracted policy paths, not full
generated COMMON layout, physical aggregation or native reducer correctness.

## Wrapper rounding and IEEE state

`rounding/README.md` describes the separate actual-wrapper fixture, frozen
source/driver inputs, all supported directed rounding modes, sticky flags and
trap masks. The final result is `rounding/run-3/execution.json`. Earlier failed
development attempts remain explicit: run 1 did not compile, and run 2 showed
that removing a final status restore was not a discriminating mutation on this
compiler. The successful original-wrapper observations in run 2 are retained;
that mutation is not claimed as detected.

## Reproduce and verify

From the repository directory:

```sh
source tools/tirust_env.sh
python3 docs/benchmarkfolder/experiments/qcd-born-fallback-prototype/host-edge-fixtures/run_policy.py --label qualification-2
python3 docs/benchmarkfolder/experiments/qcd-born-fallback-prototype/host-edge-fixtures/audit.py
```

The first command writes a **new** evidence directory and places executable and
module files only under ignored `.tirust-build/`. Choose another unused label
for subsequent runs. The inventory verifies the frozen evidence, not a new
run added later; regenerate it explicitly with `audit.py --write` only when
intentionally publishing an updated evidence collection. The rounding fixture
has its own reproduction command and inventory. Generated binary/module work
products live under ignored `.tirust-build/`, outside the report tree. An
explicit `rounding/work-relocation.json` maps the original work paths to their
new locations and unchanged hashes; historical execution logs retain their
true original paths. Generated bytecode was removed. Both audit scripts reject
ELF binaries or bytecode anywhere below their report namespaces.
