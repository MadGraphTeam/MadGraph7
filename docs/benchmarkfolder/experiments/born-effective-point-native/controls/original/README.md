# Exact generated Fortran control fixtures — prepared only

No compilation or native execution has been performed in this package. The
parent owns execution under the shared 600-second CPU14 budget. These fixtures
exercise the actual reviewed generated Fortran routines with controlled
external helper values. They do not simulate those routines in Python and
they do not require reducers, models, HELAS libraries, or physical integrals.

`prepare.py` imports the frozen learning-transaction revision 2 and checked-repair
source generators without changing either. It verifies their manifest-bound
generator identities, generates the exact reviewed bodies, and formats their
fixed-form source with the bound MadGraph `FortranWriter`. `manifest.json`
records inputs, pre-instrumentation body hashes, formatted/raw source hashes,
driver/stub hashes and compiler job/expected-result hashes. `validate_sources.py`
checks those identities, the observer's exact inverse projection, fixed-form
label columns, all source paths, and the 51 planned case records. It is a source
check only; it reports no native test success.

Four Born executables are specified: scalar and split-order aggregates, each
with three and five external legs. Each links two independent prefixes of the
same original aggregate: the exact original entry for finite-path reference,
and the reviewed checked transaction plus its wrappers. The test compares
native DP output bits, independently tabulated finite values, helper call
sequence, all learning state, and wrapper selection. MATRIX returns controlled
finite, NaN, infinity, or large finite contributions; its arithmetic and the
aggregate's actual weighting/summation/normalization run in Fortran.

A marked fixture-only observer ENTRY copies out private `NTRY` and `GOODHEL`
plus `HELRESET`. It contains no assignment into learning state and adds no
reset or setter. Removing its declaration and ENTRY blocks recovers the exact
reviewed checked aggregate bytes and exact original aggregate bytes. Both
hashes and the observer diff are retained. The observer cannot replace a
rollback: the driver takes snapshots before the real checked call and compares
every flavor afterward. Prior flavor-two learning, a pending `HELRESET`, and a
prior `USERHEL=3` deliberately expose incomplete rollback. Three-leg fixtures
exercise the all-flavor `GOODHEL` shortcut; five-leg fixtures retain distinct
flavor histories.

Each Born driver tests finite equality, valid zero, invalid-flavor early zero,
fixed helicity, raw NaN/infinity, an evaluated but excluded helicity, beam-weight
overflow, sum overflow and final normalization overflow. Split fixtures also
test overflow in learning `BUFF` and an unchosen output order. Failure must
return `OK=.FALSE.`, quiet NaNs in every output slot, restored learning/selection,
unchanged input momenta and the expected call boundary. A pre-existing
divide-by-zero sticky flag must survive; rounding and trap settings must remain
unchanged. Rejection fixtures require default disabled traps on entry and do
not disable them themselves. Enabled-trap termination, all IEEE-flag classes,
other rounding modes and optimized builds remain separate qualification work.

The fifth executable links the exact checked repair wrapper and exact
`PREPARE_MP_PS_CHECKED` host body. Original method helpers are explicit controlled
stubs, rather than real kinematic repair algorithms. They record their order,
return specified raw errors and deliberately replace their local workspace
even on failure. The driver tests primary/alternate success, both failures,
nonphysical input, supported method choices and unsupported no-op selection.
It checks raw errors, explicit outcome/adoption, call counts and order,
unchanged DP input, and preservation of the caller's C sentinel when adoption
fails. A finite failed workspace therefore cannot pass by a finite-array test.
Numerical repair correctness itself is outside this control fixture.

`compiler-jobs.json` gives explicit argv arrays and log paths for five isolated
build/run jobs. The parent must create each listed work directory, execute jobs
sequentially, retain exit codes/logs/executable hashes, and enforce its shared
budget. Each compiler is capped at 30 seconds and each executable at 15 seconds
(225 seconds total worst-case caps). All jobs use CPU14, GNU Fortran `-O0`,
`-fcheck=all`, no fast-math, and separate module/output directories. No library
is linked beyond the compiler's ordinary runtime. A failed compiler or runtime
must be retained, not interpreted as a test pass.

`expected.json` lists every expected `PASS <case>` line and known exact values.
Every case must appear once, the process must end with `ALL PASS`, and the exit
code must be zero. Repair diagnostics from the original warning flow are
permitted. Any `FAIL`, duplicate/missing case, compiler failure, timeout or
nonzero exit leaves qualification open.

Source preparation and validation, from the repository directory:

```sh
source tools/tirust_env.sh
python3 -B .tirust-build/born-effective-runtime/control-stub/prepare.py
python3 -B .tirust-build/born-effective-runtime/control-stub/validate_sources.py
```

This package owns only its new ignored `control-stub/` namespace. Production
templates, frozen generators/evidence, reference repositories, model state and
installed libraries are unchanged.
