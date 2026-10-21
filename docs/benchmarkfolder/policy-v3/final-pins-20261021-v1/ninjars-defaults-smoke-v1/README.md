This capsule checks the installed NinjaRS default for ordinary optimized MadLoop
exports and timing/profile selection. Its exact production/test/doc bindings are
in `source-bindings.json`; the source checkout is a reproduction prerequisite.
The five production files retain their original Git modes. The process-check
selection change builds on the separately reviewed v4 CLI implementation.

The fresh selected-source software run passed all 18 default-policy tests.
The subsequent live 37-test run passed 36 and exposed one stale installer fixture
that expected a fatal error for an incomplete optional prefix. That failed log
is retained. After the approved fixture correction, all 19 installer tests
passed. One compiled recording-backend Fortran test also passed, checking that
sole unavailable backend 8 fails actionably and a fallback remains usable.
These are separate observations; the initial failed run is not relabelled.

Actual `bin/madgraph` exports selected `8|6|7|1` for ordinary standalone and NLO
cards. The loop-induced ggH cards kept `7|6|1` and their disabled COLLIER pole
settings. With `set ninjars None`, an ordinary export kept `6|7|1`. Explicitly
requesting backend 8 then raised an actionable `InvalidCmd` before a numerical
check was created. This command-file dispatcher returns process status 0 even
for that caught exception; the retained diagnostic is the negative control.

The generated ordinary u-u~ to Z amplitude was compiled once using the retained
benchmark driver and evaluated at one fixed on-shell point. Its default card
returned 218 (NinjaRS), and an explicit active `MLReductionLib=6` card returned
216 (Ninja). Born and all Laurent coefficients agree at this point. The two
process groups exited successfully and were reaped in the ownership ledger.
The scalar library remains the previously accepted b0d69deb archive, bound by
the copied installation afterimage/config. Native archives and generated build
trees are retained local prerequisites and are deliberately not copied here.
These tiny measurements make no timing or broad numerical-stability claim.

From the repository root, verify the retained text and numerical records with:

```console
python3 -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/ninjars-defaults-smoke-v1/verify.py
```

Add `--live` to require the current checkout's exact source/doc/test bindings.
Historical command cards contain the original absolute working paths. To repeat
the actual exports in a new checkout, substitute a fresh local output directory
and the installed NinjaRS library prefix; never overwrite retained evidence.
The actual native smoke additionally requires gfortran, the recorded libraries,
and the driver rendering recorded by `check_sa.smoke.f` and `compile-result.json`.
