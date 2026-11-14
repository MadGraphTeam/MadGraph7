# Reported tensor completeness for the four TIRust QP points

The retained four-point TIRust fixed-QP result has been replayed with immediate
observations of the generated process's `FPE_IN_DP_REDUCTION` and
`FPE_IN_QP_REDUCTION` common-block flags. All nine calls (one bootstrap, four
warmup calls, and four final calls) reported both flags false. The final
numerical rows and `values.dat` bytes are identical to the earlier result.
The original point IDs are 39, 60, 91 and 93.

This closes the specific concern that an unchecked status 120 might conceal a
reported tensor-reduction failure and a zero contribution to the sum for these
four TIRust evaluations. The flags are observed without being reset. False
flags are not an independent accuracy test, and the fixed-QP result remains
unchecked. The earlier independent-backend comparisons supply the separate
numerical evidence. No amplitude timing or performance claim comes from this
supplement.

Only a private driver is recompiled and relinked. Its changes add the exact
generated common-block declaration and post-call output, with no numerical
control-flow or coefficient changes. `attempt-1/compilation.json` binds the
original and new driver sources, unchanged numerical object hashes, unchanged
link inputs, compiler policy and runtime libraries. The exact old cards, saved
decimal points, original result, driver patch derivation and all raw calls are
retained. The original successful result is unchanged.

Independent audit found that the initial supplement's protocol list omitted
the current numerical parser and two imported helper files. The untouched
attempt is preserved with that limitation. `verification-1/verification.json`
records a separate complete parser snapshot and successful reparse of all raw
flags, coefficients, exact inputs, and driver-only provenance. It does not
retroactively assert the omitted original execution hashes. The runner now
captures and checks these dependencies for future attempts.

Run from the MadGraph root with `source tools/tirust_env.sh`, then
`python3 docs/benchmarkfolder/experiments/qcd-n3-adjudication/fpe-completeness/run.py`.
The runner refuses to overwrite the existing attempt or private clone; a new
attempt path is required for another replay. Its native worker has a 180-second
timeout per phase and uses the shared numerical lock. The recorded timing
includes diagnostic I/O and is ineligible for performance acceptance.
