# Allocation optimization: full MadLoop process decision study

This is a private study of the final pending allocation optimization. Production
is unchanged. The measured callback-only gain in the prior performance package
does not establish a full-process speedup. This directory binds the separately
planned21 QCD3 and7 EW2 adjacent pairs, with100 fixed points per arm and normal
stability/rescue. Root authorized the exclusive MP slot and global quiet window
at 23:38 UTC on 20 October; the user subsequently requested a pause.

The fixed study is now **paused and incomplete**: 12 QCD3 pairs plus the original
arm of zero-based pair 12 completed; the allocation arm was stopped during its
timed RUN. No EW2 sample ran. The completed prefix is historical data and does
not satisfy the 21-pair gate. No performance analysis or speedup claim was made.
Do not resume the interrupted interval as an accepted measurement. Exact stopped
process identities, file descriptors, phase evidence, source/freeze signatures
and all 25 completed worker records are in `pause-20261021.json`. Root must
coordinate a new declared measurement attempt after the user resumes.

`processes-final/preparation.json`, `package-exact.json` and `exact-build/` are
the definitive build chain. Original corrected transport uses archive832fc0d1…;
allocation uses366ae91d…. The complete effective Cargo fingerprint is identical
between builds, including dependencies, profile, flags, compiler and config.
Both arms contain the exact same Fortran wrapper object4831e01d…. Generated
Fortran, cards, points, module, header, native-link metadata, compiler flags and
all other numerical libraries match. The sole production source delta is the
prevalidated minimal allocation tensor. Every worker explicitly uses one thread
for OMP, OpenBLAS, Rayon and MKL, with the effective nonsecret execution variables
retained for every sample.

Preparation history is retained transparently; no numerical sample was launched
in any of these diagnostic attempts:

- The first Cargo attempt used an incorrect dependency-cache location and
  failed offline before compilation. `build-retry-1` corrected it.
- The first process preparation linked a bare Rust archive missing the installed
  Fortran wrapper object; `package-archive.py` added the identical baseline member.
- The second process preparation linked both QCD3 arms but expected one library
  difference; its guard correctly exposed two aliases of the same private archive.
  `processes-v3` handled both aliases.
- Those initial compiled arms used `cargo build`; their effective profile differed
  from the installer's `cargo rustc -- --print native-static-libs` invocation.
  They remain diagnostic only. `exact-build.py` and `package-exact.py` corrected
  the invocation and establish full fingerprint parity for `processes-final`.
- `readiness-drafts` retains pre-sample freezes: a readiness print syntax fix
  and the independent review's explicit Rayon/MKL one-thread environment change.
  A later audit rejected the live public report-tool binding while that tool was
  being edited. No sample ran. The final comparison uses an immutable full source
  snapshot from committed81a4adff3, loading only its exact `compare` function AST.
  All unrelated report/profile imports are excluded. Thirteen differential and
  boundary checks pass against the same historical corrected-installation source.
  The stale public draft cannot be recovered byte for byte; its invalid freeze is
  retained and never used as measurement evidence. The report owner confirmed
  that the comparison function itself was unchanged throughout those edits.

After root authorizes a quiet window, run the frozen private controller with a
selected allowed CPU other than207 and a retained authorization reason:

```
python -B run-processes.py --cpu CPU --authorized-run 'exact root handoff'
python -B analyze-processes.py timings/attempt-N
```

Run from the repository root using full script paths in actual commands. The
controller records every3×3s idle preflight, launches no numerical sample on a
failed preflight, and forbids replacing any started numerical series. Each arm
creates a fresh worker, completes initialization and full-stream warmup, issues
exactly one100-point timed RUN, retains all output, then STOP/BYE and confirmed
exit. All physical coefficients, stability estimates and return codes must be
exactly equal, including raw output bytes, across both arms and all repetitions.
The analyzer reparses all raw files and applies100,000 fixed-seed paired bootstrap
draws. Primary gain requires upper95%endpoint below1; EW2's regression gate
requires upper endpoint at most1.05. No optional samples or exclusions.

All archives, object files, generated process trees, compiled executables and
Cargo caches stay disposable under `.benchmark-work`; no build product belongs
in Git. Source snapshots, controllers, text logs, signatures, every raw numerical
measurement and analysis belong in the final retained evidence seal. These
optimization studies do not replace the final eight-process benchmark against
literal-original MadLoop with Ninja.
