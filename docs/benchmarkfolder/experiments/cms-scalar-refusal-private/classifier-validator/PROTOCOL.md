# Classifier mock qualification addendum

This additive package leaves the frozen classifier, aggregate source index and
compile evidence unchanged. Its Python validator never starts a process. The
Fortran mock has not been executed by this agent.

The executable is `../build/classifier_mock`, linked only from the common
observer, exact extracted permutation tables, classifier mock hosts and fixture
driver. The actual AVH object is absent from this link. There are no runtime
cards, data files, arguments, scalar-library DSOs or Rust contexts. GNU Fortran
and its system runtime dependencies remain necessary. `runtime-static.json`
records executable/interpreter hashes, ELF dependency declarations and candidate
library paths obtained by reading ELF bytes in Python. This is a static record,
not a loader observation; the parent must bind the successful build and verify
the actual loader map before its separately authorized mock run.

Run one worker on CPU 8, with a 15-second total cap including startup, under the
shared numerical lock and reviewed process-group cleanup. No automatic retry.
Use an initially empty private directory and no stdin or executable arguments.
Capture stdout and stderr as separate regular files `stdout.log` and
`stderr.log`; these are the exact directory output closure. There are no
program-created output files. Keep process metadata and loader evidence outside
that directory. Reject preload/audit injection and unpinned loader overrides.
The parent's reviewed controller may instead merge both streams into one
declared capture file; then apply the same exact three-record gate to that
combined stream and record the merged mode. This does not establish a separately
observed empty stderr. The optional `validate_files` helper below implements
the separate-stream envelope only.

Full success requires exit code zero, no timeout, interruption or process error,
empty stderr, exact stream hashes, output closure, and these three ordered
stdout records (list-directed spaces may vary):

```text
CLASSIFIER_MOCK_COMPLETE 8 8
CLASSIFIER_MOCK_COMPLETE 16 8
COMPLETE_CLASSIFIER_MOCK_ONLY
```

Both integer fields are required: kind 8 or 16 and eight completed overloads.
The driver separately asserts at least 33 decimal digits for kind 16 before
either precision's fixtures. Each precision completion occurs after every
hard assertion, including the eight overloads' accepted callback arguments,
refusal validity/callback guards, event counts `1,1,3,3,9,9,13,13`, operand
retention, exact caller-input/output storage, strict threshold neighbors,
complex norms, permutations, bounded overflow, nonfinite publication refusal,
and nonzero values with underflowed metrics. QP also exercises the two retained
physical-transfer/external-residual decimal tokens with promoted historical
binary64 cutoff. These are completed frozen assertions, not independently
printed per-event results. The source-to-executable build binding is essential.

`SUBNORMAL_FIXTURE_UNSUPPORTED <kind>` is explicitly an incomplete qualification,
even if all three completion records follow it. Preserve that evidence and
reject full success. Missing/duplicate/reordered records, incorrect counts,
`MOCK_FAILURE`, `ERROR STOP`, runtime diagnostics, truncation, extra lines or
unexpected files also reject success. Never infer success from exit code alone.

The importable `validate_text(stdout, stderr)` supplies the stream gate.
`validate_files(stdout, stderr, event, run_dir)` additionally checks source
hashes, successful process fields, exact output inventory/hashes and directory
closure. Its event fields are `returncode`, `timed_out`, optional `error` and
`interrupted`, and exact `output_files={name:sha256}`. The parent may use the
text gate after its equivalent reviewed envelope checks. CLI use requires all
four paths, including one process-event JSON object kept outside the run dir.

Passing means limited classifier/mock control-flow qualification only. It does
not establish actual analytic scalar accuracy, full MadLoop rescue, IEEE flag
transparency, trapping-mode compatibility, rounding-mode transparency or
performance. The frozen fixture asserts finite/nonfinite publication behavior
and exact caller storage preservation; it contains no hardware sticky-flag,
trap-mask or rounding-mode assertions. Preserve any failure
without rewriting the frozen fixture or failed evidence.
