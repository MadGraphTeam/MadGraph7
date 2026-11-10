# Full-stream accepted-precision timing diagnostics

This is an additional **TODO for Hua-Sheng's later local timing run**. The
delivered adapter has Python and synthetic Fortran recording tests; it has not
run a real reducer or measured a physical amplitude. It does not change the
formal benchmark worker, campaign registration, local launcher protocol or the
existing report. Its measurements can never satisfy the 1.05 speed criterion.

Source qualification passed 19 focused tests and 20 subtests, including an
actual GNU Fortran driver linked only to recording stubs and executing two
complete synthetic 100-point traversals. That fixture validates the driver
protocol and buffered DP/QP/unchecked observations, not physical integrals.
Reproduce these checks with
`python3 -B -m pytest -q tests/unit_tests/loop/test_tirust_precision_stream.py`.

Use the new `entry-latch-cache-guard-1` source installation and complete
eleven-process, six-lane local numerical qualification from
[the successor handoff](campaigns/entry-latch-cache-guard-1/timing-handoff-1/LOCAL_TIMING_TODO.md).
Those 66 local lanes and precise timings remain pending. Earlier-source
measurements do not qualify the changed generated Fortran. The example
measurement name below is the name used in that handoff. Run these commands
from the repository root of that local checkout. This adapter requires the
unchanged sealed `numerical-prerequisites.json` and independently replays all
eleven six-lane numerical prerequisites before preparation and execution.
An incomplete numerical campaign cannot be replaced by these diagnostics.

Run these diagnostics after the formal timing attempt has finished. Keep its
results and failures. Formal timings continue to use the original worker and
their seven alternating pairs. The new diagnostic copies existing numerical
objects and links a separate executable; it never rebuilds those objects,
installs libraries or changes the original generated process. Use the same
registered CPU; no CPU reservation is requested or assumed.

## One actionable diagnostic

```bash
source tools/tirust_env.sh
python3 -B tools/tirust_benchmark_precision_stream.py --help
python3 -B tools/tirust_benchmark_precision_stream.py prepare \
  --measurement hs-entry-context-local-1 --case qcd_n1_uux_z \
  --lane candidate-9 --label qcd-n1-tirust-stream-1 \
  --loops 1 --batches 7 --timeout 3600
python3 -B tools/tirust_benchmark_precision_stream.py verify \
  --measurement hs-entry-context-local-1 --label qcd-n1-tirust-stream-1
python3 -B tools/tirust_benchmark_precision_stream.py run \
  --measurement hs-entry-context-local-1 --label qcd-n1-tirust-stream-1 --execute
python3 -B tools/tirust_benchmark_precision_stream.py verify \
  --measurement hs-entry-context-local-1 --label qcd-n1-tirust-stream-1
```

`prepare` copies qualified objects and freezes the protocol, selected lane,
point stream, cards, source/runtime identities and fixed batch schedule; it
does not compile or execute amplitudes. `run --execute` explicitly compiles
only the diagnostic driver, links its executable and runs amplitudes. The
per-measurement execution lock prevents overlap with `forward` build,
numerical or formal commands. Chains containing NinjaRS or TIRust also use the
existing shared numerical lock. The tool uses one worker process per diagnostic.

Each batch contains `--loops` complete traversals of the same 100 saved points,
always in the order 1 through 100. The default is seven batches of one traversal.
Choose a larger integer loop count before preparation if longer observations
are needed; there is no post-result calibration or favorable-rerun selection.
Trace buffering is capped at 128 MiB, and the driver explicitly requires 100
points. The timeout applies separately to each driver build command, warmup and
batch; it is not a whole-campaign time limit. A diagnostic can take substantial
time on high-multiplicity amplitudes. Initialization and normal MadLoop warmup
are retained separately.

Allowed lanes are `baseline-1`, `candidate-1`, `candidate-6`, `candidate-7`,
`candidate-8`, `candidate-9`, `baseline-chain` and `candidate-chain`. Chain lanes
use the process's recorded baseline default chain; the candidate prepends
TIRust once. This includes the distinct loop-induced defaults. There is no
arbitrary chain or precision override. A singleton lane compares against its
own sealed numerical lane. A chain uses the sealed baseline CutTools answer as
its coefficient reference and may legitimately accept a different backend or
precision. All status and accuracy differences remain explicit.

## All eleven cases and all five candidate backends

The following explicitly runs 77 separate diagnostics: eleven processes times
the five candidate singleton backends and two full chains. It performs no
automatic retry and stops at the first execution or verification failure.
Only run this after the local installation, all numerical prerequisites and
the formal campaign attempt are complete. Change the fixed loop/batch settings
before preparation if desired.

```bash
python3 -B - <<'PY'
import json
from pathlib import Path
import subprocess
import sys

measurement = "hs-entry-context-local-1"
config = json.loads((Path("docs/benchmarkfolder/local-measurements") /
                     measurement / "report/cases.json").read_text())
tool = "tools/tirust_benchmark_precision_stream.py"
lanes = ["candidate-1", "candidate-6", "candidate-7", "candidate-8",
         "candidate-9", "baseline-chain", "candidate-chain"]
for case in config["cases"]:
    for lane in lanes:
        label = case["id"] + "-" + lane + "-stream-1"
        common = ["--measurement", measurement, "--label", label]
        subprocess.run([sys.executable, "-B", tool, "prepare", *common,
                        "--case", case["id"], "--lane", lane,
                        "--loops", "1", "--batches", "7", "--timeout", "14400"],
                       check=True)
        subprocess.run([sys.executable, "-B", tool, "run", *common, "--execute"],
                       check=True)
        subprocess.run([sys.executable, "-B", tool, "verify", *common], check=True)
PY
```

Do not run the one-case example and then this complete script without excluding
that already recorded case/lane. A repeated measurement/case/lane is rejected
even under a new label. Existing labels and execution markers cannot be reused.
Failures need investigation and remain retained; this tool does not provide a
retry override. Do not create another local measurement merely to hide one.

## What the categories establish

The raw return code is `100*H + 10*T + U`. Accepted backend/precision is decoded
from the full `(H,U)` pair, so TIRust DP `(2,9)` and CutTools QP `(3,9)` are
distinct. The source layout is checked for the per-amplitude
`MP_DONE_ONCE=.FALSE.` reset and the final `T=1/2` assignment. This makes T a
call-local observation, rather than an inherited flag from an earlier point.
T records MadLoop's MP construction and does not enumerate every integral
attempt, retry, MPFR tier or internal reducer algorithm.

| Category | Meaning |
| --- | --- |
| `accepted_dp_no_observed_qp` | Checked, finite accepted DP answer with T=1; no QP observed via the return code. |
| `accepted_dp_after_qp` | Checked, finite accepted DP answer with T=2; QP was observed via the return code. |
| `accepted_qp` | Checked, finite accepted QP alias with T=2. Includes the call's natural failed DP work and rescue. |
| `unchecked` | H=1 with steady T=1/2; never admitted as a checked result. |
| `failed_or_inconsistent` | Failed, unsupported or contradictory return code/precision, missing checked accuracy or timed initialization. |
| `failed_nonfinite` | At least one answer or reported accuracy is nonfinite. |

Every call also reports its actual accepted backend, fallback relative to the
first selected backend and full H/T/U values. Accepted DP is not proof of no
internal attempt or compensated retry. The original formal `native_dp_only`
field retains its existing accepted-first-backend-DP definition.

Where the generated source exposes the exact existing software FPE COMMON
layout, its DP/QP flags are read before and after each call without resetting
them. These are software observations, not hardware IEEE flags. A sticky true
flag is retained and does not itself reclassify an otherwise accepted result.
An unavailable COMMON layout is recorded with `-1` values. No internal retry
counters or per-integral rescue counts are inferred.

## Timing and retained evidence

Each call is bracketed around the same point copying, scale calculation,
`UPDATE_AS_PARAM`, answer clearing and full `SLOOPMATRIX_THRES` call used by the
formal loop. Nothing selects points by their result. There is no extra
classification traversal or cache reset between warmup and batches. Normal
event cache boundaries and helicity/color sums remain inside MadLoop. Every
returned call's status, all answer columns and accuracies are buffered; file I/O
occurs after the timed loop. The raw outer batch time includes in-loop memory
buffering and bookkeeping but excludes file I/O. The separate Python controller
RUN duration includes output file writes and the post-batch clock controls.
These instrumentation costs prevent treating the numbers as formal timings.

The driver stores integer clock rate/max, start/end ticks and outer elapsed
ticks. The verifier checks chronological order, a possible single clock wrap,
exact call count/order, all numerical columns, status histograms and the final
traversal. Zero-tick observations are retained as resolution-limited samples.
Ten thousand back-to-back clock pairs are measured before warmup and after each
batch, with their complete raw ticks, quantization counts and timing summaries.
Clock overhead is **never subtracted**; no samples or outliers are removed.

Evidence is written beneath:

```text
docs/benchmarkfolder/local-measurements/hs-entry-context-local-1/precision-stream/<label>/
```

`batch-NNNN/result.json` contains every call's category/time and category totals,
medians, zero-tick counts, coefficient comparison and separate status/accuracy
differences. `batch-NNNN/raw/stream-values.dat` preserves all returned answers,
including unchecked, failed and nonfinite samples. `ready.json` and each batch's
request/after records preserve the normal filter/resource history. Original
process bytes and diagnostic cards/runtime are checked throughout.

`closed` means all registered diagnostic batches and raw records were retained,
not that numerical or performance acceptance passed. Read
`numerical_diagnostic_passed`, `all_calls_checked` and coefficient comparisons.
The CLI returns nonzero if the closed diagnostic has numerical failures, while
retaining its completed batches and failed samples.
An abrupt worker death may lose the current in-memory buffer; logs and any
partial files remain, and the trace stays incomplete. Missing calls are never
reconstructed or counted as successful. These results need separate review and
report integration; this tool does not overwrite the canonical benchmark PDF.
