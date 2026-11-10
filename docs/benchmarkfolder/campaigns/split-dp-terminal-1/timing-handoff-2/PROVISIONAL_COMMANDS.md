# Execution commands — final source review and runtime admission pending

The following additive entrypoint implements the approved shared-host scope.
It has not prepared private copies or launched measurements yet. Runtime still
requires root review of its exact source and prepared specification.

```bash
source tools/tirust_env.sh
python3 -B tools/tirust_benchmark_provisional.py prepare --label shared-host-1 --cpu 9
python3 -B tools/tirust_benchmark_provisional.py verify --label shared-host-1
# Only after explicit runtime admission of the prepared specification:
python3 -B tools/tirust_benchmark_provisional.py run --label shared-host-1 --execute
```

Preparation verifies the closed ordinary all-eleven numerical campaign and
copies exact compiled exports into private ignored output trees. It performs
no native evaluation, rebuild or relink. Immutable libraries remain shared;
all cards, filters, workers and raw output files are private.

Runtime fixes all eleven chain comparisons first, at 120 seconds each, followed
by the frozen standalone schedule at at most 180 seconds each. It permits three
alternating pairs targeting one second per batch, always preserving the original
full100 learning history and complete full100 timed traversals. The total
allowance is 1,800 seconds including admission waiting, synchronous verification,
initialization, calibration, measurements and cleanup. A ten-second cleanup
reserve is inside each comparison allowance; its final two seconds are reserved
for emergency cleanup, never added after the original deadline. Nested Linux signal deadlines
interrupt synchronous verification as well as child waiting. Linux subreaper
ownership catches detached descendants even when their parent exits between
samples, and guarded PID/session identities limit cleanup to owned processes.
If cleanup cannot be confirmed, no further comparison starts. Run-start and normal
summary publication are also alarm-bounded. If the deadline is exhausted, only
a small closure/failure record may be written afterward, explicitly marked
`administrative-after-deadline` and outside the work cap. That record contains
no new measurement; unfinished samples remain unavailable. No native work,
copying, hashing or replay is admitted during administrative publication.

CPU affinity is nonexclusive; no CPU reservation or quiet-host claim is made.
Backend8/9 processes still serialize on the common numerical lock. Slow or
blocked comparisons remain unavailable, later unattempted rows remain explicit,
and no automatic retry occurs. Raw observations, every actual duration,
initialization interval, numerical comparisons and final accepted precision
counts remain retained. No confidence-bound speed gate or performance pass is
produced; the entire result is performance-acceptance-ineligible.

Retained outputs are under
`docs/benchmarkfolder/campaigns/split-dp-terminal-1/experiments/provisional-timing/shared-host-1/`.
The old `timing-qualification-1` files remain unchanged and unrun. Accurate local
acceptance uses the distinct local measurement launcher described in
`LOCAL_TIMING_TODO.md`.
