# Original-input EW refusal replay

Both isolated calls reproduce the original EW point-2, loop-group-5 refusal
with unchanged inputs: four propagators, rank three, DP mode 1 and genuine QP
mode 4. The full error messages now retained are:

```text
TIRust DP: Unstable("ill-conditioned Gram matrix for [1, 2, 3]: conditioning 2.0047533625855067e-14")
TIRust QP: Unstable("ill-conditioned Gram matrix for [0, 1, 2, 3]: conditioning 5.22407811366341444758736094099234985181085186192010198472166824e-30")
```

These are the library's diagnostic strings, not independently recomputed
conditioning estimates. Both Fortran statuses are 1, while both native
processes and the orchestration exit with OS code 0. Error transport occurs
immediately after evaluation, before another structural adapter call can
clear it. Complete error frames, an unfilled error buffer, exact working-input
echoes and clean finalization were verified.

There were no input interventions, alternate libraries, retries or coefficient
buffer reads. This evidence identifies the detailed original refusals; it does
not establish their numerical cause, a projection fix, independent tensor
accuracy, amplitude accuracy, stability qualification or performance. The
broader stress contribution failures remain open.

## Exact source and execution scope

The proposal index is
`9384471349f462e7f028161c5b9a1673027698d930e82ac7da0aa6910c08ad61`.
The closed build is
`c195e4de9aa301fb3f17b00e3a89a06fca948352f359bbd212b1d6ac2dad75f3`;
the closed run is
`ec0dcaf70e5921fca8b32041bfde688430cf07c347b5ed31275ecb5845f1ce04`.
Historical source-only and build-not-yet-run flags are preserved verbatim in
the original receipts. The later closures record the separately admitted
build and run, and the root reviews remain separate records.

Each process reconstructs the original 41 DP or 82 QP low-word-first words:
shift vectors, complex squared masses, invariants and scale. The rank-three
capacity is 35 tensor components times four Laurent slots. All thirteen
plans are registered in the exact generated EW order, with one fresh context
and one tensor call per process. Options are all poles, finite enabled,
check-pass false, cache disabled, and a 268435456-byte budget. The original
adapter archive is `0c86349a…` and original TIRust shared library is `eb3d67fd…`.
The six-provider standalone runtime and executable are checked at READY and
DONE; this differs from the full-amplitude provider set in the sibling build
packet. Executables, libraries and other binary artifacts are represented by
recorded identities only, not included here.

The packet retains the 17 source payloads plus their index, three closed build
commands, four closed run commands, both worker specifications/transcripts,
four raw maps, per-worker runtime admissions, original word files,
certificates, protected-state records, terminal closures and root reviews.
Only explicit closed lists and named reviewed artifacts were captured. The
source review and its 19 passing pure tests are retained additively at
`reviews/source-review.json`; sealed originals were not edited.

## Qualified deadline interpretation

The sealed proposal's 180-second total is metadata, not an enforced hard
whole-controller limit. The actual deadline covers 120 seconds of native work
after successful shared-lock acquisition. Each worker's native protocol has a
20-second cap and its owned outer command a 30-second cap. The nominal cleanup
reserve is 60 seconds. Preflight, inherited lock wait (up to 3600 seconds), and
post-stage source/protected-state verification are outside that work deadline.
The root admitted this narrower scope explicitly. The closed receipts record
actual elapsed accountability, all observed owned processes gone, no cleanup
exceptions, protected-state equality and a successful nonblocking lock probe.
These timings are not benchmarks.

## Portable verification

Keep these declared sibling packets beside this directory:

- `../current-kinematic-trace-build-qualification-1`, seal
  `89697913341ecc5fe765fb023cc09bced783eb3eedb85436afe1ca7d4dc6c033`.
- `../current-kinematic-trace-runtime-qualification-1`, seal
  `730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4`.

From the repository directory:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/ew-original-refusal-replay-1/verify.py
```

Only Python's standard library is required. The verifier authenticates local
payloads and declared sibling seals, reconstructs both exact input files from
the original captured traces, checks the generated plan order and input
certificate, and reparses the raw protocol/error frames and map/provider joins.
Only the sealed in-memory `Protocol`/`require` parser is selected from the
retained worker; no retained controller or module initialization runs.
Verification does not compile, load a library, inspect live processes, reopen
external binary paths, evaluate an integral or read a coefficient buffer.
Recorded artifact hashes and ownership checks are historical evidence, not
fresh verification of today's binaries or processes.

`capture.py` documents one-time explicit text retention with deterministic
gzip. It is not a replay or discovery command. Native execution requires a
separate review; portable verification never performs it.
