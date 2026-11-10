# Private native DP rescue cause observer — source design only

This directory prepares an isolated observer for bridge source
`455529b929b725e8fe2d7337ab4a41fa3de9dee1b3a953afe6dc032cf99e5a4c`
and upstream TIRust `854b79dcb7fffcb35e92b4a8a3dd443369b798c4` with the
already pinned full/poles patch. Nothing here is installed, compiled, executed,
or numerically qualified. Production and reference files remain untouched.

The pure `transform.py` accepts only those exact bridge bytes. Its retained
unified diff declares `observer.rs`, observes the result of the existing single
finite-DP `reduce_batch`, records actual QP cache hits, and brackets the two
existing C evaluation entrypoints. The original guard, output-copy condition,
cache clearing, provider, options, numerical inputs, arithmetic, and returned
status stay in place. There is no added evaluation, profiling enablement,
workspace creation, coefficient read, or cache-statistics call. This is a
source claim pending isolated compilation and three-way parity qualification.

The native observations are:

- Per-call context, dimensions, modes, configured input/working precision;
  actual public status and the existing thread-local error copied immediately
  after the original `guard` returns. No failed output buffer is read.
- After a completed finite-DP `reduce_batch`, actual embedder options, routes,
  existing per-lane diagnostic strings, execution counters and attempted
  whole-lane tiers (`bits`, `rows`, `successes`). These are captured before
  `dp_checked` may remove a failed workspace. A native return error remains
  distinct from later adapter/host rejection, which the host hooks supply.
- If `reduce_batch` itself returns `Err`, only the options and failed completion
  are observed. The existing `last_*` accessors can still refer to an earlier
  call, so their stale contents are deliberately not attributed to this one.
- QP entrypoints report configured 113-bit inputs / 256-bit work and native
  status/error only. A QP cache-hit event explicitly denotes reuse, not a new
  numerical evaluation. This is not QP kernel attribution or a QP cache audit.

`Direct` is never labeled proof of pure 53-bit work. Existing public counters
cannot distinguish computed partial-DD subsets from reused rounded DD results
or count uncached/check-pass partial work. Every flush and options record says
`partial_work_observable=false`. This initial protocol qualifies only the
current default options (method choice on, partial and tiers off, default gates
and masters); a different policy permanently marks health nonzero. Attempted
tier and option records remain truthful if a future private variant is studied,
but that variant would require a new protocol, precision-accounting hooks and
qualification. No upstream core hooks are introduced here.

`observer_flags.c` only reads GNU/Linux IEEE flags, rounding mode, trap mask,
and MPFR's public flags (`MPFR_FLAGS_ALL=63`). It does not clear flags, change
rounding/traps, or save/restore state to hide observation effects. Every enabled
observation is bracketed by this read-only snapshot; any change permanently
sets health. Events retain the before/after words. Integer formatting, bit
copies, and already existing diagnostic strings are used; no floating value
is compared or formatted by the journal. This does not establish neutrality
until the later disabled/enabled/plain parity replay succeeds.

The private ABI is declared in `observer.h`:

1. Outside amplitudes, `ml_tirust_observer_open(path,length,enabled)` exclusively
   creates one UTF-8 journal path; `enabled` is 0 or 1. A disabled observer still
   emits flush envelopes, with no observed events. One owning thread/process
   and one open are supported; fresh processes provide independent static state.
2. Before each complete public call,
   `ml_tirust_observer_begin_point(int64_point,int_phase)` labels following
   events. Host hooks call `ml_tirust_observer_event(tag,words,count)` using
   signed 64-bit integer metadata or `TRANSFER` results. Each word is retained
   as its unsigned 16-digit hexadecimal bit pattern. Tags are supplied by the
   separately frozen host protocol; the journal does not reinterpret them.
3. After a complete public call, `ml_tirust_observer_flush()` writes JSONL
   `flush_begin`, event rows, then `flush_end`. It clears buffered events, but
   retains all sequence/call/drop counters and health. Flush must not occur
   during an amplitude. `close()` performs a final flush, writes a distinct
   `close` trailer with cumulative health/drop/event/native/flush counts, and
   closes the file. The trailer is required, not inferred from an empty flush.
4. Check every open/flush/close return (zero only for healthy completion), and
   retain final `ml_tirust_observer_health()` separately in the host's integer
   completion record. An effect during the final flush/close cannot be hidden
   by trusting an earlier written envelope. Failure is never reset or retried.

The per-flush bounds are 32 MiB of retained event payload plus conservative
metadata accounting, 200,000 events, 4,096 words per host event, and 65,536 bytes
per existing diagnostic string. Bounds are diagnostic storage limits, not a
whole-process RSS promise. Any dropped/oversized record fails health; there is
no claim of complete attribution after truncation. File I/O occurs only in the
explicit outside-amplitude entrypoints. Reentry, wrong thread, unsupported
policy, invalid flags or a contained observation panic also fail health.
Rust's infallible allocator can abort on OOM; this is rejected by missing close
and host completion records/nonzero exit, not claimed to be a catchable health
event. The C state reader must be compiled into the isolated bridge and linked
against the same existing MPFR/libm closure; no new numerical dependency is
required. In particular, do not add a host `-lmpfr`: the numerical DSO's MPFR
may be statically linked, and a second instance would read different TLS flags.
Final link-symbol/runtime provenance must show the reader uses the numerical
instance. That provenance is future work, not established here.

Health bits: 1 state-read failure; 2 wrong thread; 4 borrow/reentry; 8 contained
observer panic; 16 observer state change; 32 capacity; 64 call/protocol misuse;
128 file I/O; 256 unqualified numerical policy/engine. The private entrypoints
do not call the production error guard and do not modify `LAST_ERROR`.

Before any use, independently review the exact transformed source and compile
it in isolation, then compare original DSO, observer-disabled DSO and
observer-enabled DSO with identical full point/filter history. Require bitwise
outputs/status/accuracy/input parity, IEEE and MPFR state parity, unchanged
cache and provider behavior, complete ordered traces, healthy bounded flushes,
and exact source/build/runtime bindings. This package authorizes none of those
jobs and makes no performance or acceptance claim.
