# Phase-space repair instrumentation component

This is source-only preparation. Eleven Python checks pass normally and with
optimized Python. No compiler, loader, Fortran program, reducer or amplitude
was run; no runtime, ABI, numerical or performance qualification is claimed.

`instrument.instrument(text)` accepts only the actual n1 CLI repair source with
SHA256 `d511a2b32b286067c387fc2fc9b1e3d53600f8f7aaf107a53b98e18ec9a53260`.
It changes the names on the first two subroutine headers, preserves every other
original physical source byte and appends public DP/MP wrappers with the exact
original public argument lists and kinds. The private DP body still calls the
public MP wrapper. `reconstruct(text)` recovers and authenticates the complete
original source. The first two entry bodies have no alternate ENTRY or STOP;
normal returns from a private body necessarily pass its wrapper's end hook.
Other scientific routines, warning state, calls, arithmetic and ABIs stay exact.

The module is `host_repair_profile_state` and exports:

```fortran
call repair_profile_reset(mode)
call repair_profile_begin(stage,tick)
call repair_profile_end(stage,tick)
call repair_profile_flush(filename)
call repair_profile_probe(filename)
```

`mode=0` leaves calls disabled, `1` counts and `2` counts plus integer
`SYSTEM_CLOCK` ticks. `tick` is `integer(int64)`. Stage 1 covers the DP wrapper's
whole private body, including conversion and the nested MP wrapper. Stage 2
covers every actual MP body, counting whether DP is active. Stage 3 is reserved
for an empty-bracket probe in the separately compiled `check_sa.o` routine
`repair_empty_probe_loop(repetitions,whole,rate)`.

Flush always emits:

```text
# schema host-repair-profile-v1
# mode <0|1|2>
# clock_rate <positive integer>
# fields stage begin_count end_count ticks nested_count direct_count
1 <begins> <ends> <ticks> <0> <direct>
2 <begins> <ends> <ticks> <nested> <direct>
3 <begins> <ends> <ticks> <0> <direct>
# active 0 0 0
```

Counts are actual entry/exit observations, not presumed calls per amplitude.
Stage-2 ticks include direct MP calls as well as nested calls. Call that interval
the nested MP core only if the actual direct count is zero; otherwise it is the
total MP core. Stage 1 contains stage 2's nested time, so those intervals cannot
be added. Counter and clock lanes must establish their actual matching call
coverage; this component does not assume one DP or MP call per amplitude.

The empty probe writes the existing four-integer layout:
`repetitions clock_rate whole_loop_ticks scoped_ticks`, for 100,000 brackets.
Flush the measured statistics before the probe: it resets the counters, runs
stage 3 in clock mode and restores the previous mode with fresh zero counters.
Probe overhead is descriptive and must not be subtracted to manufacture gains.

The module declares no real/complex values, performs no floating-point
arithmetic, allocates nothing and contains no IEEE environment writes. That
source fact does not prove observed flags or clock-call behavior; native parity,
timing-overhead, genuine call coverage and runtime-map checks remain required.
