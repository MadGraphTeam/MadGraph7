# Private QCD loop kinematics observer — source design

This package prepares an observer for the retained `u u~ > z g g` ordinary/stress/ordinary diagnostic. It has not been compiled or executed. The private Born fallback and automatic checked loop policy remain the previously qualified prototype: its stress result is rejected with code 420 for both NinjaRS and TIRust. This observer must reproduce that result before its new records can be interpreted.

The question is which actual group failures accompany the common pole discrepancy. The corrected target is the soft-leg-4 composite channel `p1-p4`, appearing in 19 source-derived propagator pairs. All observed group visits are retained, including successful groups, repeated visits, CT4 and CT5. The 19 targets guide analysis; they do not filter collection. The original physical input, cards, `OSThres`, repaired QP point, numerator calculations, reduction order, and checked acceptance policy are unchanged.

`prepare.py` inserts marked blocks into four copied sources. Removing those blocks restores every original source byte. It checks the preceding portable evidence, the historical generated source hashes, the original object/archive hashes, and the frozen parameter card before writing. `prepared.json` records these bindings. Earlier source-only preparation passes, including an initially ambiguous insertion anchor, are retained under `source-preparation-attempt*`. None invoked a compiler or native reducer.

The inserted hooks copy existing values through `TRANSFER` into bounded integer buffers. They do not calculate another invariant, square a mass, reevaluate a clamp predicate, or inspect undefined failed outputs. Each hook saves and restores its own IEEE status. No snapshot spans a physical calculation or backend invocation. Caller traps and rounding mode are never changed. Error-text retrieval occurs only after an actual failed public native call and is wrapped in its own local IEEE preservation. Each point is flushed after the host's result and exception flags have already been captured. An independent integer-only `QO_HEALTH` stdout acknowledgement detects a final trace-write failure. Missing records, nonzero health counters, truncation, abandoned scopes, or missing final footers invalidate the observation.

The records preserve:

- point sequence, original point index, backend, actual CT mode, group, squared-order index, N, rank, and monotonically increasing call ordinal;
- original and ordered corners, ordered complex squared masses, exact native shifts and distance matrix, scale, DP/MP model inputs, and original/repaired QP momentum common blocks;
- each existing `DIFFSQ`, initial complex S entry, energy reference, every taken mass-i/mass-j/massless clamp branch in original order, and final S entry;
- original numerator coefficients for both backends and the actual sign-adjusted native numerator for NinjaRS; integer decoding will classify finiteness without evaluating NaNs in the observer;
- public native-status presence and value, available error text, successful native outputs, mapped moments and subsequent contraction/conversion stages, final adapter stability and sticky reduction-failure flag;
- original host coefficients, accuracy, status code and named IEEE flags.

An absent native status means the adapter returned before a native call. A public Fortran status can include transport failure; it is not necessarily a raw Rust status. NinjaRS error text may be empty or stale for some failures and is supplementary to the numeric status and stage record. Failed native output arrays are never read. A sticky global FPE flag does not independently identify the current group's cause.

`routing-modes.json` derives both CT4 and CT5 routing from the original generated corners and actual `MP_SWITCH_ORDER`. It retains literal shift coefficients as well as classifications modulo exact momentum conservation. Direct single-leg shells, complements of single legs, and composite channels are distinct. The first floating-point shift remains the actual summed closure residual; the analysis must not silently set it to zero. Offline exact-rational analysis will compare supplied distances with distances from the recorded Cartesian shifts, then relate each discrepancy to the recorded sequential clamp operations and original/repaired physical channels.

The later, separately authorized qualification will compile only private observer, adapter, host and driver objects, preserve all other objects and libraries, and run each backend in its own process on CPU 11 with a 180-second wall-clock cap and the shared numerical lock. It must compare every original `AMPLITUDE_*`, `BORN_RESCUE_COUNTS`, and `LOOP_NORMALIZATION` record against the frozen unobserved prototype, including code 420 and IEEE flags. Every requested target visit must either have a complete record or be explicitly reported as absent; absence is not success. No timing result is proposed.

Source equivalence and local IEEE snapshots establish the intended observer design, not compiled numerical passivity. Extra calls can affect compiler optimization. Native parity, buffer completeness, and exact output/flag comparisons remain required before any causal conclusion. A small S entry, a clamp, or an S/Cartesian mismatch alone does not establish the amplitude failure's cause, and this package proposes no production threshold or precision change.

Run the portable verifier from the MadGraph repository directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer/verify.py
```
