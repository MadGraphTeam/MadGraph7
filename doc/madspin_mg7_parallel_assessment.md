# Making `decay_generator = mg7` work with the parallel unweighting

Assessment only. Nothing in `MadSpin/interface_madspin.py` is changed by this
commit; the only file added is this one. Every measurement below was taken by
reading and exercising the shipped code from outside it. The single knob I
turned is `MADSPIN_REFILL_WAIT`, an environment variable the shipped code
already reads (`_worker_refill`, :4880) — set to 1 s instead of its 3600 s
default so a hang test finishes in seconds rather than an hour. No code change
was needed to measure anything.

Tree: `claude/mg7-npy-pool-assessment`, based on
`5f9c2eadcb74bc1aebfce5a657a8266c2a7faa57` (PR #63 merged with `main`), the
post-merge state where both sides exist in one tree. Line numbers refer to
`MadSpin/interface_madspin.py` at that SHA.

---

## Summary

The merge agent's characterisation is **half right, and wrong about which half
is hard**.

* The slicing is not the problem. Two of the five classes it named
  (`_StridedEvents`, `_LimitedEvents`) already work on an `NpyDecayPool`
  unchanged — I ran them. A third (`_ChainedEvents`) is not needed at all,
  because mg7 writes one file. Slicing a memory-mapped `.npy` N ways is a
  zero-copy view costing ~10 µs, and writing the slices out costs 0.12 s for a
  200 000-event pool. The LHE round-robin split it replaces costs 0.083 s for
  50 000. Neither is a bottleneck in either representation.
* The two genuinely hard things are things it did not name: **mg7 pools are not
  reproducible** (`RunCardMG7` has no seed parameter at all), which voids the
  entire reason main's refill machinery is built the way it is; and **a `.npy`
  pool is not self-describing** (no banner, so no cross section), which pushes
  the channel width into the owner→waiter publish contract.
* **The premise that `set nb_core 1` is a working escape hatch is false.** With
  `decay_generator = mg7`, `nb_core = 1` and two or more decaying particles —
  i.e. the PR's own `p p > t t~` benchmark — MadSpin crashes with
  `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x93`. Reproduced below
  against the real methods. The escape named in the warning does not exist for
  the flagship case.
* The guard is also **over-broad**: it demotes mg7 to madevent in spinmode
  `madspin`/`full` too, where there is no parallel unweighting to conflict with.
* **The headline 1.98x was measured with mg7 writing LHE, not `.npy`.** The
  `.npy` pool was a later, separate commit worth 1.05x on top. So "have mg7 emit
  LHE" keeps the entire headline speedup and gives up 4.6%.

Recommendation: **do not teach the refill path `.npy`.** Revert
`generate_events_mg7` to LHE output (~25 lines, a straight revert of one
commit's format choice, back to code that already shipped and was validated),
fix the two guard bugs, and make mg7 the unconditional default that way. Then
diagnose T108 separately — I found a concrete, reproducible worker-killing
defect in that machinery while looking, described in §6.

---

## 1. The interface the refill path actually needs from a pool

A decay pool lives in `evt_decayfile[pdg][channel]` and is consumed through a
deliberately narrow surface. Enumerated exhaustively:

| # | Operation | Where | Does `NpyDecayPool` satisfy it? |
|---|---|---|---|
| 1 | `next(reader)` → `Event` | `_draw_one_decay` :6897 | **Yes**, natively |
| 2 | `reader.cross` | channel choice :6879-6884, `pwidth` :2459-2481 | **Yes**, held as an attribute |
| 3 | `reader.name` | `_reader_paths` :4380 | **Yes** |
| 4 | `reader.paths` (optional) | `_reader_paths`, `_reopen_decay_pool` :6497 | absent → falls back to striding (correct) |
| 5 | `reader.close()` | :4652 | **Yes**, a no-op |
| 6 | `len(reader)` | not used on pools | — |
| 7 | survive `fork` | `Process(args=(…, evt_decayfile, …))` :6633, :7583 | **Yes** — see §2 |
| 8 | stride into a worker's view | `_StridedEvents` :629 | **Yes, unchanged** — verified |
| 9 | cut a slice ~10% short | `_LimitedEvents` :731 | **Yes, unchanged** — verified |
| 10 | count the events in a pool file | `_count_lhe_events` :4521 | **No** — returns 0 silently |
| 11 | rebuild a reader from a path | `_reader_from_paths` :4385 | **No** — raises |
| 12 | split a pool file N ways on disk | `_split_pool_round_robin` :4568 | **No** — raises |
| 13 | open this worker's refill slice | `_open_refill_slice` :4554 | **No** — LHE constructor, and no route to `cross` |
| 14 | name the per-worker slice files | `_refill_pool_paths` :4499 | cosmetic (`.lhe` in the name) |
| 15 | write one file per worker at generation | `_decay_pool_split` :3932 → `nb_unweight_output` | **No mg7 analogue** |

Measured, against a pool with `NpyDecayPool`'s real dtype:

```
_StridedEvents over Npy: 10 events of 20 at stride 2; .cross preserved   OK
_LimitedEvents over Npy: 7 of 20                                        OK
_reader_from_paths(['pool.npy'])   UnicodeDecodeError 0x93 at position 0
_split_pool_round_robin([npy],…)   UnicodeDecodeError 0x93 at position 0
_count_lhe_events('pool.npy')  ->  0        (true count 20 — silent, not an error)
```

Note #10 fails **silently**. `_count_lhe_events` swallows `IOError/OSError` and
returns 0; on a `.npy` it reads binary as text, finds no `<event`, and returns
0. `_open_refill_slice` would then hand the owner `_LimitedEvents(reader, 0)` —
a pool that is empty on its first read. That is the failure mode a naive
"just point it at the npy" patch would produce, and it would look like a refill
loop, not a type error.

So of the fifteen operations, **nine already work, four are one-function format
dispatches, and two are design problems** (§3).

## 2. Fork survival, write-back, and re-readability — the candidates, checked

The brief asked whether the difficulty hides in fork survival, write-back,
re-readability, or RNG/ordering. Checked each rather than assumed:

**Fork survival: not a problem.** `evt_decayfile` is passed as a `Process` arg
under the `fork` start method (:6633, :7583), so it is inherited through
copy-on-write, never pickled. `NpyDecayPool` holds a `numpy.memmap`, which
survives `fork` fine. Verified directly: four children each striding an
*already-primed* inherited pool (chunk buffer non-empty in the parent) each read
their 500 events correctly and independently.

```
fork survival (4 children striding an inherited, primed NpyDecayPool):
  [(0, 500, 3), (1, 500, 3), (2, 500, 3), (3, 500, 3)]
```

**There is one real serialization boundary, and it is not the one you'd
expect.** `_generate_decays` (:4437) forks *per decaying particle* to generate
the pools concurrently, and the resulting readers cannot cross back, so it
marshals them as **paths through a JSON file** (`_reader_paths` →
`_reader_from_paths`, :4423, :4487). This fires whenever there is more than one
decaying particle — **independently of `nb_core`**. That is what breaks the
escape hatch (§4).

**Write-back: not needed.** Workers only ever read pools. The only writer is the
channel owner, under an exclusive `flock` (`_owner_generate` :4703), writing a
*new* generation directory; the pool being read is never mutated.
`_generate_refill_pool` even stashes the old pool aside on the gridpack backend
so a refill only ever adds.

**Re-readability from disk by path: yes, and this is the real constraint.** The
whole owner/waiter protocol is path-based: the owner writes
`Events/ms_refill_<gen>/…`, publishes a generation counter, and every waiter
opens its own slice *by path* (`_refill_pool_path` :4509). A pool must therefore
be reconstructible from a path alone. An LHE file is — its banner carries the
cross section. **A `.npy` file is not**: it has no banner, and `cross` would have
to travel out of band. See §3.

**Ordering: unaffected.** Round-robin position `i, i+N, …` is identical in both
representations. **RNG: affected, badly** — see §3.

## 3. Where the difficulty actually is

### 3a. mg7 pools are not reproducible, and main's refill design exists to make them so

`RunCardMG7.default_setup` (`madgraph/various/banner.py` :6474-6560) declares
**no seed parameter of any kind**, and `madgraph/iolibs/template_files/mg7/run_card.toml`
contains no seed line. The event combination seeds `std::mt19937` from
`std::random_device` per thread (`madspace/src/driver/event_generator.cpp`,
`combine_to_lhe`). Commit `f96f3b653` says so outright: *"the mg7 run card has no
seed parameter, so decay pools are not reproducible run to run."*

Now read what main's refill machinery is *for*. `_channel_owner` (:4539):

> Fixing the generator this way is what makes the refilled pool (and thus the
> decayed sample) reproducible: contrast the old "whoever ran out first
> generates", whose winner was a lock race.

and `_owner_generate` (:4703) computes a deterministic `det_seed` from
`(seed_base, pdg, decay_file_nb, gen)` and assigns `self.seed` and
`self.options['seed']` so that "gen N gets the SAME seed no matter who generates
it".

**The mg7 launcher reads neither.** `generate_events_mg7` (:2701) writes
`generation.events` and `run.output_format` into the TOML and nothing else. So
under mg7 the entire fixed-owner + deterministic-seed apparatus becomes
decoration: still correct, but protecting a property that no longer holds. The
fail-safe's warning — *"Seed reproducibility is NOT guaranteed for this refill"* —
becomes unconditionally true for every refill, owner or not.

This is the deepest problem and **neither option fixes it**. It is a
cross-cutting mg7 change (a seed in `RunCardMG7`, plumbed through madspace's C++
RNG), not a MadSpin one. It is worth noting that it equally undermines the
*current* shipped behaviour for `nb_core = 1`.

### 3b. A `.npy` pool is not self-describing

`cross` is the channel's partial width, and it drives channel selection
(:6879-6884) and the branching ratio (:2459-2481). From an LHE pool it comes
free, out of the `<init>` block, at every reopen. From a `.npy` pool it comes
from the run's `info.json` — which lives in a run directory mg7 named itself,
and which the *waiter* (a different process, opening a slice by path) has no
route to.

So option A cannot be a pure I/O dispatch. The width has to be carried:
* through the `_generate_decays` JSON marshalling (feasible — `channel_widths` is
  already in that JSON, :4425);
* into the owner→waiter refill contract as a new sidecar file written before
  `_publish_gen`, since `_open_refill_slice` reconstructs from a path alone.

That second one **extends the publish protocol** — precisely the surface a
publish/open race lives on, and precisely where PR #378 ("publish a refill
generation only once its files exist") already had to fix one bug.

### 3c. mg7 has no `nb_unweight_output`

`_decay_pool_split` (:3932) makes madevent write one pool file per worker so no
worker parses another's events. mg7 writes one file. Not hard — either split once
in the parent (0.12 s for 200k events) or give `NpyDecayPool` `offset`/`stride`
arguments (a memmap slice, ~10 µs, zero-copy) — but it is work that only exists
because of the format.

## 4. Two bugs found while assessing

### 4a. `set nb_core 1` — the documented escape hatch — crashes

The warning at :3113 tells the user to `set nb_core 1` to keep mg7. That does not
work when there is more than one decaying particle, because `_generate_decays`
forks and marshals paths regardless of `nb_core`. Reproduced with the real
`_generate_decays` / `_generate_decay_entry` / `_reader_paths` /
`_reader_from_paths` methods and a stub that returns an `NpyDecayPool`:

```
escape hatch: nb_core = 1 -> mg7 generator stays selected
two decaying particles -> _generate_decays takes the forked branch
*** _generate_decays RAISES ***
'utf-8' codec can't decode byte 0x93 in position 0: invalid start byte
```

`0x93` is the second byte of the `.npy` magic `\x93NUMPY`. This is
spinmode `PA`/`onshell` only (the `run_onshell` path); spinmode
`madspin`/`full` goes through `run_bridge`, generates serially and is unaffected.
But `PA` is the mode the PR benchmarks, and `p p > t t~` with both tops decayed
is exactly two decaying particles.

**So on this tree, `decay_generator = mg7` in its own headline configuration is
not merely demoted — with the demotion disabled it is broken.** Any option must
fix `_reader_from_paths`, including "just default `nb_core = 1`".

### 4b. The guard demotes mg7 where there is nothing to conflict with

There are exactly three `fork` sites (:4465, :6624, :7577). Two are the parallel
drivers; both are reachable only from `run_onshell`, i.e. spinmode `PA`/`onshell`
(:3985). `_resolve_nb_core` drives parallelism nowhere else. But the guard at
:3101 consults `_resolve_nb_core()` unconditionally, so a user on a 16-core box
running the **default** spinmode `madspin` — which never forks a worker and never
touches the refill machinery — also gets silently moved off mg7. That demotion
buys nothing. Gating the guard on `self.options['spinmode'] in ('PA','onshell')`
is a one-line, zero-risk recovery of mg7 for the default spinmode.

## 5. What actually carries the speedup

This is the decisive number, and it settles the question.

From the PR's own back-to-back measurements (`p p > t t~`, both tops fully
decayed, 10 000 events, 18 cores, same seed):

| commit | change | total | vs prev |
|---|---|---|---|
| baseline | madevent decay pools | 79.46 s | — |
| `f96f3b653` | **mg7 generates the pools, writing LHE** | 40.07 s | **1.98x** |
| `a1afe9dda` | parallel matrix-element compile | 33.02 s | 1.71x |
| `2ee053f67` | **LHE pool → `.npy` pool** | 32.15 → 30.73 s | **1.05x** |
| `543c4dcba` | gzip level 6, stop repacking input | (100k) 4.20x on gzip | — |
| `8faeecc8d` | hoist per-trial card/pdir lookups | 1.03x on the loop | — |
| `e2e6f4c9b` | de-numpy the small density ops | 1.04x | — |
| `3b62c592d` | batch the max-weight densities | 1.14x on the scan | — |

**The 1.98x headline was measured with `output_format = 'lhe'`.** The `.npy`
representation arrived later, as a separate commit, and was worth 1.05x. Confirm
it from the diff: `2ee053f67` replaces the line
`run_card['run']['output_format'] = 'lhe'` with `'lhe_npy'`.

So the attribution is:

* **mg7 generating the pools** — the whole 1.98x. Independent of representation.
  The mechanism is refill collapse (23.74 s → 0.38 s): a madevent refill costs
  ~12 s of fixed survey/refine/combine overhead whatever its size, and mg7
  returns exactly the number of events asked for instead of madevent's 0.8x
  undershoot, so pools run dry far less often.
* **the `.npy` pool** — 1.05x, i.e. ~1.4 s of a 32 s run.
* **the other five commits** — survive either way; none reads a pool file.

I re-measured the per-event read cost independently, on realistic 5-particle
events with non-trivial floats (60 000 events, `.npy` 33.1 MB, LHE 48.5 MB):

```
npy: 10.25 µs/ev    lhe: 14.06 µs/ev    ratio 1.37x    delta 3.81 µs/ev
```

The PR measured 10.0 vs 16.6 µs. Same order; the prize is small either way
because pool reading is only ~7% of a run (`2ee053f67` corrects an earlier
double-counted 17.2% figure down to an honest 6.8%).

**Having mg7 emit LHE gives up 4.6% of wall time and keeps 100% of the headline
speedup.**

## 6. T108

**I could not find the T108 record.** It is not in this repo, not in the git
history of either side, and a full-text search across session transcripts
returns nothing. So I cannot confirm the brief's description of it. What follows
is a defect I found *by looking at the machinery T108 points at*; it should be
checked against the T108 notes before being assumed to be the same thing.

**It is not a race.** It is a deterministic ownership/liveness mismatch, and it
kills a worker exactly under heavy refill load.

In `_scan_maxwgt_parallel` (:7550) the `nb_core` handed to every worker as
`_shard_nb_core` is deliberately kept at the *pool-addressing* count, while
trailing empty ranges are filtered out so **fewer workers are forked than
`nb_core`**. The comment at :7569 reasons about this — but only about pool
*files*, not about *ownership*. `_channel_owner` (:4539) returns
`idx % _shard_nb_core`, so it can name a worker id that was never forked.

A live worker that runs such a channel dry then enters `_worker_refill`'s wait
loop (:4855) and neither fail-safe fires:

* `_read_worker_status(owner)` returns `None` (no status file was ever written),
  and the done-check tests `== ('D',)`, so it does not match;
* `_wait_cycle_to_self(owner)` sees `not st` and returns `False`, so the
  deadlock-cycle path does not fire either.

The worker therefore blocks for the full `MADSPIN_REFILL_WAIT` — **3600 s by
default** — and then raises, which fails the whole run at the parent's
"worker %s failed" check. Reproduced against the real methods (timeout shortened
to 1 s via the existing env var):

```
nb_core (pool addressing) = 8 | workers actually forked = 3
owners assigned to channels: [0, 1, 2, 3, 4, 5, 6, 7]
channels owned by a worker that was NEVER FORKED: [3, 4, 5, 6, 7]
  _read_worker_status(3) -> None     (the D fail-safe tests == ("D",); does not fire)
  _wait_cycle_to_self(3) -> False    (the deadlock fail-safe does not fire either)
  live worker 0 blocked 1.2s then DIED:
    MadSpin worker 0 waited 1s for owner worker 7 to generate gen 1 of
    channel pdg 6 (decay file 7) and gave up.
```

It is not exotic. `Nevents_for_max_weight` defaults to **75** (:1114), and the
scan clamps `nb_core` to the probe count before splitting. Phantom owners for a
75-event probe:

| cores | forked | phantom owners |
|---|---|---|
| 8 | 8 | 0 |
| 16 | 15 | 1 |
| 18 | 15 | 3 |
| 32 | 25 | 7 |

So on a typical machine the max-weight scan **always** has phantom-owned
channels; the hang fires as soon as a live worker actually exhausts one, which
is precisely "heavy refill load".

`_run_onshell_parallel` (:6593) is **not** affected: it reassigns
`nb_core = len(shard_paths)` before `_clear_worker_status` and forks every id in
range, so every owner is live. The bug is specific to the max-weight scan.

Likely one-line fix: make `_channel_owner` deal channels over the *live* worker
count, or have the parent write a `'D'` status for every un-forked id in
`_clear_worker_status`. Either wants a test, and neither should be bundled with
a format change.

## 7. The options, costed

### Option A — teach the refill path `.npy` pools

Changes: `_reader_from_paths` (+ thread `cross` through the `_generate_decays`
JSON), `_count_lhe_events` → format dispatch, `_split_pool_round_robin` → npy
variant, `_open_refill_slice` → `NpyDecayPool` + a new width sidecar in the
publish contract, `_refill_pool_paths` naming, `_reopen_decay_pool` npy fast
path, `NpyDecayPool` gains `offset`/`stride`, guard removed.

* **Size**: ~80-120 lines across 7 functions plus a protocol extension.
* **Risk**: **high**, and not because of the line count. It lands on the refill
  machinery, extends the owner→waiter publish contract, and does so on top of an
  undiagnosed worker-killing defect in that same machinery (§6). It also does not
  fix §3a — the pools stay non-reproducible, so main's fixed-owner design is
  carried but no longer buys what it was written to buy.
* **Payoff**: 1.05x over Option B.
* **Verdict**: the mechanics are more tractable than "new engineering" suggests,
  but the cost/benefit is indefensible: the highest-risk option for the smallest
  prize. **No.**

### Option B — have mg7 emit LHE

Changes: in `generate_events_mg7` (:2701), `output_format` back to `'lhe'`, read
`events.lhe` instead of `events.npy`, take the width from the `<init>` block
instead of `info.json`. Plus the two guard fixes from §4.

This is a **straight revert of `2ee053f67`'s format choice** — back to code that
already shipped in `f96f3b653`, was benchmarked at 1.98x, and had its physics
validated (partial widths agreeing with MadEvent at 0.12σ and 0.04σ). The mg7
LHE writer supplies the width: `build_lhe_meta`
(`madgraph/iolibs/template_files/mg7/madevent.py` :1067-1075) sets
`processes=[ms.LHEProcess(xsec, err, xsec, 1)]`, so `EventFile(…).cross` reads it.

Crucially, **the refill machinery does not change by one line.** It already has a
first-class path for "the backend wrote a single LHE file" — that is the gridpack
case `_generate_refill_pool` was written for: `sources != targets` →
`_materialise_refill_pool` → `_split_pool_round_robin`. mg7 lands in exactly that
branch. `_reopen_decay_pool` handles the single-file initial pool via the
`_StridedEvents` fallback; optionally split it once in the parent
(0.083 s / 50k events, existing tested code) to get the per-worker fast path and
the owner-undersize behaviour back.

* **Size**: ~25 lines reverted + ~4 lines of guard fix; optionally ~10 more for
  the parent-side split.
* **Risk**: **low**. Reverts to a validated state; touches no refill code; does
  not intersect T108.
* **Cost**: 4.6% wall (32.15 vs 30.73 s at 10k). Plus disk: LHE is ~1.5x the
  bytes.
* **Verdict**: **yes.**

### Option C — default `nb_core = 1` under mg7

* **Size**: small, *but it does not work as stated* — §4a means it must also fix
  `_reader_from_paths`, and even then only for spinmode `PA`/`onshell`.
* **Risk**: low to implement, **bad as a policy**: it silently turns off the
  parallel unweighting, which is main's headline feature, to turn on PR #63's. It
  trades an 18-way parallel unweight for a 1.05x pool read. On 18 cores that is a
  large net loss.
* **Verdict**: **no.** It resolves the conflict by picking the wrong side.

### Option D — do nothing

* Leaves the PR's headline feature off on every multicore machine, and leaves the
  §4a crash latent behind a warning that tells users to walk into it.
* **Verdict**: not acceptable, if only because of §4a.

## 8. Recommendation

1. **Fix §4a first, on its own.** `_reader_from_paths` on a `.npy` is a hard
   crash reachable today via the documented escape hatch. Three lines, plus a
   test. This is a bug fix regardless of what else is decided.
2. **Take Option B.** Revert `generate_events_mg7` to LHE output; make
   `decay_generator = mg7` the unconditional default; delete the `nb_core`
   demotion and the warning. Keep `madevent` as the fallback and keep it forced
   for gridpack (`ms_dir`), which genuinely needs `run.sh`. Cost: 4.6%. Benefit:
   the feature is on by default for everyone, and the refill machinery is not
   touched.
3. **Fix §4b in the same change** — gate any remaining backend guard on
   spinmode, so the default spinmode is never demoted for a conflict it cannot
   have.
4. **Diagnose T108 separately, and before anyone touches the refill machinery
   for any reason.** §6 is a concrete, reproducible, deterministic defect of
   exactly the reported shape and should be checked against the T108 notes
   first — it may be T108, or it may be a second bug in the same place. Either
   way the machinery has a live worker-killer in it and no format work should
   land on top of that.
5. **Raise the mg7 seed question (§3a) as its own item.** Non-reproducible decay
   pools already affect shipped `nb_core = 1` behaviour. It is an mg7/madspace
   change, not a MadSpin one, and it is a precondition for ever making Option A
   worthwhile.

Option A becomes worth revisiting only if all three of T108, the mg7 seed, and
the self-describing-pool problem are resolved first — at which point it is a
1.05x optimisation and can be judged on that basis.

## Appendix — measurements taken for this assessment

Machine: darwin arm64, python 3.14 (`mg-3.14`). All figures from the shipped
code at `5f9c2eadc`, exercised from outside.

```
memmap[3::8]                     zero-copy memmap view, 0.0093 ms
npy round-robin split 8 ways     0.121 s   (200 000 events, 68.8 MB)
_split_pool_round_robin (LHE)    0.083 s   (50 000 events, 8 ways)
npy read, realistic 5-particle   10.25 µs/ev
LHE read, same events            14.06 µs/ev   (1.37x, delta 3.81 µs/ev)
_StridedEvents  over NpyDecayPool   works unchanged
_LimitedEvents  over NpyDecayPool   works unchanged
NpyDecayPool across fork()          works (4 children, disjoint stripes)
_reader_from_paths(['x.npy'])       UnicodeDecodeError
_split_pool_round_robin([npy])      UnicodeDecodeError
_count_lhe_events('x.npy')          0, silently (true 20)
_generate_decays, mg7, nb_core=1,
  2 decaying particles              UnicodeDecodeError (§4a)
phantom-owner refill hang           worker dies after MADSPIN_REFILL_WAIT (§6)
```

Run-level timings in §5 are the PR's own back-to-back measurements from its
commit messages, not mine; they are on an 18-core machine with the real
`p p > t t~` benchmark, which I did not re-run. The per-event read costs above
are mine and corroborate their order of magnitude.
