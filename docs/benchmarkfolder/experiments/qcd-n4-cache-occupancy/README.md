# QCD n4 cache occupancy — private diagnostic proposal

This experiment asks whether the unchanged default cache budget refuses insertions
at registered QCD n4 point 1 (`ordinary-0000`). It does not change cache policy,
numerical precision, stability thresholds, or production sources. No execution or
performance result is implied by this source package.

With the installed `poles-only` build, the 256 MiB total setting gives the full-DP
store a 64 MiB ceiling, the pole store another 64 MiB, and the whole-tensor QP
cache 128 MiB. These are estimated retained-payload ceilings, not reserved memory
or resident-memory measurements. Insertion refusal returns the computed value;
it is not itself a reduction failure. Unused pole capacity is not borrowed by
the full store. Whether the limit matters for this point remains unmeasured.

The fixed input consists of the exact six momentum rows for point 1 in the
registered `split-dp-terminal-1` stream. Each fresh process makes exactly two
public calls: bootstrap 0, then call 1 at the same point, followed by ordinary
MadLoop exit. There is no adaptive warm-up, further point, timing loop, manual
cache clear, helicity assumption, or model regeneration. Internal initialization
and stability calls remain whatever the unchanged host performs. The ordinary
full-amplitude generic TIR cache is absent in this export; its clear routine is a dummy.
The observer counts actual native tensor visits, not guessed helicity reuse.

Three serial arms are required: original adapter with the common diagnostic
driver, instrumented adapter with observation disabled, and the same instrumented
executable with observation enabled. All use the same installed `eb3d67fd…` DSO,
original numerical objects, model/card resources, and compiler flags. The only
numerical object replaced in the observed arm is `TIRUST_interface.o`; removing
the tagged observation insertions must recover its exact original source bytes.
The common driver/module are private additions in both arms.

Every arm retains exact input-before/input-after, answer and accuracy words, public return codes,
and entry/exit IEEE flags, trap modes and rounding mode. These must match exactly.
The observer uses a fixed integer buffer and flushes only after both calls and
normal exit. It records actual precision, N/rank, direction, cache/check flags,
return status, generation, and aggregate/QP cache counters. Every query must
preserve the observed IEEE state; a query failure or journal overflow invalidates
qualification. Disabled observation makes no statistics query.

The public statistics routines clear error text and can lazily create a context.
To avoid these changes in active error paths, sampling occurs only after a
successful tensor evaluation, and immediately before/after the existing clear
call. The context-generation getter prevents observer-created contexts. Failed
evaluation status is recorded without querying statistics. The original clear
call has no status output and remains byte-identical; its status is recorded as
unknown, and the sampled post-clear counters must be empty. A post-clear query
can replace an unconsumed clear error; error-text parity is not established by
the public numeric/IEEE parity checks.

There is a material limit: failed DP evaluation clears its store before returning,
and QP transport failure can clear all stores. Activity created and erased inside
that call is invisible to this Fortran-only observer. Reported peaks are maxima
at the sampled boundaries and lower bounds on internal peaks. The nine aggregate
counters sum full/pole stores and add QP contributions to their final five slots.
Subtracting the separate QP counters gives full-DP activity only because this
run is checked to use full finite evaluation and the pole store is unused.
Counter maxima are per-clear epoch values, not total amplitudes or unique
integrals. The DP label identifies the 53-bit public interface; its store can
also contain internal DD results, so these counters do not establish exclusively
binary64 internal arithmetic. Independent direction checks intentionally bypass
numeric caching.

`prepare.py` binds the registered preparation, source/object/card/runtime
identities and selects point 1 without reformatting its decimals. `transform.py`
provides the insertion-only source derivation. `analyze.py` reparses retained text
and rejects missing parity, query failure, state changes or incomplete journals;
its tests use synthetic text only. The proposed build is CPU 12, one job, at most
180 seconds, with private outputs and no Rust compilation. A later separately
authorized original/off/on replay must use the shared numerical lock. This is
targeted cache profiling, never whole-amplitude performance acceptance. Any
future larger-budget contrast requires its own recorded card change and results.
