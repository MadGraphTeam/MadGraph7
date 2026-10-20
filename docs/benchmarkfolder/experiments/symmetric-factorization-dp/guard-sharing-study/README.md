# Source-only follow-up to the negative DP process timing

This census proposes no production change and performs no new compilation or
amplitude evaluation. The fixed 21-pair factor/original process result remains
1.01355 with 95% interval [0.99660, 1.03095], with the same two rescues. Neither
guard sharing nor producer fusion has been implemented or measured here.

`analysis.json` binds the actual generated QCD3/QCD4 sources and screened call
graph. `consumer-census.json` binds the adjacent producer/consumer arguments.
The v1 script/result/log are retained: tightening the lexical CALL/IEEE/EXTERNAL
screen did not change counts. These are closed-source syntactic screens, not a
general proof of Fortran or compiler runtime side effects.

| Source | Selected rank2–4 FFV updates | Strict regions | Screened ALOHA regions | If CREATE_LOOP_COEFS were proved safe |
|---|---:|---:|---:|---:|
| QCD3 | 136 | 110 | 110 | 1 |
| QCD4 | 1038 | 830 | 806 | 3 |

Strict regions allow the two selected FFV producers and ordinary public
arithmetic updaters. Every other call or control-flow barrier invalidates the
state. Closing the screened ALOHA graph helps QCD4 slightly. The last column is
conditional: CREATE_LOOP_COEFS calls MERGE_WL and split-order index functions;
the latter contain error WRITE paths. Their generated index/data domains and
all reachable effects still require a proof. It must not be used as an already
certified three-batch cache.

The retained existing-object disassembly shows five IEEE_SUPPORT_HALTING calls,
five IEEE_GET_HALTING_MODE calls, IEEE_GET_ROUNDING_MODE and the round-type
comparison on an admitted helper path. Those 12 environment runtime calls are
not folded away. Applied to the 326,400 selected calls in the existing QCD3
trace this is 3,916,800 environment runtime calls, plus 326,400 caller
IEEE_GET_FLAG queries. This counts calls, not their elapsed-time contribution.
The earlier kernel guard control passed a literal true entry flag, so its
13–26% overhead did not measure that caller flag query.

Other guard work remains substantial: with LCUT=4, ranks2/3/4 inspect respectively
640/1280/2400 real components of A and B, plus 48 complex derivative-row equality
relations (96 real-component comparisons). An environment-only optimization
must retain these per-call bounds and row checks. The strict census reduces
query groups by only 19.1% for QCD3 and 20.0% for QCD4 (22.4% with the wider
screen); a large saving needs the stronger call-effect proof.

Any future shared state should be local to the existing coefficient batch,
explicitly initialized on every entry even under -fno-automatic. Query the
actual caller INEXACT before the first selected updater; when false, use the
original fallback and poll at subsequent targets. A true observation may be
retained only over proved flag-monotone calls. Query the environment lazily
after entry-INEXACT and shape eligibility; unsupported modes always fall back.
Unknown calls, setters, I/O, unproved branches/joins, label targets and reentry
invalidate the state. Never force INEXACT true to enable the optimization.

GNU procedures using IEEE modules save/clear/restore flags. A callee-local
query is therefore not interchangeable with the explicit caller-entry value.
Source absence of a flag setter alone does not prove monotonicity across such
a boundary. The compiler/runtime restoration behavior needs compiled sticky
flag and trap tests, and removing the helper's IEEE procedure boundary would
be a separate change. A guard-only attribution experiment would be the next
useful measurement if authorized, before another factor timing.

## Adjacent producer fusion

There is a stronger but less mature opportunity. The census finds an exact
COEFS producer followed immediately by its updater, literal shape4, and
distinct WL input/output slot indices in every selected pair. Before the next
known vertex overwrite, 135/136 QCD3 and 1036/1038 QCD4 pairs have exactly one
consumer. The remaining one/two stop at control-flow tails and are deliberately
unproved; no extra consumer was found. This is a syntactic alias/use result,
not a stored-arithmetic identity proof. Producer PL input/output aliasing has
not been independently certified.

Each selected vertex assigns exactly 80 complex B entries: 16 constant, 16 q0,
48 other derivative entries. The latter comprise 24 literal-zero assignments
and 24 nonzero expressions. There is no whole-array clearing statement to
remove. A proved partial producer could preserve B0 and B1's original DAG,
derive the other rows, and avoid 48 stores, 48 complex row checks and 96 real B
component bound checks. The complete A scan would remain.

Symbolic row equality is insufficient for this fusion. It needs a nonassociative
stored real/imaginary DAG proof under the pinned compiler flags, including
signed zero and FMA lowering, plus preservation of P2 momentum output. Current
guards run after the full vertex. Omitting derivative calculations before a
guard could suppress an original serious exception or trap. A safe design must
prove a pre-producer finite/exponent domain or execute the original producer on
fallback; it cannot merely reuse today's post-producer guard. A possible
source lead is that these expressions contain at most three scalar factors,
but no complete per-node exponent or exception proof is claimed here.

This potentially saves more materialization work than strict environment
sharing, at the cost of a substantially stronger source/DAG/IEEE proof. No new
prototype or performance claim is warranted by the census alone.

`verify.py` checks this frozen scope and the recorded source/object identities
without compiling, evaluating, or creating bytecode. Source paths are the
retained generated development trees, so `--sources` additionally requires
those trees to remain available. The default verifies the packaged evidence.
