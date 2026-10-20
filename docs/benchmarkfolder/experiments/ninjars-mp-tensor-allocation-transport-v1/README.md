# Corrected transport with the allocation-only tensor patch

All planned correctness gates passed: **22 tests** (18 inherited bridge tests
plus four tensor tests), **21 golden lanes**, the complete **1,392-request** dense
callback oracle, and a separate **1,392-request** replay through reusable
production MP contexts. Both full replays reproduce all **11,136 ASCII output
fields**. The callback oracle also checks all **86,436 complex outputs** for
exact stored values, tracked precision and signs.

The inherited tests cover binary128 values beyond double precision, both zero
signs, normal/subnormal extremes, complex branch lips, and rejection of invalid
transport inputs. The complex-mass pentagon converges between 256 and 384 bits
with scaled error about 2.013e-76. Tensor controls include changing lanes and
scratch sizes, mixed precision, nonfinite products, every expansion kind and
the actual finite zero/one reset domain. General NaN-source `set_from` is not
claimed safe; the previous explicit counterexamples remain in the original
allocation experiment.

`provenance.json` binds every tracked bridge file to committed `9f13d10cc`.
The only changed production source in the private crate is `src/tensor.rs`,
copied byte-for-byte from the prior allocation experiment. Three test/oracle
modules are added. The parser uses the corrected direct widening of the 113-bit
input and preserves negative zero; ABI, provider, thresholds, precision,
dependencies and Cargo lockfile remain unchanged.

The release test compile was paused at the user's request and resumed after
exact PID/start-time/command checks. `execution-events/` retains both events;
the build wall time includes that 592.039-second pause and is not a performance
measurement. All four numerical gates then ran serially on CPU 234 under the
exclusive MP handoff. Source, executable and installed-archive hashes were
checked before and after each gate. No production files or main installation
were changed.

`summarize.py` independently validates retained sources, raw test/replay output
and exact counts without the disposable crate or a numerical engine. `verify.py`
adds the complete retention seal. Executables, compilation products and Python
bytecode are excluded; source snapshots, commands, raw logs and replay rows
are retained. The proposed minimal patch and regression tests are reviewable in
`production-proposal.md` and `production-tensor.patch`.

The approved fixed full-process design is retained in `process-plan.md`, but
has not run in this namespace. It requires separate source/archive preparation,
quiet-window authorization and all samples retained. The prior isolated
callback gain does not demonstrate a MadLoop speedup or satisfy the final
eight-process acceptance gate.
