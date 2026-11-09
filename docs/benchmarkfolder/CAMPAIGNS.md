# Frozen benchmark campaigns

The original measurements remain in `runs/` and `experiments/`. A qualified
runtime failure is authoritative for its numerical implementation and fixed
policy. Repeating the same implementation under a new label cannot replace it.
The original qualified QCD 2→1 result therefore remains a failed acceptance item
for the `c91cb85f3a` candidate.

An implemented optimization uses a new explicit campaign. Registration seals
its exporter files, numerical source snapshots, installed native artifacts,
cards policy, UFO model, phase-space points and baseline revision **before**
generation or measurement. Supply the actual validated installation prefix;
the command does not build or install it.

```sh
source tools/tirust_env.sh
python3 tools/tirust_benchmark.py register-campaign \
  --campaign optimization-id --tirust-prefix /absolute/validated/prefix \
  --predecessor legacy --reason 'Describe the implemented and validated optimization'
python3 tools/tirust_benchmark.py reuse-baseline --campaign optimization-id
python3 tools/tirust_benchmark.py build --campaign optimization-id --source baseline
python3 tools/tirust_benchmark.py generate --campaign optimization-id --source candidate
python3 tools/tirust_benchmark.py build --campaign optimization-id --source candidate
python3 tools/tirust_benchmark.py campaign --campaign optimization-id
python3 tools/tirust_benchmark.py summarize --campaign optimization-id
```

These commands are templates, not evidence that an optimized campaign has run.
Run them from the repository root. Select an allowed CPU explicitly with
`--cpu` when coordinating a quiet timing window. Registration rejects unchanged
native numerical artifacts and unchanged numerical Fortran templates under
the failed predecessor's policy. An artifact change identifies a new candidate;
it does not itself establish a performance improvement.

The numerical identity currently covers installed native libraries and numerical
Fortran templates. An optimization implemented only in a Python numerical
generator needs separate proof of changed generated Fortran before it can open
a new identity. Exporter metadata or Python edits alone cannot reset a qualified
failure. The tool deliberately rejects that case until the generated numerical
change is explicitly incorporated into the preregistration protocol.

Each namespace stores retained evidence under
`docs/benchmarkfolder/campaigns/<id>/` and ignored build artifacts under
`.benchmark-work/tirust/campaigns/<id>/`. The exact baseline source remains shared
at `.benchmark-work/tirust/baseline`. `reuse-baseline` validates and copies the
predecessor's existing source-built objects, libraries and executable into a
private output tree, with independent cards. An isolated relink first requires
the untouched old driver, objects and original library paths to reproduce the
recorded executable byte for byte. A failed proof requires a full source rebuild.
It retains their original evidence
and requires a worker relink to bind the new library paths and campaign. It
refuses active workers, changed inputs and existing destinations. Candidate
exports are generated through the actual MG CLI and built from their sources.

The manifest snapshots the predecessor's headline and retains every qualified
failure, including unselected attempts. Later completed ancestor failures are
added to an append-only discovery journal. Changed or removed evidence fails
verification. Source, native-library or policy changes after registration also
fail live execution. Historical summary reconstruction uses the retained
manifest and observations, without requiring the current checkout to remain at
an older revision.

The original TIRust installation has also been copied, with a complete hash
inventory, to the ignored prefix
`.benchmark-work/tirust/historical-installs/c91cb85f3a-49af12f0b88252a2/tirust`.
This preserves the native artifacts used by the failed original campaign.

## Numerical preparation without a second traversal

For expensive processes, a **new** numerical preparation run may reuse the
final complete steady traversal already performed during worker initialization:

```sh
python3 tools/tirust_benchmark_numerical.py \
  --campaign optimization-id --case qcd_n4_uux_zzgg \
  --label numerical-steady-warmup --warmup-only
```

The parser requires the final 100 consecutive steady calls to cover each frozen
point exactly once, allowing the cyclic order to wrap. It verifies every answer
column, backend/precision status, finite coefficient, stability estimate and
READY dimension, and preserves the verbatim trace and READY fields alongside
the normal executable, source, library, card and point bindings. It never searches
backward past an invalid final call for a more favorable window.

These observations have `seconds=null`, `seconds_per_point=null`,
`timed_status_counts=null`, and `performance_eligible=false`. The retained cold
interval covers initialization plus warmup. This option does not alter the
acceptance worker, its timed loop, or already running numerical preparations.

## Campaign execution order

Complete the separate numerical validation first: every required saved point,
explicit backend, coefficient and rescue/failure check must be assessed under
the frozen candidate policy. The `campaign` timing command does not certify that
this prerequisite has been performed. A passing rescue-chain timing comparison
does not establish agreement for explicit backends that have not been tested.

After numerical validation, `campaign` attempts all selected main TIRust-first
chain comparisons before any explicit-backend regression or control timing.
For the complete matrix this means eight main chain gates. Each retains the
same seven-pair, full-100-point protocol and per-process criterion. A failed,
ineligible or incomplete main gate remains reported; the command returns
nonzero and defers the entire later regression/control phase. Other selected
main chain gates are still attempted, so an early failure cannot hide their
outcomes. Deferred work remains incomplete, never implicitly passed.

When every selected main gate passes, the command runs the explicit backend
comparisons and controls. Every row requires numerical agreement and sufficient
batch duration. The performance bound applies to main rescue chains and main
existing-backend regressions (1, 6, 7 and 8). Standalone backend 9 versus CutTools
and control speed comparisons are descriptive; an eligible slower result in
those rows does not introduce an additional performance requirement. All failed
required gates are retained and return nonzero. This ordering avoids spending the full regression budget on
a candidate that already misses a main acceptance gate; it does not relax any
final delivery requirement or permit selecting favorable attempts.

Generated MG command files apply `--no_save` to every configuration-changing
`set` command. MG can otherwise persist options in its installation's input
directory even when `XDG_CONFIG_HOME` is private. Historical command files and
their effects are retained; this protection applies to future exports.
