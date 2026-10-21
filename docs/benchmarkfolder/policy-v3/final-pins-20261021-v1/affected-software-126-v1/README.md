# Affected software checkpoint: 126 tests

The actual cumulative run executed **126 tests: 125 passed, one unchanged
conditional skip, zero failures or errors**, across six modules on CPU 180:

| Module | Tests | Skipped |
|---|---:|---:|
| DP stability and measurement authority | 44 | 0 |
| Paper report | 31 | 1 |
| Numerator construction | 20 | 0 |
| Independent random points | 19 | 0 |
| Loop pair dispatch | 10 | 0 |
| Loop preparation | 2 | 0 |

The skipped paper-report test is
`test_actual_generated_qcd2_count_definitions_agree`: **Actual final generated
source is not retained in this checkout**. Its existing conditional skip is
preserved. Actual generated-process evidence has its own qualification; this
software run does not replace it.

The test process for every module loaded a fresh tracked source archive of
`aec2e50b8fc714729287871b345861485a2bb837`, with explicit reviewed source/test
and new point-authority overlays. Both ignored fixture parents were absent
before every module. All six child process groups exited, and all 84 bound
source/point-publication files matched their pre-run, post-run and handed-over
repository hashes. No fixture semantics changed, and no unchanged 308-test
suite was rerun. There were no native builds, generated-program calls, reduction
evaluations or Rust contexts in this task. Python fake protocols and Typst are
software controls rather than numerical observations.

`bindings.json` is the exact 84-file list: 40 software source images and the 44
files of the separate new random point publication. The source images are
retained in `software-sources.tar.gz`; the point data is referenced in its own
publication instead of duplicated here. `source-state-before.json` and
`source-state-after.json` retain the complete overlay and pre/post custody.
`run-result.json` names each actual command, source hash, test count, runtime,
log digest and process-group exit. Exact raw unittest logs are compressed in
`raw-logs.json.gz`.

The archive initially copied files because tar and the earlier immutable pool
had different write-permission bits. A separate byte/executable-bit check then
replaced 32,879 identical members with hardlinks, sharing 1,575,285,249 bytes.
The earlier source images and modes were preserved. Each overlay unlinked its
target before writing, so it could not modify the earlier proof's inode. The
large private archive is excluded from this publication.

`python3 -B verify.py` checks the capsule seal, exact module counts and skip,
raw-log footers, source images, complete custody and unchanged separate point
publication. `python3 -B verify.py --live ROOT` additionally requires all 84
files to match that current checkout. Optimized Python is explicitly rejected.
This is a historical tested software checkpoint; later report metadata changes
must receive their own validation rather than being silently included here.

To replay, create a new archive of the recorded base inside a fresh directory
under `.benchmark-work/`; overlay `software-sources.tar.gz` and the separately
bound 44-file point publication. Put `run-module.py`, `run-suite.py` and
`modules.json` beside its `source/` directory, source `.ninjars-build/dev-env.sh`,
then run the retained controller with `taskset -c 180 python3 -B run-suite.py`.
The controller removes only its private ignored fixture parents before each
module. The committed toolchain and retained source images define the tested
software environment.
