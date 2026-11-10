# Current MadLoop statistics qualification

One bounded invocation of the unchanged current
[`test_ninjars_statistics.py`](../../../../tests/unit_tests/various/test_ninjars_statistics.py)
passed **10 tests and four compiled Fortran subtests**, with exit status zero.
The complete named outcomes are in [pytest-output.txt](pytest-output.txt).

The four subtests compile and execute production tally/writer fragments for
grouped and ungrouped loop-induced LO output, fixed-order NLO output, and matched
NLO output. Synthetic return-code sequences cover every legacy units bucket,
NinjaRS DP/QP, TIRust DP/QP, and initialization codes. Production Python readers
decode the resulting XML/logs. Other tests check historical XML/log handling,
optional counter reset, mixed aggregation, alias separation and conservation of
usage totals.

This qualifies the identified **current statistics sources**. It does not
recover the invocation behind the historical 92-pass log discussed in the
[earlier audit](../acceptance-gap-audit/README.md), and does not establish an
amplitude, reduction-library, cross-section or performance result. No MadLoop
amplitude or Rust reducer was executed. Production and test sources were not
edited for this run. Only four small statistics programs were compiled.

[before.json](before.json) binds the exact command, CPU 12 affinity, private
temporary directory, pinned environment, Python/pytest/compiler/tool versions,
and hashes of the test, production Fortran producers, production Python readers
and writer, and environment inputs. [results.json](results.json) records matching
after hashes, exit status, raw-output digest and an empty temporary directory.
The run used GNU Fortran 15.2.0, Python 3.13.12 and pytest 9.0.2. Elapsed time is
administrative metadata, not a benchmark measurement.

The command used from the MadGraph root was:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python3 docs/benchmarkfolder/experiments/statistics-roundtrip-current-1/run.py
```

[run.py](run.py) invokes `taskset -c 12`, `pytest -v -p no:cacheprovider`, disables
plugin autoload and bytecode writes, and assigns
`.tirust-build/statistics-roundtrip-current-1/tmp` to `TMPDIR`. Its sole owned
process group has a 120-second timeout with bounded TERM/KILL cleanup. No timeout
or cleanup signal occurred. The runner refuses an existing attempt; reproduction
must use a separately named evidence/work directory and preserve these records.

[manifest.json](manifest.json) binds the retained runner, metadata, output and
this scope statement. Historical evidence and frozen numerical campaign inputs
were not modified.
