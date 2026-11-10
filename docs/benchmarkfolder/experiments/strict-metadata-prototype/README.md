The proposed installer correction passes all 27 existing installer/card tests
in a private module, including the four metadata regression tests retained by
the acceptance audit. Two additional checks reject JSON `1.0` for the ABI and
link-order versions. Version fields use a strict integer schema: booleans and
floating-point spellings do not qualify as version integers.

`implementation.patch` changes only the two version guards. Production remains
unchanged during the frozen numerical/performance campaign, and the worktree's
metadata regression tests remain deliberately red against production. These
results qualify the proposed correction, not its deployment.

The tests use the existing mocked installer artifacts and subprocess callbacks.
An additional process-launch guard rejects unexpected real subprocesses. No
compiler, loader, reducer, download or real installation was executed. The
first runner stopped before executing any tests because its import path lacked
the repository root; `attempt1/` preserves that source and failure description.

Reproduce from repository revision `024b266e2e`, with the retained
`../acceptance-gap-audit/metadata-tests.patch` applied to the test module:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python -B docs/benchmarkfolder/experiments/strict-metadata-prototype/run.py
```

Use a clean ignored work directory; the runner refuses to overwrite an attempt.
`results.json` binds the original and private source, test source, patch, runner
and test log. The test module and production source are checked unchanged after
the run. Applying the correction after the campaign still requires running the
tests against the actual production module.
