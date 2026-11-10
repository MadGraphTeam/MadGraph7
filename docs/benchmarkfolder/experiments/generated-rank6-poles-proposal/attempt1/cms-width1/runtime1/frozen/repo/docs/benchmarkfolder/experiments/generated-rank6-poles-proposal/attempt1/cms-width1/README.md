# Private nonzero-width correction

Attempt1's supposed complex-mass fixture used a zero WZ from the frozen physics
card and is explicitly unqualified. This separate fixture replaces only the
private driver expression `-MZ*WZ` with `-MZ*2.498770D0`, using the positive WZ
in the pinned export's default card. The existing model card, parameters,
production code, old executable and evidence remain unchanged. `fixture.json`
binds the full origin card and exact expected binary64/binary128 complex-mass
words. `driver.f` differs by that one literal substitution. The schedule selects
exactly the original15 fixture105 rows, retaining all original modes, cache
controls, backend/precision choices and thresholds.

The build stage compiles only the private driver and relinks one executable
with all original numerical objects. It reuses the recorded compiler/flags,
binds every old LOAD input before linking, and requires exactly the original
link input set with the one driver object replacement. CPU12 is nonexclusive;
two serial commands share120seconds and4GiB. No loader or native worker runs
during build. The native stage remains separately admitted: one30-second
loader check, one900-second/4GiB worker, the same numerical lock and reviewed
owned-process supervisor, exact original runtime closure and actual ELF mapping.
All raw rows, param.log, errors and failure artifacts are preserved.

The analyzer parses the15 new rows and2 fresh EW captures with the original
strict parser. Every new third mass must match the exact frozen complex value;
fresh captures must reproduce retained original inputs, outputs and status.
Only for comparison logic are those15 rows substituted into the original
retained123-row analysis. Raw attempts remain separate. The result explicitly
labels combined analysis; inherited IEEE flags, refusals and input-policy
mismatches remain open and no tolerance is relaxed. This is a tensor diagnostic,
not physical-amplitude/CMS-model or performance qualification. Final precise
timings remain a TODO for Hua-Sheng's local machine.

From the repository root after `source tools/tirust_env.sh`:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/cms-width1/controller.py verify
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/cms-width1/controller.py build
# Separate native admission required:
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/cms-width1/controller.py run
```
