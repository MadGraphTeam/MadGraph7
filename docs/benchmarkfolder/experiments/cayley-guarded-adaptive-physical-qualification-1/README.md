# Private Cayley zero-denominator guard: closed physical replay

The private guarded bridge preserves the two masked physical results of the
closed adaptive comparator. Both trapped physical workers still terminate with
SIGFPE before an observer journal is created. Trap qualification remains open.
No private source or library has been installed or promoted.

| Worker | Exit | Retained observation |
|---|---:|---|
| Checked, masked | 0 | Independent MadLoop stability checks accept DP at ordinary point and QP at threshold; 600 observed QP calls in passes `[4,5,4]` |
| Checked, trapped | -8 | SIGFPE before `mass-observer.hex`; original `FileNotFoundError` assessment retained |
| Forced QP, masked | 0 | 400 observed QP calls; public code 120 and accuracy -1 mean unchecked output |
| Forced QP, trapped | -8 | SIGFPE before `mass-observer.hex`; original `FileNotFoundError` assessment retained |
| Invalid, divide-by-zero, overflow fault controls | -8 each | All three requested trap controls demonstrated |

The masked checked and masked forced-QP amplitudes match their respective
closed adaptive references exactly, including the separately retained status,
accuracy bits, group results and call counts. Their reported maximum numerical
relative error is zero at both saved points. This is a comparison to the same
private adaptive implementation; it does not establish independent physical
accuracy. The historical original implementation is retained as a diagnostic
only: it rejected the threshold point and differs substantially in relative
pole values, despite passing the permissive pole-to-finite regression criterion.

Numerical agreement, host stability acceptance, observed QP path coverage and
trap completion remain separate in `offline-result.json.gz`. The two masked
workers have positive individual observations. The gates requiring both masked
and trapped completion all remain false. Transport completed and all owned
children were reaped; transport completion does not mean trap qualification.
Skipping a zero division may change exception flags and rescue selection, so
path identity is a diagnostic rather than an acceptance requirement. Cleaner
exception flags alone would not qualify host stability.

Root session 95330 reported TERMINAL0. The closed replay used one loader,
four physical workers and three fault controls, CPU 12 without exclusive
reservation, a 4 GiB address-space cap and a 600 second aggregate deadline.
Its total recorded elapsed time is 146.630 seconds; this is diagnostic execution
time, not a benchmark or a performance acceptance measurement. The same checked
executable and all numerical objects were reused without recompilation.

The bridge is identified by SHA256
`69722452c234f494f0aa436fa24bbabfeae59100c03b05a1f0b76336cc17886b`
and its successful build receipt by
`59167eec84a767f0712b6905285a99564458ff0667427de87ce1921534b46632`.
The [portable build qualification](../cayley-guarded-adaptive-build-qualification-1/README.md)
retains source, compiler, lockfile, failed-build and guarded-source provenance.

`manifest.json` maps original repository paths to content-addressed UTF8/gzip
payloads. Native artifacts are identities only. It includes the frozen proposal,
actual artifact binding, completed receipt, raw stdout/control/journal/map
files, every published comparison and all three frozen comparator journals.
Absent trapped journals remain absent. `analyze.py` reparses retained text and
recomputes published comparisons and gates without calling the native runner.
The independent read-only fault localization is retained under `localization/`.

From the repository directory, verify without historical build directories:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/cayley-guarded-adaptive-physical-qualification-1/verify.py
```

The folder can also be copied elsewhere and its `verify.py` run directly with
Python 3.12. It never loads the retained native identities, installs a library,
starts a numerical worker or requires the historical native paths. Do not rerun
`capture.py`: it records one completed attempt and refuses an existing manifest.

This package does not qualify a new build, independent integral truth, general
process coverage, per-call working precision or runtime performance. Precise
final timings remain a TODO for Hua-Sheng on his local machine.
