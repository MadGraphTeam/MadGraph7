# Independent ordinary-matrix retention replay

The selected sealed package has index SHA-256
`8f5f7110cc101dc79e8fe75d5dd1ef342e1e72797986a5a5026680e28d4496be`.
Both the selected-package replay and a fresh `/tmp` relocation pass, with
identical canonical replay hash
`d515f63a6f20307980141bb48f15f2787dcc2d820b5e63c555f292ead27fa62f`.
The package's exact outer inventory, 1,221 retained text members, 157,210,978
expanded bytes, source hashes, and summary were independently checked. The
original sealed package remains unchanged.

All eleven cases, six lanes per case, and 100 saved points per lane pass the
registered `3e-6` criterion when replaying the frozen parsers and supervisor
definitions. The metric is `abs(a-b)/abs(a+b)` with the frozen negligible-pole
criterion. The tighter `1e-8` target still has misses in QCD n3, QCD n4, and
EW n4. `cases.json` retains each lane's accepted DP/QP counts and status counts;
these are accepted-output labels, not internal retry telemetry.

Both replays restrict file-content access to the selected package and their
own temporary replay tree. The relocated pass cannot read the original
workspace. Process creation, process-file reads, and `ctypes.dlopen` are blocked.
Five intentional negative controls exercise these restrictions in each pass.
The verifier's fictional `/portable-checkpoint` root triggers three blocked
metadata-only probes in each pass; they are recorded explicitly and no real
filesystem query is performed for them. There are no other unexpected denied
operations. Python's standard modules are initialized before confinement; no
compiler, target executable, target loader, reducer, or scientific DSO is used.

`runtime-console-scan.json` hashes and scans exactly the 66 retained members
whose names contain `/experiments/numerical/` and end in `-output.txt`.
Case-insensitive `##W03\b` produces zero matching lines. Source files and JSON
protocol strings are excluded. This result covers those actual retained runtime
consoles only. IEEE flags and internal retry counts were not observed by the
ordinary campaign.

Four development guard failures are retained: the first two rejected unloaded
standard-library imports (`argparse`, then `__future__`); the next two rejected
the fictional-root probes in the audit's final assertion. The final harness
preloads those standard modules, continues to block all fictional-root reads,
and classifies only those exact three metadata probes separately. The frozen
package and scientific replay were never changed. The failed logs are not
numerical failures.

`audit.py` is the actual one-shot audit recipe; `proof.json` records the two
successful passes. `verify.py` checks this lean proof packet without reading
the original workspace or executing any replay/native programs. This proof
establishes portable ordinary-matrix replay only. Performance acceptance,
stress qualification, and the overall goal remain unqualified here.
