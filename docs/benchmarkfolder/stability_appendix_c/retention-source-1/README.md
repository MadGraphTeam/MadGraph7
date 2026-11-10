# Closed raw stability replay

This capsule retains the complete closed source5 raw stability pilot: all
four Standard Model processes, all five card-selected reducers, all 100
intended physical points per process, every request outcome, and complete
collected values, stdout, runtime-map and tensor-group evidence. Frozen physical
parameter cards and each materialized worker's canonical `Cards` numerical
card are retained and checked against the prelaunch readback records. The
readbacks preserve every declared destination; other post-run alias copies
are not independently recaptured. A missing or nonregular canonical card in
a guard-failed worker is recorded without following it; that lane remains
unavailable under the frozen metric policy. Source5 and
the 400 physical inputs remain unchanged. A failed or unavailable diagnostic
does not disappear from a denominator.

Large group journals and text receipts use deterministic lossless gzip chunks.
Each chunk is at most 32 MiB before compression and less than 40 MiB after it.
The index records both compressed and original byte counts and SHA-256 values,
with ordered byte-preserving chunks. Replay enforces a 512 MiB uncompressed
budget per logical text file before parsing. No compiled binary is packaged.

The pure verifier opens only capsule-local files. It checks every stored
byte, decodes the full journals, replays the exact frozen source5 diagnostic
parser and consistency calculation, validates the physical input vectors and
cuts, and reconstructs the all-point metrics and fixed 20-row summary. Saved
runtime-observer receipts are checked against retained maps and the bound
closure; historical host binaries and paths are never reopened.

```
python3 -B verify.py
```

The observable is finite/Born for three host forced-DP probes: original,
reversed reduction direction and a signed 90-degree rotation around the beam.
The displayed diagnostic is the range divided by the absolute mean, following
MadLoop's stability implementation. Its common alpha_s/(2*pi) normalization
cancels. A zero mean gives zero when all three values coincide and infinity
otherwise. This is an internal consistency estimate, not an independently
known error. The exact formula comes from MadLoop; arXiv:1604.01363 describes
the probe and tail comparison methodology.

Return code 110 is unchecked. The observation is host forced-DP/card-selected
backend; reducer-internal precision and the actual reducer are unobserved.
The frozen protocol's broader precision phrase must not be read as a claim
that every internal arithmetic operation was observed in binary64.

Every empirical tail retains 100 intended points. Lower and upper bounds are
known exceedances/100 and (known exceedances + unavailable)/100. Dotted curves
show upper bounds from unavailable outcomes, not measured failure fractions;
an entirely unavailable lane has unknown bounds [0,1]. Zero tails and a zero
lower bound cannot appear on a logarithmic axis. Ninja ratios are masked for
missing outcomes or zero Ninja exceedances. The 1% empirical resolution of
this pilot does not resolve the rare tails in a paper-sized campaign.

The source-only capture utility requires an explicitly closed root-admitted
terminal checksum. It must never be pointed at an active campaign. No native
evaluation, performance acceptance or high-precision truth calculation is
performed by capture or replay.
