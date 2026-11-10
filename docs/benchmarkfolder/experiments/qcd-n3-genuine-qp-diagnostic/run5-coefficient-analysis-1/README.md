# Closed run5 arithmetic evidence

`summary.md` gives per-point/mode values versus same-mode CutTools, all admissible pair maxima, and mode changes. `summary.json` retains all eight slots, 24 observed coefficient rows, 48 pair/point discrepancy rows, three separate metrics and explicit zero/opposite handling, threshold counts and maxima witnesses. No raw trace data are duplicated.

All finite/single/double symmetric discrepancies are below 1e-10 across the 48 allowed pairs; maximum finite 4.9434027507807906e-11, single pole 6.845195986240909e-14 and double pole 1.2482753548977742e-15. These are returned REAL*8 discrepancies from unchecked status120/accuracy[-1,-1], without an independent truth oracle, error estimate, scientific acceptance or timing qualification.

Raw input SHA256 241ed7c23c6eb52728b7c2a79a7d17ba7558652eade794947c14a0223ff0c08a; immutable combined retention index 30079cbae9a7823970d09b07e8bf6f54dce7c8c2f0f97cc84fc129271fde8830. Reproduction commands and bounded helper/tests are in sibling `run5-coefficient-analysis-source-1/README.md`; exact source index is pinned in provenance.json.
