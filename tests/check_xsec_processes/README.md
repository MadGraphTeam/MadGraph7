# Systematic MG7 cross-section tests

Regression check that MadGraph7 reproduces the reference cross-sections
recorded in `CLAUDE.md`. The source of truth is [`reference.json`](reference.json).

## What it does

For every process listed in `reference.json` the driver
[`run_check_xsec_processes.py`](run_systematic_xsec.py):

1. builds an MG5 interface and `generate`s the process (the flavour-merged
   `_quark` / `_anti_quark` particles are built into the `sm` model and used
   directly),
2. `output mg7`s it,
3. edits `Cards/run_card.toml` — sets the event count and, for the hadronic
   `t t~` decay rows, neutralises the jet cuts (per `CLAUDE.md`). Fixed
   renormalisation/factorisation scales (µ = 91.188 GeV), `e_cm = 13000` GeV and
   `NNPDF23_lo_as_0130_qed` are already the template defaults, matching how the
   reference table was produced,
4. runs `bin/generate_events -f` and reads the cross-section from the madspace
   `Events/*/info.json` (`process.mean` / `process.error`),
5. compares the **relative difference** to the reference against a tolerance.

A process passes only if `|got - ref| / ref <= tolerance`.

## Running locally

Requires the mg7 runtime stack: `madspace` built into `madspace/install`
(`pip install ./madspace -Ccmake.define.ENABLE_OPENBLAS=ON --target madspace/install`)
and the `NNPDF23_lo_as_0130_qed` LHAPDF set.

```bash
# list the processes
python tests/check_xsec_processes/run_systematic_xsec.py --list

# one process
python tests/check_xsec_processes/run_systematic_xsec.py --process gg_ttxg --events 100000

# a whole section
python tests/check_xsec_processes/run_systematic_xsec.py --section 3jets \
    --events 100000 --tolerance 0.01
```

Set `--datadir` (or `$LHAPDF_DATA_PATH`) if the NNPDF grid is not auto-detected.
Exit code is `0` if everything passed, `1` on any failure, `77` if the LHAPDF
data is unavailable (clean skip).

## CI

[`.github/workflows/check_xsec_processes.yml`](../../.github/workflows/systematic_xsec_mg7.yml)
runs this on `workflow_dispatch` with two inputs:

- `tolerance` — max relative difference (default `0.01`, i.e. 1%),
- `events` — events per process (default `100000`; the reference table used 1M).

It builds madspace once, then runs one job per section in parallel.

## Notes

- **Event count vs 1M reference.** The reference values were produced with 1M
  events; CI defaults to 100k for wall-clock reasons. With a hard 1% gate, keep
  an eye on the smallest-cross-section rows — if the run's own Monte-Carlo error
  approaches 1% the comparison can fluctuate. Raise `events` for those, or widen
  `tolerance`, as needed.
- The physical cross-section is identical whether quark flavours are summed as a
  plain multiparticle or handled by the flavour-merged `_quark` machinery, so
  the reduced-event runs remain a valid check of the reference values.
