# Closed raw-DP run4 setup failure

Source4 inherited the qualified source3 drivers and 400 frozen points without
new compiler, loader, point-generation or amplitude calls. Root inheritance
session 54810 exited 0.

Raw-run session 9030 then exited 1. All 20 requests remained unexecuted and
passed their read-only postguards. Independent numerical-card readback
succeeded; the next prelaunch check incorrectly required `param_card.dat`
in the subprocess directory. The compiled Fortran reader instead searches
the private export's `Cards` directory as a fallback. No native worker,
lock request or amplitude evaluation occurred. All 6,000 primary probes
remain unexecuted.

This capsule retains the complete closed run, all 20 request records and
written cards, plus the successful inheritance receipts. It does not copy
drivers, binaries or repeat the source3 preparation capsule. The original
passed preparation and physical sampling inputs remain bound by their
recorded hashes. Recorded host paths are metadata; the portable verifier
does not follow them.

```
python3 -B verify.py
```
