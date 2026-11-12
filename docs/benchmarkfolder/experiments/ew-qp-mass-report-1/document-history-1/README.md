# Preserved unsuccessful report renders

These files retain the exact metadata and output of two unsuccessful private
document attempts preceding the completed44-page report. They are unrelated
to scientific amplitude failures and do not alter either EW evidence capsule.

Attempt1 used document source51edd02c. Preparation passed. Typst compilation
failed when Rayon could not initialize its default thread pool (EAGAIN11).
No PDF, page-count result or composite resulted; cleanup had no survivors.

Attempt2 used source144a3b5c and an explicit recorded `RAYON_NUM_THREADS=1`.
Compilation, page-count, text and font commands succeeded, with one page.
The controller rejected its declared preview output prefix because it lay in
REPORT rather than WORK, before launching pdftoppm. There was no closed render
result/index or composite. The partial PDF identity is recorded without PDF
bytes; cleanup had no survivors.

The third sourceff2f4f9b corrected only that exact preview final-argument
allowance, retaining one thread and the same sourcea681 page, fonts, tools,
resource caps and scientific statements. Its separate completed render and
composite proof are selected by the public report ledger. This history does
not rewrite the failed statuses or count the partial render as successful.

`index.json` hashes the exact physical files. `ancestry.json` records their
original private paths as inert provenance and identifies the omitted partial
PDF. The two source indices and prepare terminals are retained identities,
not complete source namespaces or full preparation/clone packages. A fresh
portable replay of those failed controllers is not claimed. Reproducing the
published report uses the complete public Typst inputs and matching append
owner; no scientific/native evaluation is needed.
