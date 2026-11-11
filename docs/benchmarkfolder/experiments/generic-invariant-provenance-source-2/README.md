# Generic invariant provenance: portable source proposal 2

This capsule preserves the exact 21 files and source index from private source2
`0e93995f751c7a0f3dfae1e39748714afe5a194ed6dbc1c9e873ca9d89117685`.
It adds a portable verifier, this scope note and the concise root pure-verification
receipt. It changes no production source and enables no scientific correction.

The explicit ancestor is the sibling `generic-invariant-provenance-source-1`
capsule, public index `0d2bbdda8c09fc51dc2cffad348794f3a184cdd273bc1bc602ce616475f7848f`,
with original source index `72c5f2729b0b176bc9ed71779dbf03d25189ac0ba5e176f7eb76cdec86de05f7`.
Retain that sibling directory when relocating this capsule; `--ancestor-root`
can instead identify its explicit location. No files from either ancestor or
historical work directories are rewritten.

Run `python3 -B verify.py` here, or invoke this verifier from the repository root.
`--index-sha SHA256` additionally checks the expected public index digest. Default
verification performs only package and Python-environment reads:

- Authenticates all original source2 bytes and the explicit source1 capsule.
- Replays the ancestor's patch/template/table checks and its 17 pure tests.
- Derives source2's patch again from the exact source1 finalizer and the two new
  modules, then reruns all 40 unchanged semantic and source-window tests.
- Validates the frozen closed and generated records: source-hash joins, procedure
  coordinates, reference/effect status, unchanged ancestor graph summaries and
  every admission/unknown boundary. It does **not** reparse missing historical
  source files or rerun MG/model generation.

The four retained closed DP/QP windows and ten generated Born windows pass the
source-only momentum-prefix contract. Four generated loop-induced windows remain
unknown because the actual generated call/header argument counts differ. Model
initialization, unsupported model functions and full-producer WRITE/STOP effects
remain explicitly unqualified. The tests cover implicit function calls, setters,
arrays/components, numeric-dummy and forwarded momentum aliases, COMMON layouts,
overwrite detection and conservative fallback.

`source_environment_supported` remains literal `.false.`; scientific dispatch is
still the original pass-through. Source-window support does not establish native
entry/use-token behavior, compiler/ABI correctness, normal-checked amplitude
accuracy, or performance. No compiler, loader, reducer or amplitude executes in
portable replay. The root receipt reports a past pure replay with live hash checks;
it adds no numerical qualification.

The seven full MODEL/DHELAS MG generation checks and the 470 original input pins
are retained historical receipts. Their original path strings are inert provenance
here. The required full model/repository source closure is not copied into this
capsule. The sealed scripts under `source/` remain byte-exact reproduction records;
use this top-level verifier for portable replay rather than executing those original
workspace-oriented scripts.
