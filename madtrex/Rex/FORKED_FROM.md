# Forked copy of Rex

This directory is a plain copy of the Rex repository (Rex/teaRex LHE parsing
and event reweighting, Python package `pyrex`), vendored here so that MadtRex
can be built without a separate Rex checkout.

- Origin: https://github.com/zeniheisser/Rex
- Commit: 94b4ed435a4bddce2f041c210e0455f2eff01379 (committed 2026-10-05)
- Copied: 2026-10-23
- Working tree at copy time: no modified tracked files; the then-untracked
  `examples/showcase.py` was included.

This copy no longer tracks upstream. It is ordinary tracked files in the
MadGraph7 repository (not a git submodule); changes made here are not synced
back to github.com/zeniheisser/Rex, and upstream changes are not pulled in
automatically.
