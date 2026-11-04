# MadVFold overlay: vfold code

Files installed by `output DIR --vfold` (and `--vfold-with-tests`), instead
of the stock mg5amcnlo ones or in addition to them (see `vfold_overlays` in
`madgraph/iolibs/export_fks.py`). Paths mirror the mg5amcnlo tree:
- `Template/NLO/...` is overlaid on the generated process directory;
- `madgraph/<subdir>/<file>.py` replaces the copy in `bin/internal`.

Each patched stock file is first added here as a verbatim copy of the
mg5amcnlo version it is based on, so that `git diff` against that commit
gives the MadVFold patch set (e.g. to port it to later mg5amcnlo versions).
