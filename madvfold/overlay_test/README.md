# MadVFold overlay: development and test hooks

Files installed only by `output DIR --vfold-with-tests`, on top of those in
`../overlay` (a file present in both trees is taken from this one). Paths
mirror the mg5amcnlo tree, as in `../overlay`. These hooks (e.g. the
`ENV_VFOLD_CHANNELS` channel restriction and the MINT0/MINT1 overrides)
are kept separate so that they can be left out of the vfold code.

`--vfold-with-tests` also runs `madvfold/DEV/setupVfoldTests.sh`, which
prepares the process directory for the tests (e.g. relative paths, and the
cudacpp libraries of `madvfold/DEV/<proc>_vfalt` if present).
