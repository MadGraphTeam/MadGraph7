# MadVFold overlay: development and test hooks

Files installed only by `output DIR --vfold-with-tests`, on top of those in
`../overlay` (a file present in both trees is taken from this one). Paths
mirror the mg5amcnlo tree, as in `../overlay`. These hooks (e.g. the
`ENV_VFOLD_CHANNELS` channel restriction and the MINT0/MINT1 overrides)
are kept separate so that they can be left out of the vfold code.

`--vfold-with-tests` also runs `madvfold/DEV/setupVfoldTests.sh`, which
prepares the process directory for the tests (e.g. relative paths, and the
cudacpp libraries and the `vfold/test/grids` of `madvfold/DEV/<proc>_vfalt`
if present).

The generic test scripts (`vfold/myLaunch.sh`, `vfold/myStepTest.sh`,
`vfold/myBackends012.sh`, `vfold/myTimingTable.py`...) are installed from
`Template/NLO/vfold` in this tree into the `vfold` subdirectory of the
process directory. Run them from there, e.g. `./vfold/myLaunch.sh`.

The Fortran test hooks are not in this tree, but in the files of
`../overlay`, where they are enabled at compile time by the copy of
`Template/NLO/SubProcesses/vfold_tests.inc` in this tree
(`vfold_tests=.true.`), so that each Fortran file exists only once.

The copy of `Template/NLO/Source/make_opts.inc` in this tree only adds
build options for the timing tests (`-Wa,-mbranches-within-32B-boundaries`,
see road B step 11t), which `write_make_opts` in `export_fks.py` uses to
write `Source/make_opts`.
