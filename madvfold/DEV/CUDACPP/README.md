# madvfold/DEV/CUDACPP

The cudacpp libraries for mode C are no longer kept here: each process
directory generated with `output --vfold` builds its own, in
`<procdir>/CUDACPP`, and copies them to `<procdir>/lib/cudacpp` (see
`madvfold/README_CUDACPP.md`).

This directory is still read by `vfold/genCudacpp.sh` when it generates
`<procdir>/CUDACPP`:

- `.gitignore` is the template of `<procdir>/CUDACPP/.gitignore`, which
  keeps the cudacpp build products (`*.o`, `*.so`, `build.*`, the
  googletest clone...) out of `git status` in the process directories of
  `madvfold/DEV`. Do not remove it.
- with `--vfold-with-tests`, an existing googletest build found here (in
  `GTEST/googletest`, or else in `test/googletest`) is copied to
  `<procdir>/CUDACPP/GTEST`, instead of cloning and building googletest
  again for each process directory. Nothing is kept here by default
  (both are in `.gitignore`).

In the mg5amcnlo repository, the same directory also holds the cudacpp
process directories of the hand-written `madvfold/DEV/<proc>_vfalt`,
which have no `<procdir>/CUDACPP` of their own.
