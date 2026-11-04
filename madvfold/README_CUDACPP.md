# MadVFold and the cudacpp plugin

Mode C of MadVFold evaluates the Born and real-emission matrix elements with
the cudacpp plugin of madgraph4gpu (on CPU with SIMD, or on GPU). This needs
the MadVFold version of the cudacpp plugin (branch `vfold5381` of madgraph4gpu,
cudacpp step 13c and later), which adds the `--vfold` output option (also
in `standalone_cudacpp` mode, with multichannel support) and keeps googletest
in `<proc>/test/GTEST`.

## Install the cudacpp plugin

1. Create the cudacpp tarball in a madgraph4gpu clone (branch `vfold5381`):
   ```
   ./.github/workflows/archiver.sh
   ```
   This writes `cudacpp.tar.gz` (and `VERSION.txt`, also included in the
   tarball) in the top directory of the madgraph4gpu clone. The tarball
   contains a single directory `CUDACPP_OUTPUT`.
   The location where the MadVFold tarball will be kept is still to be decided.

2. Install it in the `PLUGIN` directory of this mg5amcnlo clone:
   ```
   tar xzf cudacpp.tar.gz -C PLUGIN
   ```
   This creates `PLUGIN/CUDACPP_OUTPUT`, which MG5aMC then finds as plugin
   `PLUGIN.CUDACPP_OUTPUT` (for the `madevent_simd` output used below). MG5aMC
   warns that the plugin is not validated with 3.8.1: this is expected.

## Build the cudacpp libraries for a process directory

`output <dir> --vfold` (or `--vfold-with-tests`, e.g. via
`madvfold/DEV/genProc.sh 'generate e+ e- > b b~ b b~ [QCD]'`, giving
`madvfold/DEV/epem_bbxbbx_vfgen`) builds the cudacpp libraries for mode C
automatically, if the cudacpp plugin is installed in `PLUGIN/CUDACPP_OUTPUT`:
it runs `vfold/generateCudacpp.sh` in the process directory, which runs the
three scripts below (with their logs in `CUDACPP`); with
`VFOLD_SETUP_NOCUDACPPBUILD=1` (`genProc.sh -nocudacppbuild`) it only runs the
first one. They can also be run by hand in the process directory:
```
./vfold/genCudacpp.sh [-tests] # create CUDACPP: cudacpp process directories (GTEST with -tests)
./vfold/buildCudacppLibs.sh  # build the cudacpp libraries in CUDACPP
./vfold/copyCudacppLibs.sh   # copy them to lib/cudacpp (vfold/myLaunch.sh does it in mode C)
```
- `genCudacpp.sh` reads `vfold/cudacpp_spec.txt` (created by
  `vfold/genCudacppSpec.sh` if missing) and generates one `standalone_cudacpp`
  directory per cudacpp library: the Born process `CUDACPP/<ccproc>.vfold.sa` with
  the `--vfold` option (vfold extensions and multichannel support), the
  real-emission processes `CUDACPP/<ccproc>.sa` without it.
- `CUDACPP/GTEST` (only with `--vfold-with-tests`, i.e. `genCudacpp.sh
  -tests`; without it, the libraries are built without the cudacpp unit
  tests) is the googletest installation (for the cudacpp unit tests)
  shared by these cudacpp process directories: the build of each of them creates
  its `test/GTEST` as a symlink to `../../GTEST` (if `../../GTEST` did not exist,
  `test/GTEST` would be a local directory instead). `genCudacpp.sh` copies an
  existing googletest build from `madvfold/DEV/CUDACPP` (`GTEST/googletest` or
  `test/googletest`) if there is one; otherwise googletest is cloned from GitHub
  and built by the first build (this needs git and cmake).
- `buildCudacppLibs.sh` uses ccache (if installed) with the cache directory
  `CUDACPP/ccache`. If this does not exist, it is created as a symlink to
  `madvfold/DEV/ccache` (shared by all process directories in `madvfold/DEV`,
  so that the cache survives the regeneration of a process directory) if that
  exists, or else as a local directory.
- `buildCudacppLibs.sh` and `copyCudacppLibs.sh` also work for the hand-written
  `madvfold/DEV/<proc>_vfalt`, whose cudacpp process directories are in
  `madvfold/DEV/CUDACPP` instead of `<proc>/CUDACPP` (`madevent_simd`
  directories `<ccproc>.mad`).

To run the process directory in mode C, set the `ENV_VFOLD_*` variables first,
e.g. with the example values (mode C, avx2 backend, SBORN served from cudacpp)
of `source vfold/setupVfoldDefaults.sh` (to be tuned to your needs; the test
scripts such as `vfold/myLaunch.sh` set them themselves).

The cudacpp unit tests can then be run in each cudacpp process directory, e.g.
```
cd CUDACPP/epem_bbbxbx.vfold.sa/SubProcesses/P1_Sigma_sm_epem_bbbxbx
./build.avx2_d_inl0_hrd0/runTest_cpp.exe
```
