# MadVfold and the cudacpp plugin

Mode C of MadVfold evaluates the Born and real-emission matrix elements with
the cudacpp plugin of madgraph4gpu (on CPU with SIMD, or on GPU). This needs
the MadVfold version of the cudacpp plugin (branch `vfold7` of madgraph4gpu,
cudacpp version 1.02.00), which adds the `--vfold` output option (also
in `standalone_cudacpp` mode, with multichannel support), keeps googletest
in `<proc>/test/GTEST` and also supports MadGraph7.

## Install the cudacpp plugin

1. Download the cudacpp tarball of the MadVfold release
   `valassi_cudacpp_for3.8.1_v1.02.00_vfold7on20261112`
   (https://github.com/valassi/madgraph4gpu/releases/tag/valassi_cudacpp_for3.8.1_v1.02.00_vfold7on20261112)
   into the `PLUGIN` directory of this MadGraph7 clone:
   ```
   wget -O PLUGIN/cudacpp.tar.gz https://github.com/valassi/madgraph4gpu/releases/download/valassi_cudacpp_for3.8.1_v1.02.00_vfold7on20261112/cudacpp.tar.gz
   ```
   This is the version used for the results of the MadVfold paper. The tarball
   contains a single directory `CUDACPP_OUTPUT` (with a `VERSION.txt`).
   Alternatively (e.g. for later developments), create the tarball yourself
   in a madgraph4gpu clone (branch `vfold7`), with
   `./.github/workflows/archiver.sh`, which writes `cudacpp.tar.gz` in the top
   directory of the clone, and copy it to `PLUGIN/cudacpp.tar.gz`.

2. Install it in the `PLUGIN` directory of this MadGraph7 clone:
   ```
   madvfold/cudacpp_mg7/installPlugin.sh
   ```
   Do NOT just extract the tarball: the plugin does not load in MadGraph7 as it
   is, and then `output standalone_cudacpp X` silently creates a madmatrix
   output in a directory called `standalone_cudacpp`. The script adds the GPU
   classes which MadGraph7 removed (`export_cpp_gpu.py`) to the plugin's own
   copy of the MadGraph7 `export_cpp` module, and gives it its own copy of the
   mg5amcnlo v3.8.1 `aloha_writers.py`: no MadGraph7 module is modified, so
   madmatrix and all other outputs are not affected.
   This creates `PLUGIN/CUDACPP_OUTPUT`, which MG5aMC then finds as plugin
   `PLUGIN.CUDACPP_OUTPUT` (for the `madevent_simd` output used below). MG5aMC
   warns that the plugin is not validated with MadGraph7: this is expected.

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
cd CUDACPP/epem_bbbxbx.vfold.sa/SubProcesses/P1_epem_bbbxbx
./build.avx2_d_inl0_hrd0/runTest_cpp.exe
```
(With MadGraph7, the P directories have no `Sigma_<model>_` prefix, unlike in
mg5amcnlo; the test reference files keep the mg5amcnlo names
`test/ref/dump_CPUTest.Sigma_<model>_<proc>.txt` as of the cudacpp tarball
above.)
