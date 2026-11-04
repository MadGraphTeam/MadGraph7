# MadVFold and the cudacpp plugin

Mode C of MadVFold evaluates the Born and real-emission matrix elements with
the cudacpp plugin of madgraph4gpu (on CPU with SIMD, or on GPU). This needs
the MadVFold version of the cudacpp plugin (branch `vfold5381` of madgraph4gpu,
cudacpp step 13a and later), which adds the `--vfold` output option.

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

In a process directory generated with `output <dir> --vfold-with-tests`
(e.g. by `madvfold/DEV/genProc.sh eebbbb`, giving `madvfold/DEV/eebbbb_vfgen`):
```
./vfold/genCudacpp.sh        # create CUDACPP: cudacpp process directories, test
./vfold/buildCudacppLibs.sh  # build the cudacpp libraries in CUDACPP
./vfold/copyCudacppLibs.sh   # copy them to lib/cudacpp (vfold/myLaunch.sh does it in mode C)
```
- `genCudacpp.sh` reads `vfold/cudacpp_spec.txt` (created by
  `vfold/genCudacppSpec.sh` if missing) and generates one `madevent_simd`
  directory per cudacpp library: the Born process with the `--vfold` option,
  the real-emission processes without it.
- The googletest directory `CUDACPP/test` (for the cudacpp unit tests, found as
  `../../../test` from `CUDACPP/<proc>.mad/SubProcesses/P1_<proc>`) is copied
  from `madvfold/DEV/CUDACPP/test`, including googletest if it was already
  built there (otherwise `make -f cudacpp.mk gtestlibs` clones it from GitHub,
  which needs git and cmake).
- `buildCudacppLibs.sh` uses ccache (if installed) with the cache directory
  `CUDACPP/ccache`. If this does not exist, it is created as a symlink to
  `madvfold/DEV/ccache` (shared by all process directories in `madvfold/DEV`,
  so that the cache survives the regeneration of a process directory) if that
  exists, or else as a local directory.
- `buildCudacppLibs.sh` and `copyCudacppLibs.sh` also work for the hand-written
  `madvfold/DEV/<proc>_vfalt`, whose cudacpp process directories are in
  `madvfold/DEV/CUDACPP` instead of `<proc>/CUDACPP`.

The cudacpp unit tests can then be run in each cudacpp process directory, e.g.
```
cd CUDACPP/epem_bbbxbx.mad/SubProcesses/P1_epem_bbbxbx
./build.avx2_d_inl0_hrd0/runTest_cpp.exe
```
