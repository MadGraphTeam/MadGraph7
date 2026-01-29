<p align="center">
  <img src="https://github.com/madgraph-ml/madevent7/raw/main/docs/source/_static/logo-light.png" width="450", alt="MadSpace">
</p>

### Installation

#### Package

Packages on PyPI are available for Linux and MacOS X (with Apple silicon),
for Python 3.11 to 3.14.

```sh
pip install madspace
```

#### Build

First install `scikit_build_core` with

```sh
pip install scikit_build_core
```

The pre-installed version of `cmake` under MacOS is outdated, so you might need to install a
newer version, for example with

```sh
brew install cmake
```

Then check out the `madspace` repository and build and install it with

```sh
git clone git@github.com:madgraph-ml/madspace.git
cd madspace
pip install .
```

For a development version allowing for incremental build, use the following command instead:

```sh
pip install --no-build-isolation -Cbuild-dir=build -Ccmake.build-type=RelWithDebInfo .
```

This will create a directory `build` where you can run make directly to make development
easier. To update the python module itself, make sure to also run the `pip install` command
above again. This will not happen automatically, even if you make the installation editable!
Build type `RelWithDebInfo` generates optimized code but includes debug symbols, so you
can use `lldb` or `gdb` to debug the code.

### Tests

To run the tests, you need to have the `pytest`, `numpy` and `torch` packages installed.
One test optionally requires the `lhapdf` package (can be installed via conda or built from
source) and the `NNPDF40_nlo_as_01180` PDF set.

To run the tests, go to the root directory of the repository and run
```sh
pytest tests
```
