<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/logo-wide-dark.png">
    <img src="docs/source/_static/logo-wide-light.png" width="450" alt="MadGraph7">
  </picture>
</p>

MadGraph7 is the next major release of the MadGraph event generator. Its new features
include vectorization and GPU support for matrix element evaluation and phase-space
sampling, machine-learning-accelerated phase-space integration and sampling through
MadNIS, and more accurate handling of spin correlations in MadSpin.

## Alpha release

This is an **alpha release**, meant for testing and feedback, **not for production**.
The MadGraph7 workflow currently covers leading-order event generation. The LO
workflow from MG5_aMC@NLO is still reachable through the `output madevent` mode.
The NLO workflow is also available (with more minor update).
Amplicol mode is not include in this release

If you need a stable release, use
[MadGraph5_aMC@NLO](https://github.com/mg5amcnlo/mg5amcnlo) (also on
[Launchpad](http://launchpad.net/madgraph5)).

## Installation

Download the latest release tarball from the
[releases page](https://github.com/MadGraphTeam/MadGraph7/releases), unpack it and start
the interface:

```sh
tar xzf MG7_v0.2.0.tar.gz
cd MG7_v0_2_0
./bin/madgraph
```

Python 3.12 or higher and a C++ compiler are required. The `madspace` library is
installed automatically the first time you launch a run. To build it yourself instead,
run `install madspace` from the MadGraph7 prompt.

## First steps

Type `tutorial` at the MadGraph7 prompt for a guided walkthrough of the commands.

To start your first process, you can enter the following commands:
```
generate p p > t t~
output my_first_run
launch
```

[MadBoard](https://github.com/MadGraphTeam/MadBoard) is a separate package providing a
modern web interface to MadGraph, for setting up processes, following runs from the
browser and inspecting their results. You can install it with
```
pip install madboard
```
and launch it by running the `madboard` command within your MadGraph7 directory.
