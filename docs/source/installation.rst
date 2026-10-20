Installation
============

Download the latest release tarball from the
`releases page <https://github.com/MadGraphTeam/MadGraph7/releases>`_, unpack it and
start the interface::

    tar xzf MG7_v0.2.0.tar.gz
    cd MG7_v0_2_0
    ./bin/madgraph

Python 3.12 or higher and a C++ compiler are required. The ``madspace`` library is
installed automatically the first time you launch a run. To build it yourself instead,
run ``install madspace`` from the MadGraph7 prompt, or see the
:doc:`MadSpace installation instructions <madspace/installation>`.

NinjaRS for MadLoop
------------------

The optional NinjaRS reduction backend uses oneloopmaster for scalar integrals.
From the MadGraph7 prompt, run::

    install ninjars

The installer builds the pinned sources in ``vendor/ninjars/Cargo.lock``, tests
the installed Fortran interface in double and quadruple precision, and saves the
``ninjars`` library path. It requires Rust/Cargo 1.96 or later, GNU make, C/C++
and Fortran compilers, and the GMP/MPFR numerical build dependencies. The
Fortran compiler must support ``REAL(16)``. The repository's pinned Nix
development shell provides these tools.

Select NinjaRS in the generated ``Cards/MadLoopParams.dat``::

    #MLReductionLib
    8

The existing default reduction order is retained. A chain such as ``8|6|1``
also permits Ninja and CutTools fallback. ``UseQPIntegrandForNinjaRS`` controls
whether the NinjaRS high-precision rescue receives a numerator recomputed in
quadruple precision. The CLI checks accept ``--reduction=8``.

For local development, supply source checkouts explicitly::

    install ninjars --source=/path/to/ninjars --oneloop-source=/path/to/oneloopmaster --jobs=4 --force

These sources are copied into an isolated build directory; the original
checkouts are left untouched. ``--force`` replaces an existing installation
only after the new one passes both precision checks. The default destination is
``HEPTools/ninjars``; ``--prefix`` or ``heptools_install_dir`` selects another
HEPTools prefix. Build failures retain their log in the reported staging
directory. A successful installation retains its build log and exact Cargo
manifest and lockfile. Cargo target and intermediate build caches are kept
under the installation prefix; explicit ``CARGO_TARGET_DIR`` and
``CARGO_BUILD_BUILD_DIR`` settings are respected.

Both standalone Fortran and aMC@NLO exports support NinjaRS. With
``set output_dependencies internal``, the exporter copies its archive, Fortran
module and link metadata into the output directory. The generated makefiles
still require the target platform's native compiler/runtime libraries.
