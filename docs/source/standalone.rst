Standalone matrix elements
==========================

``output standalone`` writes the matrix elements of a process as a C++ library
(the MadMatrix code, the default matrix-element code of MadGraph7) with a driver
that evaluates them on phase-space points, without the integration and the event
generation of ``output`` (``mg7``) or ``output madevent``. ``output
standalone_fortran`` is the Fortran equivalent, as in MG5_aMC@NLO::

    generate g g > t t~ g
    output standalone my_ttxg

Each subprocess gets a directory ``SubProcesses/P1_*`` holding the generated code
(``CPPProcess.cc``, ``EvaluateDiagrams.inc``, the colour data), a ``makefile`` and
``check_sa.cc``, the driver.

Building
--------

In a ``P1_*`` directory, ``make`` builds the process library and ``check_sa.exe``.
Two variables select the build; they are part of the library name, so several
builds coexist::

    make BACKEND=auto FPTYPE=m          # the defaults
    make BACKEND=simd_256 FPTYPE=d -j8

``BACKEND``
    ``scalar``, ``simd_128``, ``simd_256``, ``avx512y``, ``simd_512``, ``cuda`` or
    ``hip``. ``auto`` picks the widest SIMD flavour the host supports (``simd_128``
    on Apple silicon). The resolved choice is written to ``.resolved-backend``.

``FPTYPE``
    ``d`` (double precision everywhere), ``f`` (single), ``m`` (the default:
    amplitudes in double precision, colour algebra in single precision), ``v``
    (single-precision amplitudes with double-precision momenta and denominators),
    ``e`` (double-word expansion). Compare them with ``check precision``.

The flags are ``-O3 -std=c++17 -ffast-math`` (``OPTFLAGS`` for the optimisation
level); the compiler is ``CXX`` (``c++`` by default).

Helicity recycling
------------------

The plain code evaluates every wavefunction once per helicity configuration.
``--hel_recycling=True`` at output time builds each wavefunction once per copy
(a wavefunction depends only on the helicities of the external legs it descends
from), skips the amplitudes that vanish at a helicity and folds the others into
per-helicity tables, as the helicity recycling of madevent does. It is off by
default in the standalone and ``mg7`` outputs, on in ``madevent``::

    output standalone my_ttxg --hel_recycling=True

A warm-up at the end of the output builds a scalar double-precision probe of the
plain code and runs it on 64 RAMBO points to find the good helicities and the
amplitudes that are only a cancellation zero at each; the recycled code is written
for those. The log says what it built::

    INFO: hel_recycling: P1_gg_ttxg builds 32 helicity rows of 32, dropping 64 (helicity, amplitude) pairs that vanish

The good helicities are chosen as madevent does: a helicity is kept if its
|M|^2 exceeds ``limhel``/ncomb of the sum over helicities at one of the points,
with ``limhel`` = 1e-14 for the warm-up and 1e-8 at run time. At run time they
are chosen as in the Fortran standalone output, on the events the library is
called with: every helicity is evaluated on the first 16 events, and the ones that
contribute to one of them are kept for the rest of the run (or until a parameter
is changed through UMAMI). They are thus chosen at the energies and in the frame
the matrix element is used at. An ``mg7`` run chooses them before the run
instead, from its cards: an instance of a CPU build of each library (a scalar
double-precision one for a run on a GPU) evaluates every helicity on 16 RAMBO
points of the collider of the run card, half in the partonic centre-of-mass frame
and half in the lab frame, at its centre-of-mass energy or, when the beams have
PDFs, ISR or beamstrahlung, from just above the threshold of the process up to
it, and every instance of the run, on every thread and device, is given that
choice (``madmatrix_choose_good_helicities``, ``madmatrix_get_helicity_choice``,
``madmatrix_set_helicity_choice``). It then depends on the cards only, and a run
on a GPU evaluates the helicities of a run on a CPU. A helicity that is a power
of m/sqrt(s) below the others matters near threshold only: the all-plus
helicities of g g > h g in the ``heft`` model are (m_H^2/s)^4 of the rest, 1e-14
at 13 TeV, 1e-9 at 1.5 TeV and 20% at 150 GeV.
The warm-up chooses its rows instead on RAMBO points spread in energy from just
above the threshold of the process to 1.5 TeV, half of them boosted along the
beams when an incoming parton comes from a PDF: the matrix element is then called
in the lab frame, where the helicity of a massive leg is not boost invariant, so a
configuration that vanishes in the partonic centre-of-mass frame can contribute
there (gg > t t~ builds 16 rows where madevent keeps 12). A row that the run time
finds and the warm-up did not is computed without recycling.

Two limits keep the recycled code where it is faster than the plain one:

``--hel_recycling_max=<n>``
    Statements of the recycled block above which the output keeps the plain code
    (default 300000, 0 for no limit; ``WARNING: ... written WITHOUT recycling``
    in the log). Code that size no longer runs from the caches. gg > 5g is above
    it (426k statements).

``HRMAXMB=<n>`` (make variable or environment variable at build time)
    Size of the recycled tables per SIMD page, in MB, above which the build takes
    the plain helicity loop at run time, printing ``INFO: --hel_recycling is not
    used by this build`` once (default 10, 0 for no limit). The recycled build
    moves a fixed amount of memory per event, which wider SIMD does not make
    cheaper; with two threads per core a smaller limit can be faster, with few
    busy cores a larger one.

Other options: ``--hel_recycling_chunk=<n>`` (statements per function when the
block is cut up, default 200), ``--hel_recycling_jamp=rows|fold|expanded`` (how
the colour flows are made), ``--hel_recycling_tables=False`` (straight-line code
instead of tables). A ``cuda`` or ``hip`` build of the same output is the plain
matrix element.

Reference momenta of the external wavefunctions
-----------------------------------------------

``--axial_gauge=True`` builds the polarisation vectors of the massless vector
bosons in the axial gauge of another external leg, one reference momentum shared
by all the massless vector bosons of a gauge group (a coloured massless particle
for the gluons, a charged one for the photons, initial state first, then a
massless fermion before a vector boson; the reference particle takes the next
candidate), and quantises the spin of the massive final-state fermions along the
reference shared by at least two of the massless vectors they couple to, instead
of along their momentum. Whole diagrams and helicity configurations then vanish
and are removed by the helicity recycling; the helicity-summed |M|^2 is unchanged
(exactly for the spin axis; to the gauge dependence of a fixed width for the
reference momenta). The two spin states of such a fermion are not helicity
states: its helicity is written as 9 (unknown) in event files, and a density
matrix (``--density``) is refused for it::

    output standalone my_ttxg --hel_recycling=True --axial_gauge=True
    output standalone my_ttxg --hel_recycling=True --axial_gauge=True --fermion_reference=False

``--fermion_reference=False`` keeps the massive fermions in the helicity basis (it
is implied for a polarised process and with ``--density``). The option is also
accepted by ``output madevent``, ``output standalone_fortran`` and ``output``
(``mg7``); ``check gauge`` and ``check brs`` compare the result with the default
choice (column ``Reference``).

The driver
----------

``check_sa.exe`` has two modes. ``matrix`` (the default) evaluates one RAMBO
point, identical to the one of the Fortran and C++ ``check`` drivers at the same
energy, and prints the momenta and the matrix element::

    ./check_sa.exe                 # 1000 GeV
    ./check_sa.exe matrix 500

``perf`` evaluates ``nblocks * nthreads`` events per iteration for ``niter``
iterations on one flavour, and prints throughput counters; the parameters,
``alpha_s`` included, come from ``../../Cards/param_card.dat``::

    OMP_NUM_THREADS=1 ./check_sa.exe perf --energy 1000 1 2000 100

``TotalTime[MatrixElems] (3)`` and ``EvtsPerSec[MatrixElems] (3)`` time the
matrix-element call alone (random numbers and RAMBO excluded); the first iteration
includes the good-helicity scan. The RAMBO seed is fixed, so two builds of the same
process see the same points. Per-event files (``--dump-me``, ``--dump-momenta``,
``--momenta``, ``-e <file.lhe>``, ``--alphas``, ``--flavors``, ``--rnd-hel``) read
or write one raw ``double`` (or ``int``) per event; ``./check_sa.exe --help``
lists them. ``MG_DUMP_GOODHEL=1`` in the environment prints the good helicities.
