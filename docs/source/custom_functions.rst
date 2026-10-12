Custom functions
================

MG7 can call your own code for four parts of the calculation:

``cut``
    Decides if a phase-space point passes the cuts. It is applied on top of the
    ``[cuts]`` of the :doc:`run card <run_card>`.

``scale``
    The renormalization and factorization scale. It replaces ``dynamical_scale_choice``,
    like ``user_dynamical_scale`` in MadEvent.

``pdf``
    The parton densities. It replaces the LHAPDF set of ``[beam] pdf``.

``bias``
    An event bias. More events are generated where the bias is large, and their weights
    are lowered to compensate.

You write the function for a single phase-space point in C, C++, Fortran, CUDA or HIP.
MG7 builds it into a shared library and calls it on batches of events. The library
implements the :doc:`UMAMI interface <madspace/umami-api>`, the same interface the
matrix elements use. A generated wrapper takes care of that, so you only write the
function itself.

Creating a custom function
--------------------------

Run ``bin/init_custom_func`` in the process directory. Without options, it asks which
function and language you want:

.. code-block:: text

    $ ./bin/init_custom_func
    Which custom function do you want to create?
      1. cut      a cut, 1 keeps the event and 0 rejects it
      2. scale    the renormalization and factorization scale
      3. pdf      a parton density x f(x, Q)
      4. bias     an event bias for the unweighting
    Which language do you want to write it in?
      1. c        C
      2. cpp      C++
      3. fortran  Fortran
      4. cuda     CUDA, for runs on NVIDIA GPUs
      5. hip      HIP, for runs on AMD GPUs

The same choices are available as options:

.. code-block:: bash

    ./bin/init_custom_func --function cut --language cpp

``-f`` and ``-l`` are short for ``--function`` and ``--language``. ``-y`` takes the
defaults (``cut`` and ``cpp``) for the options you leave out instead of asking.
``--force`` overwrites an existing directory. ``--no-build`` skips the build.
``--no-set-run-card`` leaves the run card unchanged.

The script creates a directory ``custom_<function>``:

``custom_<function>.<ext>``
    The function body. This is the only file you edit.

``wrapper_cpu.c`` or ``wrapper_gpu.cu``
    The UMAMI wrapper. It unpacks the batched inputs and calls your function once per
    event.

``umami.h``
    The UMAMI header.

``Makefile``
    Builds ``lib/custom_<function>.so``.

By default, the script builds the default function right away and enables it in
``Cards/run_card.toml``:

.. code-block:: toml

    [custom_functions]
    cut = "lib/custom_cut.so"

``bin/generate_events`` runs ``make`` in ``custom_<function>`` before each run. Your
changes are therefore picked up without a manual rebuild. The build output goes to
``custom_<function>/compile.log``.

Function signatures
-------------------

The cut, scale and bias functions receive the external particles of one event. The
incoming particles come first. The momenta are in the lab frame, in GeV, with components
``(E, px, py, pz)``. The PDG ids are those of the first flavor of the subprocess for cuts
and scales, like the built-in cuts, and those of the sampled flavor for the bias.

C, C++ and CUDA/HIP:

.. code-block:: c

    double custom_cut(int particle_count, const int* pdg_ids, const double momenta[][4]);
    double custom_scale(int particle_count, const int* pdg_ids, const double momenta[][4]);
    double custom_bias(int particle_count, const int* pdg_ids, const double momenta[][4]);
    double custom_pdf(int beam_index, int pdg_id, double x, double q);

In C++, the functions are declared ``extern "C"``. In CUDA and HIP, they are
``__device__`` functions and run on the GPU, one thread per event.

Fortran:

.. code-block:: fortran

    function custom_cut(particle_count, pdg_ids, momenta) bind(C, name="custom_cut") result(value)
        use iso_c_binding
        integer(c_int), value :: particle_count
        integer(c_int), intent(in) :: pdg_ids(particle_count)
        real(c_double), intent(in) :: momenta(0:3, particle_count)
        real(c_double) :: value

The PDF takes ``integer(c_int), value :: beam_index, pdg_id`` and
``real(c_double), value :: x, q``.

What the functions return:

``cut``
    1 to keep the event, 0 to reject it. The value multiplies the event weight, so values
    in between act as a weight factor.

``scale``
    The scale in GeV, used for the renormalization scale and both factorization scales.
    ``scale_factor`` is applied on top. ``fixed_ren_scale`` and ``fixed_fact_scale`` still
    take precedence.

``pdf``
    :math:`x f(x, Q)` for the parton ``pdg_id`` (21 for the gluon) in beam ``beam_index``
    (0 or 1), at momentum fraction ``x`` and factorization scale ``q``.

``bias``
    The bias factor :math:`b > 0`. The phase space is sampled and unweighted according to
    :math:`|w|\,b`. The written events carry the weight divided by :math:`b`. Distributions
    filled with the event weights therefore stay unbiased, while regions with a large bias
    get more events. The reported cross section does not include the bias.

Examples
--------

A cut requiring :math:`p_T > 150` GeV for both particles of a ``p p > t t~`` process, in
C++:

.. code-block:: c++

    #include <cmath>

    extern "C" double custom_cut(int particle_count, const int* pdg_ids, const double momenta[][4]) {
        for (int i = 2; i < 4; ++i) {
            if (std::hypot(momenta[i][1], momenta[i][2]) < 150.) {
                return 0.;
            }
        }
        return 1.;
    }

:math:`H_T/2` as the scale, in Fortran:

.. code-block:: fortran

    function custom_scale(particle_count, pdg_ids, momenta) bind(C, name="custom_scale") result(value)
        use iso_c_binding
        implicit none
        integer(c_int), value :: particle_count
        integer(c_int), intent(in) :: pdg_ids(particle_count)
        real(c_double), intent(in) :: momenta(0:3, particle_count)
        real(c_double) :: value
        integer :: i
        value = 0d0
        do i = 3, particle_count
            value = value + sqrt(max(momenta(0, i)**2 - momenta(3, i)**2, 0d0))
        end do
        value = value / 2d0
    end function custom_scale

A bias that populates the :math:`p_T` tail of the first final-state particle, in C:

.. code-block:: c

    #include <math.h>

    double custom_bias(int particle_count, const int* pdg_ids, const double momenta[][4]) {
        double pt2 = momenta[2][1] * momenta[2][1] + momenta[2][2] * momenta[2][2];
        return fmax(pt2 / (100. * 100.), 1.);
    }

An LHAPDF set called through the LHAPDF C++ API:

.. code-block:: c++

    #include <mutex>
    #include "LHAPDF/LHAPDF.h"

    extern "C" double custom_pdf(int beam_index, int pdg_id, double x, double q) {
        static LHAPDF::PDF* pdf = LHAPDF::mkPDF("NNPDF40_lo_as_01180", 0);
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        return pdf->xfxQ(pdg_id, x, q);
    }

The function is called from several threads, so the call into LHAPDF is protected by a
lock. To build it, pass the LHAPDF flags to ``make``:

.. code-block:: bash

    make -C custom_pdf CFLAGS="$(lhapdf-config --cflags)" \
        LIBS="$(lhapdf-config --libs) -Wl,-rpath,$(lhapdf-config --libdir)"

Building
--------

The ``Makefile`` takes the usual ``CC``, ``CXX`` and ``FC`` compilers, with ``gfortran``
as the default for Fortran. ``OPTFLAGS`` defaults to ``-O3``. Extra compiler flags go into
``CFLAGS`` and extra libraries into ``LIBS``. The CUDA and HIP builds detect the GPU
architecture like the matrix elements, and accept the same ``MADGRAPH_CUDA_ARCHITECTURE``
and ``MADGRAPH_HIP_ARCHITECTURE`` overrides.

The language decides the device the function runs on. C, C++ and Fortran functions run on
the CPU, and CUDA and HIP functions run on the GPU. Use the language that matches the
``device`` of the run.

Limitations
-----------

- With a custom PDF, :math:`\alpha_s` still comes from the PDF set of ``[beam] pdf``, and
  the scale and PDF variations of ``[systematics]`` are disabled.
- With a custom bias, the weighted integration histograms of ``[run]
  weighted_histograms`` include the bias. The event weights do not.
- Decays ignore a custom scale and a custom PDF.
- A gridpack stores the custom libraries it was made with. They cannot be changed in the
  gridpack.
