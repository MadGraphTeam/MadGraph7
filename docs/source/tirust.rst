TIRust for MadLoop
=================

TIRust is the optional tensor-integral reduction backend numbered ``9``.
It uses oneloopmaster for scalar integrals and a Fortran interface for
MadLoop's numerator contraction. Installing it does not change MadLoop's
default reduction choices.

Installation and selection
--------------------------

From the MadGraph7 prompt::

    install tirust

The installer builds pinned dependencies, applies the repository's checked
pole-evaluation patches to isolated copies, and tests the installed Fortran
interface before publishing it. The supported build environment is GNU
Fortran on Linux with binary128 support, Rust/Cargo 1.96 or later, C/C++
compilers, GNU make, patch, and the GMP/MPFR build dependencies. The pinned
Nix development environment supplies these tools.

For the supplied local checkouts, launch MadGraph from its repository root
and use::

    install tirust --source=../references/TIRust --oneloop-source=../references/oneloopmaster --jobs=4

Sources are copied before building; the original checkouts are not modified.
``--update`` (or ``--force``) rebuilds an installation. ``--prefix`` selects
the HEPTools parent directory. A failed build or smoke test preserves the
previous installation; the error identifies the retained build log.

Select TIRust in a generated ``Cards/MadLoopParams.dat``::

    #MLReductionLib
    9|6|7|1

This example starts with TIRust and allows the existing backends to rescue
unstable points. ``9`` selects TIRust alone. CLI checks accept
``--reduction=9``. Keep the actual backend/precision counters when comparing
performance, since a rescued point may have used a different library.

Controls
--------

.. list-table::
   :header-rows: 1
   :widths: 32 16 52

   * - Parameter
     - Default
     - Meaning
   * - ``LoopPoleMode``
     - ``-1``
     - Legacy behaviour; for TIRust this retains UV and IR. Explicit modes
       ``0``, ``1``, ``2``, ``3`` select both, UV, IR, or neither.
   * - ``TIRustComputeFinite``
     - ``.TRUE.``
     - Compute the finite tensor coefficient. Disabling it is only for
       explicit tensor diagnostics, not a complete physical amplitude.
   * - ``UseQPIntegrandForTIRust``
     - ``.TRUE.``
     - Recompute the numerator in quadruple precision during QP rescue.
   * - ``TIRustGlobalCache``
     - ``-1``
     - Enable all qualified topologies. ``0`` disables reuse; a positive
       value limits reuse to that propagator count.
   * - ``TIRustCacheMaxBytes``
     - ``268435456``
     - Per-context numerical cache budget. A full cache stops insertion
       without changing the computed values.

``LoopPoleMode`` retains ordinary finite terms, including rational terms,
and does not change counterterms or R2. All selected rescue backends must
support an explicit UV/IR selection. For example, ``9|8|7`` can be used with
``LoopPoleMode=1``. Selecting no poles while also disabling finite evaluation
is an invalid request.

Profiling
---------

After installing both libraries, run one ``check profile`` command for each
timed backend::

    import model loop_sm
    check profile -reuse 10 u u~ > z [virt=QCD] --reduction=8 --loop_pole_mode=0 --tirust_qp_integrand=true --tirust_cache=-1 --seed=20261112
    check profile -reuse 10 u u~ > z [virt=QCD] --reduction=9 --loop_pole_mode=0 --tirust_qp_integrand=true --tirust_cache=-1 --seed=20261112

A reduction list times only its first backend; the separate stability analysis
can compare its members. The integer ``10`` selects the stability sample,
not the timing count. Compare full helicity-sum times. ``FULL−SKIP`` is an
estimated reduction time from separate full and numerator-only runs.
Explicit pole mode 0 includes UV/IR and all helicities; the configured single
helicity is unverified and may contribute zero. Reused-export initialization
may be unavailable rather than zero.

Stock timing uses fixed-DP MadLoop mode. TIRust may internally retry DP-input
tensors at 106-bit DD, 256 or 512 bits, so accepted 219 is a transport code, not
proof of native-DP-only arithmetic. Genuine QP-input transport 323 remains
separate from recomputing a QP numerator, which also requires the effective
``UseQPIntegrandForTIRust`` card. The stock driver does not emit optional route
counters; its timed internal routes remain unknown. Bridge counters also do
not cover coefficients reused by Fortran caches.

``docs/benchmarkfolder/LOCAL_FINAL_VALIDATION_TODO.md`` gives the exact current
paired full-amplitude acceptance workflow with natural stability/rescue.
Its measurements and initialization remain separate from stock profiling.

Poles-only diagnostics
----------------------

The poles-only executor uses oneloopmaster-generated pole kernels and
truncated tensor arithmetic. It preserves UV/IR separation, including
scaleless cancellations, and double-pole contributions to simple poles.
It does not silently compute the full result and discard the finite part.

Generated subprocesses expose ``<prefix>TIRUST_TENSOR_DIAGNOSTIC`` and its
``MP_TIRUST_TENSOR_DIAGNOSTIC`` counterpart. With
``TIRustComputeFinite=.FALSE.`` they return the selected simple/double-pole
tensor coefficients, set the unevaluated finite slot to NaN, and report
success through ``STABLE``. They perform no numerator contraction or
counterterm addition. Normal amplitude and Monte Carlo entrypoints reject
this partial mode with a nonzero error exit.

The initial interface domain is ordinary undotted integrals with one through
six propagators and ranks zero through six. Unsupported requests are
reported to the reduction dispatcher. The QP bridge preserves binary128
input values and evaluates with 256-bit MPFR working precision; final
numerator contraction remains in Fortran QP. Independent MadLoop checks
remain responsible for deciding numerical stability.

Exports and reproducibility
---------------------------

Standalone and NLO exporters include the Fortran adapter and an isolated
TIRust shared library. ``set output_dependencies internal`` copies the
required artifacts into the generated output and uses relative runtime
library paths. Keep these artifacts together when moving an output.

The installation manifest records source revisions, patch and native-artifact
hashes, ABI version, precision capabilities, compiler and runtime link flags.
Its dependency pins intentionally remain separate from the existing NinjaRS
installation. Upstream dependency licensing requirements still apply.

The process matrix, raw evidence, Typst source, and report reproduction
instructions are in ``docs/benchmarkfolder``. The prior NinjaRS study remains
in ``doc/benchmarkfolder``. A benchmark report marked incomplete is not a
completed numerical or performance qualification.
