# Contingent port plan

This document records a proposal, not a production change or an authorization
to promote. Normal-path overhead and the remaining actual controls must
support retention first.

1. Complete actual rotation-2 checks on retained QCD3 points, including a
   recovered DP point and failures that still require genuine QP. Run the
   corrected generated loop-induced `control-gg-h` and mixed-order
   `control-mixed` processes. Compare every split-order component against the
   retained independent backends using the unchanged prescribed tolerance.
2. Run seven alternating ordinary/retry pairs for QCD1 and EW2 on quiet CPU207,
   using the same generated objects, cards, points and GNU-first runtime link.
   Require all ordinary points to remain native DP with zero retries and the
   same physical results/statuses. Retain loop counts, every run, paired
   statistics, and external-load limitations. This measures added retry
   machinery overhead; it is not a substitute for the final eight-case user
   performance gate.
3. Only after independent review, port the guarded numerator and the separate
   lazy context entry to `vendor/ninjars`. Ordinary `DpState` and the ordinary
   DP entry remain unchanged. Promote the explicit ABI with version 3 in Rust,
   the C header, Fortran context creation, and installer metadata. Preserve
   all length checks, failure transactionality, exact MP transport and
   context lifecycle. Keep the original callback fallback for unsupported
   layouts, unsafe exponents, and unsupported hardware environments.
4. Extend the installed smoke test to exercise the explicit compensated
   entry, then return to ordinary DP on the retained context, followed by
   existing genuine MP checks. `ninjars_install.read_config` centrally rejects
   ABI2 with the existing actionable `install ninjars --force` message.
   Installer/export metadata consumers already use this check; no exporter
   ownership overlap is needed.
5. Port the wrapper and once-only retry branch to the optimized
   `loop_matrix_standalone.inc`, conditional on NinjaRS availability. Preserve
   the public signature and all early returns through the wrapper. Select the
   explicit routine in `NINJARS_interface.inc` only during retry. Initialize
   per-process retry COMMON state in a linked `BLOCK DATA` unit in the
   interface object so direct wrapper calls before `SLOOPMATRIX` are native.
   Do not change thresholds, direction selection, rotations, hard-error/FPE
   handling, other backend ordering, or genuine QP fallback.
6. Test present/absent-library templates, direct-wrapper initialization,
   rotations, backend chains, consecutive points, and explicit/native
   context interleaving. Reinstall and freshly export both standalone and
   full NLO outputs; revalidate relocation and actual poles. Root coordinates
   final documentation, source proof, and the eight-case benchmark.

Ownership, if promoted: this agent handles `vendor/ninjars`,
`madgraph/various/ninjars_install.py`, the two NinjaRS-related optimized
templates, and their focused tests. The Fortran agent owns polynomial/HELAS
generation and `loop_exporters.py`; root owns final documentation and
installation/export coordination. Production files remain untouched while
this proposal is evaluated.
