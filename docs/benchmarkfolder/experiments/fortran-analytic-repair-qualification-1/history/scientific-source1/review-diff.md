# Narrow review delta

Only `ML5_0_MP_ORIG_IMPROVE_PS_POINT_PRECISION` gains code. There are three
insertions: local binary128 guard/scratch declarations, the sequential guarded
analytic branch after the original discriminant, and label90 immediately before
the original transverse assignments. No original line is removed or changed.
`prepare.reconstruct` removes these exact insertions and recovers the measured
source byte for byte, including all sanity/close/error/warning/fallback routines
and the complete PSMC implementation.

The branch requires zero incoming masses and transverse components, zero final
transverse sum, finite totals/discriminant, the physical `E>abs(z)` branch and
finite positive `E+z` then `E-z`. No logical short-circuit behavior is assumed.
It replaces the expanded incoming shift evaluation with the algebraically
equivalent half-sum/half-difference projection, then rejoins the original code.

This is a private, unapplied GNU binary128 source candidate. Pure exact-rational
identities and guard tests have no compiled-code or performance authority.
All native build/parity/stability/timing gates remain unadmitted.
