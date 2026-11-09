# Bounded full-versus-poles tensor timing

This diagnostic compares full and poles-only evaluation of identical warmed
tensor inputs. It does not qualify physical-amplitude runtime or assume that
omitting finite work is universally faster.

Twelve settings are fixed before execution: DP and QP for a massive bubble
(N=2, rank 4), an infrared triangle (N=3, rank 3), a massive box (N=4, rank 4),
and the retained massive pentagon and hexagon (rank 2). These use the existing
generated `qcd_n4_uux_zzgg` adapter, whose rank capacity is five. Two additional
settings use the public Fortran bridge directly for a massive N=1, rank-six
tensor. They are labeled separately; no generated rank capacity is changed.
Prior DP pentagon/hexagon refusals are retained if they recur.

The historical installed DSO is pinned by SHA256
`49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12`
and copied into an isolated prefix before compiling the diagnostic driver.
The Fortran archive, module, configuration, dependencies, generated adapter,
executable and actual resolved DSO are recorded. Production files and the
installed prefix remain untouched.

Each setting runs in one process on one pinned CPU under the shared numerical
lock. Plans are initialized before measurement, numerical caches are disabled,
and both modes receive fresh input arrays for every call. Each mode calibrates
its own fixed loop count with at most four batches toward 1.1 seconds. Seven
pairs then alternate full-first and poles-first order. Wall and process CPU
times are retained for each batch. Coefficient output and comparisons occur
outside the timer. A failed preflight or calibration does not produce a speed
claim; all raw results and refusals remain in the report.

All returned full/pole coefficients are compared at every recorded batch.
Generated-adapter partial finite coefficients must be NaN+0i. The direct bridge
uses a zero placeholder for its omitted finite coefficient, which is explicitly
distinguished from the generated adapter's NaN contract. Pole comparisons use
the existing scaled 1e-10 DP / 1e-24 QP bounds. Batch timing checks and paired
log-ratio 95% intervals are reported independently of correctness. Slower
partial evaluations remain visible. CPU affinity, host-load snapshots and
wall-versus-CPU time document possible shared-host interference.
