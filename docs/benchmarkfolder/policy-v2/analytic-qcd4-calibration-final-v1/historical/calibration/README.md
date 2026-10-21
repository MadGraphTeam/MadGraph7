This private supplemental diagnostic checks the retained corrected-original
CutTools qcd4 calibration at 100 phase-space points. It checks 100 total rows
and 100 rows for the sole physical split order; those two rows coincide at each
point. It does not add 200 independent events or change any full-campaign gate.

The independent source/card derivation is
DOUBLEPOLE / Born = -(2 CF + 2 CA) G^2 / (8 pi^2), with CF=4/3 and CA=3.
The actual corrected helper source initializes G=2 sqrt(alpha_s pi), reads
alpha_s=0.118, and uses pi=3.141592653589793 in double precision. Its worker
updates MU_R but does not run alpha_s. The resulting coefficient is exactly
-0.16276245513531162 under the recorded binary64 expression. Archived FKS
getpoles and BinothLHA establish the leading-pole/interference convention;
the actual worker, model-correction postimages, card, generated source and
executable identity are bound in receipt.json. No coefficient was fitted to
the retained numerical answers.

All 200 double-pole checks pass the existing physical tolerance. The largest
relative double-pole residual is 4.646150815304775e-13; the largest fraction of
the allowed tolerance is 4.617779533184807e-6. The worst point is 33. This is
an independent leading-IR proportionality check using the same retained Born,
not an independent Born calculation or a finite/single-pole oracle.

checks.jsonl retains every prediction and residual. recipe.py reproduces the
same source/card reconstruction and raw-row postprocessing with the Python
standard library; it never imports a model, runs a worker or writes outputs.
The recipe is supplied for the root's independent final check; it has not been
run as an additional test by this audit. The original postprocessing itself
completed successfully using the same expressions. No amplitudes, compilation,
model imports, tracked edits or source/cache changes occurred.

The separately written independent source derivation is crosslinked byte for
byte. The audit's derivation and numerical coefficient were obtained before
that crosslink. Only the original-helper CutTools calibration is accepted by
this narrow diagnostic; no final repetition, other backend, timing, numerical
stability or whole-campaign acceptance follows from it.
