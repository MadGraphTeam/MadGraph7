# Corrected-original core preparation

`prepare.py` calls the frozen corrected-reference preparation helper for all
eight core processes. It copies the anchored original exports, applies only the
verified MODEL postimages, removes/rebuilds every MODEL object/archive and
subprocess object/executable, and checks unchanged original DHELAS/reduction
libraries. It starts no numerical evaluations.

Run from the repository after sourcing `.ninjars-build/dev-env.sh`:

```sh
python docs/benchmarkfolder/policy-v2/reference-preparation/prepare.py
```

For a quiet timing window, create
`.benchmark-work/policy-v2/prepared-references/pause-before-next-case`.
The current build finishes and preparation exits before the next case. Remove
that file and rerun the same command to resume; completed cases are revalidated.

Numerical collection remains separately authorized. The frozen input-root
adapter can reuse these outputs with `representative:true`, but that mode also
requires another fresh-process parity evaluation. The ordinary frozen
corrected-reference collector can run two commands without fresh parity, at the
cost of repeating its clean preparation/build. These mechanical constraints are
recorded explicitly; preparation does not bypass either guard.

The final reference cohort uses all 100 points and all three legacy backends,
bound as `original_helper_corrected`. Its times are diagnostic only. The common
core configuration stays `MLStabThres=1e-9`, Collier accuracy `1e-12`.
