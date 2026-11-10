# Attempt 4: complete generated model-reader inputs

The first attempt-3 worker stopped inside the actual generated `SETPARA`
reader, before `LC_OPEN` or an amplitude call. Its bare `param_card.dat` path
made the generated reader search for `/ident_card.dat`; the Fortran `STOP`
returned exit status zero. The new empty `param.log` separately failed the
strict input guard. All prior source freezes, build products, input bytes,
logs and the failure record remain unchanged. This is a harness correction;
there are no numerical implementation changes or successful lifecycle data
from attempt 3.

The physical driver changes exactly three string arguments: the nominal
path becomes `nominal/param_card.dat` and the alternate path becomes
`alternate/param_card.dat`. Both directories contain the identical nominal
or alternate physical card bytes retained with the failed worker and the
original export's exact `Cards/ident_card.dat`. The generated reader requires
the basename `param_card.dat` to update its parent directory. Model logging
remains enabled; `param.log` is an explicitly declared, retained and hashed
output alongside `records.txt` and `trace.txt`. Every other file must remain
identical, with no new files allowed.

The frozen patch and rendered private drivers precede compilation. Only the
new physical driver and a separate card-reader smoke driver are compiled,
using the exact actual compiler recipe retained from attempt 2 with source
and object names substituted. Fresh plain, observed and smoke executables
use the actual attempt-3 link recipes. The 33 other generated/common
numerical objects, existing observer objects, compile flags, ordered link
inputs and qualified runtime dependencies remain identical and are checked
before and after. All actual linker `LOAD` inputs must be in the prebound
hash inventory. Earlier sources, objects, executables and maps are retained.

Before the four physical workers, one 30-second card-only smoke invokes the
actual generated `SETPARA` reader with nominal, alternate and nominal cards.
It checks the exact binary64 MZ values 91.188, 92 and the original bits. It
requires complete `DONE` and `CLOSED` records, absent module/observer contexts,
and exactly three observed `BEGIN` records with no native reducer export.
This verifies actual card parsing and avoids interpreting another exit-zero
Fortran stop as success. Its separate executable links the same real model
and generated objects; no mock reader or model is used.

After independent source review and root authorization, run from the
repository root, with the pinned environment loaded:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle/attempt4/resume.py freeze
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle/attempt4/resume.py build
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle/attempt4/resume.py smoke
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle/attempt4/resume.py run
```

Freeze is provenance only and does not authorize subsequent stages. CPU 10,
one shared 600-second build deadline, a 30-second smoke and four sequential
180-second physical worker limits apply. Every executable invocation uses
the common numerical lock and the inherited bounded owned-process cleanup.
All outputs and failures are retained; no automatic retry or continuation
follows a failure. The original frozen analyzer and all 19 physical plus 24
tensor-key actions remain unchanged. There is no performance acceptance
claim or rank-six extension in this qualification.
