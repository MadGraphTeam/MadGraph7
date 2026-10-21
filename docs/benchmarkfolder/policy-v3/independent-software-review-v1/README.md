# Independent v3 software review

The exact final source versions passed the combined **73/73 MG7 software tests**
on CPU4, with zero skips, failures or errors: 28 policy, 22 pair-controller and
23 report tests. The raw unittest footer reports 27.971 seconds; the owned runner
wall time was 28.453 seconds. Source hashes before and after agree exactly.

These tests use mocked numerical data and synthetic Typst renders. They execute
no amplitude, model import, numerical-library build, MP campaign or real paired
benchmark. Typst compilation is an explicit software-test dependency. No temporary
PDF, PNG, cache, executable, library, private reference asset or active core-run
file is included here. The finalizer's separately reported four synthetic tests
are not added to the combined73 count.

`review.json` records the bounded source-review findings and repaired boundaries.
The accepted calibrations remain numerical references with zero timing repetitions;
failed parent/external observations are preserved. The original-Ninja denominator
has a separate same-outcome gate. Missing physics, runtime qualification, profiling,
seven-pair timing and final publication evidence remain pending. This is approval
of a campaign-tool software milestone, not achievement of the integration goal.

Run `python -B docs/benchmarkfolder/policy-v3/independent-software-review-v1/verify.py`
for a standard-library-only check of the retained hashes/source versions/footer.
`command.sh` records the real combined test command, using the retained developer
environment for Python, Typst and PDF text tools. Existing owner proofs retain
individual attempts, authentic-old-metadata rehearsals and visual checks separately.
