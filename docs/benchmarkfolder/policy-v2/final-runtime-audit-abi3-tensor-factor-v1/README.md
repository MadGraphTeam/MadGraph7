# Held final runtime audit

`audit.py` is the prior read-only runtime audit with only its default generation namespace changed to `generation-abi3-tensor-factor-v1`. Run after all 13 exports/builds complete. It rehashes generated sources, clean executables and all recorded dependencies, then requires the eight-target representative assumption of one exact shared runtime group per backend. A split remains a failure and requires additional complete DP/QP coverage before proof. No numerical evaluation is performed.

`check-reuse.py` then checks every generated model postimage against the existing correction and every numerical runtime signature/module/link-metadata hash against the prior final inventory. A change stops reuse and requires explicit updated ABI controls and coverage. The new candidate source proof remains mandatory. Its result is written separately as `reuse-check.json`.
