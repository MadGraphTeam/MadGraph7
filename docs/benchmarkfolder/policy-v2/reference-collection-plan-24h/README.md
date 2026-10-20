# Full-eight corrected-reference retry

The original full-eight attempt remains failed: QCD4 CutTools did not reach READY within 7200 seconds. The first six cases and every rejected snapshot remain in their original namespace. This retry starts all eight cases afresh, with all 100 points, all three corrected-original backends, two persistent commands, CPU 207, and unchanged strict numerical settings. It does not extract or relabel the completed prefix.

The only collection-control differences are the run ID/output namespace and an explicit 86400-second timeout for each worker phase (and any configured fresh subprocess). Configuration bytes, point bytes, correction, frozen collector, worker, and policy sources are unchanged. Repository-local TMPDIR is explicitly restored after sourcing the toolchain.

Launch requires root scheduling coordination. `bash command.sh` performs the ordinary frozen collector, including its normal rebuilds; it starts no additional fresh-parity evaluations. A successful complete run must still qualify through the frozen validators and receive a new runtime binding. No benchmark acceptance claim is made by this plan.
