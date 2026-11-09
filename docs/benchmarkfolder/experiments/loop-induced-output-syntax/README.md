# Preserved initial loop-induced command

`baseline-li_gg_h.mg5` is the initial standalone command file using
`generate g g > h [noborn=QCD]`. The standalone campaign subsequently used
`[sqrvirt=QCD]`, the explicit squared-loop-amplitude mode. The current retained
command and successful generation record are in
`../../commands/baseline-li_gg_h.mg5` and
`../../inputs/baseline/li_gg_h/generation.json`.

Only the initial command file was preserved in this folder. Its original CLI
stdout/stderr and exit status were not retained, so this evidence cannot
establish the exact error text or claim that a numerical comparison ran. The
matrix continues to identify the intended loop-induced physical process;
the successful command records the standalone syntax actually executed.

Recompute the old/new command hashes and verify the successful command against
its generation record with
`python3 docs/benchmarkfolder/experiments/audit_inactive_preparations.py` from the
repository root after sourcing `tools/tirust_env.sh`.
