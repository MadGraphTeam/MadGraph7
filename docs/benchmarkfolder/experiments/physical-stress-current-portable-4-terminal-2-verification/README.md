# Terminal stress capsule portability checks

This separate supplement verifies the closed terminal-2 capsule together with its immutable preparation sibling. The checks use Python only: no CLI, compiler, loader, amplitude, or subprocess execution.

The full verifier runs through an exact in-memory indexed callback while filesystem reads are denied, then on two relocated siblings while workspace reads are denied. Negative checks reject changed bytes, a missing preparation dependency, an incomplete declared path closure, reserved-namespace shadowing, and a forged all-lanes agreement summary. A final comparison proves terminal-2 changed the public JSON serialization of terminal-1 without changing the numerical records, derived summary, postclosure bindings, verifier, or policy.

Both source packets remain unchanged. Their integrity and the four completed host checks do not establish physical numerical agreement: the retained threshold single-pole discrepancies remain open, with one deferred lane and nineteen unstarted lanes.

```sh
source tools/tirust_env.sh
python -B docs/benchmarkfolder/experiments/physical-stress-current-portable-4-terminal-2-verification/check.py
```

The check writes its result once, preserving the actual execution receipt separately. This supplement does not change either evidence capsule or the frozen current campaign.
