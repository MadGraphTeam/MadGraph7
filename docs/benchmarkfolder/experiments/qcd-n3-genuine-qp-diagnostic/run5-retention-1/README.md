This packet retains every output from terminal run5, including all six new raw
native journals, stdout/stderr, parameter/input logs, warnings, mapped-runtime
captures and snapshots, strict loader traces, alias guards, parsed records,
combined adjudication and actual result. It also retains every sealed source5
payload. Large text uses deterministic gzip with raw hashes preserved; runtime
symlink targets are metadata only, and no executable, DSO or archive is copied.

The ten inherited native workers remain in the immutable sibling
`failed-run4-retention-1`, index `fcd4f753…`. Keep both packages as siblings or
pass an explicit prefix path to `verify(root, prefix_root)`. The verifier reads
only these indexed packages. It verifies the original failure and runtime
proofs, replays all ten inherited raw traces and all six new raw traces, then
reproduces the exact combined sixteen-worker adjudication.

All sixteen workers satisfy the diagnostic protocol and common-input gates;
the original run4 remains failed. Every fixed-mode amplitude is unchecked
status120/accuracy-1. Agreement among these returned values is consistency,
not a true-error estimate, independent-oracle validation, ordinary-campaign
acceptance or a timing benchmark. Captured elapsed seconds establish only that
the original combined resource budget was respected.
