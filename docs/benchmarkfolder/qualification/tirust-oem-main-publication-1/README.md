# TIRust OEM publication on main

TIRust's [`main` publication](https://github.com/huashengshao/TIRust/commit/de60dc480fec6475d2373a533c64d1c40ce23b2e)
contains the scoped OEM activation. Its required functional validation passed
on parent `54263c08a1400612eebbd3a0018bf38e26fca9c5`: 37 preflight, 662
core/masters and 708 default-member tests, with zero failures. Core and default
retain the same 31 ignored cases. All suites ran without a personal Symbolica
key; all 466 tracked file hashes remained stable and cleanup left no owned
processes. The [summary](summary.json) retains stage/log hashes and scope.

The published child changes only upstream benchmark-summary metadata. It
preserves the qualified parent's numerical sources and private signing module,
as well as all historical measurement fields. The upstream benchmark PDF
rebuild is byte-identical (`7fd799fbd08802101524ee20029d672ccfc35f71541f2b2f58576f8681962841`).
This records functional validation and publication; no performance campaign is
represented by these Rust tests or their elapsed execution times.

MadGraph intentionally continues to pin signed predecessor
`10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1`, now an ancestor on upstream main.
Its [separate bridge and physical smoke qualification](../tirust-oem-signing-1/summary.json)
retains that exact source identity. Newer upstream triangle-rotation changes
remain present on main and are not silently imported into MadGraph's validated
numerical baseline. The earlier summary's pending-main statement is its
historical snapshot; this publication note supersedes that statement only.

The MadGraph report, frozen evidence and earlier handoffs are unchanged by this
publication. Physical-stress accuracy, new signed-source full-amplitude
qualification and precise local runtime acceptance remain open. The integration
goal is not complete.
