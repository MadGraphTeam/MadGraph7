# SPDX-License-Identifier: 0BSD
"""Shared-recursion one-loop engine v2 (docs/development/loop_v2/DESIGN.md).

The v2 engine never enumerates loop or counterterm Feynman diagrams. Its
modules build process-blind kernels (``catalogue``), tree blocks, loop-prefix
families and schedules that NinjaRS reduces family by family.
"""
