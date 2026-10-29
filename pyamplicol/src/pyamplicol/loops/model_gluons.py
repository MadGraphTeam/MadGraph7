# SPDX-License-Identifier: 0BSD
"""Bounded source-gluon tensor controls for the initial fixed-SU(3) model path.

The exact expressions below were compared with generated ALOHA and MadGraph
colour algebra. No coupling coefficient or general-Nc continuation is inferred.
Ordinary representatives additionally require exact native basis/coupling
identity, including ordered particle ports, before they can be reused by a CT.
"""

from __future__ import annotations

_TWO_POINT = (
    "P(-1,1)*P(-1,1)*Metric(1,2)",
    "P(1,1)*P(2,1)",
    "Metric(1,2)",
)
_FOUR_R2_LORENTZ = (
    "Metric(1,4)*Metric(2,3)",
    "Metric(1,3)*Metric(2,4)",
    "Metric(1,2)*Metric(3,4)",
    "Metric(1,4)*Metric(2,3)+Metric(1,3)*Metric(2,4)+Metric(1,2)*Metric(3,4)",
)
_FOUR_R2_COLOR = (
    "d(-1,1,3)*d(-1,2,4)",
    "d(-1,1,3)*f(-1,2,4)",
    "d(-1,1,4)*d(-1,2,3)",
    "d(-1,1,4)*f(-1,2,3)",
    "d(-1,2,3)*f(-1,1,4)",
    "d(-1,2,4)*f(-1,1,3)",
    "f(-1,1,2)*f(-1,3,4)",
    "f(-1,1,3)*f(-1,2,4)",
    "f(-1,1,4)*f(-1,2,3)",
    "Identity(1,2)*Identity(3,4)",
    "Identity(1,3)*Identity(2,4)",
    "Identity(1,4)*Identity(2,3)",
)
_ORDINARY_COLOR = (
    "f(1,2,3)",
    "f(-1,1,2)*f(3,4,-1)",
    "f(-1,1,3)*f(2,4,-1)",
    "f(-1,1,4)*f(2,3,-1)",
)
_ORDINARY_LORENTZ = (
    "P(3,1)*Metric(1,2)-P(3,2)*Metric(1,2)-P(2,1)*Metric(1,3)"
    "+P(2,3)*Metric(1,3)+P(1,2)*Metric(2,3)-P(1,3)*Metric(2,3)",
    "Metric(1,4)*Metric(2,3)-Metric(1,3)*Metric(2,4)",
    "Metric(1,4)*Metric(2,3)-Metric(1,2)*Metric(3,4)",
    "Metric(1,3)*Metric(2,4)-Metric(1,2)*Metric(3,4)",
)
