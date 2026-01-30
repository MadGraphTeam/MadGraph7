"""
This module contains functions and classes to train neural importance sampling networks and
evaluate the integration and sampling performance.
"""

from .buffer import Buffer
from .channel_grouping import ChannelData, ChannelGroup, ChannelGrouping
from .distribution import Distribution
from .integrand import Integrand
from .integrator import Integrator, SampleBatch, TrainingStatus
from .interface import (
    MADNIS_INTEGRAND_FLAGS,
    IntegrandDistribution,
    IntegrandFunction,
    build_madnis_integrand,
)
from .losses import (
    kl_divergence,
    multi_channel_loss,
    rkl_divergence,
    stratified_variance,
    variance,
)

__all__ = [
    "Integrator",
    "TrainingStatus",
    "SampleBatch",
    "Integrand",
    "Buffer",
    "multi_channel_loss",
    "stratified_variance",
    "variance",
    "kl_divergence",
    "rkl_divergence",
    "ChannelGroup",
    "ChannelData",
    "ChannelGrouping",
    "Distribution",
    "MADNIS_INTEGRAND_FLAGS",
    "IntegrandDistribution",
    "IntegrandFunction",
    "build_madnis_integrand",
]
