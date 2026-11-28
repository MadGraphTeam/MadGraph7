"""The adaptive discrete choice and the virtual trick of the Python driver,
the rules the iteration states of the C++ integrands follow
(NLOIterationState)."""

import numpy as np


class DiscreteChoice:
    """The adaptive choice of one of count options (the FKS configuration), as
    MC_integer of MadFKS: probabilities proportional to the mean absolute
    integrand of each option, mixed with the uniform choice by floor, so
    that every option keeps being sampled. The integrand of an option is the
    part of the weight that depends on the choice (for FKS the event and its
    counter-events, without the n-body part) times the probability of its
    choice."""

    def __init__(self, count, floor=0.1):
        self.probabilities = np.full(count, 1.0 / count)
        self.floor = floor
        self.sums = np.zeros(count)
        self.counts = np.zeros(count)

    def sample(self, rng, points):
        index = rng.choice(len(self.probabilities), size=points, p=self.probabilities)
        return index, 1.0 / self.probabilities[index]

    def add(self, index, integrands):
        n = len(self.probabilities)
        self.sums += np.bincount(index, weights=np.abs(integrands), minlength=n)
        self.counts += np.bincount(index, minlength=n)

    def adapt(self):
        seen = self.counts > 0
        if not seen.any():
            return
        target = np.zeros_like(self.sums)
        target[seen] = self.sums[seen] / self.counts[seen]
        if target.sum() > 0:
            target /= target.sum()
            uniform = np.full_like(target, 1.0 / len(target))
            self.probabilities = (1 - self.floor) * target + self.floor * uniform
        self.sums[:] = 0.0
        self.counts[:] = 0.0


class VirtualTrick:
    """The virtual on a fraction of the points of a channel (the virtual
    trick of MadFKS): where it is evaluated (V - r B) / f, plus r B
    everywhere, with r the average of V / B per NLO order and B the sum of
    the Born orders."""

    def __init__(self, orders, fraction, min_fraction):
        self.ratios = np.zeros(orders)
        self.fraction = fraction
        self.min_fraction = min_fraction
        self.virtual_sums = np.zeros(orders)
        self.born_sum = 0.0

    def inputs(self, virtual, born, evaluated):
        approximation = self.ratios[None, :] * born.sum(1)[:, None]
        residual = np.where(
            evaluated[:, None], (virtual - approximation) / self.fraction, 0.0
        )
        return approximation + residual, approximation

    def add(self, virtual, born, evaluated):
        self.virtual_sums += virtual[evaluated].sum(0)
        self.born_sum += born[evaluated].sum()

    def adapt(self, error_virtual, error_total):
        if self.born_sum != 0:
            self.ratios = self.virtual_sums / self.born_sum
        self.virtual_sums[:] = 0.0
        self.born_sum = 0.0
        if error_total > 0:
            factor = max(min(2.0 * error_virtual / error_total, 2.0), 0.25)
            self.fraction = max(min(self.fraction * factor, 1.0), self.min_fraction)
