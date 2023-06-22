# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import softplus

from torch.distributions import NegativeBinomial, Poisson, Gamma, LogNormal, Beta
from pyro.distributions import TorchDistribution
from pyro.distributions.util import broadcast_shape
from pyro.distributions.util import is_identically_one, is_identically_zero


class ZeroInflatedDistribution(TorchDistribution):
    """
    Generic Zero Inflated distribution.

    This can be used directly or can be used as a base class as e.g. for
    :class:`ZeroInflatedPoisson` and :class:`ZeroInflatedNegativeBinomial`.

    Parameters
    ----------
    gate : torch.Tensor
        Probability of extra zeros given via a Bernoulli distribution.
    base_dist : torch.distributions.Distribution
        The base distribution.

    Source
    ------
    https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/zero_inflated.py#L18
    """

    arg_constraints = {
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }

    def __init__(self, base_dist, *, gate=None, gate_logits=None, validate_args=None):
        if (gate is None) == (gate_logits is None):
            raise ValueError(
                "Either `gate` or `gate_logits` must be specified, but not both."
            )
        if gate is not None:
            batch_shape = broadcast_shape(gate.shape, base_dist.batch_shape)
            self.gate = gate.expand(batch_shape)
        else:
            batch_shape = broadcast_shape(gate_logits.shape, base_dist.batch_shape)
            self.gate_logits = gate_logits.expand(batch_shape)
        if base_dist.event_shape:
            raise ValueError(
                "ZeroInflatedDistribution expected empty "
                "base_dist.event_shape but got {}".format(base_dist.event_shape)
            )

        self.base_dist = base_dist.expand(batch_shape)
        event_shape = torch.Size()

        super().__init__(batch_shape, event_shape, validate_args)

    @constraints.dependent_property
    def support(self):
        return self.base_dist.support

    @lazy_property
    def gate(self):
        return logits_to_probs(self.gate_logits)

    @lazy_property
    def gate_logits(self):
        return probs_to_logits(self.gate)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        zero_idx = (value == 0)
        support = self.support
        epsilon = abs(torch.finfo(value.dtype).eps)

        if hasattr(support, "lower_bound"):
            if is_identically_zero(getattr(support, "lower_bound", None)):
                value = value.clamp_min(epsilon)

        if hasattr(support, "upper_bound"):
            if is_identically_one(getattr(support, "upper_bound", None)) & (value.max() == 1.0):
                value = value.clamp_max(1 - epsilon)

        if "gate" in self.__dict__:
            gate, value = broadcast_all(self.gate, value)
            log_prob = (-gate).log1p() + self.base_dist.log_prob(value)
            log_prob = torch.where(zero_idx, (gate + log_prob.exp()).log(), log_prob)
        else:
            gate_logits, value = broadcast_all(self.gate_logits, value)
            log_prob_minus_log_gate = -gate_logits + self.base_dist.log_prob(value)
            log_gate = -softplus(-gate_logits)
            log_prob = log_prob_minus_log_gate + log_gate
            zero_log_prob = softplus(log_prob_minus_log_gate) + log_gate
            log_prob = torch.where(zero_idx, zero_log_prob, log_prob)
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            mask = torch.bernoulli(self.gate.expand(shape)).bool()
            samples = self.base_dist.expand(shape).sample()
            samples = torch.where(mask, samples.new_zeros(()), samples)
        return samples

    @lazy_property
    def mean(self):
        return (1 - self.gate) * self.base_dist.mean

    @lazy_property
    def variance(self):
        return (1 - self.gate) * (
                self.base_dist.mean**2 + self.base_dist.variance
        ) - self.mean**2

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)
        gate = self.gate.expand(batch_shape) if "gate" in self.__dict__ else None
        gate_logits = (
            self.gate_logits.expand(batch_shape)
            if "gate_logits" in self.__dict__
            else None
        )
        base_dist = self.base_dist.expand(batch_shape)
        ZeroInflatedDistribution.__init__(
            new, base_dist, gate=gate, gate_logits=gate_logits, validate_args=False
        )
        new._validate_args = self._validate_args
        return new


class ZeroInflatedPoisson(ZeroInflatedDistribution):
    """
    A Zero-Inflated Poisson distribution.

    Parameter
    ---------
    rate: torch.Tensor
        The rate of the Poisson distribution.
    gate: torch.Tensor
        Probability of extra zeros given via a Bernoulli distribution.

    Source
    ------
    https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/zero_inflated.py#L121
    """
    arg_constraints = {
        "rate": constraints.positive,
        "gate": constraints.unit_interval,
    }
    support = constraints.nonnegative_integer

    def __init__(self, rate, gate=None, validate_args=None):
        base_dist = Poisson(rate=rate, validate_args=False)
        base_dist._validate_args = validate_args

        super().__init__(base_dist, gate=gate, validate_args=validate_args)

    @property
    def rate(self):
        return self.base_dist.rate


class ZeroInflatedNegativeBinomial(ZeroInflatedDistribution):
    """
    A Zero Inflated Negative Binomial distribution.

    Parameter
    ---------
    total_count: torch.Tensor
        Non-negative number of negative Bernoulli trial.
    probs: torch.Tensor
        Event probabilities of success in the half open interval [0, 1).
    logits: torch.Tensor
        Event log-odds of success (log(p/(1-p))).
    gate: torch.Tensor
        Probability of extra zeros given via a Bernoulli distribution.

    Source
    ------
    - https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/zero_inflated.py#L150
    """

    arg_constraints = {
        "total_count": constraints.greater_than_eq(0),
        "probs": constraints.half_open_interval(0.0, 1.0),
        "logits": constraints.real,
        "gate": constraints.unit_interval,
    }
    support = constraints.nonnegative_integer

    def __init__(self, total_count, probs=None, gate=None, validate_args=None):
        base_dist = NegativeBinomial(total_count=total_count, probs=probs, logits=None, validate_args=False)
        base_dist._validate_args = validate_args

        super().__init__(base_dist, gate=gate, validate_args=validate_args)

    @property
    def total_count(self):
        return self.base_dist.total_count

    @property
    def probs(self):
        return self.base_dist.probs

    @property
    def logits(self):
        return self.base_dist.logits


class ZeroAdjustedGamma(ZeroInflatedDistribution):
    """
    A Zero-Adjusted Gamma distribution.

    Parameter
    ---------
    concentration: torch.Tensor
        shape parameter of the distribution (often referred to as alpha)
    rate: torch.Tensor
        rate = 1 / scale of the distribution (often referred to as beta)
    gate: torch.Tensor
        Probability of zeros given via a Bernoulli distribution.

    Source
    ------
    https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/zero_inflated.py
    """
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
        "gate": constraints.unit_interval,
    }
    support = constraints.nonnegative

    def __init__(self, concentration, rate, gate=None, validate_args=None):
        base_dist = Gamma(concentration=concentration, rate=rate, validate_args=False)
        base_dist._validate_args = validate_args

        super().__init__(base_dist, gate=gate, validate_args=validate_args)

    @property
    def concentration(self):
        return self.base_dist.concentration

    @property
    def rate(self):
        return self.base_dist.rate


class ZeroAdjustedLogNormal(ZeroInflatedDistribution):
    """
    A Zero-Adjusted Log-Normal distribution.

    Parameter
    ---------
    loc: torch.Tensor
        Mean of log of distribution.
    scale: torch.Tensor
        Standard deviation of log of the distribution.
    gate: torch.Tensor
        Probability of zeros given via a Bernoulli distribution.

    Source
    ------
    https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/zero_inflated.py
    """
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "gate": constraints.unit_interval,
    }
    support = constraints.nonnegative

    def __init__(self, loc, scale, gate=None, validate_args=None):
        base_dist = LogNormal(loc=loc, scale=scale, validate_args=False)
        base_dist._validate_args = validate_args

        super().__init__(base_dist, gate=gate, validate_args=validate_args)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale


class ZeroAdjustedBeta(ZeroInflatedDistribution):
    """
    A Zero-Adjusted Beta distribution.

    Parameter
    ---------
    concentration1: torch.Tensor
        1st concentration parameter of the distribution (often referred to as alpha).
    concentration0: torch.Tensor
        2nd concentration parameter of the distribution (often referred to as beta).
    gate: torch.Tensor
        Probability of zeros given via a Bernoulli distribution.

    Source
    ------
    https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/zero_inflated.py
    """
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
        "gate": constraints.unit_interval,
    }
    support = constraints.unit_interval

    def __init__(self, concentration1, concentration0, gate=None, validate_args=None):
        base_dist = Beta(concentration1=concentration1, concentration0=concentration0, validate_args=False)
        base_dist._validate_args = validate_args

        super().__init__(base_dist, gate=gate, validate_args=validate_args)

    @property
    def concentration1(self):
        return self.base_dist.concentration1

    @property
    def concentration0(self):
        return self.base_dist.concentration0
