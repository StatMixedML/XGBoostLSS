# Copyright (c) 2024
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all, lazy_property
from torch.distributions import Beta
from pyro.distributions import TorchDistribution
from pyro.distributions.util import broadcast_shape

class ZeroOneInflatedDistribution(TorchDistribution):
    """
    Generic Zero-One Inflated distribution base class.
    
    This handles double-bounded fractional data with point masses at exactly 0 and exactly 1.
    
    Parameters
    ----------
    base_dist : torch.distributions.Distribution
        The continuous base distribution defined strictly on (0, 1).
    gate_zo : torch.Tensor
        Probability of the observation being discrete (exactly 0 or 1).
    gate_one : torch.Tensor
        Probability of the observation being exactly 1, conditional on it being discrete.
    """
    arg_constraints = {
        "gate_zo": constraints.unit_interval,
        "gate_one": constraints.unit_interval,
    }

    def __init__(self, base_dist, gate_zo, gate_one, validate_args=None):
        if gate_zo is None or gate_one is None:
            raise ValueError(
                "ZeroOneInflatedDistribution requires both gate_zo and gate_one; got "
                f"gate_zo={gate_zo}, gate_one={gate_one}."
            )
        batch_shape = broadcast_shape(gate_zo.shape, gate_one.shape, base_dist.batch_shape)
        self.gate_zo = gate_zo.expand(batch_shape)
        self.gate_one = gate_one.expand(batch_shape)
        
        if base_dist.event_shape:
            raise ValueError(
                "ZeroOneInflatedDistribution expected empty "
                "base_dist.event_shape but got {}".format(base_dist.event_shape)
            )

        self.base_dist = base_dist.expand(batch_shape)
        event_shape = torch.Size()

        super().__init__(batch_shape, event_shape, validate_args)

    @constraints.dependent_property
    def support(self):
        return constraints.unit_interval

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        zero_idx = (value == 0.0)
        one_idx = (value == 1.0)
        
        epsilon = abs(torch.finfo(value.dtype).eps)
        safe_value = value.clamp(epsilon, 1.0 - epsilon)

        gate_zo, gate_one, value = broadcast_all(self.gate_zo, self.gate_one, value)

        log_prob_0 = torch.log(gate_zo + epsilon) + torch.log1p(-gate_one + epsilon)
        log_prob_1 = torch.log(gate_zo + epsilon) + torch.log(gate_one + epsilon)
        log_prob_cont = torch.log1p(-gate_zo + epsilon) + self.base_dist.log_prob(safe_value)

        log_prob = torch.where(zero_idx, log_prob_0, log_prob_cont)
        log_prob = torch.where(one_idx, log_prob_1, log_prob)

        return log_prob

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            is_zo = torch.bernoulli(self.gate_zo.expand(shape)).bool()
            is_one = torch.bernoulli(self.gate_one.expand(shape)).bool()
            
            samples = self.base_dist.expand(shape).sample()
            discrete_vals = torch.where(is_one, samples.new_ones(()), samples.new_zeros(()))
            samples = torch.where(is_zo, discrete_vals, samples)
            
        return samples

    @lazy_property
    def mean(self):
        prob_1 = self.gate_zo * self.gate_one
        prob_cont = 1.0 - self.gate_zo
        return prob_1 + prob_cont * self.base_dist.mean


class ZeroOneInflatedBeta(ZeroOneInflatedDistribution):
    """
    A Zero-One Inflated Beta (ZOIB) distribution parameterized by Mean and Precision.

    This distribution is used to model fractional or proportional data on the closed 
    interval [0, 1] that contains exact zeros and exact ones. It is constructed as a 
    mixture of a discrete Bernoulli process (for the boundary values) and a continuous 
    Beta distribution (for values strictly in the open interval (0, 1)).

    Following Ospina and Ferrari (2012), the continuous Beta component is parameterized 
    in terms of its mean (mu) and precision (phi), rather than the traditional 
    shape parameters (alpha, beta). This allows for direct regression modeling 
    of the expected value. The standard shape parameters are recovered under the hood 
    as alpha = mu * phi and beta = (1 - mu) * phi.

    Parameters
    ----------
    mu : torch.Tensor
        The expected value (mean) of the continuous Beta distribution component. 
        Must be strictly in the open interval (0, 1).
    phi : torch.Tensor
        The precision parameter of the continuous Beta distribution component. 
        Must be strictly positive. Higher values indicate lower variance.
    gate_zo : torch.Tensor
        The probability that the observation is discrete (exactly 0 or exactly 1).
        Must be in the closed interval [0, 1].
    gate_one : torch.Tensor
        The conditional probability that the observation is exactly 1, given that
        it is a discrete observation. Must be in the closed interval [0, 1].
    validate_args : bool, optional
        Whether to validate input with constraints. Default is None.

    Mathematical Formulation
    ------------------------
    The piecewise probability density function is given by:
    
    P(Y = 0) = gate_zo * (1 - gate_one)
    P(Y = 1) = gate_zo * gate_one
    P(Y = y) = (1 - gate_zo) * Beta(y | mu * phi, (1 - mu) * phi), for y in (0, 1)

    References
    ----------
    Ospina, R., & Ferrari, S. L. P. (2012). A general class of zero-or-one inflated 
    beta regression models. Computational Statistics & Data Analysis, 56(6), 1609-1623.
    """
    arg_constraints = {
        "mu": constraints.unit_interval,
        "phi": constraints.positive,
        "gate_zo": constraints.unit_interval,
        "gate_one": constraints.unit_interval,
    }
    support = constraints.unit_interval

    def __init__(self, mu, phi, gate_zo, gate_one, validate_args=None):
        self.mu_param = mu
        self.phi_param = phi
        
        concentration1 = mu * phi
        concentration0 = (1.0 - mu) * phi

        base_dist = Beta(concentration1=concentration1, concentration0=concentration0, validate_args=False)
        base_dist._validate_args = validate_args

        super().__init__(base_dist, gate_zo=gate_zo, gate_one=gate_one, validate_args=validate_args)

    @property
    def mu(self):
        return self.mu_param

    @property
    def phi(self):
        return self.phi_param