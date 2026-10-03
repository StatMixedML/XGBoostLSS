# Copyright (c) 2024
# SPDX-License-Identifier: Apache-2.0

import torch
from .zero_one_inflated import ZeroOneInflatedBeta as ZeroOneInflatedBeta_Torch
from .distribution_utils import DistributionClass
from ..utils import *


class ZOIB(DistributionClass):
    """
    Zero-One Inflated Beta (ZOIB) distribution class.

    This distribution is used to model fractional or proportional data on the closed 
    interval [0, 1] that contains exact zeros and exact ones. It is constructed as a 
    mixture of a discrete Bernoulli process (for the boundary values) and a continuous 
    Beta distribution (for values strictly in the open interval (0, 1)).

    Following Ospina and Ferrari (2012), the continuous Beta component is parameterized 
    in terms of its mean and precision, rather than the traditional shape parameters. 
    This allows the XGBoost trees to directly predict the expected value of the 
    continuous fractions.

    Distributional Parameters
    -------------------------
    mu: torch.Tensor
        The expected value (mean) of the continuous Beta distribution component. 
        Must be strictly in the open interval (0, 1).
    phi: torch.Tensor
        The precision parameter of the continuous Beta distribution component. 
        Must be strictly positive. Higher values indicate lower variance.
    gate_zo: torch.Tensor
        The probability that the observation is discrete (exactly 0 or exactly 1).
    gate_one: torch.Tensor
        The conditional probability that the observation is exactly 1, given that 
        it is a discrete observation.

    References
    ----------
    Ospina, R., & Ferrari, S. L. P. (2012). A general class of zero-or-one inflated 
    beta regression models. Computational Statistics & Data Analysis, 56(6), 1609-1623.
    """
    def __init__(self,
                 stabilization: str = "None",
                 response_fn: str = "softplus",
                 loss_fn: str = "nll",
                 initialize: bool = False,
                 ):

        # Input Checks
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError("Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'.")
        if loss_fn not in ["nll"]:
            raise ValueError("Invalid loss function. Please select 'nll'.")
        if not isinstance(initialize, bool):
            raise ValueError("Invalid initialize. Please choose from True or False.")

        # Specify Response Functions for the strictly positive precision parameter
        response_functions = {"exp": exp_fn, "softplus": softplus_fn}
        if response_fn in response_functions:
            response_fn = response_functions[response_fn]
        else:
            raise ValueError("Invalid response function. Please choose from 'exp' or 'softplus'.")

        # Set the PyTorch distribution backend
        distribution = ZeroOneInflatedBeta_Torch
        
        # Map XGBoost linear predictors to the valid parameter support spaces
        param_dict = {
            "mu": sigmoid_fn,        # Mean must be in (0, 1) -> Logit link
            "phi": response_fn,      # Precision must be > 0 -> Log or Softplus link
            "gate_zo": sigmoid_fn,   # Probability must be in [0, 1] -> Logit link
            "gate_one": sigmoid_fn   # Conditional Prob must be in [0, 1] -> Logit link
        }
        
        torch.distributions.Distribution.set_default_validate_args(False)

        # Initialize the agnostic DistributionClass
        super().__init__(distribution=distribution,
                         univariate=True,
                         discrete=False,
                         n_dist_param=len(param_dict),
                         stabilization=stabilization,
                         param_dict=param_dict,
                         distribution_arg_names=list(param_dict.keys()),
                         loss_fn=loss_fn,
                         initialize=initialize)