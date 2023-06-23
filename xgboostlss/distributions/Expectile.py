import torch
from torch.distributions import Distribution
from scipy.stats import norm
from ..utils import *
from .distribution_utils import DistributionClass
import numpy as np

from typing import List


class Expectile(DistributionClass):
    """
    Expectile distribution class.

    Distributional Parameters
    -------------------------
    expectile: List
        List of specified expectiles.

    Parameters
    -------------------------
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    expectiles: List
        List of expectiles in increasing order.
    penalize_crossing: bool
        Whether to include a penalty term to discourage crossing of expectiles.
    """
    def __init__(self,
                 stabilization: str = "None",
                 expectiles: List = [0.1, 0.5, 0.9],
                 penalize_crossing: bool = False,
                 ):
        # Set the parameters specific to the distribution
        distribution = Expectile_Torch
        torch.distributions.Distribution.set_default_validate_args(False)
        expectiles.sort()
        param_dict = {}
        for expectile in expectiles:
            key = f"expectile_{expectile}"
            param_dict[key] = identity_fn

        # Specify Distribution Class
        super().__init__(distribution=distribution,
                         univariate=True,
                         discrete=False,
                         n_dist_param=len(param_dict),
                         stabilization=stabilization,
                         param_dict=param_dict,
                         distribution_arg_names=list(param_dict.keys()),
                         loss_fn="nll",
                         tau=torch.tensor(expectiles),
                         penalize_crossing=penalize_crossing
                         )


class Expectile_Torch(Distribution):
    """
    PyTorch implementation of expectiles.

    Arguments
    ---------
    expectiles : List[torch.Tensor]
        List of expectiles.
    penalize_crossing : bool
        Whether to include a penalty term to discourage crossing of expectiles.
    """
    def __init__(self,
                 expectiles: List[torch.Tensor],
                 penalize_crossing: bool = False,
                 ):
        super(Expectile_Torch).__init__()
        self.expectiles = expectiles
        self.penalize_crossing = penalize_crossing
        self.__class__.__name__ = "Expectile"

    def log_prob(self, value: torch.Tensor, tau: List[torch.Tensor]) -> torch.Tensor:
        """
        Returns the log of the probability density function evaluated at `value`.

        Arguments
        ---------
        value : torch.Tensor
            Response for which log probability is to be calculated.
        tau : List[torch.Tensor]
            List of asymmetry parameters.

        Returns
        -------
        torch.Tensor
            Log probability of `value`.
        """
        value = value.reshape(-1, 1)
        loss = torch.tensor(0.0, dtype=torch.float32)
        penalty = torch.tensor(0.0, dtype=torch.float32)

        # Calculate loss
        predt_expectiles = []
        for expectile, tau_value in zip(self.expectiles, tau):
            weight = torch.where(value - expectile >= 0, tau_value, 1 - tau_value)
            loss += torch.nansum(weight * (value - expectile) ** 2)
            predt_expectiles.append(expectile.reshape(-1, 1))

        # Penalty term to discourage crossing of expectiles
        if self.penalize_crossing:
            predt_expectiles = torch.cat(predt_expectiles, dim=1)
            penalty = torch.mean(
                (~torch.all(torch.diff(predt_expectiles, dim=1) > 0, dim=1)).float()
            )

        loss = (loss * (1 + penalty)) / len(self.expectiles)

        return -loss


def expectile_pnorm(tau: np.ndarray = 0.5,
                    m: np.ndarray = 0,
                    sd: np.ndarray = 1
                    ):
    """
    Normal Expectile Distribution Function.
    For more details and distributions see https://rdrr.io/cran/expectreg/man/enorm.html

    Arguments
    _________
    tau : np.ndarray
        Vector of expectiles from the respective distribution.
    m : np.ndarray
        Mean of the Normal distribution.
    sd : np.ndarray
        Standard deviation of the Normal distribution.

    Returns
    _______
    tau : np.ndarray
        Expectiles from the Normal distribution.
    """
    z = (tau - m) / sd
    p = norm.cdf(z, loc=m, scale=sd)
    d = norm.pdf(z, loc=m, scale=sd)
    u = -d - z * p
    tau = u / (2 * u + z)

    return tau


def expectile_norm(tau: np.ndarray = 0.5,
                   m: np.ndarray = 0,
                   sd: np.ndarray = 1):
    """
    Calculates expectiles from Normal distribution for given tau values.
    For more details and distributions see https://rdrr.io/cran/expectreg/man/enorm.html

    Arguments
    _________
    tau : np.ndarray
        Vector of expectiles from the respective distribution.
    m : np.ndarray
        Mean of the Normal distribution.
    sd : np.ndarray
        Standard deviation of the Normal distribution.

    Returns
    _______
    np.ndarray
    """
    tau[tau > 1 or tau < 0] = np.nan
    zz = 0 * tau
    lower = np.array(-10, dtype="float")
    lower = np.repeat(lower[np.newaxis, ...], len(tau), axis=0)
    upper = np.array(10, dtype="float")
    upper = np.repeat(upper[np.newaxis, ...], len(tau), axis=0)
    diff = 1
    index = 0
    while (diff > 1e-10) and (index < 1000):
        root = expectile_pnorm(zz) - tau
        root[np.isnan(root)] = 0
        lower[root < 0] = zz[root < 0]
        upper[root > 0] = zz[root > 0]
        zz = (upper + lower) / 2
        diff = np.nanmax(np.abs(root))
        index = index + 1
    zz[np.isnan(tau)] = np.nan

    return zz * sd + m
