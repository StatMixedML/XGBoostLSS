import torch
from torch.distributions import Distribution
from scipy.stats import norm
from xgboostlss.utils import *
from .distribution_utils import *

from typing import List


class Expectile:
    """
    Expectile distribution class.

    Distributional Parameters
    -------------------------
    expectile: List
        List of specified expectiles.

    """
    def __init__(self,
                 stabilization: str,
                 expectiles: List
                 ):

        # When a custom objective and metric are provided, XGBoost doesn't know its response and link function. Hence,
        # the user is responsible for specifying the transformations.

        # Specify Response and Link Functions
        param_dict = {}
        for expectile in expectiles:
            key = f"expectile_{expectile}"
            param_dict[key] = identity_fn
        param_dict_inv = param_dict
        distribution_arg_names = list(param_dict.keys())

        # Specify Distribution
        self.dist_class = DistributionClass(distribution=Expectile_Torch,
                                            univariate=True,
                                            discrete=False,
                                            n_dist_param=len(param_dict),
                                            stabilization=stabilization,
                                            param_dict=param_dict,
                                            param_dict_inv=param_dict_inv,
                                            distribution_arg_names=distribution_arg_names,
                                            tau=torch.tensor(expectiles)
                                            )


class Expectile_Torch(Distribution):
    """
    PyTorch implementation of expectiles.

    Arguments
    ---------
    expectiles : List[torch.Tensor]
        List of expectiles.
    """
    def __init__(self, expectiles: List[torch.Tensor]):
        super(Expectile_Torch).__init__()
        self.expectiles = expectiles
        self.__class__.__name__ = "Expectile"

    def log_prob(self, value: torch.Tensor, tau: List[torch.Tensor]) -> torch.Tensor:
        """
        Returns the log of the probability density/mass function evaluated at `value`.

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
        loss = torch.tensor(0.0, dtype=torch.float32)

        for expectile, tau in zip(self.expectiles, tau):
            loss += torch.nansum(
                tau * (value - expectile) ** 2 * (value - expectile >= 0) +
                (1 - tau) * (value - expectile) ** 2 * (value - expectile < 0)
            )
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
