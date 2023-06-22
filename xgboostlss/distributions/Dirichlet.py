from torch.distributions import Dirichlet as Dirichlet_Torch
from xgboostlss.utils import *
from .multivariate_distribution_utils import Multivariate_DistributionClass

from typing import Dict, Optional, List, Callable
import pandas as pd


class Dirichlet(Multivariate_DistributionClass):
    """
    Dirichlet distribution class.

    The Dirichlet distribution is commonly used for modelling non-negative compositional data, i.e., data that consist
    of sub-sets that are fractions of some total. Compositional data are typically represented as proportions or
    percentages summing to 1, so that the Dirichlet extends the univariate beta-distribution to the multivariate case.

    Distributional Parameters
    -------------------------
    concentration: torch.Tensor
        Concentration parameter of the distribution (often referred to as alpha).

    Source
    -------------------------
    https://pytorch.org/docs/stable/distributions.html#dirichlet

    Parameters
    -------------------------
    D: int
        Number of targets.
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    response_fn: str
        When a custom objective and metric are provided, XGBoost doesn't know its response and link function. Hence,
        the user is responsible for specifying the transformations. Options are "exp" or "softplus".
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood).
    """
    def __init__(self,
                 D: int,
                 stabilization: str = "None",
                 response_fn: str = "exp",
                 loss_fn: str = "nll"
                 ):
        # Specify Response Functions
        if response_fn == "exp":
            response_fn = exp_fn
        elif response_fn == "softplus":
            response_fn = softplus_fn
        elif response_fn == "relu":
            response_fn = relu_fn
        else:
            raise ValueError("Invalid response function. Please choose from 'exp', 'relu' or 'softplus'.")

        # Set the parameters specific to the distribution
        distribution = Dirichlet_Torch
        param_dict = Dirichlet.create_param_dict(n_targets=D, response_fn=response_fn)
        distribution_arg_names = ["concentration"]

        # Specify Distribution Class
        super().__init__(distribution=distribution,
                         univariate=False,
                         distribution_arg_names=distribution_arg_names,
                         n_targets=D,
                         n_dist_param=len(param_dict),
                         param_dict=param_dict,
                         param_transform=Dirichlet.param_transform,
                         get_dist_params=Dirichlet.get_dist_params,
                         discrete=False,
                         stabilization=stabilization,
                         loss_fn=loss_fn
                         )

    @staticmethod
    def create_param_dict(n_targets: int,
                          response_fn: Callable
                          ) -> Dict:
        """ Function that transforms the distributional parameters to the desired scale.

        Arguments
        ---------
        n_targets: int
            Number of targets.
        response_fn: Callable
            Response function.

        Returns
        -------
        param_dict: Dict
            Dictionary of distributional parameters.
        """
        # Concentration
        param_dict = {"concentration_" + str(i + 1): response_fn for i in range(n_targets)}

        return param_dict

    @staticmethod
    def param_transform(params: List[torch.Tensor],
                        param_dict: Dict,
                        n_targets: int,
                        rank: Optional[int],
                        n_obs: int,
                        ) -> List[torch.Tensor]:
        """ Function that returns a list of parameters for a Dirichlet distribution.

        Arguments
        ---------
        params: List[torch.Tensor]
            List of distributional parameters.
        param_dict: Dict
        n_targets: int
            Number of targets.
        rank: Optional[int]
            Rank of the low-rank form of the covariance matrix.
        n_obs: int
            Number of observations.

        Returns
        -------
        params: List[torch.Tensor]
            List of parameters.
        """
        # Transform Parameters to respective scale
        params = torch.cat([
            response_fun(params[i].reshape(-1, 1)) for i, (dist_param, response_fun) in enumerate(param_dict.items())
        ], dim=1)

        return params

    @staticmethod
    def get_dist_params(n_targets: int,
                        dist_pred: torch.distributions.Distribution,
                        ) -> pd.DataFrame:
        """
        Function that returns the predicted distributional parameters.

        Arguments
        ---------
        n_targets: int
            Number of targets.
        dist_pred: torch.distributions.Distribution
            Predicted distribution.

        Returns
        -------
        dist_params_df: pd.DataFrame
            DataFrame with predicted distributional parameters.
        """
        # Concentration
        dist_params_df = pd.DataFrame(dist_pred.concentration.numpy())
        dist_params_df.columns = [f"concentration_{i + 1}" for i in range(n_targets)]

        # # Normalize to sum to 1
        # dist_params_df = dist_params_df.div(dist_params_df.sum(axis=1), axis=0)

        return dist_params_df
