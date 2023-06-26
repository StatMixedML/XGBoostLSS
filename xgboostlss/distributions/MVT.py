from pyro.distributions import MultivariateStudentT as MultivariateStudentT_Torch
from .multivariate_distribution_utils import Multivariate_DistributionClass
from ..utils import *

from typing import Dict, Optional, List, Callable
import numpy as np
import pandas as pd
from itertools import combinations


class MVT(Multivariate_DistributionClass):
    """
    Multivariate Student-T distribution class.

    The multivariate Student-T distribution is parameterized by a degree of freedom df vector, a mean vector and a
    lower-triangular matrix L with positive-valued diagonal entries, such that Σ=LL'. This triangular matrix can be
    obtained via, e.g., a Cholesky decomposition of the covariance.

    Distributional Parameters
    -------------------------
    df: torch.Tensor
        Degrees of freedom.
    loc: torch.Tensor
        Mean of the distribution (often referred to as mu).
    scale_tril: torch.Tensor
        Lower-triangular factor of covariance, with positive-valued diagonal.

    Source
    -------------------------
    https://docs.pyro.ai/en/stable/distributions.html#multivariatestudentt

    Parameters
    -------------------------
    D: int
        Number of targets.
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    response_fn: str
        Response function for transforming the distributional parameters to the correct support. Options are
        "exp" (exponential) or "softplus" (softplus).
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
            response_fn_df = exp_fn_df
        elif response_fn == "softplus":
            response_fn = softplus_fn
            response_fn_df = softplus_fn_df
        else:
            raise ValueError("Invalid response function. Please choose from 'exp' or 'softplus'.")

        # Set the parameters specific to the distribution
        distribution = MultivariateStudentT_Torch
        param_dict = MVT.create_param_dict(n_targets=D, response_fn=response_fn, response_fn_df=response_fn_df)
        distribution_arg_names = ["df", "loc", "scale_tril"]
        torch.distributions.Distribution.set_default_validate_args(False)

        # Specify Distribution Class
        super().__init__(distribution=distribution,
                         univariate=False,
                         distribution_arg_names=distribution_arg_names,
                         n_targets=D,
                         n_dist_param=len(param_dict),
                         param_dict=param_dict,
                         param_transform=MVT.param_transform,
                         get_dist_params=MVT.get_dist_params,
                         discrete=False,
                         stabilization=stabilization,
                         loss_fn=loss_fn
                         )

    @staticmethod
    def create_param_dict(n_targets: int,
                          response_fn: Callable,
                          response_fn_df: Callable
                          ) -> Dict:
        """ Function that transforms the distributional parameters to the desired scale.

        Arguments
        ---------
        n_targets: int
            Number of targets.
        response_fn: Callable
            Response function.
        response_fn_df: Callable
            Response function for the degrees of freedom.

        Returns
        -------
        param_dict: Dict
            Dictionary of distributional parameters.
        """

        # Df
        param_dict = {"df": response_fn_df}

        # Location
        loc_dict = {"location_" + str(i + 1): identity_fn for i in range(n_targets)}
        param_dict.update(loc_dict)

        # Tril
        tril_indices = torch.tril_indices(row=n_targets, col=n_targets, offset=0)
        tril_idx = (tril_indices.detach().numpy()) + 1
        n_tril = int((n_targets * (n_targets + 1)) / 2)
        tril_diag = tril_idx[0] == tril_idx[1]

        tril_dict = {}

        for i in range(n_tril):
            if tril_diag[i]:
                tril_dict.update({"scale_tril_diag_" + str(tril_idx[:, i][1]): response_fn})
            else:
                tril_dict.update({"scale_tril_offdiag_" + str(tril_idx[:, i][1]) + str(tril_idx[:, i][0]): identity_fn})

        param_dict.update(tril_dict)

        return param_dict

    @staticmethod
    def param_transform(params: List[torch.Tensor],
                        param_dict: Dict,
                        n_targets: int,
                        rank: Optional[int],
                        n_obs: int,
                        ) -> List[torch.Tensor]:
        """ Function that returns a list of parameters for a multivariate Student-T, parameterized
        by a location vector and the lower triangular matrix of the covariance matrix (Cholesky).

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
        params = [
            response_fun(params[i].reshape(-1, 1)) for i, (dist_param, response_fun) in enumerate(param_dict.items())
        ]

        # Df
        df = params[0].reshape(-1, )

        # Location
        loc = torch.cat(params[1:(n_targets + 1)], axis=1)

        # Scale Tril
        tril_predt = torch.cat(params[(n_targets + 1):], axis=1)
        tril_indices = torch.tril_indices(row=n_targets, col=n_targets, offset=0)
        scale_tril = torch.zeros(n_obs, n_targets, n_targets, dtype=tril_predt.dtype)
        scale_tril[:, tril_indices[0], tril_indices[1]] = tril_predt

        params = [df, loc, scale_tril]

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
        # Df
        Df_df = pd.DataFrame(dist_pred.df.detach().numpy())
        Df_df.columns = ["df"]

        # Location
        location_df = pd.DataFrame(dist_pred.loc.numpy())
        location_df.columns = [f"location_{i + 1}" for i in range(n_targets)]

        # Scale
        scale_df = pd.DataFrame(dist_pred.stddev.detach().numpy())
        scale_df.columns = [f"scale_{i + 1}" for i in range(n_targets)]

        # Rho
        n_obs = location_df.shape[0]
        n_rho = int((n_targets * (n_targets - 1)) / 2)
        # The covariance is df / (df - 2) * covariance_matrix
        df = torch.broadcast_to(dist_pred.df.reshape(-1, 1).unsqueeze(-1), dist_pred.covariance_matrix.shape)
        cov_mat = dist_pred.covariance_matrix * (df / (df - 2))
        rho_df = pd.DataFrame(
            np.concatenate([MVT.covariance_to_correlation(cov_mat[i]).reshape(-1, n_rho) for i in range(n_obs)], axis=0)
        )
        rho_idx = list(combinations(range(1, n_targets + 1), 2))
        rho_df.columns = [f"rho_{''.join(map(str, rho_idx[i]))}" for i in range(n_targets)]

        # Concatenate
        dist_params_df = pd.concat([Df_df, location_df, scale_df, rho_df], axis=1)

        return dist_params_df

    @staticmethod
    def covariance_to_correlation(cov_mat: torch.Tensor) -> np.ndarray:
        """ Function that calculates the correlation matrix from the covariance matrix.

        Arguments
        ---------
        cov_mat: torch.Tensor
            Covariance matrix.

        Returns
        -------
        cor_mat: np.ndarray
            Correlation matrix.
        """
        cov_mat = np.array(cov_mat)
        diag = np.sqrt(np.diag(np.diag(cov_mat)))
        diag_inv = np.linalg.inv(diag)
        cor_mat = diag_inv @ cov_mat @ diag_inv
        cor_mat = cor_mat[np.tril_indices_from(cor_mat, k=-1)]

        return cor_mat
