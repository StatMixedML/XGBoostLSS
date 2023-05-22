import torch
from torch.autograd import grad as autograd
from torch.optim import LBFGS

import numpy as np
import pandas as pd
import xgboost as xgb

from typing import Any, Dict, Optional, List, Tuple


class DistributionClass:
    """
    Generic class that contains general functions for all distributions.

    Arguments
    ---------
    distribution: torch.distributions.Distribution
        PyTorch Distribution class.
    univariate: bool
        Whether the distribution is univariate or multivariate.
    discrete: bool
        Whether the support of the distribution is discrete or continuous.
    n_dist_param: int
        Number of distributional parameters.
    stabilization: str
        Stabilization method.
    param_dict: Dict[str, Any]
        Dictionary that maps distributional parameters to their response scale.
    param_dict_inv: Dict[str, Any]
        Dictionary that maps distributional parameters to their inverse response scale.
    distribution_arg_names: List
        List of distributional parameter names.
    tau: List
        List of expectiles. Only used for Expectile distributon.

    """
    def __init__(self,
                 distribution: torch.distributions.Distribution = None,
                 univariate: bool = True,
                 discrete: bool = False,
                 n_dist_param: int = None,
                 stabilization: str = "None",
                 param_dict: Dict[str, Any] = None,
                 param_dict_inv: Dict[str, Any] = None,
                 distribution_arg_names: List = None,
                 tau: Optional[List[torch.Tensor]] = None,
                 ):

        self.distribution = distribution
        self.univariate = univariate
        self.discrete = discrete
        self.n_dist_param = n_dist_param
        self.stabilization = stabilization
        self.param_dict = param_dict
        self.param_dict_inv = param_dict_inv
        self.distribution_arg_names = distribution_arg_names
        self.tau = tau

    def objective_fn(self, predt: np.ndarray, data: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:

        """
        Function to estimate gradients and hessians of distributional parameters.

        Arguments
        ---------
        predt: np.ndarray
            Predicted values.
        data: xgb.DMatrix
            Data used for training.

        Returns
        -------
        grad: np.ndarray
            Gradient.
        hess: np.ndarray
            Hessian.
        """

        # Weights
        target = data.get_label().reshape(-1, 1)
        if data.get_weight().size == 0:
            # Use 1 as weight if no weights are specified
            weights = np.ones_like(target, dtype=target.dtype)
        else:
            weights = data.get_weight()

        predt, nll = self.get_params_nll(predt, data, requires_grad=True)
        grad, hess = compute_gradients_and_hessians(nll, predt, weights, self.stabilization)

        return grad, hess

    def metric_fn(self, predt: np.ndarray, data: xgb.DMatrix) -> Tuple[str, np.ndarray]:
        """
        Function that evaluates the predictions using the negative log-likelihood.

        Arguments
        ---------
        predt: np.ndarray
            Predicted values.
        data: xgb.DMatrix
            Data used for training.

        Returns
        -------
        name: str
            Name of the evaluation metric.
        nll: float
            Negative log-likelihood.
        """
        _, nll = self.get_params_nll(predt, data, requires_grad=False)

        return "NegLogLikelihood", nll

    def calculate_start_values(self, target: np.ndarray) -> np.ndarray:
        """
        Function that calculates the starting values for each distributional parameter.

        Arguments
        ---------
        target: np.ndarray
            Data from which starting values are calculated.

        Returns
        -------
        start_values: np.ndarray
            Starting values for each distributional parameter.
        """
        def neg_log_likelihood(params, target):
            if self.tau is None:
                dist = self.distribution(*params)
                nll = -torch.nansum(dist.log_prob(target))
            else:
                dist = self.distribution(params)
                nll = -torch.nansum(dist.log_prob(target, self.tau))
            return nll

        # Convert target to torch.tensor
        target = torch.tensor(target, dtype=torch.float32)

        # Initialize parameters
        params = [torch.tensor(0.5, requires_grad=True) for _ in range(self.n_dist_param)]

        # Minimize negative log-likelihood to estimate unconditional parameters
        optimizer = LBFGS(params, lr=0.1, max_iter=100)

        def closure():
            optimizer.zero_grad()
            loss = neg_log_likelihood(params, target)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Transform parameters to inverse response scale
        start_values = np.array(
            [inv_response_fun(params[i]).detach()
             for i, (dist_param, inv_response_fun) in enumerate(self.param_dict_inv.items())]
        )

        return start_values

    def get_params_nll(self,
                       predt: np.ndarray,
                       data: xgb.DMatrix,
                       requires_grad: bool
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function that returns the predicted parameters and the negative log-likelihood.

        Arguments
        ---------
        predt: np.ndarray
            Predicted values.
        data: xgb.DMatrix
            Data used for training.
        requires_grad: bool
            Whether to add to the computational graph or not.

        Returns
        -------
        predt: torch.Tensor
            Predicted parameters.
        nll: torch.tensor
            Negative log-likelihood.
        """

        # Target
        target = torch.tensor(data.get_label().reshape(-1, 1))

        # Predicted Parameters
        predt = predt.reshape(-1, self.n_dist_param)
        predt = [
            torch.tensor(predt[:, i].reshape(-1, 1), requires_grad=requires_grad) for i in range(self.n_dist_param)
        ]

        # Predicted Parameters transformed to response scale
        predt_transformed = [
            response_fn(predt[i].reshape(-1, 1)) for i, response_fn in enumerate(self.param_dict.values())
        ]

        if self.tau is None:
            # Specify Distribution and NLL
            dist_kwargs = dict(zip(self.distribution_arg_names, predt_transformed))
            dist_fit = self.distribution(**dist_kwargs)
            nll = -torch.nansum(dist_fit.log_prob(target))
        else:
            # Specify Distribution and NLL
            dist_kwargs = predt_transformed
            dist_fit = self.distribution(dist_kwargs)
            nll = -torch.nansum(dist_fit.log_prob(target, self.tau))

        return predt, nll

    def draw_samples(self,
                     predt_params: pd.DataFrame,
                     n_samples: int = 1000,
                     seed: int = 123
                     ) -> pd.DataFrame:
        """
        Function that draws n_samples from a predicted distribution.

        Arguments
        ---------
        predt_params: pd.DataFrame
            pd.DataFrame with predicted distributional parameters.
        n_samples: int
            Number of sample to draw from predicted response distribution.
        seed: int
            Manual seed.

        Returns
        -------
        pred_dist: pd.DataFrame
            DataFrame with n_samples drawn from predicted response distribution.

        """
        torch.manual_seed(seed)

        if self.tau is None:
            pred_params = torch.tensor(predt_params.values)
            dist_kwargs = {arg_name: param for arg_name, param in zip(self.distribution_arg_names, pred_params.T)}
            dist_pred = self.distribution(**dist_kwargs)
            dist_samples = dist_pred.sample((n_samples,)).squeeze().T.detach().numpy()
            dist_samples = pd.DataFrame(dist_samples)
            dist_samples.columns = [str("y_sample") + str(i) for i in range(dist_samples.shape[1])]
        else:
            dist_samples = None

        if self.discrete:
            dist_samples = dist_samples.astype(int)

        return dist_samples


def compute_gradients_and_hessians(nll: torch.tensor,
                                   predt: torch.tensor,
                                   weights: np.ndarray,
                                   stabilization: str) -> Tuple[np.ndarray, np.ndarray]:

    """
    Calculates gradients and hessians.

    Output gradients and hessians have shape (n_samples, n_outputs).

    Arguments:
    ---------
    nll: torch.Tensor
        Calculated NLL.
    predt: torch.Tensor
        List of predicted parameters.
    weights: np.ndarray
        Weights.
    stabilization: str
        Specifies the type of stabilization for gradients and hessians.

    Returns:
    -------
    grad: torch.Tensor
        Gradients.
    hess: torch.Tensor
        Hessians.
    """

    # Gradient and Hessian
    grad_list = autograd(nll, inputs=predt, create_graph=True)
    hess_list = [autograd(grad_list[i].nansum(), inputs=predt[i], retain_graph=True)[0] for i in range(len(grad_list))]

    # Stabilization of Derivatives
    grad = [stabilize_derivative(grad_list[i], type=stabilization) for i in range(len(grad_list))]
    hess = [stabilize_derivative(hess_list[i], type=stabilization) for i in range(len(grad_list))]

    # Reshape
    grad = torch.cat(grad, axis=1).detach().numpy()
    hess = torch.cat(hess, axis=1).detach().numpy()

    # Weighting
    grad *= weights
    hess *= weights

    # Flatten
    grad = grad.flatten()
    hess = hess.flatten()

    return grad, hess


def stabilize_derivative(input_der: torch.Tensor, type: str = "MAD") -> torch.Tensor:
    """
    Function that stabilizes Gradients and Hessians.

    As XGBoostLSS updates the parameter estimates by optimizing Gradients and Hessians, it is important
    that these are comparable in magnitude for all distributional parameters. Due to imbalances regarding the ranges,
    the estimation might become unstable so that it does not converge (or converge very slowly) to the optimal solution.
    Another way to improve convergence might be to standardize the response variable. This is especially useful if the
    range of the response differs strongly from the range of the Gradients and Hessians. Both, the stabilization and
    the standardization of the response are not always advised but need to be carefully considered.
    Source: https://github.com/boost-R/gamboostLSS/blob/7792951d2984f289ed7e530befa42a2a4cb04d1d/R/helpers.R#L173

    Parameters
    ----------
    input_der : torch.Tensor
        Input derivative, either Gradient or Hessian.
    type: str
        Stabilization method. Can be either "None", "MAD" or "L2".

    Returns
    -------
    stab_der : torch.Tensor
        Stabilized Gradient or Hessian.
    """

    if type == "MAD":
        input_der = torch.nan_to_num(input_der, nan=float(torch.nanmean(input_der)))
        div = torch.nanmedian(torch.abs(input_der - torch.nanmedian(input_der)))
        div = torch.where(div < torch.tensor(1e-04), torch.tensor(1e-04), div)
        stab_der = input_der / div

    if type == "L2":
        input_der = torch.nan_to_num(input_der, nan=float(torch.nanmean(input_der)))
        div = torch.sqrt(torch.nanmean(input_der.pow(2)))
        div = torch.where(div < torch.tensor(1e-04), torch.tensor(1e-04), div)
        div = torch.where(div > torch.tensor(10000.0), torch.tensor(10000.0), div)
        stab_der = input_der / div

    if type == "None":
        stab_der = torch.nan_to_num(input_der, nan=float(torch.nanmean(input_der)))

    return stab_der
