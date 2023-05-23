import torch
from torch.autograd import grad as autograd
from torch.optim import LBFGS

import numpy as np
import pandas as pd
import xgboost as xgb

from typing import Any, Dict, Optional, List, Tuple

import plotnine
from plotnine import *


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
        nll: float
            Negative log-likelihood.
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

        nll = optimizer.step(closure).detach().numpy()

        # Transform parameters to inverse response scale
        start_values = np.array(
            [inv_response_fun(params[i]).detach()
             for i, (dist_param, inv_response_fun) in enumerate(self.param_dict_inv.items())]
        )

        return nll, start_values

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
            dist_samples = dist_pred.sample((n_samples,)).squeeze().detach().numpy().T
            dist_samples = pd.DataFrame(dist_samples)
            dist_samples.columns = [str("y_sample") + str(i) for i in range(dist_samples.shape[1])]
        else:
            dist_samples = None

        if self.discrete:
            dist_samples = dist_samples.astype(int)

        return dist_samples

    def predict_dist(self,
                     booster: xgb.Booster,
                     dtest: xgb.DMatrix,
                     pred_type: str = "parameters",
                     n_samples: int = 1000,
                     quantiles: list = [0.1, 0.5, 0.9],
                     seed: str = 123
                     ) -> pd.DataFrame:
        """
        Function that predicts from the trained model.

        Arguments
        ---------
        booster : xgb.Booster
            Trained model.
        dtest : xgb.DMatrix
            Test data.
        pred_type : str
            Type of prediction:
            - "samples" draws n_samples from the predicted distribution.
            - "quantile" calculates the quantiles from the predicted distribution.
            - "parameters" returns the predicted distributional parameters.
            - "expectiles" returns the predicted expectiles.
        n_samples : int
            Number of samples to draw from the predicted distribution.
        quantiles : List[float]
            List of quantiles to calculate from the predicted distribution.
        seed : int
            Seed for random number generator used to draw samples from the predicted distribution.

        Returns
        -------
        pred : pd.DataFrame
            Predictions.
        """
        predt = np.array(booster.predict(dtest, output_margin=True)).reshape(-1, self.n_dist_param)
        predt = torch.tensor(predt, dtype=torch.float32)

        # Transform predicted parameters to response scale
        dist_params_predt = np.concatenate(
            [
                response_fun(
                    predt[:, i].reshape(-1, 1)).numpy() for i, (dist_param, response_fun) in
                enumerate(self.param_dict.items())
            ],
            axis=1,
        )
        dist_params_predt = pd.DataFrame(dist_params_predt)
        dist_params_predt.columns = self.param_dict.keys()

        # Draw samples from predicted response distribution
        pred_samples_df = self.draw_samples(predt_params=dist_params_predt,
                                            n_samples=n_samples,
                                            seed=seed)

        # Calculate quantiles from predicted response distribution
        pred_quant_df = pred_samples_df.quantile(quantiles, axis=1).T
        pred_quant_df.columns = [str("quant_") + str(quantiles[i]) for i in range(len(quantiles))]
        if self.discrete:
            pred_quant_df = pred_quant_df.astype(int)

        if pred_type == "parameters":
            return dist_params_predt

        elif pred_type == "expectiles":
            return dist_params_predt

        elif pred_type == "samples":
            return pred_samples_df

        elif pred_type == "quantiles":
            return pred_quant_df


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


def dist_select(target: np.ndarray,
                candidate_distributions: list,
                plot: bool = False,
                figure_size: tuple = (10, 5),
                ) -> pd.DataFrame:
    """
    Function that selects the most suitable distribution among the candidate_distributions for the target variable,
    based on the NegLogLikelihood (lower is better).

    Parameters
    ----------
    target: np.ndarray
        Response variable.
    candidate_distributions: List
        List of candidate distributions.
    plot: bool
        If True, a density plot of the actual and fitted distribution is created.
    figure_size: tuple
        Figure size of the density plot.

    Returns
    -------
    dist_nll: pd.DataFrame
        Dataframe with the negative log-likelihoods of the fitted candidate distributions.
    """
    dist_list = []
    for i in range(len(candidate_distributions)):
        dist_name = candidate_distributions[i].__name__.split(".")[2]
        dist_sel = getattr(candidate_distributions[i], dist_name)().dist_class
        nll, params = dist_sel.calculate_start_values(target)
        dist_nll = pd.DataFrame.from_dict({"NegLogLikelihood": nll.reshape(-1,),
                                           "distribution": str(dist_name)
                                           })
        dist_nll["params"] = [params] * len(dist_nll)
        dist_list.append(dist_nll)
        dist_nll = pd.concat(dist_list).sort_values(by="NegLogLikelihood", ascending=True)
        dist_nll["rank"] = dist_nll["NegLogLikelihood"].rank().astype(int)
        dist_nll.set_index(dist_nll["rank"], inplace=True)

    if plot:
        # Select best distribution
        best_dist = dist_nll[dist_nll["rank"] == 1].reset_index(drop=True)
        for dist in candidate_distributions:
            if dist.__name__.split(".")[2] == best_dist["distribution"].values[0]:
                best_dist_sel = dist
                break
        best_dist_sel = getattr(best_dist_sel, best_dist["distribution"].values[0])().dist_class
        params = torch.tensor(best_dist["params"][0]).reshape(-1, best_dist_sel.n_dist_param)

        # Transform parameters to the response scale and draw samples
        fitted_params = np.concatenate(
            [
                response_fun(params[:, i].reshape(-1, 1)).numpy()
                for i, (dist_param, response_fun) in enumerate(best_dist_sel.param_dict.items())
            ],
            axis=1,
        )
        fitted_params = pd.DataFrame(fitted_params, columns=best_dist_sel.param_dict.keys())
        fitted_params.columns = best_dist_sel.param_dict.keys()
        dist_samples = best_dist_sel.draw_samples(fitted_params,
                                                  n_samples=1000,
                                                  seed=123).values

        # Plot actual and fitted distribution
        plot_df_actual = pd.DataFrame({"y": target, "type": "Actual"})
        plot_df_fitted = pd.DataFrame({"y": dist_samples.reshape(-1,),
                                       "type": f"Best-Fit: {best_dist['distribution'].values[0]}"})
        plot_df = pd.concat([plot_df_actual, plot_df_fitted])

        print(
            ggplot(plot_df,
                   aes(x="y",
                       color="type")) +
            geom_density(alpha=0.5) +
            theme_bw(base_size=15) +
            theme(figure_size=figure_size,
                  legend_position="right",
                  legend_title=element_blank(),
                  plot_title=element_text(hjust=0.5)) +
            labs(title=f"Actual vs. Fitted Density")
        )

    dist_nll.drop(columns=["rank", "params"], inplace=True)

    return dist_nll
