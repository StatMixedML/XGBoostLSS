import torch
from torch.distributions import Categorical, MixtureSameFamily
from torch.autograd import grad as autograd
from torch.optim import LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau

import xgboost as xgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Any, Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from xgboostlss import distributions


# Get all distribution names
def get_component_distributions():
    """
    Function that returns component distributions for creating a mixing distribution.

    Arguments
    ---------
    None

    Returns
    -------
    distns: List
        List of all available distributions.
    """
    # Get all distribution names
    mixture_distns = [dist for dist in dir(distributions) if dist[0].isupper()]

    # Remove specific distributions
    distns_remove = [
        "Dirichlet",
        "Expectile",
        "MVN",
        "MVN_LoRa",
        "MVT",
        "Mixture",
        "SplineFlow"
    ]

    mixture_distns = [item for item in mixture_distns if item not in distns_remove]

    return mixture_distns


class MixtureDistributionClass:
    """
    Generic class that contains general functions for mixed-density distributions.

    Arguments
    ---------
    distribution: torch.distributions.Distribution
        PyTorch Distribution class.
    M: int
        Number of components in the mixture distribution.
    temperature: float
        Temperature for the Gumbel-Softmax distribution.
    hessian_mode: str
        Mode for computing the Hessian. Must be one of the following:

            - "individual": Each parameter is treated as a separate tensor. As a result, when the Hessian is calculated
            for each gradient element, this corresponds to the second derivative with respect to that specific tensor
            element only. This means the resulting Hessians capture the curvature of the loss w.r.t. each individual
            parameter. This is usually more runtime intensive, but can also be more accurate.

            - "grouped": Each parameter is a tensor containing all values for a specific parameter type,
            e.g., loc, scale, or mixture probabilities for a Gaussian Mixture. When computing the Hessian for each
            gradient element, the Hessian matrix for all the values in the respective tensor are calculated together.
            The resulting Hessians capture the curvature of the loss w.r.t. the entire parameter type tensor. This is
            usually less runtime intensive, but can be less accurate.
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
    distribution_arg_names: List
        List of distributional parameter names.
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood).
    """
    def __init__(self,
                 distribution: torch.distributions.Distribution = None,
                 M: int = 2,
                 temperature: float = 1.0,
                 hessian_mode: str = "individual",
                 univariate: bool = True,
                 discrete: bool = False,
                 n_dist_param: int = None,
                 stabilization: str = "None",
                 param_dict: Dict[str, Any] = None,
                 distribution_arg_names: List = None,
                 loss_fn: str = "nll",
                 ):

        self.distribution = distribution
        self.M = M
        self.temperature = temperature
        self.hessian_mode = hessian_mode
        self.univariate = univariate
        self.discrete = discrete
        self.n_dist_param = n_dist_param
        self.stabilization = stabilization
        self.param_dict = param_dict
        self.distribution_arg_names = distribution_arg_names
        self.loss_fn = loss_fn

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
        # Target
        target = torch.tensor(data.get_label().reshape(-1, 1), dtype=torch.float32)

        # Weights
        if data.get_weight().size == 0:
            # Use 1 as weight if no weights are specified
            weights = np.ones_like(target, dtype="float32")
        else:
            weights = data.get_weight().reshape(-1, 1)

        # Start values (needed to replace NaNs in predt)
        start_values = data.get_base_margin().reshape(-1, self.n_dist_param)[0, :].tolist()

        # Calculate gradients and hessians
        predt, loss = self.get_params_loss(predt, target.flatten(), start_values, requires_grad=True)
        grad, hess = self.compute_gradients_and_hessians(loss, predt, weights)

        return grad, hess

    def metric_fn(self, predt: np.ndarray, data: xgb.DMatrix) -> Tuple[str, np.ndarray]:
        """
        Function that evaluates the predictions using the specified loss function.

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
        loss: float
            Loss value.
        """
        # Target
        target = torch.tensor(data.get_label().reshape(-1, 1), dtype=torch.float32)

        # Start values (needed to replace NaNs in predt)
        start_values = data.get_base_margin().reshape(-1, self.n_dist_param)[0, :].tolist()

        # Calculate loss
        _, loss = self.get_params_loss(predt, target.flatten(), start_values, requires_grad=False)

        return self.loss_fn, loss

    def create_mixture_distribution(self,
                                    params: List[torch.Tensor],
                                    ) -> torch.distributions.Distribution:
        """
        Function that creates a mixture distribution.

        Arguments
        ---------
        params: torch.Tensor
            Distributional parameters.

        Returns
        -------
        dist: torch.distributions.Distribution
            Mixture distribution.
        """

        # Create Mixture Distribution
        mixture_cat = Categorical(probs=params[-1])
        mixture_comp = self.distribution.distribution(*params[:-1])
        mixture_dist = MixtureSameFamily(mixture_cat, mixture_comp)

        return mixture_dist

    def loss_fn_start_values(self,
                             params: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
        """
        Function that calculates the loss for a given set of distributional parameters. Only used for calculating
        the loss for the start values.

        Parameter
        ---------
        params: torch.Tensor
            Distributional parameters.
        target: torch.Tensor
            Target values.

        Returns
        -------
        loss: torch.Tensor
            Loss value.
        """
        # Replace NaNs and infinity values with 0.5
        nan_inf_idx = torch.isnan(torch.stack(params)) | torch.isinf(torch.stack(params))
        params = torch.where(nan_inf_idx, torch.tensor(0.5), torch.stack(params)).reshape(1, -1)
        params = torch.split(params, self.M, dim=1)

        # Transform parameters to response scale
        params = [response_fn(params[i]) for i, response_fn in enumerate(self.param_dict.values())]

        # Specify Distribution and Loss
        dist = self.create_mixture_distribution(params)
        loss = -torch.nansum(dist.log_prob(target))

        return loss

    def calculate_start_values(self,
                               target: np.ndarray,
                               max_iter: int = 50
                               ) -> Tuple[float, np.ndarray]:
        """
        Function that calculates the starting values for each distributional parameter.

        Arguments
        ---------
        target: np.ndarray
            Data from which starting values are calculated.
        max_iter: int
            Maximum number of iterations.

        Returns
        -------
        loss: float
            Loss value.
        start_values: np.ndarray
            Starting values for each distributional parameter.
        """
        # Convert target to torch.tensor
        target = torch.tensor(target, dtype=torch.float32).flatten()

        # Initialize parameters
        params = [torch.tensor(0.5, requires_grad=True) for _ in range(self.n_dist_param)]

        # Specify optimizer
        optimizer = LBFGS(params, lr=0.1, max_iter=np.min([int(max_iter/4), 20]), line_search_fn="strong_wolfe")

        # Define learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

        # Define closure
        def closure():
            optimizer.zero_grad()
            loss = self.loss_fn_start_values(params, target)
            loss.backward()
            return loss

        # Optimize parameters
        loss_vals = []
        tolerance = 1e-5
        patience = 5
        best_loss = float("inf")
        epochs_without_change = 0

        for epoch in range(max_iter):
            optimizer.zero_grad()
            loss = optimizer.step(closure)
            lr_scheduler.step(loss)
            loss_vals.append(loss.item())

            # Stopping criterion (no improvement in loss)
            if loss.item() < best_loss - tolerance:
                best_loss = loss.item()
                epochs_without_change = 0
            else:
                epochs_without_change += 1

            if epochs_without_change >= patience:
                break

        # Get final loss
        loss = np.array(loss_vals[-1])

        # Get start values
        start_values = np.array([params[i].detach() for i in range(self.n_dist_param)])

        # Replace any remaining NaNs or infinity values with 0.5
        start_values = np.nan_to_num(start_values, nan=0.5, posinf=0.5, neginf=0.5)

        return loss, start_values

    def get_params_loss(self,
                        predt: np.ndarray,
                        target: torch.Tensor,
                        start_values: List[float],
                        requires_grad: bool = False,
                        ) -> Tuple[List[torch.Tensor], np.ndarray]:
        """
        Function that returns the predicted parameters and the loss.

        Arguments
        ---------
        predt: np.ndarray
            Predicted values.
        target: torch.Tensor
            Target values.
        start_values: List
            Starting values for each distributional parameter.
        requires_grad: bool
            Whether to add to the computational graph or not.

        Returns
        -------
        predt: List of torch.Tensors
            Predicted parameters.
        loss: torch.Tensor
            Loss value.
        """
        # Predicted Parameters
        predt = predt.reshape(-1, self.n_dist_param)

        # Replace NaNs and infinity values with unconditional start values
        nan_inf_mask = np.isnan(predt) | np.isinf(predt)
        predt[nan_inf_mask] = np.take(start_values, np.where(nan_inf_mask)[1])

        if self.hessian_mode == "grouped":
            # Convert to torch.Tensor: splits the parameters into tensors for each parameter-type
            predt = torch.split(torch.tensor(predt, requires_grad=requires_grad), self.M, dim=1)
            # Transform parameters to response scale
            predt_transformed = [response_fn(predt[i]) for i, response_fn in enumerate(self.param_dict.values())]

        else:
            # Convert to torch.Tensor: splits the parameters into tensors for each parameter individually
            predt = torch.split(torch.tensor(predt, requires_grad=requires_grad), 1, dim=1)
            # Transform parameters to response scale
            keys = list(self.param_dict.keys())
            max_index = len(self.param_dict) * self.M
            index_ranges = []
            for i in range(0, max_index, self.M):
                if i + self.M >= max_index:
                    index_ranges.append((i, None))
                    break
                index_ranges.append((i, i + self.M))

            predt_transformed = []
            for key, (start, end) in zip(keys, index_ranges):
                predt_transformed.append(self.param_dict[key](torch.cat(predt[start:end], dim=1)))

        # Specify Distribution and Loss
        dist_fit = self.create_mixture_distribution(predt_transformed)
        loss = -torch.nansum(dist_fit.log_prob(target))

        return predt, loss

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

        pred_params = torch.tensor(predt_params.values).reshape(-1, self.n_dist_param)
        pred_params = torch.split(pred_params, self.M, dim=1)
        dist_pred = self.create_mixture_distribution(pred_params)
        dist_samples = dist_pred.sample((n_samples,)).squeeze().detach().numpy().T
        dist_samples = pd.DataFrame(dist_samples)
        dist_samples.columns = [str("y_sample") + str(i) for i in range(dist_samples.shape[1])]

        if self.discrete:
            dist_samples = dist_samples.astype(int)

        return dist_samples

    def predict_dist(self,
                     booster: xgb.Booster,
                     start_values: np.ndarray,
                     data: xgb.DMatrix,
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
        start_values : np.ndarray
            Starting values for each distributional parameter.
        data : xgb.DMatrix
            Data to predict from.
        pred_type : str
            Type of prediction:
            - "samples" draws n_samples from the predicted distribution.
            - "quantiles" calculates the quantiles from the predicted distribution.
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
        # Set base_margin as starting point for each distributional parameter. Requires base_score=0 in parameters.
        base_margin_predt = (np.ones(shape=(data.num_row(), 1))) * start_values
        data.set_base_margin(base_margin_predt.flatten())

        predt = np.array(booster.predict(data, output_margin=True)).reshape(-1, self.n_dist_param)
        predt = torch.split(torch.tensor(predt, dtype=torch.float32), self.M, dim=1)

        # Transform predicted parameters to response scale
        dist_params_predt = np.concatenate(
            [
                response_fun(predt[i]).numpy() for i, (dist_param, response_fun) in enumerate(self.param_dict.items())
            ],
            axis=1,
        )
        dist_params_predt = pd.DataFrame(dist_params_predt)
        dist_params_predt.columns = self.distribution_arg_names

        # Draw samples from predicted response distribution
        pred_samples_df = self.draw_samples(predt_params=dist_params_predt,
                                            n_samples=n_samples,
                                            seed=seed)

        if pred_type == "parameters":
            return dist_params_predt

        elif pred_type == "samples":
            return pred_samples_df

        elif pred_type == "quantiles":
            # Calculate quantiles from predicted response distribution
            pred_quant_df = pred_samples_df.quantile(quantiles, axis=1).T
            pred_quant_df.columns = [str("quant_") + str(quantiles[i]) for i in range(len(quantiles))]
            if self.discrete:
                pred_quant_df = pred_quant_df.astype(int)
            return pred_quant_df

    def compute_gradients_and_hessians(self,
                                       loss: torch.Tensor,
                                       predt: List[torch.Tensor],
                                       weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        Calculates gradients and hessians.

        Output gradients and hessians have shape (n_samples*n_outputs, 1).

        Arguments:
        ---------
        loss: torch.Tensor
            Loss.
        predt: torch.Tensor
            List of predicted parameters.
        weights: np.ndarray
            Weights.

        Returns:
        -------
        grad: torch.Tensor
            Gradients.
        hess: torch.Tensor
            Hessians.
        """
        # Gradient and Hessian
        grad = autograd(loss, inputs=predt, create_graph=True)
        hess = [autograd(grad[i].nansum(), inputs=predt[i], retain_graph=True)[0] for i in range(len(grad))]

        # Stabilization of Derivatives
        if self.stabilization != "None":
            grad = [self.stabilize_derivative(grad[i], type=self.stabilization) for i in range(len(grad))]
            hess = [self.stabilize_derivative(hess[i], type=self.stabilization) for i in range(len(hess))]

        # Reshape
        grad = torch.cat(grad, axis=1).detach().squeeze(-1).numpy()
        hess = torch.cat(hess, axis=1).detach().squeeze(-1).numpy()

        # Weighting
        grad *= weights
        hess *= weights

        # Flatten
        grad = grad.flatten()
        hess = hess.flatten()

        return grad, hess

    def stabilize_derivative(self, input_der: torch.Tensor, type: str = "MAD") -> torch.Tensor:
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

    def dist_select(self,
                    target: np.ndarray,
                    candidate_distributions: List,
                    max_iter: int = 100,
                    plot: bool = False,
                    figure_size: tuple = (8, 5),
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
        max_iter: int
            Maximum number of iterations for the optimization.
        plot: bool
            If True, a density plot of the actual and fitted distribution is created.
        figure_size: tuple
            Figure size of the density plot.

        Returns
        -------
        fit_df: pd.DataFrame
            Dataframe with the loss values of the fitted candidate distributions.
        """
        dist_list = []
        total_iterations = len(candidate_distributions)
        with tqdm(total=total_iterations, desc="Fitting candidate distributions") as pbar:
            for i in range(len(candidate_distributions)):
                dist_name = candidate_distributions[i].distribution.__class__.__name__
                n_mix = candidate_distributions[i].M
                tau = candidate_distributions[i].temperature
                dist_name = f"Mixture({dist_name}, tau={tau}, M={n_mix})"
                pbar.set_description(f"Fitting {dist_name} distribution")
                try:
                    loss, params = candidate_distributions[i].calculate_start_values(target=target, max_iter=max_iter)
                    fit_df = pd.DataFrame.from_dict(
                        {candidate_distributions[i].loss_fn: loss.reshape(-1, ),
                         "distribution": str(dist_name),
                         "params": [params],
                         "dist_pos": i,
                         "M": candidate_distributions[i].M
                         }
                    )
                except Exception as e:
                    warnings.warn(f"Error fitting {dist_name} distribution: {str(e)}")
                    fit_df = pd.DataFrame(
                        {candidate_distributions[i].loss_fn: np.nan,
                         "distribution": str(dist_name),
                         "params": [np.nan] * self.n_dist_param,
                         "dist_pos": i,
                         "M": candidate_distributions[i].M
                         }
                    )
                dist_list.append(fit_df)
                pbar.update(1)
            pbar.set_description(f"Fitting of candidate distributions completed")
            fit_df = pd.concat(dist_list).sort_values(by=candidate_distributions[i].loss_fn, ascending=True)
            fit_df["rank"] = fit_df[candidate_distributions[i].loss_fn].rank().astype(int)
            fit_df.set_index(fit_df["rank"], inplace=True)

        if plot:
            # Select best distribution
            best_dist = fit_df[fit_df["rank"] == fit_df["rank"].min()].reset_index(drop=True).iloc[[0]]
            best_dist_pos = int(best_dist["dist_pos"].values[0])
            best_dist_sel = candidate_distributions[best_dist_pos]
            params = torch.tensor(best_dist["params"][0]).reshape(1, -1)
            params = torch.split(params, best_dist_sel.M, dim=1)

            fitted_params = np.concatenate(
                [
                    response_fun(params[i]).numpy()
                    for i, (dist_param, response_fun) in enumerate(best_dist_sel.param_dict.items())
                ],
                axis=1,
            )

            fitted_params = pd.DataFrame(fitted_params, columns=best_dist_sel.distribution_arg_names)
            n_samples = np.max([10000, target.shape[0]])
            n_samples = np.where(n_samples > 500000, 100000, n_samples)
            dist_samples = best_dist_sel.draw_samples(fitted_params,
                                                      n_samples=n_samples,
                                                      seed=123).values

            # Plot actual and fitted distribution
            plt.figure(figsize=figure_size)
            sns.kdeplot(target.reshape(-1,), label="Actual")
            sns.kdeplot(dist_samples.reshape(-1,), label=f"Best-Fit: {best_dist['distribution'].values[0]}")
            plt.legend()
            plt.title("Actual vs. Best-Fit Density")
            plt.show()

        fit_df.drop(columns=["rank", "params", "dist_pos", "M"], inplace=True)

        return fit_df
