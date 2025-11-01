import torch
from torch.autograd import grad as autograd
from torch.optim import LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import Transform

import xgboost as xgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Any, Dict, Optional, List, Tuple
import warnings


class NormalizingFlowClass:
    """
    Generic class that contains general functions for normalizing flows.

    Arguments
    ---------
    base_dist: torch.distributions.Distribution
        PyTorch Distribution class. Currently only Normal is supported.
    flow_transform: Transform
        Specify the normalizing flow transform.
    count_bins: Optional[int]
        The number of segments comprising the spline. Only used if flow_transform is Spline.
    bound: Optional[float]
        The quantity "K" determining the bounding box, [-K,K] x [-K,K] of the spline. By adjusting the
        "K" value, you can control the size of the bounding box and consequently control the range of inputs that
        the spline transform operates on. Larger values of "K" will result in a wider valid range for the spline
        transformation, while smaller values will restrict the valid range to a smaller region. Should be chosen
        based on the range of the data. Only used if flow_transform is Spline.
    order: Optional[str]
        The order of the spline. Options are "linear" or "quadratic". Only used if flow_transform is Spline.
    n_dist_param: int
        Number of parameters.
    param_dict: Dict[str, Any]
        Dictionary that maps parameters to their response scale.
    distribution_arg_names: List
        List of distributional parameter names.
    target_transform: Transform
        Specify the target transform.
    discrete: bool
        Whether the target is discrete or not.
    univariate: bool
        Whether the distribution is univariate or multivariate.
    stabilization: str
        Stabilization method. Options are "None", "MAD" or "L2".
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood) or "crps" (continuous ranked probability score).
        Note that if "crps" is used, the Hessian is set to 1, as the current CRPS version is not twice differentiable.
        Hence, using the CRPS disregards any variation in the curvature of the loss function.
    """
    def __init__(self,
                 base_dist: torch.distributions.Distribution = None,
                 flow_transform: Transform = None,
                 count_bins: Optional[int] = 8,
                 bound: Optional[float] = 3.0,
                 order: Optional[str] = "quadratic",
                 n_dist_param: int = None,
                 param_dict: Dict[str, Any] = None,
                 distribution_arg_names: List = None,
                 target_transform: Transform = None,
                 discrete: bool = False,
                 univariate: bool = True,
                 stabilization: str = "None",
                 loss_fn: str = "nll",
                 ):

        self.base_dist = base_dist
        self.flow_transform = flow_transform
        self.count_bins = count_bins
        self.bound = bound
        self.order = order
        self.n_dist_param = n_dist_param
        self.param_dict = param_dict
        self.distribution_arg_names = distribution_arg_names
        self.target_transform = target_transform
        self.discrete = discrete
        self.univariate = univariate
        self.stabilization = stabilization
        self.loss_fn = loss_fn

    def objective_fn(self, predt: np.ndarray, data: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:

        """
        Function to estimate gradients and hessians of normalizing flow parameters.

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
        target = torch.tensor(data.get_label().reshape(-1, 1))

        # Weights
        if data.get_weight().size == 0:
            # Use 1 as weight if no weights are specified
            weights = torch.ones_like(target, dtype=target.dtype).numpy()
        else:
            weights = data.get_weight().reshape(-1, 1)

        # Start values (needed to replace NaNs in predt)
        start_values = data.get_base_margin().reshape(-1, self.n_dist_param)[0, :].tolist()

        # Calculate gradients and hessians
        predt, loss = self.get_params_loss(predt, target, start_values)
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
        target = torch.tensor(data.get_label().reshape(-1, 1))

        # Start values (needed to replace NaNs in predt)
        start_values = data.get_base_margin().reshape(-1, self.n_dist_param)[0, :].tolist()

        # Calculate loss
        _, loss = self.get_params_loss(predt, target, start_values)

        return self.loss_fn, loss

    def calculate_start_values(self,
                               target: np.ndarray,
                               max_iter: int = 50
                               ) -> Tuple[float, np.ndarray]:
        """
        Function that calculates starting values for each parameter.

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
            Starting values for each parameter.
        """
        # Convert target to torch.tensor
        target = torch.tensor(target).reshape(-1, 1)

        # Create Normalizing Flow
        flow_dist = self.create_spline_flow(input_dim=1)

        # Specify optimizer
        optimizer = LBFGS(flow_dist.transforms[0].parameters(),
                          lr=0.3,
                          max_iter=np.min([int(max_iter/4), 50]),
                          line_search_fn="strong_wolfe")

        # Define learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        # Define closure
        def closure():
            optimizer.zero_grad()
            loss = -torch.nansum(flow_dist.log_prob(target))
            loss.backward()
            flow_dist.clear_cache()
            return loss

        # Optimize parameters
        loss_vals = []
        tolerance = 1e-5           # Tolerance level for loss change
        patience = 5               # Patience level for loss change
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
        start_values = list(flow_dist.transforms[0].parameters())
        start_values = torch.cat([param.view(-1) for param in start_values]).detach().numpy()

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
            Starting values for each parameter.

        Returns
        -------
        predt: torch.Tensor
            Predicted parameters.
        loss: torch.Tensor
            Loss value.
        """
        # Reshape Target
        target = target.view(-1)

        # Predicted Parameters
        predt = predt.reshape(-1, self.n_dist_param)

        # Replace NaNs and infinity values with unconditional start values
        nan_inf_mask = np.isnan(predt) | np.isinf(predt)
        predt[nan_inf_mask] = np.take(start_values, np.where(nan_inf_mask)[1])

        # Convert to torch.tensor
        predt = torch.tensor(predt, dtype=torch.float32)

        # Specify Normalizing Flow
        flow_dist = self.create_spline_flow(target.shape[0])

        # Replace parameters with estimated ones
        params, flow_dist = self.replace_parameters(predt, flow_dist)

        # Calculate loss
        if self.loss_fn == "nll":
            loss = -torch.nansum(flow_dist.log_prob(target))
        elif self.loss_fn == "crps":
            torch.manual_seed(123)
            dist_samples = flow_dist.rsample((30,)).squeeze(-1)
            loss = torch.nansum(self.crps_score(target, dist_samples))
        else:
            raise ValueError("Invalid loss function. Please select 'nll' or 'crps'.")

        return params, loss

    def create_spline_flow(self,
                           input_dim: int = None,
                           ) -> Transform:

        """
        Function that constructs a Normalizing Flow.

        Arguments
        ---------
        input_dim: int
            Input dimension.

        Returns
        -------
        spline_flow: Transform
            Normalizing Flow.
        """

        # Create flow distribution (currently only Normal)
        loc, scale = torch.zeros(input_dim), torch.ones(input_dim)
        flow_dist = self.base_dist(loc, scale)

        # Create Spline Transform
        torch.manual_seed(123)
        spline_transform = self.flow_transform(input_dim,
                                               count_bins=self.count_bins,
                                               bound=self.bound,
                                               order=self.order)

        # Create Normalizing Flow
        spline_flow = TransformedDistribution(flow_dist, [spline_transform, self.target_transform])

        return spline_flow

    def replace_parameters(self,
                           params: torch.Tensor,
                           flow_dist: Transform,
                           ) -> Tuple[List, Transform]:
        """
        Replace parameters with estimated ones.

        Arguments
        ---------
        params: torch.Tensor
            Estimated parameters.
        flow_dist: Transform
            Normalizing Flow.

        Returns
        -------
        params_list: List
            List of estimated parameters.
        flow_dist: Transform
            Normalizing Flow with estimated parameters.
        """

        # Split parameters into list
        if self.order == "quadratic":
            params_list = torch.split(
                params, [self.count_bins, self.count_bins, self.count_bins - 1],
                dim=1)
        elif self.order == "linear":
            params_list = torch.split(
                params, [self.count_bins, self.count_bins, self.count_bins - 1, self.count_bins],
                dim=1)

        # Replace parameters
        for param, new_value in zip(flow_dist.transforms[0].parameters(), params_list):
            param.data = new_value

        # Get parameters (including require_grad=True)
        params_list = list(flow_dist.transforms[0].parameters())

        return params_list, flow_dist

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

        # Specify Normalizing Flow
        pred_params = torch.tensor(predt_params.values)
        flow_dist_pred = self.create_spline_flow(pred_params.shape[0])

        # Replace parameters with estimated ones
        _, flow_dist_pred = self.replace_parameters(pred_params, flow_dist_pred)

        # Draw samples
        flow_samples = pd.DataFrame(flow_dist_pred.sample((n_samples,)).squeeze().detach().numpy().T)
        flow_samples.columns = [str("y_sample") + str(i) for i in range(flow_samples.shape[1])]

        if self.discrete:
            flow_samples = flow_samples.astype(int)

        return flow_samples

    def predict_dist(self,
                     booster: xgb.Booster,
                     start_values: np.ndarray,
                     data: xgb.DMatrix,
                     pred_type: str = "parameters",
                     n_samples: int = 1000,
                     quantiles: list = [0.1, 0.5, 0.9],
                     seed: int = 123
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

        # Predict distributional parameters
        dist_params_predt = pd.DataFrame(
            np.array(booster.predict(data, output_margin=True)).reshape(-1, self.n_dist_param)
        )
        dist_params_predt.columns = self.param_dict.keys()

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
                                       loss: torch.tensor,
                                       predt: torch.tensor,
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
        if self.loss_fn == "nll":
            # Gradient and Hessian
            grad = autograd(loss, inputs=predt, create_graph=True)
            hess = [autograd(grad[i].nansum(), inputs=predt[i], retain_graph=True)[0] for i in range(len(grad))]
        elif self.loss_fn == "crps":
            # Gradient and Hessian
            grad = autograd(loss, inputs=predt, create_graph=True)
            hess = [torch.ones_like(grad[i]) for i in range(len(grad))]

        # Stabilization of Derivatives
        if self.stabilization != "None":
            grad = [self.stabilize_derivative(grad[i], type=self.stabilization) for i in range(len(grad))]
            hess = [self.stabilize_derivative(hess[i], type=self.stabilization) for i in range(len(hess))]

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

    def stabilize_derivative(self, input_der: torch.Tensor, type: str = "MAD") -> torch.Tensor:
        """
        Function that stabilizes Gradients and Hessians.

        Since parameters are estimated by optimizing Gradients and Hessians, it is important that these are comparable
        in magnitude for all parameters. Due to imbalances regarding the ranges, the estimation might become unstable
        so that it does not converge (or converge very slowly) to the optimal solution. Another way to improve
        convergence might be to standardize the response variable. This is especially useful if the range of the
        response differs strongly from the range of the Gradients and Hessians. Both, the stabilization and the
        standardization of the response are not always advised but need to be carefully considered.

        Source
        ---------
        https://github.com/boost-R/gamboostLSS/blob/7792951d2984f289ed7e530befa42a2a4cb04d1d/R/helpers.R#L173

        Arguments
        ---------
        input_der : torch.Tensor
            Input derivative, either Gradient or Hessian.
        type: str
            Stabilization method. Can be either "None", "MAD" or "L2".

        Returns
        ---------
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

    def crps_score(self, y: torch.tensor, yhat_dist: torch.tensor) -> torch.tensor:
        """
        Function that calculates the Continuous Ranked Probability Score (CRPS) for a given set of predicted samples.

        Arguments
        ---------
        y: torch.Tensor
            Response variable of shape (n_observations,1).
        yhat_dist: torch.Tensor
            Predicted samples of shape (n_samples, n_observations).

        Returns
        ---------
        crps: torch.Tensor
            CRPS score.

        References
        ---------
        Gneiting, Tilmann & Raftery, Adrian. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation.
        Journal of the American Statistical Association. 102. 359-378.

        Source
        ---------
        https://github.com/elephaint/pgbm/blob/main/pgbm/torch/pgbm_dist.py#L549
        """
        # Get the number of observations
        n_samples = yhat_dist.shape[0]

        # Sort the forecasts in ascending order
        yhat_dist_sorted, _ = torch.sort(yhat_dist, 0)

        # Create temporary tensors
        y_cdf = torch.zeros_like(y)
        yhat_cdf = torch.zeros_like(y)
        yhat_prev = torch.zeros_like(y)
        crps = torch.zeros_like(y)

        # Loop over the predicted samples generated per observation
        for yhat in yhat_dist_sorted:
            yhat = yhat.reshape(-1, 1)
            flag = (y_cdf == 0) * (y < yhat)
            crps += flag * ((y - yhat_prev) * yhat_cdf ** 2)
            crps += flag * ((yhat - y) * (yhat_cdf - 1) ** 2)
            crps += (~flag) * ((yhat - yhat_prev) * (yhat_cdf - y_cdf) ** 2)
            y_cdf += flag
            yhat_cdf += 1 / n_samples
            yhat_prev = yhat

        # In case y_cdf == 0 after the loop
        flag = (y_cdf == 0)
        crps += flag * (y - yhat)

        return crps

    def flow_select(self,
                    target: np.ndarray,
                    candidate_flows: List,
                    max_iter: int = 100,
                    plot: bool = False,
                    figure_size: tuple = (10, 5),
                    ) -> pd.DataFrame:
        """
        Function that selects the most suitable normalizing flow specification among the candidate_flow for the
        target variable, based on the NegLogLikelihood (lower is better).

        Parameters
        ----------
        target: np.ndarray
            Response variable.
        candidate_flows: List
            List of candidate normalizing flow specifications.
        max_iter: int
            Maximum number of iterations for the optimization.
        plot: bool
            If True, a density plot of the actual and fitted distribution is created.
        figure_size: tuple
            Figure size of the density plot.

        Returns
        -------
        fit_df: pd.DataFrame
            Dataframe with the loss values of the fitted normalizing flow.
        """
        flow_list = []
        total_iterations = len(candidate_flows)

        with tqdm(total=total_iterations, desc="Fitting candidate normalizing flows") as pbar:
            for flow in candidate_flows:
                flow_name = str(flow.__class__).split(".")[-1].split("'>")[0]
                flow_spec = f"(count_bins: {flow.count_bins}, order: {flow.order})"
                flow_name = flow_name + flow_spec
                pbar.set_description(f"Fitting {flow_name}")
                flow_sel = flow
                try:
                    loss, params = flow_sel.calculate_start_values(target=target, max_iter=max_iter)
                    fit_df = pd.DataFrame.from_dict(
                        {flow_sel.loss_fn: loss.reshape(-1, ),
                         "NormFlow": str(flow_name),
                         "params": [params]
                         }
                    )
                except Exception as e:
                    warnings.warn(f"Error fitting {flow_sel} NormFlow: {str(e)}")
                    fit_df = pd.DataFrame(
                        {flow_sel.loss_fn: np.nan,
                         "NormFlow": str(flow_sel),
                         "params": [np.nan] * flow_sel.n_dist_param
                         }
                    )
                flow_list.append(fit_df)
                pbar.update(1)
            pbar.set_description(f"Fitting of candidate normalizing flows completed")
            fit_df = pd.concat(flow_list).sort_values(by=flow_sel.loss_fn, ascending=True)
            fit_df["rank"] = fit_df[flow_sel.loss_fn].rank().astype(int)
            fit_df.set_index(fit_df["rank"], inplace=True)
        if plot:
            from skbase.utils.dependencies import _check_soft_dependencies

            msg = (
                "flow_select with plot=True requires 'matplotlib' and 'seaborn' "
                "to be installed. Please install the packages to use this feature. "
                "Installing via pip install xgboostlss[all_extras] also installs "
                "the required dependencies."
            )
            _check_soft_dependencies(["matplotlib", "seaborn"], msg=msg)

            import matplotlib.pyplot as plt
            import seaborn as sns

            # Select normalizing flow with the lowest loss
            best_flow = fit_df[fit_df["rank"] == 1].reset_index(drop=True)
            for flow in candidate_flows:
                flow_name = str(flow.__class__).split(".")[-1].split("'>")[0]
                flow_spec = f"(count_bins: {flow.count_bins}, order: {flow.order})"
                flow_name = flow_name + flow_spec
                if flow_name == best_flow["NormFlow"].values[0]:
                    best_flow_sel = flow
                    break

            # Draw samples from distribution
            flow_params = torch.tensor(best_flow["params"][0]).reshape(1, -1)
            flow_dist_sel = best_flow_sel.create_spline_flow(input_dim=1)
            _, flow_dist_sel = best_flow_sel.replace_parameters(flow_params, flow_dist_sel)
            n_samples = np.max([10000, target.shape[0]])
            n_samples = np.where(n_samples > 500000, 100000, n_samples)
            flow_samples = pd.DataFrame(flow_dist_sel.sample((n_samples,)).squeeze().detach().numpy().T).values

            # Plot actual and fitted distribution
            plt.figure(figsize=figure_size)
            sns.kdeplot(target.reshape(-1, ), label="Actual")
            sns.kdeplot(flow_samples.reshape(-1, ), label=f"Best-Fit: {best_flow['NormFlow'].values[0]}")
            plt.legend()
            plt.title("Actual vs. Best-Fit Density", fontweight="bold", fontsize=16)
            plt.show()

        fit_df.drop(columns=["rank", "params"], inplace=True)

        return fit_df
