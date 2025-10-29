import torch
from torch.autograd import grad as autograd
from torch.optim import LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau

import xgboost as xgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Any, Dict, Optional, List, Tuple, Callable
import seaborn as sns
import warnings


class Multivariate_DistributionClass:
    """
    Generic class that contains general functions for multivariate distributions.

    Arguments
    ---------
    distribution: torch.distributions.Distribution
        PyTorch Distribution class.
    univariate: bool
        Whether the distribution is univariate or multivariate.
    distribution_arg_names: List
        List of distributional parameter names.
    n_targets: int
        Number of targets.
    rank: Optional[int]
        Rank of the low-rank form of the covariance matrix.
    n_dist_param: int
        Number of distributional parameters.
    param_dict: Dict[str, Any]
        Dictionary that maps distributional parameters to their response scale.
    param_transform: Callable
        Function that transforms the distributional parameters into the required format.
    get_dist_params: Callable
        Function that returns the distributional parameters.
    discrete: bool
        Whether the support of the distribution is discrete or continuous.
    stabilization: str
        Stabilization method.
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood).
    """
    def __init__(self,
                 distribution: torch.distributions.Distribution = None,
                 univariate: bool = False,
                 distribution_arg_names: List = None,
                 n_targets: int = 2,
                 rank: Optional[int] = None,
                 n_dist_param: int = None,
                 param_dict: Dict[str, Any] = None,
                 param_transform: Callable = None,
                 get_dist_params: Callable = None,
                 discrete: bool = False,
                 stabilization: str = "None",
                 loss_fn: str = "nll",
                 ):

        self.distribution = distribution
        self.univariate = univariate
        self.distribution_arg_names = distribution_arg_names
        self.n_targets = n_targets
        self.rank = rank
        self.n_dist_param = n_dist_param
        self.param_dict = param_dict
        self.param_transform = param_transform
        self.get_dist_params = get_dist_params
        self.discrete = discrete
        self.stabilization = stabilization
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
        target = torch.tensor(data.get_label().reshape(-1, self.n_dist_param))[:, :self.n_targets]

        # Weights
        if data.get_weight().size == 0:
            # Use 1 as weight if no weights are specified
            weights = torch.ones_like(target[:, 0], dtype=target.dtype).numpy().reshape(-1, 1)
        else:
            weights = data.get_weight().reshape(-1, 1)

        # Start values (needed to replace NaNs in predt)
        start_values = data.get_base_margin().reshape(-1, self.n_dist_param)[0, :].tolist()

        # Calculate gradients and hessians
        predt, loss = self.get_params_loss(predt, target, start_values, requires_grad=True)
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
        target = torch.tensor(data.get_label().reshape(-1, self.n_dist_param))[:, :self.n_targets]

        # Start values (needed to replace NaNs in predt)
        start_values = data.get_base_margin().reshape(-1, self.n_dist_param)[0, :].tolist()

        # Calculate loss
        _, loss = self.get_params_loss(predt, target, start_values, requires_grad=False)

        return self.loss_fn, loss

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
        params = [
            torch.where(torch.isnan(tensor) | torch.isinf(tensor), torch.tensor(0.5), tensor) for tensor in params
        ]

        # Transform parameters to response scale
        params = self.param_transform(params, self.param_dict, self.n_targets, rank=self.rank, n_obs=1)

        # Specify Distribution and Loss
        if self.distribution.__name__ == "Dirichlet":
            dist_kwargs = dict(zip(self.distribution_arg_names, [params]))
        else:
            dist_kwargs = dict(zip(self.distribution_arg_names, params))
        dist_fit = self.distribution(**dist_kwargs)
        loss = -torch.nansum(dist_fit.log_prob(target))

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
        target = torch.tensor(target.reshape(-1, self.n_dist_param))[:, :self.n_targets]

        # Initialize parameters
        params = [
            torch.tensor(0.5, dtype=torch.float64).reshape(-1, 1).requires_grad_(True) for _ in range(self.n_dist_param)
        ]

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
        for epoch in range(max_iter):
            loss = optimizer.step(closure)
            lr_scheduler.step(loss)
            loss_vals.append(loss.item())

        # Get final loss
        loss = np.array(loss_vals[-1])

        # Get start values
        start_values = np.array([params[i][0].detach().numpy() for i in range(self.n_dist_param)])

        # Replace any remaining NaNs or infinity values with 0.5
        start_values = np.nan_to_num(start_values, nan=0.5, posinf=0.5, neginf=0.5).reshape(-1,)

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
        predt: torch.Tensor
            Predicted parameters.
        loss: torch.Tensor
            Loss value.
        """
        # Number of observations
        n_obs = target.shape[0]

        # Predicted Parameters
        predt = predt.reshape(-1, self.n_dist_param)

        # Replace NaNs and infinity values with unconditional start values
        nan_inf_mask = np.isnan(predt) | np.isinf(predt)
        predt[nan_inf_mask] = np.take(start_values, np.where(nan_inf_mask)[1])

        # Convert to torch.tensor
        predt = [
            torch.tensor(predt[:, i].reshape(-1, 1), requires_grad=requires_grad) for i in range(self.n_dist_param)
        ]

        # Predicted Parameters transformed to response scale
        predt_transformed = self.param_transform(predt, self.param_dict, self.n_targets, rank=self.rank, n_obs=n_obs)

        # Specify Distribution and Loss
        if self.distribution.__name__ == "Dirichlet":
            dist_kwargs = dict(zip(self.distribution_arg_names, [predt_transformed]))
        else:
            dist_kwargs = dict(zip(self.distribution_arg_names, predt_transformed))
        dist_fit = self.distribution(**dist_kwargs)
        loss = -torch.nansum(dist_fit.log_prob(target))

        return predt, loss

    def draw_samples(self,
                     dist_pred: torch.distributions.Distribution,
                     n_samples: int = 1000,
                     seed: int = 123
                     ) -> pd.DataFrame:
        """
        Function that draws n_samples from a predicted distribution.

        Arguments
        ---------
        dist_pred: torch.distributions.Distribution
            Predicted distribution.
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
        dist_samples = dist_pred.sample((n_samples,)).detach().numpy().T
        if self.discrete:
            dist_samples = dist_samples.astype(int)

        samples_list = []
        for i in range(self.n_targets):
            target_df = pd.DataFrame.from_dict({"target": [f"y{i + 1}" for _ in range(dist_samples.shape[1])]})
            df_samples = pd.DataFrame(dist_samples[i, :])
            df_samples.columns = [str("y_sample") + str(i) for i in range(n_samples)]
            samples_list.append(pd.concat([target_df, df_samples], axis=1))

        samples_df = pd.concat(samples_list, axis=0).reset_index(drop=True)

        return samples_df

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
        base_margin_pred = (np.ones(shape=(data.num_row(), 1))) * start_values
        data.set_base_margin(base_margin_pred.flatten())

        # Predict from model
        n_obs = data.num_row()
        predt = np.array(booster.predict(data, output_margin=True)).reshape(-1, self.n_dist_param)
        predt = [torch.tensor(predt[:, i].reshape(-1, 1), dtype=torch.float32) for i in range(self.n_dist_param)]
        dist_params_predt = self.param_transform(predt, self.param_dict, self.n_targets, rank=self.rank, n_obs=n_obs)

        # Predicted Distributional Parameters
        if self.distribution.__name__ == "Dirichlet":
            dist_kwargs = dict(zip(self.distribution_arg_names, [dist_params_predt]))
        else:
            dist_kwargs = dict(zip(self.distribution_arg_names, dist_params_predt))
        dist_pred = self.distribution(**dist_kwargs)

        # Draw samples from predicted response distribution
        pred_samples_df = self.draw_samples(dist_pred=dist_pred, n_samples=n_samples, seed=seed)

        # Get predicted distributional parameters
        predt_params_df = self.get_dist_params(n_targets=self.n_targets, dist_pred=dist_pred)

        if pred_type == "parameters":
            return predt_params_df

        elif pred_type == "samples":
            return pred_samples_df

        elif pred_type == "quantiles":
            # Calculate quantiles from predicted response distribution
            targets = pred_samples_df["target"]
            pred_quant_df = pred_samples_df.drop(columns="target")
            pred_quant_df = pred_quant_df.quantile(quantiles, axis=1).T
            pred_quant_df.columns = [str("quant_") + str(quantiles[i]) for i in range(len(quantiles))]
            if self.discrete:
                pred_quant_df = pred_quant_df.astype(int)
            pred_quant_df = pd.concat([targets, pred_quant_df], axis=1)

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
        # Calculate gradients and hessians
        grad = autograd(loss, inputs=predt, create_graph=True)
        hess = [autograd(grad[i].nansum(), inputs=predt[i], retain_graph=True)[0] for i in range(len(grad))]

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
            input_der = torch.nan_to_num(input_der, nan=float(torch.nansum(input_der)))
            div = torch.nanmedian(torch.abs(input_der - torch.nanmedian(input_der)))
            div = torch.where(div < torch.tensor(1e-04), torch.tensor(1e-04), div)
            stab_der = input_der / div

        if type == "L2":
            input_der = torch.nan_to_num(input_der, nan=float(torch.nansum(input_der)))
            div = torch.sqrt(torch.nansum(input_der.pow(2)))
            div = torch.where(div < torch.tensor(1e-04), torch.tensor(1e-04), div)
            div = torch.where(div > torch.tensor(10000.0), torch.tensor(10000.0), div)
            stab_der = input_der / div

        if type == "None":
            stab_der = torch.nan_to_num(input_der, nan=float(torch.nansum(input_der)))

        return stab_der


    def dist_select(self,
                    target: np.ndarray,
                    candidate_distributions: List,
                    max_iter: int = 100,
                    plot: bool = False,
                    ncol: int = 3,
                    height: float = 4,
                    sharex: bool = True,
                    sharey: bool = True,
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
        ncol: int
            Number of columns for the facetting of the density plots.
        height: Float
            Height (in inches) of each facet.
        sharex: bool
            Whether to share the x-axis across the facets.
        sharey: bool
            Whether to share the y-axis across the facets.

        Returns
        -------
        fit_df: pd.DataFrame
            Dataframe with the loss values of the fitted candidate distributions.
        """
        dist_list = []
        total_iterations = len(candidate_distributions)

        with tqdm(total=total_iterations, desc="Fitting candidate distributions") as pbar:
            for i in range(len(candidate_distributions)):
                dist_name = candidate_distributions[i].__class__.__name__
                if dist_name == "MVN_LoRa":
                    dist_name = dist_name + f"(rank={candidate_distributions[i].rank})"
                pbar.set_description(f"Fitting {dist_name} distribution")
                dist_sel = candidate_distributions[i]
                target_expand = dist_sel.target_append(target, dist_sel.n_targets, dist_sel.n_dist_param)
                try:
                    loss, params = dist_sel.calculate_start_values(target=target_expand, max_iter=max_iter)
                    fit_df = pd.DataFrame.from_dict(
                        {dist_sel.loss_fn: loss.reshape(-1,),
                         "distribution": str(dist_name),
                         "params": [params]
                         }
                    )
                except Exception as e:
                    warnings.warn(f"Error fitting {dist_name} distribution: {str(e)}")
                    fit_df = pd.DataFrame(
                        {dist_sel.loss_fn: np.nan,
                         "distribution": str(dist_name),
                         "params": [np.nan] * dist_sel.n_dist_param
                        }
                    )
                dist_list.append(fit_df)
                pbar.update(1)
            pbar.set_description(f"Fitting of candidate distributions completed")
            fit_df = pd.concat(dist_list).sort_values(by=dist_sel.loss_fn, ascending=True)
            fit_df["rank"] = fit_df[dist_sel.loss_fn].rank().astype(int)
            fit_df.set_index(fit_df["rank"], inplace=True)
        if plot:
            from skbase.utils.dependencies import _check_soft_dependencies

            msg = (
                "dist_select with plot=True requires 'seaborn' "
                "to be installed. Please install the packages to use this feature. "
                "Installing via pip install xgboostlss[all_extras] also installs "
                "the required dependencies."
            )
            _check_soft_dependencies(["seaborn"], msg=msg)

            import seaborn as sns

            warnings.simplefilter(action='ignore', category=UserWarning)
            # Select distribution
            best_dist = fit_df[fit_df["rank"] == 1].reset_index(drop=True)
            for dist in candidate_distributions:
                dist_name = dist.__class__.__name__
                if dist_name == "MVN_LoRa":
                    dist_name = dist_name + f"(rank={dist.rank})"
                if dist_name == best_dist["distribution"].values[0]:
                    best_dist_sel = dist
                    break

            # Draw samples from distribution
            dist_params = [
                torch.tensor(best_dist["params"][0][i].reshape(-1, 1)) for i in range(best_dist_sel.n_dist_param)
            ]
            dist_params = best_dist_sel.param_transform(dist_params,
                                                        best_dist_sel.param_dict,
                                                        n_targets=best_dist_sel.n_targets,
                                                        rank=best_dist_sel.rank,
                                                        n_obs=1)

            if best_dist["distribution"][0] == "Dirichlet":
                dist_kwargs = dict(zip(best_dist_sel.distribution_arg_names, [dist_params]))
            else:
                dist_kwargs = dict(zip(best_dist_sel.distribution_arg_names, dist_params))
            dist_fit = best_dist_sel.distribution(**dist_kwargs)
            n_samples = np.max([1000, target.shape[0]])
            n_samples = np.where(n_samples > 10000, 1000, n_samples)
            df_samples = best_dist_sel.draw_samples(dist_fit, n_samples=n_samples, seed=123)

            # Plot actual and fitted distribution
            df_samples["type"] = f"Best-Fit: {best_dist['distribution'].values[0]}"
            df_samples = df_samples.melt(id_vars=["target", "type"]).drop(columns="variable")

            df_actual = pd.DataFrame(target)
            df_actual.columns = [f"y{i + 1}" for i in range(best_dist_sel.n_targets)]
            df_actual["type"] = "Actual"
            df_actual = df_actual.melt(id_vars="type", var_name="target")[df_samples.columns]

            plot_df = pd.concat([df_actual, df_samples])

            g = sns.FacetGrid(
                plot_df,
                col="target",
                hue="type",
                col_wrap=ncol,
                height=height,
                sharex=sharex,
                sharey=sharey,
            )
            g.map(sns.kdeplot, "value", lw=2.5)
            handles, labels = g.axes[0].get_legend_handles_labels()
            g.fig.legend(handles, labels, loc='upper center', ncol=len(labels), title="", bbox_to_anchor=(0.5, 0.92))
            g.fig.suptitle("Actual vs. Best-Fit Density", weight="bold", fontsize=16)
            g.fig.tight_layout(rect=[0, 0, 1, 0.9])

        fit_df.drop(columns=["rank", "params"], inplace=True)

        return fit_df

    def target_append(self,
                      target: np.ndarray,
                      n_targets: int,
                      n_dist_param: int
                      ) -> np.ndarray:
        """
        Function that appends target to the number of specified parameters.

        Arguments
        ---------
        target: np.ndarray
            Target variables.
        n_targets: int
            Number of targets.
        n_dist_param: int
            Number of distribution parameters.

        Returns
        -------
        label: np.ndarray
            Array with appended targets.
        """
        label = target.reshape(-1, n_targets)
        n_obs = label.shape[0]
        n_fill = n_dist_param - n_targets
        np_fill = np.ones((n_obs, n_fill))
        label_append = np.concatenate([label, np_fill], axis=1).reshape(-1, n_dist_param)

        return label_append
