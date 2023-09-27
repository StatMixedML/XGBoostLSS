import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.core import (
    Booster, 
    DMatrix,
)

from xgboost.callback import (
    TrainingCallback,
)

import os
from xgboostlss.utils import *
import shap
from typing import Any, Dict, Optional, Sequence, Tuple, Union


class XGBoostLSS:
    """
    XGBoostLSS model class

    Parameters
    ----------
    dist : Distribution
        DistributionClass object.
    start_values : np.ndarray
        Starting values for each distributional parameter.
    """
    def __init__(self, dist):
        self.dist = dist             # Distribution object
        self.start_values = None     # Starting values for distributional parameters
        self.multivariate_label_expand = False
        self.multivariate_eval_label_expand = False

    def set_params_adj(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set parameters for distributional model.

        Arguments
        ---------
        params : Dict[str, Any]
            Parameters for model.

        Returns
        -------
        params : Dict[str, Any]
            Updated Parameters for model.
        """
        params_adj = {
            "objective": None,
            "base_score": 0,
            "num_target": self.dist.n_dist_param,
            "disable_default_eval_metric": True
        }
        params.update(params_adj)

        return params

    def adjust_labels(self, dmatrix: DMatrix) -> None:
        """
        Adjust labels for multivariate distributions.

        Arguments
        ---------
        dmatrix : DMatrix
            DMatrix object.

        Returns
        -------
        None
        """
        if not (self.dist.univariate or self.multivariate_label_expand):
            self.multivariate_label_expand = True
            label = self.dist.target_append(
                dmatrix.get_label(),
                self.dist.n_targets,
                self.dist.n_dist_param
            )
            dmatrix.set_label(label)

    def set_base_margin(self, dmatrix: DMatrix) -> None:
        """
        Set base margin for distributions.

        Arguments
        ---------
        dmatrix : DMatrix
            DMatrix object.

        Returns
        -------
        None
        """
        if self.start_values is None:
            _, self.start_values = self.dist.calculate_start_values(dmatrix.get_label())
        base_margin = np.ones(shape=(dmatrix.num_row(), 1)) * self.start_values
        dmatrix.set_base_margin(base_margin.flatten())

    def train(
            self,
            params: Dict[str, Any],
            dtrain: DMatrix,
            num_boost_round: int = 10,
            *,
            evals: Optional[Sequence[Tuple[DMatrix, str]]] = None,
            early_stopping_rounds: Optional[int] = None,
            evals_result: Optional[TrainingCallback.EvalsLog] = None,
            verbose_eval: Optional[Union[bool, int]] = True,
            xgb_model: Optional[Union[str, os.PathLike, Booster, bytearray]] = None,
            callbacks: Optional[Sequence[TrainingCallback]] = None,
    ) -> Booster:
            """
            Train a booster with given parameters.

            Arguments
            ---------
            params :
                Booster params.
            dtrain :
                Data to be trained.
            num_boost_round :
                Number of boosting iterations.
            evals :
                List of validation sets for which metrics will evaluated during training.
                Validation metrics will help us track the performance of the model.
            early_stopping_rounds :
                Activates early stopping. Validation metric needs to improve at least once in
                every **early_stopping_rounds** round(s) to continue training.
                Requires at least one item in **evals**.
                The method returns the model from the last iteration (not the best one).  Use
                custom callback or model slicing if the best model is desired.
                If there's more than one item in **evals**, the last entry will be used for early
                stopping.
                If there's more than one metric in the **eval_metric** parameter given in
                **params**, the last metric will be used for early stopping.
                If early stopping occurs, the model will have two additional fields:
                ``bst.best_score``, ``bst.best_iteration``.
            evals_result :
                This dictionary stores the evaluation results of all the items in watchlist.
                Example: with a watchlist containing
                ``[(dtest,'eval'), (dtrain,'train')]`` and
                a parameter containing ``('eval_metric': 'logloss')``,
                the **evals_result** returns
                .. code-block:: python
                    {'train': {'logloss': ['0.48253', '0.35953']},
                     'eval': {'logloss': ['0.480385', '0.357756']}}
            verbose_eval :
                Requires at least one item in **evals**.
                If **verbose_eval** is True then the evaluation metric on the validation set is
                printed at each boosting stage.
                If **verbose_eval** is an integer then the evaluation metric on the validation set
                is printed at every given **verbose_eval** boosting stage. The last boosting stage
                / the boosting stage found by using **early_stopping_rounds** is also printed.
                Example: with ``verbose_eval=4`` and at least one item in **evals**, an evaluation metric
                is printed every 4 boosting stages, instead of every boosting stage.
            xgb_model :
                Xgb model to be loaded before training (allows training continuation).
            callbacks :
                List of callback functions that are applied at end of each iteration.
                It is possible to use predefined callbacks by using
                :ref:`Callback API <callback_api>`.
                .. note::
                   States in callback are not preserved during training, which means callback
                   objects can not be reused for multiple training sessions without
                   reinitialization or deepcopy.
                .. code-block:: python
                    for params in parameters_grid:
                        # be sure to (re)initialize the callbacks before each run
                        callbacks = [xgb.callback.LearningRateScheduler(custom_rates)]
                        xgboost.train(params, Xy, callbacks=callbacks)

            Returns
            -------
            Booster:
                The trained booster model.
            """
            self.set_params_adj(params)
            self.adjust_labels(dtrain)
            self.set_base_margin(dtrain)

            # Set base_margin for evals
            if evals:
                self.set_eval_margin(evals, self.start_values)

            self.booster = xgb.train(params,
                                     dtrain,
                                     num_boost_round=num_boost_round,
                                     evals=evals,
                                     obj=self.dist.objective_fn,
                                     custom_metric=self.dist.metric_fn,
                                     xgb_model=xgb_model,
                                     callbacks=callbacks,
                                     verbose_eval=verbose_eval,
                                     evals_result=evals_result,
                                     maximize=False,
                                     early_stopping_rounds=early_stopping_rounds)
            return self.booster.copy()

    def predict(self,
                data: xgb.DMatrix,
                pred_type: str = "parameters",
                n_samples: int = 1000,
                quantiles: list = [0.1, 0.5, 0.9],
                seed: str = 123):
        """
        Function that predicts from the trained model.

        Arguments
        ---------
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
        predt_df : pd.DataFrame
            Predictions.
        """

        # Predict
        predt_df = self.dist.predict_dist(booster=self.booster,
                                          start_values=self.start_values,
                                          data=data,
                                          pred_type=pred_type,
                                          n_samples=n_samples,
                                          quantiles=quantiles,
                                          seed=seed)

        return predt_df

    def plot(self,
             X: pd.DataFrame,
             feature: str = "x",
             parameter: str = "loc",
             max_display: int = 15,
             plot_type: str = "Partial_Dependence"):
        """
        XGBoostLSS SHap plotting function.

        Arguments:
        ---------
        X: pd.DataFrame
            Train/Test Data
        feature: str
            Specifies which feature is to be plotted.
        parameter: str
            Specifies which distributional parameter is to be plotted.
        max_display: int
            Specifies the maximum number of features to be displayed.
        plot_type: str
            Specifies the type of plot:
                "Partial_Dependence" plots the partial dependence of the parameter on the feature.
                "Feature_Importance" plots the feature importance of the parameter.
        """
        explainer = shap.TreeExplainer(self.booster)
        shap_values = explainer(X)

        param_pos = self.dist.distribution_arg_names.index(parameter)

        if plot_type == "Partial_Dependence":
            if self.dist.n_dist_param == 1:
                shap.plots.scatter(shap_values[:, feature], color=shap_values[:, feature])
            else:
                shap.plots.scatter(shap_values[:, feature][:, param_pos], color=shap_values[:, feature][:, param_pos])
        elif plot_type == "Feature_Importance":
            if self.dist.n_dist_param == 1:
                shap.plots.bar(shap_values, max_display=max_display if X.shape[1] > max_display else X.shape[1])
            else:
                shap.plots.bar(
                    shap_values[:, :, param_pos], max_display=max_display if X.shape[1] > max_display else X.shape[1]
                )

    def expectile_plot(self,
                       X: pd.DataFrame,
                       feature: str = "x",
                       expectile: str = "0.05",
                       plot_type: str = "Partial_Dependence"):
        """
        XGBoostLSS function for plotting expectile SHapley values.

        X: pd.DataFrame
            Train/Test Data
        feature: str
            Specifies which feature to use for plotting Partial_Dependence plot.
        expectile: str
            Specifies which expectile to plot.
        plot_type: str
            Specifies which SHapley-plot to visualize. Currently, "Partial_Dependence" and "Feature_Importance"
            are supported.
        """
        explainer = shap.TreeExplainer(self.booster)
        shap_values = explainer(X)

        expect_pos = list(self.dist.param_dict.keys()).index(expectile)

        if plot_type == "Partial_Dependence":
            shap.plots.scatter(shap_values[:, feature][:, expect_pos], color=shap_values[:, feature][:, expect_pos])
        elif plot_type == "Feature_Importance":
            shap.plots.bar(shap_values[:, :, expect_pos], max_display=15 if X.shape[1] > 15 else X.shape[1])

    def set_eval_margin(self,
                        eval_set: list,
                        start_values: np.ndarray
                        ) -> list:

        """
        Function that sets the base margin for the evaluation set.

        Arguments
        ---------
        eval_set : list
            List of tuples containing the train and evaluation set.
        start_values : np.ndarray
            Array containing the start values for each distributional parameter.

        Returns
        -------
        eval_set : list
            List of tuples containing the train and evaluation set.
        """
        eval_label_expanded = False
        for d_eval, _ in eval_set:
            base_margin = (np.ones(shape=(d_eval.num_row(), 1))) * start_values
            d_eval.set_base_margin(base_margin.flatten())

            # Adjust labels to number of distributional parameters
            if not (self.dist.univariate or self.multivariate_eval_label_expand):
                eval_label_expanded = True
                new_label = self.dist.target_append(d_eval.get_label(), self.dist.n_targets, self.dist.n_dist_param)
                d_eval.set_label(new_label)

        if eval_label_expanded:
            self.multivariate_eval_label_expand = True
