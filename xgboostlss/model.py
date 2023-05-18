import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.core import (
    Booster,
    DMatrix,
)

from xgboost.callback import (
    CallbackContainer,
    EarlyStopping,
    EvaluationMonitor,
    TrainingCallback,
)

from xgboost._typing import FPreProcCallable
from xgboost.compat import DataFrame, XGBStratifiedKFold

import os
from xgboostlss.utils import *
import optuna
from optuna.samplers import TPESampler
import shap
from typing import Any, Dict, Optional, Sequence, Tuple, Union


class XGBoostLSS:
    """
    XGBoostLSS model class
    """
    def __init__(self, dist):
        self.dist = dist.dist_class  # Distribution object

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

            params_adj = {"objective": None,
                          "base_score": 0,
                          "num_target": self.dist.n_dist_param,
                          "disable_default_eval_metric": True}

            params.update(params_adj)

            # Set base_margin as starting point for each distributional parameter. Requires base_score=0 in parameters.
            self.start_values = self.dist.calculate_start_values(dtrain.get_label())
            base_margin = (np.ones(shape=(dtrain.num_row(), 1))) * self.start_values
            dtrain.set_base_margin(base_margin.flatten())

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
            return self.booster

    def cv(
        self,
        params: Dict[str, Any],
        dtrain: DMatrix,
        num_boost_round: int = 10,
        nfold: int = 3,
        stratified: bool = False,
        folds: XGBStratifiedKFold = None,
        early_stopping_rounds: Optional[int] = None,
        fpreproc: Optional[FPreProcCallable] = None,
        as_pandas: bool = True,
        verbose_eval: Optional[Union[int, bool]] = None,
        show_stdv: bool = True,
        seed: int = 0,
        callbacks: Optional[Sequence[TrainingCallback]] = None,
        shuffle: bool = True,
    ) -> Union[Dict[str, float], DataFrame]:
        # pylint: disable = invalid-name

        """
        Cross-validation with given parameters.

        Arguments
        ----------
        params : dict
            Booster params.
        dtrain : DMatrix
            Data to be trained.
        num_boost_round : int
            Number of boosting iterations.
        nfold : int
            Number of folds in CV.
        stratified : bool
            Perform stratified sampling.
        folds : a KFold or StratifiedKFold instance or list of fold indices
            Sklearn KFolds or StratifiedKFolds object.
            Alternatively may explicitly pass sample indices for each fold.
            For ``n`` folds, **folds** should be a length ``n`` list of tuples.
            Each tuple is ``(in,out)`` where ``in`` is a list of indices to be used
            as the training samples for the ``n`` th fold and ``out`` is a list of
            indices to be used as the testing samples for the ``n`` th fold.
        early_stopping_rounds: int
            Activates early stopping. Cross-Validation metric (average of validation
            metric computed over CV folds) needs to improve at least once in
            every **early_stopping_rounds** round(s) to continue training.
            The last entry in the evaluation history will represent the best iteration.
            If there's more than one metric in the **eval_metric** parameter given in
            **params**, the last metric will be used for early stopping.
        fpreproc : function
            Preprocessing function that takes (dtrain, dtest, param) and returns
            transformed versions of those.
        as_pandas : bool, default True
            Return pd.DataFrame when pandas is installed.
            If False or pandas is not installed, return np.ndarray
        verbose_eval : bool, int, or None, default None
            Whether to display the progress. If None, progress will be displayed
            when np.ndarray is returned. If True, progress will be displayed at
            boosting stage. If an integer is given, progress will be displayed
            at every given `verbose_eval` boosting stage.
        show_stdv : bool, default True
            Whether to display the standard deviation in progress.
            Results are not affected, and always contains std.
        seed : int
            Seed used to generate the folds (passed to numpy.random.seed).
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
        shuffle : bool
            Shuffle data before creating folds.

        Returns
        -------
        evaluation history : list(string)
        """
        params_adj = {"objective": None,
                      "base_score": 0,
                      "num_target": self.dist.n_dist_param,
                      "disable_default_eval_metric": True}

        params.update(params_adj)

        # Set base_margin as starting point for each distributional parameter. Requires base_score=0 in parameters.
        self.start_values = self.dist.calculate_start_values(dtrain.get_label())
        base_margin = (np.ones(shape=(dtrain.num_row(), 1))) * self.start_values
        dtrain.set_base_margin(base_margin.flatten())

        bstLSS_cv = xgb.cv(params,
                           dtrain,
                           num_boost_round=num_boost_round,
                           nfold=nfold,
                           stratified=stratified,
                           folds=folds,
                           obj=self.dist.objective_fn,
                           custom_metric=self.dist.metric_fn,
                           maximize=False,
                           early_stopping_rounds=early_stopping_rounds,
                           fpreproc=fpreproc,
                           as_pandas=as_pandas,
                           verbose_eval=verbose_eval,
                           show_stdv=show_stdv,
                           seed=seed,
                           callbacks=callbacks,
                           shuffle=shuffle)

        return bstLSS_cv

    def hyper_opt(
        self,
        hp_dict: Dict,
        dtrain: DMatrix,
        num_boost_round=500, 
        nfold=10, 
        early_stopping_rounds=20,
        max_minutes=10, 
        n_trials=None, 
        study_name=None, 
        silence=False,
        seed=None,
        hp_seed=None
    ):
        """
        Function to tune hyperparameters using optuna.

        Arguments
        ----------
        hp_dict: dict
            Dictionary of hyperparameters to tune.
        dtrain: xgb.DMatrix
            Training data.
        num_boost_round: int
            Number of boosting iterations.
        nfold: int
            Number of folds in CV.
        early_stopping_rounds: int
            Activates early stopping. Cross-Validation metric (average of validation
            metric computed over CV folds) needs to improve at least once in
            every **early_stopping_rounds** round(s) to continue training.
            The last entry in the evaluation history will represent the best iteration.
            If there's more than one metric in the **eval_metric** parameter given in
            **params**, the last metric will be used for early stopping.
        max_minutes: int
            Time budget in minutes, i.e., stop study after the given number of minutes.
        n_trials: int
            The number of trials. If this argument is set to None, there is no limitation on the number of trials.
        study_name: str
            Name of the hyperparameter study.
        silence: bool
            Controls the verbosity of the trail, i.e., user can silence the outputs of the trail.
        seed: int
            Seed used to generate the folds (passed to numpy.random.seed).
        hp_seed: int
            Seed for random number generator used in the Bayesian hyper-parameter search.  

        Returns
        -------
        opt_params : dict
            Optimal hyper-parameters.
        """       
        
        def objective(trial):

            hyper_params = {}

            for param_name, param_value in hp_dict.items():

                param_type = param_value[0]
    
                if param_type == "categorical" or param_type == "none":
                    hyper_params.update({param_name: trial.suggest_categorical(param_name, param_value[1])})

                elif param_type == "float":
                    param_constraints = param_value[1]
                    param_low = param_constraints["low"]
                    param_high = param_constraints["high"]
                    param_log = param_constraints["log"]
                    hyper_params.update(
                        {param_name: trial.suggest_float(param_name,
                                                         low=param_low,
                                                         high=param_high,
                                                         log=param_log
                                                         )
                         })

                elif param_type == "int":
                    param_constraints = param_value[1]
                    param_low = param_constraints["low"]
                    param_high = param_constraints["high"]
                    param_log = param_constraints["log"]
                    hyper_params.update(
                        {param_name: trial.suggest_int(param_name,
                                                       low=param_low,
                                                       high=param_high,
                                                       log=param_log
                                                       )
                         })
                    
            # Add booster if not included in dictionary
            if "booster" not in hyper_params.keys():  
                hyper_params.update({"booster": trial.suggest_categorical("booster", ["gbtree"])})
            
            # Add pruning
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-NegLogLikelihood")

            xgblss_param_tuning = self.cv(params=hyper_params,
                                          dtrain=dtrain,
                                          num_boost_round=num_boost_round,
                                          nfold=nfold,
                                          early_stopping_rounds=early_stopping_rounds,
                                          callbacks=[pruning_callback],
                                          seed=seed,
                                          verbose_eval=False
                                          )

            opt_rounds = xgblss_param_tuning["test-NegLogLikelihood-mean"].idxmin() + 1
            trial.set_user_attr("opt_round", int(opt_rounds))

            # Extract the best score
            best_score = np.min(xgblss_param_tuning["test-NegLogLikelihood-mean"])

            return best_score
        
        if study_name is None:
            study_name = "XGBoostLSS Hyper-Parameter Optimization"

        if silence:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
        if hp_seed is not None:
            sampler = TPESampler(seed=hp_seed) 
        else:
            sampler = TPESampler()  

        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize", study_name=study_name)
        study.optimize(objective, n_trials=n_trials, timeout=60 * max_minutes, show_progress_bar=True)

        print("\nHyper-Parameter Optimization successfully finished.")
        print("  Number of finished trials: ", len(study.trials))
        print("  Best trial:")
        opt_param = study.best_trial

        # Add optimal stopping round
        opt_param.params["opt_rounds"] = study.trials_dataframe()["user_attrs_opt_round"][
            study.trials_dataframe()["value"].idxmin()]
        opt_param.params["opt_rounds"] = int(opt_param.params["opt_rounds"])            

        print("    Value: {}".format(opt_param.value))
        print("    Params: ")
        for key, value in opt_param.params.items():
            print("    {}: {}".format(key, value))

        return opt_param.params

    def predict(self,
                dtest: xgb.DMatrix, 
                pred_type: str,
                n_samples: int = 1000, 
                quantiles: list = [0.1, 0.5, 0.9], 
                seed: str = 123):
        """
        Predicts the distributional parameters of the specified distribution.

        Arguments
        ---------
        dtest : xgb.DMatrix
            Test data.
        pred_type : str
            Type of prediction:
            "samples" draws n_samples from the predicted distribution.
            "quantile" calculates the quantiles from the predicted distribution.
            "parameters" returns the predicted distributional parameters.
            "expectiles" returns the predicted expectiles.
        n_samples : int
            Number of samples to draw from the predicted distribution.
        quantiles : list
            List of quantiles to calculate from the predicted distribution.
        seed : int
            Seed for random number generator used to draw samples from the predicted distribution.
        """

        # Set base_margin as starting point for each distributional parameter. Requires base_score=0 in parameters.
        base_margin = (np.ones(shape=(dtest.num_row(), 1))) * self.start_values
        dtest.set_base_margin(base_margin.flatten())

        predt = np.array(self.booster.predict(dtest, output_margin=True)).reshape(-1, self.dist.n_dist_param)
        predt = torch.tensor(predt, dtype=torch.float32)

        dist_params_predt = np.concatenate(
            [
                response_fun(
                    predt[:, i].reshape(-1, 1)).numpy() for i, (dist_param, response_fun) in enumerate(self.dist.param_dict.items())
            ],
            axis=1,
        )

        dist_params_predt = pd.DataFrame(dist_params_predt)
        dist_params_predt.columns = self.dist.param_dict.keys()

        # Draw samples from predicted response distribution
        pred_samples_df = self.dist.draw_samples(predt_params=dist_params_predt,
                                                 n_samples=n_samples,
                                                 seed=seed)

        if pred_type == "parameters":
            return dist_params_predt

        elif pred_type == "expectiles":
            return dist_params_predt

        elif pred_type == "samples":
            return pred_samples_df

        elif pred_type == "quantiles":
            pred_quant_df = pred_samples_df.quantile(quantiles, axis=1).T
            pred_quant_df.columns = [str("quant_") + str(quantiles[i]) for i in range(len(quantiles))]
            return pred_quant_df

    def plot(self,
             X: pd.DataFrame, 
             feature: str = "x", 
             parameter: str = "loc",
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
            Specifies which parameter is to be plotted. Valid parameters are "location", "scale", "df", "tau".
        plot_type: str
            Specifies the type of plot:
                "Partial_Dependence" plots the partial dependence of the parameter on the feature.
                "Feature_Importance" plots the feature importance of the parameter.
        """
        shap.initjs()
        explainer = shap.TreeExplainer(self.booster)
        shap_values = explainer(X)

        param_pos = list(self.dist.param_dict.keys()).index(parameter)

        if plot_type == "Partial_Dependence":
            if self.dist.n_dist_param == 1:
                shap.plots.scatter(shap_values[:, feature], color=shap_values[:, feature])
            else:
                shap.plots.scatter(shap_values[:, feature][:, param_pos], color=shap_values[:, feature][:, param_pos])
        elif plot_type == "Feature_Importance":
            if self.dist.n_dist_param == 1:
                shap.plots.bar(shap_values, max_display=15 if X.shape[1] > 15 else X.shape[1])
            else:
                shap.plots.bar(shap_values[:, :, param_pos], max_display=15 if X.shape[1] > 15 else X.shape[1])

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

        shap.initjs()
        explainer = shap.TreeExplainer(self.booster)
        shap_values = explainer(X)

        expect_pos = list(self.dist.param_dict.keys()).index(expectile)

        if plot_type == "Partial_Dependence":
            shap.plots.scatter(shap_values[:, feature][:, expect_pos], color=shap_values[:, feature][:, expect_pos])
        elif plot_type == "Feature_Importance":
            shap.plots.bar(shap_values[:, :, expect_pos], max_display=15 if X.shape[1] > 15 else X.shape[1])
