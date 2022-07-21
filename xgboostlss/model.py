import optuna
import shap
from optuna.samplers import TPESampler
import numpy as np
import xgboost as xgb
import pandas as pd


class xgboostlss:
    """
    XGBoostLSS model class

    """

    def train(params, dtrain, dist, num_boost_round=10, evals=(),
              maximize=False, early_stopping_rounds=None, evals_result=None,
              verbose_eval=True, xgb_model=None, callbacks=None):
        """Train a xgboostlss model with given parameters.

        Parameters
        ----------
        params : dict
            Booster params.
        dtrain : DMatrix
            Data to be trained.
        dist: xgboostlss.distributions class
            Specifies distributional assumption.
        num_boost_round: int
            Number of boosting iterations.
        evals: list of pairs (DMatrix, string)
            List of validation sets for which metrics will evaluated during training.
            Validation metrics will help us track the performance of the model.
        maximize : bool
            Whether to maximize custom_metric.
        early_stopping_rounds: int
            Activates early stopping. Validation metric needs to improve at least once in
            every **early_stopping_rounds** round(s) to continue training.
            Requires at least one item in **evals**.
            The method returns the model from the last iteration (not the best one).  Use
            custom callback or model slicing if the best model is desired.
            If there's more than one item in **evals**, the last entry will be used for early
            stopping.
            If there's more than one metric in the **eval_metric** parameter given in
            **params**, the last metric will be used for early stopping.
            If early stopping occurs, the model will have three additional fields:
            ``bst.best_score``, ``bst.best_iteration``.
        evals_result: dict
            This dictionary stores the evaluation results of all the items in watchlist.
            Example: with a watchlist containing
            ``[(dtest,'eval'), (dtrain,'train')]`` and
            a parameter containing ``('eval_metric': 'logloss')``,
            the **evals_result** returns
            .. code-block:: python
                {'train': {'logloss': ['0.48253', '0.35953']},
                 'eval': {'logloss': ['0.480385', '0.357756']}}
        verbose_eval : bool or int
            Requires at least one item in **evals**.
            If **verbose_eval** is True then the evaluation metric on the validation set is
            printed at each boosting stage.
            If **verbose_eval** is an integer then the evaluation metric on the validation set
            is printed at every given **verbose_eval** boosting stage. The last boosting stage
            / the boosting stage found by using **early_stopping_rounds** is also printed.
            Example: with ``verbose_eval=4`` and at least one item in **evals**, an evaluation metric
            is printed every 4 boosting stages, instead of every boosting stage.
        xgb_model : file name of stored xgb model or 'Booster' instance
            Xgb model to be loaded before training (allows training continuation).
        callbacks : list of callback functions
            List of callback functions that are applied at end of each iteration.
            It is possible to use predefined callbacks by using
            :ref:`Callback API <callback_api>`.
            Example:
            .. code-block:: python
                [xgb.callback.LearningRateScheduler(custom_rates)]
        """

        params_adj = {"objective": None,
                      "base_score": 0,
                      "num_class": dist.n_dist_param(),
                      "disable_default_eval_metric": True}

        params.update(params_adj)

        # Set base_margin as starting point for each distributional parameter. Requires base_score=0 in parameters.
        dist.start_values = dist.initialize(dtrain.get_label())
        base_margin = (np.ones(shape=(dtrain.num_row(), 1))) * dist.start_values
        dtrain.set_base_margin(base_margin.flatten())

        bstLSS_train = xgb.train(params,
                                 dtrain,
                                 num_boost_round=num_boost_round,
                                 evals=evals,
                                 obj=dist.Dist_Objective,
                                 custom_metric=dist.Dist_Metric,
                                 xgb_model=xgb_model,
                                 callbacks=callbacks,
                                 verbose_eval=verbose_eval,
                                 evals_result=evals_result,
                                 maximize=False,
                                 early_stopping_rounds=early_stopping_rounds)
        return bstLSS_train

    def cv(params, dtrain, dist, num_boost_round=10, nfold=3, stratified=False, folds=None,
           maximize=False, early_stopping_rounds=None, fpreproc=None, as_pandas=True,
           verbose_eval=None, show_stdv=True, seed=123, callbacks=None, shuffle=True):
        """Function to cross-validate a xgboostlss model with given parameters.

        Parameters
        ----------
        params : dict
            Booster params.
        dtrain : DMatrix
            Data to be trained.
        dist: xgboostlss.distributions class
            Specifies distributional assumption.
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
        maximize : bool
            Whether to maximize custom_metric.
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
        callbacks : list of callback functions
            List of callback functions that are applied at end of each iteration.
            It is possible to use predefined callbacks by using
            :ref:`Callback API <callback_api>`.
            Example:
            .. code-block:: python
                [xgb.callback.LearningRateScheduler(custom_rates)]
        shuffle : bool
            Shuffle data before creating folds.
        Returns
        -------
        evaluation history : list(string)
        """

        params_adj = {"objective": None,
                      "base_score": 0,
                      "num_class": dist.n_dist_param(),
                      "disable_default_eval_metric": True}

        params.update(params_adj)

        # Set base_margin as starting point for each distributional parameter. Requires base_score=0 in parameters.
        dist.start_values = dist.initialize(dtrain.get_label())
        base_margin = (np.ones(shape=(dtrain.num_row(), 1))) * dist.start_values
        dtrain.set_base_margin(base_margin.flatten())

        bstLSS_cv = xgb.cv(params,
                           dtrain,
                           num_boost_round=num_boost_round,
                           nfold=nfold,
                           stratified=stratified,
                           folds=folds,
                           obj=dist.Dist_Objective,
                           custom_metric=dist.Dist_Metric,
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

    def hyper_opt(params, dtrain, dist, num_boost_round=500, nfold=10, early_stopping_rounds=20,
                  max_minutes=10, n_trials=None, study_name="XGBoostLSS-HyperOpt", silence=False):
        """Function to tune hyperparameters using optuna.

        Parameters
        ----------
        params : dict
            Booster params in the form of "params_name": [min_val, max_val].
        dtrain : DMatrix
            Data to be trained.
        dist: xgboostlss.distributions class
            Specifies distributional assumption.
        num_boost_round : int
            Number of boosting iterations.
        nfold : int
            Number of folds in CV.
        early_stopping_rounds: int
            Activates early stopping. Cross-Validation metric (average of validation
            metric computed over CV folds) needs to improve at least once in
            every **early_stopping_rounds** round(s) to continue training.
            The last entry in the evaluation history will represent the best iteration.
            If there's more than one metric in the **eval_metric** parameter given in
            **params**, the last metric will be used for early stopping.
        max_minutes : int
            Time budget in minutes, i.e., stop study after the given number of minutes.
        n_trials : int
            The number of trials. If this argument is set to None, there is no limitation on the number of trials.
        study_name : str
            Name of the hyperparameter study.
        silence : bool
            Controls the verbosity of the trail, i.e., user can silence the outputs of the trail.

        Returns
        -------
        opt_params : Dict() with optimal parameters.
        """

        def objective(trial):

            hyper_params = {
                "booster": "gbtree",
                "eta": trial.suggest_loguniform("eta", params["eta"][0], params["eta"][1]),
                "max_depth": trial.suggest_int("max_depth", params["max_depth"][0], params["max_depth"][1]),
                "gamma": trial.suggest_loguniform("gamma", params["gamma"][0], params["gamma"][1]),
                "subsample": trial.suggest_loguniform("subsample", params["subsample"][0], params["subsample"][1]),
                "colsample_bytree": trial.suggest_loguniform("colsample_bytree", params["colsample_bytree"][0],
                                                             params["colsample_bytree"][1]),
                "min_child_weight": trial.suggest_int("min_child_weight", params["min_child_weight"][0],
                                                      params["min_child_weight"][1])
            }

            # Add pruning
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-NegLogLikelihood")

            xgblss_param_tuning = xgboostlss.cv(hyper_params,
                                                dtrain=dtrain,
                                                dist=dist,
                                                num_boost_round=num_boost_round,
                                                nfold=nfold,
                                                early_stopping_rounds=early_stopping_rounds,
                                                callbacks=[pruning_callback],
                                                seed=123,
                                                verbose_eval=False,
                                                maximize=False)

            # Add opt_rounds as a trial attribute, accessible via study.trials_dataframe(). # https://github.com/optuna/optuna/issues/1169
            opt_rounds = xgblss_param_tuning["test-NegLogLikelihood-mean"].idxmin() + 1
            trial.set_user_attr("opt_round", int(opt_rounds))

            # Extract the best score
            best_score = np.min(xgblss_param_tuning["test-NegLogLikelihood-mean"])

            return best_score

        if silence:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = TPESampler(seed=123)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize", study_name=study_name)
        study.optimize(objective, n_trials=n_trials, timeout=60 * max_minutes, show_progress_bar=True)

        print("Hyper-Parameter Optimization successfully finished.")
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        opt_param = study.best_trial

        # Add optimal stopping round
        opt_param.params["opt_rounds"] = study.trials_dataframe()["user_attrs_opt_round"][
            study.trials_dataframe()["value"].idxmin()]
        opt_param.params["opt_rounds"] = int(opt_param.params["opt_rounds"])

        print("  Value: {}".format(opt_param.value))
        print("  Params: ")
        for key, value in opt_param.params.items():
            print("    {}: {}".format(key, value))

        return opt_param.params

    def predict(booster: xgb.Booster, dtest: xgb.DMatrix, dist, pred_type: str,
                n_samples: int = 1000, quantiles: list = None, seed: int = 123):
        '''A customized xgboostlss prediction function.

        booster: xgb.Booster
            Trained XGBoostLSS-Model
        X: xgb.DMatrix
            Test Data
        dist: DistributionType
            Specifies the distributional assumption.
        pred_type: str
            Specifies what is to be predicted:
                "response" draws n_samples from the predicted response distribution.
                "quantile" calculates the quantiles from the predicted response distribution.
                "parameters" returns the predicted distributional parameters.
                "expectiles" returns the predicted expectiles.
        n_samples: int
            If pred_type="response" specifies how many samples are drawn from the predicted response distribution.
        quantiles: list
            If pred_type="quantiles" calculates the quantiles from the predicted response distribution.
        seed: int
            If pred_type="response" specifies the seed for drawing samples from the predicted response distribution.

        '''

        dict_param = dist.param_dict()

        # Set base_margin as starting point for each distributional parameter. Requires base_score=0 in parameters.
        base_margin = (np.ones(shape=(dtest.num_row(), 1))) * dist.start_values
        dtest.set_base_margin(base_margin.flatten())

        predt = booster.predict(dtest, output_margin=True)

        dist_params_predts = []

        for i, (dist_param, response_fun) in enumerate(dict_param.items()):
            dist_params_predts.append(response_fun(predt[:, i]))

        dist_params_df = pd.DataFrame(dist_params_predts).T
        dist_params_df.columns = dict_param.keys()

        if pred_type == "parameters":
            return dist_params_df

        elif pred_type == "expectiles":
            return dist_params_df

        elif pred_type == "response":
            pred_resp_df = dist.pred_dist_rvs(
                pred_params=dist_params_df,
                n_samples=n_samples,
                seed=seed,
            )

            pred_resp_df.columns = [str("y_pred_sample_") + str(i) for i in range(pred_resp_df.shape[1])]
            return pred_resp_df

        elif pred_type == "quantiles":
            if quantiles is None:
                quantiles = [0.1, 0.5, 0.9]
            pred_quant_df = dist.pred_dist_quantile(
                quantiles=quantiles,
                pred_params=dist_params_df,
            )

            pred_quant_df.columns = [str("quant_") + str(quantiles[i]) for i in range(len(quantiles))]
            return pred_quant_df

    def plot(booster: xgb.Booster, X: pd.DataFrame, feature: str = "x", parameter: str = "location",
             plot_type: str = "Partial_Dependence"):
        '''A customized xgboostlss plotting function.

        booster: xgb.Booster
            Trained XGBoostLSS-Model
        X: pd.DataFrame
            Train/Test Data
        feature: str
            Specifies which feature to use for plotting Partial_Dependence plot.
        parameter: str
            Specifies which distributional parameter to plot. Valid parameters are "location", "scale", "nu", "tau".
        plot_type: str
            Specifies which SHapley-plot to visualize. Currently "Partial_Dependence" and "Feature_Importance" are supported.

        '''

        shap.initjs()
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer(X)

        if parameter == "location":
            param_pos = 0
        if parameter == "scale":
            param_pos = 1
        if parameter == "nu":
            param_pos = 2
        if parameter == "tau":
            param_pos = 3

        if plot_type == "Partial_Dependence":
            shap.plots.scatter(shap_values[:, feature][:, param_pos], color=shap_values[:, :, param_pos])
        elif plot_type == "Feature_Importance":
            shap.plots.bar(shap_values[:, :, param_pos], max_display=15 if X.shape[1] > 15 else X.shape[1])

    def expectile_plot(booster: xgb.Booster, X: pd.DataFrame, dist, feature: str = "x", expectile: str = "0.05",
                       plot_type: str = "Partial_Dependence"):
        '''A customized xgboostlss plotting function.

        booster: xgb.Booster
            Trained XGBoostLSS-Model
        X: pd.DataFrame
            Train/Test Data
        dist: xgboostlss.distributions class
            Specifies distributional assumption
        feature: str
            Specifies which feature to use for plotting Partial_Dependence plot.
        expectile: str
            Specifies which expectile to plot.
        plot_type: str
            Specifies which SHapley-plot to visualize. Currently "Partial_Dependence" and "Feature_Importance" are supported.

        '''

        shap.initjs()
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer(X)

        expect_pos = dist.expectiles.index(float(expectile))

        if plot_type == "Partial_Dependence":
            shap.plots.scatter(shap_values[:, feature][:, expect_pos], color=shap_values[:, :, expect_pos])
        elif plot_type == "Feature_Importance":
            shap.plots.bar(shap_values[:, :, expect_pos], max_display=15 if X.shape[1] > 15 else X.shape[1])
