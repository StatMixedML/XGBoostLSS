"""Implementation of the scikit-learn API for XGB accelerated failure time regression."""
from typing import Any, Optional, Sequence, Tuple, Union

import xgboost
import xgboost.callback
import xgboost.compat
from xgboost._typing import ArrayLike
from xgboostlss.model import XGBoostLSS
from xgboostlss import distributions

_DISTRIBUTIONS = {
    "Beta": distributions.Beta.Beta,
    "Cauchy": distributions.Cauchy.Cauchy,
    "Dirichlet": distributions.Dirichlet.Dirichlet,
    "Expectile": distributions.Expectile.Expectile,
    "Gamma": distributions.Gamma.Gamma,
    "Gaussian": distributions.Gaussian.Gaussian,
    "Gumbel": distributions.Gumbel.Gumbel,
    "Laplace": distributions.Laplace.Laplace,
    "LogNormal": distributions.LogNormal.LogNormal,
    "MVN": distributions.MVN.MVN,
    "MVN_LoRa": distributions.MVN_LoRa.MVN_LoRa,
    "MVT": distributions.MVT.MVT,
    "NegativeBinomial": distributions.NegativeBinomial.NegativeBinomial,
    "Poisson": distributions.Poisson.Poisson,
    "SplineFlow": distributions.SplineFlow.SplineFlow,
    "StudentT": distributions.StudentT.StudentT,
    "Weibull": distributions.Weibull.Weibull,
    "ZABeta": distributions.ZABeta.ZABeta,
    "ZAGamma": distributions.ZAGamma.ZAGamma,
    "ZALN": distributions.ZALN.ZALN,
    "ZINB": distributions.ZINB.ZINB,
    "ZIPoisson": distributions.ZIPoisson.ZIPoisson,
}

class XGBLSSRegressor(xgboost.sklearn.XGBModel, xgboost.compat.XGBRegressorBase):
    """Implementation of the scikit-learn API for XGBoostLSS models."""

    def __init__(
        self,
        *,
        dist: str,
        dist_params: Optional[dict[str, str | float | int | bool]]=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.dist = dist
        self.dist_params = dist_params

    def get_params(self, **kwargs: Any) -> dict[str, Any]:  # pylint: disable=arguments-differ
        """Get the parameters of this estimator.

        Dev note: Consider overriding get_xgb_params instead if this ever turns out to be too general.
        """
        params = super().get_params(**kwargs)
        # 'dist' and 'dist_params' is specific to this class; not an XGBRegressor parameter
        del params["dist"]
        del params["dist_params"]
        return params

    # The fit function is largely copying XGBModel.fit, but with our own create_dmatrix function
    # to add lower bound and upper bound labels.
    # See https://github.com/dmlc/xgboost/blob/release_1.6.0/python-package/xgboost/sklearn.py#L862
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: Optional[ArrayLike] = None,
        base_margin: Optional[ArrayLike] = None,
        eval_set: Optional[Sequence[Tuple[ArrayLike, ArrayLike]]] = None,
        eval_metric: Optional[Union[str, Sequence[str], xgboost.core.Metric]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: Optional[bool] = True,
        xgb_model: Optional[Union[xgboost.Booster, str, xgboost.XGBModel]] = None,
        sample_weight_eval_set: Optional[list[ArrayLike]] = None,
        base_margin_eval_set: Optional[Sequence[ArrayLike]] = None,
        feature_weights: Optional[ArrayLike] = None,
        callbacks: Optional[Sequence[xgboost.callback.TrainingCallback]] = None,
    ) -> "XGBLSSRegressor":
        """Fit a XGBLSS model.

        Note that calling ``fit()`` multiple times will cause the model object to be
        re-fit from scratch. To resume training from a previous checkpoint, explicitly
        pass ``xgb_model`` argument.

        Parameters
        ----------
        X :
            Feature matrix
        y :
            Labels. This should be a 2-dimensional array of shape (n_samples, 2),
            where the first column is the lower bound of the label,
            and the second column is the upper bound of the label.
        sample_weight :
            instance weights
        base_margin :
            global bias for each instance.
        eval_set :
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.

        eval_metric : str, list of str, or callable, optional
            .. deprecated:: 1.6.0
                Use `eval_metric` in :py:meth:`__init__` or :py:meth:`set_params` instead.

        early_stopping_rounds : int
            .. deprecated:: 1.6.0
                Use `early_stopping_rounds` in :py:meth:`__init__` or
                :py:meth:`set_params` instead.
        verbose :
            If `verbose` and an evaluation set is used, writes the evaluation metric
            measured on the validation set to stderr.
        xgb_model :
            file name of stored XGBoost model or 'Booster' instance XGBoost model to be
            loaded before training (allows training continuation).
        sample_weight_eval_set :
            A list of the form [L_1, L_2, ..., L_n], where each L_i is an array like
            object storing instance weights for the i-th validation set.
        base_margin_eval_set :
            A list of the form [M_1, M_2, ..., M_n], where each M_i is an array like
            object storing base margin for the i-th validation set.
        feature_weights :
            Weight for each feature, defines the probability of each feature being
            selected when colsample is being used.  All values must be greater than 0,
            otherwise a `ValueError` is thrown.

        callbacks :
            .. deprecated:: 1.6.0
                Use `callbacks` in :py:meth:`__init__` or :py:meth:`set_params` instead.
        """
        evals_result: xgboost.callback.TrainingCallback.EvalsLog = {}
        train_dmatrix, evals = xgboost.sklearn._wrap_evaluation_matrices(
            missing=self.missing,
            X=X,
            y=y,
            group=None,
            qid=None,
            sample_weight=sample_weight,
            base_margin=base_margin,
            feature_weights=feature_weights,
            eval_set=eval_set,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin_eval_set=base_margin_eval_set,
            eval_group=None,
            eval_qid=None,
            create_dmatrix=lambda **kwargs: xgboost.DMatrix(nthread=self.n_jobs, **kwargs),
            enable_categorical=self.enable_categorical,
        )
        params = self.get_xgb_params()

        # if callable(self.objective):
        #     obj: Optional[
        #         Callable[[np.ndarray, xgboost.DMatrix], Tuple[np.ndarray, np.ndarray]]
        #     ] = xgboost.sklearn._objective_decorator(self.objective)
        # else:
        #     obj = None

        model, metric, params, early_stopping_rounds, callbacks = self._configure_fit(
            xgb_model, eval_metric, params, early_stopping_rounds, callbacks
        )
        del metric
        
        self._estimator = XGBoostLSS(dist=_DISTRIBUTIONS[self.dist](**(self.dist_params or {})))

        self._Booster = self._estimator.train(
            params,
            train_dmatrix,
            self.get_num_boosting_rounds(),
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=verbose,
            xgb_model=model,
            callbacks=callbacks,
        )

        self._set_evaluation_result(evals_result)
        return self

    def predict(
        self,
        X: ArrayLike,
    ) -> ArrayLike:
        """Predict with `X`. Returns distribution parameters.

        Parameters
        ----------
        X :
            Data to predict with.

        Returns
        -------
        prediction

        """
        test = xgboost.DMatrix(
            X,
            missing=self.missing,
            nthread=self.n_jobs,
            enable_categorical=self.enable_categorical
        )
        preds = self._estimator.predict(
            data=test,
            pred_type="parameters"
        )

        return preds.to_numpy()
