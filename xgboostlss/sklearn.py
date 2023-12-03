""""""

import numpy as np

from xgboost import XGBModel

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from xgboost.core import (
    Booster,
    DMatrix,
    Metric,
)


from xgboost.compat import SKLEARN_INSTALLED, XGBRegressorBase
from xgboost._typing import ArrayLike, FeatureTypes
from xgboost.callback import TrainingCallback

from xgboostlss.model import XGBoostLSS
from xgboost.config import config_context

from xgboost.sklearn import _wrap_evaluation_matrices
# Do not use class names on scikit-learn directly.  Re-define the classes on
# .compat to guarantee the behavior without scikit-learning installed.


class XGBModelLSS(XGBModel):
    def __init__(
        self,
        dist,
        max_depth: Optional[int] = None,
        max_leaves: Optional[int] = None,
        max_bin: Optional[int] = None,
        grow_policy: Optional[str] = None,
        learning_rate: Optional[float] = None,
        n_estimators: Optional[int] = None,
        verbosity: Optional[int] = None,
        objective: None = None,
        booster: Optional[str] = None,
        tree_method: Optional[str] = None,
        n_jobs: Optional[int] = None,
        gamma: Optional[float] = None,
        min_child_weight: Optional[float] = None,
        max_delta_step: Optional[float] = None,
        subsample: Optional[float] = None,
        sampling_method: Optional[str] = None,
        colsample_bytree: Optional[float] = None,
        colsample_bylevel: Optional[float] = None,
        colsample_bynode: Optional[float] = None,
        reg_alpha: Optional[float] = None,
        reg_lambda: Optional[float] = None,
        scale_pos_weight: Optional[float] = None,
        base_score: Optional[float] = 0,
        random_state: Optional[
            Union[np.random.RandomState, np.random.Generator, int]
        ] = None,
        missing: float = np.nan,
        num_parallel_tree: Optional[int] = None,
        monotone_constraints: Optional[Union[Dict[str, int], str]] = None,
        interaction_constraints: Optional[Union[str, Sequence[Sequence[str]]]] = None,
        importance_type: Optional[str] = None,
        device: Optional[str] = None,
        validate_parameters: Optional[bool] = None,
        enable_categorical: bool = False,
        feature_types: Optional[FeatureTypes] = None,
        max_cat_to_onehot: Optional[int] = None,
        max_cat_threshold: Optional[int] = None,
        multi_strategy: Optional[str] = None,
        eval_metric: Optional[Union[str, List[str], Callable]] = None,
        early_stopping_rounds: Optional[int] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        **kwargs: Any
    ) -> None:
        if not SKLEARN_INSTALLED:
            raise ImportError(
                "sklearn needs to be installed in order to use this module"
            )
        self.dist = dist

        self.n_estimators = n_estimators

        if objective is not None:
            raise ValueError("XGBoostLSS does not support objective function")
        else:
            self.objective = objective

        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bin = max_bin
        self.grow_policy = grow_policy
        self.learning_rate = learning_rate
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.sampling_method = sampling_method
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight

        if base_score != 0:
            raise ValueError("XGBoostLSS base_score must be 0.")
        else:
            self.base_score = 0

        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.device = device
        self.validate_parameters = validate_parameters
        self.enable_categorical = enable_categorical
        self.feature_types = feature_types
        self.max_cat_to_onehot = max_cat_to_onehot
        self.max_cat_threshold = max_cat_threshold
        self.multi_strategy = multi_strategy
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.callbacks = callbacks
        self.kwargs = kwargs

        self.start_values = None     # Starting values for distributional parameters
        self.multivariate_label_expand = False
        self.multivariate_eval_label_expand = False

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: Optional[ArrayLike] = None,
        base_margin: Optional[ArrayLike] = None,
        eval_set: Optional[Sequence[Tuple[ArrayLike, ArrayLike]]] = None,
        eval_metric: Optional[Union[str, Sequence[str], Metric]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: Optional[Union[bool, int]] = True,
        xgb_model: Optional[Union[Booster, str, "XGBModel"]] = None,
        sample_weight_eval_set: Optional[Sequence[ArrayLike]] = None,
        base_margin_eval_set: Optional[Sequence[ArrayLike]] = None,
        feature_weights: Optional[ArrayLike] = None,
        callbacks: Optional[Sequence[TrainingCallback]] = None,
    ) -> "XGBModel":
        with config_context(verbosity=self.verbosity):
            evals_result: TrainingCallback.EvalsLog = {}

            train_dmatrix, evals = _wrap_evaluation_matrices(
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
                create_dmatrix=self._create_dmatrix,
                enable_categorical=self.enable_categorical,
                feature_types=self.feature_types,
            )

            params = self.get_xgb_params()
            params.pop("dist", None)

            params_adj = {
                "num_target": self.dist.n_dist_param,
                "disable_default_eval_metric": True
            }

            params.update(params_adj)
            evals = None if not bool(evals) else evals

            self._BoosterLSS = XGBoostLSS(self.dist)
            self._BoosterLSS.train(
                params,
                train_dmatrix,
                self.get_num_boosting_rounds(),
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                evals_result=evals_result,
                verbose_eval=verbose,
                # xgb_model=self._Booster.booster,
                callbacks=callbacks,
            )

            self._Booster = self._BoosterLSS.booster

    def predict(
        self,
        X: ArrayLike,
        pred_type: str = "parameters",
        quantiles: Optional[List[float]] = None,
        n_samples: Optional[int] = None,
        # validate_features: bool = True,
        base_margin: Optional[ArrayLike] = None,
    ) -> ArrayLike:

        with config_context(verbosity=self.verbosity):
            n_samples_ = n_samples or 1000

            test = DMatrix(
                X,
                base_margin=base_margin,
                missing=self.missing,
                nthread=self.n_jobs,
                feature_types=self.feature_types,
                enable_categorical=self.enable_categorical,
            )
            return self._BoosterLSS.predict(
                data=test,
                pred_type=pred_type,
                quantiles=quantiles,
                n_samples=n_samples_,
            )


class XGBLSSRegressor(XGBModelLSS, XGBRegressorBase):
    # pylint: disable=missing-docstring
    def __init__(
        self, dist, **kwargs: Any
    ) -> None:
        """Some docs here."""
        super().__init__(dist, **kwargs)
