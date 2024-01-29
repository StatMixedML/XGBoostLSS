""""""

import numpy as np
import pandas as pd

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

from xgboost.config import config_context

from xgboost.sklearn import _wrap_evaluation_matrices
from xgboost.compat import SKLEARN_INSTALLED, XGBRegressorBase
from xgboost._typing import ArrayLike, FeatureTypes
from xgboost.callback import TrainingCallback

from xgboostlss.model import XGBoostLSS
from xgboostlss.distributions.Gaussian import Gaussian
# Do not use class names on scikit-learn directly.  Re-define the classes on
# .compat to guarantee the behavior without scikit-learning installed.


class XGBModelLSS(XGBModel):
    def __init__(
        self,
        dist: Optional[int] = None,
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

        if dist is not None:
            self.dist = dist
        else:
            self.dist = Gaussian()

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

        self.start_values = None  # Starting values for distributional parameters
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
        quantiles: Optional[Union[List[float], float]] = None,
        n_samples: Optional[int] = None,
        validate_features: bool = True,
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
            y_pred = self._BoosterLSS.predict(
                data=test,
                pred_type=pred_type,
                quantiles=quantiles,
                n_samples=n_samples_,
                validate_features=validate_features,
            )

            if isinstance(y_pred, pd.DataFrame):
                return y_pred.values
            else:
                return y_pred


class XGBLSSRegressor(XGBModelLSS, XGBRegressorBase):
    """
    Implementation of the scikit-learn API for XGBoostLSS.

    Parameters
    ----------
    dist : Distribution
        DistributionClass object.  Default is Gaussian.

    max_depth :  Optional[int]
        Maximum tree depth for base learners.
    max_leaves :
        Maximum number of leaves; 0 indicates no limit.
    max_bin :
        If using histogram-based algorithm, maximum number of bins per feature
    grow_policy :
        Tree growing policy. 0: favor splitting at nodes closest to the node, i.e. grow
        depth-wise. 1: favor splitting at nodes with highest loss change.
    learning_rate : Optional[float]
        Boosting learning rate (xgb's "eta")
    verbosity : Optional[int]
        The degree of verbosity. Valid values are 0 (silent) - 3 (debug).

    booster: Optional[str]
        Specify which booster to use: `gbtree`, `gblinear` or `dart`.
    tree_method: Optional[str]
        Specify which tree method to use.  Default to auto.  If this parameter is set to
        default, XGBoost will choose the most conservative option available.  It's
        recommended to study this option from the parameters document :doc:`tree method
        </treemethod>`
    n_jobs : Optional[int]
        Number of parallel threads used to run xgboost.  When used with other
        Scikit-Learn algorithms like grid search, you may choose which algorithm to
        parallelize and balance the threads.  Creating thread contention will
        significantly slow down both algorithms.
    gamma : Optional[float]
        (min_split_loss) Minimum loss reduction required to make a further partition on a
        leaf node of the tree.
    min_child_weight : Optional[float]
        Minimum sum of instance weight(hessian) needed in a child.
    max_delta_step : Optional[float]
        Maximum delta step we allow each tree's weight estimation to be.
    subsample : Optional[float]
        Subsample ratio of the training instance.
    sampling_method :
        Sampling method. Used only by the GPU version of ``hist`` tree method.
          - ``uniform``: select random training instances uniformly.
          - ``gradient_based`` select random training instances with higher probability
            when the gradient and hessian are larger. (cf. CatBoost)
    colsample_bytree : Optional[float]
        Subsample ratio of columns when constructing each tree.
    colsample_bylevel : Optional[float]
        Subsample ratio of columns for each level.
    colsample_bynode : Optional[float]
        Subsample ratio of columns for each split.
    reg_alpha : Optional[float]
        L1 regularization term on weights (xgb's alpha).
    reg_lambda : Optional[float]
        L2 regularization term on weights (xgb's lambda).
    scale_pos_weight : Optional[float]
        Balancing of positive and negative weights.
    base_score : Optional[float]
        The initial prediction score of all instances, global bias.
    random_state : Optional[Union[numpy.random.RandomState, numpy.random.Generator, int]]
        Random number seed.

        .. note::

           Using gblinear booster with shotgun updater is nondeterministic as
           it uses Hogwild algorithm.

    missing : float, default np.nan
        Value in the data which needs to be present as a missing value.
    num_parallel_tree: Optional[int]
        Used for boosting random forest.
    monotone_constraints : Optional[Union[Dict[str, int], str]]
        Constraint of variable monotonicity.  See :doc:`tutorial </tutorials/monotonic>`
        for more information.
    interaction_constraints : Optional[Union[str, List[Tuple[str]]]]
        Constraints for interaction representing permitted interactions.  The
        constraints must be specified in the form of a nested list, e.g. ``[[0, 1], [2,
        3, 4]]``, where each inner list is a group of indices of features that are
        allowed to interact with each other.  See :doc:`tutorial
        </tutorials/feature_interaction_constraint>` for more information
    importance_type: Optional[str]
        The feature importance type for the feature_importances\\_ property:

        * For tree model, it's either "gain", "weight", "cover", "total_gain" or
          "total_cover".
        * For linear model, only "weight" is defined and it's the normalized coefficients
          without bias.

    device : Optional[str]

        .. versionadded:: 2.0.0

        Device ordinal, available options are `cpu`, `cuda`, and `gpu`.

    validate_parameters : Optional[bool]

        Give warnings for unknown parameter.

    enable_categorical : bool

        See the same parameter of :py:class:`DMatrix` for details.

    feature_types : Optional[FeatureTypes]

        .. versionadded:: 1.7.0

        Used for specifying feature types without constructing a dataframe. See
        :py:class:`DMatrix` for details.

    max_cat_to_onehot : Optional[int]

        .. versionadded:: 1.6.0

        .. note:: This parameter is experimental

        A threshold for deciding whether XGBoost should use one-hot encoding based split
        for categorical data.  When number of categories is lesser than the threshold
        then one-hot encoding is chosen, otherwise the categories will be partitioned
        into children nodes. Also, `enable_categorical` needs to be set to have
        categorical feature support. See :doc:`Categorical Data
        </tutorials/categorical>` and :ref:`cat-param` for details.

    max_cat_threshold : Optional[int]

        .. versionadded:: 1.7.0

        .. note:: This parameter is experimental

        Maximum number of categories considered for each split. Used only by
        partition-based splits for preventing over-fitting. Also, `enable_categorical`
        needs to be set to have categorical feature support. See :doc:`Categorical Data
        </tutorials/categorical>` and :ref:`cat-param` for details.

    multi_strategy : Optional[str]

        .. versionadded:: 2.0.0

        .. note:: This parameter is working-in-progress.

        The strategy used for training multi-target models, including multi-target
        regression and multi-class classification. See :doc:`/tutorials/multioutput` for
        more information.

        - ``one_output_per_tree``: One model for each target.
        - ``multi_output_tree``:  Use multi-target trees.

    eval_metric : Optional[Union[str, List[str], Callable]]

        .. versionadded:: 1.6.0

        Metric used for monitoring the training result and early stopping.  It can be a
        string or list of strings as names of predefined metric in XGBoost (See
        doc/parameter.rst), one of the metrics in :py:mod:`sklearn.metrics`, or any
        other user defined metric that looks like `sklearn.metrics`.

        If custom objective is also provided, then custom metric should implement the
        corresponding reverse link function.

        Unlike the `scoring` parameter commonly used in scikit-learn, when a callable
        object is provided, it's assumed to be a cost function and by default XGBoost
        will minimize the result during early stopping.

        For advanced usage on Early stopping like directly choosing to maximize instead
        of minimize, see :py:obj:`xgboost.callback.EarlyStopping`.

        See :doc:`/tutorials/custom_metric_obj` and :ref:`custom-obj-metric` for more
        information.

        .. note::

             This parameter replaces `eval_metric` in :py:meth:`fit` method.  The old
             one receives un-transformed prediction regardless of whether custom
             objective is being used.

        .. code-block:: python

            from sklearn.datasets import load_diabetes
            from sklearn.metrics import mean_absolute_error
            X, y = load_diabetes(return_X_y=True)
            reg = xgb.XGBRegressor(
                tree_method="hist",
                eval_metric=mean_absolute_error,
            )
            reg.fit(X, y, eval_set=[(X, y)])

    early_stopping_rounds : Optional[int]

        .. versionadded:: 1.6.0

        - Activates early stopping. Validation metric needs to improve at least once in
          every **early_stopping_rounds** round(s) to continue training.  Requires at
          least one item in **eval_set** in :py:meth:`fit`.

        - If early stopping occurs, the model will have two additional attributes:
          :py:attr:`best_score` and :py:attr:`best_iteration`. These are used by the
          :py:meth:`predict` and :py:meth:`apply` methods to determine the optimal
          number of trees during inference. If users want to access the full model
          (including trees built after early stopping), they can specify the
          `iteration_range` in these inference methods. In addition, other utilities
          like model plotting can also use the entire model.

        - If you prefer to discard the trees after `best_iteration`, consider using the
          callback function :py:class:`xgboost.callback.EarlyStopping`.

        - If there's more than one item in **eval_set**, the last entry will be used for
          early stopping.  If there's more than one metric in **eval_metric**, the last
          metric will be used for early stopping.

        .. note::

            This parameter replaces `early_stopping_rounds` in :py:meth:`fit` method.

    callbacks : Optional[List[TrainingCallback]]
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
                reg = xgboost.XGBRegressor(**params, callbacks=callbacks)
                reg.fit(X, y)

    kwargs : dict, optional
        Keyword arguments for XGBoost Booster object.  Full documentation of parameters
        can be found :doc:`here </parameter>`.
        Attempting to set a parameter via the constructor args and \\*\\*kwargs
        dict simultaneously will result in a TypeError.

        .. note:: \\*\\*kwargs unsupported by scikit-learn

            \\*\\*kwargs is unsupported by scikit-learn.  We do not guarantee
            that parameters passed via this argument will interact properly
            with scikit-learn.
    """
    def __init__(
        self, **kwargs: Any
    ) -> None:
        """Some docs here."""
        super().__init__(**kwargs)
