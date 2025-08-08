"""
Sklearn-compatible interface for XGBoostLSS

This module provides a simplified sklearn-compatible interface that addresses
the key issues identified in the sktime integration:
1. Simple fit/predict workflow
2. Automatic distribution detection and defaults
3. Optional dependency handling
4. Better user experience
"""

import warnings
from typing import Optional, Union, Dict, Any, Literal
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost is required for XGBoostLSS functionality")

# Import distributions with graceful fallbacks
DISTRIBUTIONS_AVAILABLE = False
DISTRIBUTION_MAP = {}
Gaussian = Gamma = Beta = StudentT = None

try:
    from .distributions.Gaussian import Gaussian
    from .distributions.Gamma import Gamma  
    from .distributions.Beta import Beta
    from .distributions.StudentT import StudentT
    DISTRIBUTIONS_AVAILABLE = True
    
    # Distribution mapping for string-based selection
    DISTRIBUTION_MAP = {
        'gaussian': Gaussian,
        'normal': Gaussian,
        'gamma': Gamma,
        'beta': Beta,
        'studentt': StudentT,
        't': StudentT
    }
except ImportError as e:
    warnings.warn(f"Distribution modules not available: {e}. Install torch dependencies: pip install xgboostlss[torch]")

# Optional dependencies
MODEL_AVAILABLE = False
XGBoostLSS = None

try:
    from .model import XGBoostLSS
    MODEL_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"XGBoostLSS model not available: {e}")


class XGBoostLSSRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible XGBoostLSS regressor.
    
    A simplified interface for XGBoostLSS that provides:
    - Standard sklearn fit/predict workflow
    - Automatic distribution detection
    - Sensible defaults
    - Optional dependency handling
    
    Parameters
    ----------
    distribution : str or Distribution, default='auto'
        Distribution to use. Options:
        - 'auto': Automatically detect based on target characteristics
        - 'gaussian'/'normal': Gaussian distribution
        - 'gamma': Gamma distribution (positive values)
        - 'beta': Beta distribution (values in [0,1])
        - 'studentt'/'t': Student's t-distribution
        Or pass a Distribution instance directly.
        
    n_estimators : int, default=100
        Number of boosting rounds.
        
    learning_rate : float, default=0.3
        Step size shrinkage used in update to prevents overfitting.
        
    max_depth : int, default=6
        Maximum depth of trees.
        
    random_state : int, optional
        Random seed for reproducibility.
        
    **kwargs
        Additional XGBoost parameters.
        
    Attributes
    ----------
    distribution_ : Distribution
        The fitted distribution object.
        
    feature_importances_ : array-like of shape (n_features,)
        Feature importances.
        
    Examples
    --------
    >>> from xgboostlss.sklearn_compat import XGBoostLSSRegressor
    >>> import numpy as np
    >>> 
    >>> # Simple usage with auto distribution detection
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randn(100)
    >>> 
    >>> model = XGBoostLSSRegressor()
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> 
    >>> # With specific distribution
    >>> model = XGBoostLSSRegressor(distribution='gamma', n_estimators=200)
    >>> model.fit(X, np.abs(y))  # Gamma requires positive values
    >>> y_pred = model.predict(X)
    """
    
    def __init__(
        self,
        distribution: Union[str, Any] = 'auto',
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 6,
        random_state: Optional[int] = None,
        **kwargs
    ):
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is required but not installed. Install with: pip install xgboost")
        
        if not MODEL_AVAILABLE:
            raise ImportError("XGBoostLSS model not available. Please check installation.")
            
        self.distribution = distribution
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Will be set during fit
        self.distribution_ = None
        self._model = None
        self.feature_importances_ = None
        
    def _detect_distribution(self, y: np.ndarray) -> str:
        """
        Automatically detect appropriate distribution based on target characteristics.
        
        Parameters
        ----------
        y : array-like
            Target values
            
        Returns
        -------
        str
            Recommended distribution name
        """
        y = np.asarray(y)
        
        # Check for values in [0, 1] - suggest Beta
        if np.all((y >= 0) & (y <= 1)) and not np.all((y == 0) | (y == 1)):
            return 'beta'
            
        # Check for positive values - suggest Gamma
        elif np.all(y > 0):
            # Check skewness to decide between Gamma and Gaussian
            skewness = np.abs(np.mean(((y - np.mean(y)) / np.std(y)) ** 3))
            if skewness > 1.0:  # Highly skewed
                return 'gamma'
            else:
                return 'gaussian'
                
        # Check for heavy tails - suggest Student's t
        elif len(y) > 20:  # Need sufficient data for kurtosis
            kurtosis = np.mean(((y - np.mean(y)) / np.std(y)) ** 4) - 3
            if kurtosis > 2.0:  # Heavy tails
                return 'studentt'
                
        # Default to Gaussian
        return 'gaussian'
    
    def _get_distribution(self, distribution_spec: Union[str, Any], y: np.ndarray = None):
        """
        Get distribution instance from specification.
        
        Parameters
        ----------
        distribution_spec : str or Distribution
            Distribution specification
        y : array-like, optional
            Target values for auto-detection
            
        Returns
        -------
        Distribution
            Distribution instance
        """
        if isinstance(distribution_spec, str):
            if distribution_spec == 'auto':
                if y is None:
                    raise ValueError("Target values required for auto distribution detection")
                distribution_spec = self._detect_distribution(y)
                
            if distribution_spec not in DISTRIBUTION_MAP:
                available = list(DISTRIBUTION_MAP.keys())
                raise ValueError(f"Unknown distribution '{distribution_spec}'. Available: {available}")
                
            # Use sensible defaults for each distribution
            if distribution_spec in ['gaussian', 'normal']:
                return Gaussian(stabilization="MAD", response_fn="softplus", loss_fn="nll")
            elif distribution_spec == 'gamma':
                return Gamma(stabilization="MAD", response_fn="softplus", loss_fn="nll")
            elif distribution_spec == 'beta':
                return Beta(stabilization="MAD", response_fn="softplus", loss_fn="nll")
            elif distribution_spec in ['studentt', 't']:
                return StudentT(stabilization="MAD", response_fn="softplus", loss_fn="nll")
        else:
            # Assume it's already a distribution instance
            return distribution_spec
            
    def fit(self, X, y, **fit_params):
        """
        Fit the XGBoostLSS model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **fit_params
            Additional fitting parameters passed to XGBoost.
            
        Returns
        -------
        self : XGBoostLSSRegressor
            Returns self.
        """
        # Validate inputs
        X, y = check_X_y(X, y)
        
        # Get distribution
        self.distribution_ = self._get_distribution(self.distribution, y)
        
        # Create XGBoostLSS model
        self._model = XGBoostLSS(self.distribution_)
        
        # Prepare XGBoost parameters
        params = {
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            **self.kwargs
        }
        
        if self.random_state is not None:
            params['random_state'] = self.random_state
            
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Train model
        self._model.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False,
            **fit_params
        )
        
        # Set feature importances if available
        if hasattr(self._model.booster, 'get_score'):
            importance_dict = self._model.booster.get_score(importance_type='gain')
            n_features = X.shape[1]
            self.feature_importances_ = np.zeros(n_features)
            
            for i in range(n_features):
                feature_name = f'f{i}'
                if feature_name in importance_dict:
                    self.feature_importances_[i] = importance_dict[feature_name]
                    
        return self
        
    def predict(self, X, return_type: Literal['mean', 'samples', 'quantiles'] = 'mean', **predict_params):
        """
        Predict using the fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        return_type : str, default='mean'
            Type of prediction to return:
            - 'mean': Point predictions (mean of distribution)
            - 'samples': Sample from predictive distribution
            - 'quantiles': Return quantiles
        **predict_params
            Additional parameters for prediction.
            
        Returns
        -------
        array-like
            Predictions.
        """
        # Check if fitted
        if self._model is None:
            raise ValueError("Model must be fitted before prediction")
            
        # Validate input
        X = check_array(X)
        
        # Create DMatrix
        dtest = xgb.DMatrix(X)
        
        # Get predictions based on type
        if return_type == 'mean':
            # Return mean predictions
            pred_params = self._model.predict(dtest, pred_type="parameters")
            if hasattr(self.distribution_, 'mean'):
                return self.distribution_.mean(pred_params)
            else:
                # Fallback: return first parameter (often the mean/location)
                return pred_params.iloc[:, 0].values
                
        elif return_type == 'samples':
            n_samples = predict_params.get('n_samples', 1)
            samples = self._model.predict(dtest, pred_type="samples", n_samples=n_samples)
            return samples.values
            
        elif return_type == 'quantiles':
            quantiles = predict_params.get('quantiles', [0.1, 0.5, 0.9])
            pred_quantiles = self._model.predict(dtest, pred_type="quantiles", quantiles=quantiles)
            return pred_quantiles.values
            
        else:
            raise ValueError(f"Unknown return_type '{return_type}'. Options: 'mean', 'samples', 'quantiles'")
            
    def predict_proba(self, X, **predict_params):
        """
        Return samples from the predictive distribution.
        
        This is an alias for predict(X, return_type='samples') to maintain
        sklearn probabilistic interface conventions.
        
        Parameters
        ----------
        X : array-like
            Samples.
        **predict_params
            Additional parameters passed to predict.
            
        Returns
        -------
        array-like
            Samples from predictive distribution.
        """
        return self.predict(X, return_type='samples', **predict_params)
        
    def predict_quantiles(self, X, quantiles=None, **predict_params):
        """
        Predict quantiles.
        
        Parameters
        ----------
        X : array-like
            Samples.
        quantiles : list, optional
            Quantiles to predict. Default is [0.1, 0.5, 0.9].
        **predict_params
            Additional parameters.
            
        Returns
        -------
        array-like
            Predicted quantiles.
        """
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        return self.predict(X, return_type='quantiles', quantiles=quantiles, **predict_params)