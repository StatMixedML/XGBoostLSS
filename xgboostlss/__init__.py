"""XGBoostLSS - An extension of XGBoost to probabilistic forecasting"""

# Import sklearn-compatible interface for simplified usage
try:
    from .sklearn_compat import XGBoostLSSRegressor
    __all__ = ['XGBoostLSSRegressor']
except ImportError:
    # Graceful fallback if dependencies are missing
    __all__ = []