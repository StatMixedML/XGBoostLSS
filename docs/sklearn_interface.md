# Sklearn-Compatible Interface

This document describes the new simplified sklearn-compatible interface for XGBoostLSS, which addresses the key issues identified in the sktime integration.

## Overview

The sklearn-compatible interface provides:

- **Simplified workflow**: Standard `fit()`/`predict()` methods instead of complex multi-step process
- **Automatic distribution detection**: Intelligent selection based on target characteristics  
- **sklearn ecosystem compatibility**: Works with pipelines, cross-validation, model selection
- **Python 3.12 support**: Updated dependency management and optional dependencies
- **Better user experience**: Sensible defaults and intuitive API

## Quick Start

### Basic Usage

```python
from xgboostlss import XGBoostLSSRegressor
import numpy as np

# Generate sample data
X = np.random.randn(1000, 5)
y = np.random.randn(1000)

# Simple 2-step workflow
model = XGBoostLSSRegressor()  # Auto-detects distribution
model.fit(X, y)
y_pred = model.predict(X)
```

### With Specific Distribution

```python
# Specify distribution explicitly
model = XGBoostLSSRegressor(
    distribution='gamma',  # For positive-valued targets
    n_estimators=200,
    learning_rate=0.1
)
model.fit(X, np.abs(y))  # Gamma requires positive values
```

## Automatic Distribution Detection

The interface can automatically select appropriate distributions based on target characteristics:

| Data Characteristics | Detected Distribution | Use Case |
|---------------------|----------------------|----------|
| Values in [0, 1] | Beta | Proportions, probabilities |
| Positive values, skewed | Gamma | Count data, waiting times |
| Heavy tails (high kurtosis) | Student's t | Robust to outliers |
| General real values | Gaussian | Default fallback |

### Example

```python
from xgboostlss import XGBoostLSSRegressor

model = XGBoostLSSRegressor()

# Beta data (values in [0,1])
y_beta = np.random.beta(2, 2, 1000)
model.fit(X, y_beta)  # Automatically detects 'beta' distribution

# Gamma data (positive, skewed)  
y_gamma = np.random.gamma(2, 2, 1000)
model.fit(X, y_gamma)  # Automatically detects 'gamma' distribution
```

## sklearn Ecosystem Integration

### Pipeline Compatibility

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', XGBoostLSSRegressor(distribution='gaussian'))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

model = XGBoostLSSRegressor(n_estimators=100)
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.3, 0.5],
    'max_depth': [3, 6, 9]
}

grid_search = GridSearchCV(
    XGBoostLSSRegressor(), 
    param_grid, 
    cv=3, 
    scoring='neg_mean_squared_error'
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

## Prediction Types

The interface supports multiple prediction types for uncertainty quantification:

### Point Predictions (Mean)

```python
# Default: returns mean of predictive distribution
y_mean = model.predict(X_test)
```

### Quantile Predictions

```python
# Get prediction intervals
y_quantiles = model.predict(X_test, return_type='quantiles')
# Returns 10th, 50th (median), 90th percentiles by default

# Custom quantiles
y_custom = model.predict(
    X_test, 
    return_type='quantiles',
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
)
```

### Sampling from Predictive Distribution

```python
# Sample from posterior predictive distribution
y_samples = model.predict(X_test, return_type='samples', n_samples=100)
# Returns array of shape (n_test_samples, n_samples)
```

### Probabilistic Interface

```python
# sklearn-style probabilistic predictions
y_proba = model.predict_proba(X_test, n_samples=50)

# Dedicated quantile method
y_intervals = model.predict_quantiles(X_test, quantiles=[0.1, 0.9])
```

## Installation and Dependencies

### Core Installation (Lightweight)

```bash
pip install xgboostlss
```

Installs only core dependencies: `xgboost`, `scikit-learn`, `numpy`, `pandas`, `scipy`, `tqdm`

### Optional Dependencies

#### Visualization and Interpretation
```bash
pip install xgboostlss[viz]
```
Adds: `matplotlib`, `seaborn`, `plotnine`, `shap`

#### PyTorch-based Distributions
```bash
pip install xgboostlss[torch]  
```
Adds: `torch`, `pyro-ppl`

#### Hyperparameter Optimization
```bash
pip install xgboostlss[optim]
```
Adds: `optuna`

#### All Features
```bash
pip install xgboostlss[all]
```
Installs all optional dependencies for full compatibility.

## Python 3.12 Compatibility

The new dependency management ensures Python 3.12 compatibility:

- **Flexible version constraints**: Uses `>=` instead of `~=` for better compatibility
- **Optional heavy dependencies**: PyTorch, Pyro moved to optional installs
- **Core functionality preserved**: Basic probabilistic modeling works with minimal deps
- **Gradual migration**: `[all]` option maintains backward compatibility

## Migration Guide

### From Old Interface

**Before (Complex - 6+ steps):**
```python
from xgboostlss.model import XGBoostLSS
from xgboostlss.distributions.Gaussian import Gaussian
import xgboost as xgb

# Manual data preparation
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# Distribution configuration
dist = Gaussian(stabilization="MAD", response_fn="softplus", loss_fn="nll")

# Model setup and training
model = XGBoostLSS(dist)
params = {'learning_rate': 0.1, 'max_depth': 6}
model.train(params, dtrain, num_boost_round=100)

# Prediction
pred_params = model.predict(dtest, pred_type="parameters")
```

**After (Simple - 2 steps):**
```python
from xgboostlss import XGBoostLSSRegressor

# One-step setup and training
model = XGBoostLSSRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Simple prediction
y_pred = model.predict(X_test)
```

### Maintaining Advanced Features

For users who need access to the original low-level interface:

```python
# Access the underlying XGBoostLSS model
underlying_model = model._model  # After fitting

# Access the distribution object
distribution = model.distribution_  # After fitting

# Custom predictions with original interface
dtest = xgb.DMatrix(X_test)
custom_pred = underlying_model.predict(dtest, pred_type="expectiles")
```

## API Reference

### XGBoostLSSRegressor

Main sklearn-compatible interface class.

#### Parameters

- **distribution** (str or Distribution, default='auto'): Distribution to use
  - `'auto'`: Automatic detection based on target
  - `'gaussian'`/`'normal'`: Gaussian distribution  
  - `'gamma'`: Gamma distribution (positive values)
  - `'beta'`: Beta distribution (values in [0,1])
  - `'studentt'`/`'t'`: Student's t-distribution
  - Or pass Distribution instance directly

- **n_estimators** (int, default=100): Number of boosting rounds
- **learning_rate** (float, default=0.3): Step size shrinkage  
- **max_depth** (int, default=6): Maximum depth of trees
- **random_state** (int, optional): Random seed
- **kwargs**: Additional XGBoost parameters

#### Methods

- **fit(X, y, **fit_params)**: Fit the model
- **predict(X, return_type='mean', **predict_params)**: Make predictions
- **predict_proba(X, **predict_params)**: Sample from predictive distribution
- **predict_quantiles(X, quantiles=None, **predict_params)**: Predict quantiles

#### Attributes

- **distribution_**: Fitted distribution object
- **feature_importances_**: Feature importance scores

## Examples

See `examples/sklearn_interface_demo.py` for comprehensive examples comparing old vs new interfaces.

## Benefits Summary

The new sklearn-compatible interface provides:

✅ **Simplified workflow**: 2 steps vs 6+ steps  
✅ **Automatic configuration**: No need for distribution expertise  
✅ **sklearn compatibility**: Works with entire ecosystem  
✅ **Python 3.12 support**: Modern Python compatibility  
✅ **Optional dependencies**: Lighter installations  
✅ **Better UX**: Intuitive and discoverable API  
✅ **Maintained power**: All advanced features still accessible