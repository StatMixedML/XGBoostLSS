<img align="right" width="156.5223" height="181.3" src="../master/logo/XGBoostLSS_inv.png">

# XGBoostLSS - An extension of XGBoost to probabilistic forecasting
We propose a new framework of XGBoost that predicts the entire conditional distribution of a univariate response variable. In particular, **XGBoostLSS** models all moments of a parametric distribution, i.e., mean, location, scale and shape (LSS), instead of the conditional mean only. Choosing from a wide range of continuous, discrete and mixed discrete-continuous distribution, modelling and predicting the entire conditional distribution greatly enhances the flexibility of XGBoost, as it allows to create probabilistic forecasts from which prediction intervals and quantiles of interest can be derived.

## News
- [2021-11-14] XGBoostLSS 0.1.0 is released!

## Features
- In contrast to previous implementations, XGBoostLSS now supports simultaneous training and updating of all distributional parameters.
- Automate hyperparameter search is done via [Optuna](https://optuna.org/).
- Currently, XGBoostLSS only supports the Gaussian distribution (both location and scale). More continuous distributions (e.g., Student-T, Gamma, ...), as well as discrete, mixed discrete-continuous and Zero-Inflated distributions are to come soon.
- XGBoostLSS is currently available in Python only.

## Installation
```python
$ pip install git+https://github.com/StatMixedML/XGBoostLSS.git
```
## Quick start
We refer to the [examples section](https://github.com/StatMixedML/XGBoostLSS/tree/master/examples) for an example notebook.

## Reference Paper
März, Alexander (2019) [*"XGBoostLSS - An extension of XGBoost to probabilistic forecasting"*](https://arxiv.org/abs/1907.03178). 
