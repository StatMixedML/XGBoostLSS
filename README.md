<img align="right" width="156.5223" height="181.3" src="../master/logo/XGBoostLSS_inv.png">

# XGBoostLSS - An extension of XGBoost to probabilistic forecasting
We propose a new framework of XGBoost that predicts the entire conditional distribution of a univariate response variable. In particular, **XGBoostLSS** models all moments of a parametric distribution, i.e., mean, location, scale and shape (LSS), instead of the conditional mean only. Choosing from a wide range of continuous, discrete and mixed discrete-continuous distribution, modelling and predicting the entire conditional distribution greatly enhances the flexibility of XGBoost, as it allows to create probabilistic forecasts from which prediction intervals and quantiles of interest can be derived.

## News
:boom: [2021-11-14] XGBoostLSS v0.1.0 is released!

## Features
:white_check_mark: XGBoostLSS supports simultaneous training and updating of all distributional parameters. <br/>
:heavy_check_mark: Automated hyper-parameter search is done via [Optuna](https://optuna.org/). <br/>
:white_check_mark: The output of XGBoostLSS is explained using [SHapley Additive exPlanations](https://github.com/slundberg/shap). <br/>
:white_check_mark: XGBoostLSS is available in Python. <br/>

## Available Distributions

Currently, XGBoostLSS supports the Gaussian distribution (both location and scale). More continuous distributions (e.g., Student-t, Gamma, ...), as well as discrete, mixed discrete-continuous and zero-inflated distributions are to come soon.

|        Distribution           |              Type               |            Support            |           Location            |           Scale               |           Shape 1             |           Shape 2             |    
| ----------------------------- | ------------------------------- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------------- | 
|          Gaussian             |  Continuous <br/> (Univariate)  |   $`- \infty < y < \infty`$   |  $`- \infty < \mu < \infty`$  |    $`0 < \sigma < \infty`$    |          $`None`$             |          $`None`$             |
...
<!-- |   Student-t    |   Continuous  | $`- \infty < y < \infty`$   | location: $`\mu_{i}`$, scale: $`\sigma_{i}`$, shape $`\nu_{i}`$ |  -->

## Installation
```python
$ pip install git+https://github.com/StatMixedML/XGBoostLSS.git
```
## Quick start
We refer to the [examples section](https://github.com/StatMixedML/XGBoostLSS/tree/master/examples) for an example notebook.

## Reference Paper
März, Alexander (2019) [*"XGBoostLSS - An extension of XGBoost to probabilistic forecasting"*](https://arxiv.org/abs/1907.03178). 
