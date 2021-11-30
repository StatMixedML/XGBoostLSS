<img align="right" width="156.5223" height="181.3" src="../master/figures/XGBoostLSS_inv.png">

# XGBoostLSS - An extension of XGBoost to probabilistic forecasting
We propose a new framework of XGBoost that predicts the entire conditional distribution of a univariate response variable. In particular, **XGBoostLSS** models all moments of a parametric distribution, i.e., mean, location, scale and shape (LSS), instead of the conditional mean only. Choosing from a wide range of continuous, discrete, and mixed discrete-continuous distribution, modelling and predicting the entire conditional distribution greatly enhances the flexibility of XGBoost, as it allows to create probabilistic forecasts from which prediction intervals and quantiles of interest can be derived.

## News
:boom: [2021-11-14] XGBoostLSS v0.1.0 is released!

## Features
:white_check_mark: XGBoostLSS supports simultaneous training and updating of all distributional parameters. <br/>
:white_check_mark: Automated hyper-parameter search is done via [Optuna](https://optuna.org/). <br/>
:white_check_mark: The output of XGBoostLSS is explained using [SHapley Additive exPlanations](https://github.com/slundberg/shap). <br/>
:white_check_mark: XGBoostLSS is available in Python. <br/>

## Work in Progress
:construction: Functions that facilitates the choice and evaluation of a candidate distribution (e.g., quantile residual plots, ...). <br/>
:construction: Calling XGBoostLSS from R via the [reticulate package](https://rstudio.github.io/reticulate/). <br/>
 
## Available Distributions
Currently, XGBoostLSS supports the following distributions. More continuous distributions, as well as discrete, mixed discrete-continuous and zero-inflated distributions are to come soon.

<img align="center" src="../master/figures/distr.png">

## A Note on Stabilization
Since XGBoostLSS updates the parameter estimates by optimizing Gradients and Hessians, it is important that these are comparable in magnitude for all distributional parameters. Due to imbalances regarding the ranges, the estimation of Gradients and Hessians might become unstable so that XGBoostLSS does not converge or converge very slowly. To mitigate these effects, we have a built-in stabilization of Gradients and Hessians. 

An additional option to improve convergence might be to standardize the response variable, e.g., ```y/10``` or ```y/100```. This is especially useful if the range of the response differs strongly from the range of the Gradients and Hessians. Both, the in-built stabilization, and the standardization of the response need to be carefully considered given the data at hand.

## Installation
```python
$ pip install git+https://github.com/StatMixedML/XGBoostLSS.git
```
## Quick Start
We refer to the [examples section](https://github.com/StatMixedML/XGBoostLSS/tree/master/examples) for an example notebook.

## Reference Paper
März, Alexander (2019) [*"XGBoostLSS - An extension of XGBoost to probabilistic forecasting"*](https://arxiv.org/abs/1907.03178). 
