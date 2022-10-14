<img align="right" width="156.5223" height="181.3" src="../master/figures/XGBoostLSS_inv.png">

# XGBoostLSS - An extension of XGBoost to probabilistic forecasting
We propose a new framework of XGBoost that predicts the entire conditional distribution of a univariate response variable. In particular, **XGBoostLSS** models all moments of a parametric distribution, i.e., mean, location, scale and shape (LSS), instead of the conditional mean only. Choosing from a wide range of continuous, discrete, and mixed discrete-continuous distribution, modelling and predicting the entire conditional distribution greatly enhances the flexibility of XGBoost, as it allows to create probabilistic forecasts from which prediction intervals and quantiles of interest can be derived.

## News
:boom: [2022-10-14] XGBoostLSS now supports multi-target regression. <br/>
:boom: [2022-01-03] XGBoostLSS now supports estimation of the Gamma distribution. <br/>
:boom: [2021-12-22] XGBoostLSS now supports estimating the full predictive distribution via [Expectile Regression](https://epub.ub.uni-muenchen.de/31542/1/1471082x14561155.pdf). <br/>
:boom: [2021-12-20] XGBoostLSS is initialized with suitable starting values to improve convergence of estimation. <br/>
:boom: [2021-12-04] XGBoostLSS now supports automatic derivation of Gradients and Hessians. <br/>
:boom: [2021-12-02] XGBoostLSS now supports pruning during hyperparameter optimization. <br/>
:boom: [2021-11-14] XGBoostLSS v0.1.0 is released!

## Features
:white_check_mark: Simultaneous updating of all distributional parameters. <br/>
:white_check_mark: Automatic derivation of Gradients and Hessian of all distributional parameters using [PyTorch](https://pytorch.org/docs/stable/autograd.html). <br/>
:white_check_mark: Automated hyper-parameter search, including pruning, is done via [Optuna](https://optuna.org/). <br/>
:white_check_mark: The output of XGBoostLSS is explained using [SHapley Additive exPlanations](https://github.com/slundberg/shap). <br/>
:white_check_mark: XGBoostLSS is available in Python. <br/>

## Work in Progress
:construction: Functions that facilitates the choice and evaluation of a candidate distribution (e.g., quantile residual plots, ...). <br/>
:construction: Calling XGBoostLSS from R via the [reticulate package](https://rstudio.github.io/reticulate/). <br/>
:construction: Estimation of full predictive distribution without relying on a distributional assumption.  <br/>
 
## Available Distributions
Currently, XGBoostLSS supports the following distributions. More continuous distributions, as well as discrete, mixed discrete-continuous and zero-inflated distributions are to come soon.

<img align="center" src="../master/figures/distr.png">

## Some Notes
### Stabilization
Since XGBoostLSS updates the parameter estimates by optimizing Gradients and Hessians, it is important that these are comparable in magnitude for all distributional parameters. Due to variability regarding the ranges, the estimation of Gradients and Hessians might become unstable so that XGBoostLSS might not converge or might converge very slowly. To mitigate these effects, we have implemented a stabilization of Gradients and Hessians. 

An additional option to improve convergence can be to standardize the (continuous) response variable, e.g., ```y/100```. This is especially useful if the range of the response differs strongly from the range of Gradients and Hessians. Both, the in-built stabilization, and the standardization of the response need to be carefully considered given the data at hand.

### Runtime
Since XGBoostLSS updates all distributional parameters simultaneously, it requires training ```[number of iterations] * [number of distributional parameters]``` trees. Hence, the runtime of XGBoostLSS is generally slightly higher as compared to XGBoost, which requires training ```[number of iterations]``` trees only. 

### Feedback
Please provide feedback on how to improve XGBoostLSS, or if you request additional distributions to be implemented, by opening a new issue.

## Installation
```python
$ pip install git+https://github.com/StatMixedML/XGBoostLSS.git
```
## How to use
We refer to the [examples section](https://github.com/StatMixedML/XGBoostLSS/tree/master/examples) for example notebooks.

## Reference Paper
März, Alexander (2022) [*"Multi-Target XGBoostLSS Regression"*](https://arxiv.org/abs/2210.06831). 
März, Alexander (2019) [*"XGBoostLSS - An extension of XGBoost to probabilistic forecasting"*](https://arxiv.org/abs/1907.03178). 
