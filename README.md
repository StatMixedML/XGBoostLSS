# 
<h4 align="center">

![Python Version](https://img.shields.io/badge/python-3.9%20|%203.10-lightblue.svg)
![GitHub Release (with filter)](https://img.shields.io/github/v/release/StatMixedML/XGBoostLSS?color=lightblue&label=release)
[![Github License](https://img.shields.io/badge/license-Apache%202.0-lightblue.svg)](https://opensource.org/licenses/Apache-2.0)
![Build Status](https://github.com/StatMixedML/XGBoostLSS/workflows/build%20status/badge.svg)
<img src="https://codecov.io/gh/StatMixedML/XGBoostLSS/branch/master/graph/badge.svg" alt="Code coverage status badge">
<!---
![GitHub repo size](https://img.shields.io/github/repo-size/StatMixedML/XGBoostLSS?label=repo%20size&color=lightblue)
![Code Coverage](https://raw.githubusercontent.com/StatMixedML/XGBoostLSS/coverage-badge/coverage.svg?raw=true)
![GitHub all releases](https://img.shields.io/github/downloads/StatMixedML/XGBoostLSS/total?color=lightblue)
[![HitCount](https://img.shields.io/endpoint?url=https%3A%2F%2Fhits.dwyl.com%2FStatMixedML%2FXGBoostLSS.json%3Fcolor%3Dgreen)](http://hits.dwyl.com/StatMixedML/XGBoostLSS)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
-->

</h4>

#
<img align="right" width="156.5223" height="181.3" src="../master/figures/XGBoostLSS_inv.png">

# XGBoostLSS - An extension of XGBoost to probabilistic modelling
We propose a new framework of XGBoost that predicts the entire conditional distribution of univariate and multivariate responses. In particular, **XGBoostLSS** models all moments of a parametric distribution, i.e., mean, location, scale and shape (LSS), instead of the conditional mean only. Choosing from a wide range of continuous, discrete, and mixed discrete-continuous distribution, modelling and predicting the entire conditional distribution greatly enhances the flexibility of XGBoost, as it allows to create probabilistic forecasts from which prediction intervals and quantiles of interest can be derived.

## Features
:white_check_mark: Estimation of all distributional parameters. <br/>
:white_check_mark: Multi-target regression allows modelling of multivariate responses and their dependencies. <br/>
:white_check_mark: Normalizing Flows allow modelling of complex and multi-modal distributions. <br/>
:white_check_mark: Automatic derivation of Gradients and Hessian of all distributional parameters using [PyTorch](https://pytorch.org/docs/stable/autograd.html). <br/>
:white_check_mark: Automated hyper-parameter search, including pruning, is done via [Optuna](https://optuna.org/). <br/>
:white_check_mark: The output of XGBoostLSS is explained using [SHapley Additive exPlanations](https://github.com/dsgibbons/shap). <br/>
:white_check_mark: XGBoostLSS provides full compatibility with all the features and functionality of XGBoost. <br/>
:white_check_mark: XGBoostLSS is available in Python. <br/>

## News
:boom: [2023-07-19] Release of v0.3.0 introduces Normalizing Flows. See the [release notes](https://github.com/StatMixedML/XGBoostLSS/releases) for an overview. <br/>
:boom: [2023-06-22] Release of v0.2.2. See the [release notes](https://github.com/StatMixedML/XGBoostLSS/releases) for an overview. <br/>
:boom: [2023-06-21] XGBoostLSS now supports multi-target regression. <br/>
:boom: [2023-06-07] XGBoostLSS now supports Zero-Inflated and Zero-Adjusted Distributions. <br/>
:boom: [2023-05-26] Release of v0.2.1. See the [release notes](https://github.com/StatMixedML/XGBoostLSS/releases) for an overview. <br/>
:boom: [2023-05-18] Release of v0.2.0. See the [release notes](https://github.com/StatMixedML/XGBoostLSS/releases) for an overview. <br/>
:boom: [2021-12-22] XGBoostLSS now supports estimating the full predictive distribution via [Expectile Regression](https://epub.ub.uni-muenchen.de/31542/1/1471082x14561155.pdf). <br/>
:boom: [2021-12-20] XGBoostLSS is initialized with suitable starting values to improve convergence of estimation. <br/>
:boom: [2021-12-04] XGBoostLSS now supports automatic derivation of Gradients and Hessians. <br/>
:boom: [2021-12-02] XGBoostLSS now supports pruning during hyperparameter optimization. <br/>
:boom: [2021-11-14] XGBoostLSS v0.1.0 is released!

## Installation
To install XGBoostLSS, please first run
```python
pip install git+https://github.com/StatMixedML/XGBoostLSS.git
```
Then, to install the shap-dependency, run
```python
pip install git+https://github.com/dsgibbons/shap.git
```

## How to use
We refer to the [example section](https://github.com/StatMixedML/XGBoostLSS/tree/master/examples) for example notebooks.

## Available Distributions
XGBoostLSS currently supports the following distributions.

| Distribution                                                                                                                         |   Usage                   |Type                                        | Support                         | Number of Parameters            |
| :----------------------------------------------------------------------------------------------------------------------------------: |:------------------------: |:-------------------------------------:     | :-----------------------------: | :-----------------------------: | 
| [Beta](https://pytorch.org/docs/stable/distributions.html#beta)                                                                      | `Beta()`                  | Continuous <br /> (Univariate)             | $y \in (0, 1)$                  | 2                               |
| [Cauchy](https://pytorch.org/docs/stable/distributions.html#cauchy)                                                                  | `Cauchy()`                | Continuous <br /> (Univariate)             | $y \in (-\infty,\infty)$        | 2                               |
| [Dirichlet](https://pytorch.org/docs/stable/distributions.html#dirichlet)                                                            | `Dirichlet(D)`            | Continuous <br /> (Multivariate)           | $y_{D} \in (0, 1)$              | D                               |
| [Expectile](https://epub.ub.uni-muenchen.de/31542/1/1471082x14561155.pdf)                                                            | `Expectile()`             | Continuous <br /> (Univariate)             | $y \in (-\infty,\infty)$        | Number of expectiles            |
| [Gamma](https://pytorch.org/docs/stable/distributions.html#gamma)                                                                    | `Gamma()`                 | Continuous <br /> (Univariate)             | $y \in (0, \infty)$             | 2                               |
| [Gaussian](https://pytorch.org/docs/stable/distributions.html#normal)                                                                | `Gaussian()`              | Continuous <br /> (Univariate)             | $y \in (-\infty,\infty)$        | 2                               |
| [Gumbel](https://pytorch.org/docs/stable/distributions.html#gumbel)                                                                  | `Gumbel()`                | Continuous <br /> (Univariate)             | $y \in (-\infty,\infty)$        | 2                               |
| [Laplace](https://pytorch.org/docs/stable/distributions.html#laplace)                                                                | `Laplace()`               | Continuous <br /> (Univariate)             | $y \in (-\infty,\infty)$        | 2                               |
| [LogNormal](https://pytorch.org/docs/stable/distributions.html#lognormal)                                                            | `LogNormal()`             | Continuous <br /> (Univariate)             | $y \in (0,\infty)$              | 2                               |
| [Multivariate Normal (Cholesky)](https://pytorch.org/docs/stable/distributions.html#multivariatenormal)                              | `MVN(D)`                  | Continuous <br /> (Multivariate)           | $y_{D} \in (-\infty,\infty)$    | D(D + 3)/2                      |
| [Multivariate Normal (Low-Rank)](https://pytorch.org/docs/stable/distributions.html#lowrankmultivariatenormal)                       | `MVN_LoRa(D, rank)`       | Continuous <br /> (Multivariate)           | $y_{D} \in (-\infty,\infty)$    | D(2+rank)                       |
| [Multivariate Student-T](https://docs.pyro.ai/en/stable/distributions.html#multivariatestudentt)                                     | `MVT(D)`                  | Continuous <br /> (Multivariate)           | $y_{D} \in (-\infty,\infty)$    | 1 + D(D + 3)/2                  |
| [Negative Binomial](https://pytorch.org/docs/stable/distributions.html#negativebinomial)                                             | `NegativeBinomial()`      | Discrete Count <br /> (Univariate)         | $y \in (0, 1, 2, 3, \ldots)$    | 2                               |
| [Poisson](https://pytorch.org/docs/stable/distributions.html#poisson)                                                                | `Poisson()`               | Discrete Count <br /> (Univariate)         | $y \in (0, 1, 2, 3, \ldots)$    | 1                               |
| [Spline Flow](https://docs.pyro.ai/en/stable/distributions.html#pyro.distributions.transforms.Spline)                                | `SplineFlow()`            | Continuous \& Discrete Count <br /> (Univariate)   | $y \in (-\infty,\infty)$ <br /> <br /> $y \in [0, \infty)$  <br  /> <br /> $y \in [0, 1]$  <br  />  <br /> $y \in (0, 1, 2, 3, \ldots)$ | 2xcount_bins + (count_bins-1) (order=quadratic)  <br  /> <br  />  3xcount_bins + (count_bins-1) (order=linear)                            |
| [Student-T](https://pytorch.org/docs/stable/distributions.html#studentt)                                                             | `StudentT()`              | Continuous <br /> (Univariate)             | $y \in (-\infty,\infty)$        | 3                               |
| [Weibull](https://pytorch.org/docs/stable/distributions.html#weibull)                                                                | `Weibull()`               | Continuous <br /> (Univariate)             | $y \in [0, \infty)$             | 2                               |
| [Zero-Adjusted Beta](https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/zero_inflated.py)                                  | `ZABeta()`                | Discrete-Continuous <br /> (Univariate)    | $y \in [0, 1)$                  | 3                               |
| [Zero-Adjusted Gamma](https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/zero_inflated.py)                                 | `ZAGamma()`               | Discrete-Continuous <br /> (Univariate)    | $y \in [0, \infty)$             | 3                               |
| [Zero-Adjusted LogNormal](https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/zero_inflated.py)                             | `ZALN()`                  | Discrete-Continuous <br /> (Univariate)    | $y \in [0, \infty)$             | 3                               |
| [Zero-Inflated Negative Binomial](https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/zero_inflated.py#L150)                | `ZINB()`                  | Discrete-Count <br /> (Univariate)         | $y \in [0, 1, 2, 3, \ldots)$    | 3                               |
| [Zero-Inflated Poisson](https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/zero_inflated.py#L121)                          | `ZIPoisson()`             | Discrete-Count <br /> (Univariate)         | $y \in [0, 1, 2, 3, \ldots)$    | 2                               |


## Some Notes
### Stabilization
Since XGBoostLSS updates the parameter estimates by optimizing Gradients and Hessians, it is important that these are comparable in magnitude for all distributional parameters. Due to variability regarding the ranges, the estimation of Gradients and Hessians might become unstable so that XGBoostLSS might not converge or might converge very slowly. To mitigate these effects, we have implemented a stabilization of Gradients and Hessians. 

For improved convergence, an alternative approach is to standardize the (continuous) response variable, such as dividing it by 100 (e.g., y/100). This approach proves especially valuable when the response range significantly differs from that of Gradients and Hessians. Nevertheless, it is essential to carefully evaluate and apply both the built-in stabilization and response standardization techniques in consideration of the specific dataset at hand.

### Runtime
Since XGBoostLSS is based on a *one vs. all estimation strategy*, where a separate tree is grown for each distributional parameter, it requires training ```[number of iterations] * [number of distributional parameters]``` trees. Hence, the runtime of XGBoostLSS is generally slightly higher for univariate distributions as compared to XGBoost, which requires training ```[number of iterations]``` trees only. Moreover, for a dataset with multivariate targets, estimation can become computationally expensive.

## Feedback
We encourage you to provide feedback on how to enhance XGBoostLSS or request the implementation of additional distributions by opening a new discussion.

## Reference Paper

März, Alexander (2022): [*Multi-Target XGBoostLSS Regression*](https://arxiv.org/abs/2210.06831). <br/>
März, A. and Kneib, T.: (2022) [*Distributional Gradient Boosting Machines*](https://arxiv.org/abs/2204.00778). <br/>
März, Alexander (2019): [*XGBoostLSS - An extension of XGBoost to probabilistic forecasting*](https://arxiv.org/abs/1907.03178). 

<!---
[![Arxiv link](https://img.shields.io/badge/arXiv-Multi%20Target%20XGBoostLSS%20Regression-color=brightgreen)](https://arxiv.org/abs/2210.06831) <br/>
[![Arxiv link](https://img.shields.io/badge/arXiv-Distributional%20Gradient%20Boosting%20Machines-color=brightgreen)](https://arxiv.org/abs/2204.00778) <br/>
[![Arxiv link](https://img.shields.io/badge/arXiv-XGBoostLSS%3A%20An%20extension%20of%20XGBoost%20to%20probabilistic%20forecasting-color=brightgreen)](https://arxiv.org/abs/1907.03178) <br/>
--->
