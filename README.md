<h4 align="center">

|  | **[Documentation](https://statmixedml.github.io/XGBoostLSS/)** · **[Release Notes](https://github.com/StatMixedML/XGBoostLSS/releases)** |
|---|---|
| **Open&#160;Source** | [![Apache 2.0](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/StatMixedML/XGBoostLSS/blob/master/LICENSE.txt) |
| **CI/CD** | [![github-actions](https://img.shields.io/github/actions/workflow/status/statmixedml/xgboostlss/unit-tests.yml?logo=github)](https://github.com/statmixedml/xgboostlss/actions/workflows/unit-tests.yml) <img src="https://github.com/StatMixedML/XGBoostLSS/actions/workflows/mkdocs.yaml/badge.svg" alt="Documentation status badge"> |
| **Code** |  [![!pypi](https://img.shields.io/pypi/v/xgboostlss?color=orange)](https://pypi.org/project/xgboostlss/) [![!python-versions](https://img.shields.io/pypi/pyversions/xgboostlss)](https://www.python.org/) <img src="https://codecov.io/gh/StatMixedML/XGBoostLSS/branch/master/graph/badge.svg" alt="Code coverage status badge"> |
| **Downloads** | ![Pepy Total Downlods](https://img.shields.io/pepy/dt/xgboostlss?label=PyPI%20Downloads%2FMonth&color=green) |
| **Citation** | [![Arxiv link](https://img.shields.io/badge/arXiv-XGBoostLSS%3A%20An%20extension%20of%20XGBoost%20to%20probabilistic%20forecasting-color=brightgreen)](https://arxiv.org/abs/1907.03178) |

<!---
[![Github License](https://img.shields.io/badge/license-Apache%202.0-lightblue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://github.com/StatMixedML/XGBoostLSS/actions/workflows/mkdocs.yaml/badge.svg)](https://StatMixedML.github.io/XGBoostLSS/)
![Build Status](https://github.com/StatMixedML/XGBoostLSS/workflows/build%20status/badge.svg)
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
We introduce a comprehensive framework that models and predicts the full conditional distribution of univariate and multivariate targets as a function of covariates. Choosing from a wide range of continuous, discrete, and mixed discrete-continuous distributions, modelling and predicting the entire conditional distribution greatly enhances the flexibility of XGBoost, as it allows to create probabilistic forecasts from which prediction intervals and quantiles of interest can be derived.

## `Features`
:white_check_mark: Estimation of all distributional parameters. <br/>
:white_check_mark: Normalizing Flows allow modelling of complex and multi-modal distributions. <br/>
:white_check_mark: Mixture-Densities can model a diverse range of data characteristics. <br/>
:white_check_mark: Multi-target regression allows modelling of multivariate responses and their dependencies. <br/>
:white_check_mark: Zero-Adjusted and Zero-Inflated Distributions for modelling excess of zeros in the data. <br/>
:white_check_mark: Automatic derivation of Gradients and Hessian of all distributional parameters using [PyTorch](https://pytorch.org/docs/stable/autograd.html). <br/>
:white_check_mark: Automated hyper-parameter search, including pruning, is done via [Optuna](https://optuna.org/). <br/>
:white_check_mark: The output of XGBoostLSS is explained using [SHapley Additive exPlanations](https://github.com/dsgibbons/shap). <br/>
:white_check_mark: XGBoostLSS provides full compatibility with all the features and functionality of XGBoost. <br/>
:white_check_mark: XGBoostLSS is available in Python. <br/>

## `News`
:boom: [2025-10-31] Release of v0.5.0 - python 3.13 support and dependency minimization. See the [release notes](https://github.com/StatMixedML/XGBoostLSS/releases) for an overview. <br/>
:boom: [2024-01-19] Release of XGBoostLSS to [PyPI](https://pypi.org/project/xgboostlss/). <br/>
:boom: [2023-08-25] Release of v0.4.0 introduces Mixture-Densities. See the [release notes](https://github.com/StatMixedML/XGBoostLSS/releases) for an overview. <br/>
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

## `Installation`
To install the development version, please use
```python
pip install git+https://github.com/StatMixedML/XGBoostLSS.git
```
For the PyPI version, please use
```python
pip install xgboostlss
```

## `Available Distributions`
Our framework is built upon PyTorch and Pyro, enabling users to harness a diverse set of distributional families. XGBoostLSS currently supports the [following distributions](https://statmixedml.github.io/XGBoostLSS/distributions/). 

## `How to Use`
Please visit the [example section](https://statmixedml.github.io/XGBoostLSS/examples/Gaussian_Regression/) for guidance on how to use the framework.

## `Documentation`
For more information and context, please visit the [documentation](https://statmixedml.github.io/XGBoostLSS/).

## `Feedback`
We encourage you to provide feedback on how to enhance XGBoostLSS or request the implementation of additional distributions by opening a [new discussion](https://github.com/StatMixedML/XGBoostLSS/discussions).

## `How to Cite`
If you use XGBoostLSS in your research, please cite it as:

```bibtex
@misc{Maerz2023,
  author = {Alexander M\"arz},
  title = {{XGBoostLSS: An Extension of XGBoost to Probabilistic Modelling}},
  year = {2023},
  note = {GitHub repository, Version 0.5.0},
  howpublished = {\url{https://github.com/StatMixedML/XGBoostLSS}}
}
```

## `Reference Paper`
[![Arxiv link](https://img.shields.io/badge/arXiv-Multi%20Target%20XGBoostLSS%20Regression-color=brightgreen)](https://arxiv.org/abs/2210.06831) <br/>
[![Arxiv link](https://img.shields.io/badge/arXiv-Distributional%20Gradient%20Boosting%20Machines-color=brightgreen)](https://arxiv.org/abs/2204.00778) <br/>
[![Arxiv link](https://img.shields.io/badge/arXiv-XGBoostLSS%3A%20An%20extension%20of%20XGBoost%20to%20probabilistic%20forecasting-color=brightgreen)](https://arxiv.org/abs/1907.03178) <br/>

<!---
März, Alexander (2022): [*Multi-Target XGBoostLSS Regression*](https://arxiv.org/abs/2210.06831). <br/>
März, A. and Kneib, T.: (2022) [*Distributional Gradient Boosting Machines*](https://arxiv.org/abs/2204.00778). <br/>
März, Alexander (2019): [*XGBoostLSS - An extension of XGBoost to probabilistic forecasting*](https://arxiv.org/abs/1907.03178). 
--->

## `Star History`
<a href="https://star-history.com/#StatMixedML/XGBoostLSS&Date">
    <img src="https://api.star-history.com/svg?repos=StatMixedML/XGBoostLSS&type=Date" width="450">
</a>
