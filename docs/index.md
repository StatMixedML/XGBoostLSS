<img align="right" width="156.5223" height="181.3" src="XGBoostLSS_inv.png">

# XGBoostLSS - An extension of XGBoost to probabilistic modelling and prediction
We propose a new framework of XGBoost that predicts the entire conditional distribution of univariate and multivariate responses. In particular, **XGBoostLSS** models all moments of a parametric distribution, i.e., mean, location, scale and shape (LSS), instead of the conditional mean only. Choosing from a wide range of continuous, discrete, and mixed discrete-continuous distributions, modelling and predicting the entire conditional distribution greatly enhances the flexibility of XGBoost, as it allows to create probabilistic forecasts from which prediction intervals and quantiles of interest can be derived.

## Features
- Estimation of all distributional parameters. <br/>
- Multi-target regression allows modelling of multivariate responses and their dependencies. <br/>
- Normalizing Flows allow modelling of complex and multi-modal distributions. <br/>
- Zero-Adjusted and Zero-Inflated Distributions for modelling excess of zeros in the data. <br/>
- Automatic derivation of Gradients and Hessian of all distributional parameters using [PyTorch](https://pytorch.org/docs/stable/autograd.html). <br/>
- Automated hyper-parameter search, including pruning, is done via [Optuna](https://optuna.org/). <br/>
- The output of XGBoostLSS is explained using [SHapley Additive exPlanations](https://github.com/dsgibbons/shap). <br/>
- XGBoostLSS provides full compatibility with all the features and functionality of XGBoost. <br/>
- XGBoostLSS is available in Python. <br/>

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