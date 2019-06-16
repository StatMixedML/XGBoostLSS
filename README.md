# XGBoostLSS
We propose a new framework of XGBoost that predicts the entire conditional distribution of a univariate response variable. In particular, *XGBoostLSS* models all moments of a parametric distribution, i.e., mean, location, scale and shape (LSS), instead of the conditional mean only. Chosing from a wide range of continuous, discrete and mixed discrete-continuous distribution, modeling and predicting the entire conditional distribution greatly enhances the flexibility of XGBoost, as it allows to gain additional insight into the data generating process, as well as to create probabilistic forecasts from which prediction intervals and quantiles of interest can be derived. 

In its current implementation, *XGBoostLSS* is available in *R* and extensions to *Julia* and *Python* are in progress. 

