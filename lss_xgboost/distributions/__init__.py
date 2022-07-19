"""XGBoostLSS - An extension of XGBoost to probabilistic forecasting"""
from typing import Union

from lss_xgboost.distributions.Gaussian import *
from lss_xgboost.distributions.Gaussian_AutoGrad import *
from lss_xgboost.distributions.StudentT import *
from lss_xgboost.distributions.BCT import *
from lss_xgboost.distributions.Beta import *
from lss_xgboost.distributions.NegativeBinomial import *
from lss_xgboost.distributions.Expectile import *
from lss_xgboost.distributions.Gamma import *

DistributionType = Union[Gaussian, StudentT, BCT, Beta, NBI, Expectile, Gamma]
