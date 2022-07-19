"""XGBoostLSS - An extension of XGBoost to probabilistic forecasting"""
from typing import Union

from xgboost_lss.distributions.Gaussian import *
from xgboost_lss.distributions.Gaussian_AutoGrad import *
from xgboost_lss.distributions.StudentT import *
from xgboost_lss.distributions.BCT import *
from xgboost_lss.distributions.Beta import *
from xgboost_lss.distributions.NegativeBinomial import *
from xgboost_lss.distributions.Expectile import *
from xgboost_lss.distributions.Gamma import *

DistributionType = Union[Gaussian, StudentT, BCT, Beta, NBI, Expectile, Gamma]
