"""XGBoostLSS - An extension of XGBoost to probabilistic forecasting"""
from typing import Union

from xgboostlss.distributions.Gaussian import *
from xgboostlss.distributions.Gaussian_AutoGrad import *
from xgboostlss.distributions.StudentT import *
from xgboostlss.distributions.BCT import *
from xgboostlss.distributions.Beta import *
from xgboostlss.distributions.NegativeBinomial import *
from xgboostlss.distributions.Expectile import *
from xgboostlss.distributions.Gamma import *

DistributionType = Union[Gaussian, StudentT, BCT, Beta, NBI, Expectile, Gamma]
