from ..utils import BaseTestClass
import numpy as np
import pandas as pd

from xgboostlss.distributions import *
from xgboostlss.distributions.distribution_utils import DistributionClass


class TestClass(BaseTestClass):
    def test_dist_select(self):
        # Create data for testing
        target = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        candidate_distributions = [Beta, Gaussian, StudentT, Gamma, Cauchy, LogNormal, Weibull, Gumbel, Laplace]

        # Call the function
        dist_df = DistributionClass().dist_select(
            target, candidate_distributions, n_samples=10, plot=False
        ).reset_index(drop=True)

        # Assertions
        assert isinstance(dist_df, pd.DataFrame)
        assert not dist_df.isna().any().any()
        assert isinstance(dist_df["distribution"].values[0], str)
        assert np.issubdtype(dist_df["nll"].dtype, np.float64)
        assert not np.isnan(dist_df["nll"].values).any()
        assert not np.isinf(dist_df["nll"].values).any()
