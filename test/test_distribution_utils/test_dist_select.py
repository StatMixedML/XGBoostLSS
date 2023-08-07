from ..utils import BaseTestClass
import numpy as np
import pandas as pd

from xgboostlss.distributions import Beta, Gaussian, StudentT, Gamma, Cauchy, LogNormal, Weibull, Gumbel, Laplace
from xgboostlss.distributions.SplineFlow import *
from xgboostlss.distributions.MVN import *
from xgboostlss.distributions.MVT import *
from xgboostlss.distributions.MVN_LoRa import *
from xgboostlss.distributions.distribution_utils import DistributionClass as univariate_dist_class
from xgboostlss.distributions.multivariate_distribution_utils import Multivariate_DistributionClass as multivariate_dist_class
from xgboostlss.distributions.flow_utils import NormalizingFlowClass as flow_dist_class


class TestClass(BaseTestClass):
    ####################################################################################################################
    # Univariate Distribution
    ####################################################################################################################
    def test_univar_dist_select(self):
        # Create data for testing
        target = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        candidate_distributions = [Beta, Gaussian, StudentT, Gamma, Cauchy, LogNormal, Weibull, Gumbel, Laplace]

        # Call the function
        dist_df = univariate_dist_class().dist_select(
            target, candidate_distributions, n_samples=10, plot=False
        ).reset_index(drop=True)

        # Assertions
        assert isinstance(dist_df, pd.DataFrame)
        assert not dist_df.isna().any().any()
        assert isinstance(dist_df["distribution"].values[0], str)
        assert np.issubdtype(dist_df["nll"].dtype, np.float64)
        assert not np.isnan(dist_df["nll"].values).any()
        assert not np.isinf(dist_df["nll"].values).any()

    def test_univar_dist_select_plot(self):
        # Create data for testing
        target = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        candidate_distributions = [Beta, Gaussian, StudentT, Gamma, Cauchy, LogNormal, Weibull, Gumbel, Laplace]

        # Call the function
        dist_df = univariate_dist_class().dist_select(
            target, candidate_distributions, n_samples=10, plot=True
        ).reset_index(drop=True)

        # Assertions
        assert isinstance(dist_df, pd.DataFrame)
        assert not dist_df.isna().any().any()
        assert isinstance(dist_df["distribution"].values[0], str)
        assert np.issubdtype(dist_df["nll"].dtype, np.float64)
        assert not np.isnan(dist_df["nll"].values).any()
        assert not np.isinf(dist_df["nll"].values).any()

    ####################################################################################################################
    # Normalizing Flows
    ####################################################################################################################
    def test_flow_select(self):
        # Create data for testing
        target = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        bound = np.max([np.abs(target.min()), target.max()])
        target_support = "real"

        candidate_flows = [
            SplineFlow(target_support=target_support, count_bins=2, bound=bound, order="linear"),
            SplineFlow(target_support=target_support, count_bins=2, bound=bound, order="quadratic")
        ]

        # Call the function
        dist_df = flow_dist_class().flow_select(
            target, candidate_flows, n_samples=10, plot=False
        ).reset_index(drop=True)

        # Assertions
        assert isinstance(dist_df, pd.DataFrame)
        assert not dist_df.isna().any().any()
        assert isinstance(dist_df["NormFlow"].values[0], str)
        assert np.issubdtype(dist_df["nll"].dtype, np.float64)
        assert not np.isnan(dist_df["nll"].values).any()
        assert not np.isinf(dist_df["nll"].values).any()

    def test_flow_select_plot(self):
        # Create data for testing
        target = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        bound = np.max([np.abs(target.min()), target.max()])
        target_support = "real"

        candidate_flows = [
            SplineFlow(target_support=target_support, count_bins=2, bound=bound, order="linear"),
            SplineFlow(target_support=target_support, count_bins=2, bound=bound, order="quadratic")
        ]

        # Call the function
        dist_df = flow_dist_class().flow_select(
            target, candidate_flows, n_samples=10, plot=True
        ).reset_index(drop=True)

        # Assertions
        assert isinstance(dist_df, pd.DataFrame)
        assert not dist_df.isna().any().any()
        assert isinstance(dist_df["NormFlow"].values[0], str)
        assert np.issubdtype(dist_df["nll"].dtype, np.float64)
        assert not np.isnan(dist_df["nll"].values).any()
        assert not np.isinf(dist_df["nll"].values).any()

    ####################################################################################################################
    # Multivariate Distribution
    ####################################################################################################################
    def test_multivar_dist_select(self):
        # Create data for testing
        multivar_dist_class = MVN()
        target = np.arange(0.1, 0.9, 0.1)
        target = multivar_dist_class.target_append(
            target,
            multivar_dist_class.n_targets,
            multivar_dist_class.n_dist_param
        )
        candidate_distributions = [MVN(), MVT(), MVN_LoRa()]

        # Call the function
        dist_df = multivariate_dist_class().dist_select(
            target, candidate_distributions, n_samples=10, plot=False
        ).reset_index(drop=True)

        # Assertions
        assert isinstance(dist_df, pd.DataFrame)
        assert not dist_df.isna().any().any()
        assert isinstance(dist_df["distribution"].values[0], str)
        assert np.issubdtype(dist_df["nll"].dtype, np.float64)
        assert not np.isnan(dist_df["nll"].values).any()
        assert not np.isinf(dist_df["nll"].values).any()
