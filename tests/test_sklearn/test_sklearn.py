"""Test scikit learn API for XGBoostLSS."""

import numpy as np
import pandas as pd

from xgboost import Booster

from xgboostlss.sklearn import XGBLSSRegressor
from xgboostlss.model import XGBoostLSS

from xgboostlss.distributions.Gaussian import Gaussian
from xgboostlss.distributions.Mixture import Mixture
from xgboostlss.distributions.SplineFlow import SplineFlow
from xgboostlss.distributions.Expectile import Expectile
from xgboostlss.datasets.data_loader import load_simulated_gaussian_data
import pytest
from pytest import approx


@pytest.fixture
def univariate_data():
    train, test = load_simulated_gaussian_data()
    X_train, y_train = train.filter(regex="x"), train["y"].values
    X_test, y_test = test.filter(regex="x"), test["y"].values

    return X_train, y_train, X_test, y_test


@pytest.fixture
def univariate_xgblss():
    params = {
        "learning_rate": 0.10015347345470738,
        "max_depth": 8,
        "gamma": 24.75078796889987,
        "subsample": 0.6161756203438147,
        "colsample_bytree": 0.851057889242629,
        "min_child_weight": 147.09687376037445,
        "booster": "gbtree",
        "n_estimators": 98,
    }
    return XGBLSSRegressor(Gaussian(), **params)


@pytest.fixture
def mixture_xgblss():
    params = {"learning_rate": 0.1, "n_estimators": 10}
    return XGBLSSRegressor(Mixture(Gaussian()), **params)


@pytest.fixture
def flow_xgblss():
    params = {"learning_rate": 0.1, "n_estimators": 10}
    spline_flow = SplineFlow(target_support="real", count_bins=2)
    return XGBLSSRegressor(spline_flow, **params)


@pytest.fixture
def expectile_xgblss():
    params = {
        "learning_rate": 0.7298897353706068,
        "max_depth": 2,
        "gamma": 5.90940257278992e-06,
        "subsample": 0.9810129322454306,
        "colsample_bytree": 0.9546244491014185,
        "min_child_weight": 113.32324947486019,
        "booster": "gbtree",
    }

    return XGBLSSRegressor(Expectile(), **params)


class TestClass:
    def test_model_univ_train(self, univariate_data, univariate_xgblss):
        # Unpack
        X_train, y_train, _, _ = univariate_data
        xgblss = univariate_xgblss

        # Train the model
        xgblss.fit(X_train, y_train)

        # Assertions
        assert isinstance(xgblss._Booster, Booster)
        assert isinstance(xgblss._BoosterLSS, XGBoostLSS)

    def test_model_mixture_train(self, univariate_data, mixture_xgblss):
        # Unpack
        X_train, y_train, _, _ = univariate_data
        xgblss = mixture_xgblss

        # Train the model
        xgblss.fit(X_train, y_train)

        # Assertions
        assert isinstance(xgblss._Booster, Booster)
        assert isinstance(xgblss._BoosterLSS, XGBoostLSS)

    def test_model_flow_train(self, univariate_data, flow_xgblss):
        # Unpack
        X_train, y_train, _, _ = univariate_data
        xgblss = flow_xgblss

        # Train the model
        xgblss.fit(X_train, y_train)

        # Assertions
        assert isinstance(xgblss._Booster, Booster)
        assert isinstance(xgblss._BoosterLSS, XGBoostLSS)

    def test_model_expectile_train(self, univariate_data, expectile_xgblss):
        # Unpack
        X_train, y_train, _, _ = univariate_data
        xgblss = expectile_xgblss

        # Train the model
        xgblss.fit(X_train, y_train)

        # Assertions
        assert isinstance(xgblss._Booster, Booster)
        assert isinstance(xgblss._BoosterLSS, XGBoostLSS)

    def test_model_predict(self, univariate_data, univariate_xgblss):
        # Unpack
        X_train, y_train, X_test, y_test = univariate_data
        # opt_params, n_rounds = univariate_params
        xgblss = univariate_xgblss

        # Train the model
        xgblss.fit(X_train, y_train)

        # Call the predict method
        n_samples = 100
        quantiles = [0.1, 0.5, 0.9]

        pred_params = xgblss.predict(X_test, pred_type="parameters")
        pred_samples = xgblss.predict(
            X_test, pred_type="samples", n_samples=n_samples
        )
        pred_quantiles = xgblss.predict(
            X_test, pred_type="quantiles", quantiles=quantiles
        )

        # Assertions
        assert isinstance(pred_params, (pd.DataFrame, type(None)))
        assert not pred_params.isna().any().any()
        assert not np.isinf(pred_params).any().any()
        assert pred_params.shape[1] == xgblss.dist.n_dist_param
        assert approx(pred_params["loc"].mean(), abs=0.2) == 10.0

        assert isinstance(pred_samples, (pd.DataFrame, type(None)))
        assert not pred_samples.isna().any().any()
        assert not np.isinf(pred_samples).any().any()
        assert pred_samples.shape[1] == n_samples

        assert isinstance(pred_quantiles, (pd.DataFrame, type(None)))
        assert not pred_quantiles.isna().any().any()
        assert not np.isinf(pred_quantiles).any().any()
        assert pred_quantiles.shape[1] == len(quantiles)
