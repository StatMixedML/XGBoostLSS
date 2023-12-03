"""Test scikit learn API for XGBoostLSS."""

import numpy as np
import pandas as pd

from xgboostlss.sklearn import XGBLSSRegressor
from xgboostlss.model import XGBoostLSS
from xgboost import Booster
from xgboostlss.distributions.Gaussian import Gaussian
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
        "eta": 0.10015347345470738,
        "max_depth": 8,
        "gamma": 24.75078796889987,
        "subsample": 0.6161756203438147,
        "colsample_bytree": 0.851057889242629,
        "min_child_weight": 147.09687376037445,
        "booster": "gbtree",
        "n_estimators": 98,
    }
    return XGBLSSRegressor(Gaussian(), **params)


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
