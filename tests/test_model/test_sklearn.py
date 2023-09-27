import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from .test_model import univariate_params, multivariate_params

from xgboostlss.sklearn import XGBLSSRegressor
from xgboostlss.datasets.data_loader import load_simulated_gaussian_data, load_simulated_multivariate_gaussian_data

@pytest.fixture
def univariate_data():
    train, test = load_simulated_gaussian_data()
    X_train, y_train = train.filter(regex="x"), train["y"].values
    X_test, y_test = test.filter(regex="x"), test["y"].values

    return X_train, y_train, X_test, y_test

@pytest.fixture
def multivariate_data():
    data_sim = load_simulated_multivariate_gaussian_data()

    # Create 60%, 20%, 20% split for train, validation and test
    train, validate, test = np.split(data_sim.sample(frac=1, random_state=123),
                                     [int(0.6 * len(data_sim)), int(0.8 * len(data_sim))])

    # Train
    x_train = train.filter(regex="x")
    y_train = train.filter(regex="y").values

    # Validation
    x_eval = validate.filter(regex="x")
    y_eval = validate.filter(regex="y").values

    # Test
    x_test = test.filter(regex="x")

    return x_train, y_train, x_eval, y_eval, x_test


@pytest.fixture
def multivariate_xgblss():
    return XGBLSSRegressor(dist="MVN", dist_params=dict(D=3))


@pytest.fixture
def univariate_xgblss():
    return XGBLSSRegressor(dist="Gaussian")


@pytest.fixture
def flow_xgblss():
    return XGBLSSRegressor(dist="SplineFlow", dist_params=dict(target_support="real", count_bins=2))



def test_model_univ_train(univariate_data, univariate_xgblss, univariate_params):
    # Unpack
    X_train, y_train, _, _ = univariate_data
    opt_params, n_rounds = univariate_params
    xgblss = univariate_xgblss
    xgblss.set_params(**opt_params)
    xgblss.set_params(n_estimators=n_rounds)

    # Train the model
    xgblss.fit(X_train, y_train)

    # Assertions
    assert isinstance(xgblss.get_booster(), xgb.Booster)

def test_model_univ_train_eval(univariate_data, univariate_xgblss, univariate_params):
    # Unpack
    X_train, y_train, X_test, y_test = univariate_data
    opt_params, n_rounds = univariate_params
    xgblss = univariate_xgblss
    xgblss.set_params(**opt_params)
    xgblss.set_params(n_estimators=n_rounds)

    # Train the model
    xgblss.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    # Assertions
    assert isinstance(xgblss.get_booster(), xgb.Booster)


@pytest.mark.skip("Multivariate not working right now.")
def test_model_mvn_train(multivariate_data, multivariate_xgblss, multivariate_params):
        # Unpack
    X_train, y_train, X_eval, y_eval, _ = multivariate_data
    opt_params, n_rounds = multivariate_params
    xgblss = multivariate_xgblss
    xgblss.set_params(**opt_params)
    xgblss.set_params(n_estimators=n_rounds)

    # Train the model
    xgblss.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_eval, y_eval)])

    # Assertions
    assert isinstance(xgblss.get_booster(), xgb.Booster)

def test_model_predict(univariate_data, univariate_xgblss, univariate_params):
        # Unpack
    X_train, y_train, X_test, _ = univariate_data
    opt_params, n_rounds = univariate_params
    xgblss = univariate_xgblss
    xgblss.set_params(**opt_params)
    xgblss.set_params(n_estimators=n_rounds)

    # Train the model
    xgblss.fit(X_train, y_train)

    pred_params = xgblss.predict(X_test)

    # Assertions
    assert isinstance(pred_params, (np.ndarray, type(None)))
    assert not np.any(np.isnan(pred_params))
    assert not np.isinf(pred_params).any().any()
    assert pred_params.shape[1] == xgblss._estimator.dist.n_dist_param
    assert pytest.approx(pred_params[:, 0].mean(), abs=0.2) == 10.0


def test_model_flow_train(univariate_data, flow_xgblss):
    # Unpack
    X_train, y_train, _, _ = univariate_data
    flow_xgblss.set_params(**{"eta": 0.1, "n_estimators": 10})

    # Train the model
    flow_xgblss.fit(X_train, y_train)

    # Assertions
    assert isinstance(flow_xgblss.get_booster(), xgb.Booster)