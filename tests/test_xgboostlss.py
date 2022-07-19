import numpy as np
import pandas as pd
from pytest import approx

from tests.conftest import get_scale_shares
from lss_xgboost.datasets.data_loader import load_simulated_data
from lss_xgboost.distributions.Gaussian import Gaussian
from lss_xgboost.model import xgb, xgboostlss


def test_xgblss_train():
    # arrange
    # The data is a simulated Gaussian as follows, where x is the only true feature and all others are noise variables
    # loc = 10
    # scale = 1 + 4*((0.3 < x) & (x < 0.5)) + 2*(x > 0.7)
    train, test = load_simulated_data()
    n_cpu = 1

    X_train, y_train = train.iloc[:, 1:], train.iloc[:, 0]

    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=n_cpu)

    distribution = Gaussian  # Estimates both location and scale parameters of the Gaussian simultaneously.
    distribution.stabilize = "None"  # Option to stabilize Gradient/Hessian. Options are "None", "MAD", "L2"

    xgb_params = {
        "eta": 0.8430275405809186,
        "max_depth": 2,
        "gamma": 18.527802448042262,
        "subsample": 0.8536413558773828,
        "colsample_bytree": 0.5824797484661761,
        "min_child_weight": 205,
    }
    n_rounds = 25
    np.random.seed(123)

    # act
    res = xgboostlss.train(xgb_params,
                           dtrain,
                           dist=distribution,
                           num_boost_round=n_rounds)

    # assert
    assert isinstance(res, xgb.Booster)


def test_xgblss_estimated_parameter():
    # arrange
    # The data is a simulated Gaussian as follows, where x is the only true feature and all others are noise variables
    # loc = 10
    # scale = 1 + 4*((0.3 < x) & (x < 0.5)) + 2*(x > 0.7)
    train, test = load_simulated_data()
    n_cpu = 1

    X_train, y_train = train.iloc[:, 1:], train.iloc[:, 0]
    X_test, y_test = test.iloc[:, 1:], test.iloc[:, 0]

    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=n_cpu)
    dtest = xgb.DMatrix(X_test, nthread=n_cpu)

    distribution = Gaussian  # Estimates both location and scale parameters of the Gaussian simultaneously.
    distribution.stabilize = "None"  # Option to stabilize Gradient/Hessian. Options are "None", "MAD", "L2"

    xgb_params = {
        "eta": 0.8430275405809186,
        "max_depth": 2,
        "gamma": 18.527802448042262,
        "subsample": 0.8536413558773828,
        "colsample_bytree": 0.5824797484661761,
        "min_child_weight": 205,
    }
    n_rounds = 25
    seed = 123
    np.random.seed(seed)
    xgboostlss_model = xgboostlss.train(xgb_params,
                                        dtrain,
                                        dist=distribution,
                                        num_boost_round=n_rounds)

    # act
    # Returns predicted distributional parameters
    res = xgboostlss.predict(xgboostlss_model,
                                     dtest,
                                     dist=distribution,
                                     pred_type="parameters")

    # assert
    # assert that parameter estimates are correct
    # mu = 10
    res.location = res.location.round(2)
    assert approx(res.location.unique()[0], abs=0.1) == 10.0

    # sigma
    # scale formula is:
    # scale = 1 + 4*((0.3 < x) & (x < 0.5)) + 2*(x > 0.7)
    # we should find a scale
    # for x < 0.3 the scale is 1
    # for x between 0.3 and 0.5 the scale is 5
    # for x between 0.5 and 0.7 the scale is 1
    # for x > 0.7 the scale is 3
    # x is between 0 and 1
    # so 50% of the samples should have scale = 1
    # 30% of the samples should have scale = 3
    # 20% of the samples should have scale = 5
    pred_scale_shares = res.scale.round().value_counts() / res.shape[0]
    pred_scale_shares = pred_scale_shares.reset_index()
    pred_scale_shares.rename(columns={"index": "scale", "scale": "share"}, inplace=True)

    assert approx(pred_scale_shares.loc[pred_scale_shares.scale == 1, "share"], abs=0.05) == 0.5
    assert approx(pred_scale_shares.loc[pred_scale_shares.scale == 3, "share"], abs=0.05) == 0.3
    assert approx(pred_scale_shares.loc[pred_scale_shares.scale == 5, "share"], abs=0.05) == 0.2


def test_xgblss_prediction_quantiles():
    # arrange
    # The data is a simulated Gaussian as follows, where x is the only true feature and all others are noise variables
    # loc = 10
    # scale = 1 + 4*((0.3 < x) & (x < 0.5)) + 2*(x > 0.7)
    train, test = load_simulated_data()
    n_cpu = 1

    X_train, y_train = train.iloc[:, 1:], train.iloc[:, 0]
    X_test, y_test = test.iloc[:, 1:], test.iloc[:, 0]

    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=n_cpu)
    dtest = xgb.DMatrix(X_test, nthread=n_cpu)

    distribution = Gaussian  # Estimates both location and scale parameters of the Gaussian simultaneously.
    distribution.stabilize = "None"  # Option to stabilize Gradient/Hessian. Options are "None", "MAD", "L2"

    xgb_params = {
        "eta": 0.8430275405809186,
        "max_depth": 2,
        "gamma": 18.527802448042262,
        "subsample": 0.8536413558773828,
        "colsample_bytree": 0.5824797484661761,
        "min_child_weight": 205,
    }
    n_rounds = 25
    quant_sel = [0.05, 0.95]
    seed = 123
    np.random.seed(seed)
    xgboostlss_model = xgboostlss.train(xgb_params,
                                        dtrain,
                                        dist=distribution,
                                        num_boost_round=n_rounds)

    # act
    # Using predicted distributional parameters, calculate quantiles
    res = xgboostlss.predict(xgboostlss_model,
                                        dtest,
                                        dist=distribution,
                                        pred_type="quantiles",
                                        quantiles=quant_sel,
                                        seed=seed)

    # assert
    # so 50% of the samples have scale = 1
    # 30% of the samples have scale = 3
    # 20% of the samples have scale = 5

    pred_quantiles = res.round(1)
    share_05_s1, share_05_s3, share_05_s5 = get_scale_shares(pred_quantiles, quant_sel="quant_0.05")
    assert approx(share_05_s1, abs=0.05) == 0.5
    assert approx(share_05_s3, abs=0.05) == 0.3
    assert approx(share_05_s5, abs=0.05) == 0.2
    share_95_s1, share_95_s3, share_95_s5 = get_scale_shares(pred_quantiles, quant_sel="quant_0.95")
    assert approx(share_95_s1, abs=0.05) == 0.5
    assert approx(share_95_s3, abs=0.05) == 0.3
    assert approx(share_95_s5, abs=0.05) == 0.2


def test_xgblss_prediction_simulation():
    # arrange
    # The data is a simulated Gaussian as follows, where x is the only true feature and all others are noise variables
    # loc = 10
    # scale = 1 + 4*((0.3 < x) & (x < 0.5)) + 2*(x > 0.7)
    train, test = load_simulated_data()
    n_cpu = 1

    X_train, y_train = train.iloc[:, 1:], train.iloc[:, 0]
    X_test, y_test = test.iloc[:, 1:], test.iloc[:, 0]

    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=n_cpu)
    dtest = xgb.DMatrix(X_test, nthread=n_cpu)

    distribution = Gaussian  # Estimates both location and scale parameters of the Gaussian simultaneously.
    distribution.stabilize = "None"  # Option to stabilize Gradient/Hessian. Options are "None", "MAD", "L2"

    xgb_params = {
        "eta": 0.8430275405809186,
        "max_depth": 2,
        "gamma": 18.527802448042262,
        "subsample": 0.8536413558773828,
        "colsample_bytree": 0.5824797484661761,
        "min_child_weight": 205,
    }
    n_rounds = 25
    seed = 123
    np.random.seed(seed)
    xgboostlss_model = xgboostlss.train(xgb_params,
                                        dtrain,
                                        dist=distribution,
                                        num_boost_round=n_rounds)
    # Number of samples to draw from predicted distribution
    n_samples = 100

    # act
    # Using predicted distributional parameters, sample from distribution
    res = xgboostlss.predict(xgboostlss_model,
                                dtest,
                                dist=distribution,
                                pred_type="response",
                                n_samples=n_samples,
                                seed=seed)

    # assert
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (3000, n_samples)
    assert approx(res.quantile(0.99).mean(), abs=0.05) == 11.79
    assert approx(res.quantile(0.01).mean(), abs=0.05) == 8.36
