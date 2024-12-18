from xgboostlss.model import *
from xgboostlss.distributions.Gaussian import *
from xgboostlss.distributions.Mixture import *
from xgboostlss.distributions.Expectile import *
from xgboostlss.distributions.MVN import *
from xgboostlss.distributions.SplineFlow import *
from xgboostlss.datasets.data_loader import load_simulated_gaussian_data, load_simulated_multivariate_gaussian_data
import pytest
from pytest import approx


@pytest.fixture
def univariate_data():
    train, test = load_simulated_gaussian_data()
    X_train, y_train = train.filter(regex="x"), train["y"].values
    X_test, y_test = test.filter(regex="x"), test["y"].values
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    deval = xgb.DMatrix(X_test, label=y_test)

    return dtrain, dtest, deval, X_test


@pytest.fixture
def univariate_xgblss():
    return XGBoostLSS(Gaussian())


@pytest.fixture
def mixture_xgblss():
    return XGBoostLSS(Mixture(Gaussian()))


@pytest.fixture
def flow_xgblss():
    return XGBoostLSS(SplineFlow(target_support="real", count_bins=2))


@pytest.fixture
def expectile_xgblss():
    return XGBoostLSS(Expectile())


@pytest.fixture
def univariate_params():
    opt_params = {
        "eta": 0.10015347345470738,
        "max_depth": 8,
        "gamma": 24.75078796889987,
        "subsample": 0.6161756203438147,
        "colsample_bytree": 0.851057889242629,
        "min_child_weight": 147.09687376037445,
        "booster": "gbtree",
    }
    n_rounds = 98

    return opt_params, n_rounds


@pytest.fixture
def expectile_params():
    opt_params = {
        "eta": 0.7298897353706068,
        "max_depth": 2,
        "gamma": 5.90940257278992e-06,
        "subsample": 0.9810129322454306,
        "colsample_bytree": 0.9546244491014185,
        "min_child_weight": 113.32324947486019,
        "booster": "gbtree",
    }
    n_rounds = 3

    return opt_params, n_rounds


@pytest.fixture
def multivariate_data():
    data_sim = load_simulated_multivariate_gaussian_data()

    # Create 60%, 20%, 20% split for train, validation and test
    train, validate, test = np.split(
        data_sim.sample(frac=1, random_state=123), [int(0.6 * len(data_sim)), int(0.8 * len(data_sim))]
    )

    # Train
    x_train = train.filter(regex="x")
    y_train = train.filter(regex="y").values
    dtrain_mvn = xgb.DMatrix(x_train, label=y_train)

    # Validation
    x_eval = validate.filter(regex="x")
    y_eval = validate.filter(regex="y").values
    deval_mvn = xgb.DMatrix(x_eval, label=y_eval)

    # Test
    x_test = test.filter(regex="x")
    dtest_mvn = xgb.DMatrix(x_test)

    return dtrain_mvn, deval_mvn, dtest_mvn, x_test


@pytest.fixture
def multivariate_xgblss():
    return XGBoostLSS(MVN(D=3))


@pytest.fixture
def multivariate_params():
    opt_params = {
        "eta": 0.06058920928573687,
        "max_depth": 2,
        "gamma": 2.8407158704437237e-05,
        "subsample": 0.5214068113552733,
        "colsample_bytree": 0.8185136492497096,
        "min_child_weight": 8.847572679915343,
        "booster": "gbtree",
    }
    n_rounds = 100

    return opt_params, n_rounds


class TestClass:
    def test_model_univ_train(self, univariate_data, univariate_xgblss, univariate_params):
        # Unpack
        dtrain, _, _, _ = univariate_data
        opt_params, n_rounds = univariate_params
        xgblss = univariate_xgblss

        # Train the model
        xgblss.train(opt_params, dtrain, n_rounds)

        # Assertions
        assert isinstance(xgblss.booster, xgb.Booster)

    def test_model_univ_train_eval(self, univariate_data, univariate_xgblss, univariate_params):
        # Unpack
        dtrain, dtest, deval, _ = univariate_data
        opt_params, n_rounds = univariate_params
        xgblss = univariate_xgblss

        # Add evaluation set
        eval_set = [(dtrain, "train"), (deval, "evaluation")]
        eval_result = {}

        # Train the model
        xgblss.train(opt_params, dtrain, n_rounds, evals=eval_set, evals_result=eval_result)

        # Assertions
        assert isinstance(xgblss.booster, xgb.Booster)

    def test_model_mvn_train(self, multivariate_data, multivariate_xgblss, multivariate_params):
        # Unpack
        dtrain, deval, _, _ = multivariate_data
        opt_params, n_rounds = multivariate_params
        xgblss_mvn = multivariate_xgblss
        # Add evaluation set
        eval_set = [(dtrain, "train"), (deval, "evaluation")]
        eval_result = {}

        # Train the model
        xgblss_mvn.train(opt_params, dtrain, n_rounds, evals=eval_set, evals_result=eval_result)

        # Assertions
        assert isinstance(xgblss_mvn.booster, xgb.Booster)

    def test_model_predict(self, univariate_data, univariate_xgblss, univariate_params):
        # Unpack
        dtrain, dtest, _, _ = univariate_data
        opt_params, n_rounds = univariate_params
        xgblss = univariate_xgblss

        # Train the model
        xgblss.train(opt_params, dtrain, n_rounds)

        # Call the predict method
        n_samples = 100
        quantiles = [0.1, 0.5, 0.9]

        pred_params = xgblss.predict(dtest, pred_type="parameters")
        pred_samples = xgblss.predict(dtest, pred_type="samples", n_samples=n_samples)
        pred_quantiles = xgblss.predict(dtest, pred_type="quantiles", quantiles=quantiles)

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

    def test_model_mixture_train(self, univariate_data, mixture_xgblss):
        # Unpack
        dtrain, _, _, _ = univariate_data
        params, n_rounds = {"eta": 0.1}, 10
        xgblss = mixture_xgblss

        # Train the model
        xgblss.train(params, dtrain, n_rounds)

        # Assertions
        assert isinstance(xgblss.booster, xgb.Booster)

    def test_model_flow_train(self, univariate_data, flow_xgblss):
        # Unpack
        dtrain, _, _, _ = univariate_data
        params, n_rounds = {"eta": 0.1}, 10
        xgblss = flow_xgblss

        # Train the model
        xgblss.train(params, dtrain, n_rounds)

        # Assertions
        assert isinstance(xgblss.booster, xgb.Booster)
