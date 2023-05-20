import numpy as np
import pandas as pd
from pytest import approx

from xgboostlss.model import XGBoostLSS
from xgboostlss.datasets.data_loader import load_simulated_gaussian_data
from scipy.stats import norm
import xgboost as xgb
from xgboostlss.distributions.Gaussian import Gaussian

import multiprocessing


class TestXgboostLss:
    train, test = load_simulated_gaussian_data()
    n_cpu = max(multiprocessing.cpu_count() -1, 1)

    X_train, y_train = train.filter(regex="x"), train["y"].values
    X_test, y_test = test.filter(regex="x"), test["y"].values

    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=n_cpu)
    dtest = xgb.DMatrix(X_test, label=y_test, nthread=n_cpu)

    distribution = Gaussian(stabilization="None", response_fn="exp")

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

    def test_xgboostlss_train(self):
        # arrange
        # The data is a simulated Gaussian as follows, where x is the only true feature and all others are noise variables
        # loc = 10
        # scale = 1 + 4*((0.3 < x) & (x < 0.5)) + 2*(x > 0.7)
        np.random.seed(self.seed)

        xgblss = XGBoostLSS(self.distribution)

        # act
        res = xgblss.train(
            self.xgb_params,
            self.dtrain,
            num_boost_round=self.n_rounds
        )

        # assert
        assert isinstance(res, xgb.Booster)

    def test_xgboostlss_estimated_parameter(self):
        np.random.seed(self.seed)

        xgblss = XGBoostLSS(self.distribution)

        xgblss.train(
            self.xgb_params,
            self.dtrain,
            num_boost_round=self.n_rounds
        )

        # act
        # Returns predicted distributional parameters
        res = xgblss.predict(self.dtest, pred_type="parameters")

        # assert that parameter estimates are correct
        # mu = 10
        res["loc"] = res["loc"].round(2)
        assert approx(res["loc"].mean(), abs=0.1) == 10.0

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
        pred_scale_shares.rename(columns={"count": "share"}, inplace=True)

        assert approx(pred_scale_shares.loc[pred_scale_shares.scale == 1, "share"], abs=0.05) == 0.5
        assert approx(pred_scale_shares.loc[pred_scale_shares.scale == 3, "share"], abs=0.05) == 0.3
        assert approx(pred_scale_shares.loc[pred_scale_shares.scale == 5, "share"], abs=0.05) == 0.2

    def test_xgboostlss_prediction_quantiles(self):
        # arrange
        np.random.seed(self.seed)

        xgblss = XGBoostLSS(self.distribution)

        xgblss.train(
            self.xgb_params,
            self.dtrain,
            num_boost_round=self.n_rounds
        )
        n_samples = 1000
        quant_sel = [0.05, 0.95]  # Quantiles to calculate from predicted distribution

        # act
        # Using predicted distributional parameters, calculate quantiles
        res = xgblss.predict(
            self.dtest,
            pred_type="quantiles",
            n_samples=n_samples,
            quantiles=quant_sel,
        )

        # assert
        # so 50% of the samples have scale = 1
        # 30% of the samples have scale = 3
        # 20% of the samples have scale = 5

        pred_quantiles = res.round(1)
        share_05_s1, share_05_s3, share_05_s5 = self.get_scale_shares(pred_quantiles, quant_sel="quant_0.05")
        assert approx(share_05_s1, abs=0.05) == 0.5
        assert approx(share_05_s3, abs=0.05) == 0.3
        assert approx(share_05_s5, abs=0.05) == 0.2
        share_95_s1, share_95_s3, share_95_s5 = self.get_scale_shares(pred_quantiles, quant_sel="quant_0.95")
        assert approx(share_95_s1, abs=0.05) == 0.5
        assert approx(share_95_s3, abs=0.05) == 0.3
        assert approx(share_95_s5, abs=0.05) == 0.2

    def test_xgboostlss_prediction_simulation(self):
        # arrange
        np.random.seed(self.seed)

        xgblss = XGBoostLSS(self.distribution)

        xgblss.train(
            self.xgb_params,
            self.dtrain,
            num_boost_round=self.n_rounds
        )
        n_samples = 100

        # act
        # Returns predicted distributional parameters
        res = xgblss.predict(
            self.dtest,
            pred_type="samples",
            n_samples=n_samples,
            seed=self.seed,
        )

        # assert
        assert isinstance(res, pd.DataFrame)
        assert res.shape == (3000, n_samples)

        assert approx(res.mean().mean(), abs=0.05) == 10
        assert approx(res.quantile(0.99).mean(), abs=0.05) == 18
        assert approx(res.quantile(0.01).mean(), abs=0.05) == 2

    @staticmethod
    def get_scale_shares(pred_quantiles, quant_sel):
        n_samples = pred_quantiles.shape[0]
        quantile = float(quant_sel.split("_")[1])

        # compare estimated quantiles with the theoretical quantiles
        scale_1_filter = np.isclose(pred_quantiles[quant_sel], norm.ppf(quantile, loc=10, scale=1), atol=1)
        scale_3_filter = np.isclose(pred_quantiles[quant_sel], norm.ppf(quantile, loc=10, scale=3), atol=1)
        scale_5_filter = np.isclose(pred_quantiles[quant_sel], norm.ppf(quantile, loc=10, scale=5), atol=1)

        share_s1 = pred_quantiles.loc[scale_1_filter, quant_sel].count() / n_samples
        share_s3 = pred_quantiles.loc[scale_3_filter, quant_sel].count() / n_samples
        share_s5 = pred_quantiles.loc[scale_5_filter, quant_sel].count() / n_samples
        return share_s1, share_s3, share_s5
