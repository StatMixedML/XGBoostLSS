from ..utils import BaseTestClass
import numpy as np
import pandas as pd
import xgboost as xgb


class TestClass(BaseTestClass):
    def test_predict_dist(self, dist_class, pred_type):
        # Create data for testing
        np.random.seed(123)
        X_dta = np.random.rand(10).reshape(-1, 1)
        y_dta = np.random.rand(10).reshape(-1, 1)
        dtrain = xgb.DMatrix(X_dta, label=y_dta)

        # Train the model
        params = {"eta": 0.01}
        dist_class.train(params, dtrain, num_boost_round=2)

        # Call the function
        if dist_class.dist.tau is not None and pred_type in ["quantiles", "samples"]:
            pred_type = "parameters"
        predt_df = dist_class.dist.predict_dist(dist_class.booster,
                                                dist_class.start_values,
                                                dtrain,
                                                pred_type,
                                                n_samples=100,
                                                quantiles=[0.1, 0.5, 0.9]
                                                )

        # Assertions
        assert isinstance(predt_df, pd.DataFrame)
        assert not predt_df.isna().any().any()
        if pred_type == "parameters" or pred_type == "expectiles":
            assert predt_df.shape[1] == dist_class.dist.n_dist_param
        if dist_class.dist.tau is None:
            if pred_type == "samples":
                assert predt_df.shape[1] == 100
            elif pred_type == "quantiles":
                assert predt_df.shape[1] == 3
