import pandas as pd

from ..utils import BaseTestClass
from xgboostlss.utils import softplus_fn
import pytest
import torch
import numpy as np


class TestClass(BaseTestClass):
    def test_init(self, multivariate_dist):
        with pytest.raises(ValueError, match="Invalid dimensionality type."):
            multivariate_dist(D="1")
        with pytest.raises(ValueError, match="Invalid dimensionality."):
            multivariate_dist(D=1)

        assert isinstance(multivariate_dist().stabilization, str)
        assert multivariate_dist().stabilization is not None
        with pytest.raises(ValueError, match="Invalid stabilization method."):
            multivariate_dist(stabilization="invalid_stabilization")

        with pytest.raises(ValueError, match="Invalid response function."):
            multivariate_dist(response_fn="invalid_response_fn")

        assert isinstance(multivariate_dist().loss_fn, str)
        assert multivariate_dist().loss_fn is not None
        with pytest.raises(ValueError, match="Invalid loss function."):
            multivariate_dist(loss_fn="invalid_loss_fn")

    def test_distribution_parameters(self, multivariate_dist):
        assert isinstance(multivariate_dist().param_dict, dict)
        assert all(callable(func) for func in multivariate_dist().param_dict.values())
        assert isinstance(multivariate_dist().n_dist_param, int)
        assert isinstance(multivariate_dist().distribution_arg_names, list)
        assert isinstance(multivariate_dist().n_targets, int)
        assert multivariate_dist().n_targets > 0
        assert multivariate_dist().param_transform is not None
        assert callable(multivariate_dist().param_transform) is True
        assert multivariate_dist().create_param_dict is not None
        assert callable(multivariate_dist().create_param_dict) is True
        assert multivariate_dist().get_dist_params is not None
        assert callable(multivariate_dist().get_dist_params) is True

    def test_defaults(self, multivariate_dist):
        assert isinstance(multivariate_dist().univariate, bool)
        assert multivariate_dist().univariate is False
        assert isinstance(multivariate_dist().discrete, bool)
        assert multivariate_dist().discrete is False
        if multivariate_dist.__name__ == "MVN_LoRa":
            assert multivariate_dist().rank is not None
        else:
            assert multivariate_dist().rank is None

    def test_create_param_dict(self, multivariate_dist):
        if multivariate_dist.__name__ in ["MVN", "Dirichlet"]:
            param_dict = multivariate_dist.create_param_dict(n_targets=2, response_fn=softplus_fn)
        if multivariate_dist.__name__ == "MVN_LoRa":
            param_dict = multivariate_dist.create_param_dict(n_targets=2, rank=1, response_fn=softplus_fn)
        if multivariate_dist.__name__ == "MVT":
            param_dict = multivariate_dist.create_param_dict(n_targets=2, response_fn=softplus_fn, response_fn_df=softplus_fn)
        assert isinstance(param_dict, dict)
        assert all(callable(func) for func in param_dict.values())

    def test_param_transform(self, multivariate_dist):
        # Initialize distribution and create data for testing
        mult_dist = multivariate_dist()
        params = [torch.tensor(0.5).reshape(-1, 1) for _ in range(mult_dist.n_dist_param)]
        param_dict = mult_dist.param_dict
        n_targets = mult_dist.n_targets
        rank = 1
        n_obs = 4

        # Call the function
        params = multivariate_dist.param_transform(params, param_dict, n_targets, rank, n_obs)

        # Assertions
        if multivariate_dist.__name__ == "Dirichlet":
            assert isinstance(params, torch.Tensor)
            assert not torch.isnan(params).any()
            assert not torch.isinf(params).any()
        else:
            assert isinstance(params, list)
            for param in params:
                assert isinstance(param, torch.Tensor)
                assert not torch.isnan(param).any()
                assert not torch.isinf(param).any()

    def test_get_dist_params(self, multivariate_dist):
        # Initialize distribution and create data for testing
        n_obs = 10
        params = torch.ones(n_obs, multivariate_dist().n_dist_param)
        predt = [params[:, i].reshape(-1, 1) for i in range(multivariate_dist().n_dist_param)]

        # Transform parameters
        dist_params_predt = multivariate_dist.param_transform(
            predt,
            multivariate_dist().param_dict,
            multivariate_dist().n_targets,
            rank=multivariate_dist().rank,
            n_obs=n_obs)

        # Create distribution
        if multivariate_dist.__name__ == "Dirichlet":
            dist_kwargs = dict(zip(multivariate_dist().distribution_arg_names, [dist_params_predt]))
        else:
            dist_kwargs = dict(zip(multivariate_dist().distribution_arg_names, dist_params_predt))
        dist_pred = multivariate_dist().distribution(**dist_kwargs)

        # Call the function
        predt_params_df = multivariate_dist.get_dist_params(n_targets=multivariate_dist().n_targets,
                                                            dist_pred=dist_pred)

        # Assertions
        assert isinstance(predt_params_df, pd.DataFrame)
        assert not predt_params_df.isna().any().any()

    def test_covariance_to_correlation(self, multivariate_dist):
        # Create data for testing
        cov_mat = torch.tensor([[1, 0.5], [0.5, 1]])

        if not multivariate_dist.__name__ == "Dirichlet":
            # Call the function
            cor_mat = multivariate_dist.covariance_to_correlation(cov_mat)

            # Assertions
            assert isinstance(cor_mat, np.ndarray)
            assert not np.isnan(cor_mat).any()
            assert not np.isinf(cor_mat).any()
