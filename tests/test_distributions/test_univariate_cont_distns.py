from ..utils import BaseTestClass
import pytest


class TestClass(BaseTestClass):
    def test_init(self, univariate_cont_dist):
        assert isinstance(univariate_cont_dist().stabilization, str)
        assert univariate_cont_dist().stabilization is not None
        with pytest.raises(ValueError, match="Invalid stabilization method."):
            univariate_cont_dist(stabilization="invalid_stabilization")

        with pytest.raises(ValueError, match="Invalid response function."):
            univariate_cont_dist(response_fn="invalid_response_fn")

        assert isinstance(univariate_cont_dist().loss_fn, str)
        assert univariate_cont_dist().loss_fn is not None
        with pytest.raises(ValueError, match="Invalid loss function."):
            univariate_cont_dist(loss_fn="invalid_loss_fn")

    def test_distribution_parameters(self, univariate_cont_dist):
        assert isinstance(univariate_cont_dist().param_dict, dict)
        assert set(univariate_cont_dist().param_dict.keys()) == set(univariate_cont_dist().distribution_arg_names)
        assert all(callable(func) for func in univariate_cont_dist().param_dict.values())
        assert univariate_cont_dist().n_dist_param == len(univariate_cont_dist().distribution_arg_names)
        assert isinstance(univariate_cont_dist().n_dist_param, int)
        assert isinstance(univariate_cont_dist().distribution_arg_names, list)
        assert univariate_cont_dist().distribution_arg_names == list(univariate_cont_dist().distribution.arg_constraints.keys())

    def test_defaults(self, univariate_cont_dist):
        assert isinstance(univariate_cont_dist().univariate, bool)
        assert univariate_cont_dist().univariate is True
        assert isinstance(univariate_cont_dist().discrete, bool)
        assert univariate_cont_dist().discrete is False
        assert univariate_cont_dist().tau is None
        assert isinstance(univariate_cont_dist().penalize_crossing, bool)
