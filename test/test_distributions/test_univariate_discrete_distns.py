from ..utils import BaseTestClass
import pytest


class TestClass(BaseTestClass):
    def test_init(self, univariate_discrete_dist):
        assert isinstance(univariate_discrete_dist().stabilization, str)
        assert univariate_discrete_dist().stabilization is not None
        with pytest.raises(ValueError, match="Invalid stabilization method."):
            univariate_discrete_dist(stabilization="invalid_stabilization")

        if univariate_discrete_dist.__name__ in ["NegativeBinomial", "ZINB"]:
            with pytest.raises(ValueError, match="Invalid response function for total_count."):
                univariate_discrete_dist(response_fn_total_count="invalid_response_fn")
            with pytest.raises(ValueError, match="Invalid response function for probs."):
                univariate_discrete_dist(response_fn_probs="invalid_response_fn")
        else:
            with pytest.raises(ValueError, match="Invalid response function."):
                univariate_discrete_dist(response_fn="invalid_response_fn")

        assert isinstance(univariate_discrete_dist().loss_fn, str)
        assert univariate_discrete_dist().loss_fn is not None
        with pytest.raises(ValueError, match="Invalid loss function."):
            univariate_discrete_dist(loss_fn="invalid_loss_fn")

    def test_distribution_parameters(self, univariate_discrete_dist):
        assert isinstance(univariate_discrete_dist().param_dict, dict)
        assert set(univariate_discrete_dist().param_dict.keys()) == set(univariate_discrete_dist().distribution_arg_names)
        assert all(callable(func) for func in univariate_discrete_dist().param_dict.values())
        assert univariate_discrete_dist().n_dist_param == len(univariate_discrete_dist().distribution_arg_names)
        assert isinstance(univariate_discrete_dist().n_dist_param, int)
        assert isinstance(univariate_discrete_dist().distribution_arg_names, list)

    def test_defaults(self, univariate_discrete_dist):
        assert isinstance(univariate_discrete_dist().univariate, bool)
        assert univariate_discrete_dist().univariate is True
        assert isinstance(univariate_discrete_dist().discrete, bool)
        assert univariate_discrete_dist().discrete is True
        assert univariate_discrete_dist().tau is None
        assert isinstance(univariate_discrete_dist().penalize_crossing, bool)
