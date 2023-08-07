from ..utils import BaseTestClass
import pytest


class TestClass(BaseTestClass):
    def test_init(self, flow_dist):
        with pytest.raises(ValueError, match="target_support must be a string."):
            flow_dist(target_support=1)
        with pytest.raises(ValueError, match="Invalid target_support."):
            flow_dist(target_support="invalid_target_support")

        with pytest.raises(ValueError, match="count_bins must be an integer."):
            flow_dist(count_bins=1.0)
            flow_dist(count_bins="1.0")
        with pytest.raises(ValueError, match="count_bins must be a positive integer > 0"):
            flow_dist(count_bins=0)

        with pytest.raises(ValueError, match="bound must be a float."):
            flow_dist(bound=1)
            flow_dist(bound="1")

        with pytest.raises(ValueError, match="order must be a string."):
            flow_dist(order=1)
            flow_dist(order="invalid_order")

        with pytest.raises(ValueError, match="Invalid order specification."):
            flow_dist(order="invalid_order")

        assert isinstance(flow_dist().stabilization, str)
        assert flow_dist().stabilization is not None
        with pytest.raises(ValueError, match="Invalid stabilization method."):
            flow_dist(stabilization="invalid_stabilization")
        with pytest.raises(ValueError, match="stabilization must be a string."):
            flow_dist(stabilization=1)

        assert isinstance(flow_dist().loss_fn, str)
        assert flow_dist().loss_fn is not None
        with pytest.raises(ValueError, match="loss_fn must be a string."):
            flow_dist(loss_fn=1)
        with pytest.raises(ValueError, match="Invalid loss_fn."):
            flow_dist(loss_fn="invalid_loss_fn")

    def test_distribution_parameters(self, flow_dist):
        assert isinstance(flow_dist().param_dict, dict)
        assert all(callable(func) for func in flow_dist().param_dict.values())

    def test_defaults(self, flow_dist):
        assert isinstance(flow_dist().univariate, bool)
        assert flow_dist().univariate is True
        assert isinstance(flow_dist().discrete, bool)
