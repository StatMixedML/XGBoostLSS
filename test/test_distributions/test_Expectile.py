from ..utils import BaseTestClass
from xgboostlss.distributions.Expectile import *
import torch
import pytest


class TestClass(BaseTestClass):
    def test_init(self, expectile_dist):
        assert isinstance(expectile_dist().stabilization, str)
        assert expectile_dist().stabilization is not None
        with pytest.raises(ValueError, match="Invalid stabilization method."):
            expectile_dist(stabilization="invalid_stabilization")

        with pytest.raises(ValueError, match="Expectiles must be a list."):
            expectile_dist(expectiles=0.1)

        with pytest.raises(ValueError, match="Expectiles must be between 0 and 1."):
            expectile_dist(expectiles=[-0.1, 0.1, 1.1])

        with pytest.raises(ValueError, match="penalize_crossing must be a boolean."):
            expectile_dist(penalize_crossing=0.1)

        assert isinstance(expectile_dist().loss_fn, str)
        assert expectile_dist().loss_fn is not None

    def test_expectile_distribution_parameters(self, expectile_dist):
        assert isinstance(expectile_dist().param_dict, dict)
        assert set(expectile_dist().param_dict.keys()) == set(expectile_dist().distribution_arg_names)
        assert all(callable(func) for func in expectile_dist().param_dict.values())
        assert expectile_dist().n_dist_param == len(expectile_dist().distribution_arg_names)
        assert isinstance(expectile_dist().n_dist_param, int)
        assert isinstance(expectile_dist().distribution_arg_names, list)
        assert isinstance(expectile_dist().tau, torch.Tensor)

    def test_defaults(self, expectile_dist):
        assert isinstance(expectile_dist().univariate, bool)
        assert expectile_dist().univariate is True
        assert isinstance(expectile_dist().discrete, bool)
        assert expectile_dist().discrete is False
        assert expectile_dist().tau is not None
        assert isinstance(expectile_dist().penalize_crossing, bool)

    def test_expectile_init(self):
        # Create an instance of Expectile_Torch with example expectiles
        expectiles = [torch.tensor([0.1, 0.5, 0.9])]
        expectile_instance = Expectile_Torch(expectiles)

        # Assertions
        assert expectile_instance.expectiles == expectiles
        assert isinstance(expectile_instance.penalize_crossing, bool)
        assert expectile_instance.__class__.__name__ == "Expectile"

    def test_expectile_log_prob(self):
        # Create an instance of Expectile_Torch with example expectiles
        expectiles = [torch.tensor([0.1, 0.5, 0.9])]
        expectile_instance_penalize = Expectile_Torch(expectiles, penalize_crossing=True)
        expectile_instance_no_penalize = Expectile_Torch(expectiles, penalize_crossing=False)
        value = torch.tensor([0.2, 0.4, 0.6, 0.8])

        # Call the function
        loss_penalize = expectile_instance_penalize.log_prob(value, expectiles)
        loss_no_penalize = expectile_instance_no_penalize.log_prob(value, expectiles)

        # Assertions
        assert isinstance(loss_penalize, torch.Tensor)
        assert not torch.isnan(loss_penalize).any()
        assert not torch.isinf(loss_penalize).any()

        assert isinstance(loss_no_penalize, torch.Tensor)
        assert not torch.isnan(loss_no_penalize).any()
        assert not torch.isinf(loss_no_penalize).any()


def test_expectile_pnorm():
    # Create example data
    tau = np.array([0.5], dtype="float")
    m = np.array([0.2, 0.4, 0.8]).reshape(-1, 1)
    sd = np.array([0.1, 0.2, 0.3]).reshape(-1, 1)

    # Call the function
    out = expectile_pnorm(tau, m, sd)

    # Assertions
    assert isinstance(out, np.ndarray)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()


def test_expectile_norm():
    # Create example data
    tau = np.array([0.5], dtype="float")
    m = np.array([0.2, 0.4, 0.8]).reshape(-1, 1)
    sd = np.array([0.1, 0.2, 0.3]).reshape(-1, 1)

    # Call the function
    out = expectile_norm(tau, m, sd)

    # Assertions
    assert isinstance(out, np.ndarray)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
