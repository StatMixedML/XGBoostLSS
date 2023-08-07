from ..utils import BaseTestClass
import torch


class TestClass(BaseTestClass):
    def test_crps_score(self, flow_class):
        # Create data for testing
        torch.manual_seed(123)
        n_obs = 10
        n_samples = 20
        y = torch.rand(n_obs, 1)
        yhat_dist = torch.rand(n_samples, n_obs)

        # Call the function
        loss = flow_class.dist.crps_score(y, yhat_dist)

        # Assertions
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
        assert loss.shape == y.shape
