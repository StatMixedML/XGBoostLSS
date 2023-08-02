from xgboostlss.model import *
from ..utils import BaseTestClass
from xgboostlss.distributions.Expectile import *


class TestClass(BaseTestClass):
    def test_loss_fn_start_values(self, dist_class, loss_fn):
        # Create data for testing
        torch.manual_seed(123)
        params = torch.randn(
            dist_class.dist.n_dist_param, dtype=torch.float64
        ).reshape(-1, dist_class.dist.n_dist_param)
        params = [params[:, i] for i in range(params.shape[1])]
        target = torch.tensor([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Call the function
        loss = dist_class.dist.loss_fn_start_values(params, target)

        # Assertions
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()

    def test_loss_fn_start_values_expectile(self, loss_fn):
        # Create data for testing
        torch.manual_seed(123)
        dist_class = XGBoostLSS(Expectile())
        params = torch.randn(
            dist_class.dist.n_dist_param, dtype=torch.float64
        ).reshape(-1, dist_class.dist.n_dist_param)
        params = [params[:, i] for i in range(params.shape[1])]
        target = torch.tensor([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Set the tau for testing
        dist_class.dist.tau = [0.05, 0.95]

        # Call the function
        loss = dist_class.dist.loss_fn_start_values(params, target)

        # Assertions
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
