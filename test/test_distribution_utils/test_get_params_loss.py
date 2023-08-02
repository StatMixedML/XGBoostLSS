from ..utils import BaseTestClass
from xgboostlss.distributions.Expectile import *
from xgboostlss.model import *


class TestClass(BaseTestClass):
    def test_get_params_loss(self, dist_class, loss_fn, requires_grad):
        # Create data for testing
        predt = np.random.rand(dist_class.dist.n_dist_param * 4).reshape(-1, dist_class.dist.n_dist_param)
        target = torch.tensor([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        start_values = [0.5 for _ in range(dist_class.dist.n_dist_param)]

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Call the function
        predt, loss = dist_class.dist.get_params_loss(predt, target, start_values, requires_grad)

        # Assertions
        assert isinstance(predt, List)
        assert len(predt) == dist_class.dist.n_dist_param
        for i in range(len(predt)):
            assert isinstance(predt[i], torch.Tensor)
            assert not torch.isnan(predt[i]).any()
            assert not torch.isinf(predt[i]).any()
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()


    def test_get_params_loss_crps(self, dist_class_crps, requires_grad):
        # Create data for testing
        predt = np.random.rand(dist_class_crps.dist.n_dist_param * 4).reshape(-1, dist_class_crps.dist.n_dist_param)
        target = torch.tensor([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        start_values = [0.5 for _ in range(dist_class_crps.dist.n_dist_param)]

        # Set the loss function for testing
        dist_class_crps.dist.loss_fn = "crps"

        # Call the function
        predt, loss = dist_class_crps.dist.get_params_loss(predt, target, start_values, requires_grad)

        # Assertions
        assert isinstance(predt, List)
        assert len(predt) == dist_class_crps.dist.n_dist_param
        for i in range(len(predt)):
            assert isinstance(predt[i], torch.Tensor)
            assert not torch.isnan(predt[i]).any()
            assert not torch.isinf(predt[i]).any()
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

    def test_get_params_loss_expectile(self, dist_class, loss_fn, requires_grad):
        dist_class = XGBoostLSS(Expectile())
        # Create data for testing
        predt = np.random.rand(dist_class.dist.n_dist_param * 4).reshape(-1, dist_class.dist.n_dist_param)
        target = torch.tensor([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        start_values = [0.5 for _ in range(dist_class.dist.n_dist_param)]

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Set the tau for testing
        dist_class.dist.tau = [0.05, 0.95]

        # Call the function
        predt, loss = dist_class.dist.get_params_loss(predt, target, start_values, requires_grad)

        # Assertions
        assert isinstance(predt, List)
        assert len(predt) == dist_class.dist.n_dist_param
        for i in range(len(predt)):
            assert isinstance(predt[i], torch.Tensor)
            assert not torch.isnan(predt[i]).any()
            assert not torch.isinf(predt[i]).any()
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
