from ..utils import BaseTestClass, gen_test_data
import torch


class TestClass(BaseTestClass):
    def test_loss_fn_start_values(self, dist_class, loss_fn):
        # Create data for testing
        _, target, _ = gen_test_data(dist_class)
        predt = [
            torch.tensor(0.5, dtype=torch.float64).reshape(-1, 1).requires_grad_(True) for _ in
            range(dist_class.dist.n_dist_param)
        ]
        if dist_class.dist.univariate:
            target = torch.tensor(target)
        else:
            target = torch.tensor(target)[:, :dist_class.dist.n_targets]

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Call the function
        if hasattr(dist_class.dist, "base_dist"):
            pass
        else:
            loss = dist_class.dist.loss_fn_start_values(predt, target)
            # Assertions
            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss).any()
            assert not torch.isinf(loss).any()
