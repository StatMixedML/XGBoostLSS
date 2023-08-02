from ..utils import BaseTestClass
import numpy as np


class TestClass(BaseTestClass):
    def test_calculate_start_values(self, dist_class, loss_fn):
        # Create data for testing
        target = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Call the objective_fn method
        loss, start_values = dist_class.dist.calculate_start_values(target)

        # Assertions
        assert isinstance(loss, np.ndarray)
        assert not np.isnan(loss).any()
        assert not np.isinf(loss).any()

        assert isinstance(start_values, np.ndarray)
        assert start_values.shape[0] == dist_class.dist.n_dist_param
        assert not np.isnan(start_values).any()
        assert not np.isinf(start_values).any()
