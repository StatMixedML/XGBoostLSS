from ..utils import BaseTestClass, gen_test_data
import numpy as np
import torch


class TestClass(BaseTestClass):
    def test_metric_fn_weight(self, dist_class, loss_fn):
        # Create data for testing
        predt, labels, weights, dmatrix = gen_test_data(dist_class, weights=True)

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Call the function
        loss_fn, loss = dist_class.dist.metric_fn(predt, dmatrix)

        # Assertions
        assert isinstance(loss_fn, str)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

    def test_metric_fn_no_weight(self, dist_class, loss_fn):
        # Create data for testing
        predt, labels, dmatrix = gen_test_data(dist_class, weights=False)

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Call the function
        loss_fn, loss = dist_class.dist.metric_fn(predt, dmatrix)

        # Assertions
        assert isinstance(loss_fn, str)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

    def test_metric_fn_nans(self, dist_class, loss_fn):
        # Create data for testing and et some predt to nan
        predt, labels, weights, dmatrix = gen_test_data(dist_class, weights=True)
        predt[0, 0] = np.nan

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Call the function
        loss_fn, loss = dist_class.dist.metric_fn(predt, dmatrix)

        # Assertions
        assert isinstance(loss_fn, str)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

    def test_metric_fn_crps(self, dist_class_crps):
        # Create data for testing
        predt, labels, weights, dmatrix = gen_test_data(dist_class_crps, weights=True)

        # Set the loss function for testing
        dist_class_crps.dist.loss_fn = "crps"

        # Call the function
        loss_fn, loss = dist_class_crps.dist.metric_fn(predt, dmatrix)

        # Assertions
        assert isinstance(loss_fn, str)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
