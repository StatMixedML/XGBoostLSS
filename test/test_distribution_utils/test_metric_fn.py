from ..utils import BaseTestClass
import xgboost as xgb
import numpy as np
import torch


class TestClass(BaseTestClass):
    def test_metric_fn_weight(self, dist_class, loss_fn):
        # Create data for testing
        np.random.seed(123)
        predt = np.random.rand(dist_class.dist.n_dist_param * 4).reshape(-1, dist_class.dist.n_dist_param)
        labels = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        weights = np.ones_like(labels)
        dmatrix = xgb.DMatrix(predt, label=labels, weight=weights)
        dist_class.set_base_margin(dmatrix)

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
        np.random.seed(123)
        predt = np.random.rand(dist_class.dist.n_dist_param * 4).reshape(-1, dist_class.dist.n_dist_param)
        labels = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        dmatrix = xgb.DMatrix(predt, label=labels)
        dist_class.set_base_margin(dmatrix)

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
        np.random.seed(123)
        predt = np.random.rand(dist_class.dist.n_dist_param * 4).reshape(-1, dist_class.dist.n_dist_param)
        predt[0, 0] = np.nan
        labels = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        weights = np.ones_like(labels)
        dmatrix = xgb.DMatrix(predt, label=labels, weight=weights)
        dist_class.set_base_margin(dmatrix)

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
        np.random.seed(123)
        predt = np.random.rand(dist_class_crps.dist.n_dist_param * 4).reshape(-1, dist_class_crps.dist.n_dist_param)
        labels = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        weights = np.ones_like(labels)
        dmatrix = xgb.DMatrix(predt, label=labels, weight=weights)
        dist_class_crps.set_base_margin(dmatrix)

        # Set the loss function for testing
        dist_class_crps.dist.loss_fn = "crps"

        # Call the function
        loss_fn, loss = dist_class_crps.dist.metric_fn(predt, dmatrix)

        # Assertions
        assert isinstance(loss_fn, str)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
