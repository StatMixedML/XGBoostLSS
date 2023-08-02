from ..utils import BaseTestClass
from typing import List
import numpy as np
import torch


class TestClass(BaseTestClass):
    def test_compute_gradients_and_hessians(self, dist_class, loss_fn, stabilization):
        # Create data for testing
        np.random.seed(123)
        params = np.random.rand(dist_class.dist.n_dist_param * 4).reshape(-1, dist_class.dist.n_dist_param)
        target = torch.tensor([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        start_values = np.array([0.5 for _ in range(dist_class.dist.n_dist_param)])
        weights = np.ones_like(target)

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Set the stabilization for testing
        dist_class.dist.stabilization = stabilization

        # Call the function
        predt, loss = dist_class.dist.get_params_loss(params, target, start_values, requires_grad=True)
        grad, hess = dist_class.dist.compute_gradients_and_hessians(loss, predt, weights)

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

        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == params.flatten().shape
        assert hess.shape == params.flatten().shape
        assert not np.isnan(grad).any()
        assert not np.isnan(hess).any()

    def test_compute_gradients_and_hessians_crps(self, dist_class_crps, stabilization):
        # Create data for testing
        np.random.seed(123)
        params = np.random.rand(dist_class_crps.dist.n_dist_param * 4).reshape(-1, dist_class_crps.dist.n_dist_param)
        target = torch.tensor([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        start_values = np.array([0.5 for _ in range(dist_class_crps.dist.n_dist_param)])
        weights = np.ones_like(target)

        # Set the loss function for testing
        dist_class_crps.dist.loss_fn = "crps"

        # Set the stabilization for testing
        dist_class_crps.dist.stabilization = stabilization

        # Call the function
        predt, loss = dist_class_crps.dist.get_params_loss(params, target, start_values, requires_grad=True)
        grad, hess = dist_class_crps.dist.compute_gradients_and_hessians(loss, predt, weights)

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

        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == params.flatten().shape
        assert hess.shape == params.flatten().shape
        assert not np.isnan(grad).any()
        assert not np.isnan(hess).any()

    def test_compute_gradients_and_hessians_nans(self, dist_class, loss_fn, stabilization):
        # Create data for testing
        np.random.seed(123)
        params = np.random.rand(dist_class.dist.n_dist_param * 4).reshape(-1, dist_class.dist.n_dist_param)
        params[0, 0] = np.nan
        target = torch.tensor([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        start_values = np.array([0.5 for _ in range(dist_class.dist.n_dist_param)])
        weights = np.ones_like(target)

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Set the stabilization for testing
        dist_class.dist.stabilization = stabilization

        # Call the function
        predt, loss = dist_class.dist.get_params_loss(params, target, start_values, requires_grad=True)
        grad, hess = dist_class.dist.compute_gradients_and_hessians(loss, predt, weights)

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

        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == params.flatten().shape
        assert hess.shape == params.flatten().shape
        assert not np.isnan(grad).any()
        assert not np.isnan(hess).any()
